"""CPU regression for bounded Self-Forcing KV transfers.

The source methods are extracted without importing GPU and diffusion packages.
Projection and RoPE are identity functions; attention is dense CPU attention.
"""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "inferix/models/self_forcing/causal_model.py"
WRAPPER = ROOT / "inferix/kvcache_manager/model/self_forcing_kv_cache_manager.py"


def source_method(path, class_name, name):
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body
                                if isinstance(node, ast.FunctionDef) and node.name == name))
    method.returns = None
    for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    ast.fix_missing_locations(method)
    scope = dict(torch=torch, math=math,
                 causal_rope_apply=lambda x, *args, **kwargs: x,
                 causal_rope_apply_chunked=lambda x, *args, **kwargs: x)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def dense_attention(q, k, v):
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v)


class Backing:
    def __init__(self, capacity, batch, dim):
        self.device = torch.device("cpu")
        self.data = {str(i): torch.full((2, capacity, 1, 1, dim), float("nan"))
                     for i in range(batch)}
        self.reads = []
        self.writes = []

    def get_raw(self, request, layer):
        return self.data[request.request_id]

    def get(self, request, layer):
        return self.get_raw(request, layer).clone()

    def set(self, request, layer, start, size, value):
        assert 0 <= start <= start + size <= self.data[request.request_id].shape[1]
        self.writes.append((request.request_id, start, size))
        self.data[request.request_id][:, start:start + size] = value


class Oracle:
    """Token-list model of the existing append, overwrite and eviction rules."""

    def __init__(self, batch, capacity, sink, local_size):
        self.tokens = [[] for _ in range(batch)]
        self.global_end = 0
        self.capacity = capacity
        self.sink = sink
        self.local_size = local_size

    def step(self, start, x):
        length = x.shape[1]
        old_end = len(self.tokens[0])
        evicts = (self.local_size != -1 and start + length > self.global_end
                  and old_end + length > self.capacity)
        discarded = old_end + length - self.capacity if evicts else 0
        end = old_end + start + length - self.global_end - discarded
        write_start = end - length
        outputs = []
        for request, old in enumerate(self.tokens):
            tokens = list(old)
            if evicts:
                tokens = tokens[:self.sink] + tokens[self.sink + discarded:]
            tokens[write_start:end] = [x[request, i].clone() for i in range(length)]
            tokens = tokens[:end]
            self.tokens[request] = tokens
            kv = torch.stack(tokens).unsqueeze(0).unsqueeze(2)
            query = x[request:request + 1].unsqueeze(2)
            outputs.append(x[request:request + 1] + dense_attention(query, kv, kv).flatten(2))
        self.global_end = start + length
        return torch.cat(outputs), end, evicts, write_start

    def reset(self):
        self.tokens = [[] for _ in self.tokens]
        self.global_end = 0


class SelfForcingKVTransferTest(unittest.TestCase):
    def test_cache_state_attention_and_transfer_ranges(self):
        attention_forward = source_method(MODEL, "CausalWanSelfAttention", "forward")
        block_forward = source_method(MODEL, "CausalWanAttentionBlock", "forward")
        get_cache = source_method(WRAPPER, "SelfForcingKVCacheManager", "get_kv_cache")
        set_cache = source_method(WRAPPER, "SelfForcingKVCacheManager", "set_kv_cache")

        window = [(0, 2, 1), (0, 2, 7), (0, 2, 11),
                  (2, 2, 19), (4, 2, 29), (6, 2, 37),
                  (6, 2, 41), (8, 2, 47)]
        global_blocks = [(0, 1, 1), (0, 1, 7), (0, 1, 11)]
        for block in range(7):
            start = 1 + 2 * block
            global_blocks.extend(((start, 2, 19 + 8 * block),
                                  (start, 2, 23 + 8 * block)))
        scenarios = [(6, 6, sink, window) for sink in (0, 1)]
        scenarios.append((16, -1, 0, global_blocks))

        for capacity, local_size, sink, sequence in scenarios:
            for offload in (True, False):
                for parallel in (None, SimpleNamespace(world_size=1, rank=0)):
                    with self.subTest(capacity=capacity, sink=sink, offload=offload,
                                      parallel=parallel):
                        self._run_case(attention_forward, block_forward, get_cache,
                                       set_cache, capacity, local_size, sink,
                                       sequence, offload, parallel)

    def _run_case(self, attention_forward, block_forward, get_cache, set_cache,
                  capacity, local_size, sink, sequence, offload, parallel):
        batch, dim = 2, 2
        backing = Backing(capacity, batch, dim)
        oracle = Oracle(batch, capacity, sink, local_size)
        requests = [SimpleNamespace(request_id=str(i)) for i in range(batch)]
        wrapper = SimpleNamespace(layer_number=0, enable_kv_offload=offload)

        def tracked_get(**kwargs):
            length = kwargs.get("read_length")
            backing.reads.append((kwargs["kv_cache_request"].request_id,
                                  capacity if length is None else length))
            return get_cache(wrapper, **kwargs)

        wrapper.get_kv_cache = tracked_get
        wrapper.set_kv_cache = lambda **kwargs: set_cache(wrapper, **kwargs)
        identity = lambda x: x
        attn = SimpleNamespace(num_heads=1, head_dim=dim, norm_q=identity,
                               norm_k=identity, q=identity, k=identity,
                               v=identity, o=identity, attention=dense_attention,
                               parallel_config=parallel, local_attn_size=local_size,
                               sink_size=sink)

        class AttentionAdapter:
            def __call__(self, *args, **kwargs):
                return attention_forward(attn, *args, **kwargs)

        block = SimpleNamespace(modulation=torch.zeros(1, 6, dim), norm1=identity,
                                norm2=identity, norm3=identity,
                                self_attn=AttentionAdapter(),
                                cross_attn=lambda x, *args, **kwargs: torch.zeros_like(x),
                                ffn=lambda x: torch.zeros_like(x),
                                kv_cache_manager=wrapper, enable_kv_offload=offload,
                                parallel_config=parallel, local_attn_size=local_size)
        meta = dict(global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))

        def execute(start, length, seed):
            x = (torch.arange(batch * length * dim, dtype=torch.float32)
                 .reshape(batch, length, dim) + seed) / 13
            e = torch.zeros(batch, length, 6, dim)
            e[:, :, 2] = 1
            old_end = meta["local_end_index"].item()
            old_global = meta["global_end_index"].item()
            read_count, write_count = len(backing.reads), len(backing.writes)
            expected, end, evicts, write_start = oracle.step(start, x)
            empty = torch.empty
            poisoned = [0]

            def poison(*args, **kwargs):
                value = empty(*args, **kwargs)
                if tuple(value.shape) == (2, capacity, 1, 1, dim):
                    value.fill_(float("nan"))
                    poisoned[0] += 1
                return value

            if offload:
                torch.empty = poison
            try:
                actual = block_forward(block, x, e, torch.tensor([length] * batch),
                                       torch.tensor([[length, 1, 1]] * batch),
                                       None, None, None, None, kv_cache_meta=meta,
                                       current_start=start, kv_cache_manager=backing,
                                       kv_cache_requests=requests)
            finally:
                torch.empty = empty
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(meta["local_end_index"].item(), end)
            self.assertEqual(meta["global_end_index"].item(), start + length)
            for request in requests:
                raw = backing.get_raw(request, "layer_0")
                token_values = torch.stack(oracle.tokens[int(request.request_id)])
                torch.testing.assert_close(raw[0, :end, 0, 0], token_values, rtol=0, atol=0)
                torch.testing.assert_close(raw[1, :end, 0, 0], token_values, rtol=0, atol=0)
            expected_read = (old_end if evicts else old_end + start - old_global)
            expected_write = min(sink, write_start) if evicts else write_start
            self.assertEqual(backing.reads[read_count:],
                             [(str(i), expected_read if offload else capacity)
                              for i in range(batch)])
            self.assertEqual(backing.writes[write_count:],
                             [(str(i), expected_write if offload else 0,
                               end - (expected_write if offload else 0))
                              for i in range(batch)])
            self.assertEqual(poisoned[0], batch if offload else 0)

        for start, length, seed in sequence:
            execute(start, length, seed)
        oracle.reset()
        meta["global_end_index"].fill_(0)
        meta["local_end_index"].fill_(0)
        execute(0, 2, 59)


if __name__ == "__main__":
    unittest.main()
