"""CPU Gloo regression for Self-Forcing Ulysses and ring KV ownership."""

import ast
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]


def source_method(path, class_name, name, **namespace):
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body
                                if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    method.returns = None
    for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    ast.fix_missing_locations(method)
    scope = dict(torch=torch, math=math, Tensor=torch.Tensor, List=list,
                 causal_rope_apply=lambda x, *args, **kwargs: x,
                 causal_rope_apply_chunked=lambda x, *args, **kwargs: x)
    scope.update(namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def dense(q, k, v):
    score = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    return torch.einsum("bhqk,bkhd->bqhd", score.softmax(dim=-1), v)


def gather(tensor, group, size):
    if size == 1:
        return [tensor]
    result = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(result, tensor.contiguous(), group=group)
    return result


class SeqAllToAllCPU:
    """Same scatter/gather dimension contract as Yunchang's SeqAllToAll4D."""

    @staticmethod
    def apply(group, tensor, scatter_idx, gather_idx):
        size = dist.get_world_size(group)
        if size == 1:
            return tensor
        pieces = [part.contiguous() for part in tensor.chunk(size, dim=scatter_idx)]
        assert len(pieces) == size and all(part.shape == pieces[0].shape for part in pieces)
        send = torch.cat([part.reshape(-1) for part in pieces])
        receive = torch.empty_like(send)
        dist.all_to_all_single(receive, send, group=group)
        count = pieces[0].numel()
        received = [receive[i * count:(i + 1) * count].reshape(pieces[0].shape)
                    for i in range(size)]
        return torch.cat(received, dim=gather_idx)


class Backing:
    def __init__(self, capacity, heads, head_dim):
        self.device = torch.device("cpu")
        self.cache = torch.full((2, capacity, 1, heads, head_dim), float("nan"))
        self.reads = []
        self.writes = []

    def get_raw(self, request, layer):
        return self.cache

    def get(self, request, layer):
        return self.cache.clone()

    def set(self, request, layer, start, size, value):
        assert 0 <= start <= start + size <= self.cache.shape[1]
        self.writes.append((start, size))
        self.cache[:, start:start + size] = value


def worker(rank, ulysses, ring, init_file):
    world = ulysses * ring
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)
    try:
        ulysses_groups = [dist.new_group(list(range(r * ulysses, (r + 1) * ulysses)))
                          for r in range(ring)]
        ring_groups = [dist.new_group([r * ulysses + u for r in range(ring)])
                       for u in range(ulysses)]
        ring_index, ulysses_index = divmod(rank, ulysses)
        ulysses_group = ulysses_groups[ring_index]
        ring_group = ring_groups[ulysses_index]

        model = ROOT / "inferix/models/self_forcing/causal_model.py"
        wrapper_file = ROOT / "inferix/kvcache_manager/model/self_forcing_kv_cache_manager.py"
        distributed = ROOT / "inferix/models/attention/distributed.py"
        self_forward = source_method(model, "CausalWanSelfAttention", "forward")
        block_forward = source_method(model, "CausalWanAttentionBlock", "forward")
        core_forward = source_method(distributed, "CoreAttention", "forward",
                                     SeqAllToAll4D=SeqAllToAllCPU)
        get_cache = source_method(wrapper_file, "SelfForcingKVCacheManager", "get_kv_cache")
        set_cache = source_method(wrapper_file, "SelfForcingKVCacheManager", "set_kv_cache")

        def scenario(offload, sink, multiframe):
            heads, head_dim = 2, 2
            frame_tokens = world * 2
            pre_tokens_per_frame = frame_tokens // world
            sink_tokens = sink * frame_tokens // ring
            local_size = 6 if multiframe else 3
            capacity = local_size * frame_tokens // ring
            backing = Backing(capacity, heads // ulysses, head_dim)
            request = SimpleNamespace(request_id="0")
            wrapper = SimpleNamespace(layer_number=0, enable_kv_offload=offload)

            def tracked_get(**kwargs):
                backing.reads.append(kwargs.get("read_length"))
                return get_cache(wrapper, **kwargs)

            wrapper.get_kv_cache = tracked_get
            wrapper.set_kv_cache = lambda **kwargs: set_cache(wrapper, **kwargs)

            def ring_dense(q, k, v, **kwargs):
                all_k = torch.cat(gather(k, ring_group, ring), dim=1)
                all_v = torch.cat(gather(v, ring_group, ring), dim=1)
                return dense(q, all_k, all_v)

            core = SimpleNamespace(use_pack_qkv=False, ulysses_pg=ulysses_group,
                                   ring_pg=ring_group, scatter_idx=2, gather_idx=1,
                                   _select_strategy=lambda *args: "pass-kv",
                                   _attention_forward=ring_dense,
                                   q_descale=None, k_descale=None, v_descale=None)
            config = SimpleNamespace(world_size=world, rank=rank, ulysses_size=ulysses,
                                     ring_size=ring)
            identity = lambda x: x
            attn = SimpleNamespace(num_heads=heads, head_dim=head_dim,
                                   norm_q=identity, norm_k=identity, q=identity,
                                   k=identity, v=identity, o=identity,
                                   attention=lambda *args, **kwargs: core_forward(core, *args, **kwargs),
                                   parallel_config=config, local_attn_size=local_size, sink_size=sink)

            class SelfAttentionAdapter:
                def __call__(self, *args, **kwargs):
                    return self_forward(attn, *args, **kwargs)

            block = SimpleNamespace(modulation=torch.zeros(1, 6, heads * head_dim),
                                    norm1=identity, norm2=identity, norm3=identity,
                                    self_attn=SelfAttentionAdapter(),
                                    cross_attn=lambda x, *args, **kwargs: torch.zeros_like(x),
                                    ffn=lambda x: torch.zeros_like(x),
                                    kv_cache_manager=wrapper, enable_kv_offload=offload,
                                    parallel_config=config, local_attn_size=local_size)
            meta = dict(global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))
            expected_cache = torch.empty((0, heads // ulysses, head_dim))
            expected_global_end = 0

            def step(frame_start, frames, seed):
                nonlocal expected_cache, expected_global_end
                global_start = frame_start * frame_tokens
                pre_tokens = frames * pre_tokens_per_frame
                post_tokens = pre_tokens * ulysses
                x = (torch.arange(pre_tokens * heads * head_dim, dtype=torch.float32)
                     .reshape(1, pre_tokens, heads * head_dim) + seed + rank * 0.37) / 17
                e = torch.zeros(1, frames, 6, heads * head_dim)
                e[:, :, 2] = 1

                # Independent post-Ulysses oracle: gather source sequences from
                # group members and select this rank's head shard.
                source = x.reshape(1, pre_tokens, heads, head_dim)
                gathered = gather(source, ulysses_group, ulysses)
                head_start = ulysses_index * (heads // ulysses)
                post = torch.cat([part[:, :, head_start:head_start + heads // ulysses]
                                  for part in gathered], dim=1)
                old_end = expected_cache.shape[0]
                old_global = expected_global_end
                start_local = global_start // ring
                end_global = start_local + post_tokens
                evicts = end_global > expected_global_end and old_end + post_tokens > capacity
                discarded = old_end + post_tokens - capacity if evicts else 0
                end = old_end + end_global - expected_global_end - discarded
                offset = end - post_tokens
                if evicts:
                    expected_cache = torch.cat((expected_cache[:sink_tokens],
                                                expected_cache[sink_tokens + discarded:]))
                expected_cache = torch.cat((expected_cache[:offset], post[0],
                                            expected_cache[offset + post_tokens:]))[:end]
                expected_global_end = end_global

                # Ring attention sees every ring rank's head-sharded cache.
                all_expected = torch.cat(gather(expected_cache.unsqueeze(0), ring_group, ring), dim=1)
                expected_post_output = dense(post, all_expected, all_expected)
                peer_outputs = gather(expected_post_output, ulysses_group, ulysses)
                expected_y = torch.cat([peer[:, ulysses_index * pre_tokens:(ulysses_index + 1) * pre_tokens]
                                        for peer in peer_outputs], dim=2).flatten(2)

                read_before, write_before = len(backing.reads), len(backing.writes)
                original_empty = torch.empty
                scratch_allocations = [0]

                def poisoned_empty(*args, **kwargs):
                    value = original_empty(*args, **kwargs)
                    if tuple(value.shape) == tuple(backing.cache.shape):
                        value.fill_(float("nan"))
                        scratch_allocations[0] += 1
                    return value

                if offload:
                    torch.empty = poisoned_empty
                try:
                    output = block_forward(block, x, e, torch.tensor([pre_tokens]),
                                           torch.tensor([[frames, 1, frame_tokens]]), None,
                                           None, None, None, kv_cache_meta=meta,
                                           current_start=global_start, kv_cache_manager=backing,
                                           kv_cache_requests=[request])
                finally:
                    torch.empty = original_empty
                assert scratch_allocations[0] == (1 if offload else 0)
                torch.testing.assert_close(output, x + expected_y, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(backing.cache[0, :end, 0], expected_cache,
                                           rtol=0, atol=0)
                torch.testing.assert_close(backing.cache[1, :end, 0], expected_cache,
                                           rtol=0, atol=0)
                assert meta["local_end_index"].item() == end
                assert meta["global_end_index"].item() == expected_global_end
                expected_read = old_end if evicts else old_end + start_local - old_global
                assert len(backing.reads) == read_before + 1
                assert backing.reads[-1] == (expected_read if offload else None)
                assert len(backing.writes) == write_before + 1
                expected_dirty = min(sink_tokens, offset) if evicts else offset
                write_start = expected_dirty if offload else 0
                assert backing.writes[-1] == (write_start, end - write_start)

            # Multi-frame Ulysses cache order is rank-major within each block.
            # The oracle retains exactly the stored prefix for sink eviction.
            sequence = (((0, 3, 1), (0, 3, 9), (3, 3, 17), (3, 3, 25),
                         (6, 3, 33), (6, 3, 41)) if multiframe else
                        ((0, 1, 1), (0, 1, 9), (1, 1, 17), (2, 1, 25),
                         (3, 1, 33), (3, 1, 41), (4, 1, 49)))
            for frame_start, frames, seed in sequence:
                step(frame_start, frames, seed)
            expected_cache = expected_cache[:0]
            expected_global_end = 0
            meta["global_end_index"].fill_(0)
            meta["local_end_index"].fill_(0)
            step(0, 3 if multiframe else 1, 57)
        for offload in (True, False):
            for sink in (0, 1):
                for multiframe in (False, True):
                    scenario(offload, sink, multiframe)
    finally:
        dist.destroy_process_group()


class SelfForcingDistributedKVTest(unittest.TestCase):
    def test_ulysses_ring_and_combined(self):
        for ulysses, ring in ((2, 1), (1, 2), (2, 2)):
            with self.subTest(ulysses=ulysses, ring=ring):
                with tempfile.TemporaryDirectory() as directory:
                    mp.spawn(worker, args=(ulysses, ring, str(Path(directory) / "gloo")),
                             nprocs=ulysses * ring)


if __name__ == "__main__":
    unittest.main()
