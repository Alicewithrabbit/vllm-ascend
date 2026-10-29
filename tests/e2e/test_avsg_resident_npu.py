# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real native resident allocation/reshape, without an OmniCache pool."""
import importlib.abc
import sys
from types import SimpleNamespace as NS

import pytest


class NoCore(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "omni_cache":
            self.attempts.append(name)
            raise ImportError(name)


@pytest.fixture
def no_core():
    assert not any(name.split(".")[0] == "omni_cache" for name in sys.modules)
    guard = NoCore()
    sys.meta_path.insert(0, guard)
    try:
        yield guard
        assert not guard.attempts
    finally:
        sys.meta_path.remove(guard)


@pytest.mark.parametrize("compressed", [False, True])
def test_native_resident_bytes_without_host_main_allocation(compressed, no_core):
    import torch
    import torch_npu
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheGroupSpec, UniformTypeKVCacheSpecs
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
    import vllm_ascend.ops
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from avsg.runtime.dsa import C4_PROFILE, UNCOMPRESSED_PROFILE
    from vllm_ascend.kv_offload.avsg_host import resident_kv_config
    torch.npu.set_device(0)
    common = dict(block_size=128, num_kv_heads=1)
    if compressed:
        main = AscendMLAAttentionSpec(**common, head_size=512, dtype=torch.bfloat16, compress_ratio=4)
        index = AscendMLAAttentionSpec(**common, head_size=128, dtype=torch.int8,
            compress_ratio=4, scale_dim=1, scale_dtype=torch.float16)
    else:
        main = AscendMLAAttentionSpec(**common, head_size=656, dtype=torch.int8, cache_sparse_c8=True)
        index = AscendSFAIndexerCacheSpec(**common, head_size=128, dtype=torch.int8,
            cache_sparse_c8=True, scale_dim=1, scale_dtype=torch.float16)
    main_name, index_name = "layer0.self_attn.attn", "layer0.self_attn.indexer.k_cache"
    specs = {main_name: main, index_name: index}
    kv = KVCacheConfig(num_blocks=4, kv_cache_groups=[KVCacheGroupSpec(list(specs),
        UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))],
        kv_cache_tensors=[KVCacheTensor(size=4*s.page_size_bytes, shared_by=[n]) for n,s in specs.items()])
    projected = resident_kv_config(kv, C4_PROFILE if compressed else UNCOMPRESSED_PROFILE)
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = NS(kv_transfer_config=None)
    runner.device, runner.pin_memory = torch.device("npu:0"), False
    runner.use_sparse, runner.use_compress, runner.use_hybrid_blocks = True, compressed, False
    runner.runner_only_attn_layers = set()
    runner.c8_k_cache_dtype = torch.int8
    backend = NS(get_kv_cache_shape=lambda n,b,h,w: (n,b,h,w))
    runner.attn_backend = backend
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        NS(backend=backend, kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names)
        for g in projected.kv_cache_groups if g.layer_names]
    raw = runner._allocate_kv_cache_tensors(projected)
    cache = runner._reshape_kv_cache_tensors(projected, raw)
    assert set(raw) == set(cache) == {index_name}
    assert kv.kv_cache_groups[0].layer_names == [main_name, index_name]
    assert projected.num_blocks == kv.num_blocks == 4
    payload = cache[index_name]
    assert len(payload) == 2
    assert payload[0].shape == (4,128,1,128) and payload[0].dtype == torch.int8
    assert payload[1].shape == (4,128,1,1) and payload[1].dtype == torch.float16
    payload[0].fill_(17)
    payload[1].fill_(3)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        payload[0].fill_(19)
        payload[1].fill_(5)
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(payload[0].cpu() == 19)
    assert torch.all(payload[1].cpu() == 5)
    assert not no_core.attempts


@pytest.mark.parametrize("final_bind", [False, True, "pd_backing"])
@pytest.mark.parametrize("compressed", [False, True])
def test_native_main_storage_bootstrap_retains_host_and_zeroes_only_resident(tmp_path, compressed, final_bind, no_core):
    import gc
    import torch
    import torch_npu
    import vllm_ascend.ops
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheGroupSpec, UniformTypeKVCacheSpecs
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from avsg.runtime.dsa import C4_PROFILE, UNCOMPRESSED_PROFILE
    from avsg.runtime.main_kv import create_main_kv_file
    from avsg.runtime.arena import HostFile, HostMapping
    from dataclasses import replace
    from avsg.runtime.resident_rows import NativeResidentRows, ResidentMirror
    from vllm_ascend.kv_offload import avsg_host

    assert hasattr(avsg_host, "prepare_native_storage")
    torch.npu.set_device(0)
    profile = C4_PROFILE if compressed else UNCOMPRESSED_PROFILE
    common = dict(block_size=128, num_kv_heads=1)
    main = AscendMLAAttentionSpec(**common, head_size=512 if compressed else 656,
        dtype=torch.bfloat16 if compressed else torch.int8,
        compress_ratio=4 if compressed else 1, cache_sparse_c8=not compressed)
    index = (AscendMLAAttentionSpec(**common, head_size=128, dtype=torch.int8,
        compress_ratio=4, scale_dim=1, scale_dtype=torch.float16) if compressed
        else AscendSFAIndexerCacheSpec(**common, head_size=128, dtype=torch.int8,
        cache_sparse_c8=True, scale_dim=1, scale_dtype=torch.float16))
    specs = {"model.layers.0.self_attn.attn": main, "model.layers.0.self_attn.indexer.k_cache": index}
    if not compressed:
        specs["model.layers.1.self_attn.attn"] = AscendMLAAttentionSpec(**common, head_size=576, dtype=torch.bfloat16)
    kv = KVCacheConfig(4, [KVCacheTensor(size=4*s.page_size_bytes, shared_by=[n])
        for n, s in specs.items()], [KVCacheGroupSpec(list(specs),
        UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))])
    descriptors = None
    if final_bind == "pd_backing":
        classes, canonical = avsg_host.describe_main_kv(kv, profile)
        file = HostFile.create(tmp_path / "pd-host", 2,
            tuple((size + 256, count) for size, count in classes), ranks=1)
        descriptors = tuple(replace(layer, components=tuple(replace(span, offset=span.offset + 128)
            for span in layer.components)) for layer in canonical)
    else:
        file, _ = create_main_kv_file(tmp_path / "main-host", profile,
            avsg_host.main_kv_specs(kv, profile), capacity=2, ranks=1, budget_bytes=4194304)
    mapping = HostMapping(file, rank=0)
    mirror = ResidentMirror(1, logical_capacity=4)
    rows = NativeResidentRows(1, logical_capacity=4)
    tx = rows.reserve({0: 1})
    rows.bind(tx, 0, (2,))
    rows.finish(tx)
    delta = rows.drain()
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = NS(kv_transfer_config=None)
    runner.device, runner.pin_memory = torch.device("npu:0"), False
    runner.use_sparse, runner.use_compress, runner.use_hybrid_blocks = True, compressed, False
    runner.runner_only_attn_layers = set()
    runner.c8_k_cache_dtype = torch.int8
    runner.kv_caches = []
    backend = NS(get_kv_cache_shape=lambda n,b,h,w: (n,b,h,w))
    runner.attn_backend = backend
    projected = avsg_host.resident_kv_config(kv, profile)
    runner.attn_groups = [[
        NS(backend=backend, kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names)]
        for g in kv.kv_cache_groups]
    original_skips = runner.runner_only_attn_layers
    storage = avsg_host.prepare_native_storage(runner, kv, profile, mapping, mirror, layouts=descriptors)
    mirror.apply(delta)
    runner.kv_cache_config = kv
    if final_bind:
        runner.shared_kv_cache_layers = {}
        runner.model_config = NS(hf_text_config=NS(model_type="arbitrary_dsa", num_hidden_layers=2))
        runner.compilation_config = NS(static_forward_context={name: NS(kv_cache=None) for name in specs})
        caches = runner.initialize_kv_cache_tensors(kv)
        raw = storage._raw
        assert len(runner.kv_caches) == len(caches)
        for name, cache in caches.items():
            bound = runner.compilation_config.static_forward_context[name].kv_cache
            assert (bound[0] is cache) if compressed else (bound is cache)
    else:
        raw = runner._allocate_kv_cache_tensors(kv)
        caches = runner._reshape_kv_cache_tensors(kv, raw)
    assert runner.runner_only_attn_layers is original_skips
    assert "_kv_cache_spec_attn_group_iterator" not in vars(runner)
    assert set(raw) == {"model.layers.0.self_attn.indexer.k_cache"} and set(caches) == set(specs)
    assert caches["model.layers.0.self_attn.attn"][0] is storage.views.layer("model.layers.0.self_attn.attn").view
    assert storage.views.layer("model.layers.0.self_attn.attn").view.shape[0] == 2
    assert caches["model.layers.0.self_attn.indexer.k_cache"][0].shape[0] == 4
    storage.views._cpu.fill_(31)
    resident_raw = raw["model.layers.0.self_attn.indexer.k_cache"]
    resident_raw = (resident_raw,) if isinstance(resident_raw, torch.Tensor) else resident_raw
    for tensor in resident_raw:
        tensor.fill_(9)
    runner._init_kv_zero_meta()
    runner._zero_block_ids([2])
    torch.npu.synchronize()
    assert torch.all(storage.views._cpu == 31)
    raw_rows = resident_raw[0].reshape(4, -1)
    assert torch.all(raw_rows[2].cpu() == 0)
    assert torch.all(raw_rows[[0,1,3]].cpu() == 9)
    raw_rows[2].fill_(7)
    with pytest.raises(KeyError):
        runner._zero_block_ids([2,3])
    torch.npu.synchronize()
    assert torch.all(raw_rows[2].cpu() == 7)
    with pytest.raises(RuntimeError):
        runner._allocate_kv_cache_tensors(kv)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        caches["model.layers.0.self_attn.indexer.k_cache"][0].fill_(17)
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(caches["model.layers.0.self_attn.indexer.k_cache"][0].cpu() == 17)
    assert torch.all(storage.views._cpu == 31)
    # Destroy graphs/caches/runner before unmapping registered main backing.
    del graph, caches, raw_rows, raw, storage, runner, resident_raw, tensor
    gc.collect()
    mapping.close()
