# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent native storage COW with real NPU fences and Host bytes."""
import gc
import time
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu
import vllm_ascend.ops
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.kv_offload import avsg_host
from avsg.runtime.dsa import UNCOMPRESSED_PROFILE
from avsg.runtime.arena import HostMapping
from avsg.runtime.main_kv import create_main_kv_file
from avsg.runtime.resident_rows import ResidentMirror
from avsg.runtime.page_mirror import HostPageMirror
from vllm_ascend.kv_offload.avsg_adapter.native_cow import NativeCOW
import test_native_resident_admission as transaction


@pytest.mark.parametrize("cancel", [False, True])
def test_native_storage_cow_keeps_source_pinned_until_real_completion(tmp_path, cancel):
    assert hasattr(avsg_host, "prepare_native_cow")
    torch.npu.set_device(0)
    f = transaction.setup(logical=12, host=11)
    old, source_ref = transaction.cache(f)
    a, b = transaction.request(f, "a"), transaction.request(f, "b")
    transaction.reuse(f, a); transaction.reuse(f, b)
    cow = NativeCOW(f.admission, ranks=(0,), is_current=lambda req: req is a or req is b)
    batch = cow.prepare(a, 0, 0, 125)
    common = dict(block_size=128, num_kv_heads=1)
    specs = {
        "model.layers.0.self_attn.attn": AscendMLAAttentionSpec(**common, head_size=656, dtype=torch.int8, cache_sparse_c8=True),
        "model.layers.1.self_attn.attn": AscendMLAAttentionSpec(**common, head_size=576, dtype=torch.bfloat16),
        "model.layers.0.self_attn.indexer.k_cache": AscendSFAIndexerCacheSpec(**common, head_size=128, dtype=torch.int8,
            cache_sparse_c8=True, scale_dim=1, scale_dtype=torch.float16),
    }
    kv = KVCacheConfig(12, [KVCacheTensor(12*s.page_size_bytes, [n]) for n,s in specs.items()],
        [KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))])
    file, _ = create_main_kv_file(tmp_path / "host", UNCOMPRESSED_PROFILE,
        avsg_host.main_kv_specs(kv, UNCOMPRESSED_PROFILE), capacity=11, ranks=1, budget_bytes=4194304)
    mapping = HostMapping(file, rank=0)
    pages = ResidentMirror(1, logical_capacity=12)
    host = HostPageMirror((0,), (11,), logical_capacity=12)
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = NS(kv_transfer_config=None)
    runner.device, runner.pin_memory = torch.device("npu:0"), False
    runner.use_sparse, runner.use_compress, runner.use_hybrid_blocks = True, False, False
    runner.runner_only_attn_layers = set()
    runner.c8_k_cache_dtype, runner.kv_caches = torch.int8, []
    backend = NS(get_kv_cache_shape=lambda n,b,h,w: (n,b,h,w))
    runner.attn_backend = backend
    runner.attn_groups = [[NS(backend=backend, kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names)]
                          for g in kv.kv_cache_groups]
    storage = avsg_host.prepare_native_storage(runner, kv, UNCOMPRESSED_PROFILE, mapping, pages)
    runner.kv_cache_config = kv
    raw = runner._allocate_kv_cache_tensors(kv)
    caches = runner._reshape_kv_cache_tensors(kv, raw)
    pages.apply(f.pages.drain())
    host.apply(f.apc.drain_mappings())
    copier = avsg_host.prepare_native_cow(storage, host, rank=0)
    storage.views._cpu.fill_(19)
    for layer in storage.views.layers():
        for tensor in (layer.view,) if layer.rope is None else (layer.view, layer.rope):
            tensor[batch.source.ref.slot].fill_(3)
    for tensors in storage.resident.values():
        for tensor in tensors:
            tensor.fill_(21)
            tensor[batch.source_bindings[0].slot].fill_(7)
    torch.npu.synchronize()  # Fixture initialization must finish before the CPU byte snapshot.
    before = storage.views._cpu.clone()
    with pytest.raises(ValueError):
        copier.submit(replace(batch, source=replace(batch.source,
            ref=replace(batch.source.ref, generation=999))))
    assert not copier.pending and torch.equal(before, storage.views._cpu)
    copier.submit(batch)
    assert f.tables[0].snapshot(batch.source.ref).pins > 0
    if cancel:
        assert cow.cancel(a)
    assert f.tables[0].is_live(batch.target.ref)
    deadline = time.monotonic() + 15
    reports = ()
    while not reports:
        reports = copier.poll()
        assert time.monotonic() < deadline
    assert len(reports) == 1
    for layer in storage.views.layers():
        for tensor in (layer.view,) if layer.rope is None else (layer.view, layer.rope):
            # Audit the canonical CPU mapping after the completion fence.
            # torch_npu .cpu() on an unaligned registered-Host subview can
            # crash even with no COW; it is not a valid byte-check transport.
            alias = storage.views._cpu.view(tensor.dtype).as_strided(
                tensor.shape, tensor.stride(), tensor.storage_offset())
            assert torch.all(alias[batch.target.ref.slot, :125] == 3)
            assert torch.all(alias[batch.target.ref.slot, 125:].view(torch.uint8) == 19)
            del alias
    for tensors in storage.resident.values():
        for tensor in tensors:
            assert torch.all(tensor[batch.target_bindings[0].slot, :125].cpu() == 7)
            assert torch.all(tensor[batch.target_bindings[0].slot, 125:].cpu() == 21)
    result, = reports
    assert cow.complete(result.batch, result.rank, main_success=True, companions_success=True)[1] is not cancel
    assert f.manager.get_blocks("b").blocks[0][0] is old
    assert not copier.pending
    del layer, tensor, tensors, copier, storage, runner, caches, raw
    gc.collect()
    mapping.close()
