# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native specs determine Host bytes; names never determine model capability."""
import gc
from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from avsg.runtime.arena import HostMapping
from avsg.runtime.dsa import C4_PROFILE, UNCOMPRESSED_PROFILE
from avsg.runtime.main_kv import MainKVViews, create_main_kv_file
from vllm_ascend.kv_offload.avsg_host import describe_main_kv, main_kv_specs


def geometry(compressed):
    common = dict(block_size=128, num_kv_heads=1)
    if compressed:
        specs = {
            "arbitrary-main": AscendMLAAttentionSpec(**common, head_size=512,
                dtype=torch.bfloat16, compress_ratio=4),
            "arbitrary-index": AscendMLAAttentionSpec(**common, head_size=128,
                dtype=torch.int8, compress_ratio=4, scale_dim=1, scale_dtype=torch.float16),
        }
    else:
        specs = {
            "arbitrary-main": AscendMLAAttentionSpec(**common, head_size=656,
                dtype=torch.int8, cache_sparse_c8=True),
            "draft-main": AscendMLAAttentionSpec(**common, head_size=576, dtype=torch.bfloat16),
            "arbitrary-index": AscendSFAIndexerCacheSpec(**common, head_size=128, dtype=torch.bfloat16),
        }
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))
    return KVCacheConfig(num_blocks=8, kv_cache_groups=[group], kv_cache_tensors=[])


@pytest.mark.parametrize("compressed", [False, True])
def test_native_plan_reuses_existing_host_bytes(tmp_path, compressed):
    profile = C4_PROFILE if compressed else UNCOMPRESSED_PROFILE
    kv = geometry(compressed)
    classes, layouts = describe_main_kv(kv, profile)
    assert {x.name for x in layouts} == ({"arbitrary-main"} if compressed else {"arbitrary-main", "draft-main"})
    assert all(x.group == 0 for x in layouts)
    file, created_layouts = create_main_kv_file(tmp_path / "host", profile,
        main_kv_specs(kv, profile), capacity=4, ranks=1, budget_bytes=4194304)
    assert file.page_classes == classes and created_layouts == layouts
    mapping = HostMapping(file, rank=0)
    views = MainKVViews(mapping, profile, layouts)
    views._cpu.zero_()
    views.layer("arbitrary-main").view[1].fill_(7)
    assert torch.count_nonzero(views._cpu) > 0
    assert all(x.page_class < len(classes) for layer in layouts for x in layer.components)
    del views
    gc.collect()
    mapping.close()
    assert "arbitrary-index" in kv.kv_cache_groups[0].layer_names


@pytest.mark.parametrize("change", [
    dict(block_size=32), dict(head_size=1024), dict(num_kv_heads=2),
    dict(dtype=torch.float16), dict(scale_dim=1), dict(compress_ratio=8),
])
def test_reject_unproven_main_geometry(change):
    kv = geometry(True)
    group = kv.kv_cache_groups[0]
    group.kv_cache_spec.kv_cache_specs["arbitrary-main"] = replace(
        group.kv_cache_spec.kv_cache_specs["arbitrary-main"], **change)
    with pytest.raises(ValueError):
        describe_main_kv(kv, C4_PROFILE)


def test_preserve_group_ids_and_reject_missing_or_duplicate_specs():
    kv = geometry(True)
    second = geometry(True).kv_cache_groups[0]
    second.layer_names = ["second-main"]
    second.kv_cache_spec = replace(second.kv_cache_spec,
        kv_cache_specs={"second-main": second.kv_cache_spec.kv_cache_specs["arbitrary-main"]})
    kv.kv_cache_groups.append(second)
    classes, layouts = describe_main_kv(kv, C4_PROFILE)
    assert [x.group for x in layouts] == [0, 1]
    assert classes == ((128 * 1024, 2),)
    second.layer_names = ["arbitrary-main"]
    with pytest.raises(ValueError):
        describe_main_kv(kv, C4_PROFILE)


def test_no_scheduler_representative_spec_or_unknown_profile():
    kv = geometry(True)
    kv.kv_cache_groups[0].kv_cache_spec = kv.kv_cache_groups[0].kv_cache_spec.kv_cache_specs["arbitrary-main"]
    with pytest.raises(ValueError, match="finalized"):
        describe_main_kv(kv, C4_PROFILE)
    with pytest.raises(ValueError):
        describe_main_kv(geometry(True), replace(C4_PROFILE, topk=1024))

def test_bad_main_cannot_disappear_as_an_indexer_when_other_main_exists():
    kv = geometry(True)
    group = kv.kv_cache_groups[0]
    specs = group.kv_cache_spec.kv_cache_specs
    specs["other-main"] = specs["arbitrary-main"]
    group.layer_names.append("other-main")
    specs["arbitrary-main"] = replace(specs["arbitrary-main"], head_size=128)
    with pytest.raises(ValueError, match="indexer"):
        describe_main_kv(kv, C4_PROFILE)

@pytest.mark.parametrize("compressed", [False, True])
def test_resident_projection_preserves_ids_sharing_and_original_config(compressed):
    from vllm.v1.kv_cache_interface import KVCacheTensor
    from vllm_ascend.kv_offload import avsg_host
    assert hasattr(avsg_host, "resident_kv_config")
    kv = geometry(compressed)
    group = kv.kv_cache_groups[0]
    specs = group.kv_cache_spec.kv_cache_specs
    kv.kv_cache_tensors = [KVCacheTensor(size=kv.num_blocks * s.page_size_bytes,
                                       shared_by=[n]) for n,s in specs.items()]
    if compressed:
        peer = "companion.attn"
        specs[peer] = replace(specs["arbitrary-main"], compress_ratio=128)
        group.layer_names.append(peer)
        kv.kv_cache_tensors[0].shared_by.append(peer)
    before = [(x.size, tuple(x.shared_by)) for x in kv.kv_cache_tensors]
    projected = avsg_host.resident_kv_config(kv, C4_PROFILE if compressed else UNCOMPRESSED_PROFILE)
    assert projected.num_blocks == kv.num_blocks
    assert len(projected.kv_cache_groups) == len(kv.kv_cache_groups)
    assert set(projected.kv_cache_groups[0].layer_names) == (
        {"arbitrary-index", "companion.attn"} if compressed else {"arbitrary-index"})
    assert before == [(x.size, tuple(x.shared_by)) for x in kv.kv_cache_tensors]
    assert "arbitrary-main" in group.layer_names
    assert all("arbitrary-main" not in t.shared_by for t in projected.kv_cache_tensors)
    assert sum(t.size for t in projected.kv_cache_tensors) <= sum(t.size for t in kv.kv_cache_tensors)


def test_resident_projection_rejects_missing_duplicate_and_malformed_backing():
    from vllm.v1.kv_cache_interface import KVCacheTensor
    from vllm_ascend.kv_offload import avsg_host
    assert hasattr(avsg_host, "resident_kv_config")
    kv = geometry(False)
    specs = kv.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    complete = [KVCacheTensor(size=kv.num_blocks * s.page_size_bytes, shared_by=[n])
                for n,s in specs.items()]
    for plans in ([], complete[:-1], complete + [complete[0]],
                  [replace(complete[0], size=1), *complete[1:]]):
        kv.kv_cache_tensors = plans
        with pytest.raises(ValueError):
            avsg_host.resident_kv_config(kv, UNCOMPRESSED_PROFILE)


def test_failed_native_reshape_cannot_be_retried():
    from types import SimpleNamespace as NS
    from vllm_ascend.kv_offload.avsg_host import NativeMainStorage
    storage = object.__new__(NativeMainStorage)
    config, raw, calls = object(), object(), []
    storage.config, storage.projected, storage._raw = config, config, raw
    storage.views = NS(_check=lambda: None)
    storage.resident = {}
    storage._reshape_started = False
    storage._main_names = ("main",)
    skipped = {"native-only"}
    storage.runner = NS(runner_only_attn_layers=skipped)
    def fail(*args):
        calls.append(args)
        raise OSError("partial native binding")
    storage._reshape = fail
    with pytest.raises(OSError, match="partial native"):
        storage.reshape(config, raw)
    with pytest.raises(RuntimeError, match="exactly once"):
        storage.reshape(config, raw)
    assert len(calls) == 1
    assert storage.runner.runner_only_attn_layers is skipped


@pytest.mark.parametrize("compressed", [False, True])
def test_explicit_pd_layout_requires_exact_native_main_coverage(tmp_path, compressed):
    from avsg.runtime.arena import HostFile
    from avsg.runtime.main_kv import ComponentSpan, MainKVLayout
    from vllm_ascend.kv_offload import avsg_host
    profile = C4_PROFILE if compressed else UNCOMPRESSED_PROFILE
    kv = geometry(compressed)
    classes, canonical = describe_main_kv(kv, profile)
    classes = tuple((size + 256, count) for size, count in classes)
    layouts = tuple(replace(layer, components=tuple(replace(span, offset=span.offset + 128)
                    for span in layer.components)) for layer in canonical)
    file = HostFile.create(tmp_path / "pd-host", 4, classes, ranks=1)
    mapping = HostMapping(file, rank=0)
    try:
        resolved = avsg_host.resolve_main_layouts(kv, profile, mapping, layouts)
        assert resolved == layouts
        views = MainKVViews(mapping, profile, resolved)
        views._cpu.zero_()
        views.layer("arbitrary-main").view[1].fill_(7)
        assert torch.count_nonzero(views._cpu) > 0
        del views
        for bad in (layouts[:-1], layouts + layouts[:1],
                    (replace(layouts[0], group=1), *layouts[1:]),
                    (replace(layouts[0], block_size=32), *layouts[1:]),
                    (replace(layouts[0], layout="bf16_512"), *layouts[1:]) if not compressed
                        else (replace(layouts[0], layout="packed_c8_656"),)):
            with pytest.raises(ValueError):
                avsg_host.resolve_main_layouts(kv, profile, mapping, bad)
        with pytest.raises(ValueError, match="backing"):
            avsg_host.resolve_main_layouts(kv, profile, mapping)
        # Semantic acceptance never bypasses MainKVViews byte bounds validation.
        bad = (replace(layouts[0], components=(ComponentSpan(999, 0, 0),)), *layouts[1:])
        with pytest.raises(ValueError, match="page class"):
            MainKVViews(mapping, profile, avsg_host.resolve_main_layouts(kv, profile, mapping, bad))
    finally:
        gc.collect()
        mapping.close()

@pytest.mark.parametrize("compression", [1, 4])
def test_main_binding_retains_aliases_and_native_containers(compression):
    from types import SimpleNamespace as NS
    from vllm_ascend.kv_offload.avsg_host import bind_main_caches
    main, rope, indexer = (torch.zeros(2, 4) for _ in range(3))
    layers = (NS(name="packed", view=main, rope=None),
              NS(name="split", view=main, rope=rope))
    views = NS(profile=NS(compression=compression), layers=lambda: layers)
    resident = {"indexer": (indexer,)}
    caches = bind_main_caches(views, resident)
    assert type(caches["packed"]) is (list if compression == 4 else tuple)
    assert caches["packed"][0] is main
    assert caches["split"][0] is main and caches["split"][1] is rope
    assert caches["indexer"] is resident["indexer"]
    with pytest.raises(ValueError, match="overlap"):
        bind_main_caches(views, {"packed": (indexer,)})


def test_resident_zero_views_preserve_domains_shared_backing_and_padding():
    from vllm_ascend.kv_offload.avsg_host import resident_zero_views
    first, second = torch.full((4, 32), 17, dtype=torch.int8), torch.full((2, 48), 29, dtype=torch.int8)
    raw = {"a": first, "shared": first, "b": (second,)}
    views = resident_zero_views(raw, (("a", "shared"), ("b",)), (4, 2))
    assert len(views[0]) == len(views[1]) == 1
    assert views[0][0].data_ptr() == first.data_ptr()
    assert views[1][0].data_ptr() == second.data_ptr()
    views[0][0][2].zero_()
    assert torch.all(first[2] == 0)
    assert torch.all(first[[0, 1, 3]] == 17)
    assert torch.all(second == 29)
    # Zero the complete original row, including padding and scale bytes.
    views[1][0][1].zero_()
    assert torch.all(second[1] == 0) and torch.all(second[0] == 29)


@pytest.mark.parametrize("raw", [
    torch.ones(16, dtype=torch.float32),
    torch.ones(4, 8, dtype=torch.int8)[:, ::2],
    torch.ones(15, dtype=torch.int8),
])
def test_resident_zero_views_reject_invalid_raw_before_writing(raw):
    from vllm_ascend.kv_offload.avsg_host import resident_zero_views
    good = torch.full((4, 16), 23, dtype=torch.int8)
    with pytest.raises(ValueError, match="aligned"):
        resident_zero_views({"good": good, "bad": raw}, (("good", "bad"),), (4,))
    assert torch.all(good == 23)
