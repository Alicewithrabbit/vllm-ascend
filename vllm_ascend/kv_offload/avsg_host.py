# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate finalized native DSA specs to AVSG main-only Host geometry.

This is a worker-side layout boundary, not a PD transport or scheduler. Resident
indexers, compressed companions and compressor state retain their native specs.
The returned layout is not an OmniCache/OX wire layout.
"""
from vllm_ascend.kv_offload.avsg_adapter.source_contract import source_matches
from dataclasses import replace
from copy import deepcopy
from functools import partial

import torch

from avsg.runtime.dsa import C4_PROFILE, UNCOMPRESSED_PROFILE
from avsg.runtime.main_kv import MainKVSpec, pack_main_kv, validate_main_layouts

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)


def main_kv_specs(kv_config, profile):
    if profile not in (C4_PROFILE, UNCOMPRESSED_PROFILE):
        raise ValueError("Unsupported DSA capabilities")
    selected, names = [], set()
    for gid, group in enumerate(kv_config.kv_cache_groups):
        specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        if specs is None or set(specs) != set(group.layer_names):
            raise ValueError("Main Host layout requires finalized worker specs")
        for name in group.layer_names:
            if name in names:
                raise ValueError("Duplicate native KV layer")
            names.add(name)
            spec = specs[name]
            ratio = getattr(spec, "compress_ratio", 1)
            if type(spec) is AscendSlidingWindowMLASpec and profile == C4_PROFILE:
                # Native SWA/state storage may have blocks smaller than 128.
                continue
            if type(spec) is AscendSFAIndexerCacheSpec and profile == UNCOMPRESSED_PROFILE:
                continue
            if type(spec) is not AscendMLAAttentionSpec:
                raise ValueError(f"Unsupported native KV spec for {name}")
            if profile == C4_PROFILE:
                if ratio == 128:
                    continue
                if ratio != 4:
                    raise ValueError("Unsupported main/companion compression")
                if spec.head_size == 128:
                    if (spec.num_kv_heads != 1 or spec.dtype != torch.int8
                            or spec.scale_dim != 1 or spec.scale_dtype != torch.float16
                            or spec.cache_sparse_c8):
                        raise ValueError("Unsupported compressed indexer byte layout")
                    profile.require_service_block(spec.block_size)
                    continue  # The native compressed indexer remains resident.
                layout, width, dtype, c8 = "bf16_512", 512, torch.bfloat16, False
            elif ratio == 1:
                if spec.cache_sparse_c8:
                    layout, width, dtype, c8 = "packed_c8_656", 656, torch.int8, True
                else:
                    layout, width, dtype, c8 = "bf16_512_rope64", 576, torch.bfloat16, False
            else:
                raise ValueError("Unsupported main compression")
            if (spec.num_kv_heads != 1 or spec.head_size != width or spec.dtype != dtype
                    or spec.scale_dim != 0 or spec.cache_sparse_c8 != c8
                    or spec.page_size_bytes != spec.block_size * width * dtype.itemsize):
                raise ValueError(f"Unsupported main KV byte layout for {name}")
            selected.append(MainKVSpec(name, gid, layout, spec.block_size))
    return tuple(selected)


def describe_main_kv(kv_config, profile):
    return pack_main_kv(profile, main_kv_specs(kv_config, profile))


def resolve_main_layouts(kv_config, profile, mapping, layouts=None):
    """Accept explicit backend placement only for the exact native main specs."""
    specs = main_kv_specs(kv_config, profile)
    if layouts is not None:
        return validate_main_layouts(profile, specs, layouts)
    classes, layouts = pack_main_kv(profile, specs)
    if mapping.file.page_classes != classes:
        raise ValueError("Native main backing differs from finalized spec geometry")
    return layouts


def project_resident_config(kv_config, selected, *, require_nonempty=False):
    """Filter only allocation names; retain IDs, sizes and shared backing."""
    selected = frozenset(selected)
    specs = {}
    for group in kv_config.kv_cache_groups:
        individual = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            if name in specs or individual is not None and name not in individual:
                raise ValueError("Duplicate or missing native layer spec")
            specs[name] = individual[name] if individual is not None else group.kv_cache_spec
    if not selected or not selected <= specs.keys():
        raise ValueError("Host main layers differ from native specs")
    seen, tensors, groups = set(), [], []
    for tensor in kv_config.kv_cache_tensors:
        if (getattr(tensor, "offset", 0) or getattr(tensor, "block_stride", 0)
                or len(tensor.shared_by) != len(set(tensor.shared_by))):
            raise ValueError("Unsupported shared native backing")
        for name in tensor.shared_by:
            if (name not in specs or name in seen
                    or tensor.size != kv_config.num_blocks * specs[name].page_size_bytes):
                raise ValueError("Native backing coverage or byte geometry differs")
            seen.add(name)
        peers = [name for name in tensor.shared_by if name not in selected]
        if peers:
            tensors.append(replace(tensor, shared_by=peers))
    if seen != set(specs):
        raise ValueError("Native backing coverage is incomplete")
    for group in kv_config.kv_cache_groups:
        names = [name for name in group.layer_names if name not in selected]
        if require_nonempty and not names:
            raise ValueError("Group must retain resident companion layers")
        spec = group.kv_cache_spec
        if names and getattr(spec, "kv_cache_specs", None) is not None:
            spec = replace(spec, kv_cache_specs={n: spec.kv_cache_specs[n] for n in names})
        groups.append(replace(group, layer_names=names, kv_cache_spec=spec))
    return replace(kv_config, kv_cache_tensors=tensors, kv_cache_groups=groups)


def resident_kv_config(kv_config, profile):
    """Apply native service/layout admission before allocation projection."""
    selected = {spec.name for spec in main_kv_specs(kv_config, profile)}
    describe_main_kv(kv_config, profile)
    return project_resident_config(kv_config, selected)


def prepare_native_storage(runner, kv_config, profile, mapping, pages, *, layouts=None):
    """Install only worker storage; no PD handshake or readiness is published."""
    from avsg.runtime.arena import HostMapping
    from avsg.runtime.resident_rows import ResidentMirror

    if (type(mapping) is not HostMapping or type(pages) is not ResidentMirror
            or runner.device.type != "npu" or type(runner.device.index) is not int
            or runner.device.index < 0
            or pages.sequence or pages.logical_capacity != kv_config.num_blocks
            or len(pages.group_domains) != len(kv_config.kv_cache_groups)
            or getattr(runner, "_avsg_native_storage", None) is not None
            or getattr(runner, "_avsg_owned_storage", None) is not None
            or getattr(runner, "_physical_storage", None) is not None
            or getattr(runner, "kv_cache_config", None) is not None or runner.kv_caches):
        raise ValueError("Native storage requires fresh identity rows before KV initialization")
    for name in ("initialize_kv_cache_tensors", "_allocate_kv_cache_tensors", "_reshape_kv_cache_tensors"):
        method = getattr(runner, name)
        if (method.__name__ != name
                or not source_matches(method.__func__.__code__.co_filename, 'runner')):
            raise RuntimeError("Unsupported native allocation/binding source")
    return NativeMainStorage(runner, kv_config, profile, mapping, pages, layouts=layouts)


def zero_resident_rows(pages, logical_ids, views, device, *, by_group=False):
    """Resolve all rows before submitting zeroing; never touch Host aliases."""
    selected = [[] for _ in views]
    for logical in logical_ids:
        if type(logical) is not int:
            raise ValueError("Invalid native zeroing ID")
        if logical == 0:
            for rows in selected:
                rows.append(0)
        else:
            binding = pages.binding(logical)
            selected[binding.group if by_group else binding.domain].append(binding.slot)
    for ids, tensors in zip(selected, views):
        if ids and tensors:
            indices = torch.tensor(ids, dtype=torch.int64, device=device)
            for tensor in tensors:
                tensor.index_fill_(0, indices, 0)


class NativeMainStorage:
    """Main Host aliases plus unshrunk native resident storage, instance-scoped."""
    def __init__(self, runner, kv_config, profile, mapping, pages, *, layouts=None):
        from avsg.runtime.main_kv import MainKVViews
        from avsg.runtime.registration import register_host

        self.config = deepcopy(kv_config)
        self.projected = resident_kv_config(self.config, profile)
        layouts = resolve_main_layouts(self.config, profile, mapping, layouts)
        self.runner, self.mapping, self.pages = runner, mapping, pages
        self.profile = profile
        self._main_names = tuple(layer.name for layer in layouts)
        self._allocate, self._reshape = runner._allocate_kv_cache_tensors, runner._reshape_kv_cache_tensors
        self._started, self._reshape_started = False, False
        self._raw, self._zero_views = None, None
        self.resident = {}
        # Retain partial construction on failure; teardown, not retry, releases it.
        runner._avsg_native_storage = self
        self.views = MainKVViews(mapping, profile, layouts,
            register=partial(register_host, device=runner.device.index))
        runner._allocate_kv_cache_tensors = self.allocate
        runner._reshape_kv_cache_tensors = self.reshape
        runner._init_kv_zero_meta = self.init_zero
        runner._zero_block_ids = self.zero

    def _check(self, config):
        self.views._check()
        if config != self.config:
            raise ValueError("Native cache config changed after Host registration")

    def allocate(self, config):
        self._check(config)
        if self._started:
            raise RuntimeError("Native storage already allocated or failed; restart worker")
        self._started = True
        self._raw = self._allocate(self.projected)
        return self._raw

    def reshape(self, config, raw):
        self._check(config)
        if self._raw is None or raw is not self._raw or self._reshape_started:
            raise RuntimeError("Native reshape requires the original allocation exactly once")
        self._reshape_started = True
        # Native iteration retains the original attention groups. Main views
        # already exist in Host; skip them only inside this native reshape.
        skipped = self.runner.runner_only_attn_layers
        self.runner.runner_only_attn_layers = skipped | set(self._main_names)
        try:
            self.resident = self._reshape(self.projected, raw)
        finally:
            self.runner.runner_only_attn_layers = skipped
        caches = {}
        for layer in self.views.layers():
            components = (layer.view,) if layer.rope is None else (layer.view, layer.rope)
            caches[layer.name] = list(components) if self.profile.compression == 4 else components
        if caches.keys() & self.resident.keys():
            raise ValueError("Native main and resident layers overlap")
        caches.update(self.resident)
        return caches

    def init_zero(self):
        if self._raw is None or not self.resident or self._zero_views is not None:
            raise RuntimeError("Initialize native zeroing once after cache binding")
        capacity = self.pages.logical_capacity
        groups = []
        for group in self.projected.kv_cache_groups:
            seen, views = set(), []
            for name in group.layer_names:
                raw = self._raw[name]
                components = (raw,) if isinstance(raw, torch.Tensor) else raw
                for tensor in components:
                    if (tensor.dtype != torch.int8 or not tensor.is_contiguous()
                            or tensor.numel() % (capacity * 4)):
                        raise ValueError("Native resident backing must contain aligned byte rows")
                    if tensor.data_ptr() not in seen:
                        seen.add(tensor.data_ptr())
                        views.append(tensor.reshape(capacity, -1).view(torch.int32))
            groups.append(tuple(views))
        self._zero_views = tuple(groups)

    def zero(self, logical_ids):
        if not logical_ids:
            return
        if self._zero_views is None:
            raise RuntimeError("Native resident zeroing is not initialized")
        zero_resident_rows(self.pages, logical_ids, self._zero_views,
                           self.runner.device, by_group=True)


def describe_cow_groups(kv_config, resident):
    """Translate finalized native row coverage for either resident backend."""
    from avsg.runtime.cow_copy import COWGroup
    groups = []
    for group in kv_config.kv_cache_groups:
        specs = group.kv_cache_spec.kv_cache_specs
        # SWA/compressor state managers count original tokens even when their
        # spec inherits a compression attribute. Only MLA groups compress rows.
        shapes = {(s.block_size, getattr(s, "compress_ratio", 1)
                   if type(s) is AscendMLAAttentionSpec else 1) for s in specs.values()}
        if len(shapes) != 1:
            raise ValueError("Native COW group has inconsistent row coverage")
        block, ratio = shapes.pop()
        groups.append(COWGroup(block, ratio,
            tuple(n for n in group.layer_names if n in resident)))
    return tuple(groups)


def prepare_native_cow(storage, host_pages, *, rank):
    """Retain exact native storage aliases; no pool or backing is created."""
    from avsg.runtime.cow_copy import COWStorage, MainKVCOW
    if (type(storage) is not NativeMainStorage
            or storage.runner._avsg_native_storage is not storage
            or not storage.resident):
        raise ValueError("Native COW requires initialized canonical storage")
    storage._check(storage.config)
    groups = describe_cow_groups(storage.config, storage.resident)
    block = storage.views.layers()[0].view.shape[1]
    storage.profile.require_service_block(block)
    return MainKVCOW(COWStorage(storage.views, storage.pages, storage.resident,
        storage.runner.device, block, storage.profile, groups), host_pages, rank=rank)


def prepare_native_execution(storage, payloads, *, capacity, rank):
    """Bind portable execution to initialized native aliases; publish no readiness."""
    from avsg.runtime.model_execution import CompressedExecution, UncompressedExecution
    from avsg.runtime.pages import _integer
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm_ascend.kv_offload.avsg_tables import NativeHostTable, NativeResidentTable

    if (type(storage) is not NativeMainStorage
            or storage.runner._avsg_native_storage is not storage
            or not storage.resident or storage._zero_views is None):
        raise ValueError("Native execution requires initialized canonical storage")
    runner, profile = storage.runner, storage.profile
    storage._check(runner.kv_cache_config)
    tables = getattr(runner, "_avsg_native_tables", ())
    current = runner.input_batch.block_table.block_tables
    selected = {layer.group for layer in storage.views.layers()}
    if (not tables or len(tables) != len(storage.config.kv_cache_groups)
            or len(current) != len(tables)
            or any(a is not b for a, b in zip(tables, current))
            or any(type(table) is not (NativeHostTable if group in selected else NativeResidentTable)
                   or table.pages is not storage.pages or table.host_pages is not tables[0].host_pages
                   for group, table in enumerate(tables))):
        raise ValueError("Native execution requires the installed canonical tables")
    layers = storage.views.layers()
    resident = {id(t) for tensors in storage.resident.values() for t in tensors}
    if not isinstance(payloads, dict) or set(payloads) != {layer.name for layer in layers}:
        raise ValueError("Native execution payloads differ from main layers")
    for layer in layers:
        main = (layer.view,) if layer.rope is None else (layer.view, layer.rope)
        values = payloads[layer.name]
        if (not isinstance(values, (tuple, list)) or len(values) < len(main)
                or any(a is not b for a, b in zip(values, main))
                or any(id(t) not in resident for t in values[len(main):])):
            raise ValueError("Native execution must borrow canonical main and resident tensors")
    spec = runner.speculative_config
    if spec is not None and (spec.method != profile.proposer
            or not _integer(spec.num_speculative_tokens, 1)
            or spec.num_speculative_tokens >= profile.max_queries):
        raise ValueError("Unsupported native execution proposer/query capability")
    if profile == UNCOMPRESSED_PROFILE:
        count = runner.model_config.hf_text_config.num_hidden_layers
        if not _integer(count, 1) or len(tables) != 1:
            raise ValueError("Uncompressed execution requires native target layers and one group")
        draft = tuple(layer.name for layer in layers if extract_layer_index(layer.name) >= count)
        if len(draft) != int(spec is not None):
            raise ValueError("Native target/MTP partition differs from the proposer")
        return UncompressedExecution(storage.views, tables[0], payloads,
            capacity=capacity, rank=rank, draft_names=draft)
    coverage = []
    for group in storage.config.kv_cache_groups:
        sizes = {s.block_size * getattr(s, "compress_ratio", 1)
                 for s in group.kv_cache_spec.kv_cache_specs.values()}
        if len(sizes) != 1:
            raise ValueError("Native group has inconsistent original-token coverage")
        coverage.append(sizes.pop())
    return CompressedExecution(storage.views, tables, capacity=capacity, rank=rank,
        draft=spec is not None, coverage=coverage)
