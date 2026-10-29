# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native token arithmetic with joint Host/companion capacity admission.

All callbacks execute on the scheduler thread. ``components(group, binding)``
describes the components backed by one physical row. ``progress_reserve``
returns (headroom_by_physical_domain, headroom_by_Host_group); the scheduler
adapter supplies its actual in-flight progress budget. No prefix index lives
here. Eviction uses the native free queue and the native eviction operation.
"""
from functools import wraps
import hashlib
import inspect
from pathlib import Path

from avsg.runtime.pages import _integer
from avsg.runtime.resident_rows import NativeResidentRows


def _source(obj, expected):
    if hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() not in expected:
        raise ValueError('Admission source differs from the audited contract')


class NativeAdmission:
    def __init__(self, manager, apc, pages, *, components, progress_reserve):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.single_type_kv_cache_manager import (
            FullAttentionManager, SlidingWindowManager, ChunkedLocalAttentionManager,
            SingleTypeKVCacheManager)

        groups = manager.coordinator.single_type_managers
        native_group = '26d78d1bd457f7e26debaeb2918734c7b1f0ca2830c4d90b054dff7c940741b1'
        if not callable(progress_reserve):
            raise ValueError('An explicit progress budget callback is required')
        if (type(manager) is not KVCacheManager or manager.block_pool is not apc.pool
                or hasattr(manager, '_avsg_native_admission')
                or apc._entries or apc.pool.get_num_free_blocks() != apc.pool.num_gpu_blocks - 1
                or len(groups) != len(pages.group_domains)
                or set(apc.tables) != set(range(len(groups)))
                or pages.logical_capacity != apc.pool.num_gpu_blocks
                or pages.available != tuple(n - 1 for n in pages.capacities)
                or any(table.available != table.slot_pool.capacity for table in apc.tables.values())
                or not callable(components)):
            raise ValueError('Install admission on matching fresh native/Host/physical pools')
        _source(manager.allocate_slots, {'be9c50918c8cd01736102de4020e2a9a8650675bcd4eb4227d40d2bede6bb853'})
        for method in (manager.coordinator.get_num_blocks_to_allocate,
                       manager.coordinator.allocate_new_computed_blocks):
            _source(method, {'2851d4fb414baaba68773e3bcc22d40423b26f95443a64f65370ee167f1c834c'})
        # Native rows keep IDs assigned by this pool; compact mapping remains
        # an optional, separately source-checked backend.
        if type(pages) is not NativeResidentRows:
            _source(type(pages), {'18edae3ba63fab5acaabb68cc6a7e651778102346942858c3f01e87924bffbdb'})
        _source(type(apc.pool.free_block_queue), {'3ca561818019cef84c43c6d3839627145fd049f9479d40fc733ae142d6d71df3'})
        for index, group in enumerate(groups):
            if (not isinstance(group, (FullAttentionManager, SlidingWindowManager, ChunkedLocalAttentionManager))
                    or group._get_num_evictable_blocks.__func__ is not SingleTypeKVCacheManager._get_num_evictable_blocks.__func__):
                raise ValueError('Unsupported native group admission semantics')
            for name in ('get_num_blocks_to_allocate', 'allocate_new_blocks',
                         'allocate_external_computed_blocks'):
                _source(getattr(group, name), {native_group, '4f55ca45bec165ae0d6d5bb6b0fee314a3a440a406d9560479b7c58ec42949da'})
            coverage = group.block_size * getattr(group, 'compress_ratio', 1)
            if apc.tables[index].block_size != coverage:
                raise ValueError('Host group coverage must match native logical tokens')
        self.manager, self.apc, self.pages = manager, apc, pages
        self.pool, self.tables = apc.pool, apc.tables
        self._host_pools = tuple(dict.fromkeys(table.slot_pool for table in self.tables.values()))
        self._host_domains = {g: self._host_pools.index(table.slot_pool) for g, table in self.tables.items()}
        self.components, self.progress_reserve = components, progress_reserve
        self._request = self._group = self._counts = self._capture = None
        self._pending = self._transaction = None
        self._progress_hits = None
        self._incoming = set()
        self._fresh = set()
        self._allocate_base = self.pool.get_new_blocks
        self.pool.get_new_blocks = self._allocate
        for index, group in enumerate(groups):
            group._get_num_evictable_blocks = self._wrap_hits(group._get_num_evictable_blocks)
            group.get_num_blocks_to_allocate = self._wrap_count(group.get_num_blocks_to_allocate, index)
            for name in ('allocate_new_blocks', 'allocate_external_computed_blocks'):
                setattr(group, name, self._wrap_group(getattr(group, name), index))
        coordinator = manager.coordinator
        coordinator.get_num_blocks_to_allocate = self._wrap_demand(coordinator.get_num_blocks_to_allocate)
        manager.allocate_slots = self._wrap_slots(manager.allocate_slots)
        manager._avsg_native_admission = self

    def token_limits(self, num_tokens):
        """Generated logical range, using each audited manager's compression.

        SWA specs can carry a ratio without a compressed manager; consulting
        the spec would incorrectly truncate ordinary sliding-window tokens.
        """
        self.apc._guard()
        if not _integer(num_tokens) or num_tokens > self.manager.max_model_len:
            raise ValueError('Invalid native token limit')
        ratios = (getattr(group, 'compress_ratio', 1)
                  for group in self.manager.coordinator.single_type_managers)
        return tuple(num_tokens // ratio * ratio for ratio in ratios)

    def _wrap_hits(self, original):
        @wraps(original)
        def hits(blocks):
            result = original(blocks)
            if self._capture is not None:
                self._capture[0] += result
                self._capture[1] += sum(b.block_id in self.apc._held for b in blocks if not b.is_null)
                self._incoming.update(b.block_id for b in blocks if not b.is_null)
            return result
        return hits

    def _wrap_count(self, original, group):
        @wraps(original)
        def count(*args, **kwargs):
            self.apc._guard()
            if self._capture is not None:
                raise RuntimeError('Nested native group demand')
            capture = self._capture = [0, 0]
            try:
                raw = original(*args, **kwargs)
            finally:
                self._capture = None
            physical, logical = raw - capture[0], raw - capture[1]
            if min(physical, logical) < 0:
                raise RuntimeError('Native hit accounting exceeds demand')
            if self._counts is not None:
                if group in self._counts:
                    raise RuntimeError('Duplicate group demand')
                self._counts[group] = physical
            return physical if self._progress_hits is not None else logical
        return count

    def _headroom(self, counts, new_computed_blocks):
        if self._progress_hits is not None:
            raise RuntimeError('Nested progress reservation')
        # Native logical demand includes free-queue hits. Their backing already
        # exists, so progress must count physical growth with the same hit list.
        self._progress_hits = new_computed_blocks
        try:
            physical, host = self.progress_reserve(self._request, dict(counts))
        finally:
            self._progress_hits = None
        physical, host = tuple(physical), tuple(host)
        if (len(physical) != len(self.pages.capacities) or len(host) != len(self.tables)
                or any(type(n) is not int or n < 0 for n in physical + host)):
            raise ValueError('Invalid physical/Host progress headroom')
        return physical, host

    def _plan(self, counts, headroom):
        needed = list(headroom[0])
        host_needed = [0] * len(self._host_pools)
        for group, count in counts.items():
            needed[self.pages.group_domains[group]] += count
            host_needed[self._host_domains[group]] += count + headroom[1][group]
        free = list(self.pages.available)
        host_free = [pool.available for pool in self._host_pools]

        def fits():
            return all(a >= b for a, b in zip(free, needed)) and all(
                a >= b for a, b in zip(host_free, host_needed))

        victims = []
        if fits():
            return victims
        # Walk only on capacity pressure, stopping as soon as enough native
        # eviction candidates exist. Never materialize a full-history list.
        block = self.pool.free_block_queue.fake_free_list_head.next_free_block
        while block is not None and block.next_free_block is not None:
            entry = self.apc._entries.get(block.block_id)
            if entry is not None and block.block_id not in self._incoming:
                binding = self.pages.binding(block.block_id)
                group, domain = entry.group, binding.domain
                host_domain = self._host_domains[group]
                if free[domain] < needed[domain] or host_free[host_domain] < host_needed[host_domain]:
                    if block.ref_cnt or entry.table.snapshot(entry.ref).pins:
                        raise RuntimeError('Native eviction candidate still has consumers')
                    victims.append(block)
                    free[domain] += 1
                    host_free[host_domain] += 1
                    if fits():
                        return victims
            block = block.next_free_block
        return None

    def _wrap_demand(self, original):
        @wraps(original)
        def demand(*args, **kwargs):
            self.apc._guard()
            if self._counts is not None or self._pending is not None:
                raise RuntimeError('Nested or repeated admission before allocation')
            counts = self._counts = {}
            self._incoming = set()
            try:
                logical = original(*args, **kwargs)
            finally:
                self._counts = None
            if set(counts) != set(self.tables):
                raise RuntimeError('Native coordinator omitted a cache group')
            if not sum(counts.values()):
                return logical
            hits = kwargs['new_computed_blocks'] if 'new_computed_blocks' in kwargs else args[2]
            headroom = self._headroom(counts, hits)
            victims = self._plan(counts, headroom)
            if victims is None:
                return self.pool.num_gpu_blocks + 1
            dry = kwargs.get('apply_admission_cap', args[6] if len(args) > 6 else False)
            if self._request is not None and not dry:
                self._pending = counts, headroom, victims
            return logical
        return demand

    def _wrap_slots(self, original):
        @wraps(original)
        def slots(request, *args, **kwargs):
            self.apc._guard()
            if self._request is not None:
                raise RuntimeError('Nested native allocate_slots')
            request_id = request.request_id
            # Constant work per group; never copy or traverse existing history.
            before = [(group, request_id in group.req_to_blocks,
                       len(group.req_to_blocks.get(request_id, ())),
                       group.num_cached_block.get(request_id), len(group.new_block_ids))
                      for group in self.manager.coordinator.single_type_managers]
            self._request = request
            try:
                return original(request, *args, **kwargs)
            except BaseException as error:
                try:
                    if self.apc._fault is None:
                        self._rollback(request_id, before)
                except BaseException as rollback_error:
                    self.apc._fault = rollback_error
                    raise
                finally:
                    if self.apc._fault is None:
                        self.apc._fault = error
                raise
            finally:
                if self._transaction is not None:
                    self.pages.finish(self._transaction)
                self._pending = self._transaction = self._request = None
                self._incoming.clear()
                self._fresh.clear()
        return slots

    def _rollback(self, request_id, before):
        # Revoke only newly allocated blocks. Existing cache hits retain their
        # backing; release only the references acquired by this failed call.
        self.pool.evict_blocks(self._fresh)
        for group, existed, length, cached, new_ids in before:
            blocks = group.req_to_blocks.get(request_id, [])
            added = blocks[length:]
            self.pool.free_blocks(reversed(added))
            self._fresh.difference_update(b.block_id for b in added)
            del blocks[length:]
            if not existed:
                group.req_to_blocks.pop(request_id, None)
            if cached is None:
                group.num_cached_block.pop(request_id, None)
            else:
                group.num_cached_block[request_id] = cached
            del group.new_block_ids[new_ids:]
        # Covers an exception between returning newly bound blocks and their
        # insertion into the native request list.
        self.pool.free_blocks(self.pool.blocks[i] for i in self._fresh)

    def _wrap_group(self, original, group):
        @wraps(original)
        def allocate(*args, **kwargs):
            previous = self._group
            if previous not in (None, group):
                raise RuntimeError('Nested allocation across groups')
            self._group = group
            try:
                return original(*args, **kwargs)
            finally:
                self._group = previous
        return allocate

    def _allocate(self, num_blocks):
        self.apc._guard()
        if not num_blocks:
            return self._allocate_base(num_blocks)
        if self._pending is None or self._group is None:
            raise RuntimeError('Physical allocation has no native admission')
        if self._transaction is None:
            counts, headroom, victims = self._pending
            # Native logical admission has now passed and every group's
            # incoming cache hits have been touched before external allocation.
            for victim in victims:
                self.pool.evict_blocks({victim.block_id})
            self._transaction = self.pages.reserve(counts, protected=headroom[0])
            if self._transaction is None:
                raise RuntimeError('Admitted physical credit disappeared')
        tx = self._transaction
        if num_blocks > tx.remaining.get(self._group, 0):
            raise RuntimeError('Allocation exceeds admitted group demand')
        blocks = self._allocate_base(num_blocks)
        bindings = ()
        try:
            bindings = self.pages.bind(tx, self._group, [b.block_id for b in blocks])
            for block, binding in zip(blocks, bindings):
                group = self._group
                self.apc.bind(
                    group, block, self.components(group, binding),
                    current_bindings=lambda b=binding, g=group: self.components(g, self.pages.binding(b.logical_id)),
                    on_retire=lambda b=binding: self.pages.release(b))
        except BaseException:
            unbound = [b for b in bindings if b.logical_id not in self.apc._entries]
            try:
                self.pool.free_blocks(blocks)
                for binding in unbound:
                    if not self.pages.release(binding):
                        raise RuntimeError('Failed to unwind unpublished physical binding')
            except BaseException as cleanup_error:
                self.apc._fault = cleanup_error
                raise
            raise
        self._fresh.update(b.block_id for b in blocks)
        return blocks
