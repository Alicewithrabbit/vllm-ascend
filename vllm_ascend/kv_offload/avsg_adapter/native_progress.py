# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Scheduler progress credit shared by independent Host and Device backing."""
from vllm.v1.request import RequestStatus

from avsg.runtime.pages import _integer
from vllm_ascend.kv_offload.avsg_adapter.native_admission import _source


class NativeProgressReserve:
    def __init__(self, manager, requests, *, advance, lookahead):
        if not _integer(advance) or not advance or not _integer(lookahead):
            raise ValueError('Invalid native progress budget')
        self.manager, self.requests = manager, requests
        self.advance, self.lookahead = advance, lookahead

    def _admission(self):
        admission = getattr(self.manager, '_avsg_native_admission', None)
        if admission is None or admission.progress_reserve is not self:
            raise RuntimeError('Progress reserve requires matching guarded native admission')
        admission.apc._guard()
        return admission

    def _counts(self, request_id, computed, advance):
        main = min(computed + advance, self.manager.max_model_len)
        total = min(main + self.lookahead, self.manager.max_model_len)
        admission = self.manager._avsg_native_admission
        incoming = (admission._progress_hits if admission._request is not None
                    and admission._request.request_id == request_id else None)
        # Native managers own compression, window and draft-token arithmetic,
        # including skipped/null incoming hits. Count only additional backing.
        return tuple(group.get_num_blocks_to_allocate(
            request_id=request_id, num_tokens=total,
            new_computed_blocks=() if incoming is None else incoming[index],
            total_computed_tokens=computed, num_tokens_main_model=main)
            for index, group in enumerate(self.manager.coordinator.single_type_managers))

    def __call__(self, current, demand):
        admission = self._admission()
        groups = self.manager.coordinator.single_type_managers
        current_id = None if current is None else current.request_id
        ids = set() if current is None else {current_id}
        for group in groups:
            ids.update(group.req_to_blocks)
        base, growth, growing = ([0] * len(groups) for _ in range(3))
        # Called only when native admission needs new bindings. Visit owners,
        # including PD waiters, never their historical block lists.
        for request_id in ids:
            request = current if request_id == current_id else self.requests.get(request_id)
            if request is None:
                continue  # Asynchronously held bytes remain charged to their pools.
            computed = request.num_computed_tokens
            if request.status != RequestStatus.RUNNING:
                # Pending PD must retain its future prompt. Local replay is
                # already running: its future window has not been allocated.
                computed = max(computed, request.num_prompt_tokens)
            floor = self._counts(request_id, computed, 0)
            ceiling = self._counts(request_id, computed, self.advance)
            for group, (lo, hi) in enumerate(zip(floor, ceiling)):
                paid = demand.get(group, 0) if request_id == current_id else 0
                lo, hi = max(0, lo - paid), max(0, hi - paid)
                if hi < lo:
                    raise RuntimeError('Native progress demand decreased without a release')
                base[group] += lo
                growth[group] += hi - lo
                growing[group] += hi > lo
        host = []
        physical = [0] * len(admission.pages.capacities)
        for index, group in enumerate(groups):
            # One scheduling budget B is shared, not repeated for every owner.
            # Page staircases grow at most once per q tokens after their first
            # boundary; compressed groups grow no faster than this bound.
            q, owners = group.block_size, min(growing[index], self.advance)
            shared = (self.advance + owners * (q - 1)) // q
            count = base[index] + min(growth[index], shared)
            host.append(count)
            physical[admission.pages.group_domains[index]] += count
        return tuple(physical), tuple(host)

    def validate_single_request(self):
        """Reject an impossible configured maximum before starting service."""
        from vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowManager, ChunkedLocalAttentionManager)

        admission = self._admission()
        maximum, counts = self.manager.max_model_len, []
        for group in self.manager.coordinator.single_type_managers:
            count = group.get_num_blocks_to_allocate(
                request_id=object(), num_tokens=maximum, new_computed_blocks=(),
                total_computed_tokens=0, num_tokens_main_model=maximum)
            if isinstance(group, (SlidingWindowManager, ChunkedLocalAttentionManager)):
                peak = group.kv_cache_spec.max_admission_blocks_per_request
                _source(peak, {'28fb0bf520e7047ed20f9240fe884050bb8b3812f14fd3ed434dfc7526a73a7b'})
                # Native recycling bounds include boundary alignment. Sampling
                # only computed=max_length can land just after reclamation and
                # miss the earlier peak. Include this scheduler's draft budget.
                count = min(count, peak(self.advance + self.lookahead, maximum))
            counts.append(count)
        physical = [0] * len(admission.pages.capacities)
        host = {}
        for group, count in enumerate(counts):
            physical[admission.pages.group_domains[group]] += count
            pool = admission.tables[group].slot_pool
            host[pool] = host.get(pool, 0) + count
        if sum(counts) >= self.manager.block_pool.num_gpu_blocks:
            raise ValueError('logical capacity cannot admit one maximum-length request')
        if any(count >= capacity for count, capacity in zip(physical, admission.pages.capacities)):
            raise ValueError('physical capacity cannot admit one maximum-length request')
        if any(count > pool.capacity for pool, count in host.items()):
            raise ValueError('Host capacity cannot admit one maximum-length request')
