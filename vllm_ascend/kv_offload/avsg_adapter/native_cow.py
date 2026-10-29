# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native-owned COW leases and atomic request-page replacement.

The caller must suspend the request while has_pending(request) is true, submit
Host-main and current Device-companion copies, and report real quiescence.
After commit it must apply the returned replacement to worker request metadata
before any model forward. This module neither executes copies nor publishes
scheduler/worker metadata; selecting it alone does not enable service COW.
"""
from dataclasses import dataclass, field
from uuid import uuid4
from avsg.runtime.cow_records import CowBatch, CowResult, CowReplacement

from avsg.runtime.pages import ComponentBinding, PageRef, PageUse, _integer








@dataclass
class _Pending:
    batch: CowBatch
    request: object
    blocks: list
    original: object
    replacement: object
    progress: tuple
    receipt: object
    reports: set = field(default_factory=set)
    cancelled: bool = False


class NativeCOW:
    def __init__(self, admission, *, ranks, is_current):
        from vllm_ascend.kv_offload.avsg_adapter.native_admission import NativeAdmission

        ranks = tuple(ranks)
        if (type(admission) is not NativeAdmission
                or admission.manager._avsg_native_admission is not admission
                or not callable(is_current) or not ranks
                or any(not _integer(r) for r in ranks) or len(set(ranks)) != len(ranks)):
            raise ValueError("COW requires native admission, request identity and unique ranks")
        self.admission, self.owner = admission, admission.apc
        self.manager, self.pool, self.pages = admission.manager, admission.pool, admission.pages
        self.ranks, self.is_current = frozenset(ranks), is_current
        self._epoch, self._serial = uuid4().hex, 0
        self._pending, self._requests = {}, {}
        self._submissions = []

    def take_batches(self):
        batches = tuple(b for b in self._submissions if b.identity in self._pending)
        self._submissions.clear()
        return batches

    def has_pending(self, request=None):
        return bool(self._pending) if request is None else id(request) in self._requests

    @staticmethod
    def _progress(request):
        return (request.num_preemptions, request.num_computed_tokens,
                request.num_tokens, request.num_output_placeholders)

    def _headroom(self, request):
        # A replacement is not future decode growth: do not deduct its credit
        # from the existing request's progress reservation.
        physical, host = map(tuple, self.admission.progress_reserve(request, {}))
        if (len(physical) != len(self.pages.capacities)
                or len(host) != len(self.owner.tables)
                or any(not _integer(n) for n in physical + host)):
            raise ValueError("Invalid native COW progress reservation")
        return physical, host

    def prepare(self, request, group, position, tokens):
        self.owner._guard()
        a = self.admission
        if (a._request is not None or a._pending is not None or a._transaction is not None
                or a._counts is not None or a._progress_hits is not None):
            raise RuntimeError("COW must start outside an active native allocation")
        if (self.has_pending(request) or not self.is_current(request) or request.is_finished()
                or not _integer(group) or group not in self.owner.tables
                or not _integer(position)):
            raise ValueError("COW requires a current request with no pending replacement")
        manager = self.manager.coordinator.single_type_managers[group]
        blocks = manager.req_to_blocks.get(request.request_id)
        if blocks is None or position >= len(blocks) or blocks[position].is_null:
            raise ValueError("COW source is not an allocated native request page")
        original = blocks[position]
        ref = self.owner.ref_for(group, original.block_id)
        table = self.owner.tables[group]
        state = table.snapshot(ref)
        # Request progress is in original tokens; compressed groups may only
        # have generated a shorter prefix. Use the same audited boundary as PD
        # admission, including uncompressed SWA/state managers.
        generated = a.token_limits(request.num_computed_tokens)[group]
        required = min(table.block_size, max(0,
            generated - position * table.block_size))
        if (state.references < 2 or not _integer(tokens, 1)
                or tokens > state.ready_tokens or tokens < required
                or a.token_limits(tokens)[group] != tokens):
            raise ValueError("COW requires a shared, jointly ready prefix")
        headroom = self._headroom(request)
        if self.pool.get_num_free_blocks() < 1 + sum(headroom[1]):
            return None
        victims = a._plan({group: 1}, headroom)
        if victims is None:
            return None

        source = self.owner.pin(ref)
        mapping_version = self.pages.sequence, self.owner._mapping_sequence
        tx = block = binding = receipt = target = None
        try:
            for victim in victims:
                self.pool.evict_blocks({victim.block_id})
            tx = self.pages.reserve({group: 1}, protected=headroom[0])
            if tx is None:
                raise RuntimeError("Admitted COW physical credit disappeared")
            block, = a._allocate_base(1)
            binding, = self.pages.bind(tx, group, [block.block_id])
            target = self.owner.bind(group, block, a.components(group, binding),
                current_bindings=lambda b=binding, g=group: a.components(g, self.pages.binding(b.logical_id)),
                on_retire=lambda b=binding: self.pages.release(b))
            receipt = self.owner.begin_receive(target, tokens, ranks=self.ranks)
            self._serial += 1
            batch = CowBatch(f"{self._epoch}:{self._serial}", request.request_id,
                group, position, tokens, source, receipt.ticket, state.bindings,
                table.snapshot(target).bindings)
            self._pending[batch.identity] = _Pending(batch, request, blocks, original,
                block, self._progress(request), receipt)
            self._requests[id(request)] = batch.identity
            self._submissions.append(batch)
            return batch
        except BaseException:
            try:
                if receipt is not None:
                    for rank in self.ranks:
                        receipt.complete(receipt.ticket, rank, "main", success=False)
                if block is not None:
                    entry = self.owner._entries.get(block.block_id)
                    bound = entry is not None
                    target = entry.ref if bound else target
                    self.pool.free_blocks((block,))
                    if binding is not None and not bound and not self.pages.release(binding):
                        raise RuntimeError("Unpublished COW binding could not be released")
                if not self.owner.complete(source):
                    raise RuntimeError("COW source lease disappeared during rollback")
                if mapping_version != (self.pages.sequence, self.owner._mapping_sequence):
                    raise RuntimeError("Uncommitted COW mappings escaped during construction")
                # These new bindings never escaped prepare(). Keep any real
                # eviction of an older native cached row, even if its logical
                # ID was reused for this failed allocation. Generations differ.
                if binding is not None:
                    self.pages.discard_unpublished(binding, mapping_version[0])
                if target is not None:
                    self.owner.discard_unpublished(target, mapping_version[1])
            except BaseException as error:
                self.owner._fault = error
                raise
            raise
        finally:
            if tx is not None:
                self.pages.finish(tx)

    def cancel(self, request):
        identity = self._requests.get(id(request))
        if identity is None:
            return False
        pending = self._pending[identity]
        if pending.cancelled:
            return False
        pending.cancelled = True
        pending.receipt.cancel()
        return True

    def _current(self, pending):
        batch, request = pending.batch, pending.request
        manager = self.manager.coordinator.single_type_managers[batch.group]
        blocks = manager.req_to_blocks.get(batch.request_id)
        return (self.is_current(request) and not request.is_finished()
                and self._progress(request) == pending.progress
                and blocks is pending.blocks and batch.position < len(blocks)
                and blocks[batch.position] is pending.original)

    def complete(self, batch, rank, *, main_success, companions_success):
        self.owner._guard()
        pending = self._pending.get(batch.identity)
        if pending is None:
            return None
        if (batch != pending.batch or not _integer(rank) or rank not in self.ranks
                or type(main_success) is not bool or type(companions_success) is not bool
                or companions_success and not main_success):
            raise ValueError("Foreign COW completion or invalid stage order")
        if rank in pending.reports:
            return None
        if not self._current(pending):
            self.cancel(pending.request)
        try:
            self.owner._usable(batch.source.ref)
        except RuntimeError:
            self.cancel(pending.request)
        receipt = pending.receipt
        receipt.complete(batch.target, rank, "main", success=main_success)
        if main_success:
            for use in receipt.uses[1:]:
                receipt.complete(batch.target, rank, use.component, success=companions_success)
        pending.reports.add(rank)
        if pending.reports != self.ranks:
            return None
        if not receipt.done:
            raise RuntimeError("COW completion omitted a component")
        success = receipt.success and not pending.cancelled and self._current(pending)
        change = None
        try:
            if success:
                # Worker row replacement is an explicit result, not an implicit
                # mutation of already-created SchedulerOutput objects.
                manager = self.manager.coordinator.single_type_managers[batch.group]
                pending.blocks[batch.position] = pending.replacement
                manager.num_cached_block[batch.request_id] = min(
                    manager.num_cached_block.get(batch.request_id, 0), batch.position)
                self.pool.free_blocks((pending.original,))
                change = CowReplacement(batch.request_id, batch.group, batch.position,
                                        batch.source.ref, batch.target.ref)
            else:
                self.pool.free_blocks((pending.replacement,))
            if not self.owner.complete(batch.source):
                raise RuntimeError("COW source lease was released outside its transaction")
        except BaseException as error:
            self.owner._fault = error
            raise
        del self._pending[batch.identity]
        del self._requests[id(pending.request)]
        return batch.request_id, success, change
