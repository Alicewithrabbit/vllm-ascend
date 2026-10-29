# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native bounded write reservations shared by supported compression layouts."""
from avsg.runtime.dsa import resolve_profile
from avsg.runtime.pages import _integer
from avsg.runtime.write_geometry import C4WriteGeometry


class ModelWrites:
    """Reserve only the current write window; publish after validated completion."""
    def prepare(self, ticket, records):
        from vllm_ascend.kv_offload.avsg_adapter.native_model import ModelPage

        writes = []
        try:
            if len(ticket.requests) != len(records):
                raise ValueError('DSA model records differ from the scheduled requests')
            for window, (request, _) in zip(ticket.requests, records):
                if window.request_id != request.request_id:
                    raise ValueError('DSA model record belongs to another request')
                lane = f'{ticket.epoch}:{id(request)}:{window.preemptions}'
                end = min(window.start + window.count + self.scheduler.num_lookahead_tokens,
                          self.scheduler.max_model_len)
                for gid, (group, shape) in enumerate(zip(self.groups, self.geometry)):
                    blocks = group.req_to_blocks[window.request_id]
                    for position, start, stop in shape.spans(window.min_start, end):
                        if position >= len(blocks) or blocks[position].is_null:
                            raise RuntimeError('DSA write window lacks a native physical allocation')
                        ref = self.owner.ref_for(gid, blocks[position].block_id)
                        state = self.owner.tables[gid].snapshot(ref)
                        if state.references != 1:
                            raise RuntimeError('A shared DSA destination requires native COW')
                        if state.cached and start < state.ready_tokens:
                            self.owner.pool.evict_blocks({ref.block_id})
                        use = self.owner.reserve_model(ref, lane, start, stop)
                        # Native block IDs are global across groups. The owner
                        # retains the group, so no second wire identity is added.
                        writes.append(ModelPage(window.request_id, position, use))
        except BaseException as error:
            self.owner._fault = error
            raise
        return tuple(writes)

    def complete(self, result):
        accepted = {r.request_id: r for r in result.requests}
        scheduled = {r.request_id: r for r in result.step.requests}
        try:
            for page in result.step.writes:
                use = page.use
                group = self.owner._entry(use.ref).group
                shape = self.geometry[group]
                outcome, window = accepted[page.request_id], scheduled[page.request_id]
                base = page.position * shape.coverage
                start = max(use.start, shape.generated(outcome.start) - base)
                generated_end = outcome.start + window.count + self.scheduler.num_lookahead_tokens
                if self.clip_completion:
                    generated_end = min(self.scheduler.max_model_len, generated_end)
                end = min(use.end, shape.generated(generated_end) - base)
                tokens = None
                if start < end:
                    if outcome.publish and shape.generated(outcome.accepted) > base + start:
                        tokens = min(end, shape.generated(outcome.accepted) - base)
                else:
                    start, end = use.start, use.end  # No row was generated in this reservation.
                if not self.owner.complete_model(use, start=start, end=end, accepted=tokens):
                    raise RuntimeError('DSA reservation completed outside its native step')
        except BaseException as error:
            self.owner._fault = error
            raise


class C4ModelWrites(ModelWrites):
    def __init__(self, scheduler, owner, *, host_groups):
        manager = scheduler.kv_cache_manager
        admission = getattr(manager, '_avsg_native_admission', None)
        groups = tuple(manager.coordinator.single_type_managers)
        selected = tuple(host_groups)
        if (resolve_profile(scheduler.vllm_config).compression != 4
                or admission is None or admission.apc is not owner
                or owner.pool is not manager.block_pool
                or set(owner.tables) != set(range(len(groups)))
                or not selected or len(set(selected)) != len(selected)
                or any(not _integer(g) or g >= len(groups) for g in selected)
                or not _integer(scheduler.num_spec_tokens) or scheduler.num_spec_tokens > 7
                or not _integer(scheduler.num_lookahead_tokens) or scheduler.num_lookahead_tokens > 7
                or not _integer(scheduler.max_model_len, 1)):
            raise ValueError('DSv4 writes require audited native admission, layout groups and query1..8')
        geometry = tuple(C4WriteGeometry(group.block_size, getattr(group, 'compress_ratio', 1))
                         for group in groups)
        for gid, shape in enumerate(geometry):
            table = owner.tables[gid]
            if (table.block_size != shape.coverage or table.host_main != (gid in selected)
                    or table.host_main and shape.compression != 4):
                raise ValueError('DSv4 write coverage or residency differs from the native owner')
        self.scheduler, self.owner = scheduler, owner
        self.groups, self.geometry = groups, geometry
        self.clip_completion = True


class UncompressedModelWrites(ModelWrites):
    def __init__(self, scheduler, owner):
        groups = scheduler.kv_cache_manager.coordinator.single_type_managers
        if (len(groups) != 1 or set(owner.tables) != {0}
                or owner.pool is not scheduler.kv_cache_manager.block_pool
                or getattr(groups[0], 'compress_ratio', 1) != 1
                or groups[0].block_size not in (32, 128)
                or owner.tables[0].block_size != groups[0].block_size
                or not 0 <= scheduler.num_spec_tokens <= 3
                or not 0 <= scheduler.num_lookahead_tokens <= 3):
            raise ValueError('GLM model writes require one native uncompressed block32/128 group')
        self.scheduler, self.owner = scheduler, owner
        self.group, self.table = groups[0], owner.tables[0]
        self.groups = (self.group,)
        self.geometry = (C4WriteGeometry(self.table.block_size, 1),)
        self.clip_completion = False
