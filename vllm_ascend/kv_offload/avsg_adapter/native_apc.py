# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host lifecycle hooks with optional native prefix reuse for AVSG pages.

vLLM alone owns prefix hashes and request references. Bound-page annotations
carry alias lengths, not a second lookup index. All callbacks run on the
scheduler thread. Companion leases must cover every physical consumer and
remain valid until ``on_retire``; model/PD code supplies those leases.
"""
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path

from avsg.runtime.page_mirror import HostDelta


_NATIVE_SHA256 = '3fc5dfe0ab7ccc6206f6eb7886b2aad15384f92708e6b9f7e5ec7838b7379abe'


@dataclass
class _BoundPage:
    group: int
    block: object
    table: object
    ref: object
    current_bindings: object
    on_retire: object
    ranges: dict = field(default_factory=dict)


class NativeAPC:
    def __init__(self, pool, tables):
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        hooks = {'get_new_blocks': self._get_new, 'free_blocks': self._free,
                 'touch': self._touch, '_maybe_evict_cached_block': self._evict,
                 'cache_full_blocks': self._cache_full,
                 'cache_partial_block': self._cache_partial,
                 'reset_prefix_cache': self._reset, 'get_cached_block': self._lookup}
        if (type(pool) is not BlockPool
                or pool.enable_kv_cache_events
                or pool.get_num_free_blocks() != pool.num_gpu_blocks - 1
                or len(pool.cached_block_hash_to_block)):
            raise ValueError('Install only on a fresh native pool without D cache events')
        if hashlib.sha256(Path(inspect.getsourcefile(BlockPool)).read_bytes()).hexdigest() != _NATIVE_SHA256:
            raise ValueError('Native BlockPool source does not match the audited contract')
        if not pool.enable_caching:
            for name in ('cache_full_blocks', 'cache_partial_block',
                         'reset_prefix_cache', 'get_cached_block'):
                del hooks[name]
        for name in hooks:
            method = getattr(pool, name)
            if (getattr(method, '__func__', None) is not getattr(BlockPool, name)
                    or method.__module__ != BlockPool.__module__):
                raise ValueError('Native BlockPool already has lifecycle hooks')
        if (not tables or any(type(group) is not int or group < 0 for group in tables)
                or len({id(table) for table in tables.values()}) != len(tables)):
            raise ValueError('Each native group needs its own Host page table')
        if any(table.available != table.slot_pool.capacity for table in tables.values()):
            raise ValueError('Install only on empty Host pools, including shared groups')
        self.pool, self.tables = pool, dict(tables)
        self._key = make_block_hash_with_group_id
        self._base = {name: getattr(pool, name) for name in hooks}
        self._entries, self._owners = {}, {}
        self._held = set()
        self._mapping_sequence, self._mapping_changes = 0, []
        self._fault = None
        for name, method in hooks.items():
            setattr(pool, name, method)

    def _guard(self):
        if self._fault is not None:
            raise RuntimeError('Native APC adapter faulted; storage must not be reused') from self._fault

    def bind(self, group, block, bindings, *, current_bindings, on_retire):
        self._guard()
        if (group not in self.tables or block.is_null or block.ref_cnt <= 0
                or not 0 <= block.block_id < len(self.pool.blocks)
                or self.pool.blocks[block.block_id] is not block
                or block.block_hash is not None or block.block_id in self._entries
                or not callable(current_bindings) or not callable(on_retire)):
            raise ValueError('Bind a newly allocated native block with explicit companion leases')
        table = self.tables[group]
        bindings = tuple(bindings)
        if tuple(current_bindings()) != bindings:
            raise ValueError('Companion generation changed before Host binding')
        ref = table.allocate(block.block_id, bindings, block.ref_cnt)
        self._entries[block.block_id] = _BoundPage(
            group, block, table, ref, current_bindings, on_retire)
        self._owners[ref.owner] = table
        self._mapping_changes.append(('bind', group, ref))
        return ref

    def discard_unpublished(self, ref, sequence):
        """Discard only this retired Host binding; preserve other evictions."""
        events = [action for action, group, value in self._mapping_changes if value == ref]
        if (sequence != self._mapping_sequence or ref.block_id in self._entries
                or events != ['bind', 'free']):
            raise ValueError('Only a retired, unpublished Host binding can be discarded')
        self._mapping_changes[:] = [(action, group, value)
            for action, group, value in self._mapping_changes if value != ref]

    def drain_mappings(self):
        """Emit address changes only; readiness still belongs to this owner."""
        self._guard()
        if not self._mapping_changes:
            return None
        delta = HostDelta(self._mapping_sequence, self._mapping_sequence + 1,
                          tuple(self._mapping_changes))
        self._mapping_changes.clear()
        self._mapping_sequence += 1
        return delta

    def _entry(self, ref):
        self._guard()
        entry = self._entries.get(ref.block_id)
        if entry is None or entry.ref != ref:
            raise RuntimeError('Stale or foreign native Host binding')
        return entry

    def ref_for(self, group, block_id):
        self._guard()
        entry = self._entries.get(block_id)
        if entry is None or entry.group != group:
            raise RuntimeError('Logical block has no matching Host binding')
        return self._usable(entry.ref).ref

    def _valid(self, entry):
        if entry.table.check_bindings(entry.ref, entry.current_bindings()):
            return True
        # Revoke native membership as well as joint readiness. Active request
        # references and asynchronous tickets still keep their original leases.
        self.pool.evict_blocks({entry.block.block_id})
        return False

    def _usable(self, ref):
        entry = self._entry(ref)
        if not self._valid(entry):
            raise RuntimeError('Companion binding is no longer valid')
        return entry

    def pin(self, ref):
        entry = self._usable(ref)
        use = entry.table.pin(ref)
        self._sync(entry)
        return use

    def begin_write(self, ref, component, start, end):
        entry = self._usable(ref)
        use = entry.table.begin_write(ref, component, start, end)
        self._sync(entry)
        return use

    def begin_receive(self, ref, tokens, *, ranks):
        """Reserve the full canonical-row footprint before any PD submission."""
        from avsg.runtime.receive import PageReceive

        return PageReceive(self, self._usable(ref).table, ref, tokens, ranks)

    def reserve_model(self, ref, lane, start, end):
        entry = self._usable(ref)
        use = entry.table.reserve_model(ref, lane, start, end)
        self._sync(entry)
        return use

    def complete_model(self, use, *, start, end, accepted):
        self._guard()
        table = self._owners.get(use.ref.owner)
        if table is None:
            raise ValueError('Model completion belongs to another Host owner')
        entry = self._entries.get(use.ref.block_id)
        if entry is not None and entry.ref == use.ref:
            self._valid(entry)
        completed = table.complete_model(use, start=start, end=end, accepted=accepted)
        entry = self._entries.get(use.ref.block_id)
        if entry is not None and entry.ref == use.ref:
            self._sync(entry)
        return completed

    def accept(self, ref, tokens):
        entry = self._usable(ref)
        entry.table.accept(ref, tokens)

    def complete(self, use, success=True):
        self._guard()
        table = self._owners.get(use.ref.owner)
        if table is None:
            raise ValueError('Completion belongs to another Host owner')
        entry = self._entries.get(use.ref.block_id)
        if entry is not None and entry.ref == use.ref:
            self._valid(entry)
        completed = table.complete(use, success)
        entry = self._entries.get(use.ref.block_id)
        if entry is not None and entry.ref == use.ref:
            self._sync(entry)
        return completed

    def invalidate(self, ref):
        entry = self._entry(ref)
        entry.table.invalidate(ref)
        self.pool.evict_blocks({entry.block.block_id})

    def _restore(self, block):
        if block.block_id not in self._held:
            return
        assert block.ref_cnt == 0
        if block.block_hash is None:
            self.pool.free_block_queue.prepend_n([block])
        else:
            self.pool.free_block_queue.append_n([block])
        self._held.remove(block.block_id)

    def _sync(self, entry):
        block, table, ref = entry.block, entry.table, entry.ref
        if table.is_live(ref):
            table.native_state(ref, block.ref_cnt, block.block_hash is not None)
        if not table.is_live(ref):
            try:
                if entry.on_retire() is False:
                    raise RuntimeError('Companion lease retirement did not match its generation')
            except BaseException as error:
                self._fault = error
                raise
            self._mapping_changes.append(('free', entry.group, ref))
            del self._entries[block.block_id]
            self._restore(block)
        elif block.ref_cnt == 0 and table.snapshot(ref).pins:
            if block.block_id not in self._held:
                self.pool.free_block_queue.remove(block)
                self._held.add(block.block_id)
        elif block.block_id in self._held:
            self._restore(block)

    def _sync_blocks(self, blocks):
        for block in blocks:
            entry = self._entries.get(block.block_id)
            if entry is not None:
                self._sync(entry)

    def _get_new(self, num_blocks):
        self._guard()
        return self._base['get_new_blocks'](num_blocks)

    def _free(self, ordered_blocks):
        self._guard()
        blocks = tuple(ordered_blocks)
        self._base['free_blocks'](blocks)
        self._sync_blocks(blocks)

    def _touch(self, blocks):
        self._guard()
        blocks = tuple(blocks)
        for block in blocks:
            self._restore(block)
        self._base['touch'](blocks)
        self._sync_blocks(blocks)

    def _evict(self, block):
        self._guard()
        result = self._base['_maybe_evict_cached_block'](block)
        self._sync_blocks((block,))
        return result

    def _check_publication(self, block, group, block_size):
        entry = self._entries.get(block.block_id)
        if (entry is None or entry.group != group
                or entry.table.block_size != block_size):
            raise RuntimeError('Native cache publication needs a matching group binding')
        return entry

    def _annotate(self, entry, key, tokens):
        block = entry.block
        active = self.pool.cached_block_hashes_by_block.get(block.block_id, set())
        entry.ranges = {k: n for k, n in entry.ranges.items()
                        if k == block.block_hash or k in active}
        entry.ranges[key] = tokens
        self._sync(entry)

    def _cache_full(self, request, blocks, num_cached_blocks, num_full_blocks,
                    block_size, kv_cache_group_id, block_mask=None):
        self._guard()
        from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize

        selected = blocks[num_cached_blocks:num_full_blocks]
        mask = [True] * len(selected) if block_mask is None else list(block_mask)
        assert len(mask) == len(selected)
        entries, hashes = [], None
        for index, block in enumerate(selected):
            if block.is_null or not mask[index]:
                continue
            entry = self._check_publication(block, kv_cache_group_id, block_size)
            position = num_cached_blocks + index
            # COW rewinds the native cache cursor to its replacement. Later
            # unchanged pages retain their hashes; they are not promotions.
            if block.block_hash_num_tokens == (position + 1) * block_size:
                if hashes is None:
                    hashes = (request.block_hashes if block_size == self.pool.hash_block_size
                              else BlockHashListWithBlockSize(request.block_hashes,
                                  self.pool.hash_block_size, block_size))
                key = self._key(hashes[position], kv_cache_group_id)
                if block.block_hash == key and entry.ranges.get(key) == block_size:
                    if not self._valid(entry):
                        raise RuntimeError("Companion binding is no longer valid")
                    mask[index] = False
                    continue
            entries.append(entry)
        self._base['cache_full_blocks'](request, blocks, num_cached_blocks,
                                        num_full_blocks, block_size, kv_cache_group_id,
                                        block_mask=mask)
        for entry in entries:
            self._annotate(entry, entry.block.block_hash, block_size)

    def _cache_partial(self, request, block, num_tokens, kv_cache_group_id, block_size):
        self._guard()
        entry = None if block.is_null else self._check_publication(block, kv_cache_group_id, block_size)
        key = self._base['cache_partial_block'](request, block, num_tokens,
                                                kv_cache_group_id, block_size)
        if entry is not None and key is not None:
            self._annotate(entry, key, num_tokens % block_size)
        return key

    def _reset(self):
        self._guard()
        result = self._base['reset_prefix_cache']()
        if result:
            for entry in tuple(self._entries.values()):
                self._sync(entry)
        return result

    def _lookup(self, block_hash, kv_cache_group_ids):
        self._guard()
        found = []
        for group in kv_cache_group_ids:
            key = self._key(block_hash, group)
            # Enumerate candidates in the native map, including duplicates.
            # These private details are protected by the source guard above.
            candidates = self.pool.cached_block_hash_to_block._cache.get(key)
            candidates = tuple(candidates.values()) if isinstance(candidates, dict) else (candidates,)
            for block in candidates:
                entry = None if block is None else self._entries.get(block.block_id)
                if (entry is not None and entry.group == group and key in entry.ranges
                        and self._valid(entry)
                        and entry.table.can_reuse(entry.ref, entry.ranges[key])):
                    found.append(block)
                    break
            else:
                return None
        return found
