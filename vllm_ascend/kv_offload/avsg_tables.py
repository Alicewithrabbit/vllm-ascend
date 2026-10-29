# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity-only native table adapter; no compact pool or second table buffer."""

from vllm_ascend.kv_offload.avsg_adapter.source_contract import source_matches
from avsg.runtime.history_table import HostHistoryTable
from avsg.runtime.mapped_table import MappedBlockTable, guard_state_update
from avsg.runtime.page_mirror import HostPageMirror
from avsg.runtime.pages import _integer
from avsg.runtime.resident_rows import ResidentMirror
from vllm_ascend.worker import block_table as native


class NativeResidentTable(MappedBlockTable):
    def __init__(self, table, pages, host_pages, *, group):
        if not source_matches(native.__file__, 'block_table'):
            raise RuntimeError("Unsupported native block table source")
        if (type(table) is not native.BlockTable or type(pages) is not ResidentMirror
                or type(host_pages) is not HostPageMirror or table.num_blocks_per_row.any()
                or pages.logical_capacity != host_pages.logical_capacity
                or len(pages.group_domains) != len(host_pages.group_domains)
                or not _integer(group) or group >= len(pages.group_domains)
                or table.physical_block_size not in (2, 8, 32, 128) or table.dcp_world_size != 1):
            raise ValueError("Native identity tables require empty native storage and matching mirrors")
        super().__init__(table, table, pages, group)
        self.host_pages = host_pages


class NativeHostTable(HostHistoryTable, NativeResidentTable):
    def __init__(self, table, pages, host_pages, *, group):
        if table.physical_block_size != 128:
            raise ValueError("AVSG main history requires service block128")
        super().__init__(table, pages, host_pages, group=group)
        self._init_history(table, host_pages)


def install_native_tables(runner, connector):
    """Attach initialized native storage before requests, warmup or handshake.

    This does not enable attention or publish service readiness. Partial native
    state-update failure requires teardown; mapping validation failures remain
    governed by the connector's existing transactional rules.
    """
    from vllm_ascend.kv_offload.avsg_adapter.native_connector import OwnedConnector
    from vllm_ascend.kv_offload.avsg_host import NativeMainStorage
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    storage = getattr(runner, "_avsg_native_storage", None)
    if (type(runner) is not NPUModelRunner or type(storage) is not NativeMainStorage
            or storage.runner is not runner or not storage.resident or storage._zero_views is None
            or not isinstance(connector, OwnedConnector)
            or type(connector).attach_host_mappings is not OwnedConnector.attach_host_mappings
            or connector.pages is not storage.pages
            or connector.worker is None or storage.pages.sequence
            or getattr(runner, "_avsg_native_tables", None) is not None
            or getattr(runner, "_avsg_owned_tables", None) is not None):
        raise ValueError("Native tables require the same freshly initialized storage and connector")
    storage._check(runner.kv_cache_config)
    tables = runner.input_batch.block_table.block_tables
    groups = storage.config.kv_cache_groups
    if (len(tables) != len(groups) or len(tables) != len(storage.pages.group_domains)
            or any(t.physical_block_size != g.kv_cache_spec.block_size for t, g in zip(tables, groups))):
        raise ValueError("Native table groups differ from initialized KV geometry")
    original = runner._update_states
    if (getattr(original, "__func__", None) is not NPUModelRunner._update_states
            or not source_matches(original.__func__.__code__.co_filename, 'runner')):
        raise RuntimeError("Unsupported native Runner state-update ordering")
    selected = {layer.group for layer in storage.views.layers()}
    if not selected or not selected.issubset(range(len(tables))):
        raise ValueError("Native main Host layers differ from table groups")
    host = HostPageMirror((0,) * len(tables), (storage.mapping.file.capacity,),
                          logical_capacity=storage.pages.logical_capacity)
    wrapped = tuple((NativeHostTable if group in selected else NativeResidentTable)(
        table, storage.pages, host, group=group) for group, table in enumerate(tables))
    connector.attach_host_mappings(host, wrapped)
    update = guard_state_update(runner, connector, "_avsg_native_storage", tables, wrapped)

    tables[:] = wrapped
    runner._update_states = update
    runner._avsg_native_tables = wrapped
    return wrapped[0] if len(wrapped) == 1 else wrapped
