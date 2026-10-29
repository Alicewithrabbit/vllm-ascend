# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real native table arithmetic and independent Host history."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
import torch
import vllm_ascend.ops
from vllm_ascend.worker import block_table as native
from avsg.runtime.resident_rows import ResidentMirror, ResidentBinding, ResidentDelta
from avsg.runtime.page_mirror import HostPageMirror, HostDelta
from avsg.runtime.pages import PageRef


def make(device="cpu"):
    from vllm_ascend.kv_offload.avsg_tables import NativeHostTable
    with patch.object(native, "get_dcp_group", return_value=NS(world_size=1, rank_in_group=0)):
        original = native.BlockTable(128, 4, 8, 32, False, torch.device(device), kernel_sizes=[64])
    pages = ResidentMirror(1, logical_capacity=16)
    host = HostPageMirror((0,), (8,), logical_capacity=16)
    bindings = (ResidentBinding(0, 5, 1, 0, 5), ResidentBinding(0, 9, 1, 0, 9))
    refs = (PageRef("owner", 5, 1, 0), PageRef("owner", 9, 1, 6))
    pages.apply(ResidentDelta(0, 1, tuple(("bind", b) for b in bindings)))
    host.apply(HostDelta(0, 1, tuple(("bind", 0, ref) for ref in refs)))
    return NativeHostTable(original, pages, host, group=0), original, pages, host, bindings, refs


def test_identity_table_has_one_backing_and_stable_history():
    table, original, pages, host, bindings, refs = make()
    assert table.logical is table.physical is original
    table.add_row([5], 0)
    before = table.history(0, 128)
    table.append_row([9], 0)
    table.commit_block_table(1)
    assert table.get_numpy_array()[0, :4].tolist() == [10, 11, 18, 19]
    assert table.get_device_tensor(1)[0, :4].tolist() == [10, 11, 18, 19]
    assert table.history(0, 129).boundary(before, 8) == 128
    target = torch.empty((4, 8), dtype=torch.int32)
    table.project(target, (129,))
    with patch.object(host, "binding", side_effect=AssertionError("stable history scan")), \
         patch.object(torch.Tensor, "scatter_", side_effect=AssertionError("stable copy")):
        table.project(target, (130,))
    assert target[0, :2].tolist() == [0, 6]


def test_row_copy_split_swap_and_retirement_preserve_generations():
    table, original, pages, host, bindings, refs = make()
    table.add_row([5], 0)
    table.move_row(0, 1)
    table.append_row([9], 0)
    with patch.object(table, "_drop", side_effect=AssertionError("swap rebuilt native retirement index")):
        table.swap_row(0, 2)
    assert table.get_numpy_array()[1, :2].tolist() == [10, 11]
    assert table.get_numpy_array()[2, :4].tolist() == [10, 11, 18, 19]
    table.clear_row(0)
    table.retire(bindings[0]); table.retire_host(refs[0])
    assert table.get_numpy_array()[1, :2].tolist() == [0, 0]
    assert table.get_numpy_array()[2, :4].tolist() == [0, 0, 18, 19]
    with pytest.raises(ValueError):
        table.history(2, 128)
    new_binding = ResidentBinding(0, 5, 2, 0, 5)
    new_ref = PageRef("owner", 5, 2, 1)
    pages.apply(ResidentDelta(1, 2, (("free", bindings[0]), ("bind", new_binding))))
    host.apply(HostDelta(1, 2, (("free", 0, refs[0]), ("bind", 0, new_ref))))
    table.add_row([5], 1)
    table.retire(bindings[0]); table.retire_host(refs[0])
    assert table.get_numpy_array()[1, :2].tolist() == [10, 11]
    assert tuple(table.history(1, 128)) == (new_ref,)


def test_invalid_rows_or_ids_cannot_partially_modify_native_table():
    table, original, pages, host, bindings, refs = make()
    table.add_row([5], 0)
    before = table.get_cpu_tensor().clone()
    for ids, row in (([9, 3], 0), ([5] * 9, 0), ([5], -1), ([5], 4)):
        with pytest.raises((ValueError, KeyError)):
            table.add_row(ids, row)
        assert torch.equal(table.get_cpu_tensor(), before)
    table.clear()
    assert not table.num_blocks_per_row.any()


def test_npu_table_slot_arithmetic_and_host_projection_graph_replay():
    import numpy as np
    import torch_npu
    torch.npu.set_device(0)
    table, original, pages, host, bindings, refs = make("npu:0")
    table.add_row([5, 9], 0)
    table.add_row([9, 5], 1)
    table.commit_block_table(2)
    table.compute_slot_mapping_draft(np.array([0, 0]), np.array([0, 129]))
    torch.npu.synchronize()
    assert table.slot_mapping.gpu[:2].cpu().tolist() == [640, 1153]
    target = torch.empty((4, 8), dtype=torch.int32, device="npu:0")
    output = torch.empty_like(target)
    table.project(target, (129, 129))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output.copy_(target)
    graph.replay()
    torch.npu.synchronize()
    assert output[:2, :2].cpu().tolist() == [[0, 6], [6, 0]]
    table.swap_row(0, 1)
    table.project(target, (129, 129))
    graph.replay()
    torch.npu.synchronize()
    assert output[:2, :2].cpu().tolist() == [[6, 0], [0, 6]]
