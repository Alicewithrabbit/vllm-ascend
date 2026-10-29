# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual native tables/storage feeding portable execution and device fences."""
import gc
import os
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.v1.worker import gpu_model_runner as base_runner
from vllm_ascend.worker import model_runner_v1 as ascend_runner
from vllm_ascend.kv_offload import avsg_host, avsg_tables
from avsg.runtime.model_records import ModelStep, ModelPage, ScheduledRange
from test_avsg_native_worker_tables_npu import bootstrap, output, consume
import test_native_resident_admission as transaction


@pytest.mark.parametrize("compressed", [False, True])
def test_native_storage_executes_bounded_host_write_and_waits_for_device(tmp_path, compressed):
    torch.ops.load_library(os.environ["AVSG_TEST_ASCEND_LIBRARY"])
    assert hasattr(avsg_host, "prepare_native_execution")
    f, runner, storage, connector, mapping, caches = bootstrap(tmp_path, compressed)
    table = avsg_tables.install_native_tables(runner, connector)
    req = transaction.request(f, "decode", 129)
    scheduled = output(f, 1, new=(req,))
    with patch.object(ascend_runner, "get_pp_group", return_value=NS(is_last_rank=True)),          patch.object(base_runner, "get_pp_group", return_value=NS(is_last_rank=True)):
        runner._update_states(scheduled)
    consume(connector, scheduled)
    storage.views._cpu.zero_()
    payloads = {v.name: caches[v.name] for v in storage.views.layers()}
    execution = avsg_host.prepare_native_execution(storage, payloads, capacity=4, rank=0)
    start, count = (0, 8) if compressed else (128, 1)
    position = 0 if compressed else 1
    ref = table.history(0, 129)[position]
    use = f.apc.reserve_model(ref, "native-model", 0, 8 if compressed else 1)
    ticket = ModelStep("model", 1, (ScheduledRange("decode", start, count, start, 0),),
                       (ModelPage("decode", position, use),))
    execution.begin_model(ticket)
    bad = execution.step((("decode", start, count),))
    original = execution.ticket
    execution.ticket = replace(ticket, writes=(replace(ticket.writes[0],
        use=replace(use, ref=replace(ref, generation=999))),))
    with pytest.raises((ValueError, KeyError)):
        bad.begin() if compressed else bad.begin(padded=4)
    assert execution.take_reports() == ()
    execution.ticket = original
    execution.active = None  # No device work was submitted by the rejected fixture.
    step = execution.step((("decode", start, count),))
    if compressed:
        step.begin()
        rows = step.rows(0, 128, capacity=2).to("npu:0")
        expected_rows = [0, 1]
    else:
        step.begin(padded=4)
        rows = execution.writes.rows
        expected_rows = [0]
    target = storage.views.layers()[0].view
    values = torch.full((len(rows), 1, target.shape[-1]), 7, dtype=target.dtype, device="npu:0")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops._C_ascend.npu_scatter_nd_update_v2(target, rows, values)
    graph.replay()
    event = torch.npu.Event()
    event.record()
    step.finish_target(NS(_event=event))
    assert not execution.take_reports()  # Submission is not completion/publication.
    execution.finish_writes()
    report, = execution.take_reports()
    assert report.sequence == 1 and report.requests[0].count == count
    assert f.tables[0].snapshot(ref).pins > 0  # Worker cannot release scheduler lease.
    alias = storage.views._cpu.view(target.dtype).as_strided(target.shape, target.stride(), target.storage_offset())
    for row in expected_rows:
        assert torch.all(alias[ref.slot, row] == 7)
    assert torch.all(alias[ref.slot, 127] == 0)
    with pytest.raises(ValueError, match="sequence"):
        execution.begin_model(ticket)
    del graph, target, alias, values, rows, step, bad, execution, payloads, table, runner, storage, caches, connector
    gc.collect()
    mapping.close()


def test_native_execution_preserves_mixed_target_draft_and_rejects_foreign_aliases(tmp_path):
    f, runner, storage, connector, mapping, caches = bootstrap(tmp_path, False, draft=True)
    payloads = {v.name: caches[v.name] for v in storage.views.layers()}
    with pytest.raises(ValueError, match="tables"):
        avsg_host.prepare_native_execution(storage, payloads, capacity=4, rank=0)
    table = avsg_tables.install_native_tables(runner, connector)
    target_name, draft_name = "model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"
    index = storage.resident["model.layers.0.self_attn.indexer.k_cache"]
    payloads[target_name] = (*payloads[target_name], *index)
    foreign = dict(payloads)
    foreign[target_name] = (*payloads[target_name][:-1], index[-1].clone())
    with pytest.raises(ValueError, match="canonical"):
        avsg_host.prepare_native_execution(storage, foreign, capacity=4, rank=0)
    with patch.object(runner, "speculative_config", None), pytest.raises(ValueError, match="partition"):
        avsg_host.prepare_native_execution(storage, payloads, capacity=4, rank=0)
    with patch.object(runner.speculative_config, "num_speculative_tokens", 7), pytest.raises(ValueError, match="query"):
        avsg_host.prepare_native_execution(storage, payloads, capacity=4, rank=0)
    execution = avsg_host.prepare_native_execution(storage, payloads, capacity=4, rank=0)
    target, = execution.target.layers()
    draft, = execution.draft.layers()
    assert target.name == target_name and draft.name == draft_name
    assert target.view is caches[target_name][0] and target.view.dtype == torch.int8
    assert draft.view is caches[draft_name][0] and draft.view.dtype == torch.bfloat16
    assert draft.rope is caches[draft_name][1]
    assert execution.payloads[target_name][-1] is index[-1]
    assert execution.target._base is execution.draft._base is storage.views._base
    # No extra Host history or resident component is allocated by partitioning.
    assert execution.target.mapping is execution.draft.mapping is mapping
    torch.npu.synchronize()
    del execution, target, draft, table, payloads, foreign, index, runner, storage, caches, connector
    gc.collect()
    mapping.close()
