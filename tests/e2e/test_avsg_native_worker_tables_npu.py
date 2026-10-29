# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native worker ordering, not a loaded-model or PD transport test."""
import gc
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch_npu
import vllm_ascend.ops
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData, CachedRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker import gpu_model_runner as base_runner
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.worker import model_runner_v1 as ascend_runner, block_table as native
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.kv_offload import avsg_host, avsg_tables
from avsg.runtime.dsa import C4_PROFILE, UNCOMPRESSED_PROFILE
from avsg.runtime.main_kv import create_main_kv_file
from avsg.runtime.arena import HostMapping
from avsg.runtime.resident_rows import ResidentMirror
from vllm_ascend.kv_offload.avsg_adapter.native_connector import OwnedConnector, ResidentMetadata
import test_native_resident_admission as transaction


def bootstrap(tmp_path, compressed, *, draft=False):
    torch.npu.set_device(0)
    f = transaction.setup(logical=12, host=11)
    profile = C4_PROFILE if compressed else UNCOMPRESSED_PROFILE
    common = dict(block_size=128, num_kv_heads=1)
    specs = {
        "model.layers.0.self_attn.attn": AscendMLAAttentionSpec(**common,
            head_size=512 if compressed else 656,
            dtype=torch.bfloat16 if compressed else torch.int8,
            compress_ratio=4 if compressed else 1, cache_sparse_c8=not compressed),
        "model.layers.0.self_attn.indexer.k_cache": (
            AscendMLAAttentionSpec(**common, head_size=128, dtype=torch.int8,
                compress_ratio=4, scale_dim=1, scale_dtype=torch.float16) if compressed else
            AscendSFAIndexerCacheSpec(**common, head_size=128, dtype=torch.int8,
                cache_sparse_c8=True, scale_dim=1, scale_dtype=torch.float16)),
    }
    if draft:
        assert not compressed
        specs["model.layers.1.self_attn.attn"] = AscendMLAAttentionSpec(
            **common, head_size=576, dtype=torch.bfloat16)
    kv = KVCacheConfig(12, [KVCacheTensor(12*s.page_size_bytes, [n]) for n,s in specs.items()],
        [KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))])
    file, _ = create_main_kv_file(tmp_path / "host", profile,
        avsg_host.main_kv_specs(kv, profile), capacity=11, ranks=1, budget_bytes=4194304)
    mapping = HostMapping(file, rank=0)
    pages = ResidentMirror(1, logical_capacity=12)
    runner = object.__new__(ascend_runner.NPUModelRunner)
    runner.vllm_config = NS(kv_transfer_config=None)
    runner.device, runner.pin_memory = torch.device("npu:0"), False
    runner.use_sparse, runner.use_compress, runner.use_hybrid_blocks = True, compressed, False
    runner.runner_only_attn_layers = set()
    runner.c8_k_cache_dtype, runner.kv_caches = torch.int8, []
    backend = NS(get_kv_cache_shape=lambda n,b,h,w: (n,b,h,w))
    runner.attn_backend = backend
    runner.attn_groups = [[NS(backend=backend, kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names)]
                          for g in kv.kv_cache_groups]
    storage = avsg_host.prepare_native_storage(runner, kv, profile, mapping, pages)
    runner.kv_cache_config = kv
    runner.shared_kv_cache_layers = {}
    runner.model_config = NS(hf_text_config=NS(model_type="arbitrary_dsa", num_hidden_layers=1))
    runner.compilation_config = NS(static_forward_context={n: NS(kv_cache=None) for n in specs})
    caches = runner.initialize_kv_cache_tensors(kv)
    runner._init_kv_zero_meta()
    with patch.object(native, "get_dcp_group", return_value=NS(world_size=1, rank_in_group=0)):
        runner.input_batch = NPUInputBatch(4, 1024, 256, runner.device, False,
            256, [128], [[128]], max_num_blocks_per_req=[8])
    runner.requests, runner.num_prompt_logprobs, runner.encoder_cache = {}, {}, {}
    runner.late_interaction_runner = NS(on_requests_finished=lambda ids: None,
                                       register_request=lambda *args: None)
    runner.use_async_scheduling = runner.use_async_spec_decode = False
    runner.is_pooling_model = runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.speculative_config = runner.reorder_batch_threshold = None
    if draft:
        runner.speculative_config = NS(method="mtp", num_speculative_tokens=3)
    connector = OwnedConnector(NS(kv_transfer_config=NS(kv_connector_extra_config={})),
                               KVConnectorRole.WORKER, kv)
    # Strict empty transport only. No remote completion or model report is faked.
    connector.attach_worker(NS(connector_metadata_type=ResidentMetadata), pages)
    return f, runner, storage, connector, mapping, caches


def output(f, sequence, *, new=(), finished=()):
    step = SchedulerOutput.make_empty()
    for req in new:
        blocks = f.manager.allocate_slots(req, 129, delay_cache_blocks=True)
        assert blocks is not None
        ids = tuple([b.block_id for b in group] for group in blocks.blocks)
        step.scheduled_new_reqs.append(NewRequestData.from_request(req, ids))
        step.num_scheduled_tokens[req.request_id] = 129
    step.total_num_scheduled_tokens = sum(step.num_scheduled_tokens.values())
    step.finished_req_ids = set(finished)
    delta = f.pages.drain()
    step.new_block_ids_to_zero = [b.logical_id for a,b in (() if delta is None else delta.changes) if a == "bind"]
    step.kv_connector_metadata = ResidentMetadata("native-worker", sequence, delta, (), (),
                                                 host_delta=f.apc.drain_mappings())
    return step


def consume(connector, step):
    connector.bind_connector_metadata(step.kv_connector_metadata)
    connector.start_load_kv(None)
    connector.clear_connector_metadata()


@pytest.mark.parametrize("compressed", [False, True])
def test_native_worker_maps_before_zero_rows_and_retirement(tmp_path, compressed):
    assert hasattr(avsg_tables, "install_native_tables")
    f, runner, storage, connector, mapping, caches = bootstrap(tmp_path, compressed)
    original = runner.input_batch.block_table.block_tables[0]
    storage.views._cpu.fill_(31)
    for raw in storage._raw.values():
        for tensor in (raw,) if isinstance(raw, torch.Tensor) else raw:
            tensor.fill_(17)
    torch.npu.synchronize()
    with pytest.raises(RuntimeError):
        connector.get_handshake_metadata()  # This entry cannot publish service readiness.
    for name, value in (("_zero_views", None), ("pages", ResidentMirror(1, logical_capacity=12))):
        with patch.object(storage, name, value), pytest.raises(ValueError):
            avsg_tables.install_native_tables(runner, connector)
        assert connector._host_pages is None
        assert runner.input_batch.block_table.block_tables[0] is original
    with patch.object(runner, "_update_states", lambda output: None), pytest.raises(RuntimeError):
        avsg_tables.install_native_tables(runner, connector)
    assert connector._host_pages is None
    table = avsg_tables.install_native_tables(runner, connector)
    assert table.physical is table.logical is original
    with pytest.raises(ValueError):
        avsg_tables.install_native_tables(runner, connector)
    a, b = transaction.request(f, "a", 129), transaction.request(f, "b", 129, prefix=2)
    step = output(f, 1, new=(a, b))
    with patch.object(ascend_runner, "get_pp_group", return_value=NS(is_last_rank=True)),          patch.object(base_runner, "get_pp_group", return_value=NS(is_last_rank=True)):
        # Malformed paired delta must not partially mutate native requests/tables/HBM.
        bad = replace(step, kv_connector_metadata=replace(step.kv_connector_metadata, host_delta=None))
        with pytest.raises(ValueError, match="same logical pages"):
            runner._update_states(bad)
        assert not runner.requests and not table.num_blocks_per_row.any()
        assert storage.pages.sequence == 0
        runner._update_states(step)
        with pytest.raises(ValueError, match="already"):
            runner._update_states(step)
        consume(connector, step)
        with pytest.raises(ValueError, match="already"):
            runner._update_states(step)
        for req in (a, b):
            row = runner.input_batch.req_id_to_index[req.request_id]
            ids = [v.block_id for v in f.manager.get_blocks(req.request_id).blocks[0]]
            assert tuple(table.history(row, 129)) == tuple(f.apc.ref_for(0, i) for i in ids)
            for logical in ids:
                for raw in storage._raw.values():
                    for tensor in (raw,) if isinstance(raw, torch.Tensor) else raw:
                        assert torch.all(tensor.reshape(12, -1)[logical].cpu() == 0)
        assert torch.all(storage.views._cpu == 31)
        projected = torch.empty((4, 8), dtype=torch.int32, device="npu:0")
        result = torch.empty_like(projected)
        table.project(projected, (129, 129))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result.copy_(projected)
        graph.replay()
        torch.npu.synchronize()
        b_refs = tuple(table.history(runner.input_batch.req_id_to_index["b"], 129))
        # A finishes: actual native condense moves B into its row. Retire only A.
        f.manager.free(a)
        finished = output(f, 2, finished=("a",))
        finished.scheduled_cached_reqs = CachedRequestData(
            ["b"], set(), [], {}, [None], [128], [0])
        finished.num_scheduled_tokens, finished.total_num_scheduled_tokens = {"b": 1}, 1
        runner._update_states(finished)
        consume(connector, finished)
        assert runner.input_batch.req_id_to_index == {"b": 0}
        assert tuple(table.history(0, 129)) == b_refs
        table.project(projected, (129,))
        graph.replay()
        torch.npu.synchronize()
        assert result[0, :2].cpu().tolist() == [r.slot for r in b_refs]
        assert torch.all(result[1:].cpu() == 0)
        with pytest.raises(ValueError, match="ordered"):
            runner._update_states(step)
        assert tuple(table.history(0, 129)) == b_refs
        f.manager.free(b)
        ended = output(f, 3, finished=("b",))
        runner._update_states(ended)
        consume(connector, ended)
        assert not runner.input_batch.num_reqs and not connector._host_pages._live
        assert storage.pages.sequence == connector._host_pages.sequence == 3
    assert caches["model.layers.0.self_attn.attn"][0] is storage.views.layers()[0].view
    with pytest.raises(RuntimeError):
        connector.get_handshake_metadata()
    del tensor, raw, graph, projected, result, table, original, runner, storage, caches, connector
    gc.collect()
    mapping.close()


def test_native_update_failure_requires_teardown(tmp_path):
    f, runner, storage, connector, mapping, caches = bootstrap(tmp_path, False)
    table = avsg_tables.install_native_tables(runner, connector)
    req = transaction.request(f, "failure", 129)
    step = output(f, 1, new=(req,))
    with patch.object(ascend_runner, "get_pp_group", return_value=NS(is_last_rank=True)), \
         patch.object(base_runner, "get_pp_group", return_value=NS(is_last_rank=True)), \
         patch.object(runner, "_zero_block_ids", side_effect=RuntimeError("injected native failure")):
        with pytest.raises(RuntimeError, match="injected native failure"):
            runner._update_states(step)
    assert storage.pages.sequence == connector._host_pages.sequence == 1
    with pytest.raises(RuntimeError, match="worker teardown"):
        runner._update_states(step)
    with pytest.raises(RuntimeError):
        connector.get_handshake_metadata()
    del table, runner, storage, connector, caches
    gc.collect()
    mapping.close()


def test_consumed_step_cannot_zero_or_install_native_rows(tmp_path):
    f, runner, storage, connector, mapping, caches = bootstrap(tmp_path, False)
    table = avsg_tables.install_native_tables(runner, connector)
    req = transaction.request(f, "out-of-order", 129)
    step = output(f, 1, new=(req,))
    consume(connector, step)
    with patch.object(runner, "_zero_block_ids", side_effect=AssertionError("late zero")):
        with pytest.raises(ValueError, match="already consumed"):
            runner._update_states(step)
    assert not runner.requests and not table.num_blocks_per_row.any()
    del table, runner, storage, connector, caches
    gc.collect()
    mapping.close()
