# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Thin runner/SFA adapter for the external GLM decode runtimes."""


def glm_model_forward(runner, context, forward, padded_tokens=None):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from avsg.providers.omnicache_glm import GlmHostViews, GlmStoredStep, GlmMtpStep
    from avsg.runtime.glm import GlmDecode
    from avsg.runtime.glm_graph import GlmGraphDecode
    from vllm_ascend import envs

    config = runner.vllm_config
    direct = envs.VLLM_ASCEND_AVSG_HOST_PROVIDER == "omnicache-glm-direct"
    transfer = config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer or context.attn_metadata is None:
        return forward()
    if getattr(runner, "_avsg_glm_failed", False):
        raise RuntimeError("GLM runner failed; retain owners until teardown")
    if getattr(runner, "_avsg_glm_step", None) is not None:
        raise RuntimeError("GLM previous step still requires draft completion")
    full_mode = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    state = getattr(runner, "_avsg_glm_state", None)
    if full_mode and state is None:
        raise RuntimeError("GLM FULL graph has not been warmed")
    if context.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        raise RuntimeError("GLM supports eager or FULL decode only")
    mtp = runner.speculative_config is not None
    if mtp and (not getattr(runner.speculative_config, "enforce_eager", False)
                or not 1 <= getattr(runner.speculative_config, "num_speculative_tokens", 0) <= 3):
        raise RuntimeError("GLM MTP requires one to three eager draft passes")
    if config.model_config.hf_text_config.model_type != "glm_moe_dsa":
        raise ValueError("GLM adapter requires glm_moe_dsa")
    if not isinstance(context.attn_metadata, dict):
        raise ValueError("GLM runner requires per-layer attention metadata")
    batch = runner.input_batch
    count = batch.num_reqs
    connector = get_kv_transfer_group()
    if state is None:
        if direct:
            from avsg.providers.omnicache_glm_store import GlmMainWrites

            worker = connector._get_connector_worker()._offloading
            views = getattr(runner, "_avsg_glm_views", None)
            if views is None or views._kv_config is not runner.kv_cache_config:
                raise RuntimeError("GLM direct runner has no registered borrowed main views")
            writes = GlmMainWrites(worker, views, max_tokens=config.scheduler_config.max_num_seqs * 4)
        else:
            worker, views = GlmHostViews.attach(connector, runner.kv_cache_config, config)
            writes = None
        decode = None if direct else GlmDecode(config.scheduler_config.max_num_seqs)
        graph = GlmGraphDecode(views.split_execution()[0] if mtp else views,
                               config.scheduler_config.max_num_seqs, writes=writes)
        state = connector, runner.kv_cache_config, worker, views, decode, graph
        runner._avsg_glm_state = state
        runner._avsg_glm_mtp_enabled = mtp
    if state[0] is not connector or state[1] is not runner.kv_cache_config:
        raise RuntimeError("GLM connector or cache allocation changed; restart the worker")
    _, _, worker, views, eager, graph = state
    if (graph.main_writes is not None) != direct:
        raise RuntimeError("GLM runtime selection changed; restart the worker")
    if getattr(runner, "_avsg_glm_mtp_enabled", mtp) != mtp:
        raise RuntimeError("GLM speculative configuration changed; restart the worker")
    execution_views = views.split_execution()[0] if mtp else views
    main_metadata = tuple(context.attn_metadata.get(layer.name) for layer in execution_views.layers())
    if any(item is None for item in main_metadata):
        raise ValueError("GLM runner requires metadata for every registered main layer")
    # Split indexer entries legitimately carry None; shared main metadata is
    # scoped once even when multiple layer names refer to the same object.
    metadata = tuple({id(item): item for item in main_metadata}.values())
    if not metadata or any(getattr(m, "avsg_decode", None) is not None for m in metadata):
        raise RuntimeError("GLM attention metadata has no available model-step scope")
    if not count:
        if worker._active_plans:
            raise RuntimeError("GLM capture cannot run with active Core plans")
        requests = len(metadata[0].cum_query_lens)
        if (not padded_tokens or not requests or padded_tokens % requests
                or any(len(m.cum_query_lens) != requests for m in metadata)):
            raise ValueError("GLM capture requires a uniform padded query bucket")
        counts = (padded_tokens // requests,) * requests
        widths = {m.block_table.shape[1] for m in metadata}
        if len(widths) != 1:
            raise ValueError("GLM capture requires equal uncompressed table widths")
        empty_replay = full_mode and counts in graph.captured
        try:
            graph.prepare_bucket(counts, widths.pop(),
                {v.name: views.payload(worker, v.name) for v in execution_views.layers()})
            if empty_replay:
                graph.begin_empty(counts)
            else:
                graph.begin_capture(counts)
        except BaseException:
            graph.failed = True
            torch.npu.synchronize()
            raise
        decode = graph
    else:
        if context.capturing:
            raise RuntimeError("GLM cannot capture a live request batch")
        starts = runner.query_start_loc.cpu[:count + 1]
        if starts.device.type != "cpu" or starts.ndim != 1 or len(starts) != count + 1:
            raise ValueError("GLM query starts must use the runner CPU metadata")
        offsets = starts.tolist()
        if offsets[0] != 0:
            raise ValueError("GLM query offsets must begin at zero")
        before = batch.num_computed_tokens_cpu[:count]
        if getattr(runner, "use_async_spec_decode", False):
            # _prepare_inputs has waited for the previous sampling-copy event
            # and corrected these CPU lengths using the native batch remap.
            # InputBatch itself deliberately retains optimistic bookkeeping.
            ends = getattr(runner, "optimistic_seq_lens_cpu", None)
            if (not getattr(runner, "_needs_seq_lens_cpu_sync", False)
                    or not isinstance(ends, torch.Tensor) or ends.device.type != "cpu"
                    or ends.ndim != 1 or ends.dtype not in (torch.int32, torch.int64) or len(ends) < count):
                raise RuntimeError("GLM async decode requires corrected CPU sequence lengths")
            corrected = [end - (offsets[i + 1] - offsets[i]) for i, end in enumerate(ends[:count].tolist())]
            if any(not 0 <= int(optimistic) - actual <= runner.speculative_config.num_speculative_tokens
                   for optimistic, actual in zip(before, corrected)):
                raise ValueError("GLM async length correction exceeds the speculative window")
            before = corrected
        scheduled = tuple((rid, int(before[i]), offsets[i + 1] - offsets[i])
                          for i, rid in enumerate(batch.req_ids[:count]))
        if direct:
            from avsg.providers.omnicache_glm_direct import GlmDirectStep

            step = GlmDirectStep(views, worker, scheduled, graph.main_writes)
        else:
            step = (GlmMtpStep if mtp else GlmStoredStep)(views, worker, scheduled)
        if mtp:
            runner._avsg_glm_step = step
        decode = graph if full_mode or direct else eager
        if full_mode or direct:
            counts = step.query_counts()
            if full_mode and padded_tokens != sum(counts):
                if (not padded_tokens or len(set(counts)) != 1 or padded_tokens % counts[0]
                        or padded_tokens < sum(counts)):
                    raise ValueError("GLM padding requires a uniform captured query bucket")
                counts = (counts[0],) * (padded_tokens // counts[0])
            try:
                if direct and not full_mode:
                    widths = {m.block_table.shape[1] for m in metadata}
                    if len(widths) != 1:
                        raise ValueError("GLM direct eager metadata has inconsistent table widths")
                    graph.prepare_bucket(counts, widths.pop(),
                        {v.name: views.payload(worker, v.name) for v in execution_views.layers()})
                    graph.begin(step, counts, eager=True, padded=padded_tokens or sum(counts))
                else:
                    graph.begin(step, counts)
            except BaseException:
                graph.failed = True
                torch.npu.synchronize()
                raise
        else:
            eager.begin(step)
    previous = []
    success = False
    try:
        for item in metadata:
            previous.append((item, hasattr(item, "avsg_decode"), getattr(item, "avsg_decode", None)))
            item.avsg_decode = decode
        result = forward()
        success = True
        return result
    finally:
        if mtp and not success:
            runner._avsg_glm_failed = True
        try:
            if not count:
                if not success:
                    graph.failed = True
                    torch.npu.synchronize()
                else:
                    torch.npu.synchronize()
                    if empty_replay:
                        if context.capturing:
                            raise RuntimeError("GLM empty replay unexpectedly recaptured the model")
                        graph.end_empty()
                    else:
                        graph.end_capture(captured=context.capturing)
            elif full_mode or direct:
                dependency = worker._record_device_dependency()
                if dependency is None:
                    raise RuntimeError("GLM graph completion was not recorded")
                graph.end(dependency, success=success)
            else:
                event = torch.npu.Event()
                event.record()
                if success:
                    eager.finish(event)
                else:
                    eager.abort(event)
        except BaseException:
            decode.failed = True
            runner._avsg_glm_failed = True
            if full_mode or direct or not count:
                torch.npu.synchronize()
            raise
        finally:
            for item, existed, prior in previous:
                if existed:
                    item.avsg_decode = prior
                else:
                    del item.avsg_decode


def glm_draft_forward(proposer, context, forward, multi_metadata, *, dummy=False):
    """Scope one whole merged draft; the native proposer still owns token logic."""
    import torch
    from vllm.config import CUDAGraphMode
    from avsg.runtime.glm import GlmDraft

    runner = proposer.runner
    transfer = runner.vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer:
        return forward()
    if getattr(runner, "_avsg_glm_failed", False):
        raise RuntimeError("GLM runner failed; retain owners until teardown")
    state = getattr(runner, "_avsg_glm_state", None)
    step = getattr(runner, "_avsg_glm_step", None)
    if step is None:
        if (dummy or runner.input_batch.num_reqs == 0) and (state is None or not state[2]._active_plans):
            return forward()
        raise RuntimeError("GLM draft has no completed target step")
    worker = state[2]
    previous = []
    try:
        batch = runner.input_batch
        if tuple(batch.req_ids[:batch.num_reqs]) != tuple(rid for rid, _, _ in step.scheduled):
            raise RuntimeError("GLM request slots changed between target and draft")
        if (dummy or proposer.method != "mtp" or proposer.parallel_drafting
                or context.cudagraph_runtime_mode != CUDAGraphMode.NONE
                or context.capturing):
            raise RuntimeError("GLM merged draft requires serial eager execution")
        name = step.draft_views.layers()[0].name
        if not multi_metadata or len(multi_metadata) != proposer.num_speculative_tokens:
            raise ValueError("GLM requires metadata for every draft pass")
        metadata = []
        for entries in multi_metadata:
            item = entries.get(name) if isinstance(entries, dict) else None
            if item is None or getattr(item, "avsg_decode", None) is not None or getattr(item, "avsg_mtp", None) is not None:
                raise ValueError("GLM draft metadata is missing or already in use")
            metadata.append(item)
        factory = getattr(step, "draft_scope", None)
        scope = factory(proposer, tuple(metadata)) if factory is not None else GlmDraft(step, proposer.num_speculative_tokens)
        runner._avsg_glm_draft = scope
        for item in {id(m): m for m in metadata}.values():
            previous.append((item, hasattr(item, "avsg_mtp"), getattr(item, "avsg_mtp", None)))
            item.avsg_mtp = scope
        result = forward()
        dependency = worker._record_device_dependency()
        if dependency is None:
            raise RuntimeError("GLM draft completion was not recorded")
        scope.finish(dependency)
        runner._avsg_glm_step = None
        runner._avsg_glm_draft = None
        return result
    except BaseException:
        runner._avsg_glm_failed = True
        step.failed = True
        # Includes failures before event creation or halfway through a pass.
        # Retain all references until the caller tears down the worker.
        torch.npu.synchronize()
        raise
    finally:
        for item, existed, prior in previous:
            if existed:
                item.avsg_mtp = prior
            else:
                del item.avsg_mtp


def glm_finalize_step(runner, finalize):
    """Never let a skipped/failed draft reach the connector publication path."""
    if getattr(runner, "_avsg_glm_failed", False) or getattr(runner, "_avsg_glm_step", None) is not None:
        import torch

        runner._avsg_glm_failed = True
        torch.npu.synchronize()
        raise RuntimeError("GLM step still requires successful draft completion")
    return finalize()


def execute_glm_attention(impl, decode, q_nope, q_rope, cache, indices, metadata):
    # Import after worker operator registration; DeviceOperator participates
    # in Ascend's operator/quantization initialization order.
    from vllm_ascend.device.device_op import DeviceOperator

    if getattr(decode, "graph_mode", False):
        if impl.enable_dsa_cp or q_nope.shape[0] != indices.shape[0] or q_rope.shape[0] != indices.shape[0]:
            raise ValueError("GLM graph attention requires padded TP1 query geometry")
        selected, table, remapped, cumulative, lengths = decode.select(impl.layer_name, cache, indices)
        pool = decode.consumer.pool(impl.layer_name)
        components = ((selected,) if pool.source_rope.shape[-1] == 0 else
                      (selected, pool.selection_rope.unsqueeze(2)))
        return DeviceOperator.execute_sparse_flash_attention_process(
            impl, q_nope, q_rope, components, remapped, metadata, cumulative, lengths, block_table=table)

    count = sum(decode.active.query_counts())
    if (impl.enable_dsa_cp or q_nope.shape[0] < count
            or q_rope.shape[0] != q_nope.shape[0] or indices.shape[0] < count):
        raise ValueError("GLM AVSG requires D TP1 and the exact scheduled query prefix")
    selected, table, remapped, cumulative, lengths = decode.select(
        impl.layer_name, cache, indices[:count])
    result = DeviceOperator.execute_sparse_flash_attention_process(
        impl, q_nope[:count], q_rope[:count], selected, remapped, metadata,
        cumulative, lengths, block_table=table)
    if q_nope.shape[0] == count:
        return result
    padded = result.new_zeros((q_nope.shape[0], *result.shape[1:]))
    padded[:count].copy_(result)
    return padded
