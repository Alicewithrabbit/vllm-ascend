# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DSA owned runner; selected by layout capabilities."""


"""Native target/proposer scopes for owned GLM execution, with no Core STORE."""

from avsg.runtime.dsa import resolve_profile, UNCOMPRESSED_PROFILE
from vllm_ascend.kv_offload.avsg_adapter.source_contract import source_matches
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from avsg.runtime.graph import UncompressedGraphDecode


def require_replay_runner(runner, *, native=False):
    method = runner._build_attn_state
    if (method.__name__ != '_build_attn_state'
            or not source_matches(method.__func__.__code__.co_filename, 'runner')):
        raise RuntimeError('Owned worker requires the replay-capable native Runner')



def _state(runner):
    state = getattr(runner, '_avsg_owned_worker', None)
    if state is None or state.native_config is not runner.kv_cache_config:
        raise RuntimeError('Owned GLM runner requires its original bootstrapped native storage')
    state._ready()
    return state


def _uncompressed_replay_attention_state(runner, num_reqs, num_scheduled_tokens):
    """Route bounded local replay through the same causal GLM decode kernels."""
    from avsg.runtime.pages import _integer
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    _state(runner)
    config, spec = runner.vllm_config, runner.speculative_config
    transfer = config.kv_transfer_config
    drafts = 0 if spec is None else spec.num_speculative_tokens
    if (resolve_profile(config).compression != 1
            or transfer is None or not transfer.is_kv_consumer or transfer.kv_connector != 'OwnedConnector'
            or not _integer(drafts) or drafts >= UNCOMPRESSED_PROFILE.max_queries or spec is not None and spec.method != UNCOMPRESSED_PROFILE.proposer
            or not _integer(num_reqs, 1) or num_reqs != runner.input_batch.num_reqs
            or num_scheduled_tokens.shape != (num_reqs,)
            or num_scheduled_tokens.dtype.kind not in 'iu'
            or any(not 1 <= n <= drafts + 1 for n in num_scheduled_tokens.tolist())):
        raise ValueError('Owned GLM replay requires native query1..4 and matching MTP configuration')
    return AscendAttentionState.DecodeOnly if spec is None else AscendAttentionState.SpecDecoding


def _dependency():
    # GLM D TP1/CP1 main/indexer writes run on the model stream. Native
    # collective consumers must join it before the model returns; this is
    # not a replacement for a transport event, Host publication or rank ACK.
    event = torch.npu.Event()
    event.record()
    return SimpleNamespace(_event=event)


def _fail(state, error):
    state._fault = state.execution.failed = error
    state.graph.failed = True
    torch.npu.synchronize()


@contextmanager
def _scope(items, attribute, value):
    items = tuple({id(item): item for item in items}.values())
    if not items or any(item is None or getattr(item, 'avsg_decode', None) is not None
                        or getattr(item, 'avsg_mtp', None) is not None for item in items):
        raise ValueError('Owned GLM metadata is missing or already scoped')
    saved = [(item, hasattr(item, attribute), getattr(item, attribute, None)) for item in items]
    try:
        for item in items:
            setattr(item, attribute, value)
        yield
    finally:
        for item, existed, before in saved:
            if existed:
                setattr(item, attribute, before)
            else:
                delattr(item, attribute)


def _bucket(items, tokens):
    count = len(items[0].cum_query_lens)
    widths = {m.block_table.shape[1] for m in items}
    if (not count or not tokens or tokens % count or tokens // count not in (1, 2, 3, 4)
            or any(len(m.cum_query_lens) != count for m in items) or len(widths) != 1):
        raise ValueError('Owned GLM warmup requires a uniform query1..4 bucket')
    return (tokens // count,) * count, widths.pop()


def _scheduled(runner):
    batch, count = runner.input_batch, runner.input_batch.num_reqs
    starts = runner.query_start_loc.cpu[:count + 1]
    if starts.device.type != 'cpu' or starts.ndim != 1 or len(starts) != count + 1:
        raise ValueError('Owned GLM requires native CPU query offsets')
    offsets = starts.tolist()
    if offsets[0] != 0:
        raise ValueError('Owned GLM query offsets must start at zero')
    before = batch.num_computed_tokens_cpu[:count]
    if getattr(runner, 'use_async_spec_decode', False):
        ends = getattr(runner, 'optimistic_seq_lens_cpu', None)
        if (not getattr(runner, '_needs_seq_lens_cpu_sync', False) or not isinstance(ends, torch.Tensor)
                or ends.device.type != 'cpu' or ends.ndim != 1 or len(ends) < count
                or ends.dtype not in (torch.int32, torch.int64)):
            raise RuntimeError('Owned GLM async execution requires corrected native CPU lengths')
        actual = [end - offsets[i + 1] + offsets[i] for i, end in enumerate(ends[:count].tolist())]
        if any(not 0 <= int(old) - new <= runner.speculative_config.num_speculative_tokens
               for old, new in zip(before, actual)):
            raise ValueError('Owned GLM native correction exceeded its speculative window')
        before = actual
    return tuple((rid, int(before[i]), offsets[i + 1] - offsets[i])
                 for i, rid in enumerate(batch.req_ids[:count]))


def _uncompressed_model_forward(runner, context, forward, padded_tokens=None):
    transfer = runner.vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer or context.attn_metadata is None:
        return forward()  # Native memory profiling has no KV consumers.
    state = _state(runner)
    execution, graph = state.execution, state.graph
    if not isinstance(context.attn_metadata, dict):
        raise ValueError('Owned GLM requires native per-layer attention metadata')
    items = tuple(context.attn_metadata.get(v.name) for v in execution.target.layers())
    full = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    if context.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        raise RuntimeError('Owned GLM requires eager or FULL decode')
    count = runner.input_batch.num_reqs
    payloads = {v.name: execution.payloads[v.name] for v in execution.target.layers()}
    try:
        with _scope(items, 'avsg_decode', graph):
            if not count:
                if execution.ticket is not None:
                    raise RuntimeError('Owned empty capture/replay cannot replace a dispatched model ticket')
                counts, width = _bucket(items, padded_tokens)
                empty = full and counts in graph.captured
                graph.prepare_bucket(counts, width, payloads)
                graph.begin_empty(counts) if empty else graph.begin_capture(counts)
                result = forward()
                torch.npu.synchronize()
                if empty:
                    if context.capturing:
                        raise RuntimeError('Owned empty replay unexpectedly recaptured a model')
                    graph.end_empty()
                else:
                    graph.end_capture(captured=context.capturing)
                return result
            if context.capturing:
                raise RuntimeError('Owned model cannot capture a live request')
            step = execution.step(_scheduled(runner))
            counts = step.query_counts()
            padded = sum(counts) if padded_tokens is None else padded_tokens
            # Eager EP padding adds token lanes, not requests. Only FULL replay
            # needs the captured uniform bucket; eager attention uses real rows.
            if full and padded != sum(counts):
                if len(set(counts)) != 1 or padded < sum(counts) or padded % counts[0]:
                    raise ValueError('Owned GLM graph padding must match its uniform query bucket')
                counts = (counts[0],) * (padded // counts[0])
            if not full:
                widths = {m.block_table.shape[1] for m in items}
                if len(widths) != 1:
                    raise ValueError('Owned GLM eager tables have different uncompressed widths')
                graph.prepare_bucket(counts, widths.pop(), payloads)
            graph.begin(step, counts, eager=not full, padded=padded)
            if execution.draft is not None:
                # These inputs depend on the retained ticket, not target KV.
                # Upload before target submission to avoid fencing its replay.
                step.prepare_draft()
            result = forward()
            graph.end(_dependency())
            return result
    except BaseException as error:
        _fail(state, error)
        raise


class _EmptyDraft:
    """Eager dummy MTP stages main writes but never authorizes a Host row."""
    graph_mode = True

    def __init__(self, state, metadata):
        execution = state.execution
        if execution.draft is None or execution.ticket is not None:
            raise RuntimeError('Owned dummy draft requires no live model ticket')
        graph = getattr(state, '_empty_draft_graph', None)
        if graph is None:
            graph = state._empty_draft_graph = UncompressedGraphDecode(execution.draft, execution.capacity,
                                                           writes=execution.writes)
        self.graph, self.metadata, self.calls = graph, metadata, 0
        self.name = execution.draft.layers()[0].name
        self.payloads = {self.name: execution.payloads[self.name]}
        self.consumer, self.main_writes = graph.consumer, graph.main_writes

    def prepare(self, name):
        if name != self.name or self.calls >= len(self.metadata):
            raise ValueError('Owned dummy MTP changed its layer or pass count')
        item = self.metadata[self.calls]
        counts, width = _bucket((item,), len(item.slot_mapping))
        self.graph.prepare_bucket(counts, width, self.payloads)
        self.graph.begin_capture(counts)
        self.graph.prepare(name)
        self.slot_mapping = self.graph.slot_mapping
        self.calls += 1

    def preprocess(self, name, forward, **kwargs):
        return self.graph.preprocess(name, forward, **kwargs)

    def select(self, name, payload, indices):
        result = self.graph.select(name, payload, indices)
        self.graph.end_capture(captured=False)
        return result


def _uncompressed_draft_forward(proposer, context, forward, multi_metadata, *, dummy=False):
    transfer = proposer.runner.vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer or context.attn_metadata is None and not multi_metadata:
        return forward()  # Profiling without any attention/cache access.
    runner = proposer.runner
    state = _state(runner)
    execution = state.execution
    try:
        if (proposer.method != 'mtp' or proposer.parallel_drafting
                or context.cudagraph_runtime_mode != CUDAGraphMode.NONE or context.capturing
                or not 1 <= proposer.num_speculative_tokens < UNCOMPRESSED_PROFILE.max_queries
                or len(multi_metadata) != proposer.num_speculative_tokens):
            raise ValueError('Owned GLM requires one to three serial eager MTP passes')
        if execution.draft is None:
            raise RuntimeError('Owned GLM has no physical draft layer')
        name = execution.draft.layers()[0].name
        metadata = tuple(entry.get(name) if isinstance(entry, dict) else None for entry in multi_metadata)
        if dummy or not runner.input_batch.num_reqs:
            scope, attribute = _EmptyDraft(state, metadata), 'avsg_decode'
        else:
            step = execution.active
            if step is None or tuple(runner.input_batch.req_ids[:runner.input_batch.num_reqs]) != tuple(
                    rid for rid, _, _ in step.scheduled):
                raise RuntimeError('Owned GLM request slots changed between target and draft')
            scope, attribute = step.draft_scope(proposer, metadata), 'avsg_mtp'
        with _scope(metadata, attribute, scope):
            result = forward()
            if isinstance(scope, _EmptyDraft):
                if scope.calls != len(metadata) or scope.graph.capture_key is not None:
                    raise RuntimeError('Owned dummy MTP did not finish every pass')
                torch.npu.synchronize()
            else:
                scope.finish(_dependency())
            return result
    except BaseException as error:
        _fail(state, error)
        raise


def finish_step(runner, finalize):
    transfer = runner.vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer:
        return finalize()
    execution = _state(runner).execution
    if execution.ticket is not None and (execution.active is None or execution.active.phase != 'submitted'):
        raise RuntimeError('Owned GLM completion requires successful target and draft execution')
    return finalize()


"""Native DSv4 target/DSpark scopes; only C4 main is redirected to Host."""

from avsg.runtime.dsa import resolve_profile, C4_PROFILE

from avsg.runtime.pages import _integer


def require_native_hooks():
    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.spec_decode import dspark_proposer, llm_base_proposer
    for module, name in ((dsa_v1, 'dsa'), (llm_base_proposer, 'proposer'), (dspark_proposer, 'dspark')):
        if not source_matches(module.__file__, name):
            raise RuntimeError('Owned DSA requires matching target, DSpark and dummy hooks')



def _draft(runner):
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    spec = runner.speculative_config
    if spec is None:
        return 0
    if (spec.method != C4_PROFILE.proposer or not _integer(spec.num_speculative_tokens, 1)
            or spec.num_speculative_tokens >= C4_PROFILE.max_queries or not isinstance(runner.drafter, AscendDSparkProposer)
            or runner.drafter.use_cuda_graph):
        raise ValueError('Owned DSv4 requires the native eager DSpark proposer with at most seven drafts')
    return spec.num_speculative_tokens


def _c4_replay_attention_state(runner, num_reqs, num_scheduled_tokens):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    _state(runner)
    config, drafts = runner.vllm_config, _draft(runner)
    transfer = config.kv_transfer_config
    if (resolve_profile(config).compression != 4
            or transfer is None or not transfer.is_kv_consumer or transfer.kv_connector != 'OwnedConnector'
            or not _integer(num_reqs, 1) or num_reqs != runner.input_batch.num_reqs
            or num_scheduled_tokens.shape != (num_reqs,) or num_scheduled_tokens.dtype.kind not in 'iu'
            or any(not 1 <= n <= drafts + 1 for n in num_scheduled_tokens.tolist())):
        raise ValueError('Owned DSv4 replay requires exact native query1..8')
    return AscendAttentionState.SpecDecoding if drafts else AscendAttentionState.DecodeOnly


def _c4_model_forward(runner, context, forward, padded_tokens=None):
    transfer = runner.vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_consumer or context.attn_metadata is None:
        return forward()
    state = _state(runner)
    execution, graph = state.execution, state.graph
    _draft(runner)
    if not isinstance(context.attn_metadata, dict):
        raise ValueError('Owned DSv4 requires native per-layer attention metadata')
    items, widths = [], {}
    for layer in execution.views.layers():
        item = context.attn_metadata.get(layer.name)
        decode = getattr(item, 'decode', None)
        if decode is None or getattr(item, 'num_prefills', 0):
            raise ValueError('Owned DSv4 requires all C4 queries to use decode metadata')
        width = decode.block_table.shape[1]
        if widths.setdefault(layer.group, width) != width:
            raise ValueError('Owned DSv4 C4 group changed its native table width')
        items.append(item)
    full = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    if context.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        raise RuntimeError('Owned DSv4 requires eager or FULL decode')
    try:
        with _scope(items, 'avsg_decode', graph):
            if not runner.input_batch.num_reqs:
                if execution.ticket is not None:
                    raise RuntimeError('Owned empty DSv4 step cannot replace a model ticket')
                # DSA builders have already copied native slot mappings. Clear
                # the actual graph inputs; clearing BlockTable alone is too late.
                # Physical page zero is the native reserved dummy page, including
                # compressor/indexer state whose destinations derive from tables.
                cleared = set()
                for item in context.attn_metadata.values():
                    decode = getattr(item, 'decode', None)
                    if decode is None:
                        raise ValueError('Owned DSv4 idle requires native decode metadata')
                    for value, fill in ((decode.slot_mapping, -1), (decode.block_table, 0)):
                        if value is not None and id(value) not in cleared:
                            value.fill_(fill)
                            cleared.add(id(value))
                offsets = items[0].decode.query_start_loc_cpu.tolist()
                counts = tuple(b - a for a, b in zip(offsets, offsets[1:]))
                if not counts or len(set(counts)) != 1 or sum(counts) != padded_tokens:
                    raise ValueError('Owned DSv4 capture requires uniform native query offsets')
                empty = full and counts in graph.captured
                graph.prepare_bucket(counts, widths)
                graph.begin_empty(counts) if empty else graph.begin_capture(counts)
                result = forward()
                torch.npu.synchronize()
                if empty:
                    if context.capturing:
                        raise RuntimeError('Owned idle DSv4 unexpectedly recaptured a model')
                    graph.end_empty()
                else:
                    graph.end_capture(captured=context.capturing)
                return result
            if context.capturing:
                raise RuntimeError('Owned DSv4 cannot capture a live request')
            step = execution.step(_scheduled(runner))
            counts = step.query_counts()
            if full and padded_tokens != sum(counts):
                if (len(set(counts)) != 1 or padded_tokens is None or padded_tokens < sum(counts)
                        or padded_tokens % counts[0]):
                    raise ValueError('Owned DSv4 padding differs from its uniform graph bucket')
                counts = (counts[0],) * (padded_tokens // counts[0])
            if not full:
                graph.prepare_bucket(counts, widths, eager=True)
            graph.begin(step, counts, eager=not full)
            result = forward()
            graph.end(_dependency())
            return result
    except BaseException as error:
        _fail(state, error)
        raise


def _c4_draft_forward(proposer, context, forward, multi_metadata):
    runner = proposer.runner
    state = _state(runner)
    try:
        _draft(runner)
        step = state.execution.active
        if (context.cudagraph_runtime_mode != CUDAGraphMode.NONE or context.capturing
                or step is None or step.phase != 'draft'
                or tuple(runner.input_batch.req_ids[:runner.input_batch.num_reqs]) != tuple(
                    rid for rid, _, _ in step.scheduled)):
            raise RuntimeError('Owned DSpark requires its retained target batch and eager native execution')
        # The native runnable includes context-KV precompute, all SWA draft
        # writes and sampling. Preserve its physical mappings/causal semantics.
        result = forward()
        step.finish_draft(_dependency())
        return result
    except BaseException as error:
        _fail(state, error)
        raise


def dummy_forward(proposer, forward):
    runner = proposer.runner
    state = _state(runner)
    if state.execution.ticket is not None:
        raise RuntimeError('Owned DSpark dummy cannot replace a live model ticket')
    # Native dummy execution can retain buffers from the previous live pass.
    # Clear every context destination before its precompute scatter executes.
    for value in proposer._per_group_context_slot_mapping_buffers.values():
        value.fill_(-1)
    for value in proposer._per_group_query_slot_mapping_buffers.values():
        value.fill_(-1)
    result = forward()
    torch.npu.synchronize()
    return result


def _adapter(runner):
    state = getattr(runner, '_avsg_owned_worker', None)
    profile = getattr(state, 'profile', None)
    if profile is None:
        profile = resolve_profile(runner.vllm_config)
    if profile.compression == 4:
        return _C4_RUNNER
    if profile.compression == 1:
        return _UNCOMPRESSED_RUNNER
    raise ValueError('Unsupported owned execution compression')


_C4_RUNNER = SimpleNamespace(replay_attention_state=_c4_replay_attention_state,
                            model_forward=_c4_model_forward, draft_forward=_c4_draft_forward)
_UNCOMPRESSED_RUNNER = SimpleNamespace(replay_attention_state=_uncompressed_replay_attention_state,
                                      model_forward=_uncompressed_model_forward,
                                      draft_forward=_uncompressed_draft_forward)


def replay_attention_state(runner, *args):
    return _adapter(runner).replay_attention_state(runner, *args)


def model_forward(runner, *args):
    return _adapter(runner).model_forward(runner, *args)


def draft_forward(proposer, *args, **kwargs):
    return _adapter(proposer.runner).draft_forward(proposer, *args, **kwargs)
