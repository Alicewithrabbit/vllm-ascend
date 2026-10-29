# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native D connector boundary; bootstrap must supply the existing owned pages.

No Core Host owner or prefix catalog is constructed. Model execution is attached
separately: selecting this transport boundary alone cannot run attention/STORE.
"""
from dataclasses import dataclass, replace
from copy import copy
import hashlib
import inspect
from pathlib import Path
from uuid import uuid4

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorWorkerMetadata,
    KVConnectorRole, SupportsHMA)
from vllm.v1.core.sched import scheduler as native
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus
from avsg.runtime.pages import _integer
from avsg.runtime.receive import ReceiveResult
from avsg.runtime.resident_rows import ResidentDelta
from vllm_ascend.kv_offload.avsg_adapter.native_model import ModelStep, ModelReport
from avsg.runtime.page_mirror import HostDelta, HostPageMirror
from vllm_ascend.kv_offload.avsg_adapter.native_cow import CowBatch, CowResult, NativeCOW


@dataclass(frozen=True)
class OwnedMetadata(KVConnectorMetadata):
    epoch: str
    sequence: int
    # No backend is a strict empty transport contract, never fake PD success.
    delta: None
    batches: tuple[()]
    cancellations: tuple[str, ...]
    model: ModelStep | None = None
    host_delta: HostDelta | None = None
    cows: tuple[CowBatch, ...] = ()

    def __post_init__(self):
        if type(self) is OwnedMetadata and (self.delta is not None or self.batches != ()):
            raise ValueError('Transport payload requires an attached backend metadata schema')


@dataclass(frozen=True)
class ResidentMetadata(OwnedMetadata):
    """Identity rows without transport work; never imply remote KV completion."""
    delta: ResidentDelta | None

    def __post_init__(self):
        if (self.batches != ()
                or self.delta is not None and type(self.delta) is not ResidentDelta):
            raise ValueError('Resident metadata requires identity deltas and no remote batches')


@dataclass(frozen=True)
class OwnedWorkerMetadata(KVConnectorWorkerMetadata):
    results: tuple[ReceiveResult, ...]
    model_reports: tuple[ModelReport, ...] = ()
    cow_results: tuple[CowResult, ...] = ()

    def aggregate(self, other):
        if not isinstance(other, OwnedWorkerMetadata):
            raise ValueError('Cannot aggregate a foreign owned receive result')
        return OwnedWorkerMetadata(self.results + other.results,
                                   self.model_reports + other.model_reports,
                                   self.cow_results + other.cow_results)


def _backend_metadata_type(backend, pages=None):
    factory = getattr(backend, 'connector_metadata_type_for', None)
    if pages is not None and factory is not None:
        if not callable(factory):
            raise ValueError('Backend metadata selector must be callable')
        schema = factory(pages)
    else:
        schema = getattr(backend, 'connector_metadata_type', OwnedMetadata)
    if not isinstance(schema, type) or not issubclass(schema, OwnedMetadata):
        raise ValueError('Backend must provide a concrete owned metadata schema')
    return schema


def collect_model_reports(connector, output):
    """Join post-MTP reports into this forward's still-local native output.

    Native collection can precede deferred draft finalization. Never send the
    report in a later step or discard it because the early PD result was empty.
    """
    early = output.kv_connector_worker_meta
    if early is not None and not isinstance(early, OwnedWorkerMetadata):
        raise TypeError('Owned execution cannot merge foreign connector metadata')
    late = connector.build_connector_worker_meta()
    combined = late if early is None else early if late is None else early.aggregate(late)
    if combined is None:
        return
    seen = set()
    for report in combined.model_reports:
        key = report.epoch, report.sequence, report.rank
        if key[:2] != (connector._worker_epoch, connector._worker_sequence):
            raise ValueError('Post-draft report differs from the current worker step')
        if key in seen:
            raise ValueError('Post-draft output contains a duplicate model report')
        seen.add(key)
    output.kv_connector_worker_meta = combined


class OwnedConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.pd = self.scheduler = self.worker = self.pages = None
        self._metadata_type = OwnedMetadata
        self._epoch, self._sequence = uuid4().hex, 0
        self._worker_epoch, self._worker_sequence = None, 0
        self._batches, self._cancellations = [], []
        self._failures = None
        self._finish_writes = None
        self._model_progress = self._complete_model = None
        self._begin_model = self._take_model_reports = None
        self._host_pages, self._mapping_tables = None, ()
        self._mapped_meta, self._mapping_fault = None, None
        self._bootstrap = self._bootstrap_fault = self._owned_handshake = None
        self._owned_worker = None
        self._cow = self._cow_worker = None
        self._cow_schedule = None
        self._cow_results, self._cow_completions = [], []

    def register_kv_caches(self, caches):
        if self._owned_worker is None:
            raise RuntimeError('Owned worker must prepare storage before native cache allocation')
        self._owned_worker.register(caches)

    def prepare_avsg_worker(self, runner, kv_config):
        raise NotImplementedError("AVSG requires an explicit storage backend")

    def set_xfer_handshake_metadata(self, metadata):
        raise NotImplementedError("AVSG requires a backend handshake")

    def get_handshake_metadata(self):
        if self._owned_handshake is None:
            raise RuntimeError('Owned worker requires storage/execution bootstrap before handshake')
        return self._owned_handshake

    def attach_scheduler(self, scheduler, pd):
        from vllm_ascend.core import recompute_scheduler as recompute

        expected = 'e25d4c9a95abdbe8e516714ed02574d929ca0d5e8c11c4cc73b84d3a3b905443'
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unsupported native scheduler completion order')
        update_source = native
        update_owner = AsyncScheduler if scheduler.scheduler_config.async_scheduling else native.Scheduler
        if isinstance(scheduler, recompute.RecomputeScheduler):
            # This completion emits a terminal retry signal on capacity
            # exhaustion; accepting its source does not provide P retry.
            expected = '5c6fcd50a1bca555f176dc8bb8fe980b96080ef20f26ec56d202c9344dacb128'
            if hashlib.sha256(Path(recompute.__file__).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Unsupported native recompute completion order')
            update_source, update_owner = recompute, recompute.RecomputeScheduler
        for name in ('update_from_output', '_handle_invalid_blocks'):
            method = getattr(scheduler, name).__func__
            target_source = update_source if name == 'update_from_output' else native
            while Path(method.__code__.co_filename).resolve() != Path(target_source.__file__).resolve():
                # Retain the image's SWA and PP/MTP backports, including their
                # real delegation targets; do not blindly unwrap other patches.
                source = Path(method.__code__.co_filename)
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                wrapped = getattr(method, '__wrapped__', None)
                if digest == '614bc2f87c2ccde2ee120812951757edc595b57fede1f59083bd208b80f75563':
                    target = method.__globals__.get('_orig_update_from_output')
                elif digest == '8d2ddaa5e88558fc2d8f39172592c41b8f0959526579a75ca13224733167aea2':
                    target = inspect.getclosurevars(method).nonlocals.get('original_update_from_output')
                else:
                    raise RuntimeError('Unsupported native completion method source')
                if wrapped is None or target is not wrapped:
                    raise RuntimeError('Unsupported native completion method source')
                method = wrapped
            if (Path(method.__code__.co_filename).resolve() != Path(target_source.__file__).resolve()
                    or method.__name__ != name):
                raise RuntimeError('Unsupported native completion method source')
        if (self.role != KVConnectorRole.SCHEDULER or self.scheduler is not None
                or scheduler.connector is not self or scheduler.kv_cache_manager is not pd.manager
                or scheduler.requests or scheduler.recompute_kv_load_failures
                or scheduler.update_from_output.__func__ is not update_owner.update_from_output
                or scheduler._handle_invalid_blocks.__func__ is not native.Scheduler._handle_invalid_blocks):
            raise ValueError('Attach once to the actual empty native scheduler with failure policy fail')
        self._metadata_type = _backend_metadata_type(pd)
        self.scheduler, self.pd, self.pages = scheduler, pd, pd.physical
        self._native_update = scheduler.update_from_output
        scheduler.update_from_output = self._update_from_output
        scheduler._handle_invalid_blocks = self._handle_invalid_blocks

    def attach_model_progress(self, progress, complete):
        self._scheduler_ready()
        if (self._model_progress is not None or self._sequence or self.scheduler.requests
                or progress.scheduler is not self.scheduler or not callable(complete)):
            raise ValueError('Attach model acceptance before scheduling on the same native owner')
        self._model_progress, self._complete_model = progress, complete
        # Native request references protect unchanged history even for sync
        # cancellation. The model report fences target, MTP and all consumers.
        self.scheduler.defer_block_free = True

    def attach_worker(self, worker, pages, *, finish_writes=None,
                      begin_model=None, take_model_reports=None):
        if self.role != KVConnectorRole.WORKER or self.worker is not None:
            raise ValueError('Attach once to the rank-local owned worker')
        if ((begin_model is None) != (take_model_reports is None)
                or begin_model is not None and not all(map(callable,
                    (begin_model, take_model_reports, finish_writes)))):
            raise ValueError('Owned model execution requires begin, drain and report callbacks')
        self._metadata_type = _backend_metadata_type(worker, pages)
        self.worker, self.pages, self._finish_writes = worker, pages, finish_writes
        self._begin_model, self._take_model_reports = begin_model, take_model_reports

    def _scheduler_ready(self):
        if self.scheduler is None:
            raise RuntimeError('Owned connector requires post-manager bootstrap')

    def attach_cow_scheduler(self, cow):
        self._scheduler_ready()
        if (type(cow) is not NativeCOW or self._cow is not None or self._sequence
                or self.scheduler.requests or cow.manager is not self.scheduler.kv_cache_manager
                or cow.owner is not self.pd.owner or cow.pages is not self.pages):
            raise ValueError('COW must attach to the same empty native scheduler and page owner')
        self._cow = cow

    def take_cow_completions(self):
        completed = tuple(self._cow_completions)
        self._cow_completions.clear()
        return completed

    def attach_cow_worker(self, copier):
        from avsg.runtime.cow_copy import MainKVCOW
        if (self.role != KVConnectorRole.WORKER or self._cow_worker is not None
                or self._worker_sequence or self._mapped_meta is not None
                or not isinstance(copier, MainKVCOW)
                or copier.storage.pages is not self.pages
                or copier.host_pages is not self._host_pages):
            raise ValueError('COW worker requires the same initialized storage and Host mirror')
        self._cow_worker = copier

    def _poll_cow(self):
        if self._cow_worker is not None:
            self._cow_results.extend(self._cow_worker.poll())

    def attach_host_mappings(self, host_pages, tables=()):
        if (self.worker is None or self._host_pages is not None or self._worker_sequence
                or self._mapped_meta is not None or not isinstance(host_pages, HostPageMirror)
                or host_pages.sequence or host_pages.logical_capacity != self.pages.logical_capacity
                or len(host_pages.group_domains) != len(self.pages.group_domains)
                or any(t.pages is not self.pages or t.host_pages is not host_pages for t in tables)):
            raise ValueError('Attach fresh Host mappings to the same bootstrapped worker before dispatch')
        self._host_pages, self._mapping_tables = host_pages, tuple(tables)

    def prepare_mappings(self, metadata):
        """Run before native _update_states; start_load_kv shares this boundary."""
        if self.worker is None or self._mapping_fault is not None:
            raise RuntimeError('Owned mapping bootstrap failed; retain storage until teardown') from self._mapping_fault
        meta = msgspec.convert(metadata, self._metadata_type, strict=True)
        if (not meta.epoch or not _integer(meta.sequence, 1)
                or self._worker_epoch not in (None, meta.epoch)):
            raise ValueError('A new scheduler owner requires worker teardown')
        previous = self._mapped_meta
        if previous is not None and meta.sequence == previous.sequence:
            if meta != previous:
                raise ValueError('Previously mapped step changed before consumption')
            return meta
        if (meta.sequence != self._worker_sequence + 1
                or previous is not None and previous.sequence != self._worker_sequence):
            raise ValueError('Owned mapping metadata must arrive in ordered steps')
        if self._host_pages is not None:
            host = () if meta.host_delta is None else meta.host_delta.changes
            physical = () if meta.delta is None else meta.delta.changes
            if tuple((a, g, r.block_id) for a, g, r in host) != tuple(
                    (a, b.group, b.logical_id) for a, b in physical):
                raise ValueError('Host and companion mapping changes must cover the same logical pages')
            if meta.host_delta is not None:
                self._host_pages.validate(meta.host_delta)
        try:
            if meta.delta is not None:
                self.pages.apply(meta.delta)
            if self._host_pages is not None and meta.host_delta is not None:
                self._host_pages.apply(meta.host_delta)
            for table in self._mapping_tables:
                for action, binding in (() if meta.delta is None else meta.delta.changes):
                    if action == 'free':
                        table.retire(binding)
                for action, group, ref in (() if meta.host_delta is None else meta.host_delta.changes):
                    if action == 'free' and group == table.group:
                        table.retire_host(ref)
        except BaseException as error:
            self._mapping_fault = error
            raise
        self._mapped_meta = meta
        return meta

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._scheduler_ready()
        if self._cow is not None and self._cow.has_pending(request):
            return None, False
        if (request.num_preemptions and self._model_progress is not None
                and self._model_progress.has_pending(request)):
            # Older native outputs may still append accepted tokens and release
            # deferred pages. Never bind a resumed generation ahead of them.
            return None, False
        if self.pd.has_pending(request.request_id):
            return None, False
        if not (request.kv_transfer_params or {}).get('do_remote_prefill'):
            return 0, False
        count = max(0, request.num_prompt_tokens - num_computed_tokens)
        return count, count > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._scheduler_ready()
        if (request.kv_transfer_params or {}).get('do_remote_prefill'):
            self._batches.append(self.pd.prepare(request, step_id=self._sequence + 1))

    def build_connector_meta(self, scheduler_output):
        self._scheduler_ready()
        if self._cow_schedule is not None:
            self._cow_schedule.prepare(scheduler_output)
        self._sequence += 1
        model = (self._model_progress.prepare(self._epoch, self._sequence, scheduler_output)
                 if self._model_progress is not None else None)
        meta = self._metadata_type(self._epoch, self._sequence, self.pages.drain(),
                             tuple(self._batches), tuple(self._cancellations), model,
                             self.pd.owner.drain_mappings(),
                             () if self._cow is None else self._cow.take_batches())
        self._batches.clear()
        self._cancellations.clear()
        return meta

    def request_finished_all_groups(self, request, block_ids):
        self._scheduler_ready()
        if self._cow is not None:
            self._cow.cancel(request)
        identity = self.pd.cancel(request.request_id)
        if identity is not None:
            self._cancellations.append(identity)
        # Native references may be released; owned async pins retain the bytes.
        return False, None

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def has_pending_push_work(self):
        return ((self.pd is not None and self.pd.has_pending())
                or (self._model_progress is not None and self._model_progress.pending)
                or (self._cow is not None and self._cow.has_pending())
                or (self._cow_schedule is not None and self._cow_schedule.pending)
                or bool(self._cow_completions))

    def _update_from_output(self, step, output):
        progress = self._model_progress
        state = None
        if progress is not None:
            connector_output = output.kv_connector_output
            raw = None if connector_output is None else connector_output.kv_connector_worker_meta
            meta = None if raw is None else msgspec.convert(raw, OwnedWorkerMetadata, strict=True)
            ticket = step.kv_connector_metadata.model
            state = progress.before_native(ticket, () if meta is None else meta.model_reports, output)
        try:
            result = self._update_owned_output(step, output)
            if progress is not None:
                progress.after_native(state, self._complete_model)
            return result
        except BaseException as error:
            if progress is not None:
                progress.fail(error)
            raise

    def _update_owned_output(self, step, output):
        connector_output = output.kv_connector_output
        if connector_output is None:
            return self._native_update(step, output)
        if (connector_output.finished_sending or connector_output.finished_recving
                or connector_output.invalid_block_ids):
            raise ValueError('Owned PD rejects unqualified native completion/block IDs')
        meta = connector_output.kv_connector_worker_meta
        if meta is None:
            return self._native_update(step, output)
        meta = msgspec.convert(meta, OwnedWorkerMetadata, strict=True)
        if meta.model_reports and self._model_progress is None:
            raise ValueError('Model reports require an attached native acceptance owner')
        if meta.cow_results and self._cow is None:
            raise ValueError('COW reports require an attached native transaction owner')
        for report in meta.cow_results:
            done = self._cow.complete(report.batch, report.rank,
                main_success=report.main_success, companions_success=report.companions_success)
            if done is not None:
                self._cow_completions.append(done)
        finished, invalid, failed = set(), set(), set()
        for report in meta.results:
            request = self.pd.request_for(report.identity)
            done = self.pd.complete(report)
            if done is None or self.scheduler.requests.get(done[0]) is not request:
                continue
            if request.is_finished():
                finished.add(request.request_id)  # Native retained cancelled receive.
                continue
            if request.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
                continue  # A ready full hit needing only P ACK.
            if done[1]:
                finished.add(request.request_id)
            else:
                if not done[2]:
                    raise RuntimeError('Failed receive has no qualified destination')
                failed.add(request.request_id)
                finished.add(request.request_id)  # Native error output precedes final free.
                invalid.update(done[2])
        normalized = replace(connector_output, finished_recving=finished,
            invalid_block_ids=invalid, kv_connector_worker_meta=None)
        normalized_output = copy(output)  # Preserve Ascend's non-dataclass MTP fields.
        normalized_output.kv_connector_output = normalized
        self._failures = (invalid, failed)
        try:
            return self._native_update(step, normalized_output)
        finally:
            self._failures = None

    def _handle_invalid_blocks(self, block_ids, num_scheduled_tokens):
        # These are only terminal owned PD failures, associated with original
        # Request objects above. Native's single-group recompute path is unsafe
        # for hybrid D and is not needed under the explicit fail policy.
        if self._failures is None or block_ids != self._failures[0]:
            raise ValueError('Invalid blocks lack an owned receive completion')
        return self._failures[1]

    def bind_connector_metadata(self, metadata):
        super().bind_connector_metadata(msgspec.convert(metadata, self._metadata_type, strict=True))

    def start_load_kv(self, forward_context, **kwargs):
        if self.worker is None:
            raise RuntimeError('Owned worker requires Host/physical/transport bootstrap')
        incoming = self._get_connector_metadata()
        if incoming.cows and self._cow_worker is None:
            raise RuntimeError('COW execution is not attached')
        blocked = {b.request_id for b in incoming.cows}
        if self._cow_worker is not None:
            blocked.update(self._cow_worker.request_ids)
        if incoming.model is not None and any(r.request_id in blocked for r in incoming.model.requests):
            raise ValueError('Model execution must wait for COW and the updated request row')
        meta = self.prepare_mappings(self._get_connector_metadata())
        if meta.sequence <= self._worker_sequence:
            return  # Previously applied ordered step; never repeat its writes.
        if meta.model is not None and self._begin_model is None:
            raise RuntimeError('Owned model execution is not attached')
        for batch in meta.batches:
            self.worker.submit(batch)
        for identity in meta.cancellations:
            self.worker.cancel(identity)
        for batch in meta.cows:
            self._cow_worker.submit(batch)
        if meta.model is not None:
            self._begin_model(meta.model)
        self._worker_epoch, self._worker_sequence = meta.epoch, meta.sequence

    def get_finished(self, finished_req_ids):
        if self.worker is None:
            raise RuntimeError('Owned worker is not bootstrapped')
        self.worker.poll()
        self._poll_cow()
        return None, None  # Only scheduler-side generation checks may emit IDs.

    def build_connector_worker_meta(self):
        self._poll_cow()
        results = self.worker.take_results()
        reports = () if self._take_model_reports is None else tuple(self._take_model_reports())
        cows = tuple(self._cow_results)
        self._cow_results.clear()
        return OwnedWorkerMetadata(results, reports, cows) if results or reports or cows else None

    def wait_for_layer_load(self, layer_name):
        pass  # PD completed before native scheduler admits model execution.

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass  # Owned execution writes directly; no generic per-layer STORE.

    def wait_for_save(self):
        if self._finish_writes is None:
            raise RuntimeError('Owned model write completion is not attached')
        self._finish_writes()

    def shutdown(self):
        if self._owned_worker is not None:
            self._owned_worker.quiesce()
