# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qualify model completion with native, post-sampling token accounting.

This is not a page allocator or byte-access ticket. The execution owner must
retain its generation-qualified pages until the completion callback succeeds.
Worker reports cover target, draft and all dependent readers/writes, after real
device completion; enqueue notifications and timeouts cannot produce reports.
"""
from dataclasses import dataclass, replace
import hashlib
import inspect
from pathlib import Path

from vllm.v1 import request as native_request
from vllm.v1.core.sched import async_scheduler, scheduler as native_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from avsg.runtime.pages import PageUse, _integer


from avsg.runtime.model_records import (
    ScheduledRange, ModelPage, ModelStep, ExecutedRange, ModelReport,
    AcceptedRange, ModelCompletion)


class NativeModelProgress:
    def __init__(self, scheduler, *, ranks, writes=None):
        ranks = tuple(ranks)
        if (not ranks or len(set(ranks)) != len(ranks)
                or any(not _integer(rank) for rank in ranks)
                or scheduler.num_sampled_tokens_per_step != 1
                or scheduler.parallel_config.pipeline_parallel_size != 1
                or scheduler.use_v2_model_runner):
            raise ValueError('Model accounting requires AR decode with the audited V1/PP1 order')
        for module, digest in (
                (native_request, '06b1c6ad3cc0223c2036b4a2e317868fa4988391b4b024bc5e5f54672b7d2c80'),
                (native_scheduler, 'e25d4c9a95abdbe8e516714ed02574d929ca0d5e8c11c4cc73b84d3a3b905443'),
                (async_scheduler, 'da6343d7e7c394a1738cf72905cbecc208003ffa461ccb441268333a3eb9f884')):
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Unsupported native accepted-token accounting')
        expected = (async_scheduler.AsyncScheduler if scheduler.scheduler_config.async_scheduling else Scheduler)
        def check_source(method, name, source):
            # Retain the image's SWA -> PP/MTP -> native delegation, also when
            # AsyncScheduler calls it through super(). PP is dormant at PP1.
            for _ in range(3):
                path = Path(method.__code__.co_filename)
                if path.resolve() == Path(source.__file__).resolve():
                    break
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if name == '_update_after_schedule' and digest == '614bc2f87c2ccde2ee120812951757edc595b57fede1f59083bd208b80f75563':
                    original = method.__globals__.get('_orig_update_after_schedule')
                elif name == '_update_after_schedule' and digest == '8d2ddaa5e88558fc2d8f39172592c41b8f0959526579a75ca13224733167aea2':
                    original = inspect.getclosurevars(method).nonlocals.get('original_update_after_schedule')
                else:
                    raise RuntimeError('Unsupported native accounting/deferred-free source')
                if original is None or getattr(method, '__wrapped__', None) is not original:
                    raise RuntimeError('Unsupported native inflight accounting delegation')
                method = original
            if method.__name__ != name or Path(method.__code__.co_filename).resolve() != Path(source.__file__).resolve():
                raise RuntimeError('Unsupported native accounting/deferred-free source')
        for name in ('_update_request_with_output', '_update_after_schedule',
                     '_free_request_blocks', '_drain_deferred_frees'):
            method = getattr(scheduler, name).__func__
            source = (async_scheduler if expected is async_scheduler.AsyncScheduler
                      and name in ('_update_request_with_output', '_update_after_schedule') else native_scheduler)
            if method is not getattr(expected, name):
                raise RuntimeError('Unsupported native accounting/deferred-free source')
            check_source(method, name, source)
        if expected is async_scheduler.AsyncScheduler:
            for name in ('_update_request_with_output', '_update_after_schedule'):
                check_source(getattr(expected.__mro__[1], name), name, native_scheduler)
        self.scheduler, self.ranks = scheduler, frozenset(ranks)
        self.writes = writes
        self._pending = {}
        self._fault = None

    @property
    def pending(self):
        return bool(self._pending)

    def has_pending(self, request):
        self._guard()
        return any(previous is request for _, records, _ in self._pending.values()
                   for previous, _ in records)

    def _guard(self):
        if self._fault is not None:
            raise RuntimeError('Model completion faulted; retain pages until teardown') from self._fault

    def describe(self, output):
        self._guard()
        windows, records = [], []
        for rid, count in output.num_scheduled_tokens.items():
            request = self.scheduler.requests[rid]
            if not _integer(count, 1) or count > self.scheduler.num_spec_tokens + 1:
                raise ValueError('Independent D model step requires bounded decode queries')
            # Earlier in-flight rejections may correct the worker's CPU start.
            # No scan of historical blocks is needed to bound that correction.
            correction = sum(drafts for _, old, _ in self._pending.values()
                             for previous, drafts in old if previous is request)
            start = request.num_computed_tokens
            windows.append(ScheduledRange(rid, start, count, max(0, start - correction),
                                          request.num_preemptions))
            records.append((request, len(output.scheduled_spec_decode_tokens.get(rid, ()))))
        return tuple(windows), tuple(records)

    def prepare(self, epoch, sequence, output):
        self._guard()
        if not output.num_scheduled_tokens:
            return None
        if not epoch or not _integer(sequence, 1) or sequence in self._pending:
            raise ValueError('Model step must have a unique scheduler identity')
        windows, records = self.describe(output)
        ticket = ModelStep(epoch, sequence, tuple(windows))
        if self.writes is not None:
            try:
                ticket = replace(ticket, writes=self.writes.prepare(ticket, records))
            except BaseException as error:
                self._fault = error
                raise
        self._pending[sequence] = ticket, tuple(records), {w.request_id: w.start for w in windows}
        return ticket

    def before_native(self, ticket, reports, output):
        self._guard()
        if ticket is None:
            if reports:
                raise ValueError('Model report has no scheduled model step')
            return None
        record = self._pending.get(ticket.sequence)
        if (record is None or record[0] != ticket or ticket.sequence != min(self._pending)
                or len(reports) != len(self.ranks) or {r.rank for r in reports} != self.ranks):
            raise ValueError('Model completion requires the ordered step and every rank exactly once')
        expected = {w.request_id: w for w in ticket.requests}
        ranges = None
        for report in reports:
            if (not _integer(report.rank) or report.epoch != ticket.epoch
                    or report.sequence != ticket.sequence):
                raise ValueError('Model completion belongs to another scheduler step')
            actual = {w.request_id: w for w in report.requests}
            if len(actual) != len(report.requests) or set(actual) != set(expected):
                raise ValueError('Model completion omitted or repeated a request')
            for rid, window in actual.items():
                scheduled = expected[rid]
                if (not _integer(window.start) or not _integer(window.count, 1)
                        or window.count != scheduled.count
                        or not scheduled.min_start <= window.start <= scheduled.start
                        or window.start != record[2][rid]):
                    raise ValueError('Executed range differs from the scheduled decode window')
            if ranges is not None and ranges != actual:
                raise ValueError('Model ranks disagree about their executed token ranges')
            ranges = actual
        eligible, rejections = [], []
        for window, (request, drafts) in zip(ticket.requests, record[1]):
            current = self.scheduler.requests.get(window.request_id)
            if current is not None and current is not request:
                raise RuntimeError('Model result cannot enter native accounting for a reused request ID')
            eligible.append(current is request and request.status == RequestStatus.RUNNING
                            and request.num_preemptions == window.preemptions
                            and request.async_tokens_to_discard == 0)
            samples = (output.sampled_token_ids[output.req_id_to_index[window.request_id]]
                       if output.sampled_token_ids else ())
            if len(samples) > drafts + 1:
                raise ValueError('Model samples exceed the scheduled speculative window')
            # Capture rejection sampling before native stop handling truncates
            # the list. EOS/max-token truncation limits publication, not the
            # worker's correction of an already in-flight frame.
            rejections.append(drafts - max(len(samples) - 1, 0) if samples else 0)
        return ticket, record[1], ranges, tuple(eligible), tuple(rejections)

    def after_native(self, state, complete):
        if state is None:
            return
        ticket, records, ranges, eligible, rejections = state
        try:
            results = []
            for window, (request, _), valid, rejected in zip(ticket.requests, records, eligible, rejections):
                executed = ranges[window.request_id]
                valid = (valid and request.num_preemptions == window.preemptions
                         and request.status in (RequestStatus.RUNNING,
                             RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED))
                accepted = None
                if valid:
                    # Native counters can include a later, already scheduled
                    # frame. Confirmed IDs also cap EOS/max-token truncation;
                    # the final sampled token has no KV until the next forward.
                    accepted = min(executed.start + executed.count,
                        request.num_computed_tokens - request.num_output_placeholders,
                        request.num_tokens - 1)
                    if accepted < executed.start:
                        raise RuntimeError('Native accepted prefix precedes the executed decode window')
                # Cancelled/completed requests still drain previously executed
                # frames. Track their worker correction without publishing KV.
                for future, later, starts in self._pending.values():
                    if future.sequence <= ticket.sequence:
                        continue
                    for future_window, (other, _) in zip(future.requests, later):
                        if other is request and future_window.preemptions == window.preemptions:
                            starts[future_window.request_id] -= rejected
                results.append(AcceptedRange(window.request_id, executed.start, accepted, valid))
            completion = ModelCompletion(ticket, tuple(results))
            if self.writes is not None:
                self.writes.complete(completion)
            complete(completion)
            del self._pending[ticket.sequence]
        except BaseException as error:
            self._fault = error
            raise

    def fail(self, error):
        self._fault = error  # Native may have partially consumed an output; never replay it.
