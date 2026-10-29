# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Retain owned model completion through the frozen native async output."""
import hashlib
from pathlib import Path

import vllm
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker import gpu_model_runner

from vllm_ascend.kv_offload.avsg_adapter.native_connector import OwnedWorkerMetadata
from vllm_ascend.kv_offload.avsg_adapter.owned_runner import _state


def async_completion_enabled(config):
    # Experimental, non-sensitive connector option. False preserves synchronous
    # completion; True requires the source-matched native two-step async queue.
    enabled = config.kv_transfer_config.kv_connector_extra_config.get('avsg_owned_async_completion', False)
    if type(enabled) is not bool or enabled and not config.scheduler_config.async_scheduling:
        raise ValueError('Owned async completion requires a boolean option and native async scheduling')
    if enabled:
        root = Path(vllm.__file__).parent
        for name, expected in (
                ('v1/worker/gpu_model_runner.py', '6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e'),
                ('v1/engine/core.py', '27b23827a86fd488f1f7cc9a2722fe76d284087ca7cfefef5a30c5e74f21bc71')):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Unsupported native async output completion order')
    return enabled


def wrap_async_output(runner, output):
    state = _state(runner)
    execution = state.execution
    if not execution.defer_reports or execution._uncollected is None:
        return output
    try:
        if type(output) is not gpu_model_runner.AsyncGPUModelRunnerOutput:
            raise TypeError('Owned completion requires its source-matched native async output')
        frame = execution.take_completion()
        return _OwnedOutput(state, output, frame)
    except BaseException as error:
        if execution.failed is None:
            execution.failed = error
        raise


class _OwnedOutput(AsyncModelRunnerOutput):
    def __init__(self, state, native, frame):
        self.state, self.native, self.frame = state, native, frame
        self.execution = state.execution
        self.output = None

    def get_output(self):
        if self.output is not None:
            return self.output
        execution, frame = self.execution, self.frame
        try:
            if self.state._ready() is not execution:
                raise RuntimeError('Owned output lost its original worker')
            execution.check_completion(frame)
            # Native conversion joins D2H; the separate final target/draft event
            # still fences all KV consumers. Neither report means submission.
            output = self.native.get_output()
            frame.dependency._event.synchronize()
            report = frame.report
            if (output is not self.native._model_runner_output
                    or tuple(output.req_ids) != tuple(r.request_id for r in report.requests)
                    or output.kv_connector_output is None):
                raise ValueError('Owned completion differs from its captured native output')
            meta = output.kv_connector_output.kv_connector_worker_meta
            if meta is not None and not isinstance(meta, OwnedWorkerMetadata):
                raise TypeError('Owned output cannot merge foreign connector metadata')
            reports = () if meta is None else meta.model_reports
            if any((r.epoch, r.sequence) != (report.epoch, report.sequence) or r.rank == report.rank
                   for r in reports):
                raise ValueError('Owned output contains a foreign or duplicate model report')
            output.kv_connector_output.kv_connector_worker_meta = OwnedWorkerMetadata(
                () if meta is None else meta.results, reports + (report,),
                () if meta is None else meta.cow_results)
            execution.release_completion(frame)
            self.output, self.frame = output, None
            return output
        except BaseException as error:
            if execution.failed is None:
                execution.failed = error
            raise
