# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Startup selection remains default-off and rejects retired modes."""
import importlib.util
from pathlib import Path
import pytest

def test_provider_selection_is_default_off_and_rejects_unknown(monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'vllm_ascend/envs.py'
    spec = importlib.util.spec_from_file_location('isolated_envs', path)
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    monkeypatch.delenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', raising=False)
    assert envs.VLLM_ASCEND_AVSG_HOST_PROVIDER == 'legacy'
    monkeypatch.setenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', 'avsg-owned')
    assert envs.VLLM_ASCEND_AVSG_HOST_PROVIDER == 'avsg-owned'
    for value in ('typo', 'omnicache', 'omnicache-glm', 'avsg-owned-glm'):
        monkeypatch.setenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', value)
        with pytest.raises(ValueError, match='legacy or avsg-owned'):
            envs.VLLM_ASCEND_AVSG_HOST_PROVIDER
