# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Test startup selection without importing the device runtime."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2] / 'vllm_ascend'


def load(name, path):
    assert path.is_file(), 'AVSG startup hook is missing'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_selection_is_default_off_and_rejects_unknown(monkeypatch):
    envs = load('isolated_envs', ROOT / 'envs.py')
    monkeypatch.delenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', raising=False)
    assert envs.VLLM_ASCEND_AVSG_HOST_PROVIDER == 'legacy'
    package = ModuleType('vllm_ascend')
    package.envs = envs
    monkeypatch.setitem(sys.modules, 'vllm_ascend', package)
    monkeypatch.setitem(sys.modules, 'avsg', None)
    hook = load('isolated_hook', ROOT / 'patch/patch_avsg_host_provider.py')
    hook.install()
    monkeypatch.setenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', 'avsg-owned-glm')
    hook.install()  # Explicit native entry points; no Core/worker imports.
    monkeypatch.setenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', 'typo')
    with pytest.raises(ValueError, match='legacy.*omnicache'):
        hook.install()


@pytest.mark.parametrize('mode', ['omnicache-glm', 'omnicache-glm-direct'])
def test_glm_mode_registers_scheduler_without_importing_worker(monkeypatch, mode):
    package = ModuleType('vllm_ascend')
    package.envs = SimpleNamespace(VLLM_ASCEND_AVSG_HOST_PROVIDER=mode)
    monkeypatch.setitem(sys.modules, 'vllm_ascend', package)
    monkeypatch.setitem(sys.modules, 'vllm_ascend.worker', None)
    calls = []
    bridge = ModuleType('avsg.providers.omnicache_glm')
    bridge.install = lambda: calls.append('scheduler')
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    load('isolated_glm_hook', ROOT / 'patch/patch_avsg_host_provider.py').install()
    assert calls == ['scheduler']


def test_explicit_mode_delegates_and_propagates_incompatible_backend(monkeypatch):
    package = ModuleType('vllm_ascend')
    package.envs = SimpleNamespace(VLLM_ASCEND_AVSG_HOST_PROVIDER='omnicache')
    monkeypatch.setitem(sys.modules, 'vllm_ascend', package)
    calls = []
    bridge = ModuleType('avsg.providers.omnicache')
    bridge.install = lambda: calls.append('installed')
    monkeypatch.setitem(sys.modules, 'avsg.providers.omnicache', bridge)
    hook = load('isolated_hook', ROOT / 'patch/patch_avsg_host_provider.py')
    hook.install()
    assert calls == ['installed']
    def fail():
        raise RuntimeError('unsupported source')
    bridge.install = fail
    with pytest.raises(RuntimeError, match='unsupported source'):
        hook.install()


@pytest.mark.parametrize('mode', ['legacy', 'omnicache', 'omnicache-decode', 'omnicache-direct',
                                  'avsg-owned-glm'])
def test_real_connector_entry_does_not_import_worker_patches(monkeypatch, mode):
    monkeypatch.setenv('VLLM_ASCEND_AVSG_HOST_PROVIDER', mode)
    monkeypatch.setitem(sys.modules, 'vllm_ascend.logger', ModuleType('vllm_ascend.logger'))
    # Pre-existing connector registration is outside this test. The package
    # and new hook imports use Python's real package import machinery.
    for name, function in [('kv_transfer', 'register_connector'), ('weight_transfer', 'register_engine')]:
        module = ModuleType('vllm_ascend.distributed.'+name)
        setattr(module, function, lambda: None)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, 'vllm_ascend.patch.worker', None)
    calls = []
    bridge = ModuleType('avsg.providers.omnicache')
    bridge.install = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    for name in ['vllm_ascend.envs', 'vllm_ascend.patch', 'vllm_ascend.patch.patch_avsg_host_provider']:
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location('vllm_ascend', ROOT/'__init__.py')
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'vllm_ascend', package)
    spec.loader.exec_module(package)
    package._ensure_global_patch = lambda: None
    package.register_connector()
    expected = {'legacy': [], 'avsg-owned-glm': [], 'omnicache': [{}], 'omnicache-decode': [{'decode': True}],
                'omnicache-direct': [{'decode': True, 'direct': True}]}
    assert calls == expected[mode]
