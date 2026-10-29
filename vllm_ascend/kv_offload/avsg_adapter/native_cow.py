# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compatibility import for the paired OmniCache-owned Ascend adapter."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("omni_cache.v2.avsg.ascend_adapter.native_cow")
