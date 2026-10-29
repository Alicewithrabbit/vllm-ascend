# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the paired OmniCache-owned Ascend adapter."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("omni_cache.v2.avsg.ascend_adapter.tables")
