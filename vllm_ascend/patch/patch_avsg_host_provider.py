# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Select the external consumer without importing worker-only patches."""
from vllm_ascend import envs


def install():
    provider = envs.VLLM_ASCEND_AVSG_HOST_PROVIDER
    if provider in ("legacy", "avsg-owned", "avsg-owned-glm", "avsg-owned-dsv4"):
        # Owned execution uses explicit native Scheduler/connector selection and
        # Runner hooks; it must never install the Core Host lifecycle.
        return
    if provider in ("omnicache-glm", "omnicache-glm-direct"):
        from avsg.providers.omnicache_glm import install as install_glm

        install_glm()
        return
    if provider not in ("omnicache", "omnicache-decode", "omnicache-direct"):
        raise ValueError("AVSG Host provider must be legacy, omnicache, omnicache-decode, omnicache-direct, omnicache-glm or omnicache-glm-direct")
    from avsg.providers.omnicache import install as install_provider

    if provider == "omnicache-direct":
        install_provider(decode=True, direct=True)
    elif provider == "omnicache-decode":
        install_provider(decode=True)
    else:
        install_provider()
