# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exact paired native hooks; new source requires explicit revalidation."""
import hashlib
from pathlib import Path
from types import MappingProxyType

SOURCE_FILES = MappingProxyType({'runner': 'worker/model_runner_v1.py',
 'block_table': 'worker/block_table.py',
 'dsa': 'attention/dsa_v1.py',
 'dspark': 'spec_decode/dspark_proposer.py',
 'proposer': 'spec_decode/llm_base_proposer.py'})
SOURCE_SHA256 = MappingProxyType({'runner': '83ea11a812876d4494721d497fef3d453ff32e1fe785ab97c07a96af9b64411c',
 'block_table': '47b3ec1a19283ecc81bfebaf9c3b39af74e8a0dc8b2a1ed3fd0923f25f84daa7',
 'dsa': '9bae0cc54b57d9cf07f22046c60d83bcb43b6ce8291604ea3c42001ef1b6ffd6',
 'dspark': '5d1c3a39f83b6f937f2d9f2e6aac679171c4c44bad1e7b318bd5911221232f7c',
 'proposer': '8305debd5e3852b38c123394bb430871992286a92dc6ee7a0446e699e75ecdb7'})


def source_matches(path, name):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() == SOURCE_SHA256[name]
