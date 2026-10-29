# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Capacity planning without initializing the NPU plugin or a scheduler."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


source = Path(__file__).resolve().parents[3] / "vllm_ascend/kv_offload/avsg_adapter/native_admission.py"
spec = importlib.util.spec_from_file_location("avsg_native_admission_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def planner(host_free, host_domains=(0, 0), physical_free=10):
    admission = module.NativeAdmission.__new__(module.NativeAdmission)
    admission.pages = NS(available=(physical_free,), group_domains=(0, 0))
    admission._host_pools = tuple(NS(available=n) for n in host_free)
    admission._host_domains = dict(enumerate(host_domains))
    admission.pool = NS(free_block_queue=NS(fake_free_list_head=NS(next_free_block=None)))
    return admission


class HostHeadroomTests(unittest.TestCase):
    def test_sparse_cow_demand_preserves_every_groups_headroom(self):
        # One replacement plus two reserved pages requires three shared slots.
        for counts in ({0: 1}, {0: 1, 1: 0}):
            for free, expected in ((1, None), (2, None), (3, [])):
                with self.subTest(counts=counts, free=free):
                    self.assertEqual(planner((free,))._plan(counts, ((2,), (0, 2))), expected)

    def test_separate_host_pools_and_physical_reserve(self):
        for free, physical, expected in (((1, 1), 10, None),
                                          ((1, 2), 10, []),
                                          ((1, 2), 2, None)):
            with self.subTest(free=free, physical=physical):
                admission = planner(free, (0, 1), physical)
                self.assertEqual(admission._plan({0: 1}, ((2,), (0, 2))), expected)

    def test_shared_host_reserve_selects_enough_victims_without_mutation(self):
        admission = planner((1,))
        tail = NS(next_free_block=None)
        second = NS(block_id=8, ref_cnt=0, next_free_block=tail)
        first = NS(block_id=7, ref_cnt=0, next_free_block=second)
        admission.pool.free_block_queue.fake_free_list_head.next_free_block = first
        table = NS(snapshot=lambda ref: NS(pins=0))
        admission.apc = NS(_entries={n: NS(group=1, table=table, ref=n) for n in (7, 8)})
        admission.pages.binding = lambda logical: NS(domain=0)
        admission._incoming = set()
        self.assertEqual(admission._plan({0: 1}, ((2,), (0, 2))), [first, second])
        self.assertEqual(admission._host_pools[0].available, 1)
        self.assertIs(first.next_free_block, second)
        admission._incoming.add(8)
        self.assertIsNone(admission._plan({0: 1}, ((2,), (0, 2))))


if __name__ == "__main__":
    unittest.main()
