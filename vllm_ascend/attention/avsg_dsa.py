# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sparse attention bridge to the external AVSG runtime."""

def execute_dsa_attention(impl, decode, q_nope, q_rope, cache, indices, metadata):
    # Import after worker operator registration; DeviceOperator participates
    # in Ascend's operator/quantization initialization order.
    from vllm_ascend.device.device_op import DeviceOperator

    if getattr(decode, "graph_mode", False):
        if impl.enable_dsa_cp or q_nope.shape[0] != indices.shape[0] or q_rope.shape[0] != indices.shape[0]:
            raise ValueError("DSA graph attention requires padded TP1 query geometry")
        selected, table, remapped, cumulative, lengths = decode.select(impl.layer_name, cache, indices)
        pool = decode.consumer.pool(impl.layer_name)
        components = ((selected,) if pool.source_rope.shape[-1] == 0 else
                      (selected, pool.selection_rope.unsqueeze(2)))
        return DeviceOperator.execute_sparse_flash_attention_process(
            impl, q_nope, q_rope, components, remapped, metadata, cumulative, lengths, block_table=table)

    count = sum(decode.active.query_counts())
    if (impl.enable_dsa_cp or q_nope.shape[0] < count
            or q_rope.shape[0] != q_nope.shape[0] or indices.shape[0] < count):
        raise ValueError("DSA AVSG requires D TP1 and the exact scheduled query prefix")
    selected, table, remapped, cumulative, lengths = decode.select(
        impl.layer_name, cache, indices[:count])
    result = DeviceOperator.execute_sparse_flash_attention_process(
        impl, q_nope[:count], q_rope[:count], selected, remapped, metadata,
        cumulative, lengths, block_table=table)
    if q_nope.shape[0] == count:
        return result
    padded = result.new_zeros((q_nope.shape[0], *result.shape[1:]))
    padded[:count].copy_(result)
    return padded
