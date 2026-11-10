#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""HiF8 (HiFloat8) W8A8 Linear for DeepSeek-V4 family checkpoints.

Torch has no HiFloat8 dtype, so every HiF8 tensor is carried as ``torch.uint8``
(the raw 8-bit code) and the kernels are told how to read it through their
``*_dtype=torch_npu.hifloat8`` arguments:

* weight: per-channel (per output row) HiF8, produced at load time from the
  checkpoint's block-scaled FP8 matrix (dequantize -> ``npu_dynamic_quant``);
  the FP32 row scales go through ``npu_trans_quant_param`` once.
* activation: per-tensor HiF8 with unit scale, ``npu_dtype_cast`` per call.
  Inputs that are already HiF8 codes (uint8, e.g. the wo_a batch-matmul output
  feeding wo_b) are passed through.

``wo_a`` is grouped (o_groups x o_lora_rank rows) and runs through
``npu_transpose_quant_batchmatmul``, which takes one per-channel scale vector
for all groups; its rows are therefore quantized per o_lora row *across* the
groups, and the weight is left in the same [groups, group_in, o_lora_rank]
layout as the MXFP8 path.

Selected instead of ``ds_linear`` by ``VLLM_ASCEND_DSV41_HIF8_LINEAR=1``;
shared experts stay on the MXFP8 ``ds_linear`` path and MoE is unchanged.
"""

import torch
import torch_npu

from vllm_ascend.utils import FP8_METHOD

from ..registry import register_scheme
from .w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod

HIF8_LINEAR_ENV = "VLLM_ASCEND_DSV41_HIF8_LINEAR"
_HIF8_FLOAT_INPUTS = (torch.bfloat16, torch.float16, torch.float32)

# Accuracy knobs (all default to the original behaviour):
#  VLLM_ASCEND_DSV41_HIF8_WEIGHT_RANGE=<M>  per-channel weight scale maps max|row| -> M (e.g. 16).
#      "dyn" (default) keeps npu_dynamic_quant, which maps max|row| to ~3.2e4, deep in HiF8's
#      coarse large-magnitude region: measured 10.3% weight-only error vs 2.5% at M in [4, 16]
#      and 1.1% for MXFP8.
#  VLLM_ASCEND_DSV41_HIF8_ACT_CALIB_STEPS=<N>  static per-tensor activation scale per Linear,
#      calibrated as the running max|x| over the first N calls (kept on device, no host sync),
#      mapped to VLLM_ASCEND_DSV41_HIF8_ACT_RANGE (default 16). 0 (default) = unit-scale cast
#      (5.2% activation-only error vs 2.7% at max -> 16).
#  VLLM_ASCEND_DSV41_HIF8_MX_LINEARS=wq_b,wo_a  Linear name suffixes kept on MXFP8.


def _env(name, default):
    import os

    return os.environ.get(name, default)


def hif8_weight_range():
    value = _env("VLLM_ASCEND_DSV41_HIF8_WEIGHT_RANGE", "dyn")
    return None if value == "dyn" else float(value)


def hif8_act_calib_steps() -> int:
    return int(_env("VLLM_ASCEND_DSV41_HIF8_ACT_CALIB_STEPS", "0"))


def hif8_act_range() -> float:
    return float(_env("VLLM_ASCEND_DSV41_HIF8_ACT_RANGE", "16"))


def hif8_act_calib_skip_suffixes() -> tuple:
    # engram.wkv sees O(5) inputs where a max->16 static scale measured worse than the unit cast.
    value = _env("VLLM_ASCEND_DSV41_HIF8_ACT_CALIB_SKIP", "engram.wkv")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def hif8_mx_linear_suffixes() -> tuple:
    value = _env("VLLM_ASCEND_DSV41_HIF8_MX_LINEARS", "")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def keep_linear_on_mxfp8(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in hif8_mx_linear_suffixes())


def _quantize_weight_rows(dense: torch.Tensor):
    """Per-row HiF8 codes (uint8) + FP32 row scales for a [rows, cols] matrix."""
    weight_range = hif8_weight_range()
    if weight_range is None:
        quantized, row_scale = torch_npu.npu_dynamic_quant(dense.to(torch.bfloat16), dst_type=torch_npu.hifloat8)
        return _as_hif8_codes(quantized), row_scale.to(torch.float32).view(-1)
    row_scale = dense.float().abs().amax(dim=1).clamp_min(1e-12) / weight_range
    scaled = (dense.float() / row_scale.view(-1, 1)).to(torch.bfloat16)
    return _as_hif8_codes(torch_npu.npu_dtype_cast(scaled, dtype=torch_npu.hifloat8)), row_scale


def hif8_linear_enabled() -> bool:
    import os

    return os.environ.get(HIF8_LINEAR_ENV, "0") == "1"


def _dequantize_block_fp8(weight: torch.Tensor, block_scale: torch.Tensor, block: int) -> torch.Tensor:
    """[out, in] E4M3 with [ceil(out/block), ceil(in/block)] scales -> FP32 dense matrix."""
    if block_scale.dtype == torch.float8_e8m0fnu:
        block_scale = block_scale.to(torch.float32)  # numeric cast: exact powers of two
    out_features, in_features = weight.shape
    dense_scale = block_scale.to(torch.float32).repeat_interleave(block, dim=0)[:out_features]
    dense_scale = dense_scale.repeat_interleave(block, dim=1)[:, :in_features]
    return weight.to(torch.float32) * dense_scale


def _as_hif8_codes(quantized: torch.Tensor) -> torch.Tensor:
    return quantized if quantized.dtype == torch.uint8 else quantized.view(torch.uint8)


@register_scheme(FP8_METHOD, "ds_linear_hif8")
class AscendW8A8HIF8DSDynamicLinearMethod(AscendW8A8MXFP8DSDynamicLinearMethod):
    """Loads exactly like ``ds_linear`` (same parameters / checkpoint tensors), runs as HiF8."""

    supports_weight_switch = False
    weight_switch_gather_specs = ()
    weight_switch_output_gather_specs = ()

    # ops/cv_linear.py's CVLinearWrapper dispatches on
    # isinstance(scheme, AscendW8A8MXFP8DynamicLinearMethod), which this class satisfies by
    # inheritance, and its MXFP8 branch reads layer.weight / layer.weight_scale directly with
    # scale_dtype=float8_e8m0fnu. Our weights are HiF8 codes and our weight_scale is the int64
    # npu_trans_quant_param output, so that branch fails on the first forward with
    #   Tensor dtype:long int itemsize:8, is not compatible with
    #   tensor_dtype:torch_npu.float8_e8m0fnu itemsize:1
    # This flag opts wq_a / wkv / wq_b / indexer.wq_b out of the Cube/Vector fast path and back
    # onto apply() below, which quantizes the activation itself. The cost is losing that overlap.
    is_hif8_scheme = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dense = _dequantize_block_fp8(layer.weight.data, layer.weight_scale.data, self.block_size)
        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("wo_a"):
            # The layer may use OTP instead of the model's ordinary TP group, so the group count
            # comes from the loaded output shard -- the same rule as the MXFP8 ds_linear path,
            # which reads it off the already-transposed weight. ``dense`` is still [out, in].
            rank = self.o_lora_rank
            rows, cols = dense.shape
            if rows % rank:
                raise ValueError(
                    f"{prefix}: wo_a output shard {rows} is not a multiple of o_lora_rank {rank}")
            groups = rows // rank
            grouped = dense.view(groups, rank, cols).transpose(0, 1).reshape(rank, groups * cols)
            codes, row_scale = _quantize_weight_rows(grouped)
            # [rank, groups, cols] -> [groups, cols (group_in), rank]: the MXFP8 wo_a layout.
            codes = codes.view(rank, groups, cols).permute(1, 2, 0).contiguous()
            layer.weight.data = codes
            layer.weight_scale.data = torch_npu.npu_trans_quant_param(row_scale.to(torch.float32)).view(-1)
            layer.hif8_bmm_x1_scale = torch_npu.npu_trans_quant_param(
                torch.ones(1, dtype=torch.float32, device=codes.device))
            layer.hif8_quant_scale = torch.ones(1, dtype=torch.bfloat16, device=codes.device)
        else:
            codes, row_scale = _quantize_weight_rows(dense)
            codes = codes.contiguous()
            # Same weight format as the recipe's HiF8 Linear: NZ, then a transposed view.
            layer.weight.data = torch_npu.npu_format_cast(codes, 29).transpose(0, 1)
            layer.weight_scale.data = torch_npu.npu_trans_quant_param(row_scale.to(torch.float32)).view(-1)
        del dense
        layer.is_hif8 = True
        # wo_a's activation goes through apply_hif8_wo_a, not apply(); skip-listed Linears keep the unit cast.
        if not prefix.endswith("wo_a") and not any(prefix.endswith(x) for x in hif8_act_calib_skip_suffixes()):
            _init_act_calibration(layer, layer.weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            # PreparedLinearInput / (tensor, scale) are the MXFP8 fused-quant calling conventions:
            # they carry an MX per-token scale this kernel has no argument for.
            raise TypeError(
                f"{getattr(layer, 'prefix', 'linear')}: HiF8 Linear got a pre-quantized "
                f"{type(x).__name__} input, which is the MXFP8 fused-quant calling convention.")
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        act_scale = None
        if x.dtype in _HIF8_FLOAT_INPUTS:
            output_dtype = x.dtype
            x, act_scale = _quantize_activation(layer, x)
        else:
            output_dtype = torch.bfloat16
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8,
            output_dtype=output_dtype,
        )
        if act_scale is not None:
            output = output * act_scale
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)
        return output


def _init_act_calibration(layer: torch.nn.Module, device) -> None:
    steps = hif8_act_calib_steps()
    layer.hif8_calib_left = steps
    if steps > 0:
        layer.hif8_act_absmax = torch.zeros(1, dtype=torch.float32, device=device)
        layer.hif8_act_scale = torch.ones(1, dtype=torch.bfloat16, device=device)
        layer.hif8_act_inv_scale = torch.ones(1, dtype=torch.bfloat16, device=device)


def _quantize_activation(layer: torch.nn.Module, x: torch.Tensor):
    """HiF8 codes for a float activation, plus the scale the output must be multiplied by (or None)."""
    if getattr(layer, "hif8_calib_left", 0) <= 0 and not hasattr(layer, "hif8_act_absmax"):
        return torch_npu.npu_dtype_cast(x, dtype=torch_npu.hifloat8), None
    if layer.hif8_calib_left > 0 and x.numel() > 0:
        # Running max on device; the scale tensors are updated in place (stable addresses).
        torch.maximum(layer.hif8_act_absmax, x.detach().abs().max().float().view(1), out=layer.hif8_act_absmax)
        scale = layer.hif8_act_absmax.clamp_min(1e-6) / hif8_act_range()
        layer.hif8_act_scale.copy_(scale)
        layer.hif8_act_inv_scale.copy_(1.0 / scale)
        layer.hif8_calib_left -= 1
    # npu_quantize computes x * scale, so pass the inverse scale.
    codes = torch_npu.npu_quantize(x, layer.hif8_act_inv_scale, None, torch_npu.hifloat8, -1, False)
    return codes, layer.hif8_act_scale


def apply_hif8_wo_a(wo_a: torch.nn.Module, o: torch.Tensor, perm_x1, perm_x2, perm_y) -> torch.Tensor:
    """HiF8 grouped o_a projection; returns HiF8 codes that wo_b consumes without a re-cast."""
    o = torch_npu.npu_quantize(o, wo_a.hif8_quant_scale, None, torch_npu.hifloat8, -1, False)
    return torch_npu.npu_transpose_quant_batchmatmul(
        o,
        wo_a.weight,
        dtype=torch_npu.hifloat8,
        x1_scale=wo_a.hif8_bmm_x1_scale,
        x2_scale=wo_a.weight_scale,
        group_sizes=(0, 0, 0),
        perm_x1=perm_x1,
        perm_x2=perm_x2,
        perm_y=perm_y,
        batch_split_factor=1,
        x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8,
    )


__all__ = ["AscendW8A8HIF8DSDynamicLinearMethod", "apply_hif8_wo_a", "hif8_linear_enabled",
           "keep_linear_on_mxfp8", "HIF8_LINEAR_ENV"]
