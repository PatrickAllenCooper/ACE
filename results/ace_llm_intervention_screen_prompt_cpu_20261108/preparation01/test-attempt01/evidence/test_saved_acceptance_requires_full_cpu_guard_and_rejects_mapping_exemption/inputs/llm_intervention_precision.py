"""An explicitly new FP32 output-projection recipe for a frozen language base.

The original BF16 output parameter can share storage with input embeddings.
It is retained unchanged. A separate frozen FP32 buffer supplies the output
projection, so changing projection arithmetic cannot cast the tied embedding.
This does not guarantee full/selected gradient agreement on a GPU.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


class FrozenFP32Projection(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if (type(original) is not torch.nn.Linear or original.bias is not None
                or original.weight.requires_grad or original.weight.dtype != torch.bfloat16
                or not torch.isfinite(original.weight).all()):
            raise ValueError("finite frozen BF16 bias-free linear projection required")
        self.original = original
        self.register_buffer("weight_fp32", original.weight.detach().to(torch.float32).clone())
        self.in_features, self.out_features = original.in_features, original.out_features

    @property
    def weight(self):
        return self.weight_fp32

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16
                or x.device != self.weight_fp32.device or x.shape[-1] != self.in_features
                or self.weight_fp32.dtype != torch.float32 or self.weight_fp32.requires_grad
                or self.original.weight.requires_grad or self.original.weight.dtype != torch.bfloat16):
            raise ValueError("frozen projection dtype/device/layout differs")
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight_fp32)


def install_fp32_projection(model):
    original = getattr(model, "lm_head", None)
    if isinstance(original, FrozenFP32Projection):
        raise ValueError("projection is already installed")
    model.lm_head = FrozenFP32Projection(original)
    return {"recipe": "frozen_fp32_output_projection_v1", "original_dtype": str(original.weight.dtype),
            "projection_dtype": str(model.lm_head.weight.dtype), "shape": list(original.weight.shape),
            "original_parameter_retained": model.lm_head.original is original,
            "independent_storage": model.lm_head.weight.data_ptr() != original.weight.data_ptr(),
            "new_trainable_parameters": 0}
