"""Fixed-row projection and a common action reduction for a new recipe.

Full and selected modes differ only in which hidden positions are projected.
Both use singleton projection rows and the same action-position reduction.
This preserves the mathematical masked-token objective, not historical
floating-point results. Actual GPU agreement and throughput remain unqualified.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from llm_intervention_precision import FrozenFP32Projection
from llm_intervention_action_scoring import action_positions
from llm_intervention_learning import deterministic_scoring


class FixedRowFP32Projection(FrozenFP32Projection):
    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.numel() == 0
                or x.shape[-1] != self.in_features):
            raise ValueError("nonempty batch/position/hidden projection input required")
        project = super().forward
        rows = x.reshape(-1, self.in_features)
        # The row count of each matmul is independent of padding, batch and
        # action position. Do not replace this with one larger matmul silently.
        values = [project(row.unsqueeze(0)) for row in rows]
        return torch.cat(values, dim=0).reshape(x.shape[0], x.shape[1], self.out_features)


def install_canonical_projection(model):
    original = getattr(model, "lm_head", None)
    model.lm_head = FixedRowFP32Projection(original)
    return {"recipe": "fixed_row_fp32_action_reduction_v1", "projection_rows_per_call": 1,
            "original_parameter_retained": model.lm_head.original is original,
            "independent_storage": model.lm_head.weight.data_ptr() != original.weight.data_ptr(),
            "original_dtype": str(original.weight.dtype), "projection_dtype": str(model.lm_head.weight.dtype),
            "new_trainable_parameters": 0}


def sequence_log_probs_canonical(model, batch, *, selected):
    if type(selected) is not bool or not isinstance(model.lm_head, FixedRowFP32Projection):
        raise ValueError("explicit canonical projection and path required")
    positions, targets, scored = action_positions(batch)
    parameters = tuple(model.parameters())
    if not parameters or len({p.device for p in parameters}) != 1:
        raise ValueError("one-device model required")
    device = parameters[0].device
    arguments = {"input_ids": batch.input_ids.to(device), "attention_mask": batch.attention_mask.to(device),
                 "use_cache": False}
    if selected:
        arguments["logits_to_keep"] = positions.to(device)
    with deterministic_scoring(model):
        logits = model(**arguments).logits
    expected = len(positions) if selected else batch.input_ids.shape[1]
    if (type(logits) is not torch.Tensor or logits.dtype != torch.float32 or logits.ndim != 3
            or logits.shape[:2] != (batch.input_ids.shape[0], expected)
            or logits.shape[2] == 0 or not torch.isfinite(logits).all()):
        raise ValueError("canonical projection returned invalid logits")
    if not selected:
        logits = logits.index_select(1, positions.to(device))
    targets, scored = targets.to(device), scored.to(device)
    if torch.any(targets[scored] >= logits.shape[-1]):
        raise ValueError("action label exceeds vocabulary")
    safe = targets.masked_fill(~scored, 0)
    values = F.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    scores = values.masked_fill(~scored, 0).sum(dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite canonical action scores")
    return scores


def supervised_loss_canonical(model, batch, *, selected):
    scores = sequence_log_probs_canonical(model, batch, selected=selected)
    counts = (batch.labels[:, 1:] != -100).sum(dim=1).to(scores.device)
    return -(scores/counts).mean()
