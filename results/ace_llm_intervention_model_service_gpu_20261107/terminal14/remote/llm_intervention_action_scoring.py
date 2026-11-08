"""Action-token scoring without vocabulary logits for the full state prompt.

The full input sequence still passes through the causal transformer with
gradients. Only the final vocabulary projection is restricted to positions
that predict scored action tokens. The model must implement tensor-valued
logits_to_keep as qualified for the pinned Qwen implementation. There is no
fallback to an unqualified approximation or truncated context.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from llm_intervention_learning import ActionBatch, deterministic_scoring


def action_positions(batch):
    if not isinstance(batch, ActionBatch):
        raise ValueError("explicit action batch required")
    ids, attention, labels = batch.input_ids, batch.attention_mask, batch.labels
    if (any(type(x) is not torch.Tensor or x.dtype != torch.long or x.ndim != 2
            for x in (ids, attention, labels)) or ids.shape != attention.shape or ids.shape != labels.shape
            or ids.shape[0] == 0 or ids.shape[1] < 2
            or any(x.device != ids.device for x in (attention, labels))):
        raise ValueError("aligned integer action tensors required")
    scored = labels != -100
    if (not torch.all(ids >= 0) or not torch.all((attention == 0) | (attention == 1))
            or not torch.all(attention[:, 0] == 1) or torch.any(attention[:, 1:] > attention[:, :-1])
            or torch.any(scored[:, 0]) or torch.any(scored & (attention == 0))
            or not torch.equal(labels[scored], ids[scored])
            or not torch.all(scored[:, 1:].sum(dim=1) > 0)):
        raise ValueError("invalid causal labels, padding or action token identity")
    # Position j predicts token j+1. The union allows unequal action lengths.
    positions = torch.nonzero(scored[:, 1:].any(dim=0), as_tuple=False).flatten()
    targets = labels[:, 1:].index_select(1, positions)
    return positions, targets, targets != -100


def projection_profile(batch, vocabulary_size):
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        raise ValueError("positive vocabulary size required")
    positions, _, _ = action_positions(batch)
    b, length = batch.input_ids.shape
    return {"batch_size": b, "input_positions": length, "projected_positions": len(positions),
        "vocabulary_size": vocabulary_size, "full_logit_elements": b*length*vocabulary_size,
        "selected_logit_elements": b*len(positions)*vocabulary_size,
        "scope": "tensor element count only, not total memory or measured speed"}


def sequence_log_probs_selected(model, batch):
    positions, targets, scored = action_positions(batch)
    parameters = tuple(model.parameters())
    if not parameters or len({p.device for p in parameters}) != 1:
        raise ValueError("one-device model required")
    device = parameters[0].device
    with deterministic_scoring(model):
        output = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device),
                       use_cache=False, logits_to_keep=positions.to(device))
    logits = output.logits
    if (type(logits) is not torch.Tensor or logits.ndim != 3
            or tuple(logits.shape[:2]) != (batch.input_ids.shape[0], len(positions))
            or logits.shape[2] == 0 or not torch.isfinite(logits).all()):
        raise ValueError("model did not return finite selected-position vocabulary logits")
    targets, scored = targets.to(device), scored.to(device)
    if torch.any(targets[scored] >= logits.shape[-1]):
        raise ValueError("action target exceeds returned vocabulary")
    safe = targets.masked_fill(~scored, 0)
    selected = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    result = selected.masked_fill(~scored, 0).sum(dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite action sequence score")
    return result


def supervised_loss_selected(model, batch):
    scores = sequence_log_probs_selected(model, batch)
    counts = (batch.labels[:, 1:] != -100).sum(dim=1).to(scores.device)
    return -(scores/counts).mean()
