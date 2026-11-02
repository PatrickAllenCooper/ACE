"""State-conditioned action training primitives, independent of scientific data.

Adapters are implemented with torch alone so qualifying the existing runtime
does not require installing a new training framework. Actual model loading,
tokenizer/chat templates, data splits and GPU resources require separate gates.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def pack_actions(prompt_ids, action_ids, pad_token_id: int, max_tokens: int) -> ActionBatch:
    """Append action tokens to the exact state prompt, never silently truncate.

    The caller owns chat-template and completion tokenization. Reusing one
    prompt prefix for both preference actions is deliberate. Prompt and padding
    labels are masked, but their embeddings still condition action likelihood.
    """
    prompt = list(prompt_ids)
    actions = [list(x) for x in action_ids]
    if (not prompt or not actions or any(not x for x in actions)
            or type(pad_token_id) is not int or pad_token_id < 0
            or type(max_tokens) is not int or max_tokens < 2
            or any(type(x) is not int or x < 0 for seq in [prompt] + actions for x in seq)):
        raise ValueError("nonempty integer token sequences required")
    width = len(prompt) + max(map(len, actions))
    if width > max_tokens:
        raise ValueError("state/action exceeds the frozen context limit")
    ids = torch.full((len(actions), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, action in enumerate(actions):
        stop = len(prompt) + len(action)
        ids[i, :stop] = torch.tensor(prompt + action)
        attention[i, :stop] = 1
        labels[i, len(prompt):stop] = torch.tensor(action)
    return ActionBatch(ids, attention, labels)


def sequence_log_probs(model, batch: ActionBatch) -> torch.Tensor:
    device = next(model.parameters()).device
    ids, attention, labels = (x.to(device) for x in
                              (batch.input_ids, batch.attention_mask, batch.labels))
    output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    logits = output.logits[:, :-1].float()
    targets = labels[:, 1:]
    selected = targets != -100
    if not torch.all(selected.sum(dim=1) > 0):
        raise ValueError("every sequence needs at least one scored action token")
    safe_targets = targets.masked_fill(~selected, 0)
    values = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    scores = values.masked_fill(~selected, 0).sum(dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite action log probability")
    return scores


def supervised_loss(model, batch: ActionBatch) -> torch.Tensor:
    counts = (batch.labels[:, 1:] != -100).sum(dim=1)
    scores = sequence_log_probs(model, batch)
    return -(scores / counts.to(scores.device)).mean()


def preference_loss(policy_pairs: torch.Tensor, reference_pairs: torch.Tensor,
                    beta: float = 0.1) -> torch.Tensor:
    """DPO on chosen/rejected action sequence log probabilities at the SAME state."""
    if (policy_pairs.ndim != 2 or policy_pairs.shape[1] != 2 or policy_pairs.shape[0] == 0
            or reference_pairs.shape != policy_pairs.shape
            or not torch.isfinite(policy_pairs).all() or not torch.isfinite(reference_pairs).all()
            or type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0):
        raise ValueError("finite aligned [pair, chosen/rejected] scores and positive beta required")
    relative = policy_pairs - reference_pairs.detach().to(policy_pairs.device)
    return -F.logsigmoid(beta * (relative[:, 0] - relative[:, 1])).mean()


class LoRALinear(nn.Module):
    """Frozen linear base plus trainable low-rank update and fixed reference copy."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if (not isinstance(base, nn.Linear) or type(rank) is not int or rank <= 0
                or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("linear base, positive rank and alpha required")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                          dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.register_buffer("reference_A", self.A.detach().clone())
        self.register_buffer("reference_B", self.B.detach().clone())
        self.reference = False

    def forward(self, x):
        a, b = (self.reference_A, self.reference_B) if self.reference else (self.A, self.B)
        base = self.base(x)
        update = F.linear(F.linear(x.float(), a), b) * self.scale
        return base + update.to(base.dtype)

    def snapshot_reference(self):
        with torch.no_grad():
            self.reference_A.copy_(self.A)
            self.reference_B.copy_(self.B)


def install_adapters(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                     targets=("q_proj", "v_proj")) -> tuple[str, ...]:
    if any(isinstance(x, LoRALinear) for x in model.modules()):
        raise ValueError("adapters already installed")
    names = [(name, module) for name, module in model.named_modules()
             if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)]
    if not names:
        raise ValueError("no requested attention projections found")
    model.requires_grad_(False)
    for name, module in names:
        path = name.split(".")
        parent = model
        for item in path[:-1]:
            parent = getattr(parent, item)
        setattr(parent, path[-1], LoRALinear(module, rank, alpha))
    return tuple(name for name, _ in names)


def snapshot_reference(model: nn.Module) -> int:
    modules = [x for x in model.modules() if isinstance(x, LoRALinear)]
    if not modules:
        raise ValueError("no adapters to freeze as reference")
    for module in modules:
        module.snapshot_reference()
    return len(modules)


@contextmanager
def reference_policy(model: nn.Module):
    """Use fixed adapter buffers without mutating trainable parameters or graphs."""
    modules = [x for x in model.modules() if isinstance(x, LoRALinear)]
    if not modules or any(x.reference for x in modules):
        raise ValueError("missing adapters or nested reference policy")
    for module in modules:
        module.reference = True
    try:
        with torch.no_grad():
            yield model
    finally:
        for module in modules:
            module.reference = False
