"""Learned numerical action control with no language model or pretraining.

SFT and preference updates can use the same state/action supervision as the
LLM. This primitive cannot prove its caller's world separation or data lineage.
No environment, scientific generator or private evaluator is imported here.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from llm_intervention_learning import preference_loss


VERSION = "ace-numerical-ranker-v1"
ENCODING = "complete-padded-numeric-state-signed-log-v1"


class CandidateRanker(nn.Module):
    def __init__(self, feature_width, hidden=128, seed=0):
        super().__init__()
        if (type(feature_width) is not int or not 1 <= feature_width <= 20000
                or type(hidden) is not int or not 1 <= hidden <= 256
                or type(seed) is not int or seed < 0):
            raise ValueError("bounded ranker dimensions and nonnegative seed required")
        self.feature_width, self.hidden = feature_width, hidden
        # Seed only CPU. torch.manual_seed also changes accelerator RNGs, which
        # fork_rng(devices=[]) deliberately does not snapshot or initialize.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            options = {"device": "cpu", "dtype": torch.float32}
            self.network = nn.Sequential(nn.Linear(feature_width, hidden, **options), nn.Tanh(),
                                         nn.Linear(hidden, hidden, **options), nn.Tanh(),
                                         nn.Linear(hidden, 1, **options))

    def forward(self, features):
        if (features.ndim != 2 or features.shape[0] == 0
                or features.shape[1] != self.feature_width or not torch.isfinite(features).all()):
            raise ValueError("finite complete candidate feature rows required")
        result = self.network(features).squeeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite numerical policy scores")
        return result


def feature_tensor(values):
    array = np.array(values, dtype=np.float32, copy=True)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("finite candidate features required")
    return torch.from_numpy(array)


def supervised_rank_loss(scores, chosen):
    if type(chosen) is not int or not 0 <= chosen < len(scores):
        raise ValueError("valid supervised action index required")
    return -F.log_softmax(scores, dim=0)[chosen]


def fit_ranker(examples, *, scope, seed, hidden=128, sft_epochs=8, preference_epochs=4,
               learning_rate=.001, beta=.1):
    """Fixed final-iterate fitting. No score-based checkpoint or seed selection.

    Each example has features, one supervised action index, and zero or more
    chosen/rejected preference pairs. Ties must remain in the caller's full
    dataset and are represented here by no strict pair. Counts are explicit.
    """
    if scope not in ("artificial", "policy_train"):
        raise ValueError("ranker fitting is training-only")
    if (type(examples) is not tuple or not examples or len(examples) > 4096
            or type(sft_epochs) is not int or not 1 <= sft_epochs <= 100
            or type(preference_epochs) is not int or not 0 <= preference_epochs <= 100
            or type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0):
        raise ValueError("invalid fixed numerical training recipe")
    data = []
    for example in examples:
        if type(example) is not dict or set(example) != {"features", "chosen", "pairs"}:
            raise ValueError("unexpected numeric training fields")
        x = feature_tensor(example["features"])
        y, pairs = example["chosen"], example["pairs"]
        if type(y) is not int or not 0 <= y < len(x) or type(pairs) is not tuple:
            raise ValueError("invalid supervised label or preference list")
        if (any(type(pair) is not tuple or len(pair) != 2 or any(type(i) is not int or not 0 <= i < len(x) for i in pair)
                or pair[0] == pair[1] for pair in pairs) or len(set(pairs)) != len(pairs)
                or any((b, a) in pairs for a, b in pairs)):
            raise ValueError("invalid, duplicate or contradictory preference pair")
        data.append((x, y, pairs))
    width = data[0][0].shape[1]
    if any(x.shape[1] != width for x, _, _ in data):
        raise ValueError("feature width differs across training states")
    pair_count = sum(len(pairs) for _, _, pairs in data)
    if preference_epochs and not pair_count:
        raise ValueError("preference updates requested without a strict preference")
    model = CandidateRanker(width, hidden, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    events = []
    def step(loss, stage, epoch):
        if not torch.isfinite(loss):
            raise ValueError("nonfinite numeric training loss")
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError("missing or nonfinite numeric gradients")
        norm = float(torch.sqrt(sum((g*g).sum() for g in grads)))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite learned numerical parameters")
        events.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()), "gradient_norm": norm})
    for epoch in range(sft_epochs):
        loss = torch.stack([supervised_rank_loss(model(x), y) for x, y, _ in data]).mean()
        step(loss, "sft", epoch)
    sft = copy.deepcopy(model).eval().requires_grad_(False)
    reference = []
    with torch.no_grad():
        for x, _, pairs in data:
            scores = F.log_softmax(sft(x), dim=0)
            reference.extend([torch.stack([scores[a], scores[b]]) for a, b in pairs])
    ref = torch.stack(reference) if reference else torch.empty((0, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(preference_epochs):
        policy_pairs = []
        for x, _, pairs in data:
            scores = F.log_softmax(model(x), dim=0)
            policy_pairs.extend([torch.stack([scores[a], scores[b]]) for a, b in pairs])
        step(preference_loss(torch.stack(policy_pairs), ref, beta), "preference", epoch)
    return model.eval(), sft, {"version": VERSION, "scope": scope, "seed": seed,
        "feature_width": width, "hidden": hidden, "training_states": len(data),
        "strict_preferences": pair_count, "states_without_strict_preferences": sum(not p for _, _, p in data),
        "sft_epochs": sft_epochs, "preference_epochs": preference_epochs,
        "learning_rate": learning_rate, "beta": beta, "events": events,
        "scientific_world_independence_verified": False}


def save_ranker(model, path, *, training_receipt_sha256):
    if (type(training_receipt_sha256) is not str or len(training_receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in training_receipt_sha256)):
        raise ValueError("independent training receipt pin required")
    payload = {"version": VERSION, "encoding": ENCODING, "feature_width": model.feature_width,
               "hidden": model.hidden, "training_receipt_sha256": training_receipt_sha256,
               "state_dict": model.state_dict()}
    with Path(path).open("xb") as f:
        torch.save(payload, f)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ranker(path, *, sha256, training_receipt_sha256, feature_width):
    import io
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("numeric checkpoint digest differs")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (set(payload) != {"version", "encoding", "feature_width", "hidden", "training_receipt_sha256", "state_dict"}
            or payload["version"] != VERSION or payload["encoding"] != ENCODING
            or payload["training_receipt_sha256"] != training_receipt_sha256
            or payload["feature_width"] != feature_width):
        raise ValueError("numeric checkpoint contract differs")
    model = CandidateRanker(payload["feature_width"], payload["hidden"], seed=0)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite numeric checkpoint")
    return model.eval().requires_grad_(False)
