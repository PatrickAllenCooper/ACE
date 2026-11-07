"""CPU numerical control trained on the exact matched exposure proposal.

No world generation or policy evaluation. Caller owns original provenance,
scientific adoption, source/runtime pins and process/resource supervision.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

from llm_intervention_contract import canonical, digest, integer
from llm_intervention_learning import preference_loss
from llm_intervention_ranker import CandidateRanker, ENCODING, feature_tensor, load_ranker, save_ranker, supervised_rank_loss
from llm_intervention_supervision import matched_views
from llm_intervention_training_exposure import propose_exposure


VERSION = "ace-matched-numerical-training-v1"
MAX_PARAMETERS = 3000000


def numeric_recipe(*, artificial):
    if type(artificial) is not bool:
        raise ValueError("explicit artificial/scientific recipe required")
    return {"version": VERSION, "artificial_control": artificial,
        "hidden": 4 if artificial else 128, "seed": 109, "dtype": "float32", "device": "cpu",
        "optimizer": "Adam", "learning_rate": .001, "beta": .1, "betas": [.9, .999],
        "eps": 1e-8, "weight_decay": 0., "amsgrad": False, "foreach": False, "fused": False,
        "gradient_clipping": None, "optimizer_state": "continuous_across_sft_dpo",
        "torch_threads": 1, "deterministic_algorithms": True,
        "exposure": "one_sft_per_record_then_at_most_one_strict_pair_per_record"}


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    with path.open("x") as f:
        f.write(canonical(value)+"\n")
        f.flush()
        os.fsync(f.fileno())


def _tensor_file(path, value):
    with path.open("xb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())


def _signature(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        h.update(canonical([name, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _prepare(*, records, record_pins, supervision_receipt_pins, record_roster, scope,
             exposure, exposure_sha256, recipe, checkpoint_steps, outer_provenance_receipt_sha256):
    if (type(recipe) is not dict or type(recipe.get("artificial_control")) is not bool
            or canonical(recipe) != canonical(numeric_recipe(artificial=recipe["artificial_control"]))
            or recipe["artificial_control"] and scope != "artificial"
            or not _pin(outer_provenance_receipt_sha256)):
        raise ValueError("explicit fixed numerical recipe and outer provenance pin required")
    if not _pin(exposure_sha256) or digest(exposure) != exposure_sha256:
        raise ValueError("independent exposure pin required")
    expected = propose_exposure(records=records, record_pins=record_pins,
        supervision_receipt_pins=supervision_receipt_pins, record_roster=record_roster, scope=scope)
    if canonical(expected) != canonical(exposure) or not exposure["has_both_training_stages"]:
        raise ValueError("exact complete two-stage exposure proposal required")
    steps = exposure["numeric_schedule"]
    if (type(checkpoint_steps) is not tuple or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
            or any(not integer(i) or i >= len(steps) for i in checkpoint_steps)):
        raise ValueError("fixed sorted checkpoint indices required")
    features, feature_pins, widths = {}, {}, set()
    for rid in record_roster:
        views = matched_views(records[rid], record_sha256=record_pins[rid],
            supervision_receipt_sha256=supervision_receipt_pins[rid], allowed_scope=scope)
        x = feature_tensor(views["numeric"]["features"])
        features[rid] = x
        widths.add(x.shape[1])
        feature_pins[rid] = {"shape": list(x.shape), "float32_tensor_sha256": _signature([("features", x)])}
    if len(widths) != 1:
        raise ValueError("feature width differs across records")
    width = widths.pop()
    hidden = recipe["hidden"]
    parameters = width*hidden + hidden + hidden*hidden + hidden + hidden + 1
    if not 1 <= width <= 20000 or parameters > MAX_PARAMETERS:
        raise ValueError("numerical parameter bound exceeded")
    plan = {"version": VERSION, "scope": scope, "recipe": recipe,
        "raw_records_utf8": {rid: records[rid].decode("utf-8") for rid in record_roster},
        "record_pins": record_pins, "supervision_receipt_pins": supervision_receipt_pins,
        "record_roster": record_roster, "exposure": exposure, "exposure_sha256": exposure_sha256,
        "feature_pins": feature_pins, "encoding": ENCODING, "feature_width": width,
        "parameter_count": parameters, "checkpoint_steps": checkpoint_steps,
        "outer_provenance_receipt_sha256": outer_provenance_receipt_sha256,
        "scientific_provenance_qualified": False, "retry_allowed": False}
    return json.loads(canonical(plan)), features


def prepare_plan(**inputs):
    """CPU preflight. Feature hashes bind derived FP32 tensors, not their provenance."""
    return _prepare(**inputs)[0]


def fit(*, plan, plan_sha256, output):
    """Initialize one pristine CPU ranker and follow every fixed exposure step.

    Full input/feature reconstruction precedes any parameter initialization.
    A single feasible action legitimately gives zero SFT gradient. Such rows
    still receive their declared Adam step, including existing momentum.
    """
    if not _pin(plan_sha256) or digest(plan) != plan_sha256 or plan.get("version") != VERSION:
        raise ValueError("independently pinned numerical plan required")
    plan = json.loads(canonical(plan))
    checked, features = _prepare(records={k: v.encode("utf-8") for k, v in plan["raw_records_utf8"].items()},
        record_pins=plan["record_pins"], supervision_receipt_pins=plan["supervision_receipt_pins"],
        record_roster=tuple(plan["record_roster"]), scope=plan["scope"], exposure=plan["exposure"],
        exposure_sha256=plan["exposure_sha256"], recipe=plan["recipe"],
        checkpoint_steps=tuple(plan["checkpoint_steps"]),
        outer_provenance_receipt_sha256=plan["outer_provenance_receipt_sha256"])
    if canonical(checked) != canonical(plan):
        raise ValueError("prepared features or plan fields differ")
    if (torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError("caller must declare one-thread deterministic CPU execution")
    output = Path(output)
    output.mkdir(exist_ok=False)
    started, model, reference, optimizer, attempt = time.monotonic(), None, None, None, None
    events, ref_scores = [], {}
    recipe, exposure = plan["recipe"], plan["exposure"]

    def checkpoint(stem, *, gradients=False):
        if model is None:
            return
        _tensor_file(output/(stem+".state.pt"), {n: v.detach().clone() for n, v in model.state_dict().items()})
        files = {"state": _sha(output/(stem+".state.pt"))}
        if optimizer is not None:
            _tensor_file(output/(stem+".optimizer.pt"), optimizer.state_dict())
            files["optimizer"] = _sha(output/(stem+".optimizer.pt"))
        if gradients:
            _tensor_file(output/(stem+".gradients.pt"), {n: None if p.grad is None else p.grad.detach().clone()
                                                     for n, p in model.named_parameters()})
            files["gradients"] = _sha(output/(stem+".gradients.pt"))
        _save(output/(stem+".checkpoint.json"), {"files": files, "completed_steps": len(events),
            "attempt": attempt, "parameter_signature": _signature(model.named_parameters())})

    try:
        _save(output/"plan.json", plan)
        _save(output/"initialization-start.json", {"utc": _utc(), "recipe": recipe})
        model = CandidateRanker(plan["feature_width"], hidden=recipe["hidden"], seed=recipe["seed"]).eval()
        if sum(p.numel() for p in model.parameters()) != plan["parameter_count"]:
            raise ValueError("initialized parameter count differs")
        initial_signature = _signature(model.named_parameters())
        _save(output/"initialization-complete.json", {"utc": _utc(), "signature": initial_signature})
        checkpoint("initial")
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"],
            betas=tuple(recipe["betas"]), eps=recipe["eps"], weight_decay=recipe["weight_decay"],
            amsgrad=recipe["amsgrad"], foreach=recipe["foreach"], fused=recipe["fused"])
        reference_signature = None
        for index, step in enumerate(exposure["numeric_schedule"]):
            if step["stage"] == "dpo" and reference is None:
                checkpoint("sft")
                reference = copy.deepcopy(model).eval().requires_grad_(False)
                reference.zero_grad(set_to_none=True)
                reference_signature = _signature(reference.named_parameters())
                _tensor_file(output/"reference.pt", reference.state_dict())
                with torch.no_grad():
                    for rid, x in features.items():
                        ref_scores[rid] = F.log_softmax(reference(x), dim=0).clone()
                _save(output/"reference.json", {"parameter_signature": reference_signature,
                    "state_sha256": _sha(output/"reference.pt"), "sft_checkpoint_sha256": _sha(output/"sft.state.pt"),
                    "scores": {rid: v.tolist() for rid, v in ref_scores.items()}, "completed_sft_steps": len(events)})
            attempt = {"index": index, **step, "utc": _utc(), "optimizer_started": False, "optimizer_returned": False}
            _save(output/f"step-{index:04d}.start.json", attempt)
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            rid = step["record_id"]
            scores = model(features[rid])
            if step["stage"] == "sft":
                loss = supervised_rank_loss(scores, step["chosen_index"])
            else:
                ids = [step["chosen_index"], step["rejected_index"]]
                loss = preference_loss(F.log_softmax(scores, dim=0)[ids].reshape(1, 2),
                    ref_scores[rid][ids].reshape(1, 2), beta=recipe["beta"])
            value = float(loss.detach())
            _save(output/f"step-{index:04d}.loss.json", {"finite": math.isfinite(value),
                "loss": value if math.isfinite(value) else None, "loss_repr": repr(value)})
            if not math.isfinite(value):
                raise ValueError("nonfinite numerical loss")
            loss.backward()
            grads = [{"name": n, "present": p.grad is not None,
                "finite": p.grad is not None and bool(torch.isfinite(p.grad).all())} for n, p in model.named_parameters()]
            _save(output/f"step-{index:04d}.gradients.json", {"parameters": grads})
            if any(not g["present"] or not g["finite"] for g in grads):
                raise ValueError("missing or nonfinite numerical gradient")
            norm = math.sqrt(sum(float((p.grad.double()**2).sum()) for p in model.parameters()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite numerical gradient norm")
            if reference is not None and (_signature(reference.named_parameters()) != reference_signature
                    or any(p.grad is not None or p.requires_grad for p in reference.parameters())):
                raise ValueError("fixed post-SFT reference changed")
            attempt["optimizer_started"] = True
            _save(output/f"step-{index:04d}.optimizer-attempt.json", attempt)
            optimizer.step()
            attempt["optimizer_returned"] = True
            _save(output/f"step-{index:04d}.optimizer-returned.json", attempt)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("nonfinite updated numerical parameters")
            event = {**attempt, "loss": value, "gradient_norm": norm, "zero_gradient": norm == 0.,
                "candidate_count": len(scores), "wall_seconds": time.monotonic()-before}
            _save(output/f"step-{index:04d}.complete.json", event)
            events.append(event)
            if index in plan["checkpoint_steps"]:
                checkpoint(f"step-{index:04d}")
        if _signature(reference.named_parameters()) != reference_signature:
            raise ValueError("reference changed at completion")
        with torch.no_grad():
            for rid, x in features.items():
                if not torch.equal(F.log_softmax(reference(x), dim=0), ref_scores[rid]):
                    raise ValueError("fixed reference score changed")
        checkpoint("final")
        training = {"version": VERSION, "utc": _utc(), "status": "training_complete_export_pending", "plan_sha256": plan_sha256,
            "exposure_sha256": plan["exposure_sha256"], "completed_steps": len(events), "steps": events,
            "parameter_count": plan["parameter_count"], "initial_signature": initial_signature,
            "final_signature": _signature(model.named_parameters()), "reference_signature": reference_signature,
            "checkpoints": {s: _sha(output/(s+".state.pt")) for s in ("initial", "sft", "final")},
            "scientific_provenance_qualified": False}
        _save(output/"training_complete.json", training)
        training_pin, exports = _sha(output/"training_complete.json"), {}
        for role, fitted in (("sft", reference), ("dpo", model)):
            path = output/(role+".ranker.pt")
            h = save_ranker(fitted, path, training_receipt_sha256=training_pin)
            restored = load_ranker(path, sha256=h, training_receipt_sha256=training_pin,
                                   feature_width=plan["feature_width"])
            with torch.no_grad():
                for x in features.values():
                    if not torch.equal(restored(x), fitted(x)):
                        raise ValueError("fresh exported ranker scores differ")
            exports[role] = {"path": path.name, "sha256": h, "training_receipt_sha256": training_pin,
                "parameter_signature": _signature(restored.named_parameters()), "all_training_score_rows_equal": True}
        terminal = {"version": VERSION, "utc": _utc(), "status": "complete", "scope": plan["scope"],
            "plan_sha256": plan_sha256, "exposure_sha256": plan["exposure_sha256"],
            "training_receipt_sha256": training_pin, "exports": exports,
            "completed_steps": len(events), "sft_steps": exposure["counts"]["sft_steps"],
            "dpo_steps": exposure["counts"]["dpo_steps"], "zero_gradient_steps": sum(e["zero_gradient"] for e in events),
            "elapsed_seconds": time.monotonic()-started, "retry_allowed": False,
            "scientific_provenance_qualified": False, "environment_responses": 0}
        _save(output/"terminal.json", terminal)
        return terminal
    except BaseException as exc:
        failure = {"status": "failed", "utc": _utc(), "exception_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "plan_sha256": plan_sha256, "completed_steps": events,
            "attempt": attempt, "retry_allowed": False, "scientific_provenance_qualified": False,
            "update_outcome": "unknown_or_partial" if attempt and attempt["optimizer_started"] and not attempt["optimizer_returned"]
                              else "see_retained_step_events"}
        try:
            checkpoint("failed-partial", gradients=True)
            if reference is not None:
                _tensor_file(output/"failed-partial.reference.pt", reference.state_dict())
            failure["partial_state_saved"] = model is not None
        except BaseException as save_error:
            failure["checkpoint_error"] = type(save_error).__name__+": "+str(save_error)
        _save(output/"failure.json", failure)
        raise
