"""Bounded SFT -> fixed post-SFT-reference DPO, with caller-owned admission.

CPU preparation/tokenization is separate from fitting an already initialized
model. No model loading, scientific generator, environment, fallback or resume.
Artificial control tests do not qualify Qwen execution or scientific provenance.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from llm_intervention_contract import canonical, digest, integer
from llm_intervention_language_policy import pin, prepare_task_tokens
from llm_intervention_learning import LoRALinear, pack_actions, preference_loss, reference_policy, snapshot_reference
from llm_intervention_canonical_scoring import (
    FixedRowFP32Projection, sequence_log_probs_canonical, supervised_loss_canonical,
    sequence_log_probs_unpadded, supervised_loss_unpadded,
)
from llm_intervention_checkpointing import DecoderCheckpointing
from llm_intervention_supervision import matched_views
from qualify_llm_intervention_checkpoint_gpu import RECIPE as GPU09_RECIPE


VERSION = "ace-bounded-training-engine-v1"
MAX_RECORDS, MAX_STEPS, MAX_STORED_TOKENS, MAX_PARAMETERS = 256, 16384, 4194304, 1146880
NUMERIC_KEYS = ("seed", "rank", "alpha", "learning_rate", "beta", "max_tokens", "max_new_tokens",
    "attention", "sdpa_backend", "base_dtype", "adapter_dtype", "activation_checkpointing",
    "strict_determinism", "float32_matmul", "output_projection")


def gpu09_numeric_recipe():
    """Arithmetic/optimizer recipe only; qualification's 3+3 steps are not copied."""
    return {**{k: GPU09_RECIPE[k] for k in NUMERIC_KEYS}, "optimizer": "AdamW",
        "weight_decay": 0., "adam_betas": [.9, .999], "adam_eps": 1e-8,
        "amsgrad": False, "foreach": None, "fused": None, "maximize": False,
        "capturable": False, "differentiable": False, "clip_gradient_norm": 1.,
        "optimizer_state": "continuous_across_sft_dpo", "scoring_path": "canonical_selected"}


def screen_numeric_recipe():
    """Prospective singleton decoder layout; historical arithmetic stays named."""
    return {**gpu09_numeric_recipe(), "scoring_path": "canonical_selected_unpadded"}


def _recipe(value):
    if type(value) is not dict or canonical(value) not in (
            canonical(gpu09_numeric_recipe()), canonical(screen_numeric_recipe())):
        raise ValueError("exact explicit numeric recipe required; no defaults or memory rescue")


def _execution_recipe(recipe, execution_kind):
    _recipe(recipe)
    if (execution_kind not in ('qwen_gpu09', 'qwen_screen_singletons', 'artificial_control')
            or execution_kind == 'qwen_gpu09' and recipe != gpu09_numeric_recipe()
            or execution_kind == 'qwen_screen_singletons' and recipe != screen_numeric_recipe()):
        raise ValueError('execution kind and explicit scoring recipe disagree')


def _name(value):
    return type(value) is str and 0 < len(value) <= 128


def _save(path, value):
    with path.open("x") as handle:
        handle.write(canonical(value)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tensors(path, value):
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _file_pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_record(*, record_id, raw, record_sha256, supervision_receipt_sha256,
                   scope, tokenizer, tokenizer_sha256, numeric_recipe):
    """CPU-only matched-record validation and exact chat/command tokenization.

    Prefix stored once; no tokenization or grammar construction in fit(). Keep
    the returned record's digest in an independently supplied preparation manifest.
    Tokenizer bytes and provenance are authenticated by the outer caller.
    """
    _recipe(numeric_recipe)
    if not _name(record_id) or not pin(tokenizer_sha256):
        raise ValueError("bounded record identity and tokenizer pin required")
    views = matched_views(raw, record_sha256=record_sha256,
        supervision_receipt_sha256=supervision_receipt_sha256, allowed_scope=scope)
    record = json.loads(raw)
    grammar, meta = prepare_task_tokens(record["envelope"], tokenizer,
        max_tokens=numeric_recipe["max_tokens"], max_new_tokens=numeric_recipe["max_new_tokens"])
    commands = {tokenizer.decode(ids[:-1], skip_special_tokens=False): list(ids) for ids in grammar.completion_ids}
    result = {"version": VERSION, "record_id": record_id, "scope": scope,
        "raw_record_utf8": raw.decode("utf-8"), "record_sha256": record_sha256,
        "supervision_receipt_sha256": supervision_receipt_sha256, "tokenizer_sha256": tokenizer_sha256,
        "token_metadata": meta, "prompt_ids": list(grammar.prompt_ids), "commands": commands,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "supervised_command": views["language"]["supervised_completion"],
        "strict_pairs": [list(p) for p in views["language"]["preference_completions"]],
        "tied_pairs": views["alignment"]["tied_pairs"], "provenance_qualified": False}
    return json.loads(canonical(result))


def prepare_training_plan(*, records, record_pins, schedule, numeric_recipe, scope,
        execution_kind, expected_adapter_shapes, max_trainable_parameters, checkpoint_steps,
        outer_provenance_receipt_sha256):
    """CPU preflight: fixed sequence, complete exposure counts and data binding.

    One step is one supervised record or one strict chosen/rejected pair. All
    SFT precedes all DPO. Repetitions/zero exposures are explicit, not selected
    from losses. Ties remain in records and are not strict DPO examples.
    """
    _execution_recipe(numeric_recipe, execution_kind)
    if (scope not in ("artificial", "policy_train") or execution_kind not in ("qwen_gpu09", "qwen_screen_singletons", "artificial_control")
            or execution_kind == "artificial_control" and scope != "artificial"
            or not pin(outer_provenance_receipt_sha256)
            or type(records) is not tuple or not 1 <= len(records) <= MAX_RECORDS
            or type(record_pins) is not dict or type(schedule) is not tuple or not 2 <= len(schedule) <= MAX_STEPS):
        raise ValueError("bounded explicit training-only inputs and execution declaration required")
    if (type(expected_adapter_shapes) is not dict or not 1 <= len(expected_adapter_shapes) <= 112
            or not integer(max_trainable_parameters, 1) or max_trainable_parameters > MAX_PARAMETERS):
        raise ValueError("explicit bounded adapter parameter inventory required")
    count = 0
    for name, shape in expected_adapter_shapes.items():
        if (not _name(name) or not name.endswith((".A", ".B")) or type(shape) is not list
                or len(shape) != 2 or any(not integer(x, 1) for x in shape)):
            raise ValueError("exact named adapter matrix shapes required")
        count += math.prod(shape)
    if count > max_trainable_parameters:
        raise ValueError("declared adapter inventory exceeds parameter bound")
    ids, exposures, token_count, tokenizer_pins = [], {}, 0, set()
    for rec in records:
        if type(rec) is not dict or not _name(rec.get("record_id")):
            raise ValueError("prepared record required")
        rid = rec["record_id"]
        if rid in ids or rec.get("version") != VERSION or rec.get("scope") != scope or digest(rec) != record_pins.get(rid):
            raise ValueError("duplicate, foreign or unpinned prepared record")
        ids.append(rid)
        views = matched_views(rec["raw_record_utf8"].encode(), record_sha256=rec["record_sha256"],
            supervision_receipt_sha256=rec["supervision_receipt_sha256"], allowed_scope=scope)
        meta = rec["token_metadata"]
        if (meta["envelope_sha256"] != views["alignment"]["envelope_sha256"]
                or meta["task_sha256"] != views["alignment"]["task_sha256"]
                or meta["prompt_sha256"] != hashlib.sha256(views["language"]["prompt"].encode()).hexdigest()
                or rec["supervised_command"] != views["language"]["supervised_completion"]
                or canonical(rec["strict_pairs"]) != canonical(views["language"]["preference_completions"])
                or rec["tied_pairs"] != views["alignment"]["tied_pairs"] or not pin(rec["tokenizer_sha256"])
                or meta["max_tokens"] != numeric_recipe["max_tokens"]
                or meta["max_new_tokens"] != numeric_recipe["max_new_tokens"]):
            raise ValueError("tokenized data/state/label association differs")
        prompt, commands = rec["prompt_ids"], rec["commands"]
        if (type(prompt) is not list or not prompt or any(not integer(x) for x in prompt)
                or meta["prompt_tokens"] != len(prompt) or len(prompt)+32 > numeric_recipe["max_tokens"]
                or type(commands) is not dict or not 1 <= len(commands) <= 33
                or not integer(rec["pad_token_id"]) or not integer(rec["eos_token_id"])):
            raise ValueError("bounded complete prompt and command tokens required")
        for command, tokens in commands.items():
            if (type(command) is not str or type(tokens) is not list or not 1 <= len(tokens) <= 32
                    or any(not integer(x) for x in tokens) or tokens[-1] != rec["eos_token_id"]):
                raise ValueError("invalid complete action tokens")
        legal = {canonical({"stop": True} if a is None else {"action_id": a})
                 for a in views["alignment"]["action_ids"]}
        if (set(commands) != legal or meta["legal_commands"] != len(commands)
                or len({tuple(t) for t in commands.values()}) != len(commands)):
            raise ValueError("exact feasible command/token membership required")
        if (rec["supervised_command"] not in commands
                or any(len(p) != 2 or any(c not in commands for c in p) for p in rec["strict_pairs"])):
            raise ValueError("required SFT/DPO action tokens absent")
        token_count += len(prompt)+sum(map(len, commands.values()))
        tokenizer_pins.add(rec["tokenizer_sha256"])
        exposures[rid] = {"sft": 0, "dpo_pairs": [0]*len(rec["strict_pairs"]), "tied_pairs_retained": len(rec["tied_pairs"])}
    if set(record_pins) != set(ids) or len(tokenizer_pins) != 1 or token_count > MAX_STORED_TOKENS:
        raise ValueError("exact record/tokenizer membership or token storage bound differs")
    dpo_seen, sft_steps, dpo_steps = False, 0, 0
    for step in schedule:
        if (type(step) is not dict or set(step) != {"stage", "record_id", "pair_index"}
                or step["record_id"] not in exposures or step["stage"] not in ("sft", "dpo")):
            raise ValueError("exact fixed stage/record/pair sequence required")
        entry = exposures[step["record_id"]]
        if step["stage"] == "sft":
            if dpo_seen or step["pair_index"] is not None:
                raise ValueError("SFT must precede DPO and have no pair index")
            entry["sft"] += 1
            sft_steps += 1
        else:
            dpo_seen = True
            pair = step["pair_index"]
            if not integer(pair) or pair >= len(entry["dpo_pairs"]):
                raise ValueError("DPO references an absent strict preference pair")
            entry["dpo_pairs"][pair] += 1
            dpo_steps += 1
    if not sft_steps or not dpo_steps:
        raise ValueError("both predeclared SFT and DPO stages required")
    if (type(checkpoint_steps) is not tuple or any(not integer(i) or i >= len(schedule) for i in checkpoint_steps)
            or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps):
        raise ValueError("fixed sorted unique checkpoint indices required")
    return json.loads(canonical({"version": VERSION, "scope": scope, "execution_kind": execution_kind,
        "numeric_recipe": numeric_recipe, "records": records, "record_pins": record_pins, "schedule": schedule,
        "expected_adapter_shapes": expected_adapter_shapes, "max_trainable_parameters": max_trainable_parameters,
        "declared_adapter_parameters": count, "checkpoint_steps": checkpoint_steps,
        "outer_provenance_receipt_sha256": outer_provenance_receipt_sha256,
        "exposures": exposures, "sft_steps": sft_steps, "dpo_steps": dpo_steps,
        "sft_rows": sft_steps, "dpo_rows": 2*dpo_steps, "stored_tokens": token_count,
        "one_pass_sft": all(x["sft"] == 1 for x in exposures.values()),
        "one_pass_dpo": all(n == 1 for x in exposures.values() for n in x["dpo_pairs"]),
        "provenance_qualified": False, "retry_allowed": False}))


def _signature(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        h.update(canonical([name, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _inventory(model, plan):
    params = dict(model.named_parameters())
    adapters = {name+suffix: p for name, m in model.named_modules() if isinstance(m, LoRALinear)
                for suffix, p in ((".A", m.A), (".B", m.B))}
    shapes = plan["expected_adapter_shapes"]
    if (set(adapters) != set(shapes) or any(list(p.shape) != shapes[n] or p.dtype != torch.float32 for n, p in adapters.items())
            or sum(p.numel() for p in adapters.values()) > plan["max_trainable_parameters"]
            or any(p.requires_grad for n, p in params.items() if n not in adapters)
            or any(p.grad is not None for p in params.values())
            or any(not torch.isfinite(p).all() for p in adapters.values())
            or any(m.reference or m.A.shape[0] != 8 or m.scale != 2. for m in model.modules() if isinstance(m, LoRALinear))
            or type(getattr(model, "lm_head", None)) is not FixedRowFP32Projection):
        raise ValueError("model differs from exact frozen-base/adapter/projection inventory")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or not 1 <= len(layers) <= 28:
        raise ValueError("bounded explicit decoder layer list required")
    tensors = [*params.values(), *model.buffers()]
    if not tensors or len({t.device for t in tensors}) != 1 or any(b.requires_grad for b in model.buffers()):
        raise ValueError("single-device supplied model required")
    device = tensors[0].device
    if plan["execution_kind"] == "artificial_control":
        if device.type != "cpu":
            raise ValueError("artificial control path is CPU-only")
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        expected_names = {f"model.layers.{i}.self_attn.{part}.{suffix}" for i in range(28)
                          for part in ("q_proj", "v_proj") for suffix in ("A", "B")}
        if (type(model) is not Qwen3ForCausalLM or device.type != "cuda" or torch.cuda.device_count() != 1
                or model.config._attn_implementation != "sdpa" or len(layers) != 28 or set(adapters) != expected_names
                or model.config.max_position_embeddings < plan["numeric_recipe"]["max_tokens"]
                or sum(p.numel() for p in adapters.values()) != MAX_PARAMETERS
                or any(p.dtype != (torch.float32 if n in adapters else torch.bfloat16) for n, p in params.items())
                or any(b.dtype != torch.float32 for n, b in model.named_buffers() if "reference_" in n)
                or not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
                or torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
                or torch.initial_seed() != 109 or torch.cuda.initial_seed() != 109):
            raise ValueError("supplied Qwen/arithmetic state differs from GPU09 recipe")
    return params, adapters, len(layers), device


def _fit_plan_checks(plan):
    """Cheap structural admission only; no tokenization/numerical feature rebuild."""
    _execution_recipe(plan['numeric_recipe'], plan['execution_kind'])
    records, schedule, shapes = plan["records"], plan["schedule"], plan["expected_adapter_shapes"]
    if (type(records) is not list or not 1 <= len(records) <= MAX_RECORDS
            or type(schedule) is not list or not 2 <= len(schedule) <= MAX_STEPS
            or not integer(plan["max_trainable_parameters"], 1) or plan["max_trainable_parameters"] > MAX_PARAMETERS
            or type(shapes) is not dict or not 1 <= len(shapes) <= 112):
        raise ValueError("prepared plan exceeds fixed bounds")
    exposure = {}
    for rec in records:
        rid = rec["record_id"]
        if (rid in exposure or rec["scope"] != plan["scope"] or digest(rec) != plan["record_pins"].get(rid)
                or hashlib.sha256(rec["raw_record_utf8"].encode()).hexdigest() != rec["record_sha256"]
                or digest(json.loads(rec["raw_record_utf8"])["envelope"]) != rec["token_metadata"]["envelope_sha256"]):
            raise ValueError("prepared data/state binding changed")
        exposure[rid] = {"sft": 0, "dpo_pairs": [0]*len(rec["strict_pairs"]), "tied_pairs_retained": len(rec["tied_pairs"])}
    dpo_seen = False
    for step in schedule:
        if set(step) != {"stage", "record_id", "pair_index"} or step["record_id"] not in exposure:
            raise ValueError("invalid fixed step membership")
        e = exposure[step["record_id"]]
        if step["stage"] == "sft" and step["pair_index"] is None and not dpo_seen:
            e["sft"] += 1
        elif step["stage"] == "dpo" and integer(step["pair_index"]) and step["pair_index"] < len(e["dpo_pairs"]):
            dpo_seen = True
            e["dpo_pairs"][step["pair_index"]] += 1
        else:
            raise ValueError("altered SFT/DPO ordering or absent pair")
    sft = sum(e["sft"] for e in exposure.values())
    if (not 0 < sft < len(schedule) or sft != plan["sft_steps"] or len(schedule)-sft != plan["dpo_steps"]
            or exposure != plan["exposures"] or set(exposure) != set(plan["record_pins"])
            or plan["one_pass_sft"] != all(e["sft"] == 1 for e in exposure.values())
            or plan["one_pass_dpo"] != all(n == 1 for e in exposure.values() for n in e["dpo_pairs"])):
        raise ValueError("prepared exposure counts differ from actual schedule")
    indices = plan["checkpoint_steps"]
    if (type(indices) is not list or any(not integer(i) or i >= len(schedule) for i in indices)
            or sorted(set(indices)) != indices):
        raise ValueError("invalid checkpoint cadence")


def fit(*, model, plan, plan_sha256, output):
    """Fit exactly the pinned CPU-prepared plan; never initialize or reload a model.

    fit verifies the prepared plan byte content by canonical digest; full record
    validation/tokenization belongs to prepare_training_plan on CPU. Independent
    expected pin and source/runtime/model admission belong to the caller. Even
    qwen_gpu09 mode is not a scientific provenance certificate.
    """
    if not pin(plan_sha256) or digest(plan) != plan_sha256 or plan.get("version") != VERSION:
        raise ValueError("independently pinned prepared training plan required")
    # Detached plan prevents caller mutation during the run. This is not an
    # adversarial same-process sandbox or an authenticator for self-derived pins.
    plan = json.loads(canonical(plan))
    _recipe(plan["numeric_recipe"])
    _fit_plan_checks(plan)
    if (plan["execution_kind"] not in ("artificial_control", "qwen_gpu09", "qwen_screen_singletons")
            or plan["scope"] not in ("artificial", "policy_train")
            or plan["execution_kind"] == "artificial_control" and plan["scope"] != "artificial"):
        raise ValueError("invalid admitted execution scope")
    params, adapters, layers, device = _inventory(model, plan)
    base = [(n, p) for n, p in params.items() if n not in adapters]
    def reference_items():
        return [(n, b) for n, b in model.named_buffers() if "reference_" in n]
    def base_signature():
        return _signature(base + [(n, b) for n, b in model.named_buffers() if "reference_" not in n])
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    optimizer = control = None
    events, attempt, reference_signature, reference_scores = [], None, None, {}
    initial_base = initial_projection = None
    records = {r["record_id"]: r for r in plan["records"]}
    singletons = plan['numeric_recipe']['scoring_path'] == 'canonical_selected_unpadded'
    score = sequence_log_probs_unpadded if singletons else sequence_log_probs_canonical
    supervised = supervised_loss_unpadded if singletons else supervised_loss_canonical

    def batch(step):
        rec = records[step["record_id"]]
        commands = ([rec["supervised_command"]] if step["stage"] == "sft" else rec["strict_pairs"][step["pair_index"]])
        return pack_actions(rec["prompt_ids"], [rec["commands"][c] for c in commands],
                            rec["pad_token_id"], plan["numeric_recipe"]["max_tokens"])

    def pair_key(step):
        return canonical([step["record_id"], step["pair_index"]])

    def checkpoint(stem, *, gradients=False):
        paths = {}
        values = {n: p.detach().cpu().clone() for n, p in adapters.items()}
        _tensors(output/(stem+".adapter.pt"), values)
        paths["adapter"] = _file_pin(output/(stem+".adapter.pt"))
        if optimizer is not None:
            # optimizer.state_dict retains the actual Adam step/moments; no reset
            # between SFT and DPO. Serialized tensors can be loaded on CPU.
            _tensors(output/(stem+".optimizer.pt"), optimizer.state_dict())
            paths["optimizer"] = _file_pin(output/(stem+".optimizer.pt"))
        if gradients:
            _tensors(output/(stem+".gradients.pt"), {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in params.items()})
            paths["gradients"] = _file_pin(output/(stem+".gradients.pt"))
        _save(output/(stem+".checkpoint.json"), {"files": paths, "adapter_signature": _signature(adapters.items()),
            "completed_steps": len(events), "attempt": attempt, "provenance_qualified": False})

    try:
        _save(output/"plan.json", plan)
        initial_base = base_signature()
        initial_projection = _signature([("projection", model.lm_head.weight)])
        _save(output/"initial-signatures.json", {"base": initial_base, "projection": initial_projection,
            "adapter": _signature(adapters.items()), "adapter_parameters": sum(p.numel() for p in adapters.values()),
            "scope": "pristine supplied state before engine updates; base provenance belongs to caller"})
        checkpoint("initial")
        model.eval()
        for p in adapters.values():
            p.requires_grad_(True)
        control = DecoderCheckpointing(model, expected_layers=layers)
        control.set_enabled(True)
        r = plan["numeric_recipe"]
        optimizer = torch.optim.AdamW(list(adapters.values()), lr=r["learning_rate"], betas=tuple(r["adam_betas"]),
            eps=r["adam_eps"], weight_decay=r["weight_decay"], amsgrad=r["amsgrad"], foreach=r["foreach"],
            fused=r["fused"], maximize=r["maximize"], capturable=r["capturable"], differentiable=r["differentiable"])
        if device.type == "cuda":
            from torch.nn.attention import sdpa_kernel, SDPBackend
            arithmetic = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        else:
            arithmetic = nullcontext()
        with arithmetic:
            for index, step in enumerate(plan["schedule"]):
                if step["stage"] == "dpo" and reference_signature is None:
                    checkpoint("sft")
                    snapshot_reference(model)
                    reference_signature = _signature(reference_items())
                    _tensors(output/"reference.pt", {n: b.detach().cpu().clone() for n, b in reference_items()})
                    # Persist reference tensors BEFORE scoring any DPO pair.
                    _save(output/"reference-snapshot.json", {"signature": reference_signature,
                        "reference_sha256": _file_pin(output/"reference.pt"), "sft_adapter_sha256": _file_pin(output/"sft.adapter.pt"),
                        "completed_sft_steps": len(events), "policy_optimizer_updates_after_snapshot": 0})
                    with reference_policy(model):
                        for scheduled in plan["schedule"][index:]:
                            key = pair_key(scheduled)
                            if key not in reference_scores:
                                reference_scores[key] = score(model, batch(scheduled), selected=True).detach().cpu().clone()
                    reference_report = {"signature": reference_signature,
                        "scores": {k: v.tolist() for k, v in reference_scores.items()}}
                    if singletons:
                        reference_report.update(unique_pair_score_calls=len(reference_scores),
                            decoder_forwards=2*len(reference_scores), scoring_layout='ordered_unpadded_singletons')
                    else:
                        reference_report['unique_pair_forwards'] = len(reference_scores)
                    _save(output/"reference-scores.json", reference_report)
                optimizer.zero_grad(set_to_none=True)
                attempt = {"index": index, **step, "optimizer_step_started": False, "optimizer_step_returned": False}
                _save(output/f"step-{index:05d}.start.json", {**attempt, "record_pin": plan["record_pins"][step["record_id"]]})
                before, recompute = time.monotonic(), control.recompute_calls
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                action_batch = batch(step)
                loss = (supervised(model, action_batch, selected=True) if step["stage"] == "sft" else
                    preference_loss(score(model, action_batch, selected=True).reshape(1, 2),
                                    reference_scores[pair_key(step)].reshape(1, 2), beta=r["beta"]))
                value = float(loss.detach().cpu())
                _save(output/f"step-{index:05d}.loss.json", {"finite": math.isfinite(value),
                    "loss": value if math.isfinite(value) else None, "loss_repr": repr(value)})
                if not math.isfinite(value):
                    raise ValueError("nonfinite scheduled loss")
                loss.backward()
                details = [{"name": n, "shape": list(p.shape), "present": p.grad is not None,
                    "finite": p.grad is not None and bool(torch.isfinite(p.grad).all())} for n, p in adapters.items()]
                _save(output/f"step-{index:05d}.gradients.json", {"tensors": details,
                    "checkpoint_recompute_delta": control.recompute_calls-recompute})
                if (any(not d["present"] or not d["finite"] for d in details)
                        or any(p.grad is not None for _, p in base)
                        or control.recompute_calls-recompute < layers*(action_batch.input_ids.shape[0] if singletons else 1)):
                    raise ValueError("missing/nonfinite adapter gradient, frozen-base gradient or missing recomputation")
                norm = float(torch.nn.utils.clip_grad_norm_(list(adapters.values()), r["clip_gradient_norm"], error_if_nonfinite=True).cpu())
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("nonfinite/zero gradient norm")
                clipped = math.sqrt(sum(float((p.grad.detach().double()**2).sum().cpu()) for p in adapters.values()))
                _save(output/f"step-{index:05d}.update-intent.json", {"gradient_norm_before_clip": norm,
                    "gradient_norm_after_clip": clipped, "clip_limit": r["clip_gradient_norm"]})
                if reference_signature is not None and _signature(reference_items()) != reference_signature:
                    raise ValueError("fixed post-SFT reference changed before update")
                attempt["optimizer_step_started"] = True
                _save(output/f"step-{index:05d}.optimizer-attempt.json", attempt)
                optimizer.step()
                attempt["optimizer_step_returned"] = True
                _save(output/f"step-{index:05d}.update-returned.json", attempt)
                if (any(not torch.isfinite(p).all() for p in adapters.values())
                        or reference_signature is not None and _signature(reference_items()) != reference_signature):
                    raise ValueError("nonfinite adapter or fixed reference changed after update")
                event = {**attempt, "loss": value, "gradient_norm_before_clip": norm, "gradient_norm_after_clip": clipped,
                    "checkpoint_recompute_delta": control.recompute_calls-recompute,
                    "rows": action_batch.input_ids.shape[0], "input_tokens": int(action_batch.attention_mask.sum()),
                    "action_tokens": int((action_batch.labels != -100).sum())}
                if device.type == "cuda":
                    end.record()
                    torch.cuda.synchronize()
                    event.update(cuda_elapsed_ms=begin.elapsed_time(end), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
                event["wall_seconds"] = time.monotonic()-before
                _save(output/f"step-{index:05d}.complete.json", event)
                events.append(event)
                if index in plan["checkpoint_steps"]:
                    checkpoint(f"step-{index:05d}")
            verified = set()
            with reference_policy(model):
                for step in plan["schedule"][plan["sft_steps"]:]:
                    key = pair_key(step)
                    if key not in verified:
                        observed = score(model, batch(step), selected=True).detach().cpu()
                        if not torch.equal(observed, reference_scores[key]):
                            raise ValueError("fixed reference score changed")
                        verified.add(key)
        after_base = base_signature()
        after_projection = _signature([("projection", model.lm_head.weight)])
        if (after_base != initial_base or after_projection != initial_projection
                or _signature(reference_items()) != reference_signature):
            raise ValueError("pristine base, projection or fixed reference changed")
        checkpoint("final")
        result = {"status": "complete", "version": VERSION, "plan_sha256": plan_sha256,
            "completed_steps": len(events), "sft_steps": plan["sft_steps"], "dpo_steps": plan["dpo_steps"],
            "exposures": plan["exposures"], "one_pass_sft": plan["one_pass_sft"], "one_pass_dpo": plan["one_pass_dpo"],
            "base_before": initial_base, "base_after": after_base, "projection_before": initial_projection,
            "projection_after": after_projection, "reference_signature": reference_signature,
            "reference_scores_unchanged": True, "reference_pair_count": len(reference_scores),
            "checkpoint_usage": control.evidence(), "elapsed_seconds": time.monotonic()-started,
            "execution_kind": plan["execution_kind"], "outer_provenance_receipt_sha256": plan["outer_provenance_receipt_sha256"],
            "provenance_qualified": False, "retry_allowed": False, "environment_responses": 0}
        _save(output/"terminal.json", result)
        return result
    except BaseException as exc:
        failure = {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "plan_sha256": plan_sha256, "completed_steps": events,
            "failed_attempt": attempt, "retry_allowed": False, "provenance_qualified": False,
            "update_outcome": "unknown_or_partial" if attempt and attempt["optimizer_step_started"] and not attempt["optimizer_step_returned"] else "see_retained_step_events"}
        try:
            checkpoint("failed-partial", gradients=True)
            _tensors(output/"failed-partial.reference.pt", {n: b.detach().cpu().clone() for n, b in reference_items()})
            failure["partial_checkpoint_saved"] = True
        except BaseException as saved_error:
            failure["checkpoint_error"] = type(saved_error).__name__+": "+str(saved_error)
        _save(output/"failure.json", failure)
        raise
    finally:
        if control is not None:
            control.set_enabled(False)
