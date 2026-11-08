#!/usr/bin/env python3
"""Checkpointed shared-task qualification using saved artificial states only.

The pretrained base and an accepted artificial adapter are restored before
training. No scientific generator or environment callback is present. This
measures one fixed BF16/SDPA recipe, not scientific utility or universal bounds.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

SOURCES = {"qualify_llm_intervention_checkpoint_gpu.py", "supervise_llm_intervention_checkpoint_gpu.py",
           "llm_intervention_contract.py", "llm_intervention_learning.py", "llm_intervention_grammar.py",
           "llm_intervention_scm.py", "llm_intervention_episode.py", "llm_intervention_numerical_policy.py",
           "llm_intervention_target_risk.py", "llm_intervention_task.py",
           "llm_intervention_language_policy.py", "llm_intervention_action_scoring.py", "llm_intervention_precision.py", "llm_intervention_gradient_probe.py", "llm_intervention_canonical_scoring.py", "llm_intervention_checkpointing.py"}
MODELS = {"model.safetensors", "tokenizer_config.json", "config.json", "tokenizer.json",
          "generation_config.json", "merges.txt", "vocab.json"}
PACKAGES = {"torch", "transformers", "numpy", "scipy", "safetensors", "huggingface-hub", "tokenizers", "jinja2"}
ASSETS = {"case0.json", "case2.json", "case3.json", "stress.json", "profile.json", "prepared_tokens.json",
          "accepted_adapter.pt", "gpu04_terminal.json", "gpu04_acceptance.json", "cpu01_acceptance.json",
          "gpu06_terminal.json", "gpu06_acceptance.json", "gpu07_failure_disposition.json", "gpu08_failure_disposition.json"}
RECIPE = {"seed": 109, "rank": 8, "alpha": 16., "sft_steps": 3, "dpo_steps": 3,
          "learning_rate": .0001, "beta": .1, "max_tokens": 8192, "max_new_tokens": 32,
          "attention": "sdpa", "sdpa_backend": "flash_attention_only", "base_dtype": "bfloat16", "adapter_dtype": "float32",
          "activation_checkpointing": "explicit_nonreentrant_decoder_v1", "strict_determinism": True, "float32_matmul": "highest",
          "output_projection": "fixed_row_fp32_action_reduction_v1", "gradient_atol": .0001, "gradient_rtol": .01,
          "loss_atol": .0001, "loss_rtol": .00001}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def validate_freeze(freeze):
    if (freeze["schema"] != "ace-checkpoint-task-gpu-v1" or set(freeze["sources"]) != SOURCES
            or set(freeze["model_files"]) != MODELS or set(freeze["runtime"]) != PACKAGES
            or set(freeze["assets"]) != ASSETS or freeze["recipe"] != RECIPE
            or freeze["startup_seconds"] != 180 or freeze["worker_seconds"] != 540):
        raise ValueError("altered qualification design or input membership")
    for section in ("sources", "model_files", "assets"):
        for name, pin in freeze[section].items():
            if (Path(name).name != name or not isinstance(pin, str) or len(pin) != 64
                    or any(x not in "0123456789abcdef" for x in pin)):
                raise ValueError("invalid input pin")


def validate_terminal(terminal, freeze, freeze_sha256, steps):
    expected_steps = {(stage, i) for stage in ("sft", "dpo") for i in range(3)}
    if (len(steps) != 6 or {(p["stage"], p["step"]) for p in steps} != expected_steps
            or any(not math.isfinite(p["cuda_elapsed_ms"]) or p["cuda_elapsed_ms"] <= 0
                   or p["checkpoint_recompute_delta"] < 28 for p in steps)
            or terminal["status"] != "complete" or terminal["freeze_sha256"] != freeze_sha256
            or terminal["source_pins"] != freeze["sources"] or terminal["sft_steps"] != 3
            or terminal["dpo_steps"] != 3 or terminal["scientific_responses"] != 0
            or terminal["environment_responses"] != 0 or terminal["generation_count"] != 4
            or terminal["cuda_step_ms_total"] <= 0 or terminal["peak_gpu_allocated_bytes"] <= 0
            or terminal["frozen_base_before"] != terminal["frozen_base_after"]
            or terminal["reference_before"] != terminal["reference_after"]
            or terminal["adapter_before"] == terminal["adapter_after"]
            or terminal["checkpoint_gradient_equivalence"] is not True
            or terminal["accepted_adapter_restored"] is not True
            or terminal["trained_adapter_reload_score_max_difference"] != 0
            or terminal["activation_checkpointing_used"] is not True
            or terminal["checkpoint_recompute_calls"] < 28
            or terminal["same_path_repeatability"] is not True or terminal["comparison_count"] != 8
            or terminal["strict_determinism"] is not True or terminal["fp32_projection_unchanged"] is not True):
        raise ValueError("incomplete shared-task model, gradient, reload or resource evidence")


def compare_and_record(out, objective, observed):
    """Persist actual discrepancies before rejecting a failed tolerance check."""
    import torch
    baseline_loss, full_gradients = observed[0]
    checkpointed_loss, selected_gradients = observed[1]
    details = []
    for name in full_gradients:
        a, b = full_gradients[name], selected_gradients[name]
        details.append({"name": name, "shape": list(a.shape),
            "max_absolute_difference": float((a-b).abs().max()),
            "baseline_max_absolute_gradient": float(a.abs().max()),
            "checkpointed_max_absolute_gradient": float(b.abs().max()),
            "within_tolerance": bool(torch.allclose(a, b, atol=RECIPE["gradient_atol"], rtol=RECIPE["gradient_rtol"]))})
    record = {"objective": objective, "baseline_loss": baseline_loss, "checkpointed_loss": checkpointed_loss,
        "loss_absolute_difference": abs(baseline_loss-checkpointed_loss),
        "loss_within_tolerance": math.isclose(baseline_loss, checkpointed_loss, abs_tol=RECIPE["loss_atol"], rel_tol=RECIPE["loss_rtol"]),
        "max_gradient_absolute_difference": max(d["max_absolute_difference"] for d in details),
        "gradient_tensors": len(details), "tensor_details": details,
        "tolerances": {k: RECIPE[k] for k in ("loss_atol", "loss_rtol", "gradient_atol", "gradient_rtol")}}
    save(out/f"comparison-{objective}.json", record)
    return record


def finish_comparison_gate(out, comparisons, repeat_failures):
    expected = {f"{stage}-{left}__{right}" for stage in ("sft", "dpo")
        for left, right in (("off-0", "off-1"), ("on-0", "on-1"),
                            ("off-0", "on-0"), ("off-1", "on-1"))}
    if (len(comparisons) != 8 or {x["objective"] for x in comparisons} != expected
            or any(x["gradient_tensors"] != 112 for x in comparisons)):
        raise ValueError("incomplete actual-model comparison inventory")
    failures = [x["objective"] for x in comparisons if not x["loss_within_tolerance"]
                or any(not d["within_tolerance"] for d in x["tensor_details"])]
    save(out/"comparison_gate.json", {"comparisons": 8, "tolerance_failures": failures,
        "exact_repeat_failures": repeat_failures, "optimizer_updates": 0,
        "passed": not failures and not repeat_failures})
    if failures or repeat_failures:
        raise ValueError("numerical qualification failed after all comparison evidence was retained")


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    phase_number = 0
    sdpa_context = None
    gpu_ready = False
    memory_peaks = []
    def phase(name, **extra):
        nonlocal phase_number
        if gpu_ready:
            memory_peaks.append((torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
        save(out/f"phase-{phase_number:03d}.json", {"phase": name,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic_seconds": time.monotonic(), "elapsed_seconds": time.monotonic()-started,
            "process_cpu_seconds": time.process_time()-cpu, **extra})
        phase_number += 1
    def deadline(signum, frame):
        raise TimeoutError("bounded shared-task qualification deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    stack_log = (out/"runtime_stacks.log").open("x")
    faulthandler.dump_traceback_later(60, repeat=True, file=stack_log, exit=False)
    try:
        phase("input_verification_start")
        raw = Path(args.freeze).read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.freeze_sha256:
            raise ValueError("freeze digest differs")
        freeze = json.loads(raw)
        validate_freeze(freeze)
        source, model_dir = Path(args.freeze).resolve().parent, Path(freeze["model_path"])
        if (Path(__file__).resolve() != source/"qualify_llm_intervention_checkpoint_gpu.py"
                or list(sys.version_info[:3]) != freeze["python_version"]
                or {p.name for p in model_dir.iterdir()} != MODELS):
            raise ValueError("worker/Python/model membership mismatch")
        for names, base in ((freeze["sources"], source), (freeze["model_files"], model_dir), (freeze["assets"], source)):
            for name, pin in names.items():
                if sha(base/name) != pin:
                    raise ValueError("source/model/asset bytes differ: " + name)
        runtime = {name: importlib.metadata.version(name) for name in PACKAGES}
        if runtime != freeze["runtime"]:
            raise ValueError("runtime mismatch")
        prior = json.loads((source/"gpu04_terminal.json").read_bytes())
        if (prior["status"] != "complete" or prior["adapter_sha256"] != freeze["assets"]["accepted_adapter.pt"]):
            raise ValueError("accepted artificial adapter lineage differs")
        diag = json.loads((source/"gpu06_terminal.json").read_bytes())
        accepted = json.loads((source/"gpu06_acceptance.json").read_bytes())
        if (diag["status"] != "diagnostic_complete" or accepted["status"] != "accepted_diagnostic"
                or accepted["worker_terminal_sha256"] != freeze["assets"]["gpu06_terminal.json"]):
            raise ValueError("preceding diagnostic acceptance differs")
        failed = json.loads((source/"gpu07_failure_disposition.json").read_bytes())
        if (failed["status"] != "accepted_failed_attempt" or failed["numerical_qualification"] is not False
                or failed["job_id"] != "35209701" or failed["optimizer_updates"] != 0):
            raise ValueError("preceding failed precision qualification differs")
        failed08 = json.loads((source/"gpu08_failure_disposition.json").read_bytes())
        if (failed08["status"] != "accepted_failed_attempt" or failed08["job_id"] != "35209904"
                or failed08["canonical_numerical_gate_passed"] is not True or failed08["optimizer_updates"] != 1):
            raise ValueError("preceding canonical attempt differs")
        phase("input_verification_complete", packages=runtime,
              runtime_scope="declared package versions, not whole-runtime byte closure")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        sys.path.insert(0, str(source))
        phase("imports_start")
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from llm_intervention_learning import preference_loss, snapshot_reference, reference_policy
        from llm_intervention_language_policy import (restore_adapter, prepare_task_tokens,
                                                      training_action_batch, generate_task_decision)
        from llm_intervention_canonical_scoring import (install_canonical_projection,
            sequence_log_probs_canonical, supervised_loss_canonical)
        def sequence_log_probs(m, b): return sequence_log_probs_canonical(m, b, selected=False)
        def sequence_log_probs_selected(m, b): return sequence_log_probs_canonical(m, b, selected=True)
        def supervised_loss(m, b): return supervised_loss_canonical(m, b, selected=False)
        def supervised_loss_selected(m, b): return supervised_loss_canonical(m, b, selected=True)
        from llm_intervention_checkpointing import DecoderCheckpointing
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        save(out/"precision_mode.json", {"strict": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "recipe": RECIPE})
        if (not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
                or torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32):
            raise ValueError("strict arithmetic recipe differs")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        sdpa_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        sdpa_context.__enter__()
        phase("attention_backend_frozen", flash=torch.backends.cuda.flash_sdp_enabled(),
              math=torch.backends.cuda.math_sdp_enabled(), efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
              cudnn=torch.backends.cuda.cudnn_sdp_enabled())
        phase("imports_complete")
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("one BF16-compatible visible GPU required")
        torch.manual_seed(RECIPE["seed"])
        torch.cuda.init()
        torch.cuda.manual_seed_all(RECIPE["seed"])
        gpu_ready = True
        phase("cuda_initialization_complete", device_name=torch.cuda.get_device_name(0),
              total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
              cuda_build=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        cases = {name: json.loads((source/name).read_bytes()) for name in ("case0.json", "case2.json", "case3.json", "stress.json")}
        profiles = json.loads((source/"profile.json").read_bytes())
        prepared = json.loads((source/"prepared_tokens.json").read_bytes())
        batches, metadata = {}, {}
        for name, envelope in cases.items():
            grammar, meta = prepare_task_tokens(envelope, tokenizer, max_tokens=8192, max_new_tokens=32)
            if meta != profiles["profiles"][name]:
                raise ValueError("target tokenizer differs from CPU-prepared fixture")
            commands = tuple(tokenizer.decode(ids[:-1], skip_special_tokens=False) for ids in grammar.completion_ids)
            action_commands = tuple(x for x in commands if 'action_id' in x)[:2]
            if len(action_commands) != 2:
                raise ValueError("two arbitrary feasible artificial intervention labels required")
            batches[name], _ = training_action_batch(envelope, tokenizer, action_commands,
                                                      max_tokens=8192, max_new_tokens=32)
            tokens = {"prompt_ids": list(grammar.prompt_ids),
                      "completion_ids": [list(x) for x in grammar.completion_ids],
                      "input_ids": batches[name].input_ids.tolist(),
                      "attention_mask": batches[name].attention_mask.tolist(),
                      "labels": batches[name].labels.tolist()}
            if tokens != prepared[name]:
                raise ValueError("actual prompt/grammar/action/mask tokens differ from CPU preparation")
            metadata[name] = {**meta, "arbitrary_fixture_labels": action_commands}
        save(out/"task_tokens.json", metadata)
        phase("tokenization_complete", prompt_tokens={k: v["prompt_tokens"] for k, v in metadata.items()})
        phase("model_cpu_load_start")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if type(model) is not Qwen3ForCausalLM or model.config._attn_implementation != "sdpa":
            raise ValueError("actual Qwen implementation/attention differs")
        names = tuple(f"model.layers.{i}.self_attn.{part}" for i in range(28) for part in ("q_proj", "v_proj"))
        restoration = restore_adapter(model, (source/"accepted_adapter.pt").read_bytes(),
            adapter_sha256=freeze["assets"]["accepted_adapter.pt"],
            training_receipt_sha256=freeze["assets"]["gpu04_terminal.json"], expected_projection_names=names)
        save(out/"accepted_adapter_restoration.json", restoration)
        phase("model_cpu_load_and_restore_complete")
        model.to("cuda")
        projection = install_canonical_projection(model)
        save(out/"projection.json", projection)
        torch.cuda.synchronize()
        if (any(p.dtype != (torch.float32 if n.endswith((".A", ".B")) else torch.bfloat16)
                for n, p in model.named_parameters())
                or any(b.dtype != torch.float32 for n, b in model.named_buffers() if "reference_" in n)):
            raise ValueError("base/adapter/reference dtype layout differs")
        if model.is_gradient_checkpointing:
            raise ValueError("unqualified activation checkpointing enabled")
        checkpoint_control = DecoderCheckpointing(model, expected_layers=28)
        phase("gpu_transfer_complete")
        def signature(items):
            h = hashlib.sha256()
            for name, tensor in items:
                h.update(name.encode())
                h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return h.hexdigest()
        def adapters():
            return [(n, p) for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
        def base_signature():
            return signature((n, p) for n, p in model.named_parameters() if not n.endswith((".A", ".B")))
        frozen_before, adapter_before = base_signature(), signature(adapters())
        projection_before = signature([("projection", model.lm_head.weight)])
        with torch.no_grad():
            initial = sequence_log_probs_selected(model, batches["case0.json"]).cpu()
        torch.cuda.synchronize()
        phase("first_gpu_forward_complete", scores=initial.tolist(), allocated_bytes=torch.cuda.memory_allocated())
        signal.alarm(max(1, int(540-(time.monotonic()-started))))
        for _, p in adapters():
            p.requires_grad_(True)
        trainable = [p for _, p in adapters()]
        if len(trainable) != 112 or sum(p.numel() for p in trainable) != 1146880:
            raise ValueError("unexpected trainable adapter inventory")
        # New comparison isolates recomputation. Both paths use the canonical
        # selected projection already qualified in GPU08, without redoing full logits.
        from llm_intervention_gradient_probe import capture_backward, compare_captures
        comparisons, repeat_failures = [], []
        for objective in ("sft", "dpo"):
            captures = {}
            def loss_for_capture(model, batch, selected):
                score = sequence_log_probs_selected
                sft = supervised_loss_selected
                return (sft(model, batch) if objective == "sft" else
                    preference_loss(score(model, batch).reshape(1, 2), initial.reshape(1, 2), beta=.1))
            for selected in (False, True):
                checkpoint_control.set_enabled(selected)
                path = "on" if selected else "off"
                for repeat in range(2):
                    torch.cuda.reset_peak_memory_stats()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    backward_start = time.monotonic()
                    capture_recompute_before = checkpoint_control.recompute_calls
                    begin.record()
                    capture = capture_backward(model, batches["case0.json"], selected=True,
                                               loss_function=loss_for_capture)
                    end.record()
                    torch.cuda.synchronize()
                    phase("comparison_backward", objective=objective, path=path, repeat=repeat,
                        cuda_elapsed_ms=begin.elapsed_time(end), elapsed_backward_seconds=time.monotonic()-backward_start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        scope="includes observational copies, not production throughput")
                    capture["checkpointing_enabled"] = selected
                    capture["checkpoint_recompute_delta"] = checkpoint_control.recompute_calls-capture_recompute_before
                    key = f"{path}-{repeat}"
                    torch.save(capture, out/f"capture-{objective}-{key}.pt")
                    captures[key] = capture
                    if capture["checkpoint_recompute_delta"] != (28 if selected else 0):
                        raise ValueError("comparison checkpoint execution count differs")
            for left, right in (("off-0", "off-1"), ("on-0", "on-1"),
                                ("off-0", "on-0"), ("off-1", "on-1")):
                label = f"{objective}-{left}__{right}"
                a, b = captures[left], captures[right]
                boundary = compare_captures(a, b, atol=RECIPE["gradient_atol"], rtol=RECIPE["gradient_rtol"])
                save(out/f"boundary-{label}.json", boundary)
                comparisons.append(compare_and_record(out, label,
                    [(a["loss"], a["gradients"]), (b["loss"], b["gradients"])]))
                if left.split('-')[0] == right.split('-')[0] and (
                        a["loss"] != b["loss"] or any(not torch.equal(a["gradients"][n], b["gradients"][n])
                                                       for n in a["gradients"])):
                    repeat_failures.append(label)
                phase("score_gradient_comparison", **{k: v for k, v in comparisons[-1].items() if k != "tensor_details"})
            del captures, capture, a, b, boundary
        comparison_preservation = {"base_before": frozen_before, "base_after": base_signature(),
            "adapter_before": adapter_before, "adapter_after": signature(adapters()),
            "projection_before": projection_before,
            "projection_after": signature([("projection", model.lm_head.weight)]), "optimizer_updates": 0}
        save(out/"comparison_immutability.json", comparison_preservation)
        if any(comparison_preservation[x+"_before"] != comparison_preservation[x+"_after"]
               for x in ("base", "adapter", "projection")):
            raise ValueError("comparison changed model parameters")
        finish_comparison_gate(out, comparisons, repeat_failures)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        save(out/"checkpoint_gradient_equivalence.json", {"comparisons": comparisons, "recipe": RECIPE,
            "scope": "one613-token artificial state, checkpoint off/on canonical selected arithmetic, not universal equivalence"})
        checkpoint_control.set_enabled(True)
        optimizer = torch.optim.AdamW(trainable, lr=RECIPE["learning_rate"], weight_decay=0.)
        events = []
        # One five-node example followed by the longer pure length stress twice.
        schedule = ("case3.json", "stress.json", "stress.json")
        for stage in ("sft", "dpo"):
            if stage == "dpo":
                snapshot_reference(model)
                reference_signature = signature((n, b) for n, b in model.named_buffers() if "reference_" in n)
                with reference_policy(model):
                    references = {name: sequence_log_probs_selected(model, batches[name]).detach().cpu() for name in set(schedule)}
                save(out/"frozen_reference.json", {"signature": reference_signature,
                    "scores": {n: t.tolist() for n, t in references.items()}})
                torch.save({n: b.detach().cpu() for n, b in model.named_buffers() if "reference_" in n}, out/"reference.pt")
                phase("dpo_reference_frozen", signature=reference_signature)
            for step, name in enumerate(schedule):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                recompute_before = checkpoint_control.recompute_calls
                phase("training_step_start", stage=stage, step=step, case=name,
                    prompt_tokens=metadata[name]["prompt_tokens"],
                    allocated_bytes=torch.cuda.memory_allocated())
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                loss = (supervised_loss_selected(model, batches[name]) if stage == "sft" else
                    preference_loss(sequence_log_probs_selected(model, batches[name]).reshape(1, 2),
                                    references[name].reshape(1, 2), beta=.1))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                if (not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0
                        or any(p.grad is not None for p in model.parameters() if not p.requires_grad)):
                    raise ValueError("nonfinite/zero gradient or frozen-base gradient")
                if checkpoint_control.recompute_calls - recompute_before < 28:
                    raise ValueError("training backward did not recompute all decoder layers")
                optimizer.step()
                end.record()
                torch.cuda.synchronize()
                event = {"stage": stage, "step": step, "case": name, "prompt_tokens": metadata[name]["prompt_tokens"],
                    "batch_rows": 2, "checkpoint_recompute_delta": checkpoint_control.recompute_calls-recompute_before,
                    "loss": float(loss.detach().cpu()), "gradient_norm": float(norm.cpu()),
                    "cuda_elapsed_ms": begin.elapsed_time(end), "wall_seconds": time.monotonic()-before,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                events.append(event)
                phase("training_step", **event)
                del loss
        model.zero_grad(set_to_none=True)
        del optimizer
        with reference_policy(model):
            reference_after = {name: sequence_log_probs_selected(model, batches[name]).cpu() for name in references}
        if any(not torch.equal(reference_after[n], references[n]) for n in references):
            raise ValueError("fixed reference scores changed")
        reference_after_signature = signature((n, b) for n, b in model.named_buffers() if "reference_" in n)
        frozen_after, adapter_after = base_signature(), signature(adapters())
        projection_after = signature([("projection", model.lm_head.weight)])
        if projection_after != projection_before:
            raise ValueError("frozen FP32 projection changed")
        if frozen_after != frozen_before or reference_after_signature != reference_signature or adapter_after == adapter_before:
            raise ValueError("base/reference changed or adapter unchanged")
        torch.save({n: p.detach().cpu() for n, p in adapters()}, out/"adapter.pt")
        model.requires_grad_(False)
        with torch.no_grad():
            before_reload = sequence_log_probs_selected(model, batches["case3.json"]).cpu()
        phase("trained_checkpoint_saved", adapter_sha256=sha(out/"adapter.pt"))
        checkpoint_evidence = checkpoint_control.evidence()
        save(out/"checkpoint_usage.json", checkpoint_evidence)
        if checkpoint_evidence["recompute_calls"] < 28:
            raise ValueError("checkpoint recomputation not observed")
        del checkpoint_control, trainable, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        phase("trained_reload_start")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        restore_adapter(model, (out/"adapter.pt").read_bytes(), adapter_sha256=sha(out/"adapter.pt"),
            training_receipt_sha256=sha(out/"frozen_reference.json"), expected_projection_names=names)
        model.to("cuda")
        install_canonical_projection(model)
        if signature([("projection", model.lm_head.weight)]) != projection_before:
            raise ValueError("reloaded FP32 projection differs")
        with torch.no_grad():
            after_reload = sequence_log_probs_selected(model, batches["case3.json"]).cpu()
        if not torch.equal(before_reload, after_reload) or base_signature() != frozen_before or signature(adapters()) != adapter_after:
            raise ValueError("full-size trained checkpoint reload differs")
        save(out/"reload.json", {"before": before_reload.tolist(), "after": after_reload.tolist(),
            "max_absolute_difference": float((before_reload-after_reload).abs().max()),
            "adapter_sha256": sha(out/"adapter.pt"), "base_signature": frozen_before})
        phase("trained_reload_complete")
        generations = []
        for index, name in enumerate(("case2.json", "case3.json", "stress.json", "case3.json")):
            before = time.monotonic()
            completion, evidence = generate_task_decision(model, tokenizer, cases[name], max_tokens=8192, max_new_tokens=32)
            torch.cuda.synchronize()
            evidence.update({"case": name, "elapsed_seconds": time.monotonic()-before})
            save(out/f"generation-{index:02d}.json", evidence)
            generations.append(completion)
            phase("generation_complete", index=index, case=name, completion=completion,
                  elapsed_inference_seconds=evidence["elapsed_seconds"])
        if generations[1] != generations[3]:
            raise ValueError("same-state deterministic generation differs after another state")
        terminal = {"status": "complete", "freeze_sha256": args.freeze_sha256,
            "scope": "full-size checkpointed artificial qualification, no scientific utility result",
            "source_pins": freeze["sources"], "sft_steps": 3, "dpo_steps": 3,
            "scientific_responses": 0, "environment_responses": 0, "generation_count": 4,
            "strict_determinism": torch.are_deterministic_algorithms_enabled(),
            "fp32_projection_unchanged": projection_before == projection_after,
            "projection_before": projection_before, "projection_after": projection_after,
            "same_path_repeatability": True, "comparison_count": len(comparisons),
            "checkpoint_gradient_equivalence": True, "accepted_adapter_restored": True,
            "activation_checkpointing_used": True, "checkpoint_recompute_calls": checkpoint_evidence["recompute_calls"], "trained_adapter_reload_score_max_difference": 0.,
            "adapter_sha256": sha(out/"adapter.pt"), "frozen_base_before": frozen_before, "frozen_base_after": frozen_after,
            "reference_checkpoint_sha256": sha(out/"reference.pt"),
            "reference_reload_scope": "policy reload snapshots final policy as its new reference, original DPO reference saved separately",
            "reference_before": reference_signature, "reference_after": reference_after_signature,
            "adapter_before": adapter_before, "adapter_after": adapter_after,
            "cuda_step_ms_total": sum(x["cuda_elapsed_ms"] for x in events),
            "peak_gpu_allocated_bytes": max([torch.cuda.max_memory_allocated()]+[x[0] for x in memory_peaks]),
            "peak_gpu_reserved_bytes": max([torch.cuda.max_memory_reserved()]+[x[1] for x in memory_peaks]),
            "runtime_scope": "eight package versions, not whole-runtime byte closure",
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        validate_terminal(terminal, freeze, args.freeze_sha256, events)
        save(out/"terminal.json", terminal)
    except BaseException as exc:
        save(out/"failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic()-started, "process_cpu_seconds": time.process_time()-cpu})
        raise
    finally:
        try:
            phase("worker_function_exit")
        finally:
            if sdpa_context is not None:
                sdpa_context.__exit__(None, None, None)
            faulthandler.cancel_dump_traceback_later()
            stack_log.close()
            signal.alarm(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
