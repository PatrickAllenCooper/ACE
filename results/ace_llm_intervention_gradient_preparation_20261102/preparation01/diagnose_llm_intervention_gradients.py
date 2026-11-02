#!/usr/bin/env python3
"""Discriminate repeat variation from action-projection gradient differences.

Eight artificial backwards with identical weights, no optimization or responses.
The default/strict control affects more than FlashAttention, so observations do
not uniquely identify a kernel-level cause. Every raw capture is retained.
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

SOURCES = {"diagnose_llm_intervention_gradients.py", "supervise_llm_intervention_gradients.py",
           "llm_intervention_contract.py", "llm_intervention_learning.py", "llm_intervention_grammar.py",
           "llm_intervention_scm.py", "llm_intervention_episode.py", "llm_intervention_numerical_policy.py",
           "llm_intervention_target_risk.py", "llm_intervention_task.py",
           "llm_intervention_language_policy.py", "llm_intervention_action_scoring.py", "llm_intervention_gradient_probe.py"}
MODELS = {"model.safetensors", "tokenizer_config.json", "config.json", "tokenizer.json",
          "generation_config.json", "merges.txt", "vocab.json"}
PACKAGES = {"torch", "transformers", "numpy", "scipy", "safetensors", "huggingface-hub", "tokenizers", "jinja2"}
ASSETS = {"case0.json", "profile.json", "prepared_tokens.json", "accepted_adapter.pt",
          "gpu04_terminal.json", "gpu04_acceptance.json", "cpu01_acceptance.json", "gpu05_failure_disposition.json"}
RECIPE = {"seed": 109, "rank": 8, "alpha": 16., "max_tokens": 8192, "max_new_tokens": 32,
          "attention": "sdpa", "sdpa_backend": "flash_attention_only", "base_dtype": "bfloat16",
          "adapter_dtype": "float32", "activation_checkpointing": False,
          "gradient_atol": .0001, "gradient_rtol": .01, "loss_atol": .0001, "loss_rtol": .00001,
          "cublas_workspace_config": ":4096:8", "modes": ["default", "strict"],
          "paths": ["full", "selected"], "repeats": 2, "optimizer_updates": 0}


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
    if (freeze["schema"] != "ace-gradient-diagnostic-v1" or set(freeze["sources"]) != SOURCES
            or set(freeze["model_files"]) != MODELS or set(freeze["runtime"]) != PACKAGES
            or set(freeze["assets"]) != ASSETS or freeze["recipe"] != RECIPE
            or freeze["startup_seconds"] != 180 or freeze["worker_seconds"] != 250):
        raise ValueError("altered qualification design or input membership")
    for section in ("sources", "model_files", "assets"):
        for name, pin in freeze[section].items():
            if (Path(name).name != name or not isinstance(pin, str) or len(pin) != 64
                    or any(x not in "0123456789abcdef" for x in pin)):
                raise ValueError("invalid input pin")


def validate_terminal(terminal, freeze, freeze_sha256, steps):
    expected = {(m, p, r) for m in ("default", "strict") for p in ("full", "selected") for r in range(2)}
    if (len(steps) != 8 or {(p["mode"], p["path"], p["repeat"]) for p in steps} != expected
            or any(not math.isfinite(p["cuda_elapsed_ms"]) or p["cuda_elapsed_ms"] <= 0 for p in steps)
            or terminal["status"] != "diagnostic_complete" or terminal["freeze_sha256"] != freeze_sha256
            or terminal["source_pins"] != freeze["sources"] or terminal["optimizer_updates"] != 0
            or terminal["scientific_responses"] != 0 or terminal["environment_responses"] != 0
            or terminal["backward_passes"] != 8 or terminal["cuda_step_ms_total"] <= 0
            or terminal["peak_gpu_allocated_bytes"] <= 0 or terminal["model_parameters_unchanged"] is not True
            or terminal["reference_buffers_unchanged"] is not True or terminal["long_context_qualified"] is not False):
        raise ValueError("incomplete diagnostic/runtime evidence")


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
        if (Path(__file__).resolve() != source/"diagnose_llm_intervention_gradients.py"
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
        phase("input_verification_complete", packages=runtime,
              runtime_scope="declared package versions, not whole-runtime byte closure")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          CUBLAS_WORKSPACE_CONFIG=RECIPE["cublas_workspace_config"])
        sys.path.insert(0, str(source))
        phase("imports_start")
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from llm_intervention_learning import (sequence_log_probs, supervised_loss, preference_loss,
                                               snapshot_reference, reference_policy)
        from llm_intervention_language_policy import (restore_adapter, prepare_task_tokens,
                                                      training_action_batch, generate_task_decision)
        from llm_intervention_action_scoring import sequence_log_probs_selected, supervised_loss_selected
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(False, warn_only=False)
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
        cases = {name: json.loads((source/name).read_bytes()) for name in ("case0.json",)}
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
        torch.cuda.synchronize()
        if (any(p.dtype != (torch.float32 if n.endswith((".A", ".B")) else torch.bfloat16)
                for n, p in model.named_parameters())
                or any(b.dtype != torch.float32 for n, b in model.named_buffers() if "reference_" in n)):
            raise ValueError("base/adapter/reference dtype layout differs")
        if model.is_gradient_checkpointing:
            raise ValueError("unqualified activation checkpointing enabled")
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
        with torch.no_grad():
            initial = sequence_log_probs_selected(model, batches["case0.json"]).cpu()
        torch.cuda.synchronize()
        phase("first_gpu_forward_complete", scores=initial.tolist(), allocated_bytes=torch.cuda.memory_allocated())
        signal.alarm(max(1, int(250-(time.monotonic()-started))))
        for _, p in adapters():
            p.requires_grad_(True)
        trainable = [p for _, p in adapters()]
        if len(trainable) != 112 or sum(p.numel() for p in trainable) != 1146880:
            raise ValueError("unexpected trainable adapter inventory")
        from llm_intervention_gradient_probe import capture_backward, compare_captures
        parameters_before = signature(model.named_parameters())
        references_before = signature(model.named_buffers())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        save(out/"rng_states.json", {"cpu_sha256": hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
            "cuda_sha256": [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in cuda_rng]})
        events, captures = [], {}
        for mode in RECIPE["modes"]:
            strict = mode == "strict"
            torch.use_deterministic_algorithms(strict, warn_only=False)
            mode_receipt = {"mode": mode, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "flash": torch.backends.cuda.flash_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled(),
                "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(), "cudnn": torch.backends.cuda.cudnn_sdp_enabled()}
            if (mode_receipt["deterministic_algorithms"] != strict or mode_receipt["warn_only"]
                    or not mode_receipt["flash"] or any(mode_receipt[k] for k in ("math", "efficient", "cudnn"))):
                raise ValueError("frozen mode or Flash-only backend differs")
            save(out/("mode-"+mode+".json"), mode_receipt)
            for path in RECIPE["paths"]:
                for repeat in range(2):
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    capture = capture_backward(model, batches["case0.json"], selected=path == "selected")
                    end.record()
                    torch.cuda.synchronize()
                    name = f"{mode}-{path}-{repeat}"
                    if len(capture["gradients"]) != 112:
                        raise ValueError("full adapter gradient set missing")
                    torch.save(capture, out/("capture-"+name+".pt"))
                    captures[name] = capture
                    event = {"mode": mode, "path": path, "repeat": repeat, "loss": capture["loss"],
                        "cuda_elapsed_ms": begin.elapsed_time(end), "wall_seconds": time.monotonic()-start,
                        "capture_sha256": sha(out/("capture-"+name+".pt")),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "scope": "includes observational hooks and CPU capture copies, not production throughput"}
                    events.append(event)
                    phase("diagnostic_backward", **event)
        # Numerical mismatches are results of this diagnostic, never filtered.
        # All eight raw captures exist before deriving any comparison.
        comparisons = {}
        pair_names = []
        for mode in RECIPE["modes"]:
            pair_names.extend([(f"{mode}-full-0", f"{mode}-full-1"),
                               (f"{mode}-selected-0", f"{mode}-selected-1"),
                               (f"{mode}-full-0", f"{mode}-selected-0"),
                               (f"{mode}-full-1", f"{mode}-selected-1")])
        pair_names.extend((f"default-{p}-0", f"strict-{p}-0") for p in RECIPE["paths"])
        for left, right in pair_names:
            comparisons[left+"__"+right] = compare_captures(captures[left], captures[right],
                atol=RECIPE["gradient_atol"], rtol=RECIPE["gradient_rtol"])
        save(out/"comparisons.json", comparisons)
        parameters_after, references_after = signature(model.named_parameters()), signature(model.named_buffers())
        if parameters_before != parameters_after or references_before != references_after:
            raise ValueError("parameters or buffers changed without optimization")
        terminal = {"status": "diagnostic_complete", "freeze_sha256": args.freeze_sha256,
            "scope": "eight artificial backward observations, not longer-context training or intervention utility",
            "source_pins": freeze["sources"], "optimizer_updates": 0, "scientific_responses": 0,
            "environment_responses": 0, "backward_passes": 8, "comparisons": 10,
            "comparisons_sha256": sha(out/"comparisons.json"), "model_parameters_unchanged": True,
            "reference_buffers_unchanged": True, "parameters_before": parameters_before,
            "parameters_after": parameters_after, "buffers_before": references_before, "buffers_after": references_after,
            "long_context_qualified": False, "gpu05_status": "failed_and_unchanged",
            "cuda_step_ms_total": sum(e["cuda_elapsed_ms"] for e in events),
            "peak_gpu_allocated_bytes": max([torch.cuda.max_memory_allocated()]+[x[0] for x in memory_peaks]),
            "peak_gpu_reserved_bytes": max([torch.cuda.max_memory_reserved()]+[x[1] for x in memory_peaks]),
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
