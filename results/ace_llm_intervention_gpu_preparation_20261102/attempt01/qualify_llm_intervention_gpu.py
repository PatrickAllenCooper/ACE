#!/usr/bin/env python3
"""Bounded actual-model adapter smoke using artificial public states only.

This is qualification of computation and an action interface, not evidence of
experimental utility. No scientific generator, private score or training-world
preference is loaded. Every startup/compute/output phase is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

SOURCES = {"supervise_llm_intervention_gpu.py", "qualify_llm_intervention_gpu.py",
           "llm_intervention_contract.py", "llm_intervention_learning.py"}
MODELS = {"model.safetensors", "tokenizer_config.json", "config.json", "tokenizer.json",
          "generation_config.json", "merges.txt", "vocab.json"}
PACKAGES = {"torch", "transformers", "numpy", "scipy", "safetensors", "huggingface-hub"}


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
    if (freeze["schema"] != "ace-llm-gpu-smoke-v1" or set(freeze["sources"]) != SOURCES
            or set(freeze["model_files"]) != MODELS or set(freeze["runtime"]) != PACKAGES
            or freeze["training"] != {"seed": 107, "rank": 8, "alpha": 16., "sft_steps": 8,
                "dpo_steps": 4, "learning_rate": 0.0005, "beta": .1, "max_tokens": 512,
                "max_new_tokens": 32} or freeze["startup_seconds"] != 420
            or freeze["worker_seconds"] != 840):
        raise ValueError("incomplete or altered bounded smoke design")
    for section in ("sources", "model_files"):
        if any(Path(name).name != name or not isinstance(pin, str) or len(pin) != 64
               for name, pin in freeze[section].items()):
            raise ValueError("invalid input pin")


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    phase_number = 0
    def phase(name, **extra):
        nonlocal phase_number
        value = {"phase": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "elapsed_seconds": time.monotonic()-started,
                 "process_cpu_seconds": time.process_time()-cpu, **extra}
        save(out/f"phase-{phase_number:03d}.json", value)
        phase_number += 1
    def deadline(signum, frame):
        raise TimeoutError("bounded startup or worker deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    try:
        phase("input_verification_start")
        raw = Path(args.freeze).read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.freeze_sha256:
            raise ValueError("freeze digest differs")
        freeze = json.loads(raw)
        validate_freeze(freeze)
        signal.alarm(freeze["startup_seconds"])
        source, model_dir = Path(args.freeze).resolve().parent, Path(freeze["model_path"])
        if (Path(__file__).resolve() != source/"qualify_llm_intervention_gpu.py"
                or list(sys.version_info[:3]) != freeze["python_version"]
                or {p.name for p in model_dir.iterdir()} != MODELS):
            raise ValueError("worker/Python/model membership mismatch")
        for names, base in ((freeze["sources"], source), (freeze["model_files"], model_dir)):
            for name, pin in names.items():
                if sha(base/name) != pin:
                    raise ValueError("source/model bytes differ: " + name)
        runtime = {name: importlib.metadata.version(name) for name in PACKAGES}
        if runtime != freeze["runtime"]:
            raise ValueError("runtime mismatch")
        phase("input_verification_complete", packages=runtime)
        # Slurm allocation/accounting is additionally authenticated by the
        # supervisor packet and terminal acceptance, never inferred from CUDA.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        sys.path.insert(0, str(source))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llm_intervention_contract import Action, Node, MechanismSummary, ChargedExecutor, policy_prompt
        from llm_intervention_learning import (install_adapters, pack_actions, sequence_log_probs,
            supervised_loss, preference_loss, snapshot_reference, reference_policy)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("exactly one visible BF16-compatible GPU required")
        torch.manual_seed(freeze["training"]["seed"])
        torch.cuda.manual_seed_all(freeze["training"]["seed"])
        props = torch.cuda.get_device_properties(0)
        phase("imports_complete", device_name=props.name, total_device_bytes=props.total_memory,
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_cuda=torch.version.cuda)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        graph = (Node("X", (), "unit", (-1., 1.)), Node("Y", ("X",), "unit"))
        menu = (Action("a0", "intervene", (("X", -1.),), ("Y",), 1),
                Action("a1", "intervene", (("X", 1.),), ("Y",), 1))
        summary = (MechanismSummary("Y", ("intercept", "X"), (0., 0.), ((1., 0.), (0., 1.)), 1., 0),)
        with ChargedExecutor(graph, menu, 1, out/"prompt.jsonl", lambda *_: None) as ex:
            state = ex.state("Learn the response of Y to interventions on X", summary)
        prompt = policy_prompt(state)
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                            add_generation_prompt=True, enable_thinking=False)
        action_text = ['{"action_id":"a0"}', '{"action_id":"a1"}']
        action_ids = [tokenizer.encode(x, add_special_tokens=False)+[tokenizer.eos_token_id] for x in action_text]
        paired = pack_actions(prompt_ids, action_ids, tokenizer.pad_token_id, 512)
        chosen = pack_actions(prompt_ids, action_ids[:1], tokenizer.pad_token_id, 512)
        save(out/"artificial_tokens.json", {"state": state, "prompt_ids": prompt_ids,
                "action_ids": action_ids, "labels_are_arbitrary_interface_fixture": True})
        phase("tokenization_complete", prompt_tokens=len(prompt_ids), pair_width=paired.input_ids.shape[1])
        phase("model_loading_start")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.to("cuda")
        names = install_adapters(model, rank=8, alpha=16.)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(names) != 56 or sum(p.numel() for p in trainable) != 1146880:
            raise ValueError("unexpected actual adapter layout")
        phase("model_loading_complete", projections=len(names), trainable_parameters=sum(p.numel() for p in trainable))

        def weights_signature(parameters):
            h = hashlib.sha256()
            for name, parameter in parameters:
                h.update(name.encode())
                h.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return h.hexdigest()
        frozen_before = weights_signature((n, p) for n, p in model.named_parameters() if not p.requires_grad)
        adapter_before = weights_signature((n, p) for n, p in model.named_parameters() if p.requires_grad)
        with torch.no_grad():
            initial_scores = sequence_log_probs(model, paired).cpu().tolist()
            initial_sft = float(supervised_loss(model, chosen).cpu())
        torch.cuda.synchronize()
        phase("first_gpu_forward_complete", allocated_bytes=torch.cuda.memory_allocated(),
              initial_scores=initial_scores, initial_sft_loss=initial_sft)
        # A real forward has now completed. Overall deadline remains measured
        # from worker start rather than restarting a full wall cap here.
        signal.alarm(max(1, int(freeze["worker_seconds"]-(time.monotonic()-started))))
        optimizer = torch.optim.AdamW(trainable, lr=.0005, weight_decay=0.)
        events = []
        for stage, count in (("sft", 8), ("dpo", 4)):
            if stage == "dpo":
                snapshot_reference(model)
                with reference_policy(model):
                    reference_scores = sequence_log_probs(model, paired).detach()
                reference_signature = weights_signature((n, b) for n, b in model.named_buffers() if "reference_" in n)
                phase("dpo_reference_frozen", scores=reference_scores.cpu().tolist(), digest=reference_signature)
            for step in range(count):
                optimizer.zero_grad(set_to_none=True)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                loss = (supervised_loss(model, chosen) if stage == "sft" else
                        preference_loss(sequence_log_probs(model, paired).reshape(1, 2),
                                        reference_scores.reshape(1, 2), beta=.1))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                if not torch.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise ValueError("nonfinite/zero adapter gradient or frozen base gradient")
                optimizer.step()
                end.record()
                torch.cuda.synchronize()
                event = {"stage": stage, "step": step, "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm.detach().cpu()), "cuda_elapsed_ms": begin.elapsed_time(end),
                    "allocated_bytes": torch.cuda.memory_allocated(), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                events.append(event)
                phase("training_step", **event)
        with torch.no_grad():
            final_sft = float(supervised_loss(model, chosen).cpu())
            final_scores = sequence_log_probs(model, paired).cpu().tolist()
        with reference_policy(model):
            reference_after = sequence_log_probs(model, paired).cpu()
        if not torch.equal(reference_after, reference_scores.cpu()):
            raise ValueError("frozen reference behavior changed")
        if weights_signature((n, b) for n, b in model.named_buffers() if "reference_" in n) != reference_signature:
            raise ValueError("reference weights changed")
        if weights_signature((n, p) for n, p in model.named_parameters() if not p.requires_grad) != frozen_before:
            raise ValueError("frozen base bytes changed")
        adapter_after = weights_signature((n, p) for n, p in model.named_parameters() if p.requires_grad)
        if adapter_before == adapter_after or not final_sft < initial_sft:
            raise ValueError("adapter update or artificial loss improvement absent")
        model.eval()
        phase("generation_start")
        with torch.no_grad():
            ids = torch.tensor([prompt_ids], device="cuda")
            sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=32, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        sequence = sequence[0].cpu().tolist()
        # Exactly one artificial response. It tests action binding, not utility.
        with ChargedExecutor(graph, menu, 1, out/"generated_action.jsonl",
                             lambda a, r: {"Y": dict(a.targets)["X"]}) as ex:
            response = ex.execute_generated(prompt_ids, sequence,
                    lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
        if response is None:
            raise ValueError("artificial model stopped without testing intervention execution")
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, out/"adapter.pt")
        phase("final_output", final_sft_loss=final_sft, final_scores=final_scores,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), adapter_sha256=sha(out/"adapter.pt"))
        save(out/"terminal.json", {"status": "complete", "freeze_sha256": args.freeze_sha256,
            "scope": "actual pretrained GPU adapter/interface smoke, no scientific utility result",
            "source_pins": freeze["sources"], "sft_steps": 8, "dpo_steps": 4,
            "scientific_responses": 0, "artificial_responses": 1,
            "initial_sft_loss": initial_sft, "final_sft_loss": final_sft,
            "frozen_base_sha256": frozen_before, "adapter_before": adapter_before, "adapter_after": adapter_after,
            "cuda_step_ms_total": sum(x["cuda_elapsed_ms"] for x in events),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    except BaseException as exc:
        save(out/"failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic()-started, "process_cpu_seconds": time.process_time()-cpu})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
