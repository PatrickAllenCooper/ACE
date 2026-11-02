#!/usr/bin/env python3
"""CPU-only Qwen grammar, actual logits-processor and import qualification.

No pretrained model is instantiated. No CUDA query, training or environment
response is performed. This is new decoding integration, not a repeat of the
previous CPU learning and process-group fixtures.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

SOURCES = {"prepare_llm_intervention_decoding_cpu.py", "qualify_llm_intervention_gpu.py",
    "supervise_llm_intervention_gpu.py", "llm_intervention_contract.py",
    "llm_intervention_learning.py", "llm_intervention_grammar.py"}
MODELS = {"model.safetensors", "tokenizer_config.json", "config.json", "tokenizer.json",
          "generation_config.json", "merges.txt", "vocab.json"}
PACKAGES = {"torch", "transformers", "numpy", "scipy", "safetensors", "huggingface-hub",
            "tokenizers", "jinja2"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_processor(grammar, vocabulary_size):
    """Exercise the actual HF callback and mask at every distinct prefix."""
    import torch
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
    from qualify_llm_intervention_gpu import prefix_callback
    if (type(vocabulary_size) is not int or vocabulary_size <= 0
            or max(grammar.prompt_ids + tuple(t for x in grammar.completion_ids for t in x)) >= vocabulary_size):
        raise ValueError("token vocabulary differs")
    callback = prefix_callback(grammar)
    processor = PrefixConstrainedLogitsProcessor(callback, num_beams=1)
    scores = torch.arange(vocabulary_size, dtype=torch.float32).reshape(1, -1)
    prefixes = {command[:length] for command in grammar.completion_ids for length in range(len(command))}
    for prefix in sorted(prefixes):
        ids = torch.tensor([grammar.prompt_ids + prefix], dtype=torch.long)
        allowed = grammar.allowed_next(grammar.prompt_ids + prefix)
        result = processor(ids, scores.clone())
        mask = torch.ones(vocabulary_size, dtype=torch.bool)
        mask[list(allowed)] = False
        if (not torch.equal(result[0, list(allowed)], scores[0, list(allowed)])
                or not torch.isneginf(result[0, mask]).all()):
            raise ValueError("processor changed allowed logits or permitted another token")
    for batch_id, ids in ((1, torch.tensor(grammar.prompt_ids)),
                          (0, torch.tensor([vocabulary_size-1]+list(grammar.prompt_ids)))):
        try:
            callback(batch_id, ids)
        except ValueError:
            pass
        else:
            raise ValueError("callback failed to reject unsupported batch or prompt")
    return {"distinct_prefixes_checked": len(prefixes), "commands_checked": len(grammar.completion_ids),
            "vocabulary_size": vocabulary_size, "allowed_logits_preserved": True,
            "outside_tokens_negative_infinity": True, "negative_binding_checks": 2,
            "scope": "actual library logits processor on artificial CPU tensors, no model generation"}


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    number = 0
    def phase(name, **extra):
        nonlocal number
        save(out/f"phase-{number:03d}.json", {"phase": name,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic_seconds": time.monotonic(), "elapsed_seconds": time.monotonic()-started,
            "process_cpu_seconds": time.process_time()-cpu, **extra})
        number += 1
    def timeout(signum, frame):
        raise TimeoutError("CPU decoding qualification exceeded560seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(560)
    stack_log = (out/"runtime_stacks.log").open("x")
    faulthandler.dump_traceback_later(60, repeat=True, file=stack_log, exit=False)
    try:
        phase("authentication_start")
        freeze_path = Path(args.freeze).resolve()
        if sha(freeze_path) != args.freeze_sha256:
            raise ValueError("freeze pin differs")
        freeze = json.loads(freeze_path.read_bytes())
        source = freeze_path.parent
        if (freeze["schema"] != "ace-llm-decoding-cpu-v1" or set(freeze["sources"]) != SOURCES
                or Path(__file__).resolve() != source/"prepare_llm_intervention_decoding_cpu.py"
                or list(sys.version_info[:3]) != freeze["python_version"]
                or freeze["max_new_tokens"] != 32):
            raise ValueError("source, Python or CPU protocol differs")
        for name, pin in freeze["sources"].items():
            if sha(source/name) != pin:
                raise ValueError("source differs: "+name)
        sys.path.insert(0, str(source))
        model = Path(freeze["model_path"])
        if (set(freeze["model_files"]) != MODELS or {p.name for p in model.iterdir()} != MODELS
                or set(freeze["runtime"]) != PACKAGES):
            raise ValueError("model/runtime membership differs")
        for name, pin in freeze["model_files"].items():
            if sha(model/name) != pin:
                raise ValueError("model differs: "+name)
        versions = {n: importlib.metadata.version(n) for n in PACKAGES}
        if versions != freeze["runtime"]:
            raise ValueError("package versions differ")
        accepted = Path(freeze["accepted_fixture_path"])
        if sha(accepted) != freeze["accepted_fixture_sha256"]:
            raise ValueError("accepted predecessor fixture differs")
        phase("authentication_complete", packages=versions)
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        phase("torch_import_start")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        phase("torch_import_complete", origin=torch.__file__, cuda_build=torch.version.cuda)
        phase("transformers_import_start")
        import transformers
        phase("transformers_import_complete", origin=transformers.__file__)
        phase("automodel_import_start")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        phase("automodel_import_complete", model_class_imported_only=AutoModelForCausalLM.__name__)
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        phase("tokenizer_load_complete")
        from qualify_llm_intervention_gpu import build_artificial_fixture, build_decoding_fixture
        graph, menu, fixture = build_artificial_fixture(tokenizer, out/"prompt.jsonl")
        if fixture != json.loads(accepted.read_bytes()):
            raise ValueError("historical artificial prompt/target changed")
        save(out/"fixture.json", fixture)
        grammar, decoding = build_decoding_fixture(tokenizer, menu, fixture, freeze["max_new_tokens"])
        save(out/"decoding_fixture.json", decoding)
        config = json.loads((model/"config.json").read_bytes())
        checked = check_processor(grammar, config["vocab_size"])
        save(out/"processor_checks.json", checked)
        phase("decoding_qualification_complete", **checked)
        save(out/"terminal.json", {"status": "complete", "freeze_sha256": args.freeze_sha256,
            "source_pins": freeze["sources"], "package_versions": versions,
            "fixture_sha256": sha(out/"fixture.json"), "decoding_fixture_sha256": sha(out/"decoding_fixture.json"),
            "processor_checks_sha256": sha(out/"processor_checks.json"),
            "scope": "exact CPU tokenizer and logits processor, CPU imports only, no pretrained model",
            "pretrained_model_loads": 0, "gradient_updates": 0, "environment_responses": 0,
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    except BaseException as exc:
        save(out/"failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-started})
        raise
    finally:
        try:
            phase("worker_function_exit")
        finally:
            faulthandler.cancel_dump_traceback_later()
            stack_log.close()
            signal.alarm(0)


if __name__ == "__main__":
    # The -I entrypoint imports only authenticated sibling code. The launcher
    # must check all source pins before invoking this script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
