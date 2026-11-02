#!/usr/bin/env python3
"""CPU-only runtime, tokenizer and artificial-interface qualification.

No pretrained model is instantiated. The safetensors header is metadata only.
This does not qualify scientific responses, a trained policy or GPU utilization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback
import unittest

SOURCE_FILES = {"llm_intervention_contract.py", "llm_intervention_learning.py",
                "test_llm_intervention_contract.py", "test_llm_intervention_learning.py",
                "qualify_llm_intervention_cpu.py"}
MODEL_FILES = {"model.safetensors", "tokenizer_config.json", "config.json", "tokenizer.json",
               "generation_config.json", "merges.txt", "vocab.json"}
RUNTIME_PACKAGES = {"torch", "transformers", "numpy", "scipy", "safetensors", "huggingface-hub"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(args):
    start, cpu = time.monotonic(), time.process_time()
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    try:
        raw = Path(args.freeze).read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.freeze_sha256:
            raise ValueError("freeze pin mismatch")
        freeze = json.loads(raw)
        source = Path(args.freeze).resolve().parent
        model = Path(freeze["model_path"])
        if (set(freeze["sources"]) != SOURCE_FILES or set(freeze["model_files"]) != MODEL_FILES
                or set(freeze["runtime"]) != RUNTIME_PACKAGES
                or {p.name for p in model.iterdir()} != MODEL_FILES):
            raise ValueError("incomplete or unexpected qualification inventory")
        if list(sys.version_info[:3]) != freeze["python_version"]:
            raise ValueError("Python version mismatch")
        for section, base in (("sources", source), ("model_files", model)):
            for name, pin in freeze[section].items():
                if Path(name).name != name or sha(base/name) != pin:
                    raise ValueError("input pin mismatch: " + name)
        if Path(__file__).resolve() != source/"qualify_llm_intervention_cpu.py":
            raise ValueError("worker must execute from the pinned source directory")
        actual = {name: importlib.metadata.version(name) for name in freeze["runtime"]}
        if actual != freeze["runtime"]:
            raise ValueError("installed runtime version mismatch")
        write(output/"runtime.json", {"python": sys.version, "executable": sys.executable,
                                      "packages": actual, "source_files_verified": len(freeze["sources"]),
                                      "model_files_verified": len(freeze["model_files"])})
        os.environ.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})
        sys.path.insert(0, str(source))
        import torch
        torch.set_num_threads(1)
        from transformers import AutoTokenizer
        from llm_intervention_contract import ChargedExecutor, policy_prompt
        from test_llm_intervention_contract import GRAPH, MENU, SUMMARY
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                                  trust_remote_code=False)
        with ChargedExecutor(GRAPH, MENU, 4, output/"artificial_empty_history.jsonl",
                             lambda *_: (_ for _ in ()).throw(RuntimeError("response forbidden"))) as executor:
            prompt = policy_prompt(executor.state("Predict Y under legal interventions", SUMMARY))
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False)
        completion = '{"action_id":"a0"}'
        action_ids = tokenizer.encode(completion, add_special_tokens=False)
        if tokenizer.decode(action_ids, skip_special_tokens=True) != completion:
            raise ValueError("completion token roundtrip differs")
        if len(ids) + len(action_ids) + 1 > freeze["max_context_tokens"]:
            raise ValueError("artificial state exceeds proposed context limit")
        if type(tokenizer.eos_token_id) is not int or type(tokenizer.pad_token_id) is not int:
            raise ValueError("explicit tokenizer EOS and padding tokens required")
        with (model/"model.safetensors").open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > 16*1024*1024:
                raise ValueError("unexpectedly large model header")
            header = json.loads(f.read(length))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        adapter_parameters = sum(8*sum(v["shape"]) for name, v in tensors.items()
                                 if name.endswith(("q_proj.weight", "v_proj.weight")))
        profile = {"model_revision": freeze["model_revision"], "tensor_count": len(tensors),
                   "stored_elements": sum(math.prod(v["shape"]) for v in tensors.values()),
                   "stored_dtypes": sorted({v["dtype"] for v in tensors.values()}),
                   "weight_file_bytes": (model/"model.safetensors").stat().st_size,
                   "rank8_qv_adapter_parameters": adapter_parameters,
                   "artificial_prompt_tokens": len(ids), "action_tokens_including_eos": len(action_ids)+1,
                   "max_context_tokens": freeze["max_context_tokens"],
                   "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "gpu_memory_measured": False, "pretrained_model_loaded": False}
        write(output/"tokenizer_model_profile.json", profile)
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
                                   ("test_llm_intervention_contract", "test_llm_intervention_learning"))
        with (output/"artificial_tests.log").open("x") as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if not result.wasSuccessful() or result.testsRun != 18 or result.skipped:
            raise ValueError("full artificial fixture family did not pass")
        write(output/"terminal.json", {"status": "complete", "freeze_sha256": args.freeze_sha256,
              "tests_run": result.testsRun, "source_pins": freeze["sources"],
              "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-start,
              "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "scientific_responses": 0, "pretrained_model_loads": 0, "gpu_allocations": 0,
              "scope": "CPU runtime, tokenizer, weight metadata and artificial fixtures only"})
    except BaseException as exc:
        write(output/"failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
              "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic()-start,
              "process_cpu_seconds": time.process_time()-cpu})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
