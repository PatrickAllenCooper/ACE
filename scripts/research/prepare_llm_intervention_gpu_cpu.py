#!/usr/bin/env python3
"""Target-CPU fixture, startup-phase and owned-group qualification before GPU.

No pretrained model is instantiated. This does not repeat the earlier eighteen
learning/contract fixtures. It tests the new GPU packet and exact new prompt.
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
import sys
import time
import traceback
import unittest

SOURCES = {"prepare_llm_intervention_gpu_cpu.py", "qualify_llm_intervention_gpu.py",
           "supervise_llm_intervention_gpu.py", "test_llm_intervention_gpu_packet.py",
           "llm_intervention_contract.py", "llm_intervention_learning.py"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def run(args):
    out = Path(args.output); out.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    number = 0
    def phase(name, **extra):
        nonlocal number
        save(out/f"phase-{number:03d}.json", {"phase": name, "elapsed_seconds": time.monotonic()-start,
            "process_cpu_seconds": time.process_time()-cpu, **extra})
        number += 1
    def deadline(signum, frame):
        raise TimeoutError("CPU preparation deadline")
    signal.signal(signal.SIGALRM, deadline); signal.alarm(560)
    try:
        phase("source_auth_start")
        freeze_path = Path(args.freeze).resolve()
        if sha(freeze_path) != args.freeze_sha256:
            raise ValueError("freeze pin mismatch")
        freeze = json.loads(freeze_path.read_bytes()); source = freeze_path.parent
        if (set(freeze["sources"]) != SOURCES or list(sys.version_info[:3]) != freeze["python_version"]
                or Path(__file__).resolve() != source/"prepare_llm_intervention_gpu_cpu.py"):
            raise ValueError("CPU source/Python entrypoint mismatch")
        for name, pin in freeze["sources"].items():
            if sha(source/name) != pin: raise ValueError("source differs: " + name)
        sys.path.insert(0, str(source))
        from qualify_llm_intervention_gpu import PACKAGES, MODELS, build_artificial_fixture
        model = Path(freeze["model_path"])
        if (set(freeze["model_files"]) != MODELS or set(freeze["runtime"]) != PACKAGES
                or {p.name for p in model.iterdir()} != MODELS):
            raise ValueError("model/runtime membership mismatch")
        phase("source_auth_complete")
        for name, pin in freeze["model_files"].items():
            phase("model_hash_start", filename=name)
            if sha(model/name) != pin: raise ValueError("model differs: " + name)
            phase("model_hash_complete", filename=name)
        versions = {n: importlib.metadata.version(n) for n in PACKAGES}
        if versions != freeze["runtime"]: raise ValueError("package versions differ")
        phase("runtime_metadata_complete", packages=versions)
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                          TOKENIZERS_PARALLELISM="false")
        phase("torch_import_start")
        import torch
        torch.set_num_threads(1)
        phase("torch_import_complete", cuda_build=torch.version.cuda, origin=torch.__file__)
        phase("tokenizer_import_start")
        from transformers import AutoTokenizer
        phase("tokenizer_import_complete")
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        phase("tokenizer_load_complete")
        graph, menu, fixture = build_artificial_fixture(tokenizer, out/"artificial_prompt.jsonl")
        save(out/"fixture.json", fixture)
        phase("fixture_complete", fixture_sha256=sha(out/"fixture.json"), prompt_tokens=len(fixture["prompt_ids"]))
        suite = unittest.defaultTestLoader.loadTestsFromName("test_llm_intervention_gpu_packet")
        with (out/"tests.stdout").open("x") as stdout, (out/"tests.stderr").open("x") as stderr:
            import contextlib
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = unittest.TextTestRunner(stream=stderr, verbosity=2).run(suite)
        if not result.wasSuccessful() or result.testsRun != 9 or result.skipped:
            raise ValueError("new target-runtime packet/cleanup fixtures failed")
        phase("new_packet_tests_complete", tests=9)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out/"terminal.json", {"status": "complete", "freeze_sha256": args.freeze_sha256,
            "source_pins": freeze["sources"], "fixture_sha256": sha(out/"fixture.json"),
            "package_versions": versions, "cuda_build_recorded": torch.version.cuda,
            "scope": "CPU exact artificial prompt and nine new packet/cleanup fixtures, no pretrained model",
            "pretrained_model_loads": 0, "scientific_responses": 0, "gpu_allocations": 0,
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-start,
            "reaped_child_cpu_seconds": children_after.ru_utime+children_after.ru_stime-children_before.ru_utime-children_before.ru_stime,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    except BaseException as exc:
        save(out/"failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic()-start, "process_cpu_seconds": time.process_time()-cpu})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True); p.add_argument("--output", required=True)
    run(p.parse_args())
