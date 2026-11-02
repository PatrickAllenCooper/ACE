#!/usr/bin/env python3
"""Authenticate one bounded Slurm GPU smoke and preserve progress/failure.

External scheduler completion and complete local custody remain required after
this supervisor. NVIDIA process telemetry is independent of worker counters.
Whole-device utilization is not attributed to this process on shared/MIG GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def scheduler_fields(raw):
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9/]*)=", raw))
    fields = {}
    for i, match in enumerate(matches):
        if match[1] in fields:
            raise ValueError("duplicate scheduler field")
        fields[match[1]] = raw[match.end():matches[i+1].start() if i+1 < len(matches) else len(raw)].strip()
    return fields


def memory_mib(text):
    match = re.fullmatch(r"(\d+)([MG])", text)
    if not match:
        raise ValueError("unrecognized Slurm memory units")
    return int(match[1]) * (1024 if match[2] == "G" else 1)


def authenticate_allocation(raw, job_id, resources):
    expected = {"account": "ucb736_asc1", "partition": "aa100", "qos": "gpu-normal",
                "requested_cpus": 4, "max_allocated_cpus": 8, "memory_mib": 12288,
                "wall_seconds": 900, "gpus": 1, "gpu_type": "a100_3g.20gb"}
    if resources != expected or not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("altered resource request or invalid job identity")
    fields = scheduler_fields(raw)
    for name, wanted in {"JobId": job_id, "Account": "ucb736_asc1", "Partition": "aa100",
                         "QOS": "gpu-normal", "JobState": "RUNNING", "NumNodes": "1",
                         "NumTasks": "1", "CPUs/Task": "4", "TimeLimit": "00:15:00"}.items():
        if fields.get(name) != wanted:
            raise ValueError("scheduler allocation mismatch: " + name)
    if not fields.get("UserId", "").startswith("paco0228("):
        raise ValueError("unexpected job owner")
    pairs = fields.get("AllocTRES", "").split(",")
    tres = {}
    for pair in pairs:
        if pair.count("=") != 1:
            raise ValueError("invalid AllocTRES")
        key, value = pair.split("=")
        if key in tres:
            raise ValueError("duplicate allocated resource")
        tres[key] = value
    if (tres.get("gres/gpu") != "1" or tres.get("gres/gpu:a100_3g.20gb") != "1"
            or tres.get("node") != "1" or not str(tres.get("cpu", "")).isdigit()
            or not 4 <= int(tres["cpu"]) <= 8 or fields.get("NumCPUs") != tres["cpu"]
            or memory_mib(tres.get("mem", "")) != 12288
            or any(k.startswith("gres/") and k not in {"gres/gpu", "gres/gpu:a100_3g.20gb"} for k in tres)):
        raise ValueError("actual CPU/memory/GPU allocation differs")
    return {"fields": fields, "allocated_resources": tres}


def gpu_process_sample(pid):
    command = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        # Retain only our worker's process rows, never unrelated users' work.
        rows = [line for line in p.stdout.splitlines() if line.split(",")[0].strip() == str(pid)]
        return {"returncode": p.returncode, "owned_rows": rows, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "owned_rows": [], "error": type(exc).__name__ + ": " + str(exc)}


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    process = None
    try:
        freeze_path = Path(args.freeze).resolve()
        if sha(freeze_path) != args.freeze_sha256:
            raise ValueError("freeze digest mismatch")
        freeze = json.loads(freeze_path.read_bytes())
        source = freeze_path.parent
        required = {"supervise_llm_intervention_gpu.py", "qualify_llm_intervention_gpu.py",
                    "llm_intervention_contract.py", "llm_intervention_learning.py"}
        if set(freeze["sources"]) != required or Path(__file__).resolve() != source/"supervise_llm_intervention_gpu.py":
            raise ValueError("source membership/entrypoint mismatch")
        for name, pin in freeze["sources"].items():
            if sha(source/name) != pin:
                raise ValueError("source pin mismatch: " + name)
        if list(sys.version_info[:3]) != freeze["python_version"]:
            raise ValueError("supervisor Python differs")
        job_id = os.environ.get("SLURM_JOB_ID", "")
        check = subprocess.run(["scontrol", "show", "job", "-o", job_id], capture_output=True, text=True,
                               timeout=10, check=False)
        save(out/"scheduler_raw.json", {"job_id": job_id, "returncode": check.returncode,
                                       "stdout": check.stdout, "stderr": check.stderr})
        if check.returncode:
            raise ValueError("live allocation unavailable")
        allocation = authenticate_allocation(check.stdout, job_id, freeze["resources"])
        save(out/"allocation.json", allocation)
        command = [sys.executable, "-I", "-B", str(source/"qualify_llm_intervention_gpu.py"),
                   "--freeze", str(freeze_path), "--freeze-sha256", args.freeze_sha256,
                   "--output", str(out/"worker")]
        with (out/"child.stdout").open("x") as stdout, (out/"child.stderr").open("x") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                         HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1"))
            save(out/"child_start.json", {"command": command, "pid": process.pid,
                "freeze_sha256": args.freeze_sha256, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            child_start = time.monotonic()
            saw_owned_gpu = False
            first_forward = None
            with (out/"gpu_process_samples.jsonl").open("x") as telemetry:
                while process.poll() is None:
                    elapsed = time.monotonic()-child_start
                    sample = gpu_process_sample(process.pid)
                    saw_owned_gpu |= bool(sample["owned_rows"])
                    telemetry.write(json.dumps({"child_elapsed_seconds": elapsed, **sample})+"\n")
                    telemetry.flush()
                    for path in (out/"worker").glob("phase-*.json"):
                        try:
                            phase = json.loads(path.read_bytes())
                        except (OSError, ValueError):
                            continue
                        if phase.get("phase") == "first_gpu_forward_complete":
                            first_forward = phase
                    if elapsed > 870 or (elapsed > 425 and first_forward is None):
                        raise TimeoutError("supervisor startup/total limit reached")
                    time.sleep(2)
            returncode = process.wait()
        terminal_path = out/"worker"/"terminal.json"
        if returncode or not terminal_path.is_file():
            raise ValueError("worker failed or has no completion receipt")
        terminal = json.loads(terminal_path.read_bytes())
        if (terminal["status"] != "complete" or terminal["freeze_sha256"] != args.freeze_sha256
                or terminal["source_pins"] != freeze["sources"] or terminal["sft_steps"] != 8
                or terminal["dpo_steps"] != 4 or terminal["scientific_responses"] != 0
                or terminal["artificial_responses"] != 1 or terminal["cuda_step_ms_total"] <= 0
                or terminal["peak_gpu_allocated_bytes"] <= 0 or not saw_owned_gpu
                or first_forward is None or first_forward["elapsed_seconds"] > 420):
            raise ValueError("missing runtime/interface/utilization qualification")
        cleanup = gpu_process_sample(process.pid)
        save(out/"cleanup_gpu_process.json", cleanup)
        if cleanup["returncode"] != 0 or cleanup["owned_rows"]:
            raise ValueError("GPU process cleanup not independently confirmed")
        save(out/"supervisor.json", {"status": "complete", "job_id": job_id,
            "freeze_sha256": args.freeze_sha256, "worker_terminal_sha256": sha(terminal_path),
            "owned_nvidia_process_seen": saw_owned_gpu, "first_gpu_forward_seconds": first_forward["elapsed_seconds"],
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-start,
            "whole_device_utilization_not_attributed": True, "scientific_utility_qualified": False})
    except BaseException as exc:
        if process is not None:
            terminate(process)
        save(out/"supervisor_failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "child_returncode": process.returncode if process else None,
            "elapsed_seconds": time.monotonic()-start, "process_cpu_seconds": time.process_time()-cpu})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
