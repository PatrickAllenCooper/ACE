#!/usr/bin/env python3
"""Supervise a separate shared-task long-context artificial GPU qualification.

Derived from the accepted GPU04 supervisor with the same allocation, telemetry,
startup and cleanup guards. Its worker and completion contract are new.

External scheduler completion and complete local custody remain required after
this supervisor. NVIDIA process telemetry is independent of worker counters.
Whole-device utilization is not attributed to this process on shared/MIG GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=", raw))
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
    expected = {"account": "ucb736_asc1", "partition": "aa100", "qos": "gpu-testing",
                "requested_cpus": 4, "max_allocated_cpus": 8, "memory_mib": 12288,
                "wall_seconds": 600, "gpus": 1, "gpu_type": "a100_3g.20gb"}
    if resources != expected or not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("altered resource request or invalid job identity")
    fields = scheduler_fields(raw)
    for name, wanted in {"JobId": job_id, "Account": "ucb736_asc1", "Partition": "aa100",
                         "QOS": "gpu-testing", "JobState": "RUNNING", "NumNodes": "1",
                         "NumTasks": "1", "TimeLimit": "00:10:00"}.items():
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
    # Alpine may report only the typed GRES. It is the authoritative slice
    # count. If an aggregate count is also present, it must agree exactly.
    if (("gres/gpu" in tres and tres["gres/gpu"] != "1")
            or tres.get("gres/gpu:a100_3g.20gb") != "1"
            or tres.get("node") != "1" or not str(tres.get("cpu", "")).isdigit()
            or not 4 <= int(tres["cpu"]) <= 8 or fields.get("NumCPUs") != tres["cpu"]
            or fields.get("CPUs/Task") != tres["cpu"]
            or memory_mib(tres.get("mem", "")) != 12288
            or any(k.startswith("gres/") and k not in {"gres/gpu", "gres/gpu:a100_3g.20gb"} for k in tres)):
        raise ValueError("actual CPU/memory/GPU allocation differs")
    requested = {}
    for pair in fields.get("ReqTRES", "").split(","):
        if pair.count("=") != 1:
            raise ValueError("invalid requested resource evidence")
        key, value = pair.split("=")
        if key in requested:
            raise ValueError("duplicate requested resource")
        requested[key] = value
    if requested.get("cpu") != "4" or memory_mib(requested.get("mem", "")) != 12288:
        raise ValueError("CPU/memory request differs from frozen request")
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


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate(process, deadline):
    """Clean the owned group even if its leader already exited. Never raise."""
    result = {"pgid": process.pid, "term_sent": False, "kill_sent": False, "errors": []}
    try:
        process.poll()
        if group_alive(process.pid):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                result["term_sent"] = True
            except ProcessLookupError:
                pass
            grace = min(deadline-2., time.monotonic()+8.)
            while time.monotonic() < grace and group_alive(process.pid):
                process.poll()
                time.sleep(.1)
            if group_alive(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    result["kill_sent"] = True
                except ProcessLookupError:
                    pass
            while time.monotonic() < deadline and group_alive(process.pid):
                process.poll()
                time.sleep(.1)
        process.poll()
        result["group_still_exists"] = group_alive(process.pid)
    except BaseException as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        result["group_still_exists"] = None
    result["leader_returncode"] = process.returncode
    return result


def absolute_deadlines(supervisor_start, child_start):
    # Twenty-five seconds are reserved for cleanup, final validation and writes.
    supervisor_end = supervisor_start + 570.
    return {"supervisor_end": supervisor_end, "worker_end": min(child_start+540., supervisor_end-25.),
            "startup_end": min(child_start+180., supervisor_end-25.)}


def read_phases(directory):
    result = []
    for path in sorted(directory.glob("phase-*.json")):
        try:
            result.append(json.loads(path.read_bytes()))
        except (OSError, ValueError):
            continue  # May observe an exclusive receipt before its write ends.
    return result


def arm_final_deadline(start, process):
    """Keep a hard failure deadline active even if filesystem I/O stalls."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def final_timeout(signum, frame):
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        # There is intentionally no success receipt on this path. Slurm/log
        # reconciliation must preserve this uncatchable-terminal failure.
        os._exit(124)
    signal.signal(signal.SIGALRM, final_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.01, start+570.-time.monotonic()))


def finish_failure(out, failure, process, start, cpu):
    arm_final_deadline(start, process)
    persistence_errors = []
    try:
        save(out/"supervisor_failure_intent.json", failure)
    except BaseException as exc:
        persistence_errors.append("intent: " + type(exc).__name__ + ": " + str(exc))
    finally:
        # This executes even when the intent receipt cannot be persisted.
        cleanup = terminate(process, min(start+565., time.monotonic()+18.)) if process is not None else None
    gpu_cleanup = gpu_process_sample(process.pid) if process is not None else None
    result = {**failure, "cleanup": cleanup, "persistence_errors": persistence_errors,
              "post_cleanup_gpu_sample": gpu_cleanup,
              "child_returncode": process.returncode if process else None,
              "elapsed_seconds": time.monotonic()-start, "process_cpu_seconds": time.process_time()-cpu}
    try:
        save(out/"supervisor_failure.json", result)
    except BaseException as exc:
        persistence_errors.append("terminal: " + type(exc).__name__ + ": " + str(exc))
        print(json.dumps(result), file=sys.stderr, flush=True)
    # Do not cancel the final alarm. It covers propagation, receipt I/O and
    # interpreter shutdown through the570second absolute supervisor envelope.
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    process = None
    cleanup = None
    def interrupted(signum, frame):
        raise InterruptedError("supervisor received termination signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    # This covers source/authentication and telemetry calls as well as the
    # child loop. Cleanup and failure writing retain25seconds before570.
    signal.alarm(545)
    try:
        freeze_path = Path(args.freeze).resolve()
        if sha(freeze_path) != args.freeze_sha256:
            raise ValueError("freeze digest mismatch")
        freeze = json.loads(freeze_path.read_bytes())
        source = freeze_path.parent
        required = {"qualify_llm_intervention_precise_gpu.py", "supervise_llm_intervention_precise_gpu.py",
                    "llm_intervention_contract.py", "llm_intervention_learning.py", "llm_intervention_grammar.py",
                    "llm_intervention_scm.py", "llm_intervention_episode.py", "llm_intervention_numerical_policy.py",
                    "llm_intervention_target_risk.py", "llm_intervention_task.py",
                    "llm_intervention_language_policy.py", "llm_intervention_action_scoring.py", "llm_intervention_precision.py", "llm_intervention_gradient_probe.py"}
        if set(freeze["sources"]) != required or Path(__file__).resolve() != source/"supervise_llm_intervention_precise_gpu.py":
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
        command = [sys.executable, "-I", "-B", str(source/"qualify_llm_intervention_precise_gpu.py"),
                   "--freeze", str(freeze_path), "--freeze-sha256", args.freeze_sha256,
                   "--output", str(out/"worker")]
        with (out/"child.stdout").open("x") as stdout, (out/"child.stderr").open("x") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                         HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1"))
            save(out/"child_start.json", {"command": command, "pid": process.pid,
                "freeze_sha256": args.freeze_sha256, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            child_start = time.monotonic()
            limits = absolute_deadlines(start, child_start)
            saw_owned_gpu = False
            first_forward = None
            seen_phases = 0
            last_progress = child_start
            with (out/"gpu_process_samples.jsonl").open("x") as telemetry:
                while process.poll() is None:
                    elapsed = time.monotonic()-child_start
                    sample = gpu_process_sample(process.pid)
                    saw_owned_gpu |= sample["returncode"] == 0 and bool(sample["owned_rows"])
                    telemetry.write(json.dumps({"child_elapsed_seconds": elapsed, **sample})+"\n")
                    telemetry.flush()
                    phases = read_phases(out/"worker")
                    if len(phases) > seen_phases:
                        seen_phases = len(phases)
                        last_progress = time.monotonic()
                    for phase in phases:
                        if phase.get("phase") == "first_gpu_forward_complete":
                            first_forward = phase
                    now = time.monotonic()
                    if now > limits["worker_end"] or (now > limits["startup_end"] and first_forward is None):
                        raise TimeoutError("supervisor startup/total limit reached")
                    if first_forward is not None and now-last_progress > 120:
                        raise TimeoutError("no recorded GPU-worker progress for120seconds")
                    time.sleep(2)
            returncode = process.wait()
            save(out/"child_exit.json", {"returncode": returncode, "monotonic_seconds": time.monotonic(),
                "child_elapsed_seconds": time.monotonic()-child_start,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scope": "supervisor observation of process exit, includes polling interval"})
        # Include receipts written between the final live sample and exit.
        phases = read_phases(out/"worker")
        first_forward = next((p for p in phases if p.get("phase") == "first_gpu_forward_complete"), None)
        steps = [p for p in phases if p.get("phase") == "training_step"]
        cleanup = terminate(process, min(start+565., time.monotonic()+18.))
        save(out/"process_group_cleanup.json", cleanup)
        if cleanup["errors"] or cleanup["group_still_exists"] is not False:
            raise ValueError("owned process group cleanup not confirmed")
        gpu_cleanup = gpu_process_sample(process.pid)
        save(out/"cleanup_gpu_process.json", gpu_cleanup)
        if gpu_cleanup["returncode"] != 0 or gpu_cleanup["owned_rows"]:
            raise ValueError("GPU process cleanup not independently confirmed")
        terminal_path = out/"worker"/"terminal.json"
        if returncode or not terminal_path.is_file():
            raise ValueError("worker failed or has no completion receipt")
        terminal = json.loads(terminal_path.read_bytes())
        # Import the stdlib-only worker validator only after every source pin
        # has been authenticated. This does not load model packages here.
        sys.path.insert(0, str(source))
        from qualify_llm_intervention_precise_gpu import validate_freeze, validate_terminal
        validate_freeze(freeze)
        validate_terminal(terminal, freeze, args.freeze_sha256, steps)
        if (not saw_owned_gpu or first_forward is None or first_forward["elapsed_seconds"] > 180
                or terminal["adapter_sha256"] != sha(out/"worker"/"adapter.pt")):
            raise ValueError("missing independent GPU activity, startup or checkpoint evidence")
        if time.monotonic() > start+570.:
            raise TimeoutError("whole-supervisor envelope exceeded")
        save(out/"supervisor.json", {"status": "complete", "job_id": job_id,
            "freeze_sha256": args.freeze_sha256, "worker_terminal_sha256": sha(terminal_path),
            "owned_nvidia_process_seen": saw_owned_gpu, "first_gpu_forward_seconds": first_forward["elapsed_seconds"],
            "process_cpu_seconds": time.process_time()-cpu, "elapsed_seconds": time.monotonic()-start,
            "whole_device_utilization_not_attributed": True, "scientific_utility_qualified": False})
        signal.alarm(0)
    except BaseException as exc:
        failure = {"status": "failed", "exception_type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc()}
        finish_failure(out, failure, process, start, cpu)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
