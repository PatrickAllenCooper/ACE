#!/usr/bin/env python3
"""One bounded persistent-service integration on saved artificial public states.

No training or environment callback. The outer pinned packet contains two
service manifests (frozen base and artificial adapter) and existing fixtures.
External Slurm completion and complete custody remain acceptance requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import types


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def arm_final_deadline(start, process):
    """Hard owned-group fallback remains active through cleanup and final I/O."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def final_timeout(signum, frame):
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        os._exit(124)  # Scheduler/log custody must reconcile this interruption.
    signal.signal(signal.SIGALRM, final_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.01, start+580.-time.monotonic()))


def scheduler_record(out, job_id):
    command = ["scontrol", "show", "job", "-o", job_id]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except subprocess.TimeoutExpired as exc:
        text = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else x
        save(out/"scheduler_raw.json", {"job_id": job_id, "command": command,
            "status": "timeout", "stdout": text(exc.stdout), "stderr": text(exc.stderr)})
        raise
    save(out/"scheduler_raw.json", {"job_id": job_id, "command": command,
        "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    if p.returncode:
        raise RuntimeError("scheduler query failed; raw evidence retained")
    return p.stdout


def require_gpu_sample(sample, *, active):
    if sample.get("returncode") != 0 or type(sample.get("owned_rows")) is not list:
        raise ValueError("independent GPU observation unavailable")
    if bool(sample["owned_rows"]) != active:
        raise ValueError("independent GPU activity/cleanup mismatch")


def validate_session(directory, *, freeze_sha256, identity_sha256, envelope_pins):
    """Reconstruct the four accepted exchanges from saved artifacts only."""
    directory = Path(directory)
    def read(relative):
        return json.loads((directory/relative).read_bytes())
    def digest(value):
        return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    terminal, closed = read("worker/terminal.json"), read("client/closed.json")
    cleanup = closed["cleanup"]
    if (len(envelope_pins) != 4 or terminal["status"] != "complete"
            or terminal["freeze_sha256"] != freeze_sha256 or terminal["generation_count"] != 4
            or terminal["new_training_updates"] != 0 or terminal["environment_responses"] != 0
            or closed["status"] != "closed" or closed["completed_requests"] != 4
            or closed["last_request_id"] != 43 or cleanup["errors"]
            or cleanup["group_still_exists"] is not False or cleanup["leader_returncode"] != 0
            or cleanup["term_sent"] or cleanup["kill_sent"]):
        raise ValueError("incomplete terminal/clean-close evidence")
    authenticated = read("worker/authenticated.json")
    if (authenticated["freeze_sha256"] != freeze_sha256
            or digest(authenticated["identity"]) != identity_sha256
            or read("client/ready.json")["identity_sha256"] != identity_sha256):
        raise ValueError("authenticated startup differs from expected binding")
    if list(directory.rglob("failure.json")):
        raise ValueError("failure evidence forbids successful session qualification")
    phases = [json.loads(p.read_bytes())["phase"] for p in (directory/"worker").glob("phase-*.json")]
    if any(phases.count(n) != 1 for n in ("model_cpu_load_start", "model_cpu_load_complete", "model_ready")):
        raise ValueError("model must load exactly once in each session")
    if {p.name for p in (directory/"worker").glob("generation-*.json")} != {f"generation-{i:04d}.json" for i in range(4)}:
        raise ValueError("generation membership differs")
    generations = []
    for i, pin in enumerate(envelope_pins):
        g = read(f"worker/generation-{i:04d}.json")
        wire = read(f"client/request-{i:04d}.observed-response.json")
        accepted = read(f"client/request-{i:04d}.accepted.json")
        if (g["ordinal"] != i or g["envelope_sha256"] != pin or g["new_training_updates"] != 0
                or g["environment_callbacks_available"] is not False or g["tensor_versions_unchanged"] is not True
                or g["cross_request_cache_supplied"] is not False
                or not isinstance(g["cuda_interval_ms"], (int, float))
                or not math.isfinite(g["cuda_interval_ms"]) or g["cuda_interval_ms"] <= 0
                or g["peak_gpu_allocated_bytes"] <= 0 or wire["request_id"] != 40+i
                or wire["identity_sha256"] != identity_sha256 or wire["envelope_sha256"] != pin
                or wire["completion"] != g["completion"] or accepted["completion"] != g["completion"]
                or accepted["request_id"] != 40+i or accepted["response_sha256"] != digest(wire)):
            raise ValueError("generation/wire/source identity or GPU progress mismatch")
        raw = read(f"worker/raw-generation-{i:04d}.json")
        if (raw["status"] != "returned_unaccepted" or raw["truncated"] is not False
                or raw["token_prefix"] != g["sequence_ids"]
                or sha((directory/f"worker/raw-generation-{i:04d}.bin").read_bytes()) != raw["raw_sha256"]):
            raise ValueError("raw generated token lineage mismatch")
        generations.append(g)
    if (generations[0]["sequence_ids"] != generations[3]["sequence_ids"]
            or generations[0]["prompt_ids"] != generations[3]["prompt_ids"]):
        raise ValueError("separated repeated prompt changed tokens")
    return {"requests": 4, "repeated_tokens_identical": True, "clean_close": closed,
            "cuda_interval_ms": sum(g["cuda_interval_ms"] for g in generations)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--packet-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    root = Path(args.packet).absolute().parent
    start, cpu = time.monotonic(), time.process_time()
    current, monitor, supervisor, completed = None, None, None, False
    stop = threading.Event()
    def deadline(signum, frame):
        # Arm before raising: nested transport handlers may catch BaseException
        # and perform their own receipt I/O before this function regains control.
        arm_final_deadline(start, getattr(current, "process", None) if current is not None else None)
        raise TimeoutError("persistent-service integration deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.signal(signal.SIGINT, deadline)
    signal.setitimer(signal.ITIMER_REAL, 560.)
    try:
        raw = Path(args.packet).read_bytes()
        if sha(raw) != args.packet_sha256:
            raise ValueError("independent packet pin mismatch")
        packet = json.loads(raw)
        if (set(packet) != {"schema", "files", "services", "resources"}
                or packet["schema"] != "ace-persistent-model-integration-v1"
                or packet["services"] != ["frozen.json", "adapter.json"]):
            raise ValueError("fixed artificial integration contract differs")
        data = {}
        for name, expected in packet["files"].items():
            if type(name) is not str or Path(name).name != name or (root/name).is_symlink():
                raise ValueError("plain regular pinned packet inputs required")
            data[name] = (root/name).read_bytes()
            if sha(data[name]) != expected:
                raise ValueError("packet input mismatch: " + name)
        # Bootstrap only captured, independently authenticated service bytes.
        name = "llm_intervention_model_service"
        if name in sys.modules:
            raise ValueError("service module already cached")
        service = types.ModuleType(name)
        service.__file__ = str(root/(name+".py"))
        sys.modules[name] = service
        exec(compile(data[name+".py"], service.__file__, "exec"), service.__dict__)
        needed = set(service.SOURCE_NAMES) | {"qualify_llm_intervention_model_service.py",
            "supervise_llm_intervention_checkpoint_gpu.py", "frozen.json", "adapter.json",
            "case0.json", "case3.json", "stress.json", "trained_adapter.pt", "gpu09_terminal.json"}
        if set(data) != needed or Path(__file__).absolute() != root/"qualify_llm_intervention_model_service.py":
            raise ValueError("unexpected complete packet membership/entrypoint")
        freezes = [service.validate_freeze(service.decode(data[n])) for n in packet["services"]]
        for f in freezes:
            if (f["scope"] != "artificial" or f["sources"] != {n: packet["files"][n] for n in service.SOURCE_NAMES}
                    or f["limits"] != {"startup_seconds": 180, "request_seconds": 30,
                        "idle_seconds": 30, "total_seconds": 240, "max_requests": 4}):
                raise ValueError("service source/scope/deadline contract differs")
        if (freezes[0]["treatment"] != "frozen" or freezes[1]["treatment"] != "artificial_adapter"
                or freezes[0]["model_files"] != freezes[1]["model_files"]
                or freezes[0]["model_directory"] != freezes[1]["model_directory"]
                or freezes[1]["adapter"] != {"file": "trained_adapter.pt", "sha256": packet["files"]["trained_adapter.pt"],
                    "training_receipt": "gpu09_terminal.json", "training_receipt_sha256": packet["files"]["gpu09_terminal.json"],
                    "training_scope": "artificial"}):
            raise ValueError("frozen/adapter treatment bindings differ")
        previous = service.decode(data["gpu09_terminal.json"])
        if previous["status"] != "complete" or previous["adapter_sha256"] != packet["files"]["trained_adapter.pt"]:
            raise ValueError("saved artificial adapter lineage mismatch")
        if (list(sys.version_info[:3]) != [3, 10, 19]
                or {n: service.importlib.metadata.version(n) for n in service.RUNTIME} != service.RUNTIME):
            raise ValueError("integration Python/package versions differ")
        captured = {n: data[n] for n in service.SOURCE_NAMES}
        loader = service.CapturedSources(captured, root)
        sys.meta_path.insert(0, loader)
        from llm_intervention_persistent_policy import PersistentPolicy
        from llm_intervention_task import unpack_envelope
        supervisor = types.ModuleType("pinned_allocation_helpers")
        exec(compile(data["supervise_llm_intervention_checkpoint_gpu.py"],
            str(root/"supervise_llm_intervention_checkpoint_gpu.py"), "exec"), supervisor.__dict__)
        job_id = os.environ.get("SLURM_JOB_ID", "")
        scheduler = scheduler_record(out, job_id)
        save(out/"allocation.json", {"raw": scheduler,
            "validated": supervisor.authenticate_allocation(scheduler, job_id, packet["resources"])})
        cases = {n: service.decode(data[n]) for n in ("case0.json", "case3.json", "stress.json")}
        tasks = {n: unpack_envelope(v)[1] for n, v in cases.items()}
        if tasks["case0.json"] == tasks["case3.json"]:
            raise ValueError("qualification requires distinct public task bindings")
        def telemetry():
            index = 0
            while not stop.is_set():
                for process_path in sorted(out.glob("session*/client/process.json")):
                    try:
                        pid = json.loads(process_path.read_bytes())["pid"]
                        sample = supervisor.gpu_process_sample(pid)
                        save(out/f"telemetry-{index:04d}.json", {"session": process_path.parts[-3],
                            "pid": pid, "elapsed_seconds": time.monotonic()-start, **sample})
                        index += 1
                    except (OSError, ValueError) as exc:
                        save(out/f"telemetry-{index:04d}.json", {"sampling_error": str(exc)})
                        index += 1
                stop.wait(1.)
        monitor = threading.Thread(target=telemetry, daemon=True)
        monitor.start()
        sessions = []
        for i, (filename, freeze) in enumerate(zip(packet["services"], freezes)):
            directory = out/f"session{i}"
            directory.mkdir()
            command = (sys.executable, "-B", str(root/"llm_intervention_model_service.py"),
                "--freeze", str(root/filename), "--freeze-sha256", packet["files"][filename],
                "--output", str(directory/"worker"))
            # Retain this object's process handle even if its handshake fails
            # inside __init__. The independent final watchdog can then clean
            # the owned group even if the transport has marked itself closed.
            current = PersistentPolicy.__new__(PersistentPolicy)
            current.__init__(command, 35., expected_identity=service.identity(freeze),
                output=directory/"client", startup_seconds=185., cleanup_seconds=5.,
                max_stderr_bytes=1024*1024, max_requests=4, task=tasks["case0.json"])
            with current:
                a = current.call(cases["case0.json"], 40)
                current.bind_task(tasks["case3.json"])
                current.choose(cases["case3.json"]["state"], directory/"episode1", 0)
                current.choose(cases["stress.json"]["state"], directory/"episode1", 1)
                current.bind_task(tasks["case0.json"])
                repeated = current.choose(cases["case0.json"]["state"], directory/"episode2", 0)
                if a != repeated:
                    raise ValueError("separated identical public state changed decision")
            gpu_cleanup = supervisor.gpu_process_sample(current.process.pid)
            save(directory/"post_close_gpu_sample.json", gpu_cleanup)
            require_gpu_sample(gpu_cleanup, active=False)
            validation = validate_session(directory, freeze_sha256=packet["files"][filename],
                identity_sha256=service.digest(service.identity(freeze)),
                envelope_pins=[service.digest(cases[n]) for n in ("case0.json", "case3.json", "stress.json", "case0.json")])
            sessions.append({"treatment": freeze["treatment"], "pid": current.process.pid, **validation})
            current = None
        stop.set()
        monitor.join(timeout=12.)
        if monitor.is_alive():
            raise RuntimeError("GPU sampler did not terminate")
        records = [json.loads(p.read_bytes()) for p in out.glob("telemetry-*.json")]
        for session in sessions:
            if not any(r.get("pid") == session["pid"] and r.get("returncode") == 0
                       and r.get("owned_rows") for r in records):
                raise ValueError("independent owned-process GPU telemetry missing")
        arm_final_deadline(start, None)
        save(out/"terminal.json", {"status": "complete", "packet_sha256": args.packet_sha256,
            "sessions": sessions, "new_training_updates": 0, "environment_responses": 0,
            "scientific_worlds": 0, "elapsed_seconds": time.monotonic()-start,
            "supervisor_process_cpu_seconds": time.process_time()-cpu,
            "scheduler_and_external_acceptance_required": True})
        completed = True
    except BaseException as exc:
        process = getattr(current, "process", None) if current is not None else None
        arm_final_deadline(start, process)
        if current is not None:
            try:
                current.close()
            except BaseException:
                pass  # Transport records its owned-group failure evidence.
        if process is not None and supervisor is not None:
            cleanup = supervisor.terminate(process, min(start+575., time.monotonic()+10.))
            save(out/"fallback_cleanup.json", cleanup)
        stop.set()
        if monitor is not None:
            monitor.join(timeout=12.)
        save(out/"failure.json", {"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "retry_allowed": False})
        raise
    finally:
        if completed:
            signal.setitimer(signal.ITIMER_REAL, 0.)


if __name__ == "__main__":
    main()
