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
import re
import stat
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import types


VERSION = "ace-persistent-model-integration-v3"
CPU_SOURCE_SHA = "c827a53f0fe03204c447e8e7d406e6d8a2c9318b4d6cd5fdc82beb84b576b6b7"
GPU_HELPER_SHA = "ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec"
RESOURCES = {"account": "ucb736_asc2", "partition": "aa100", "qos": "gpu-testing",
    "requested_cpus": 4, "max_allocated_cpus": 8, "memory_mib": 12288,
    "wall_seconds": 600, "gpus": 1, "gpu_type": "a100_3g.20gb"}
FIRST_GPU_SECONDS = 240.


def bounded_input(path, pin, maximum):
    if type(pin) is not str or re.fullmatch('[0-9a-f]{64}', pin) is None:
        raise ValueError("independent input pin required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("regular bounded input required")
        raw = handle.read(maximum+1)
        after = os.fstat(handle.fileno())
    if (len(raw) > maximum or sha(raw) != pin or
        (before.st_ino, before.st_size, before.st_mtime_ns) !=
        (after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("input changed or independent pin differs")
    return raw


def module_from_bytes(name, path, raw):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def packet_shape(packet, cpu_inputs):
    expected = set(cpu_inputs) | {"qualify_llm_intervention_model_service.py",
        "supervise_llm_intervention_checkpoint_gpu.py", "cpu-preflight.json", "cpu-acceptance.json"}
    if (type(packet) is not dict or set(packet) != {"schema", "files", "services", "resources"}
        or packet['schema'] != VERSION or packet['services'] != ['frozen.json', 'adapter.json']
        or json.dumps(packet['resources'], sort_keys=True) != json.dumps(RESOURCES, sort_keys=True)
        or type(packet['files']) is not dict or set(packet['files']) != expected
        or any(type(p) is not str or re.fullmatch('[0-9a-f]{64}', p) is None for p in packet['files'].values())):
        raise ValueError("complete prospective GPU packet contract differs")


def check_preflight(acceptance, packet, preflight, cpu):
    # Pins bind a separately reviewed CPU result, not proof inferred from metadata.
    if (acceptance.get('status') != 'cpu_service_preflight_accepted'
        or acceptance.get('job_id') != '35348366'
        or acceptance.get('preflight_sha256') != packet['files']['cpu-preflight.json']
        or acceptance.get('cpu_preflight_qualified') is not True
        or acceptance.get('model_service_qualified') is not False
        or acceptance.get('scientific_qualified') is not False
        or any(packet['files'][name] != pin for name, pin in preflight['files'].items())
        or set(preflight['files']) != set(cpu.INPUTS)):
        raise ValueError("accepted CPU preflight or exact input lineage differs")


def service_command(prefix, root, filename, files, directory, stage_path, stage_pin, bootstrap):
    return (str(prefix/'bin/python'), '-I', '-B', '-c', bootstrap,
        str(root/'llm_intervention_model_service.py'), files['llm_intervention_model_service.py'],
        '--freeze', str(root/filename), '--freeze-sha256', files[filename],
        '--output', str(directory/'worker'), '--runtime-stage', str(stage_path),
        '--runtime-stage-sha256', stage_pin)


def child_environment(prefix, inherited):
    env = dict(inherited)
    for name in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'MKL_SERVICE_FORCE_INTEL'):
        env.pop(name, None)
    if not env.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError("scheduler GPU visibility must be preserved, not fabricated")
    env.update(LD_LIBRARY_PATH=str(prefix/'lib'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
        MKL_THREADING_LAYER='GNU')
    return env


def install_child_environment(environment):
    """Set this owned launcher's C environment before its fresh child exec.

    NumPy may have set C variables absent from Python's os.environ mapping.
    Clear disallowed C variables explicitly, then overwrite the threading layer.
    This does not change libraries already loaded in the parent process.
    """
    import ctypes
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    keys = ('MKL_THREADING_LAYER', 'MKL_SERVICE_FORCE_INTEL', 'CUDA_VISIBLE_DEVICES')
    def snapshot():
        c = {}
        for name in keys:
            value = libc.getenv(name.encode())
            if value is not None and len(value)>8192:
                raise ValueError('bounded child environment required')
            c[name] = None if value is None else value.decode('utf-8')
        return {'python':{n:os.environ.get(n) for n in keys},'c':c}
    before = snapshot()
    if (environment.get('MKL_THREADING_LAYER')!='GNU'
        or any(n in environment for n in ('MKL_SERVICE_FORCE_INTEL','PYTHONHOME','PYTHONPATH','LD_PRELOAD'))
        or not environment.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('explicit GNU child environment contract differs')
    for name in ('MKL_SERVICE_FORCE_INTEL','PYTHONHOME','PYTHONPATH','LD_PRELOAD'):
        os.unsetenv(name)
    os.environ.clear()
    os.environ.update(environment)
    after = snapshot()
    expected = {n:environment.get(n) for n in keys}
    if after['python']!=expected or after['c']!=expected:
        raise ValueError('Python/C child environment handoff differs')
    return {'before':before,'after':after,'loaded_parent_libraries_changed':False}


def require_session_time(start, now):
    if now-start >= 310.:
        raise TimeoutError("less than 250 seconds remain in the original GPU deadline")


def validate_runtime(directory, freeze, files, stage, stage_pin, cpu, service, diagnostic, checks, provenance):
    worker = Path(directory)/'worker'
    def read(name):
        with (worker/name).open('rb') as handle:
            raw = handle.read(16*1024**2+1)
        if len(raw) > 16*1024**2:
            raise ValueError("bounded runtime receipt exceeded")
        return service.decode(raw)
    # Reuse the CPU baseline/metadata/cache/provenance checks. The second
    # observation is explicitly mapped to each GPU phase, never relabeled on disk.
    for phase in ('after-model-load', 'after-requests'):
        def mapped(name):
            return read('runtime-'+phase+'.json' if name == 'runtime-after-cpu-tokenization.json' else name)
        cpu.validate_runtime_records(mapped, stage, stage_pin, service, diagnostic, checks, provenance)
    expected = {Path(n).stem: pin for n,pin in freeze['sources'].items() if n != 'llm_intervention_model_service.py'}
    for phase in ('after-service-imports', 'after-model-load', 'after-requests'):
        projects = read('runtime-'+phase+'.json')['captured_project_modules']
        if (not set(expected) <= set(projects) or not set(projects) <= set(expected)|{'__main__','__mp_main__'}
            or any(projects[n]['sha256'] != pin for n,pin in expected.items())):
            raise ValueError("GPU service project import closure differs")
    auth = read('authenticated.json')
    if (auth['runtime'] != service.RUNTIME or set(auth['models']) != service.MODEL_NAMES
        or any(row['sha256'] != freeze['model_files'][n] or type(row['bytes']) is not int or row['bytes'] <= 0
               for n,row in auth['models'].items())
        or read('terminal.json')['runtime_stage_sha256'] != stage_pin):
        raise ValueError("GPU model input or runtime-stage binding differs")
    return {'runtime_observations': ['after-service-imports','after-model-load','after-requests'],
            'stage_sha256': stage_pin, 'actual_imports_qualified': True}


def authenticate_allocation(raw, job_id, resources, helper):
    scheduler_fields, memory_mib = helper.scheduler_fields, helper.memory_mib
    expected = {"account": "ucb736_asc2", "partition": "aa100", "qos": "gpu-testing",
                "requested_cpus": 4, "max_allocated_cpus": 8, "memory_mib": 12288,
                "wall_seconds": 600, "gpus": 1, "gpu_type": "a100_3g.20gb"}
    if resources != expected or not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("altered resource request or invalid job identity")
    fields = scheduler_fields(raw)
    for name, wanted in {"JobId": job_id, "Account": "ucb736_asc2", "Partition": "aa100",
                         "QOS": "gpu-testing", "JobState": "RUNNING", "NumNodes": "1",
                         "NumTasks": "1", "TimeLimit": "00:10:00", "Requeue": "0", "Restarts": "0"}.items():
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
    signal.setitimer(signal.ITIMER_REAL, FIRST_GPU_SECONDS)
    try:
        cpu_raw = bounded_input(root/'qualify_llm_intervention_service_cpu.py', CPU_SOURCE_SHA, 128*1024)
        cpu_checks = module_from_bytes('pinned_cpu_checks', root/'qualify_llm_intervention_service_cpu.py', cpu_raw)
        raw = bounded_input(args.packet, args.packet_sha256, 128*1024)
        packet = json.loads(raw)
        packet_shape(packet, cpu_checks.INPUTS)
        data = {n: bounded_input(root/n, pin, 8*1024**2) for n,pin in packet['files'].items()}
        if (packet['files']['qualify_llm_intervention_service_cpu.py'] != CPU_SOURCE_SHA
            or packet['files']['supervise_llm_intervention_checkpoint_gpu.py'] != GPU_HELPER_SHA
            or Path(__file__).absolute() != root/'qualify_llm_intervention_model_service.py'):
            raise ValueError("independent helper pin or entrypoint differs")
        service = module_from_bytes('pinned_service', root/'llm_intervention_model_service.py', data['llm_intervention_model_service.py'])
        diagnostic = module_from_bytes('pinned_diagnostic', root/'qualify_llm_intervention_runtime_metadata.py', data['qualify_llm_intervention_runtime_metadata.py'])
        preflight = service.decode(data['cpu-preflight.json'])
        freezes = cpu_checks.authenticate(root, preflight, {n:data[n] for n in cpu_checks.INPUTS}, service, diagnostic)
        check_preflight(service.decode(data['cpu-acceptance.json']), packet, preflight, cpu_checks)
        runtime_checks = module_from_bytes('pinned_runtime_checks', root/'llm_intervention_service_runtime.py', data['llm_intervention_service_runtime.py'])
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
            "validated": authenticate_allocation(scheduler, job_id, packet["resources"], supervisor)})
        scratch = Path(os.environ.get('SLURM_SCRATCH',''))
        if not scratch.is_absolute():
            raise ValueError('absolute node scratch required')
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir() or scratch.stat().st_uid != os.getuid():
            raise ValueError('owned node scratch required')
        local = scratch/('ace-service-gpu-'+job_id)
        local.mkdir(mode=0o700, exist_ok=False)
        helper = module_from_bytes('pinned_archive_helper', root/'llm_intervention_runtime_archive.py', data['llm_intervention_runtime_archive.py'])
        event = diagnostic.events(out, start, cpu, 512*1024**2)
        event('stage_start')
        stage = diagnostic.stage_runtime(preflight, helper, local/'runtime', event)
        stage_path = out/'stage.json'
        save(stage_path, stage)
        stage_pin = sha(stage_path.read_bytes())
        event('stage_complete', staged_prefix=stage['staged_prefix'])
        prefix = Path(stage['staged_prefix'])
        provenance = diagnostic.provenance_contract(prefix)
        save(out/'independent-provenance-sources.json', provenance)
        # PersistentPolicy inherits this environment. Keep scheduler GPU visibility.
        environment = child_environment(prefix, os.environ)
        save(out/'child-environment.json', install_child_environment(environment))
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
            require_session_time(start, time.monotonic())
            command = service_command(prefix, root, filename, packet['files'], directory,
                                      stage_path, stage_pin, diagnostic.BOOTSTRAP)
            event('service_start', ordinal=i, manifest=filename)
            # Retain this object's process handle even if its handshake fails
            # inside __init__. The independent final watchdog can then clean
            # the owned group even if the transport has marked itself closed.
            current = PersistentPolicy.__new__(PersistentPolicy)
            current.__init__(command, 35., expected_identity=service.identity(freeze),
                output=directory/"client", startup_seconds=185., cleanup_seconds=5.,
                max_stderr_bytes=1024*1024, max_requests=4, task=tasks["case0.json"])
            with current:
                a = current.call(cases["case0.json"], 40)
                if i == 0:
                    sample = supervisor.gpu_process_sample(current.process.pid)
                    save(directory/'first_gpu_sample.json', sample)
                    require_gpu_sample(sample, active=True)
                    if time.monotonic()-start >= FIRST_GPU_SECONDS:
                        raise TimeoutError('first verified GPU work missed startup limit')
                    event('first_gpu_work', pid=current.process.pid)
                    signal.setitimer(signal.ITIMER_REAL, max(.01, start+560.-time.monotonic()))
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
            runtime_validation = validate_runtime(directory, freeze, packet['files'], stage, stage_pin,
                cpu_checks, service, diagnostic, runtime_checks, provenance)
            sessions.append({"treatment": freeze["treatment"], "pid": current.process.pid,
                             **validation, **runtime_validation})
            event('service_complete', ordinal=i, pid=current.process.pid)
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
        save(out/"terminal.json", {"schema": VERSION, "status": "complete", "packet_sha256": args.packet_sha256,
            "stage_sha256": stage_pin, "cpu_preflight_sha256": packet["files"]["cpu-preflight.json"],
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
