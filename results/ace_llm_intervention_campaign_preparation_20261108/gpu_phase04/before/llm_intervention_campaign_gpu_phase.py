"""Three sequential campaign services under an already admitted GPU controller.

No staging, allocation/source/training admission, CPU controls, private scoring,
or final evidence qualification here. The outer controller supplies authenticated
modules and owns the subsequent runtime and dynamic trajectory validators.
Only live objects created by the scoped campaign factory authorize cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import stat
import threading
import time
import types

import llm_intervention_campaign as campaign_api
from llm_intervention_contract import digest, finite, integer
from llm_intervention_paired_episode import _pin, _raw, _sha

VERSION = "ace-campaign-gpu-phase-v1"
POLICIES = ("frozen_llm", "sft_llm", "dpo_llm")
FIRST_GLOBAL, FIRST_SERVICE, FRESHNESS = 240., 120., 90.
SESSION_RESERVE, SESSION_CAP = 120., 180.


class GPUPhaseInterrupted(BaseException):
    """Must pass through campaign.run_policy's ordinary-failure containment."""


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _save(path, value):
    with path.open("xb") as handle:
        handle.write(_raw(value))
        handle.flush()
        os.fsync(handle.fileno())


def _read(path, maximum=1024*1024):
    """Bounded live read; callers treat partial writes as uncertainty."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, "bounded regular evidence required")
        raw = handle.read(maximum+1)
    _need(len(raw) <= maximum, "live evidence exceeds bound")
    return raw


def _callbacks(module, names):
    # Pins below record the OUTER admission, not a claim that function metadata
    # authenticates code. Capture exact module-owned functions, never ambient
    # substitutes or callbacks resolved again while work is running.
    _need(type(module) is types.ModuleType, "outer-authenticated helper module required")
    result = []
    for name in names:
        callback = getattr(module, name, None)
        _need(type(callback) is types.FunctionType and callback.__name__ == name
              and callback.__globals__ is module.__dict__, "exact module-owned helper required: "+name)
        result.append(callback)
    return result


def _gpu_active(sample, pid):
    if type(sample) is not dict or sample.get("returncode") != 0 or type(sample.get("owned_rows")) is not list:
        return False
    try:
        return bool(sample["owned_rows"]) and all(
            len(parts := row.split(",")) == 3 and int(parts[0].strip()) == pid
            and parts[1].strip() and finite(float(parts[2])) and float(parts[2]) > 0
            for row in sample["owned_rows"])
    except (ValueError, TypeError, AttributeError):
        return False


def _gpu_empty(sample):
    return type(sample) is dict and sample.get("returncode") == 0 and sample.get("owned_rows") == []


def _activity(attempt, worlds):
    """Observed host work counts, not a second trajectory validator or timer."""
    counts = dict(initial_response_attempts=0, adaptive_response_attempts=0,
                  initial_returned_responses=0, adaptive_returned_responses=0)
    uncertain = []
    for index in range(worlds):
        path = attempt / f"world-{index:02d}/episode/journal.jsonl"
        if not path.exists():
            continue
        try:
            raw = _read(path, 8*1024*1024)
            lines = raw.splitlines()
            if not raw.endswith(b"\n"):
                lines = lines[:-1]
                uncertain.append(str(path)+": partial line")
            proposals, requests = {}, {}
            for line in lines:
                row = campaign_api._decode(line)
                if row["event"] == "proposal":
                    proposals[row["proposal_id"]] = row["provenance"]["kind"]
                elif row["event"] == "reserve":
                    phase = "initial" if proposals[row["proposal_id"]] == "shared_initial_schedule" else "adaptive"
                    requests[row["request_id"]] = phase
                    counts[phase+"_response_attempts"] += 1
                elif row["event"] == "response":
                    counts[requests[row["request_id"]]+"_returned_responses"] += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            uncertain.append(str(path)+": "+type(exc).__name__)
    snapshots, histories = 0, {}
    for path in sorted((attempt / "service").glob("request-*.input.json")):
        try:
            request = campaign_api._decode(_read(path))
            mapping = campaign_api._decode(_read(path.with_name(path.name.replace(".input.json", ".episode.json"))))
            state = request["envelope"]["state"]
            _need(type(state["scm"]) is list and type(state["history"]) is list, "public state incomplete")
            binding = mapping["task_binding_index"]
            histories[binding] = max(histories.get(binding, 0), len(state["history"]))
            snapshots += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            uncertain.append(str(path)+": "+type(exc).__name__)
    return {**counts, "public_scm_snapshots": snapshots, "history_rows_exported_to_policy": sum(histories.values()),
            "scm_update_calls": None, "scm_cpu_seconds": None,
            "scope": "observed journal reservations/returns and public SCM snapshots; update calls and CPU time not instrumented",
            "uncertainties": uncertain}


class _Watch:
    def __init__(self, *, pid, worker, attempt, output, outer_started, started, deadline, sample, worlds):
        self.pid, self.worker, self.attempt, self.output = pid, worker, attempt, output
        self.outer_started, self.started, self.deadline = outer_started, started, deadline
        self.sample, self.worlds = sample, worlds
        self.policy = self.process = None
        self.cleanup_confirmed = False
        self.first_verified = self.last_gpu = self.last_progress = None
        self.generations, self.fault, self.index = {}, None, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="campaign-gpu-watch-"+pid)
        self.last_activity = None

    def owned_process(self):
        process = getattr(self.policy, "process", None) if self.policy is not None else None
        if process is not None:
            if self.process is not None and self.process is not process:
                raise GPUPhaseInterrupted("registered process handle replaced")
            _need(integer(process.pid, 1), "invalid registered process handle")
            self.process = process
        return self.process

    def observe(self):
        now = time.monotonic()
        process, sample, uncertain = self.owned_process(), None, []
        if process is not None:
            alive = process.poll() is None
            sample = self.sample(process.pid)
            now = time.monotonic()  # include bounded helper latency
            if alive and _gpu_active(sample, process.pid):
                self.last_gpu = now
            else:
                uncertain.append("no current positive owned-process GPU observation")
        for path in sorted(self.worker.glob("generation-*.json")):
            try:
                raw = _read(path)
                row = campaign_api._decode(raw)
                ordinal = row["ordinal"]
                _need(integer(ordinal) and ordinal < 8 and path.name == f"generation-{ordinal:04d}.json"
                      and finite(row["cuda_interval_ms"]) and row["cuda_interval_ms"] > 0
                      and integer(row["peak_gpu_allocated_bytes"], 1)
                      and type(row["new_training_updates"]) is int and row["new_training_updates"] == 0
                      and row["tensor_versions_unchanged"] is True
                      and row["environment_callbacks_available"] is False
                      and type(row["prompt_ids"]) is list and type(row["sequence_ids"]) is list
                      and len(row["sequence_ids"]) > len(row["prompt_ids"]), "incomplete positive generation")
                if ordinal in self.generations:
                    if self.generations[ordinal]["sha256"] != _sha(raw):
                        self.fault = "completed generation bytes changed"
                elif ordinal == len(self.generations):
                    self.generations[ordinal] = {"sha256": _sha(raw), "cuda_interval_ms": row["cuda_interval_ms"],
                                                 "peak_gpu_allocated_bytes": row["peak_gpu_allocated_bytes"]}
                    self.last_progress = now
                else:
                    uncertain.append(path.name+": noncontiguous progress")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                uncertain.append(path.name+": "+type(exc).__name__)
        if self.first_verified is None and self.generations and self.last_gpu is not None and now-self.last_gpu < FRESHNESS:
            self.first_verified = now
        activity = _activity(self.attempt, self.worlds)
        _save(self.output / f"telemetry-{self.index:05d}.json", {
            "utc": datetime.now(timezone.utc).isoformat(), "monotonic": now,
            "outer_elapsed_seconds": now-self.outer_started, "service_elapsed_seconds": now-self.started,
            "policy_id": self.pid, "registered_pid": process.pid if process else None, "gpu_process_sample": sample,
            "positive_generations": len(self.generations), "first_verified_monotonic": self.first_verified,
            "last_gpu_monotonic": self.last_gpu, "last_generation_monotonic": self.last_progress,
            "adaptive_cpu_work": activity, "uncertainties": uncertain})
        self.index += 1
        self.last_activity = activity

    def run(self):
        try:
            while not self.stop.is_set():
                self.observe()
                self.stop.wait(1.)
        except BaseException as exc:
            self.fault = type(exc).__name__+": "+str(exc)


def run_gpu_phase(*, campaign, campaign_sha256, source_plan_pins, campaign_root,
                  services, supervision_root, outer_started, work_deadline, final_deadline,
                  helper, helper_sha256, deadline_helper, deadline_helper_sha256):
    """Run only frozen/SFT/DPO once, returning retained pins and observations.

    services maps each policy to exactly worker_output_dir/freeze_sha256/
    identity_sha256. The outer admits these fixed paths, shims, source closure,
    CPU acceptance/inventories and helper module bytes BEFORE calling. The passed
    helper supplies gpu_process_sample(pid), group_alive(pgid), terminate(Popen,
    absolute_deadline); deadline_helper supplies the reviewed
    arm_final_deadline(start, Popen_or_None) with its start+580 hard boundary.

    Absolute deadlines share the original outer monotonic epoch. Reserve 120s
    per remaining service, cap each at 180s, and require first work in 120s per
    service and 240s globally. A custom BaseException interrupts run_policy.
    Failure evidence is saved before re-raising; remaining policies are explicitly
    recorded unstarted when time permits. No retry or private finalization.

    Must run on the main thread without concurrent campaign calls. Prior signal
    state is restored after confirmed cleanup. If owned cleanup remains uncertain,
    the hard final watchdog remains armed for the caller's outer failure path.
    """
    _need(threading.current_thread() is threading.main_thread(), "main-thread signal ownership required")
    sample, group_alive, terminate = _callbacks(helper, ("gpu_process_sample", "group_alive", "terminate"))
    (arm_final,) = _callbacks(deadline_helper, ("arm_final_deadline",))
    _need(_pin(helper_sha256) and _pin(deadline_helper_sha256), "independent admitted helper pins required")
    entered = time.monotonic()
    _need(all(finite(t) for t in (outer_started, work_deadline, final_deadline))
          and outer_started <= entered < work_deadline <= outer_started+550
          and work_deadline+20 <= final_deadline <= outer_started+580, "absolute outer work/cleanup deadlines required")
    c, root = campaign_api._root(campaign, campaign_sha256, source_plan_pins, campaign_root)
    _need(tuple(p for p in c["policy_ids"] if c["policy_phases"][p] == "gpu") == POLICIES
          and set(c["services"]) == set(services) == set(POLICIES)
          and c["scope"] == "artificial" and len(c["sources"]) == 2 and len(c["policy_ids"]) == 8,
          "exact artificial two-world/eight-arm/three-service campaign required")
    out = Path(supervision_root).absolute()
    attempts = {pid: campaign_api._attempt(root, c, pid) for pid in POLICIES}
    workers = {}
    for pid, spec in services.items():
        _need(set(spec) == {"worker_output_dir", "freeze_sha256", "identity_sha256"}
              and _pin(spec["freeze_sha256"]) and _pin(spec["identity_sha256"])
              and digest(c["services"][pid]["identity"]) == spec["identity_sha256"]
              and c["services"][pid]["max_requests"] <= 8, "fixed service paths/pins/limit required")
        workers[pid] = Path(spec["worker_output_dir"]).absolute()
        _need(not workers[pid].exists() and not attempts[pid].exists(), "once-only service or attempt already exists")
    # Include already sealed CPU attempts: telemetry must not mutate their
    # inventories even though this function only executes the three GPU arms.
    all_attempts = [campaign_api._attempt(root, c, pid) for pid in c["policy_ids"]]
    roots = [out, *workers.values(), *all_attempts]
    resolved = [p.resolve() for p in roots]
    _need(len(set(resolved)) == len(resolved)
          and all(a not in b.parents for a in resolved for b in resolved if a != b), "supervision/worker/attempt roots overlap")
    _need(all(not p.is_symlink() for p in roots), "symlink output root")
    out.mkdir(exist_ok=False)
    _save(out / "start.json", {"schema": VERSION, "campaign_sha256": campaign_sha256,
        "source_plan_pins": source_plan_pins, "services": {p: {**services[p], "worker_output_dir": str(workers[p])} for p in POLICIES},
        "outer_started": outer_started, "phase_entered": entered, "work_deadline": work_deadline,
        "final_deadline": final_deadline, "helper_sha256": helper_sha256, "deadline_helper_sha256": deadline_helper_sha256,
        "session_reserve_seconds": SESSION_RESERVE, "session_cap_seconds": SESSION_CAP,
        "first_global_seconds": FIRST_GLOBAL, "first_service_seconds": FIRST_SERVICE, "freshness_seconds": FRESHNESS,
        "source_runtime_training_allocation_admission": "outer caller obligation", "retry_allowed": False})
    previous_class = campaign_api.PersistentPolicy
    signals = (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    handlers = {s: signal.getsignal(s) for s in signals}
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    old_due = entered+old_timer[0] if old_timer[0] else None
    current, first_global, failure, cleanup_safe, hard_armed = None, None, None, True, False
    pins, sessions, errors, not_recorded = {}, [], [], []

    def arm_cleanup():
        nonlocal hard_armed
        process = current.owned_process() if current is not None and not current.cleanup_confirmed else None
        arm_final(final_deadline-580., process)
        hard_armed = True

    def check():
        now = time.monotonic()
        first = first_global if first_global is not None else current.first_verified if current else None
        reason = None
        if now >= work_deadline or old_due is not None and now >= old_due:
            reason = "absolute work or inherited outer alarm deadline"
        elif (first is None and now >= outer_started+FIRST_GLOBAL) or first is not None and first > outer_started+FIRST_GLOBAL:
            reason = "first global GPU verification exceeded original outer 240 seconds"
        elif current is not None:
            if current.fault:
                reason = current.fault
            elif now >= current.deadline:
                reason = "service consumed its time reservation"
            elif current.first_verified is None and now >= current.started+FIRST_SERVICE:
                reason = "service first GPU work exceeded 120 seconds"
            elif current.first_verified is not None:
                if current.first_verified > current.started+FIRST_SERVICE:
                    reason = "service first GPU verification was late"
                elif now-current.last_gpu >= FRESHNESS:
                    reason = "owned GPU telemetry stale for 90 seconds; utilization uncertain"
                elif now-current.last_progress >= FRESHNESS:
                    reason = "no advancing positive GPU generation for 90 seconds"
        if reason:
            arm_cleanup()
            raise GPUPhaseInterrupted(reason)

    def interrupt(signum, frame):
        if signum != signal.SIGALRM:
            arm_cleanup()
            raise GPUPhaseInterrupted("external signal "+str(signum))
        check()
        signal.setitimer(signal.ITIMER_REAL, .5)

    def factory(*args, **kwargs):
        _need(current is not None and current.policy is None, "one registered policy object per service")
        _need(Path(kwargs["output"]).absolute() == attempts[current.pid] / "service"
              and digest(kwargs["expected_identity"]) == services[current.pid]["identity_sha256"], "factory launch binding differs")
        obj = previous_class.__new__(previous_class)
        current.policy = obj  # register BEFORE __init__, including failed handshakes
        obj.__init__(*args, **kwargs)
        return obj

    def retain(pid, result=None):
        path = attempts[pid] / "attempt.json"
        if not path.is_file():
            errors.append({"policy_id": pid, "error": "attempt receipt unavailable; preserve original directory"})
            return None
        pin = _sha(_read(path, 32*1024*1024))
        pins[pid] = pin
        _save(out / (pid+".attempt-pin.json"), {"policy_id": pid, "sha256": pin})
        if result is not None:
            _need(result == {"policy_id": pid, "sha256": pin}, "returned attempt pin differs")
        receipt, _ = campaign_api._load_attempt(c, campaign_sha256, attempts[pid], pid, pin)
        return receipt

    try:
        campaign_api.PersistentPolicy = factory
        for s in signals:
            signal.signal(s, interrupt)
        signal.setitimer(signal.ITIMER_REAL, .5)
        for index, pid in enumerate(POLICIES):
            check()
            now = time.monotonic()
            _need(work_deadline-now >= (len(POLICIES)-index)*SESSION_RESERVE,
                  "insufficient time reserved for all remaining services")
            folder = out / pid
            folder.mkdir(exist_ok=False)
            current = _Watch(pid=pid, worker=workers[pid], attempt=attempts[pid], output=folder,
                outer_started=outer_started, started=now,
                deadline=min(now+SESSION_CAP, work_deadline-(len(POLICIES)-index-1)*SESSION_RESERVE),
                sample=sample, worlds=len(c["sources"]))
            current.thread.start()
            result, receipt, post_empty, process_closed = None, None, False, False
            service_error = None
            try:
                result = campaign_api.run_policy(campaign=c, campaign_sha256=campaign_sha256,
                    source_plan_pins=source_plan_pins, output=root, policy_id=pid, phase="gpu")
            except BaseException as exc:
                service_error = exc
                arm_cleanup()
            finally:
                current.stop.set()
                current.thread.join(timeout=min(6., max(0., final_deadline-time.monotonic()-5)))
                if current.thread.is_alive():
                    service_error = service_error or GPUPhaseInterrupted("GPU sampler did not stop")
                else:
                    try:
                        current.observe()  # consume final completed records, never invent GPU evidence
                        check()
                    except BaseException as exc:
                        service_error = service_error or exc
                process = current.owned_process()
                try:
                    if process is not None:
                        if process.poll() is None or group_alive(process.pid):
                            arm_cleanup()
                            cleanup = terminate(process, min(time.monotonic()+10., final_deadline-5.))
                            _save(folder / "fallback-cleanup.json", cleanup)
                            service_error = service_error or GPUPhaseInterrupted("owned fallback cleanup was required")
                        process_closed = process.poll() is not None and group_alive(process.pid) is False
                        _need(process_closed, "owned process group closure unconfirmed")
                        current.cleanup_confirmed = True
                        # Dedicated observation AFTER process/group exit, separate from live samples.
                        post = sample(process.pid)
                        _save(folder / "post-exit-gpu.json", {"monotonic": time.monotonic(), "registered_pid": process.pid,
                              "process_group_closed": True, "sample": post})
                        post_empty = _gpu_empty(post)
                        _need(post_empty, "dedicated empty post-exit GPU observation missing")
                        _need(process.returncode == 0, "service process exited unsuccessfully")
                    else:
                        process_closed = True  # constructor never produced a process handle
                except BaseException as exc:
                    service_error = service_error or exc
                    arm_cleanup()
                cleanup_safe = process_closed and not current.thread.is_alive()
                try:
                    receipt = retain(pid, result)
                    _need(receipt is not None and not receipt["errors"]
                          and all(a["status"] == "completed" for a in receipt["arms"]), "campaign attempt contains failures")
                except BaseException as exc:
                    service_error = service_error or exc
                if current.first_verified is None:
                    service_error = service_error or GPUPhaseInterrupted("service lacks independently verified GPU generation")
                if first_global is None:
                    first_global = current.first_verified
                summary = {"policy_id": pid, "registered_pid": process.pid if process else None,
                    "attempt_sha256": pins.get(pid), "first_verified_monotonic": current.first_verified,
                    "started_monotonic": current.started, "closed_monotonic": time.monotonic(),
                    "process_group_closed": process_closed, "post_exit_gpu_empty": post_empty,
                    "positive_generations": len(current.generations), "generation_evidence": current.generations,
                    "cuda_interval_ms": sum(g["cuda_interval_ms"] for g in current.generations.values()),
                    "adaptive_cpu_work": current.last_activity,
                    "error": type(service_error).__name__+": "+str(service_error) if service_error else None}
                sessions.append(summary)
                _save(folder / "summary.json", summary)
            if service_error is not None:
                raise service_error
            _need(process_closed and post_empty, "service closure required before next policy")
            current = None  # retire handle before next service; never authorize a PID from saved metadata
        check()
    except BaseException as exc:
        failure = exc
        arm_cleanup()
        # Never retry a claimed attempt. Missing later policies can be recorded
        # unstarted through the normal campaign API, while the final timer stays armed.
        for pid in POLICIES:
            if attempts[pid].exists():
                continue
            if final_deadline-time.monotonic() < 5:
                not_recorded.append(pid)
                continue
            try:
                result = campaign_api.run_policy(campaign=c, campaign_sha256=campaign_sha256,
                    source_plan_pins=source_plan_pins, output=root, policy_id=pid, phase="gpu",
                    unstarted_reason="GPU phase stopped; no retry: "+type(exc).__name__)
                retain(pid, result)
            except BaseException as record_error:
                errors.append({"policy_id": pid, "error": type(record_error).__name__+": "+str(record_error)})
    finally:
        campaign_api.PersistentPolicy = previous_class
        # A failure in receipt I/O can bypass the per-service cleanup block.
        if current is not None:
            current.stop.set()
            try:
                process = current.owned_process()
                if process is not None and (process.poll() is None or group_alive(process.pid)):
                    arm_cleanup()
                    cleanup = terminate(process, min(time.monotonic()+10., final_deadline-3.))
                    _save(out / "last-resort-cleanup.json", cleanup)
                    cleanup_safe = (process.poll() is not None and group_alive(process.pid) is False
                                    and not current.thread.is_alive())
                if not cleanup_safe:
                    failure = failure or GPUPhaseInterrupted("owned cleanup or sampler termination remains uncertain")
                    arm_cleanup()
            except BaseException as cleanup_error:
                cleanup_safe = False
                failure = failure or cleanup_error
                errors.append({"phase": "last_resort_cleanup", "error": type(cleanup_error).__name__+": "+str(cleanup_error)})
                arm_cleanup()
        activity_fields = ("initial_response_attempts", "adaptive_response_attempts", "initial_returned_responses",
                           "adaptive_returned_responses", "public_scm_snapshots", "history_rows_exported_to_policy")
        phase_counts = {"observed_"+k: sum((s["adaptive_cpu_work"] or {}).get(k, 0) for s in sessions)
                        for k in activity_fields}
        phase_counts.update(services_started=sum(s["registered_pid"] is not None for s in sessions),
                            positive_generations=sum(s["positive_generations"] for s in sessions),
                            scm_update_calls=None, scm_cpu_seconds=None,
                            trajectory_accounting_validated=False)
        record = {"schema": VERSION, "status": "gpu_phase_complete" if failure is None else "gpu_phase_failed",
            "campaign_sha256": campaign_sha256, "attempt_pins": pins, "sessions": sessions, "phase_counts": phase_counts,
            "unstarted_not_recorded": not_recorded, "errors": errors, "retry_allowed": False,
            "failure": type(failure).__name__+": "+str(failure) if failure else None,
            "outer_elapsed_seconds": time.monotonic()-outer_started, "phase_elapsed_seconds": time.monotonic()-entered,
            "response_accounting": "per-session adaptive_cpu_work from original charged journals; not zero-response preparation",
            "new_training_updates": 0, "cpu_control_policies_run": 0, "private_scores_computed": False,
            "source_runtime_training_allocation_qualified": False, "dynamic_evidence_qualified": False,
            "scientific_acceptance": False, "signals_restored": cleanup_safe}
        try:
            _need(time.monotonic() < final_deadline, "absolute final deadline exceeded")
            _save(out / ("terminal.json" if failure is None else "failure.json"), record)
        finally:
            if cleanup_safe:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for s, handler in handlers.items():
                    signal.signal(s, handler)
                if old_due is not None:
                    signal.setitimer(signal.ITIMER_REAL, max(.001, old_due-time.monotonic()), old_timer[1])
            elif not hard_armed:
                arm_cleanup()
    if failure is not None:
        raise failure
    return record
