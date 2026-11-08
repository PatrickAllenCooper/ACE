"""New GPU-phase control fixtures: no subprocess, signals, models or responses.

Run only this module. An external invocation harness retains these exact sources,
command, stdout/stderr and measured child CPU; all fixture attempts are preserved.
"""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import llm_intervention_campaign_gpu_phase as phase


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(phase._raw(value))


class Process:
    def __init__(self, pid):
        self.pid, self.returncode, self.group = pid, None, True

    def poll(self):
        return self.returncode


class Harness:
    def __init__(self, path, mode="success", now=0.):
        self.path, self.mode, self.now = path, mode, now
        self.calls, self.samples, self.terminations, self.hard_arms, self.watches = [], [], [], [], []
        self.processes, self.before_init, self.handlers = {}, [], {}
        self.timer, self.bad_gpu = (0., 0.), False
        self.original_handlers = {s: object() for s in (phase.signal.SIGALRM, phase.signal.SIGTERM, phase.signal.SIGINT)}
        self.handlers.update(self.original_handlers)
        self.root = path / "campaign"
        self.root.mkdir(parents=True)
        ids = ["cpu-"+str(i) for i in range(5)] + list(phase.POLICIES)
        self.campaign = {"scope": "artificial", "sources": [{}, {}], "policy_ids": ids,
            "policy_phases": {p: "gpu" if p in phase.POLICIES else "cpu" for p in ids},
            "services": {p: {"identity": {"fixture": p}, "max_requests": 8} for p in phase.POLICIES}}
        self.services = {p: {"worker_output_dir": path / "workers" / p,
                            "freeze_sha256": phase.digest(["freeze", p]),
                            "identity_sha256": phase.digest(self.campaign["services"][p]["identity"])} for p in phase.POLICIES}
        self.helper = types.ModuleType("fabricated_authenticated_gpu_helper")
        self.helper.harness = self
        exec("def gpu_process_sample(pid): return harness.sample(pid)\n"
             "def group_alive(pgid): return harness.group_alive(pgid)\n"
             "def terminate(process, deadline): return harness.terminate(process, deadline)\n", self.helper.__dict__)
        self.deadline = types.ModuleType("fabricated_authenticated_deadline_helper")
        self.deadline.harness = self
        exec("def arm_final_deadline(start, process): harness.hard_arms.append((start, process))\n", self.deadline.__dict__)
        h = self

        class FakePolicy:
            def __init__(self, *args, **kwargs):
                watch = h.watches[-1]
                h.before_init.append(watch.policy is self)
                self.process = Process(700+len(h.processes))
                self.closed = False
                h.processes[self.process.pid] = self.process
                Path(kwargs["output"]).mkdir(parents=True)
                # Deliberately hostile saved PID is never a cleanup authority.
                put(Path(kwargs["output"]) / "process.json", {"pid": 999999, "pgid": 999999})
                watch.worker.mkdir(parents=True)
                if h.mode in ("init_failure", "init_interrupt"):
                    if h.mode == "init_interrupt":
                        raise phase.GPUPhaseInterrupted("fixture initialization interrupt")
                    raise RuntimeError("fixture failed handshake")
                h.now += 13.
                watch.observe()

        class FakeThread:
            def __init__(self, *, target, **kwargs):
                self.watch = target.__self__
                h.watches.append(self.watch)

            def start(self):
                pass

            def join(self, timeout):
                pass

            def is_alive(self):
                return False

        self.policy_class, self.thread_class = FakePolicy, FakeThread
        self.args = dict(campaign=self.campaign, campaign_sha256="a"*64, source_plan_pins=[], campaign_root=self.root,
            services=self.services, supervision_root=path/"supervision", outer_started=0., work_deadline=550., final_deadline=580.,
            helper=self.helper, helper_sha256="b"*64, deadline_helper=self.deadline, deadline_helper_sha256="c"*64)

    def sample(self, pid):
        self.samples.append(pid)
        process = self.processes[pid]
        if process.returncode is not None:
            if self.mode == "post_exit_unknown":
                return {"returncode": None, "owned_rows": [], "error": "fixture nvidia-smi unavailable"}
            return {"returncode": 0, "owned_rows": []}
        if self.bad_gpu or self.mode == "no_gpu":
            return {"returncode": None, "owned_rows": [], "error": "fixture unknown telemetry"}
        if self.mode == "wrong_pid":
            return {"returncode": 0, "owned_rows": ["999999, GPU-fixture, 1024"]}
        return {"returncode": 0, "owned_rows": [f"{pid}, GPU-fixture, 1024"]}

    def terminate(self, process, deadline):
        self.terminations.append((process, deadline))
        process.returncode, process.group = -15, False
        return {"pgid": process.pid, "leader_returncode": -15, "group_still_exists": False,
                "term_sent": True, "kill_sent": False, "errors": []}

    def group_alive(self, pgid):
        if self.mode == "group_query_failure":
            raise PermissionError("fixture owned-group observation unavailable")
        return self.processes[pgid].group

    def alarm(self, which=None):
        sig = phase.signal.SIGALRM if which is None else which
        self.handlers[sig](sig, None)

    def generation(self, ordinal=0):
        watch = self.watches[-1]
        put(watch.worker / f"generation-{ordinal:04d}.json", {"ordinal": ordinal, "cuda_interval_ms": 2.,
            "peak_gpu_allocated_bytes": 1024, "new_training_updates": 0, "tensor_versions_unchanged": True,
            "environment_callbacks_available": False, "prompt_ids": [1], "sequence_ids": [1, 2, 3]})

    def activity(self, attempt):
        events = [
            {"event": "proposal", "proposal_id": 0, "provenance": {"kind": "shared_initial_schedule"}},
            {"event": "reserve", "proposal_id": 0, "request_id": 0},
            {"event": "response", "request_id": 0},
            {"event": "proposal", "proposal_id": 1, "provenance": {"kind": "process_policy"}},
            {"event": "reserve", "proposal_id": 1, "request_id": 1},
            {"event": "response", "request_id": 1}]
        p = attempt / "world-00/episode/journal.jsonl"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"".join(phase._raw(e) for e in events))
        put(attempt / "service/request-0000.input.json", {"envelope": {"state": {"scm": [], "history": [{}, {}]}}})
        put(attempt / "service/request-0000.episode.json", {"task_binding_index": 0})

    def run_policy(self, **kwargs):
        pid = kwargs["policy_id"]
        self.calls.append((pid, kwargs.get("unstarted_reason")))
        assert kwargs["phase"] == "gpu" and pid in phase.POLICIES
        attempt = phase.campaign_api._attempt(self.root, self.campaign, pid)
        attempt.mkdir(exist_ok=False)
        if kwargs.get("unstarted_reason"):
            receipt = {"arms": [{"status": "unstarted"}]*2, "errors": []}
        else:
            obj, caught = None, None
            try:
                obj = phase.campaign_api.PersistentPolicy(("never-executed",), 35., output=attempt/"service",
                    expected_identity=self.campaign["services"][pid]["identity"], startup_seconds=125., cleanup_seconds=5., max_requests=8)
                self.activity(attempt)
                if self.mode == "service_startup_timeout":
                    self.now = self.watches[-1].started+121.
                    self.watches[-1].observe()
                    self.alarm()
                if self.mode == "global_startup_timeout":
                    self.now = 241.
                else:
                    self.now += 1.
                if self.mode == "partial":
                    (self.watches[-1].worker / "generation-0000.json").write_bytes(b'{"ordinal":')
                    self.watches[-1].observe()
                    assert self.watches[-1].first_verified is None
                    self.now += 1.
                self.generation()
                self.watches[-1].observe()
                if self.mode in ("stale_gpu", "stale_progress"):
                    self.bad_gpu = self.mode == "stale_gpu"
                    self.now += 91.
                    self.watches[-1].observe()
                if self.mode == "changed_generation":
                    path = self.watches[-1].worker / "generation-0000.json"
                    value = json.loads(path.read_bytes())
                    value["cuda_interval_ms"] = 3.
                    put(path, value)
                    self.watches[-1].observe()
                if self.mode == "sigterm":
                    self.alarm(phase.signal.SIGTERM)
                self.alarm()
            except BaseException as exc:
                caught = exc
            finally:
                if obj is not None:
                    obj.process.returncode, obj.process.group, obj.closed = (1 if self.mode == "nonzero_exit" else 0), False, True
            receipt = {"arms": [{"status": "failed" if caught else "completed"}]*2,
                       "errors": [str(caught)] if caught else []}
        put(attempt / "attempt.json", receipt)
        if not kwargs.get("unstarted_reason") and caught is not None and not isinstance(caught, Exception):
            raise caught  # mirrors campaign's BaseException propagation
        return {"policy_id": pid, "sha256": phase._sha((attempt/"attempt.json").read_bytes())}

    def patches(self):
        def handler(sig, value):
            old, self.handlers[sig] = self.handlers[sig], value
            return old

        def timer(which, seconds, interval=0.):
            old, self.timer = self.timer, (seconds, interval)
            return old

        return [patch.object(phase.time, "monotonic", side_effect=lambda: self.now),
            patch.object(phase.threading, "Thread", self.thread_class),
            patch.object(phase.signal, "signal", side_effect=handler),
            patch.object(phase.signal, "getsignal", side_effect=lambda sig: self.handlers[sig]),
            patch.object(phase.signal, "getitimer", side_effect=lambda _: self.timer),
            patch.object(phase.signal, "setitimer", side_effect=timer),
            patch.object(phase.campaign_api, "PersistentPolicy", self.policy_class),
            patch.object(phase.campaign_api, "_root", return_value=(self.campaign, self.root)),
            patch.object(phase.campaign_api, "run_policy", side_effect=self.run_policy),
            patch.object(phase.campaign_api, "_load_attempt", side_effect=lambda c, cp, path, pid, pin:
                         (json.loads((path/"attempt.json").read_bytes()), None))]

    def invoke(self, **overrides):
        patches = self.patches()
        for p in patches:
            p.start()
        try:
            return phase.run_gpu_phase(**dict(self.args, **overrides))
        finally:
            assert phase.campaign_api.PersistentPolicy is self.policy_class, "factory not restored"
            for p in reversed(patches):
                p.stop()


class GPUPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("ACE_CAMPAIGN_GPU_PHASE_TEST_OUTPUT")
        cls.root = Path(explicit) if explicit else Path(tempfile.mkdtemp(prefix="ace-campaign-gpu-phase-"))
        if explicit:
            cls.root.mkdir(exist_ok=False)
        print("Retained fabricated GPU-phase controls:", cls.root, flush=True)

    def fixture(self, mode="success", now=0.):
        return Harness(self.root / self._testMethodName / mode, mode=mode, now=now)

    def test_three_sequential_services_registered_before_init_and_counts(self):
        h = self.fixture()
        result = h.invoke()
        self.assertEqual(result["status"], "gpu_phase_complete")
        self.assertEqual([p for p, _ in h.calls], list(phase.POLICIES))
        self.assertTrue(all(h.before_init))
        self.assertEqual(set(result["attempt_pins"]), set(phase.POLICIES))
        self.assertEqual(h.terminations, [])
        self.assertTrue(all(s["post_exit_gpu_empty"] for s in result["sessions"]))
        self.assertEqual([s["registered_pid"] for s in result["sessions"]], [700, 701, 702])
        self.assertNotIn(999999, h.samples)
        for session in result["sessions"]:
            self.assertEqual(session["positive_generations"], 1)
            self.assertEqual(session["adaptive_cpu_work"]["adaptive_returned_responses"], 1)
            self.assertEqual(session["adaptive_cpu_work"]["initial_returned_responses"], 1)
            self.assertEqual(session["adaptive_cpu_work"]["history_rows_exported_to_policy"], 2)
            self.assertIsNone(session["adaptive_cpu_work"]["scm_cpu_seconds"])
        self.assertEqual(h.handlers, h.original_handlers)
        self.assertFalse(any((h.root/f"policy-{i:02d}").exists() for i in range(5)))

    def test_failed_init_retains_actual_object_and_only_its_group_is_cleaned(self):
        for mode in ("init_failure", "init_interrupt"):
            h = self.fixture(mode)
            with self.assertRaises(BaseException):
                h.invoke()
            self.assertEqual(h.before_init, [True])
            self.assertEqual(len(h.terminations), 1)
            process, deadline = h.terminations[0]
            self.assertIs(process, h.watches[0].policy.process)
            self.assertEqual(process.pid, 700)
            self.assertLessEqual(deadline, 575.)
            self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])
            failure = json.loads((h.path/"supervision/failure.json").read_bytes())
            self.assertEqual(set(failure["attempt_pins"]), set(phase.POLICIES))
            self.assertTrue((h.path/"supervision/frozen_llm/post-exit-gpu.json").exists())
            self.assertEqual(h.handlers, h.original_handlers)

    def test_baseexception_interrupt_and_signal_state_restored(self):
        self.assertTrue(issubclass(phase.GPUPhaseInterrupted, BaseException))
        self.assertFalse(issubclass(phase.GPUPhaseInterrupted, Exception))
        h = self.fixture("sigterm")
        with self.assertRaisesRegex(phase.GPUPhaseInterrupted, "external signal"):
            h.invoke()
        self.assertTrue(h.hard_arms)
        self.assertTrue(all(start+580. <= 580. for start, _ in h.hard_arms))
        self.assertEqual(h.handlers, h.original_handlers)
        self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])

    def test_first_work_deadlines_are_global_and_per_service(self):
        for mode, now, match in (("global_startup_timeout", 150., "original outer 240"),
                                 ("service_startup_timeout", 0., "120 seconds")):
            h = self.fixture(mode, now=now)
            with self.assertRaisesRegex(phase.GPUPhaseInterrupted, match):
                h.invoke()
            self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])

    def test_real_generation_and_independent_owned_process_are_both_required(self):
        for mode in ("no_gpu", "wrong_pid"):
            h = self.fixture(mode)
            with self.assertRaisesRegex(phase.GPUPhaseInterrupted, "independently verified"):
                h.invoke()
            self.assertIsNone(h.watches[0].first_verified)
            self.assertEqual(len(h.watches[0].generations), 1)

    def test_fresh_gpu_and_advancing_generation_guards(self):
        for mode, match in (("stale_gpu", "telemetry stale"), ("stale_progress", "advancing positive"),
                             ("changed_generation", "bytes changed")):
            h = self.fixture(mode)
            with self.assertRaisesRegex(phase.GPUPhaseInterrupted, match):
                h.invoke()
            self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])

    def test_partial_live_record_is_uncertainty_then_real_progress(self):
        h = self.fixture("partial")
        result = h.invoke()
        self.assertEqual(result["status"], "gpu_phase_complete")
        rows = [json.loads(p.read_bytes()) for p in (h.path/"supervision/frozen_llm").glob("telemetry-*.json")]
        self.assertTrue(any(r["positive_generations"] == 0 and any("generation-0000" in x for x in r["uncertainties"]) for r in rows))
        self.assertTrue(any(r["positive_generations"] == 1 for r in rows))

    def test_time_reserved_for_three_sessions_and_no_restart(self):
        h = self.fixture(now=200.)
        with self.assertRaisesRegex(ValueError, "all remaining services"):
            h.invoke()
        self.assertEqual(len(h.watches), 0)
        self.assertEqual([bool(reason) for _, reason in h.calls], [True, True, True])
        before = list(h.calls)
        with self.assertRaisesRegex(ValueError, "once-only"):
            h.invoke()
        self.assertEqual(h.calls, before)

    def test_dedicated_postexit_observation_cannot_be_unknown(self):
        h = self.fixture("post_exit_unknown")
        with self.assertRaisesRegex(ValueError, "post-exit"):
            h.invoke()
        sample = json.loads((h.path/"supervision/frozen_llm/post-exit-gpu.json").read_bytes())
        self.assertTrue(sample["process_group_closed"])
        self.assertIsNone(sample["sample"]["returncode"])
        self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])

    def test_deadline_and_callback_admission_do_not_launch(self):
        h = self.fixture()
        for overrides in ({"work_deadline": 551.}, {"final_deadline": 581.}, {"final_deadline": 555.},
                          {"helper_sha256": "not-a-pin"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                h.invoke(**overrides)
        h.helper.gpu_process_sample = lambda pid: {"returncode": 0, "owned_rows": []}
        with self.assertRaisesRegex(ValueError, "module-owned"):
            h.invoke()
        self.assertEqual(h.calls, [])

    def test_cleanup_query_failure_is_preserved_with_final_watchdog(self):
        h = self.fixture("group_query_failure")
        with self.assertRaisesRegex(PermissionError, "owned-group"):
            h.invoke()
        failure = json.loads((h.path/"supervision/failure.json").read_bytes())
        self.assertEqual(failure["status"], "gpu_phase_failed")
        self.assertFalse(failure["signals_restored"])
        self.assertTrue(any(e.get("phase") == "last_resort_cleanup" for e in failure["errors"]))
        self.assertTrue(h.hard_arms)
        self.assertEqual(set(failure["attempt_pins"]), set(phase.POLICIES))

    def test_nonzero_actual_exit_cannot_be_upgraded_by_clean_attempt_metadata(self):
        h = self.fixture("nonzero_exit")
        with self.assertRaisesRegex(ValueError, "exited unsuccessfully"):
            h.invoke()
        failure = json.loads((h.path/"supervision/failure.json").read_bytes())
        self.assertFalse(failure["sessions"][0]["error"] is None)
        self.assertEqual([bool(reason) for _, reason in h.calls], [False, True, True])

    def test_phase_counts_and_shorter_explicit_final_boundary(self):
        h = self.fixture()
        result = h.invoke(work_deadline=500., final_deadline=530.)
        counts = result["phase_counts"]
        self.assertEqual(counts["positive_generations"], 3)
        self.assertEqual(counts["observed_adaptive_returned_responses"], 3)
        self.assertEqual(counts["observed_initial_returned_responses"], 3)
        self.assertEqual(counts["observed_public_scm_snapshots"], 3)
        self.assertFalse(counts["trajectory_accounting_validated"])
        h = self.fixture("sigterm")
        with self.assertRaises(phase.GPUPhaseInterrupted):
            h.invoke(work_deadline=500., final_deadline=530.)
        self.assertTrue(all(start+580. == 530. for start, _ in h.hard_arms))


if __name__ == "__main__":
    unittest.main()
