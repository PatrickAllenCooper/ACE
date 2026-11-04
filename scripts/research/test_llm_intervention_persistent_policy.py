"""Tiny artificial JSONL workers only; all test attempts and evidence retained."""
from dataclasses import asdict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from llm_intervention_contract import Action, Node, VERSION as STATE_VERSION, canonical, digest
from llm_intervention_episode import ProcessPolicy
from llm_intervention_persistent_policy import PersistentPolicy, VERSION, cleanup_process_group, serve_policy
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask, make_envelope


RECIPE = {"kind": "artificial_literal_menu", "numeric_dtype": "float64"}
IDENTITY = {name+"_sha256": hashlib.sha256(("artificial-"+name).encode()).hexdigest()
            for name in ("model", "source", "adapter")}
IDENTITY.update(numeric_recipe=RECIPE, numeric_recipe_sha256=digest(RECIPE))
GRAPH = (Node("X", (), "unit", (-2., 2.)), Node("Y", ("X",), "unit"))
MENU = (Action("a0", "intervene", (("X", 0.),), ("Y",), 1),
        Action("a1", "intervene", (("X", 1.),), ("Y",), 1))
TASK = PredictionTask("Y", (MENU[0],), (1.,), 1., .04)


def envelope(note="first"):
    learner = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})
    state = json.loads(canonical({"version": STATE_VERSION, "objective": "artificial prediction",
        "domain_notes": note, "graph": [asdict(n) for n in GRAPH], "menu": [asdict(a) for a in MENU],
        "history": [], "scm": [asdict(s) for s in learner.summaries()], "spent_units": 0,
        "remaining_units": 2, "requests": 0, "returned_responses": 0, "terminal": False}))
    return make_envelope(state, TASK)


def worker(mode):
    def initialize():
        print("ARTIFICIAL_INIT")
        def decide(public):
            print("ARTIFICIAL_DECISION")
            if mode == "bad-completion": return '{"action_id":"missing"}'
            chosen = "a1" if public["state"]["domain_notes"] == "second" else "a0"
            return ' {"action_id": "'+chosen+'"} '
        return decide
    if mode in ("normal", "bad-completion", "bad-handshake"):
        identity = copy.deepcopy(IDENTITY)
        if mode == "bad-handshake": identity["model_sha256"] = "0"*64
        serve_policy(initialize, identity=identity)
        return
    if mode == "startup-timeout":
        time.sleep(10)
        return
    def emit(value):
        sys.stdout.buffer.write((canonical(value)+"\n").encode())
        sys.stdout.buffer.flush()
    emit({"type": "ready", "version": VERSION, "identity": IDENTITY, "identity_sha256": digest(IDENTITY)})
    raw = sys.stdin.buffer.readline(1024*1024)
    if mode in ("shutdown-stderr", "shutdown-mixed"):
        sys.stderr.buffer.write(b"d"*(512*1024))
        sys.stderr.buffer.flush()
        if mode == "shutdown-mixed":
            emit({"type": "unsolicited_after_shutdown_diagnostics"})
            time.sleep(10)
        return
    if mode == "bad-close":
        emit({"type": "unsolicited_after_eof"})
        return
    request = json.loads(raw)
    if mode == "mismatch":
        emit({"type": "completion", "request_id": request["request_id"]+1,
            "identity_sha256": digest(IDENTITY), "envelope_sha256": request["envelope_sha256"],
            "completion": '{"action_id":"a0"}'})
    elif mode == "oversized":
        sys.stdout.buffer.write(b"x"*8192)
        sys.stdout.buffer.flush()
    elif mode == "stderr-flood":
        sys.stderr.buffer.write(b"z"*8192)
        sys.stderr.buffer.flush()
    time.sleep(10)  # Client must terminate this owned artificial worker.


class PersistentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "root"):
            cls.root = Path(tempfile.mkdtemp(prefix="ace-persistent-policy-", dir="/tmp"))

    def client(self, name, mode="normal", **overrides):
        args = dict(command=(sys.executable, "-B", str(Path(__file__).resolve()), "--worker", mode),
            timeout_seconds=2., expected_identity=IDENTITY, output=self.root/name,
            startup_seconds=5., cleanup_seconds=.8)
        args.update(overrides)
        return PersistentPolicy(**args)

    def failure(self, name):
        result = json.loads((self.root/name/"failure.json").read_text())
        self.assertFalse(result["retry_allowed"])
        self.assertFalse((self.root/name/"closed.json").exists())
        self.assertFalse(result["cleanup"]["group_still_exists"])
        self.assertEqual(result["cleanup"]["errors"], [])
        return result

    def test_load_once_two_distinct_public_requests_and_exact_completion(self):
        first, second = envelope(), envelope("second")
        with self.client("two") as policy:
            self.assertEqual(policy.call(first, 0), ' {"action_id": "a0"} ')
            self.assertEqual(policy.call(second, 7), ' {"action_id": "a1"} ')
        path = self.root/"two"
        stderr = (path/"stderr.bin").read_text()
        self.assertEqual(stderr.count("ARTIFICIAL_INIT"), 1)
        self.assertEqual(stderr.count("ARTIFICIAL_DECISION"), 2)
        closed = json.loads((path/"closed.json").read_text())
        self.assertEqual(closed["completed_requests"], 2)
        self.assertEqual(closed["cleanup"]["leader_returncode"], 0)
        self.assertFalse(closed["cleanup"]["group_still_exists"])
        for index, supplied in enumerate((first, second)):
            sent = json.loads((path/f"request-{index:04d}.input.json").read_text())
            self.assertEqual(sent["envelope"], supplied)
            self.assertEqual(sent["envelope_sha256"], digest(supplied))

    def test_existing_choose_adapter(self):
        with self.client("choose", task=TASK) as policy:
            self.assertIsInstance(policy, ProcessPolicy)
            self.assertEqual(policy.choose(envelope()["state"], self.root, 0), ' {"action_id": "a0"} ')

    def test_repeated_id_closes_session_without_second_callback_or_retry(self):
        policy = self.client("repeat")
        policy.call(envelope(), 0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            policy.call(envelope("second"), 0)
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            policy.call(envelope(), 1)
        failure = self.failure("repeat")
        self.assertEqual(failure["completed_requests"], 1)
        self.assertEqual((self.root/"repeat/stderr.bin").read_text().count("ARTIFICIAL_DECISION"), 1)

    def test_worker_independently_rejects_repeated_id_and_initializes_once(self):
        public = envelope()
        request = {"type": "request", "request_id": 0, "envelope": public,
                   "envelope_sha256": digest(public), "identity_sha256": digest(IDENTITY)}
        calls = []
        def initialize():
            calls.append("init")
            def decide(value):
                calls.append("decision")
                return '{"action_id":"a0"}'
            return decide
        wire = ((canonical(request)+"\n")*2).encode()
        out = io.BytesIO()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            serve_policy(initialize, identity=IDENTITY, input_stream=io.BytesIO(wire), output_stream=out)
        self.assertEqual(calls, ["init", "decision"])
        self.assertEqual(len(out.getvalue().splitlines()), 2)
        (self.root/"worker-repeat-wire.bin").write_bytes(out.getvalue())

    def test_mismatching_response_id_is_retained_and_rejected(self):
        policy = self.client("mismatch", "mismatch")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            policy.call(envelope(), 0)
        self.assertEqual(self.failure("mismatch")["completed_requests"], 0)
        observed = json.loads((self.root/"mismatch/request-0000.observed-response.json").read_text())
        self.assertEqual(observed["request_id"], 1)

    def test_invalid_handshake_and_startup_timeout_cleanup(self):
        with self.assertRaisesRegex(ValueError, "handshake identity"):
            self.client("bad-handshake", "bad-handshake")
        self.failure("bad-handshake")
        with self.assertRaises(TimeoutError):
            self.client("startup", "startup-timeout", startup_seconds=.1)
        self.failure("startup")

    def test_oversized_output_and_stderr_are_bounded(self):
        policy = self.client("oversized", "oversized", max_output_bytes=1024)
        with self.assertRaisesRegex(ValueError, "output line exceeds"):
            policy.call(envelope(), 0)
        self.failure("oversized")
        self.assertLessEqual((self.root/"oversized/stdout.bin").stat().st_size, 2049)
        policy = self.client("stderr", "stderr-flood", max_stderr_bytes=128)
        with self.assertRaisesRegex(ValueError, "stderr cumulative"):
            policy.call(envelope(), 0)
        failure = self.failure("stderr")
        self.assertEqual((self.root/"stderr/stderr.bin").stat().st_size, 128)
        self.assertGreater(failure["discarded_observed_bytes"], 0)

    def test_request_timeout_and_invalid_completion_have_no_fallback(self):
        policy = self.client("timeout", "timeout", timeout_seconds=.1)
        with self.assertRaises(TimeoutError): policy.call(envelope(), 0)
        self.failure("timeout")
        policy = self.client("bad-completion", "bad-completion")
        with self.assertRaises(EOFError): policy.call(envelope(), 0)
        self.failure("bad-completion")
        self.assertIn("ValueError", (self.root/"bad-completion/stderr.bin").read_text())

    def test_private_fields_and_oversized_inputs_never_cross_boundary(self):
        policy = self.client("private")
        public = envelope()
        public["private_evaluation"] = "DO_NOT_FORWARD"
        with self.assertRaises(ValueError): policy.call(public, 0)
        self.failure("private")
        self.assertFalse(list((self.root/"private").glob("*.input.json")))
        self.assertNotIn("ARTIFICIAL_DECISION", (self.root/"private/stderr.bin").read_text())
        policy = self.client("input-bound", max_input_bytes=2048)
        with self.assertRaisesRegex(ValueError, "input byte bound"):
            policy.call(envelope("x"*10000), 0)
        self.failure("input-bound")
        self.assertFalse(list((self.root/"input-bound").glob("*.input.json")))

    def test_cleanup_permission_failures_are_reported_then_test_reaps_own_worker(self):
        policy = self.client("cleanup-failed", "timeout", timeout_seconds=.1, cleanup_seconds=.08)
        real_killpg = os.killpg
        def deny_signals(pgid, sig):
            if sig: raise PermissionError("artificial cleanup denial")
            return real_killpg(pgid, sig)
        try:
            with patch("llm_intervention_persistent_policy.os.killpg", side_effect=deny_signals):
                with self.assertRaises(TimeoutError): policy.call(envelope(), 0)
            failure = json.loads((self.root/"cleanup-failed/failure.json").read_text())
            self.assertTrue(failure["cleanup"]["group_still_exists"])
            self.assertTrue(failure["cleanup"]["errors"])
            self.assertFalse((self.root/"cleanup-failed/closed.json").exists())
        finally:
            cleanup = cleanup_process_group(policy.process, 1.)
            (self.root/"cleanup-failed/test-owned-worker-final-cleanup.json").write_text(canonical(cleanup))
            self.assertFalse(cleanup["group_still_exists"])
            self.assertEqual(cleanup["errors"], [])

    def test_close_rejects_unsolicited_output(self):
        policy = self.client("bad-close", "bad-close")
        with self.assertRaisesRegex(ValueError, "unsolicited"):
            policy.close()
        self.failure("bad-close")
        self.assertIn(b"unsolicited_after_eof", (self.root/"bad-close/stdout.bin").read_bytes())

    def test_existing_output_is_not_overwritten_or_launched(self):
        path = self.root/"exists"
        path.mkdir()
        (path/"sentinel").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError): self.client("exists")
        self.assertEqual([p.name for p in path.iterdir()], ["sentinel"])
        self.assertEqual((path/"sentinel").read_bytes(), b"preserve")

    def test_review02_shutdown_drains_pipe_filling_stderr_before_clean_exit(self):
        policy = self.client("review02-shutdown", "shutdown-stderr", max_stderr_bytes=1024*1024,
                             cleanup_seconds=2.)
        policy.close()
        path = self.root/"review02-shutdown"
        self.assertEqual((path/"stderr.bin").read_bytes(), b"d"*(512*1024))
        closed = json.loads((path/"closed.json").read_text())
        self.assertEqual(closed["cleanup"]["leader_returncode"], 0)
        self.assertFalse(closed["cleanup"]["term_sent"])
        self.assertFalse(closed["cleanup"]["kill_sent"])
        self.assertFalse(closed["cleanup"]["group_still_exists"])
        self.assertEqual(closed["captured_bytes"]["stderr"], 512*1024)

    def test_review02_shutdown_detects_stdout_while_draining_stderr(self):
        policy = self.client("review02-mixed", "shutdown-mixed", max_stderr_bytes=1024*1024,
                             cleanup_seconds=2.)
        with self.assertRaisesRegex(ValueError, "unsolicited"):
            policy.close()
        self.failure("review02-mixed")
        path = self.root/"review02-mixed"
        self.assertEqual((path/"stderr.bin").read_bytes(), b"d"*(512*1024))
        self.assertIn(b"unsolicited_after_shutdown_diagnostics", (path/"stdout.bin").read_bytes())
        cleanup = json.loads((path/"close-cleanup.json").read_text())
        self.assertTrue(any("pipe service" in error for error in cleanup["errors"]))
        self.assertFalse(cleanup["group_still_exists"])

    def test_review02_shutdown_stderr_limit_still_rejects_and_cleans_up(self):
        policy = self.client("review02-limit", "shutdown-stderr", max_stderr_bytes=65536,
                             cleanup_seconds=2.)
        with self.assertRaisesRegex(ValueError, "stderr cumulative"):
            policy.close()
        failure = self.failure("review02-limit")
        self.assertEqual((self.root/"review02-limit/stderr.bin").stat().st_size, 65536)
        self.assertGreater(failure["discarded_observed_bytes"], 0)

    def test_review02_choose_session_ids_rebinding_and_episode_mapping(self):
        other_task = PredictionTask("Y", (MENU[1],), (1.,), 2., .08)
        with self.client("review02-episodes", task=TASK) as policy:
            pid = policy.process.pid
            self.assertEqual(policy.call(envelope(), 40), ' {"action_id": "a0"} ')
            self.assertEqual(policy.choose(envelope()["state"], self.root/"episode-A", 0), ' {"action_id": "a0"} ')
            self.assertIs(policy.bind_task(other_task), policy)
            self.assertEqual(policy.choose(envelope("second")["state"], self.root/"episode-B", 0), ' {"action_id": "a1"} ')
            self.assertEqual(policy.choose(envelope()["state"], self.root/"episode-B", 1), ' {"action_id": "a0"} ')
            self.assertEqual(policy.process.pid, pid)
        path = self.root/"review02-episodes"
        for index, session_id in enumerate((40, 41, 42, 43)):
            request = json.loads((path/f"request-{index:04d}.input.json").read_text())
            response = json.loads((path/f"request-{index:04d}.observed-response.json").read_text())
            self.assertEqual(request["request_id"], session_id)
            self.assertEqual(response["request_id"], session_id)
            self.assertNotIn("episode_directory", request)
            self.assertNotIn("decision_index", request)
            self.assertEqual(request["envelope"]["task"], (TASK if index < 2 else other_task).payload())
        for index, episode, local_id in ((1, "episode-A", 0), (2, "episode-B", 0), (3, "episode-B", 1)):
            mapping = json.loads((path/f"request-{index:04d}.episode.json").read_text())
            self.assertEqual(mapping["session_request_id"], 40+index)
            self.assertEqual(mapping["decision_index"], local_id)
            self.assertEqual(mapping["episode_directory"], str((self.root/episode).resolve()))
            self.assertEqual(mapping["task_sha256"], digest((TASK if index == 1 else other_task).payload()))
        binding = json.loads((path/"task-binding-0000.json").read_text())
        self.assertEqual(binding["worker_pid"], pid)
        self.assertEqual(binding["last_completed_session_request_id"], 41)
        self.assertEqual(binding["task_sha256"], digest(other_task.payload()))
        self.assertFalse(binding["model_reloaded"])
        self.assertEqual((path/"stderr.bin").read_text().count("ARTIFICIAL_INIT"), 1)
        self.assertEqual(json.loads((path/"closed.json").read_text())["completed_requests"], 4)

    def test_review02_episode_mapping_survives_rejected_response(self):
        policy = self.client("review02-mapping-failed", "mismatch", task=TASK)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            policy.choose(envelope()["state"], self.root/"failed-episode", 0)
        self.failure("review02-mapping-failed")
        mapping = json.loads((self.root/"review02-mapping-failed/request-0000.episode.json").read_text())
        self.assertEqual(mapping["session_request_id"], 0)
        self.assertEqual(mapping["decision_index"], 0)
        self.assertEqual(mapping["episode_directory"], str((self.root/"failed-episode").resolve()))


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text)
    def flush(self):
        for stream in self.streams: stream.flush()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        PersistentPolicyTests.root = Path(tempfile.mkdtemp(prefix="ace-persistent-policy-", dir="/tmp"))
        with (PersistentPolicyTests.root/"test-report.txt").open("x") as report:
            stream = _Tee(sys.stderr, report)
            stream.write("Interpreter: "+sys.executable+"\nVersion: "+sys.version.replace("\n", " ")+"\n")
            stream.write("Retained artifacts/report: "+str(PersistentPolicyTests.root)+"\n")
            stream.flush()
            unittest.main(testRunner=unittest.TextTestRunner(stream=stream, verbosity=2))
