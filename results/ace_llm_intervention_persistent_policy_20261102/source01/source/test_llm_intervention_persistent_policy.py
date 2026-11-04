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
