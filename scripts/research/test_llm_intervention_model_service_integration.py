"""Synthetic saved-session acceptance fixtures, no service/model execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qualify_llm_intervention_model_service import (validate_session, arm_final_deadline,
    scheduler_record, require_gpu_sample)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class IntegrationEvidenceTests(unittest.TestCase):
    def fixture(self, name):
        root = self.root/self._testMethodName/name
        (root/"worker").mkdir(parents=True)
        (root/"client").mkdir()
        identity = {"fixture": "arithmetic-only"}
        pins = ["a"*64, "b"*64, "c"*64, "a"*64]
        def put(name, value):
            (root/name).write_text(json.dumps(value)+"\n")
        put("worker/terminal.json", {"status": "complete", "freeze_sha256": "f"*64,
            "generation_count": 4, "new_training_updates": 0, "environment_responses": 0})
        put("client/closed.json", {"status": "closed", "completed_requests": 4, "last_request_id": 43,
            "cleanup": {"errors": [], "group_still_exists": False, "leader_returncode": 0,
                        "term_sent": False, "kill_sent": False}})
        put("worker/authenticated.json", {"freeze_sha256": "f"*64, "identity": identity})
        put("client/ready.json", {"identity_sha256": digest(identity)})
        for i, phase in enumerate(("model_cpu_load_start", "model_cpu_load_complete", "model_ready")):
            put(f"worker/phase-{i:03d}.json", {"phase": phase})
        for i, pin in enumerate(pins):
            ids = [1, 2, 3] if i in (0, 3) else [1, i+4, 3]
            completion = '{"action_id":"a00"}'
            g = {"ordinal": i, "envelope_sha256": pin, "new_training_updates": 0,
                "environment_callbacks_available": False, "tensor_versions_unchanged": True,
                "cross_request_cache_supplied": False, "cuda_interval_ms": 1.,
                "peak_gpu_allocated_bytes": 1024, "completion": completion,
                "sequence_ids": ids, "prompt_ids": ids[:1]}
            wire = {"type": "completion", "request_id": 40+i, "identity_sha256": digest(identity),
                "envelope_sha256": pin, "completion": completion}
            raw = struct.pack("<3q", *ids)
            (root/f"worker/raw-generation-{i:04d}.bin").write_bytes(raw)
            put(f"worker/raw-generation-{i:04d}.json", {"status": "returned_unaccepted", "truncated": False,
                "token_prefix": ids, "raw_sha256": hashlib.sha256(raw).hexdigest()})
            put(f"worker/generation-{i:04d}.json", g)
            put(f"client/request-{i:04d}.observed-response.json", wire)
            put(f"client/request-{i:04d}.accepted.json", {"request_id": 40+i,
                "completion": completion, "response_sha256": digest(wire)})
        return root, {"freeze_sha256": "f"*64, "identity_sha256": digest(identity), "envelope_pins": pins}

    def alter(self, root, name, change):
        path = root/name
        value = json.loads(path.read_bytes())
        (root/(name+".original")).write_bytes(path.read_bytes())
        change(value)
        path.write_text(json.dumps(value)+"\n")

    def test_complete_arithmetic_fixture(self):
        root, pins = self.fixture("complete")
        observed = validate_session(root, **pins)
        self.assertEqual(observed["requests"], 4)
        self.assertEqual(observed["cuda_interval_ms"], 4.)

    def test_corrupt_identity_command_tokens_and_progress_rejected(self):
        variants = [
            ("client/request-0001.observed-response.json", lambda v: v.update(request_id=42)),
            ("worker/authenticated.json", lambda v: v.update(freeze_sha256="e"*64)),
            ("worker/generation-0002.json", lambda v: v.update(envelope_sha256="d"*64)),
            ("worker/generation-0001.json", lambda v: v.update(completion='{"stop":true}')),
            ("worker/generation-0001.json", lambda v: v.update(cuda_interval_ms=0.)),
            ("worker/raw-generation-0002.json", lambda v: v.update(token_prefix=[999])),
        ]
        for i, (name, change) in enumerate(variants):
            with self.subTest(i=i):
                root, pins = self.fixture(str(i))
                self.alter(root, name, change)
                with self.assertRaises(ValueError):
                    validate_session(root, **pins)

    def test_updates_responses_and_unclean_exit_rejected(self):
        variants = [
            ("worker/terminal.json", lambda v: v.update(new_training_updates=1)),
            ("worker/terminal.json", lambda v: v.update(environment_responses=1)),
            ("client/closed.json", lambda v: v["cleanup"].update(group_still_exists=True)),
            ("client/closed.json", lambda v: v["cleanup"].update(term_sent=True)),
            ("client/closed.json", lambda v: v.update(completed_requests=3)),
        ]
        for i, (name, change) in enumerate(variants):
            root, pins = self.fixture(str(i))
            self.alter(root, name, change)
            with self.assertRaises(ValueError):
                validate_session(root, **pins)

    def test_extra_load_generation_failure_and_raw_corruption_rejected(self):
        for variant in ("extra_load", "extra_generation", "failure", "raw_corruption"):
            root, pins = self.fixture(variant)
            if variant == "extra_load":
                (root/"worker/phase-003.json").write_text('{"phase":"model_ready"}')
            elif variant == "extra_generation":
                (root/"worker/generation-0004.json").write_text('{}')
            elif variant == "failure":
                (root/"worker/failure.json").write_text('{"type":"fixture"}')
            else:
                (root/"worker/raw-generation-0002.bin").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_session(root, **pins)

    def test_scheduler_failures_preserve_raw_and_partial_output(self):
        for name, response in (("nonzero", SimpleNamespace(returncode=1, stdout="partial job", stderr="denied")),
            ("timeout", subprocess.TimeoutExpired(["scontrol"], 10, output=b"partial", stderr=b"delayed"))):
            root = self.root/self._testMethodName/name
            root.mkdir(parents=True)
            options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            with patch("qualify_llm_intervention_model_service.subprocess.run", **options):
                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    scheduler_record(root, "1234")
            raw = json.loads((root/"scheduler_raw.json").read_bytes())
            self.assertTrue(raw["stdout"].startswith("partial"))
            self.assertTrue(raw["stderr"])

    def test_gpu_sample_uncertainty_and_cleanup_gate(self):
        require_gpu_sample({"returncode": 0, "owned_rows": ["123, uuid, 42"]}, active=True)
        require_gpu_sample({"returncode": 0, "owned_rows": []}, active=False)
        for sample, active in (({"returncode": 1, "owned_rows": ["123, uuid, 42"]}, True),
                ({"returncode": None, "owned_rows": []}, False),
                ({"returncode": 0, "owned_rows": ["123, uuid, 42"]}, False),
                ({"returncode": 0, "owned_rows": []}, True)):
            with self.assertRaises(ValueError):
                require_gpu_sample(sample, active=active)

    def test_final_watchdog_kills_only_owned_group_and_hard_exits(self):
        handlers = {}
        with patch("qualify_llm_intervention_model_service.signal.signal", side_effect=lambda s, h: handlers.update({s:h})), \
             patch("qualify_llm_intervention_model_service.signal.setitimer") as timer, \
             patch("qualify_llm_intervention_model_service.time.monotonic", return_value=110.), \
             patch("qualify_llm_intervention_model_service.os.killpg") as kill, \
             patch("qualify_llm_intervention_model_service.os._exit", side_effect=RuntimeError("hard exit")) as leave:
            import signal
            arm_final_deadline(100., SimpleNamespace(pid=7654321))
            timer.assert_called_once_with(signal.ITIMER_REAL, 570.)
            with self.assertRaisesRegex(RuntimeError, "hard exit"):
                handlers[signal.SIGALRM](None, None)
            kill.assert_called_once_with(7654321, signal.SIGKILL)
            leave.assert_called_once_with(124)


if __name__ == "__main__":
    IntegrationEvidenceTests.root = Path(sys.argv.pop(1))
    IntegrationEvidenceTests.root.mkdir(exist_ok=False)
    unittest.main(verbosity=2)
