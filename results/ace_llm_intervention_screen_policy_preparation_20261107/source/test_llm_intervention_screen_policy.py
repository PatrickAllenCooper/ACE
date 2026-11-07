"""New public-envelope fixtures only. No world realization or environment call.

An exclusive ACE_SCREEN_POLICY_TEST_OUTPUT is required so every subprocess,
input, expected rejection and fixture remains available after the single run.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llm_intervention_contract import Action, Node, VERSION as STATE_VERSION, canonical, digest
from llm_intervention_numerical_policy import choose as historical_choose
from llm_intervention_screen_protocol import build_proposal, REQUIRED_SOURCE_PATHS
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_task import PredictionTask, TaskProcessPolicy, make_envelope
import llm_intervention_screen_policy as policy
import llm_intervention_world as world


def save(path, value):
    with path.open("x") as handle:
        handle.write(canonical(value) + "\n")


class ScreenPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["ACE_SCREEN_POLICY_TEST_OUTPUT"])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.path = self.root / self._testMethodName
        self.path.mkdir()
        self.rejections = 0
        # These are written public receipt fixtures, not sampled or returned
        # environment responses. Their posterior is a hand-computed scalar one.
        self.graph = (Node("X", (), "unit", (-2., 2.)),)
        self.menu = (Action("observe_all", "observe", (), ("X",), 1),
                     Action("set_x", "intervene", (("X", 1.),), ("X",), 2))
        self.task = PredictionTask("X", (Action("q", "observe", (), ("X",), 1),),
                                   (1.,), 2., .04)
        self.blockers = []
        for name in ("generate_world", "_normal", "private_structural_means", "training_callbacks"):
            p = patch.object(world, name, side_effect=AssertionError("forbidden world operation"))
            p.start()
            self.blockers.append(p)
            self.addCleanup(p.stop)
        p = patch.object(world.ResponseAdapter, "__call__", side_effect=AssertionError("no environment"))
        p.start()
        self.addCleanup(p.stop)

    def state(self, costly=False, initial=2):
        history = [{"request_id": i, "action_id": "observe_all", "kind": "observe",
                    "targets": {}, "values": {"X": float(2*i)}, "cost_units": 1}
                   for i in range(initial)]
        if costly:
            history.append({"request_id": initial, "action_id": "set_x", "kind": "intervene",
                            "targets": {"X": 1.}, "values": {"X": 1.}, "cost_units": 2})
        spent = initial + 2*int(costly)
        return json.loads(canonical({"version": STATE_VERSION, "objective": "Artificial public prediction",
            "domain_notes": "Hand-specified public fixture. No environment callback.",
            "graph": [asdict(n) for n in self.graph], "menu": [asdict(a) for a in self.menu],
            "history": history, "scm": [{"node": "X", "terms": ["intercept"],
                "parameter_mean": [initial*(initial-1)/(initial+1)],
                "parameter_covariance": [[1/(initial+1)]], "noise_variance": 1., "eligible_rows": initial}],
            "spent_units": spent, "remaining_units": 6-spent, "requests": len(history),
            "returned_responses": len(history), "terminal": False}))

    def binding_args(self, arm="uniform_random", wid=17, initial=2):
        namespace = digest({"protocol": policy.PROPOSAL_VERSION, "scope": "artificial",
                            "world_id": wid, "purpose": "trajectory", "arm": arm})
        return dict(seed_prefix=["artificial", wid, "policy", namespace, "behavior"],
            scope="artificial", world_id=wid, request_namespace=namespace, arm=arm, role="behavior",
            initial_actions=["observe_all"]*initial, initial_cost_units=initial,
            total_budget=6, planning_draws=32)

    def binding(self, **kwargs):
        return policy.bind_policy(**self.binding_args(**kwargs))

    def envelope(self, **kwargs):
        return make_envelope(self.state(**kwargs), self.task)

    def reject(self, label, value, function):
        self.rejections += 1
        try:
            function()
        except (ValueError, TypeError, KeyError) as exc:
            save(self.path / ("rejection-%02d.json" % self.rejections),
                 {"label": label, "input": value, "type": type(exc).__name__, "message": str(exc)})
        else:
            self.fail("expected rejection: " + label)

    def command(self, b):
        return (sys.executable, "-B", str(Path(policy.__file__).resolve()),
                "--binding-json", canonical(b), "--binding-sha256", digest(b))

    def test_exact_key_oracle_offset_and_distinct_cost_units(self):
        b = self.binding()
        random.seed(700)
        np.random.seed(701)
        r0, n0 = random.getstate(), np.random.get_state()
        receipts = []
        for index, costly in enumerate((False, True)):
            envelope = self.envelope(costly=costly)
            seed, receipt = policy.decision_rng(envelope, binding=b,
                expected_binding_sha256=digest(b), expected_decision_index=index)
            key = ["ace-linear-gaussian-world-v1", "artificial", 17, "policy",
                   [b["request_namespace"], "behavior", index]]
            raw = json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)
            expected = hashlib.sha256(raw.encode()).digest()
            self.assertEqual(seed, int.from_bytes(expected, "big"))
            self.assertEqual(receipt["seed_hex_256"], expected.hex())
            self.assertEqual(receipt["canonical_key_json"], raw)
            self.assertEqual(receipt["decision_index"], index)
            self.assertEqual(receipt["request_count"], 2+index)
            self.assertEqual(receipt["spent_units"], 2+2*index)
            receipts.append(receipt)
        self.assertEqual(random.getstate(), r0)
        n1 = np.random.get_state()
        self.assertEqual(n0[0], n1[0])
        np.testing.assert_array_equal(n0[1], n1[1])
        self.assertEqual(n0[2:], n1[2:])
        self.assertNotEqual(receipts[0]["seed_decimal"], receipts[1]["seed_decimal"])
        save(self.path/"key-oracles.json", receipts)

    def test_actual_task_process_abi_all_four_controls(self):
        retained = []
        for arm in policy.METHODS:
            b = self.binding(arm=arm)
            worker = TaskProcessPolicy(self.command(b), 15., self.task)
            directory = self.path/arm
            directory.mkdir()
            # Two completed public histories exercise the real stdin/stdout ABI
            # at decisions 0 and 1 without executing an environment or a world.
            for index, costly in enumerate((False, True)):
                state = self.state(costly=costly)
                completion = worker.choose(state, directory, index)
                result = json.loads((directory/("decision-%04d.process.json" % index)).read_text())
                receipt = json.loads(result["stderr"])
                decision = json.loads(completion)
                seed = int(receipt["seed_decimal"])
                if arm == "target_risk":
                    expected, meta = choose_target_risk(state, seed=seed, draws=32, **self.task.risk_arguments())
                else:
                    expected, meta = historical_choose(state, method=arm, seed=seed, draws=32)
                self.assertEqual(decision, expected)
                self.assertEqual(receipt["planning_metadata"], meta)
                self.assertEqual(receipt["decision_index"], index)
                self.assertEqual(receipt["declared_planning_draws"], 32)
                self.assertEqual(receipt["public_envelope_sha256"], result["input_sha256"])
                self.assertEqual(receipt["decision_sha256"], digest(decision))
                self.assertEqual(receipt["environment_responses"], 0)
                self.assertEqual(set(json.loads((directory/("decision-%04d.input.json" % index)).read_text())),
                                 {"version", "state", "task"})
                self.assertNotIn("stop", decision)
                if arm in ("local_information", "target_risk"):
                    self.assertEqual(decision, {"action_id": "observe_all"})
                    count_key = "fantasy_responses" if arm == "target_risk" else "planning_draws"
                    self.assertEqual(meta[count_key], 64)
                if arm == "menu_coverage":
                    self.assertEqual(decision, {"action_id": "set_x"})
                retained.append({"arm": arm, "index": index, "decision": decision})
        save(self.path/"abi-results.json", retained)

    def test_binding_rejects_scope_role_namespace_seed_and_draw_drift(self):
        base = self.binding_args()
        changes = [{"scope": "policy_train"}, {"scope": "confirmation"}, {"role": "teacher"},
            {"role": "snapshot"}, {"arm": "llm_sft"}, {"planning_draws": 16}, {"planning_draws": True},
            {"world_id": True}, {"request_namespace": "a"*64}, {"initial_cost_units": 0},
            {"seed_prefix": base["seed_prefix"]+[0]}, {"seed_prefix": ["wrong"]},
            {"seed_prefix": ["artificial", 17.0, "policy", base["request_namespace"], "behavior"]},
            {"initial_actions": [], "initial_cost_units": 2}]
        for i, delta in enumerate(changes):
            args = dict(base, **delta)
            self.reject(str(i), args, lambda args=args: policy.bind_policy(**args))
        b = self.binding()
        self.reject("independent pin", b, lambda: policy.validate_binding(b, expected_sha256="0"*64))
        for extra in ({"private_truth": 3}, {"version": "wrong"}, {"proposal_version": "wrong"}):
            v = dict(b, **extra)
            self.reject("extra/version", v, lambda v=v: policy.validate_binding(v, expected_sha256=digest(v)))

    def test_public_history_gate_and_no_selection_after_failure(self):
        b = self.binding()
        base = self.envelope()
        variants = []
        v = copy.deepcopy(base)
        v["state"]["history"] = v["state"]["history"][:1]
        variants.append(("incomplete initial", v))
        v = copy.deepcopy(base)
        v["state"]["returned_responses"] -= 1
        variants.append(("unknown return", v))
        v = copy.deepcopy(base)
        v["state"]["terminal"] = True
        variants.append(("already stopped", v))
        v = copy.deepcopy(base)
        v["state"]["remaining_units"] -= 1
        variants.append(("changed budget", v))
        v = self.envelope(costly=True)
        v["state"]["history"][2]["cost_units"] = 1
        variants.append(("incorrect charge", v))
        v = copy.deepcopy(base)
        v["state"]["history"][1]["request_id"] = 0
        variants.append(("duplicate receipt", v))
        v = copy.deepcopy(base)
        v["state"]["private_error"] = .1
        variants.append(("private state field", v))
        v = copy.deepcopy(base)
        v["private_parameters"] = [1.]
        variants.append(("private envelope field", v))
        for label, envelope in variants:
            with patch.object(policy, "choose_numerical", side_effect=AssertionError("must reject before choosing")):
                self.reject(label, envelope, lambda envelope=envelope: policy.choose_public(
                    envelope, binding=b, expected_binding_sha256=digest(b)))
        for index in (2, True, -1):
            self.reject("wrong call index", index, lambda index=index: policy.decision_rng(base,
                binding=b, expected_binding_sha256=digest(b), expected_decision_index=index))
        changed = dict(b, initial_actions=["set_x", "set_x"], initial_cost_units=4)
        self.reject("initial action mismatch", changed, lambda: policy.decision_rng(base,
            binding=changed, expected_binding_sha256=digest(changed)))
        changed = dict(b, initial_cost_units=3)
        self.reject("initial cost mismatch", changed, lambda: policy.decision_rng(base,
            binding=changed, expected_binding_sha256=digest(changed)))

    def test_zero_initial_and_budget_terminal_edges(self):
        b = self.binding(initial=0)
        seed, receipt = policy.decision_rng(self.envelope(initial=0), binding=b,
                                            expected_binding_sha256=digest(b))
        self.assertEqual(receipt["decision_index"], 0)
        self.assertEqual(receipt["initial_cost_units"], 0)
        save(self.path/"zero-initial.json", receipt)
        b = dict(self.binding(), total_budget=2)
        envelope = self.envelope()
        envelope["state"]["remaining_units"] = 0
        self.reject("normal executor budget stop", envelope, lambda: policy.decision_rng(envelope,
            binding=b, expected_binding_sha256=digest(b)))

    def test_proposal_development_bindings_without_realization(self):
        # Pure declared metadata comparison only. No scientific envelope or world
        # is constructed and no numerical policy is run on these identities.
        proposal = build_proposal(source_pins={p: "a"*64 for p in REQUIRED_SOURCE_PATHS})
        bindings = []
        for row in proposal["development_worlds"]:
            for episode in row["development_episodes"]:
                if episode["arm"] not in policy.METHODS:
                    continue
                b = policy.bind_policy(seed_prefix=episode["policy_seed_prefix"], scope=row["scope"],
                    world_id=row["world_id"], request_namespace=episode["measurement_binding"]["request_namespace"],
                    arm=episode["arm"], role="behavior", initial_actions=row["public"]["initial_actions"],
                    initial_cost_units=row["public"]["initial_cost_units"],
                    total_budget=episode["total_budget"], planning_draws=32)
                self.assertEqual(b["seed_prefix"], episode["policy_seed_prefix"])
                self.assertEqual(policy.validate_binding(b, expected_sha256=digest(b)), b)
                bindings.append(b)
        self.assertEqual(len(bindings), 96)
        self.assertEqual(len({b["request_namespace"] for b in bindings}), 96)
        b = bindings[0]
        args = {k: v for k, v in b.items() if k not in ("version", "proposal_version")}
        for delta in ({"initial_actions": ["observe_all"]*7, "initial_cost_units": 7},
                      {"total_budget": 33}, {"world_id": 0}):
            changed = dict(args, **delta)
            self.reject("development drift", changed, lambda changed=changed: policy.bind_policy(**changed))
        save(self.path/"declared-bindings-only.json", {"bindings": bindings, "realized_worlds": 0,
             "source_pins": "fabricated placeholders, not original source authentication"})

    def test_determinism_and_namespace_separation(self):
        envelopes = [self.envelope(), self.envelope(costly=True)]
        b = self.binding()
        before = canonical(envelopes)
        forward = [policy.choose_public(e, binding=b, expected_binding_sha256=digest(b)) for e in envelopes]
        reverse = [policy.choose_public(e, binding=b, expected_binding_sha256=digest(b)) for e in reversed(envelopes)]
        self.assertEqual(forward, list(reversed(reverse)))
        self.assertEqual(before, canonical(envelopes))
        other = self.binding(arm="menu_coverage")
        other_world = self.binding(wid=18)
        seeds = [policy.decision_rng(envelopes[0], binding=x, expected_binding_sha256=digest(x))[0]
                 for x in (b, other, other_world)]
        self.assertEqual(len(set(seeds)), 3)
        args = self.binding_args()
        detached = policy.bind_policy(**args)
        args["seed_prefix"][0] = "mutated"
        args["initial_actions"].clear()
        self.assertEqual(detached, b)
        save(self.path/"determinism.json", {"forward": forward, "distinct_seed_decimals": list(map(str, seeds))})

    def test_cli_rejects_bad_payload_and_pin_without_fallback(self):
        b = self.binding()
        command = self.command(b)
        inputs = [("duplicate-key", '{"version":1,"version":2}'),
                  ("private-field", canonical(dict(self.envelope(), private_evaluation=2))),
                  ("nonfinite", '{"value":NaN}'),
                  ("oversize", " "*(policy.MAX_INPUT_BYTES+1))]
        for label, raw in inputs:
            directory = self.path/label
            directory.mkdir()
            (directory/"stdin.txt").write_text(raw)
            result = subprocess.run(command, input=raw, text=True, capture_output=True, timeout=15, check=False)
            save(directory/"process.json", {"command": command, "returncode": result.returncode,
                                           "stdout": result.stdout, "stderr": result.stderr})
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertEqual(json.loads(result.stderr)["status"], "rejected")
        directory = self.path/"wrong-pin"
        directory.mkdir()
        bad_command = command[:-1]+("0"*64,)
        worker = TaskProcessPolicy(bad_command, 15., self.task)
        with self.assertRaisesRegex(RuntimeError, "policy process failed"):
            worker.choose(self.state(), directory, 0)
        receipt = json.loads((directory/"decision-0000.process.json").read_text())
        self.assertEqual(receipt["returncode"], 2)
        self.assertEqual(receipt["stdout"], "")

    def test_no_neural_runtime_imported(self):
        self.assertFalse(any(k.split(".")[0] in {"torch", "transformers", "safetensors", "tabpfn"}
                             for k in sys.modules))
        save(self.path/"runtime.json", {"python": sys.version, "numpy": np.__version__,
             "neural_modules_loaded": False, "environment_responses": 0,
             "scope": "public-history fixtures and working-model planning fantasies only"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
