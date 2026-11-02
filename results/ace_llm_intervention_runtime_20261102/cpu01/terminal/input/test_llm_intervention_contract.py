"""Artificial interface fixtures only. No scientific worlds or language model."""
import json
from pathlib import Path
import tempfile
import unittest

from llm_intervention_contract import (
    Action, ChargedExecutor, Decision, MechanismSummary, Node, canonical, digest,
    parse_completion, parse_generated_sequence, policy_prompt, stream_seed, validate_design,
)


GRAPH = (Node("X", (), "unit", (-1., 1.)), Node("Y", ("X",), "unit"))
MENU = (Action("a0", "intervene", (("X", -1.),), ("X", "Y"), 2),
        Action("a1", "intervene", (("X", 1.),), ("Y",), 2),
        Action("o0", "observe", (), ("X", "Y"), 1))
SUMMARY = (MechanismSummary("Y", ("intercept", "X"), (0., 1.), ((1., 0.), (0., 1.)), .1, 0),)


class ContractTests(unittest.TestCase):
    def test_completion_not_prompt_or_examples(self):
        self.assertEqual(parse_completion('{"action_id":"a0"}', MENU), Decision("a0", False))
        for text in ('', 'Example: {"action_id":"a0"}', '```json\n{"action_id":"a0"}\n```',
                     '{"action_id":"a0"} {"action_id":"a1"}',
                     '{"action_id":"a0","action_id":"a1"}', '{"action_id":true}',
                     '{"action_id":"unknown"}', '{"stop":1}', '{"action_id":"a0","truth":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_completion(text, MENU)
        decoded = []
        def decode(ids):
            decoded.append(ids)
            return '{"action_id":"a1"}'
        decision, text = parse_generated_sequence([10, 11], [10, 11, 12], decode, MENU)
        self.assertEqual(decoded, [[12]])
        self.assertEqual(decision.action_id, "a1")
        with self.assertRaises(ValueError):
            parse_generated_sequence([10, 11], [10, 12, 13], decode, MENU)

    def test_design_guards_and_observation_distinction(self):
        validate_design(GRAPH, MENU)
        with self.assertRaises(ValueError):
            validate_design(tuple(reversed(GRAPH)), MENU)
        for target in ((("Y", 0.),), (("X", 2.),)):
            with self.assertRaises(ValueError):
                validate_design(GRAPH, (Action("bad", "intervene", target, ("Y",), 1),))
        for cost in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                Action("bad", "observe", (), ("Y",), cost)
        with self.assertRaises(ValueError):
            Action("bad", "observe", (("X", 0.),), ("Y",), 1)

    def test_reservation_precedes_callback_and_repetition_is_paid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"journal.jsonl"
            calls = []
            def respond(action, request):
                latest = json.loads(path.read_text().splitlines()[-1])
                self.assertEqual(latest["event"], "reserve")
                self.assertEqual(latest["request_id"], request)
                calls.append(action.action_id)
                return {name: 1. for name in action.measure}
            with ChargedExecutor(GRAPH, MENU, 4, path, respond) as executor:
                for _ in range(2):
                    executor.execute(Decision("a0", False), '{"action_id":"a0"}')
                with self.assertRaises(ValueError):
                    executor.execute(Decision("a1", False), '{"action_id":"a1"}')
                state = executor.state("predict Y", SUMMARY)
                self.assertEqual((state["spent_units"], state["remaining_units"], state["returned_responses"]), (4, 0, 2))
                self.assertEqual(calls, ["a0", "a0"])
            previous = None
            for line in path.read_text().splitlines():
                event = json.loads(line)
                pin = event.pop("sha256")
                self.assertEqual(event["previous_sha256"], previous)
                self.assertEqual(pin, digest(event))
                previous = pin
            with self.assertRaises(FileExistsError):
                ChargedExecutor(GRAPH, MENU, 4, path, respond)

    def test_unknown_return_halts_and_preserves_charge(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"journal.jsonl"
            def fail(*args):
                raise RuntimeError("artificial device interruption")
            with ChargedExecutor(GRAPH, MENU, 4, path, fail) as executor:
                with self.assertRaises(RuntimeError):
                    executor.execute(Decision("a0", False), '{"action_id":"a0"}')
                state = executor.state("predict Y", SUMMARY)
                self.assertEqual(state["spent_units"], 2)
                self.assertEqual(state["returned_responses"], 0)
                self.assertTrue(state["terminal"])
                with self.assertRaises(RuntimeError):
                    executor.execute(Decision("a0", False), '{"action_id":"a0"}')
            event = json.loads(path.read_text().splitlines()[-1])
            self.assertIsNone(event["response_returned_to_driver"])

    def test_invalid_return_is_known_return_and_terminal(self):
        for result in ({"Y": float("nan")}, {"Y": True}, {"Y": 0., "private_score": 1.}):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as td:
                path = Path(td)/"journal.jsonl"
                with ChargedExecutor(GRAPH, MENU, 4, path, lambda *args: result) as executor:
                    with self.assertRaises(ValueError):
                        executor.execute(Decision("a1", False), '{"action_id":"a1"}')
                    self.assertTrue(executor.state("goal", SUMMARY)["terminal"])
                self.assertTrue(json.loads(path.read_text().splitlines()[-1])["response_returned_to_driver"])

    def test_detached_allowlisted_public_state(self):
        with tempfile.TemporaryDirectory() as td:
            with ChargedExecutor(GRAPH, MENU, 4, Path(td)/"journal", lambda *args: {"Y": 3.}) as executor:
                executor.execute(Decision("a1", False), '{"action_id":"a1"}')
                state = executor.state("predict Y", SUMMARY)
                state["history"][0]["values"]["Y"] = -999
                self.assertEqual(executor.state("predict Y", SUMMARY)["history"][0]["values"]["Y"], 3.)
                text = policy_prompt(executor.state("predict Y", SUMMARY))
                self.assertNotIn("_respond", text)
                state["private_score"] = 1.
                with self.assertRaises(ValueError):
                    policy_prompt(state)

    def test_binding_joint_constraints_and_stop(self):
        for mode in ("mismatch", "joint", "stop", "parse", "prefix"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                calls = []
                path = Path(td)/"journal"
                with ChargedExecutor(GRAPH, MENU, 4, path, lambda *args: calls.append(args),
                                     joint_feasible=lambda a: a.action_id != "a1") as executor:
                    if mode == "stop":
                        executor.execute(Decision(None, True), '{"stop":true}')
                    else:
                        with self.assertRaises(ValueError):
                            if mode == "prefix":
                                executor.execute_generated([1, 2], [1, 3, 4], lambda x: "{}")
                            else:
                                text = {"mismatch": '{"action_id":"a0"}',
                                        "joint": '{"action_id":"a1"}', "parse": "not an action"}[mode]
                                executor.execute(Decision("a1", False), text)
                    with self.assertRaises(RuntimeError):
                        executor.execute(Decision("a0", False), '{"action_id":"a0"}')
                    self.assertEqual(calls, [])
                terminal = json.loads(path.read_text().splitlines()[-1])
                self.assertEqual(terminal["event"], "stop" if mode == "stop" else "rejected")
                self.assertEqual(terminal["spent_units"], 0)

    def test_unhandled_driver_error_is_not_normal_close(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"journal"
            with self.assertRaises(RuntimeError):
                with ChargedExecutor(GRAPH, MENU, 4, path, lambda *args: {}) as executor:
                    raise RuntimeError("artificial generation failure")
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])["event"], "driver_failure")

    def test_generated_rejections_retain_text_and_bad_token_attempt(self):
        for mode in ("malformed", "tokens"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                path = Path(td)/"journal"
                with ChargedExecutor(GRAPH, MENU, 4, path, lambda *args: {}) as executor:
                    with self.assertRaises(ValueError):
                        executor.execute_generated([1, 2], [1, 2, 3] if mode == "malformed" else [float("nan")],
                                                   lambda ids: "This is not an action")
                    with self.assertRaises(RuntimeError):
                        executor.execute(Decision("a0", False), '{"action_id":"a0"}')
                events = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(events[-1]["event"], "rejected")
                self.assertEqual(events[-1]["proposal_id"], 0)
                self.assertEqual(events[-1]["spent_units"], 0)
                decoded = [e for e in events if e["event"] == "decoded_completion"]
                if mode == "malformed":
                    self.assertEqual(decoded[0]["completion"], "This is not an action")
                else:
                    self.assertEqual(events[-1]["stage"], "token_validation")

    def test_scopes_and_streams_are_separate(self):
        values = [stream_seed(scope, 0, stream) for scope in
                  ("artificial", "policy_train", "development", "confirmation")
                  for stream in ("mechanism", "initial", "measurement", "policy", "private_eval")]
        self.assertEqual(len(set(values)), len(values))
        self.assertEqual(stream_seed("policy_train", 1, "train_score"), stream_seed("policy_train", 1, "train_score"))
        for scope in ("development", "confirmation", "artificial"):
            with self.assertRaises(ValueError):
                stream_seed(scope, 0, "train_score")


if __name__ == "__main__":
    unittest.main()
