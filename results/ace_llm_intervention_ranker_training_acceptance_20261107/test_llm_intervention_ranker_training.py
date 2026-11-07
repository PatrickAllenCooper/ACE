"""New tiny CPU-ranker controls, with no pretrained model or scientific world."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from llm_intervention_contract import Node, Action, VERSION as CONTRACT_VERSION, canonical, digest
from llm_intervention_task import PredictionTask, make_envelope
from llm_intervention_supervision import VERSION as RECORD_VERSION, matched_views
from llm_intervention_training_exposure import propose_exposure
from llm_intervention_ranker import CandidateRanker, feature_tensor
import llm_intervention_ranker_training as training


def fixture(index, *, terminal=False):
    graph = (Node("X", (), "unit", (-1., 1.)),)
    menu = (Action("a", "intervene", (("X", -1.),), ("X",), 1),
            Action("b", "intervene", (("X", 1.),), ("X",), 1))
    state = {"version": CONTRACT_VERSION, "objective": "artificial numerical training control", "domain_notes": "",
        "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "scm": [{"node": "X", "terms": ["intercept"], "parameter_mean": [0.],
                 "parameter_covariance": [[1.]], "noise_variance": 1., "eligible_rows": 0}],
        "history": [], "spent_units": 0, "remaining_units": 0 if terminal else 2,
        "requests": 0, "returned_responses": 0, "terminal": terminal}
    pairs = [] if terminal else [{"left": "a", "right": "b", "tie": False, "chosen": "b", "rejected": "a"},
        {"left": "a", "right": None, "tie": True, "chosen": None, "rejected": None},
        {"left": "b", "right": None, "tie": False, "chosen": "b", "rejected": None}]
    return {"version": RECORD_VERSION, "scope": "artificial", "world_id": index,
        "envelope": make_envelope(json.loads(canonical(state)), PredictionTask("X", menu, (.5, .5), 1., .04)),
        "supervised_action_id": None if terminal else "b", "preferences": pairs,
        "supervision_receipt_sha256": hashlib.sha256(str(index).encode()).hexdigest()}


def arguments():
    records = {"first": canonical(fixture(0)).encode(), "terminal": canonical(fixture(1, terminal=True)).encode()}
    args = {"records": records, "record_pins": {r: hashlib.sha256(b).hexdigest() for r, b in records.items()},
        "supervision_receipt_pins": {r: json.loads(b)["supervision_receipt_sha256"] for r, b in records.items()},
        "record_roster": tuple(records), "scope": "artificial"}
    exposure = propose_exposure(**args)
    return {**args, "exposure": exposure, "exposure_sha256": digest(exposure),
        "recipe": training.numeric_recipe(artificial=True), "checkpoint_steps": (0,),
        "outer_provenance_receipt_sha256": "b"*64}


def read(path):
    return json.loads(path.read_bytes())


def tensors(path):
    return torch.load(path, weights_only=True, map_location="cpu")


class NumericTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.root = Path(os.environ["ACE_NUMERIC_TRAINING_TEST_OUTPUT"])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.output = self.root/self._testMethodName
        self.output.mkdir()
        self.args = arguments()
        self.plan = training.prepare_plan(**self.args)
        training._save(self.output/"input-plan.json", self.plan)

    def run_fit(self):
        return training.fit(plan=self.plan, plan_sha256=digest(self.plan), output=self.output/"fit")

    def test_complete_loop_matches_independent_adam_and_retains_terminal_row(self):
        result = self.run_fit()
        self.assertEqual((result["sft_steps"], result["dpo_steps"], result["zero_gradient_steps"]), (2, 1, 1))
        model = CandidateRanker(self.plan["feature_width"], hidden=4, seed=109).eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., amsgrad=False, foreach=False, fused=False)
        data = {r: feature_tensor(matched_views(b, record_sha256=self.args["record_pins"][r],
            supervision_receipt_sha256=self.args["supervision_receipt_pins"][r],
            allowed_scope="artificial")["numeric"]["features"]) for r, b in self.args["records"].items()}
        ref, losses = None, []
        for step in self.plan["exposure"]["numeric_schedule"]:
            if step["stage"] == "dpo" and ref is None:
                ref = copy.deepcopy(model).eval().requires_grad_(False)
                for n, p in ref.state_dict().items():
                    self.assertTrue(torch.equal(p, tensors(self.output/"fit/sft.state.pt")[n]))
            optimizer.zero_grad(set_to_none=True)
            logs = F.log_softmax(model(data[step["record_id"]]), dim=0)
            a = step["chosen_index"]
            if step["stage"] == "sft":
                loss = -logs[a]
            else:
                b = step["rejected_index"]
                with torch.no_grad():
                    frozen = F.log_softmax(ref(data[step["record_id"]]), dim=0)
                loss = -F.logsigmoid(.1*((logs[a]-frozen[a])-(logs[b]-frozen[b])))
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        expected = tensors(self.output/"fit/final.state.pt")
        self.assertTrue(all(torch.equal(p, expected[n]) for n, p in model.state_dict().items()))
        self.assertEqual(losses, [s["loss"] for s in read(self.output/"fit/training_complete.json")["steps"]])
        state = tensors(self.output/"fit/final.optimizer.pt")
        self.assertTrue(all(v["step"].item() == 3 for v in state["state"].values()))
        self.assertTrue(all(v["all_training_score_rows_equal"] for v in result["exports"].values()))
        self.assertFalse(result["scientific_provenance_qualified"])
        self.assertNotEqual(read(self.output/"fit/training_complete.json")["initial_signature"],
                            read(self.output/"fit/training_complete.json")["final_signature"])

    def test_ambiguous_optimizer_update_is_retained_without_retry(self):
        original, calls = torch.optim.Adam.step, []
        def fail(optimizer, *args, **kwargs):
            calls.append(1)
            out = original(optimizer, *args, **kwargs)
            if len(calls) == 3:
                raise RuntimeError("injected after actual update")
            return out
        with patch.object(torch.optim.Adam, "step", fail), self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_fit()
        failure = read(self.output/"fit/failure.json")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(failure["completed_steps"]), 2)
        self.assertEqual(failure["update_outcome"], "unknown_or_partial")
        self.assertTrue(failure["partial_state_saved"])
        self.assertFalse((self.output/"fit/terminal.json").exists())
        self.assertTrue(all(v["step"].item() == 3 for v in tensors(self.output/"fit/failed-partial.optimizer.pt")["state"].values()))

    def test_reference_mutation_fails_and_keeps_complete_prefix(self):
        original_copy, original_step, refs, calls = copy.deepcopy, torch.optim.Adam.step, [], []
        def capture(value, *args, **kwargs):
            out = original_copy(value, *args, **kwargs)
            if type(value) is CandidateRanker:
                refs.append(out)
            return out
        def mutate(optimizer, *args, **kwargs):
            calls.append(1)
            out = original_step(optimizer, *args, **kwargs)
            if len(calls) == 3:
                with torch.no_grad(): next(refs[0].parameters()).add_(1.)
            return out
        with patch.object(training.copy, "deepcopy", capture), patch.object(torch.optim.Adam, "step", mutate):
            with self.assertRaisesRegex(ValueError, "reference changed"):
                self.run_fit()
        self.assertEqual(len(read(self.output/"fit/failure.json")["completed_steps"]), 3)
        self.assertTrue((self.output/"fit/failed-partial.reference.pt").exists())
        self.assertFalse((self.output/"fit/terminal.json").exists())

    def test_export_failure_never_becomes_terminal_success(self):
        with patch.object(training, "load_ranker", side_effect=ValueError("injected reload failure")):
            with self.assertRaisesRegex(ValueError, "reload failure"):
                self.run_fit()
        self.assertTrue((self.output/"fit/training_complete.json").exists())
        self.assertEqual(read(self.output/"fit/training_complete.json")["status"], "training_complete_export_pending")
        self.assertTrue((self.output/"fit/failure.json").exists())
        self.assertFalse((self.output/"fit/terminal.json").exists())

    def test_nonfinite_loss_is_retained_before_optimizer_attempt(self):
        with patch.object(training, "supervised_rank_loss", side_effect=lambda scores, _: scores.sum()*float("nan")):
            with self.assertRaisesRegex(ValueError, "nonfinite numerical loss"):
                self.run_fit()
        self.assertEqual(read(self.output/"fit/failure.json")["completed_steps"], [])
        self.assertIsNone(read(self.output/"fit/step-0000.loss.json")["loss"])
        self.assertFalse(read(self.output/"fit/failure.json")["attempt"]["optimizer_started"])
        self.assertFalse((self.output/"fit/step-0000.optimizer-attempt.json").exists())

    def test_invalid_complete_inputs_reject_before_model_initialization(self):
        cases = {}
        bad = copy.deepcopy(self.args); bad["records"]["first"] += b" "; cases["record_pin"] = bad
        bad = copy.deepcopy(self.args); bad["exposure"]["numeric_schedule"][0]["chosen_index"] = 0
        bad["exposure_sha256"] = digest(bad["exposure"]); cases["altered_schedule"] = bad
        bad = copy.deepcopy(self.args); bad["recipe"]["hidden"] = 5; cases["altered_recipe"] = bad
        bad = copy.deepcopy(self.args); bad["scope"] = "development"; cases["forbidden_scope"] = bad
        bad = copy.deepcopy(self.args); bad["checkpoint_steps"] = (0, 0); cases["duplicate_checkpoint"] = bad
        with patch.object(training, "CandidateRanker", side_effect=AssertionError("must not initialize")):
            for name, args in cases.items():
                with self.subTest(name=name), self.assertRaises(ValueError): training.prepare_plan(**args)
                preserved = {k: v for k, v in args.items() if k != "records"}
                preserved["records"] = {r: b.decode() for r, b in args["records"].items()}
                training._save(self.output/(name+".json"), preserved)
            changed = copy.deepcopy(self.plan); changed["feature_pins"]["first"]["float32_tensor_sha256"] = "0"*64
            training._save(self.output/"changed_feature_plan.json", changed)
            with self.assertRaisesRegex(ValueError, "features or plan"):
                training.fit(plan=changed, plan_sha256=digest(changed), output=self.output/"must_not_exist")
            with self.assertRaisesRegex(ValueError, "pinned numerical plan"):
                training.fit(plan=self.plan, plan_sha256="0"*64, output=self.output/"must_not_exist")
        self.assertFalse((self.output/"must_not_exist").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
