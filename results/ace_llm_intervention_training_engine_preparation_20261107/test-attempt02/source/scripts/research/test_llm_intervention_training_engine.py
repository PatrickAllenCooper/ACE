"""Tiny arithmetic controls only; no real language model or accelerator calls."""
from contextlib import ExitStack
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from llm_intervention_contract import Action, Node, VERSION as CONTRACT_VERSION, canonical, digest
from llm_intervention_learning import LoRALinear
from llm_intervention_canonical_scoring import install_canonical_projection
from llm_intervention_supervision import VERSION as RECORD_VERSION
from llm_intervention_task import PredictionTask, make_envelope
import llm_intervention_training_engine as engine


def save(path, value):
    with path.open("x") as f:
        f.write(canonical(value)+"\n")


def read(path):
    return json.loads(path.read_text())


class Tokenizer:
    pad_token_id, eos_token_id = 0, 9
    commands = {'{"action_id":"a"}': 3, '{"action_id":"b"}': 4, '{"stop":true}': 5}
    def apply_chat_template(self, messages, **kwargs):
        return [1, 6 if "artificial-one" in messages[0]["content"] else 2]
    def encode(self, text, **kwargs):
        return [self.commands[text]]
    def decode(self, ids, **kwargs):
        return next(k for k, v in self.commands.items() if list(ids) == [v])


def prepared(index, *, tied=False):
    graph = (Node("X", (), "unit", (-1., 1.)), Node("Y", ("X",), "unit"))
    menu = (Action("a", "intervene", (("X", 0.),), ("Y",), 1),
            Action("b", "intervene", (("X", 1.),), ("Y",), 1))
    summaries = [{"node": n.name, "terms": ["intercept", *n.parents],
        "parameter_mean": [0.]*(len(n.parents)+1),
        "parameter_covariance": [[float(i == j) for j in range(len(n.parents)+1)] for i in range(len(n.parents)+1)],
        "noise_variance": 1., "eligible_rows": 0} for n in graph]
    state = json.loads(canonical({"version": CONTRACT_VERSION, "objective": "artificial-"+("one" if index else "zero"),
        "domain_notes": "", "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "scm": summaries, "history": [], "spent_units": 0, "remaining_units": 4,
        "requests": 0, "returned_responses": 0, "terminal": False}))
    task = PredictionTask("Y", menu, (.5, .5), 1., .04)
    record = {"version": RECORD_VERSION, "scope": "artificial", "world_id": index,
        "envelope": make_envelope(state, task), "supervised_action_id": "b" if index else "a",
        "preferences": [{"left": "a", "right": "b", "tie": tied,
            "chosen": None if tied else "a", "rejected": None if tied else "b"}],
        "supervision_receipt_sha256": "1"*64}
    raw = canonical(record).encode()
    return engine.prepare_record(record_id="r"+str(index), raw=raw,
        record_sha256=engine.hashlib.sha256(raw).hexdigest(), supervision_receipt_sha256="1"*64,
        scope="artificial", tokenizer=Tokenizer(), tokenizer_sha256="2"*64,
        numeric_recipe=engine.gpu09_numeric_recipe())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.q_proj = LoRALinear(nn.Linear(2, 2, bias=False, dtype=torch.bfloat16), rank=8, alpha=16.)
            self.v_proj = LoRALinear(nn.Linear(2, 2, bias=False, dtype=torch.bfloat16), rank=8, alpha=16.)
            self.lm_head = nn.Linear(2, 12, bias=False, dtype=torch.bfloat16)
        self.lm_head.requires_grad_(False)
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.fill_(0. if name.endswith(".B") else .1)
        install_canonical_projection(self)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    def forward(self, *args, **kwargs):
        raise AssertionError("real language-model scoring is not authorized in these tests")


class CounterControl:
    def __init__(self, model, *, expected_layers):
        self.enabled, self.forward_calls, self.recompute_calls = False, 0, 0
        self.layers = expected_layers
        model.control = self
    def set_enabled(self, enabled):
        self.enabled = enabled
    def evidence(self):
        return {"enabled": self.enabled, "forward_calls": self.forward_calls,
                "recompute_calls": self.recompute_calls, "scope": "artificial mocked counters only"}


class Arithmetic:
    def __init__(self, *, fail_at=None, mutate_reference_at=None):
        self.policy_calls, self.reference_calls = 0, 0
        self.fail_at, self.mutate_reference_at = fail_at, mutate_reference_at
    def scores(self, model, batch, *, selected):
        assert selected is True
        reference = model.q_proj.reference
        self.reference_calls += int(reference)
        self.policy_calls += int(not reference)
        if not reference and self.policy_calls == self.mutate_reference_at:
            model.q_proj.reference_A.add_(1.)
        values = [getattr(m, name) for m in (model.q_proj, model.v_proj)
                  for name in (("reference_A", "reference_B") if reference else ("A", "B"))]
        weight = sum(v.sum() for v in values)
        targets = batch.labels[:, 2].float()/10 + batch.input_ids[:, 1].float()/100
        scores = -(weight-targets)**2
        if not reference and self.policy_calls == self.fail_at:
            scores = scores*float("nan")
        if scores.requires_grad:
            model.control.forward_calls += model.control.layers
            def backward(grad):
                model.control.recompute_calls += model.control.layers
                return grad
            scores.register_hook(backward)
        return scores
    def sft(self, model, batch, *, selected):
        return -(self.scores(model, batch, selected=selected)/(batch.labels != -100).sum(1)).mean()


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(os.environ["ACE_TRAINING_ENGINE_TEST_OUTPUT"])
        cls.root.mkdir(exist_ok=False)
        cls.records = (prepared(0), prepared(1))
        save(cls.root/"prepared-artificial-records.json", cls.records)

    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def plan(self, **overrides):
        shapes = {name+suffix: shape for name in ("q_proj", "v_proj")
                  for suffix, shape in ((".A", [8, 2]), (".B", [2, 8]))}
        records = overrides.get("records", self.records)
        params = dict(records=records, record_pins={r["record_id"]: digest(r) for r in records},
            schedule=({"stage": "sft", "record_id": "r0", "pair_index": None},
                      {"stage": "sft", "record_id": "r1", "pair_index": None},
                      {"stage": "dpo", "record_id": "r0", "pair_index": 0},
                      {"stage": "dpo", "record_id": "r1", "pair_index": 0}),
            numeric_recipe=engine.gpu09_numeric_recipe(), scope="artificial", execution_kind="artificial_control",
            expected_adapter_shapes=shapes, max_trainable_parameters=64, checkpoint_steps=(0,),
            outer_provenance_receipt_sha256="3"*64)
        params.update(overrides)
        return engine.prepare_training_plan(**params)

    def run_fit(self, name="fit", *, model=None, plan=None, arithmetic=None):
        model, plan, arithmetic = model or TinyModel(), plan or self.plan(), arithmetic or Arithmetic()
        output = self.directory/name
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine, "DecoderCheckpointing", CounterControl))
            stack.enter_context(patch.object(engine, "sequence_log_probs_canonical", arithmetic.scores))
            stack.enter_context(patch.object(engine, "supervised_loss_canonical", arithmetic.sft))
            stack.enter_context(patch.object(torch.cuda, "init", side_effect=AssertionError("GPU forbidden")))
            result = engine.fit(model=model, plan=plan, plan_sha256=digest(plan), output=output)
        return result, model, output, arithmetic

    def test_fixed_sft_dpo_reference_checkpoints_counts_and_clipping(self):
        result, model, out, arithmetic = self.run_fit()
        self.assertEqual((result["completed_steps"], result["sft_steps"], result["dpo_steps"]), (4, 2, 2))
        self.assertTrue(result["one_pass_sft"] and result["one_pass_dpo"])
        self.assertEqual(result["base_before"], result["base_after"])
        self.assertEqual(result["projection_before"], result["projection_after"])
        self.assertFalse(result["provenance_qualified"])
        self.assertFalse(model.control.enabled)
        self.assertEqual((arithmetic.policy_calls, arithmetic.reference_calls), (4, 4))
        initial = torch.load(out/"initial.adapter.pt", weights_only=True)
        sft = torch.load(out/"sft.adapter.pt", weights_only=True)
        final = torch.load(out/"final.adapter.pt", weights_only=True)
        reference = torch.load(out/"reference.pt", weights_only=True)
        self.assertTrue(any(not torch.equal(initial[n], sft[n]) for n in sft))
        self.assertTrue(any(not torch.equal(sft[n], final[n]) for n in sft))
        for name, value in sft.items():
            prefix, suffix = name.rsplit(".", 1)
            self.assertTrue(torch.equal(value, reference[prefix+".reference_"+suffix]))
            self.assertTrue(torch.equal(reference[prefix+".reference_"+suffix], dict(model.named_buffers())[prefix+".reference_"+suffix]))
        self.assertEqual({int(v["step"]) for v in torch.load(out/"sft.optimizer.pt", weights_only=True)["state"].values()}, {2})
        self.assertEqual({int(v["step"]) for v in torch.load(out/"final.optimizer.pt", weights_only=True)["state"].values()}, {4})
        self.assertEqual([read(out/f"step-{i:05d}.complete.json")["stage"] for i in range(4)], ["sft", "sft", "dpo", "dpo"])
        first = read(out/"step-00000.complete.json")
        self.assertGreater(first["gradient_norm_before_clip"], 1.)
        self.assertLessEqual(first["gradient_norm_after_clip"], 1.000001)
        self.assertEqual(result["checkpoint_usage"]["recompute_calls"], 8)
        self.assertTrue((out/"step-00000.adapter.pt").exists())
        self.assertFalse((out/"step-00001.adapter.pt").exists())
        before = {str(p.relative_to(out)): engine._file_pin(p) for p in out.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.run_fit(model=TinyModel())
        self.assertEqual(before, {str(p.relative_to(out)): engine._file_pin(p) for p in out.rglob("*") if p.is_file()})

    def test_absent_pairs_data_state_association_and_fixed_schedule_reject(self):
        wrong = copy.deepcopy(self.records[0])
        wrong["token_metadata"]["envelope_sha256"] = self.records[1]["token_metadata"]["envelope_sha256"]
        tied = prepared(0, tied=True)
        variants = [dict(records=(wrong, self.records[1])), dict(records=(tied, self.records[1])),
            dict(schedule=({"stage": "dpo", "record_id": "r0", "pair_index": 0},
                           {"stage": "sft", "record_id": "r0", "pair_index": None})),
            dict(schedule=({"stage": "sft", "record_id": "r0", "pair_index": None},
                           {"stage": "dpo", "record_id": "r0", "pair_index": 9})),
            dict(record_pins={"r0": "0"*64, "r1": digest(self.records[1])}),
            dict(scope="development"), dict(scope="policy_train"),
            dict(numeric_recipe=dict(engine.gpu09_numeric_recipe(), learning_rate=.01))]
        errors = []
        for i, variant in enumerate(variants):
            with self.subTest(index=i), self.assertRaises(ValueError) as error:
                self.plan(**variant)
            errors.append({"case": i, "message": str(error.exception)})
        save(self.directory/"rejections.json", errors)
        # Repetitions are permitted ONLY as declared exposures, never inferred.
        schedule = tuple(self.plan()["schedule"])
        repeated = self.plan(schedule=schedule[:1]+schedule)
        self.assertFalse(repeated["one_pass_sft"])
        self.assertEqual(repeated["exposures"]["r0"]["sft"], 2)
        save(self.directory/"explicit-repeat-plan.json", repeated)

    def test_parameter_bound_inventory_and_pin_reject_before_updates(self):
        with self.assertRaisesRegex(ValueError, "parameter bound"):
            self.plan(max_trainable_parameters=63)
        model = TinyModel()
        model.q_proj.base.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.run_fit(model=model)
        self.assertFalse((self.directory/"fit").exists())
        model, plan = TinyModel(), self.plan()
        with self.assertRaisesRegex(ValueError, "pinned"):
            engine.fit(model=model, plan=plan, plan_sha256="0"*64, output=self.directory/"badpin")
        altered = copy.deepcopy(plan)
        altered["schedule"] = list(reversed(altered["schedule"]))
        with self.assertRaises(ValueError):
            engine.fit(model=model, plan=altered, plan_sha256=digest(altered), output=self.directory/"altered")
        save(self.directory/"rejections.json", {"parameter_cap": True, "extra_trainable_base": True,
            "incorrect_plan_pin": True, "reordered_schedule_even_with_new_pin": True})

    def test_reference_mutation_is_retained_and_rejected_before_dpo_update(self):
        with self.assertRaisesRegex(ValueError, "reference changed"):
            self.run_fit(arithmetic=Arithmetic(mutate_reference_at=3))
        out = self.directory/"fit"
        failure = read(out/"failure.json")
        self.assertEqual(len(failure["completed_steps"]), 2)
        self.assertFalse(failure["failed_attempt"]["optimizer_step_started"])
        old = torch.load(out/"reference.pt", weights_only=True)
        partial = torch.load(out/"failed-partial.reference.pt", weights_only=True)
        self.assertFalse(torch.equal(old["q_proj.reference_A"], partial["q_proj.reference_A"]))
        self.assertTrue((out/"sft.adapter.pt").exists())
        self.assertFalse((out/"terminal.json").exists())

    def test_failed_step_retains_prior_steps_nonfinite_scores_and_partial_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "finite aligned"):
            self.run_fit(arithmetic=Arithmetic(fail_at=4))
        out = self.directory/"fit"
        failure = read(out/"failure.json")
        self.assertEqual(len(failure["completed_steps"]), 3)
        self.assertEqual(failure["failed_attempt"]["index"], 3)
        self.assertFalse(failure["failed_attempt"]["optimizer_step_started"])
        self.assertFalse(failure["retry_allowed"])
        # The existing DPO helper rejects nonfinite scores before constructing
        # a loss. Preserve that earlier failure; do not invent a loss artifact.
        self.assertIn("finite aligned", failure["message"])
        self.assertFalse((out/"step-00003.loss.json").exists())
        self.assertTrue(failure["partial_checkpoint_saved"])
        self.assertTrue((out/"failed-partial.gradients.pt").exists())
        self.assertFalse((out/"final.adapter.pt").exists())

    def test_ambiguous_optimizer_failure_keeps_partial_state_without_retry(self):
        original, calls = torch.optim.AdamW.step, []
        def interrupted(optimizer, *args, **kwargs):
            calls.append(len(calls))
            if len(calls) == 3:
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][0].add_(.5)
                raise RuntimeError("artificial mid-update interruption")
            return original(optimizer, *args, **kwargs)
        with patch.object(torch.optim.AdamW, "step", interrupted), self.assertRaisesRegex(RuntimeError, "mid-update"):
            self.run_fit()
        out = self.directory/"fit"
        failure = read(out/"failure.json")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(failure["completed_steps"]), 2)
        self.assertEqual(failure["update_outcome"], "unknown_or_partial")
        self.assertTrue(failure["failed_attempt"]["optimizer_step_started"])
        self.assertFalse(failure["failed_attempt"]["optimizer_step_returned"])
        self.assertTrue((out/"step-00002.optimizer-attempt.json").exists())
        self.assertFalse((out/"step-00002.update-returned.json").exists())
        partial = torch.load(out/"failed-partial.adapter.pt", weights_only=True)
        sft = torch.load(out/"sft.adapter.pt", weights_only=True)
        self.assertTrue(torch.equal(partial["q_proj.A"], sft["q_proj.A"]+.5))


if __name__ == "__main__":
    print("Interpreter:", sys.executable, flush=True)
    print("Version:", sys.version.replace("\n", " "), flush=True)
    print("Torch:", torch.__version__, flush=True)
    print("Retained fixtures:", os.environ["ACE_TRAINING_ENGINE_TEST_OUTPUT"], flush=True)
    unittest.main(verbosity=2)
