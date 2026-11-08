"""Pure evidence controls only: no tensor libraries, model, CUDA or training."""
import builtins
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest


SOURCE = Path(__file__).with_name("llm_intervention_training_gpu_worker.py")
PREPARED = SOURCE.parents[2] / "results/ace_llm_intervention_training_cpu_20261108/terminal01/remote/qualification/session/worker/artifacts/prepared.json"


class ArtifactValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = json.loads(PREPARED.read_bytes())
        cls.imports = []
        real_import = builtins.__import__
        def stdlib_only(name, *args, **kwargs):
            cls.imports.append(name)
            if name.split(".")[0] in {"torch", "numpy", "scipy", "transformers"} or name.startswith("llm_"):
                raise AssertionError("numeric/project import during pure validation: " + name)
            return real_import(name, *args, **kwargs)
        # Match the parent's captured-source loader; importing the worker must
        # not require a model runtime or even a normal importable module name.
        cls.worker = types.ModuleType("captured_worker")
        cls.worker.__dict__["__builtins__"] = {**vars(builtins), "__import__": stdlib_only}
        exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), cls.worker.__dict__)

    def test_accepted_prepared_pin_and_detachment(self):
        plan = self.worker._plan(self.prepared)
        self.assertEqual(plan["sft_steps"], 4)
        self.assertEqual(plan["dpo_steps"], 2)
        self.assertEqual(len(plan["expected_adapter_shapes"]), 112)
        plan["records"][0]["prompt_ids"][0] = -1
        self.assertNotEqual(plan, self.prepared["training_plan"])
        changed = copy.deepcopy(self.prepared)
        changed["training_plan"]["scope"] = "policy_train"
        changed["training_plan_sha256"] = self.worker._digest(changed["training_plan"])
        with self.assertRaisesRegex(ValueError, "accepted artificial"):
            self.worker._plan(changed)

    def test_engine_reference_schema_is_pair_keys_and_ordered_scores(self):
        pairs = self.worker._pair_specs(self.prepared["training_plan"])
        self.assertEqual(set(pairs), {'["artificial/17/snapshot-000",0]', '["artificial/4/snapshot-000",1]'})
        self.assertEqual(pairs['["artificial/4/snapshot-000",1]'][1],
                         ['{"action_id":"c"}', '{"action_id":"b"}'])
        reference = {key: [-2.25, -7.5] for key in pairs}
        self.assertEqual(self.worker._score_difference(reference, reference), 0.)
        changed = copy.deepcopy(reference)
        changed[next(iter(changed))][0] += 1e-7
        self.assertGreater(self.worker._score_difference(changed, reference), 0.)
        changed = {key: values[::-1] for key, values in reference.items()}
        self.assertGreater(self.worker._score_difference(changed, reference), 0.)
        for malformed in ({"artificial/4/snapshot-000": [-2.25, -7.5]},
                          {key: [float("nan"), -7.5] for key in pairs},
                          {key: [-2.25] for key in pairs}):
            with self.assertRaises(ValueError):
                self.worker._score_difference(malformed, reference)

    def test_generated_tokens_bind_to_exact_prompt_and_command(self):
        record = self.prepared["training_plan"]["records"][0]
        action, other, stop = '{"action_id":"a"}', '{"action_id":"b"}', '{"stop":true}'
        sequence = record["prompt_ids"] + record["commands"][action]
        self.assertFalse(self.worker._legal_sequence(record, sequence, action))
        self.assertTrue(self.worker._legal_sequence(record, record["prompt_ids"] + record["commands"][stop], stop))
        for wrong, text in ((sequence, other), ([0] + sequence[1:], action), (sequence[:-1], action),
                            (sequence + [record["eos_token_id"]], action)):
            with self.assertRaises(ValueError):
                self.worker._legal_sequence(record, wrong, text)
        terminal = self.prepared["training_plan"]["records"][1]
        with self.assertRaises(ValueError):
            self.worker._legal_sequence(terminal, terminal["prompt_ids"] + record["commands"][action], action)

    def test_parent_validator_rejects_partial_and_failed_outputs_without_heavy_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing or linked evidence"):
                self.worker.validate_outputs(root, self.prepared)
            (root / "evidence-files.json").write_text("{}\n")
            (root / "summary.json").write_text('{"status":"worker_complete"}\n')
            with self.assertRaisesRegex(ValueError, "missing or linked evidence"):
                self.worker.validate_outputs(root, self.prepared)
            (root / "failure.json").write_text('{"status":"failed"}\n')
            with self.assertRaisesRegex(ValueError, "missing/failed"):
                self.worker.validate_outputs(root, self.prepared)
        self.assertFalse(any(name.startswith(("torch", "numpy", "transformers", "llm_")) for name in self.imports))


if __name__ == "__main__":
    unittest.main()
