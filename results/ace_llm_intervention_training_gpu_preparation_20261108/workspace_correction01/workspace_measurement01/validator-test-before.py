"""Pure evidence controls only: no tensor libraries, model, CUDA or training."""
import builtins
import copy
import hashlib
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
            with self.assertRaisesRegex(ValueError, "evidence file inventory differs"):
                self.worker.validate_outputs(root, self.prepared)
            (root / "failure.json").write_text('{"status":"failed"}\n')
            with self.assertRaisesRegex(ValueError, "missing/failed"):
                self.worker.validate_outputs(root, self.prepared)
        self.assertFalse(any(name.startswith(("torch", "numpy", "transformers", "llm_")) for name in self.imports))

    @staticmethod
    def _fixture_write(root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())

    @staticmethod
    def _fixture_pin(root, name):
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    def _fixture_manifest(self, root):
        manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*") if p.is_file() and p.name not in ("summary.json", "evidence-files.json")}
        self._fixture_write(root, "evidence-files.json", manifest)

    def _complete_fixture(self, root):
        """Fabricated JSON and opaque .pt bytes; no engine, model or tensor load.

        Construct the writer's full bundle independently of the validator's
        inventory and summary helpers, using only the accepted prepared plan.
        """
        plan = self.prepared["training_plan"]
        write = lambda name, value: self._fixture_write(root, name, value)
        pin = lambda name: self._fixture_pin(root, name)
        signature = lambda text: hashlib.sha256(text.encode()).hexdigest()
        base, projection, reference = map(signature, ("fixture base", "fixture projection", "fixture reference"))
        checkpoints = {}
        for stem, stage, count in (("initial", "initial", 0), ("sft", "sft", 4), ("final", "final", 6),
                                   ("step-00003", "sft", 4), ("step-00005", "final", 6)):
            files = {}
            for kind in (("adapter",) if stage == "initial" else ("adapter", "optimizer")):
                name = f"training/{stem}.{kind}.pt"
                write(name, f"opaque fixture {stage} {kind}\n".encode())
                files[kind] = pin(name)
            checkpoints[stem] = {"files": files, "adapter_signature": signature("fixture " + stage),
                                 "completed_steps": count, "provenance_qualified": False}
            write(f"training/{stem}.checkpoint.json", checkpoints[stem])
        write("training/plan.json", plan)
        write("training/initial-signatures.json", {"base": base, "projection": projection,
              "adapter": checkpoints["initial"]["adapter_signature"], "adapter_parameters": 1146880})
        steps = []
        for index, scheduled in enumerate(plan["schedule"]):
            stem = f"training/step-{index:05d}"
            identity = {"index": index, **scheduled}
            write(stem + ".start.json", {**identity, "record_pin": plan["record_pins"][scheduled["record_id"]],
                  "optimizer_step_started": False, "optimizer_step_returned": False})
            write(stem + ".optimizer-attempt.json", {**identity, "optimizer_step_started": True,
                  "optimizer_step_returned": False})
            returned = {**identity, "optimizer_step_started": True, "optimizer_step_returned": True}
            write(stem + ".update-returned.json", returned)
            write(stem + ".loss.json", {"finite": True, "loss": 0.5})
            write(stem + ".gradients.json", {"checkpoint_recompute_delta": 28,
                  "tensors": [{"name": n, "shape": shape, "present": True, "finite": True}
                              for n, shape in plan["expected_adapter_shapes"].items()]})
            write(stem + ".update-intent.json", {"gradient_norm_before_clip": 1.2,
                  "gradient_norm_after_clip": 1., "clip_limit": 1.})
            rec = next(r for r in plan["records"] if r["record_id"] == scheduled["record_id"])
            commands = ([rec["supervised_command"]] if scheduled["stage"] == "sft" else
                        rec["strict_pairs"][scheduled["pair_index"]])
            tokens = sum(len(rec["commands"][c]) for c in commands)
            event = {**returned, "loss": 0.5, "gradient_norm_before_clip": 1.2, "gradient_norm_after_clip": 1.,
                     "checkpoint_recompute_delta": 28, "rows": len(commands), "action_tokens": tokens,
                     "input_tokens": len(commands) * len(rec["prompt_ids"]) + tokens, "cuda_elapsed_ms": 1.,
                     "wall_seconds": 0.5, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
            write(stem + ".complete.json", event)
            steps.append(event)
        reference_scores = {json.dumps([s["record_id"], s["pair_index"]], separators=(",", ":")): [-2.25, -7.5]
                            for s in plan["schedule"] if s["stage"] == "dpo"}
        write("training/reference.pt", b"opaque fixture reference\n")
        write("training/reference-snapshot.json", {"signature": reference, "reference_sha256": pin("training/reference.pt"),
              "sft_adapter_sha256": checkpoints["sft"]["files"]["adapter"], "completed_sft_steps": 4,
              "policy_optimizer_updates_after_snapshot": 0})
        write("training/reference-scores.json", {"signature": reference, "unique_pair_forwards": 2, "scores": reference_scores})
        write("training/terminal.json", {"status": "complete", "plan_sha256": self.prepared["training_plan_sha256"],
              "execution_kind": "qwen_gpu09", "completed_steps": 6, "sft_steps": 4, "dpo_steps": 2,
              "exposures": plan["exposures"], "one_pass_sft": True, "one_pass_dpo": False,
              "reference_scores_unchanged": True, "reference_pair_count": 2, "reference_signature": reference,
              "provenance_qualified": False, "retry_allowed": False, "environment_responses": 0,
              "outer_provenance_receipt_sha256": plan["outer_provenance_receipt_sha256"],
              "base_before": base, "base_after": base, "projection_before": projection, "projection_after": projection,
              "checkpoint_usage": {"enabled": True, "use_reentrant": False, "preserve_rng_state": True,
                                   "native_checkpointing": False, "recompute_calls": 168, "forward_calls": 168},
              "elapsed_seconds": 8.})
        memory = {"peak_gpu_allocated_bytes": 100, "peak_gpu_reserved_bytes": 200}
        write("first-gpu-work.json", {"monotonic_seconds": 20., "training_updates": 0, "forward_count": 1,
              "record_id": plan["records"][0]["record_id"], "device": "cuda:0", "cuda_interval_ms": 1.,
              "scores": [-2.25], **memory})
        write("arithmetic.json", {"recipe": plan["numeric_recipe"], "strict": True, "warn_only": False,
              "float32_matmul": "highest", "allow_tf32": False, "cublas_workspace_config": ":4096:8",
              "cpu_seed": 109, "cuda_seed": 109, "visible_gpus": 1, "flash": True,
              "math": False, "efficient": False, "cudnn": False})
        projection_receipt = {"recipe": "fixed_row_fp32_action_reduction_v1", "original_parameter_retained": True,
              "independent_storage": True, "original_dtype": "torch.bfloat16", "projection_dtype": "torch.float32",
              "new_trainable_parameters": 0, "projection_rows_per_call": 1}
        write("initial-load.json", {"stage": "initial", "fresh_base": True, "restoration": None,
              "projection": projection_receipt, "wall_seconds": 1., "monotonic_seconds": 10.})
        order = {r["record_id"]: sorted(r["commands"]) for r in plan["records"]}
        final_scores = {rid: [-2. - i for i, _ in enumerate(commands)] for rid, commands in order.items()}
        for stage, clock in (("sft", 40.), ("final", 50.)):
            signatures = {"base": base, "projection": projection,
                          "adapter": checkpoints[stage]["adapter_signature"], "reference": reference}
            if stage == "final":
                write("final-before-reload.json", {"commands": order, "scores": final_scores, "optimizer_updates": 0,
                      "signatures_before": signatures, "signatures_after": signatures, "monotonic_seconds": 31.})
            restoration = {"adapter_sha256": checkpoints[stage]["files"]["adapter"],
                           "training_receipt_sha256": pin(f"training/{stage}.checkpoint.json"),
                           "rank": 8, "alpha": 16., "adapter_parameters": 1146880,
                           "projection_names": sorted({n.rsplit(".", 1)[0] for n in plan["expected_adapter_shapes"]})}
            write(stage + "-load.json", {"stage": stage, "fresh_base": True, "restoration": restoration,
                  "projection": projection_receipt, "wall_seconds": 1., "monotonic_seconds": clock})
            scores = reference_scores if stage == "sft" else final_scores
            write(stage + "-reload-scores.json", {"scores": scores, "expected_scores": scores,
                  "max_absolute_difference": 0., "monotonic_seconds": clock + 1})
            for index, rec in enumerate([*plan["records"], plan["records"][0]]):
                command = rec["supervised_command"]
                sequence = rec["prompt_ids"] + rec["commands"][command]
                row = {"stage": stage, "ordinal": index, "record_id": rec["record_id"],
                       "record_pin": plan["record_pins"][rec["record_id"]], "sequence_ids": sequence,
                       "completion": command, "stop": command == '{"stop":true}', "legal_commands": len(rec["commands"]),
                       "new_training_updates": 0, "environment_responses": 0, "tensor_versions_unchanged": True,
                       "cross_request_cache_supplied": False, "cuda_interval_ms": 1.,
                       "monotonic_seconds": clock + 2 + index, **memory}
                write(f"{stage}-generation-{index:02d}.json", row)
                write(f"{stage}-raw-generation-{index:02d}.json", {"sequence_ids": sequence,
                      "dtype": "torch.int64", "shape": [1, len(sequence)]})
            write(stage + "-reload.json", {"scores": scores, "score_max_absolute_difference": 0., "scores_exact": True,
                  "optimizer_updates": 0, "fresh_base": True, "adapter_parameters": 1146880,
                  "adapter_sha256": restoration["adapter_sha256"], "training_receipt_sha256": restoration["training_receipt_sha256"],
                  "signatures_before": signatures, "signatures_after": signatures, "tensor_versions_unchanged": True,
                  "inference_parameters_frozen": True, "monotonic_seconds": clock + 7})
        for label, clock in (("before-sft-reload", 32.), ("before-final-reload", 48.), ("after-final-inference", 59.)):
            write(label + "-memory.json", {"allocated_bytes": 0, "reserved_bytes": 0, "monotonic_seconds": clock})
        summary = {"version": "ace-bounded-artificial-training-gpu-worker-v1", "status": "worker_complete", "scope": "artificial",
            "prepared_sha256": hashlib.sha256((json.dumps(self.prepared, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
            "training_plan_sha256": self.prepared["training_plan_sha256"],
            **{k: self.prepared[k] for k in ("collection_plan_sha256", "collection_terminal_sha256", "exposure_sha256", "tokenizer_sha256")},
            "engine_terminal_sha256": pin("training/terminal.json"), "reference_sha256": pin("training/reference.pt"),
            **{s + "_adapter_sha256": checkpoints[s]["files"]["adapter"] for s in ("initial", "sft", "final")},
            "base_signature": base, "projection_signature": projection, "reference_signature": reference,
            "first_gpu_work_sha256": pin("first-gpu-work.json"), "adapter_matrices": 112, "adapter_parameters": 1146880,
            "seed": 109, "visible_gpus": 1, "records": 4, "sft_steps": 4, "dpo_steps": 2, "training_updates": 6,
            "stored_tokens": plan["stored_tokens"], "training_input_tokens": sum(s["input_tokens"] for s in steps),
            "training_action_tokens": sum(s["action_tokens"] for s in steps), "sft_rows": 4, "dpo_rows": 4,
            "inference_training_updates": 0, "generation_count": 10, "legal_generation_count": 10,
            "stop_count": 4, "action_count": 6, "prepared_legal_commands": 10, "reference_pair_count": 2, "final_score_count": 10,
            "fresh_base_load_count": 3, "prior_adapter_restored": False, "sft_reload_scores_exact": True,
            "final_reload_scores_exact": True, "same_state_repeatability": True, "sft_reload_score_max_difference": 0.,
            "final_reload_score_max_difference": 0., "inference_signatures_unchanged": True, "base_unchanged": True,
            "projection_unchanged": True, "checkpoint_recompute_calls": 168, "cuda_step_ms_total": 6.,
            "cuda_generation_ms_total": 10., "cuda_first_forward_ms": 1., "training_wall_seconds": 8.,
            "environment_responses": 0, "scientific_responses": 0, "scientific_qualified": False,
            "whole_run_accepted": False, "outer_execution_qualified": False, "retry_allowed": False,
            "elapsed_seconds": 60., "process_cpu_seconds": 1., "model_load_seconds": 3., **memory,
            "total_device_bytes": 1000, "first_gpu_work_monotonic_seconds": 20., "training_start_monotonic_seconds": 21.,
            "training_complete_monotonic_seconds": 30., "inference_complete_monotonic_seconds": 58.}
        write("summary.json", summary)
        self._fixture_manifest(root)

    def test_complete_bundle_rejects_repinned_surplus_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            checked = self.worker.validate_outputs(root, self.prepared)
            self.assertEqual(checked["status"], "validated_worker_evidence")
            self.assertEqual((checked["training_updates"], checked["generation_count"], checked["evidence_files"]), (6, 10, 95))
            self.assertEqual(checked["summary_sha256"], self._fixture_pin(root, "summary.json"))
            self.assertFalse(checked["tensor_files_deserialized"])
            self.assertFalse(checked["runtime_and_cleanup_qualified"])
            for name in ("training/step-00006.optimizer-attempt.json", "final-generation-05.json",
                         "training/failed-partial.adapter.pt", "training/failure.json", "training/step-00006.checkpoint.json",
                         "sft-raw-generation-05.json", "training/initial.optimizer.pt"):
                with self.subTest(surplus=name):
                    self._fixture_write(root, name, b"surplus fixture evidence\n")
                    self._fixture_manifest(root)
                    self.assertEqual(json.loads((root / "evidence-files.json").read_bytes())[name], self._fixture_pin(root, name))
                    with self.assertRaisesRegex(ValueError, "evidence file inventory differs from pinned recipe"):
                        self.worker.validate_outputs(root, self.prepared)
                    (root / name).unlink()
            self._fixture_manifest(root)
            # Membership stays exact, but changing bytes without re-pinning is
            # still independently rejected before any JSON/tensor inspection.
            with (root / "training/final.adapter.pt").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "worker evidence file changed: training/final.adapter.pt"):
                self.worker.validate_outputs(root, self.prepared)

    def test_complete_bundle_binds_finite_phase_clocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            self.worker.validate_outputs(root, self.prepared)
            summary_bytes = (root / "summary.json").read_bytes()
            for field, value, message in (
                ("first_gpu_work_monotonic_seconds", 19., "first GPU work clock differs"),
                ("first_gpu_work_monotonic_seconds", "-1e309", "nonfinite summary phase clock"),
                ("training_start_monotonic_seconds", True, "nonfinite summary phase clock"),
                ("inference_complete_monotonic_seconds", "1e309", "nonfinite summary phase clock"),
                ("training_complete_monotonic_seconds", 31.5, "summary phase bounds differ"),
                ("inference_complete_monotonic_seconds", 56.5, "summary phase bounds differ"),
                ("inference_complete_monotonic_seconds", 60., "summary phase bounds differ"),
            ):
                with self.subTest(summary_clock=field, value=value):
                    summary = json.loads(summary_bytes)
                    summary[field] = value
                    raw = json.dumps(summary).replace('"-1e309"', '-1e309').replace('"1e309"', '1e309')
                    (root / "summary.json").write_text(raw + "\n")
                    with self.assertRaisesRegex(ValueError, message):
                        self.worker.validate_outputs(root, self.prepared)
            (root / "summary.json").write_bytes(summary_bytes)
            for name, value in (("final-reload.json", 59.), ("sft-generation-02.json", 39.),
                                ("final-reload-scores.json", None)):
                with self.subTest(saved_clock=name):
                    path = root / name
                    original = path.read_bytes()
                    row = json.loads(original)
                    row["monotonic_seconds"] = value
                    self._fixture_write(root, name, row)
                    self._fixture_manifest(root)
                    with self.assertRaisesRegex(ValueError, "summary phase bounds differ"):
                        self.worker.validate_outputs(root, self.prepared)
                    path.write_bytes(original)
            self._fixture_manifest(root)
            self.assertEqual(self.worker.validate_outputs(root, self.prepared)["status"], "validated_worker_evidence")


if __name__ == "__main__":
    unittest.main()
