"""NEW stdlib-only screen worker controls; no engine/model/tokenizer execution.

Every fabricated output is retained under ACE_SCREEN_WORKER_TEST_OUTPUT. Inputs
have hand-written token IDs, not actual CPU-tokenizer qualification evidence.
"""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import types
import unittest

SOURCE = Path(__file__).with_name("llm_intervention_screen_training_gpu_worker.py")

def canonical(value):return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def pins(prepared):return dict(prepared_sha256=hashlib.sha256((canonical(prepared)+"\n").encode()).hexdigest(), plan_sha256=digest(prepared["training_plan"]))

def fabricated_prepared():
    """Independent declarations; no numeric/project import and no data draws."""
    recipe = {'activation_checkpointing': 'explicit_nonreentrant_decoder_v1', 'adam_betas': [0.9, 0.999], 'adam_eps': 1e-08, 'adapter_dtype': 'float32', 'alpha': 16.0, 'amsgrad': False, 'attention': 'sdpa', 'base_dtype': 'bfloat16', 'beta': 0.1, 'capturable': False, 'clip_gradient_norm': 1.0, 'differentiable': False, 'float32_matmul': 'highest', 'foreach': None, 'fused': None, 'learning_rate': 0.0001, 'max_new_tokens': 32, 'max_tokens': 8192, 'maximize': False, 'optimizer': 'AdamW', 'optimizer_state': 'continuous_across_sft_dpo', 'output_projection': 'fixed_row_fp32_action_reduction_v1', 'rank': 8, 'scoring_path': 'canonical_selected_unpadded', 'sdpa_backend': 'flash_attention_only', 'seed': 109, 'strict_determinism': True, 'weight_decay': 0.0}
    shapes = {f"model.layers.{i}.self_attn.{part}.{side}": shape
              for i in range(28) for part,width in (("q_proj",2048),("v_proj",1024))
              for side,shape in (("A",[8,1024]),("B",[width,8]))}
    names = ["observe_all","set_x_neg","set_x_zero","set_x_pos","set_y_neg","set_y_zero","set_y_pos","set_xy"]
    action = lambda name: canonical({"action_id":name})
    stop = canonical({"stop":True}); obs, neg, joint = map(action,["observe_all","set_x_neg","set_xy"])
    commands = {action(n): [i+1] * (8 if n == "set_x_neg" else 7) + [99] for i,n in enumerate(names)}
    commands[stop] = [42]*5 + [99]
    records=[]
    for i in range(4):
        table = {stop:commands[stop]} if i == 2 else copy.deepcopy(commands)
        raw = {"scope":"artificial", "envelope":{"state":{"terminal": i == 2,
            "remaining_units":0 if i == 2 else 2,"menu":[{"action_id":n,"cost_units":1} for n in names]}}}
        rawtext=canonical(raw)+"\n"
        records.append({"record_id":f"fake/r{i}","scope":"artificial","version":"ace-bounded-training-engine-v1",
            "raw_record_utf8":rawtext,"record_sha256":hashlib.sha256(rawtext.encode()).hexdigest(),
            "prompt_ids":[123]*(4744 if i==2 else 4638),"commands":table,"pad_token_id":0,"eos_token_id":99,
            "supervised_command":[obs,neg,stop,joint][i],"strict_pairs":[[[obs,neg]],[[neg,obs]],[],[[joint,obs]]][i],
            "tied_pairs":[],"tokenizer_sha256":digest("fake tokenizer"),
            "token_metadata":{"prompt_tokens":4744 if i==2 else 4638,"max_tokens":8192,"max_new_tokens":32}})
    schedule=[{"stage":"sft","record_id":r["record_id"],"pair_index":None} for r in records]
    schedule += [{"stage":"dpo","record_id":r["record_id"],"pair_index":0} for r in records[:2]]
    plan={"scope":"artificial","execution_kind":"qwen_screen_singletons","numeric_recipe":recipe,
          "sft_steps":4,"dpo_steps":2,"expected_adapter_shapes":shapes,"max_trainable_parameters":1146880,
          "declared_adapter_parameters":1146880,"checkpoint_steps":[3,5],"one_pass_sft":True,"one_pass_dpo":False,
          "records":records,"record_pins":{r["record_id"]:digest(r) for r in records},"schedule":schedule,
          "exposures":{r["record_id"]:{"sft":1,"dpo_pairs":[1] if i<2 else ([] if i==2 else [0]),"tied_pairs_retained":0} for i,r in enumerate(records)},
          "stored_tokens":sum(len(r["prompt_ids"])+sum(map(len,r["commands"].values())) for r in records),
          "sft_rows":4,"dpo_rows":4,"outer_provenance_receipt_sha256":digest("fake provenance")}
    return {"schema":"ace-screen-training-shape-preparation-v1","scope":"artificial",
            "training_plan":plan,"training_plan_sha256":digest(plan),"tokenizer_sha256":digest("fake tokenizer"),
            **{k:digest(k) for k in ["capacity_packet_sha256","capacity_acceptance_sha256","tokenization_sha256"]},
            "source_sha256":{"fabricated-source.py":digest("fake source")},"source_commit":"a"*40,
            "labels":"fabricated only","environment_responses":0,"tokenizer_calls":0,
            "scientific_qualified":False,"training_execution_qualified":False}

class ScreenWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ["ACE_SCREEN_WORKER_TEST_OUTPUT"])
        cls.output.mkdir(exist_ok=False)
        cls.prepared=fabricated_prepared();cls.pins=pins(cls.prepared)
        cls.source=SOURCE.read_bytes(); cls.imports=[]
        real_import=builtins.__import__
        def stdlib_only(name,*args,**kwargs):
            cls.imports.append(name)
            if name not in {"__future__","gc","hashlib","json","math","os","pathlib","time","traceback"}:
                raise AssertionError("heavy/unexpected import forbidden: "+name)
            return real_import(name,*args,**kwargs)
        cls.worker=types.ModuleType("captured_screen_worker")
        cls.worker.__dict__["__builtins__"]={**vars(builtins),"__import__":stdlib_only}
        exec(compile(cls.source,str(SOURCE),"exec"),cls.worker.__dict__)
        (cls.output/"fabricated-prepared.json").write_text(canonical(cls.prepared)+"\n")
        (cls.output/"independent-pins.json").write_text(canonical(cls.pins)+"\n")

    def root(self,label):
        p=self.output/self._testMethodName/label;p.mkdir(parents=True,exist_ok=False);return p

    def validate(self,root):return self.worker.validate_outputs(root,self.prepared,**self.pins)

    def mutate(self,root,name,change):
        data=json.loads((root/name).read_bytes());change(data)
        self._fixture_write(root,name,data);self._fixture_manifest(root)

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
        inventory and summary helpers, using a new fabricated 4638/4744-token plan. This fixture writer is
        adapted from the prior verifier fixture; no old test method is invoked.
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
            write(stem + ".gradients.json", {"checkpoint_recompute_delta": (28 if scheduled["stage"] == "sft" else 56),
                  "tensors": [{"name": n, "shape": shape, "present": True, "finite": True}
                              for n, shape in plan["expected_adapter_shapes"].items()]})
            write(stem + ".update-intent.json", {"gradient_norm_before_clip": 1.2,
                  "gradient_norm_after_clip": 1., "clip_limit": 1.})
            rec = next(r for r in plan["records"] if r["record_id"] == scheduled["record_id"])
            commands = ([rec["supervised_command"]] if scheduled["stage"] == "sft" else
                        rec["strict_pairs"][scheduled["pair_index"]])
            tokens = sum(len(rec["commands"][c]) for c in commands)
            event = {**returned, "loss": 0.5, "gradient_norm_before_clip": 1.2, "gradient_norm_after_clip": 1.,
                     "checkpoint_recompute_delta": (28 if scheduled["stage"] == "sft" else 56), "rows": len(commands), "action_tokens": tokens,
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
        write("training/reference-scores.json", {"signature": reference, "unique_pair_score_calls": 2, "decoder_forwards": 4, "scoring_layout": "ordered_unpadded_singletons", "scores": reference_scores})
        write("training/terminal.json", {"status": "complete", "plan_sha256": self.prepared["training_plan_sha256"],
              "execution_kind": "qwen_screen_singletons", "completed_steps": 6, "sft_steps": 4, "dpo_steps": 2,
              "exposures": plan["exposures"], "one_pass_sft": True, "one_pass_dpo": False,
              "reference_scores_unchanged": True, "reference_pair_count": 2, "reference_signature": reference,
              "provenance_qualified": False, "retry_allowed": False, "environment_responses": 0,
              "outer_provenance_receipt_sha256": plan["outer_provenance_receipt_sha256"],
              "base_before": base, "base_after": base, "projection_before": projection, "projection_after": projection,
              "checkpoint_usage": {"enabled": True, "use_reentrant": False, "preserve_rng_state": True,
                                   "native_checkpointing": False, "recompute_calls": 224, "forward_calls": 224},
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
                      "scoring_layout": "ordered_unpadded_singletons", "decoder_forwards": 28,
                      "signatures_before": signatures, "signatures_after": signatures, "monotonic_seconds": 31.})
            restoration = {"adapter_sha256": checkpoints[stage]["files"]["adapter"],
                           "training_receipt_sha256": pin(f"training/{stage}.checkpoint.json"),
                           "rank": 8, "alpha": 16., "adapter_parameters": 1146880,
                           "projection_names": sorted({n.rsplit(".", 1)[0] for n in plan["expected_adapter_shapes"]})}
            write(stage + "-load.json", {"stage": stage, "fresh_base": True, "restoration": restoration,
                  "projection": projection_receipt, "wall_seconds": 1., "monotonic_seconds": clock})
            scores = reference_scores if stage == "sft" else final_scores
            write(stage + "-reload-scores.json", {"scores": scores, "expected_scores": scores,
                  "max_absolute_difference": 0., "monotonic_seconds": clock + 1,
                  "scoring_layout": "ordered_unpadded_singletons", "decoder_forwards": 4 if stage == "sft" else 28})
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
            write(label + "-memory.json", {"allocated_bytes": 0, "reserved_bytes": 0, "monotonic_seconds": clock,
                  "cleanup_operation": "torch._C._cuda_clearCublasWorkspaces", "workspace_clear_calls": 1,
                  "workspace_clear_succeeded": True, "before_allocated_bytes": 100, "before_reserved_bytes": 200,
                  "after_workspace_clear_allocated_bytes": 0, "after_workspace_clear_reserved_bytes": 200})
        summary = {"version": "ace-screen-singleton-artificial-training-gpu-worker-v1", "execution_kind": "qwen_screen_singletons", "status": "worker_complete", "scope": "artificial",
            "prepared_sha256": hashlib.sha256((json.dumps(self.prepared, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
            "training_plan_sha256": self.prepared["training_plan_sha256"],
            **{k: self.prepared[k] for k in ("capacity_packet_sha256", "capacity_acceptance_sha256", "tokenization_sha256", "tokenizer_sha256", "labels")},
            "engine_terminal_sha256": pin("training/terminal.json"), "reference_sha256": pin("training/reference.pt"),
            **{s + "_adapter_sha256": checkpoints[s]["files"]["adapter"] for s in ("initial", "sft", "final")},
            "base_signature": base, "projection_signature": projection, "reference_signature": reference,
            "first_gpu_work_sha256": pin("first-gpu-work.json"), "adapter_matrices": 112, "adapter_parameters": 1146880,
            "seed": 109, "visible_gpus": 1, "records": 4, "sft_steps": 4, "dpo_steps": 2, "training_updates": 6,
            "stored_tokens": plan["stored_tokens"], "training_input_tokens": sum(s["input_tokens"] for s in steps),
            "training_action_tokens": sum(s["action_tokens"] for s in steps), "sft_rows": 4, "dpo_rows": 4,
            "inference_training_updates": 0, "generation_count": 10, "legal_generation_count": 10,
            "stop_count": 2, "action_count": 8, "prepared_legal_commands": 28, "reference_pair_count": 2, "final_score_count": 28,
            "reference_decoder_forwards": 4, "scoring_layout": "ordered_unpadded_singletons",
            "fresh_base_load_count": 3, "prior_adapter_restored": False, "sft_reload_scores_exact": True,
            "final_reload_scores_exact": True, "same_state_repeatability": True, "sft_reload_score_max_difference": 0.,
            "final_reload_score_max_difference": 0., "inference_signatures_unchanged": True, "base_unchanged": True,
            "projection_unchanged": True, "checkpoint_recompute_calls": 224, "cuda_step_ms_total": 6.,
            "cuda_generation_ms_total": 10., "cuda_first_forward_ms": 1., "training_wall_seconds": 8.,
            "environment_responses": 0, "scientific_responses": 0, "scientific_qualified": False,
            "whole_run_accepted": False, "outer_execution_qualified": False, "retry_allowed": False,
            "elapsed_seconds": 60., "process_cpu_seconds": 1., "model_load_seconds": 3., **memory,
            "total_device_bytes": 1000, "first_gpu_work_monotonic_seconds": 20., "training_start_monotonic_seconds": 21.,
            "training_complete_monotonic_seconds": 30., "inference_complete_monotonic_seconds": 58.}
        write("summary.json", summary)
        self._fixture_manifest(root)

    def test_independent_pins_are_required_for_both_public_apis(self):
        plan=self.worker._plan(self.prepared,**self.pins)
        self.assertEqual(len(plan['records']),4)
        plan['records'][0]['prompt_ids'][0]=0
        self.assertNotEqual(plan,self.prepared['training_plan'])
        with self.assertRaises(TypeError):self.worker.validate_outputs('/unused',self.prepared)
        with self.assertRaises(TypeError):self.worker.run_training(self.prepared,'/unused','/unused',None,None)
        for key in self.pins:
            for value in [None,'0'*64,'A'*64]:
                bad=dict(self.pins);bad[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    self.worker._plan(self.prepared,**bad)
        altered=copy.deepcopy(self.prepared);altered['training_plan']['scope']='policy_train'
        altered['training_plan_sha256']=digest(altered['training_plan'])
        with self.assertRaisesRegex(ValueError,'artifact pin'):
            self.worker._plan(altered,**self.pins)

    def test_profile_rejects_repinned_scope_recipe_shape_exposure_and_prompt_changes(self):
        changes=[('scope',lambda p:p.update(scope='policy_train')),
            ('old_kind',lambda p:p.update(execution_kind='qwen_gpu09')),
            ('old_scoring',lambda p:p['numeric_recipe'].update(scoring_path='canonical_selected')),
            ('extra_record',lambda p:p['records'].append(copy.deepcopy(p['records'][0]))),
            ('short_prompt',lambda p:p['records'][0]['prompt_ids'].pop()),
            ('shape',lambda p:p['expected_adapter_shapes'][next(iter(p['expected_adapter_shapes']))].__setitem__(0,7)),
            ('checkpoint',lambda p:p.update(checkpoint_steps=[2,5])),
            ('one_pass_dpo',lambda p:p.update(one_pass_dpo=True)),
            ('schedule',lambda p:p['schedule'][-1].update(record_id=p['records'][0]['record_id'])),
            ('hidden_exposure',lambda p:p['exposures']['fake/r3']['dpo_pairs'].__setitem__(0,1)),
            ('wrong_reverse',lambda p:p['records'][1]['strict_pairs'][0].reverse()),
            ('wrong_terminal',lambda p:p['records'][2].update(raw_record_utf8=p['records'][0]['raw_record_utf8'])),
            ('equal_scheduled_lengths',lambda p:p['records'][0]['commands']['{"action_id":"set_x_neg"}'].pop(0)),
            ('wrong_stop_length',lambda p:p['records'][2]['commands']['{"stop":true}'].pop(0))]
        for label,change in changes:
            altered=copy.deepcopy(self.prepared);p=altered['training_plan'];change(p)
            p['record_pins']={r['record_id']:digest(r) for r in p['records']}
            altered['training_plan_sha256']=digest(p)
            root=self.root(label)
            self._fixture_write(root,'rejected-prepared.json',altered)
            with self.subTest(label=label),self.assertRaises(ValueError):
                self.worker._plan(altered,**pins(altered))

    def test_both_mixed_orders_and_equal_control_stay_unpadded_singletons(self):
        plan=self.worker._plan(self.prepared,**self.pins)
        for i,lengths in [(0,[8,9]),(1,[9,8]),(3,[8,8])]:
            rec=plan['records'][i];commands=rec['strict_pairs'][0]
            self.assertEqual([len(rec['commands'][c]) for c in commands],lengths)
            for mode in ['singletons','reference_pair']:
                self.assertEqual(self.worker.scoring_layout(rec,commands,mode=mode),tuple((c,) for c in commands))
        with self.assertRaises(ValueError):
            self.worker.scoring_layout(plan['records'][0],['missing'],mode='singletons')
        with self.assertRaises(ValueError):
            self.worker.scoring_layout(plan['records'][0],list(plan['records'][0]['commands']),mode='reference_pair')

    def test_complete_new_bundle_has_28_scores_10_generations_no_tensor_load(self):
        root=self.root('complete');self._complete_fixture(root);checked=self.validate(root)
        self.assertEqual((checked['records'],checked['training_updates'],checked['generation_count']),(4,6,10))
        self.assertEqual((checked['final_score_count'],checked['prepared_legal_commands']),(28,28))
        self.assertEqual((checked['reference_decoder_forwards'],checked['checkpoint_recompute_calls']),(4,224))
        self.assertEqual(checked['scoring_layout'],'ordered_unpadded_singletons')
        self.assertFalse(checked['tensor_files_deserialized']);self.assertFalse(checked['runtime_and_cleanup_qualified'])
        self.assertEqual(checked['environment_responses'],0)
        self.assertEqual(checked['evidence_files'],95)

    def test_reference_receipt_rejects_legacy_batched_or_reversed_claims(self):
        for label,change in [
            ('legacy',lambda d:(d.pop('unique_pair_score_calls'),d.update(unique_pair_forwards=2))),
            ('decoder_count',lambda d:d.update(decoder_forwards=2)),
            ('layout',lambda d:d.update(scoring_layout='batched_pair')),
            ('pair_count',lambda d:d.update(unique_pair_score_calls=1)),
            ('reverse',lambda d:d['scores'].__setitem__(next(iter(d['scores'])),[-7.5,-2.25]))]:
            root=self.root(label);self._complete_fixture(root)
            self.mutate(root,'training/reference-scores.json',change)
            with self.subTest(label=label),self.assertRaises(ValueError):self.validate(root)

    def test_stage_specific_recomputation_is_not_28_for_mixed_dpo(self):
        for index,delta in [(0,27),(4,28),(5,55)]:
            root=self.root(str(index));self._complete_fixture(root)
            for kind in ['gradients','complete']:
                self.mutate(root,f'training/step-{index:05d}.{kind}.json',lambda d:d.update(checkpoint_recompute_delta=delta))
            with self.assertRaisesRegex(ValueError,'recomputation'):self.validate(root)
        root=self.root('terminal');self._complete_fixture(root)
        self.mutate(root,'training/terminal.json',lambda d:d['checkpoint_usage'].update(recompute_calls=168))
        with self.assertRaisesRegex(ValueError,'checkpoint evidence'):self.validate(root)

    def test_before_and_reload_evidence_must_use_full_singleton_counts(self):
        for name,count in [('final-before-reload.json',27),('sft-reload-scores.json',2),('final-reload-scores.json',27)]:
            root=self.root(name);self._complete_fixture(root)
            self.mutate(root,name,lambda d:d.update(decoder_forwards=count))
            with self.assertRaises(ValueError):self.validate(root)

    def test_signature_checkpoint_bytes_cuda_and_cleanup_fail_closed(self):
        cases=[('sft-reload.json',lambda d:d.update(tensor_versions_unchanged=False)),
               ('first-gpu-work.json',lambda d:d.update(cuda_interval_ms=0)),
               ('training/step-00004.complete.json',lambda d:d.update(peak_allocated_bytes=0)),
               ('arithmetic.json',lambda d:d.update(math=True)),
               ('before-final-reload-memory.json',lambda d:d.update(allocated_bytes=1))]
        for i,(name,change) in enumerate(cases):
            root=self.root(str(i));self._complete_fixture(root);self.mutate(root,name,change)
            with self.subTest(name=name),self.assertRaises(ValueError):self.validate(root)
        root=self.root('opaque_corruption');self._complete_fixture(root)
        with (root/'training/final.adapter.pt').open('ab') as f:f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'changed'):self.validate(root)

    def test_no_surplus_failed_or_extra_attempts_even_if_repinned(self):
        for i,name in enumerate(['failure.json','training/failure.json','training/step-00006.complete.json','final-generation-05.json']):
            root=self.root(str(i));self._complete_fixture(root)
            self._fixture_write(root,name,{'status':'failed'});self._fixture_manifest(root)
            with self.assertRaises(ValueError):self.validate(root)

    def test_legal_generation_rejects_prompt_or_token_substitution(self):
        record=self.prepared['training_plan']['records'][0];command=record['supervised_command']
        sequence=record['prompt_ids']+record['commands'][command]
        self.assertFalse(self.worker._legal_sequence(record,sequence,command))
        for wrong in [[1]+sequence[1:],sequence[:-1],sequence+[99]]:
            with self.assertRaises(ValueError):self.worker._legal_sequence(record,wrong,command)
        terminal=self.prepared['training_plan']['records'][2]
        self.assertTrue(self.worker._legal_sequence(terminal,terminal['prompt_ids']+terminal['commands']['{"stop":true}'],'{"stop":true}'))

    def test_bad_run_pin_preserves_failure_before_any_heavy_import_or_observation(self):
        parent=self.root('parent');root=parent/'artifacts'
        class Journal:
            output=parent
            def save(j,name,value):
                with (parent/name).open('x') as f:f.write(canonical(value)+'\n')
            def event(j,*args,**kwargs):raise AssertionError('no model phase')
        class Observer:
            def observe(o,*args):raise AssertionError('no GPU observation')
        with self.assertRaisesRegex(ValueError,'independent prepared'):
            self.worker.run_training(self.prepared,'/never-load',root,Observer(),Journal(),prepared_sha256=None,plan_sha256=self.pins['plan_sha256'])
        failure=json.loads((root/'failure.json').read_text())
        self.assertEqual(failure['stage'],'prepared_validation');self.assertFalse(failure['retry_allowed'])
        self.assertEqual(failure['environment_responses'],0)
        with self.assertRaisesRegex(ValueError,'no resume'):
            self.worker.run_training(self.prepared,'/never-load',root,Observer(),Journal(),**self.pins)
        self.assertFalse(any(n.startswith(('torch','numpy','transformers','llm_')) for n in self.imports))

    def test_wiring_has_new_engine_kind_recipe_unpadded_path_and_explicit_pins(self):
        tree=ast.parse(self.source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_training')
        self.assertEqual([a.arg for a in run.args.kwonlyargs],['prepared_sha256','plan_sha256'])
        self.assertEqual(run.args.kw_defaults,[None,None])
        validate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_outputs')
        self.assertEqual([a.arg for a in validate.args.kwonlyargs],['prepared_sha256','plan_sha256'])
        text=ast.unparse(run)
        self.assertIn('engine.screen_numeric_recipe()',text)
        self.assertNotIn('sequence_log_probs_canonical',text)
        self.assertIn('sequence_log_probs_unpadded(current, batch, selected=True)',text)
        self.assertIn('plan_sha256=plan_sha256',text)
        self.assertIn('SDPBackend.FLASH_ATTENTION',text)
        self.assertIn("observer.observe('after-requests')",text)
        score=next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name=='scores')
        self.assertIn('for group in layouts[key]',ast.unparse(score))
        self.assertIn('with torch.no_grad()',ast.unparse(score))
        # Diagnostic no_grad helper is separate from the engine's training fit.
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in ast.walk(run)),1)

    def test_shape_preparation_lineage_survives_full_summary_and_rejects_collection_claims(self):
        # Exact top-level schema from the parent's shape builder; values and
        # tokens remain fabricated. No collected-supervision fields are added.
        expected = {'schema','scope','training_plan','training_plan_sha256','tokenizer_sha256',
                    'capacity_acceptance_sha256','capacity_packet_sha256','tokenization_sha256',
                    'source_sha256','source_commit','scientific_qualified','training_execution_qualified',
                    'labels','environment_responses','tokenizer_calls'}
        self.assertEqual(set(self.prepared), expected)
        root=self.root('shape_complete');self._complete_fixture(root)
        checked=self.validate(root)
        for key in ['capacity_packet_sha256','capacity_acceptance_sha256','tokenization_sha256','labels']:
            self.assertEqual(checked[key],self.prepared[key])
        legacy=['collection_plan_sha256','collection_terminal_sha256','exposure_sha256']
        self.assertFalse(set(legacy).intersection(checked))
        # These repinned controls exercise the pre-import schema gate, rather
        # than merely failing the outer prepared digest.
        for key in ['capacity_packet_sha256','capacity_acceptance_sha256','tokenization_sha256','labels']:
            changed=copy.deepcopy(self.prepared);changed.pop(key)
            folder=self.root('missing-'+key);self._fixture_write(folder,'prepared.json',changed)
            with self.subTest(missing=key),self.assertRaisesRegex(ValueError,'lineage'):
                self.worker._plan(changed,**pins(changed))
        for key in legacy:
            changed=copy.deepcopy(self.prepared);changed[key]='a'*64
            folder=self.root('legacy-'+key);self._fixture_write(folder,'prepared.json',changed)
            with self.subTest(legacy=key),self.assertRaisesRegex(ValueError,'collected-supervision'):
                self.worker._plan(changed,**pins(changed))
        legacy_root=root.parent/'legacy-summary'
        shutil.copytree(root,legacy_root)
        self.mutate(legacy_root,'summary.json',lambda d:d.update(collection_plan_sha256='a'*64))
        with self.assertRaisesRegex(ValueError,'collected-supervision'):self.validate(legacy_root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
