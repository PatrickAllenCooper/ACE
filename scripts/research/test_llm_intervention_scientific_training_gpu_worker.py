"""New fabricated scientific-shaped receipts only; no real execution or data.

The fixed public roster is a schema control, not sampled scientific IDs. All
labels, short token IDs, clocks, GPU metrics and opaque .pt files are fabricated.
No tokenizer/engine/model imports or tensor deserialization. Direct execution
requires an absent external ACE_SCIENTIFIC_GPU_TEST_OUTPUT; retain every attempt.
"""
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import llm_intervention_scientific_training_gpu_worker as worker

canonical = worker._canonical
digest = worker._digest

def pins(prepared):
    return {'prepared_sha256':hashlib.sha256((canonical(prepared)+'\n').encode()).hexdigest(),
            'plan_sha256':prepared['training_plan_sha256']}

def repin(prepared):
    plan=prepared['training_plan']
    plan['record_pins']={r['record_id']:digest(r) for r in plan['records']}
    prepared['training_plan_sha256']=digest(plan)
    prepared['exposure_sha256']=digest(prepared['exposure'])
    return pins(prepared)

def fabricated(d=128):
    records=[];details=[];schedule=[];dpo=[];exposures={};raw_pins={};receipt_pins={}
    menu=[{'action_id':n,'cost_units':1} for n in ('observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy')]
    actions=[canonical({'action_id':m['action_id']}) for m in menu];stop=canonical({'stop':True})
    pairs=[list(x) for x in itertools.combinations(actions,2)]
    n=0
    for i,rid in enumerate(worker.ROSTER):
        terminal=i%5==4;selected=not terminal and n<d
        if not terminal:n+=1
        strict=copy.deepcopy(pairs) if selected else []
        tied=copy.deepcopy(pairs) if not terminal and not selected else []
        original={'scope':'policy_train','world_id':worker.WORLD_IDS[i//5],'fixture_only':True,'ordinal':i,
            'envelope':{'state':{'terminal':terminal,'menu':menu,'remaining_units':0 if terminal else 24-i%5*6}},
            'preferences':[{'left':a,'right':b,'tie':not selected} for a,b in pairs] if not terminal else []}
        raw=canonical(original)+'\n';rh=hashlib.sha256(raw.encode()).hexdigest();receipt=digest(['fabricated receipt',i])
        commands={stop:[900,151645]} if terminal else {c:[20+j,30+j,151645] for j,c in enumerate([*actions,stop])}
        supervised=stop if terminal else actions[0]
        rec={'record_id':rid,'scope':'policy_train','raw_record_utf8':raw,'record_sha256':rh,
            'supervision_receipt_sha256':receipt,'tokenizer_sha256':'a'*64,'prompt_ids':[100,101,i+200],
            'commands':commands,'token_metadata':{'prompt_tokens':3,'max_tokens':8192,'max_new_tokens':32},
            'pad_token_id':151643,'eos_token_id':151645,'supervised_command':supervised,
            'strict_pairs':strict,'tied_pairs':tied,'provenance_qualified':False}
        weights=[0]*len(strict)
        if strict:
            # Same fixed pair ordering/key rule, fabricated labels only.
            index=min(range(len(strict)),key=lambda k:(digest(['ace-matched-training-exposure-proposal-v1/unordered-command-pair',rid,sorted(strict[k])]),tuple(sorted(strict[k]))))
            weights[index]=1;dpo.append({'stage':'dpo','record_id':rid,'pair_index':index})
        records.append(rec);raw_pins[rid]=rh;receipt_pins[rid]=receipt
        details.append({'record_id':rid,'world_id':original['world_id'],'record_sha256':rh,
            'supervision_receipt_sha256':receipt,'supervised_command':supervised,'strict_pair_exposures':weights,
            'dpo_exposure_count':int(selected),'sft_exposure_count':1})
        exposures[rid]={'sft':1,'dpo_pairs':weights,'tied_pairs_retained':len(tied)}
        schedule.append({'stage':'sft','record_id':rid,'pair_index':None})
    schedule+=dpo
    counts={'records':160,'retained_pairs':3584,'strict_pairs':28*d,'selected_pairs':d,'unselected_pairs':27*d,
            'tied_pairs':28*(128-d),'records_without_strict_pairs':160-d,'sft_steps':160,'dpo_steps':d,
            'sft_action_rows':160,'dpo_action_rows':2*d}
    exposure={'scope':'policy_train','has_both_training_stages':d>0,'record_roster':list(worker.ROSTER),
        'roster_sha256':digest(worker.ROSTER),'record_pins':raw_pins,'supervision_receipt_pins':receipt_pins,
        'records':details,'schedule':schedule,'counts':counts}
    shapes={f'model.layers.{i}.self_attn.{part}.{side}':shape for i in range(28)
            for part,width in (('q_proj',2048),('v_proj',1024)) for side,shape in (('A',[8,1024]),('B',[width,8]))}
    plan={'scope':'policy_train','execution_kind':'qwen_screen_singletons','numeric_recipe':copy.deepcopy(worker.SCREEN_NUMERIC_RECIPE),
        'expected_adapter_shapes':shapes,'declared_adapter_parameters':1146880,'max_trainable_parameters':1146880,
        'checkpoint_steps':[159,159+d],'one_pass_sft':True,'one_pass_dpo':False,'provenance_qualified':False,
        'outer_provenance_receipt_sha256':'b'*64,'records':records,'exposures':exposures,'schedule':schedule,
        'sft_steps':160,'dpo_steps':d,'sft_rows':160,'dpo_rows':2*d,
        'stored_tokens':sum(len(r['prompt_ids'])+sum(map(len,r['commands'].values())) for r in records)}
    prepared={'version':'ace-complete-collection-training-plan-v1','collection_terminal_sha256':'c'*64,
        'collection_inventory_sha256':'d'*64,'collection_plan_sha256':'e'*64,'tokenizer_sha256':'a'*64,
        'exposure':exposure,'training_plan':plan,'new_environment_responses':0,'training_updates':0,
        'outer_execution_qualified':False,'scientific_qualified':False}
    repin(prepared)
    return prepared


class ScientificWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ['ACE_SCIENTIFIC_GPU_TEST_OUTPUT'])
        cls.output.mkdir(exist_ok=False)
        cls.prepared=fabricated()

    def setUp(self):
        self.work=self.output/self._testMethodName;self.work.mkdir()

    def _fixture_write(self,root,name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(value if isinstance(value,bytes) else (canonical(value)+'\n').encode())

    def _fixture_pin(self,root,name):return worker._sha(root/name)

    def _fixture_manifest(self,root):
        self._fixture_write(root,'evidence-files.json',{p.relative_to(root).as_posix():worker._sha(p)
            for p in root.rglob('*') if p.is_file() and p.name not in ('summary.json','evidence-files.json')})

    def validate(self,root):return worker.validate_outputs(root,self.prepared,**pins(self.prepared))

    def _complete_fixture(self, root):
        """Fabricated JSON and opaque .pt bytes; no engine, model or tensor load.

        Construct the writer's full bundle independently of the validator's
        inventory and summary helpers, using a new fabricated 160-record plan. This fixture writer is
        adapted from the prior verifier fixture; no old test method is invoked.
        """
        plan = self.prepared["training_plan"]
        d, total = plan["dpo_steps"], 160+plan["dpo_steps"]
        forwards = sum(len(r["commands"]) for r in plan["records"])
        recomputes = 28*(160+2*d)
        write = lambda name, value: self._fixture_write(root, name, value)
        pin = lambda name: self._fixture_pin(root, name)
        signature = lambda text: hashlib.sha256(text.encode()).hexdigest()
        base, projection, reference = map(signature, ("fixture base", "fixture projection", "fixture reference"))
        checkpoints = {}
        for stem, stage, count in (("initial", "initial", 0), ("sft", "sft", 160), ("final", "final", total),
                                   ("step-00159", "sft", 160), (f"step-{total-1:05d}", "final", total)):
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
              "sft_adapter_sha256": checkpoints["sft"]["files"]["adapter"], "completed_sft_steps": 160,
              "policy_optimizer_updates_after_snapshot": 0})
        write("training/reference-scores.json", {"signature": reference, "unique_pair_score_calls": d, "decoder_forwards": 2*d, "scoring_layout": "ordered_unpadded_singletons", "scores": reference_scores})
        write("training/terminal.json", {"status": "complete", "plan_sha256": self.prepared["training_plan_sha256"],
              "execution_kind": "qwen_screen_singletons", "completed_steps": total, "sft_steps": 160, "dpo_steps": d,
              "exposures": plan["exposures"], "one_pass_sft": True, "one_pass_dpo": False,
              "reference_scores_unchanged": True, "reference_pair_count": d, "reference_signature": reference,
              "provenance_qualified": False, "retry_allowed": False, "environment_responses": 0,
              "outer_provenance_receipt_sha256": plan["outer_provenance_receipt_sha256"],
              "base_before": base, "base_after": base, "projection_before": projection, "projection_after": projection,
              "checkpoint_usage": {"enabled": True, "use_reentrant": False, "preserve_rng_state": True,
                                   "native_checkpointing": False, "recompute_calls": recomputes, "forward_calls": recomputes},
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
        for stage, clock in (("sft", 40.), ("final", 250.)):
            signatures = {"base": base, "projection": projection,
                          "adapter": checkpoints[stage]["adapter_signature"], "reference": reference}
            if stage == "final":
                write("final-before-reload.json", {"commands": order, "scores": final_scores, "optimizer_updates": 0,
                      "scoring_layout": "ordered_unpadded_singletons", "decoder_forwards": forwards,
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
                  "scoring_layout": "ordered_unpadded_singletons", "decoder_forwards": 2*d if stage == "sft" else forwards})
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
                  "inference_parameters_frozen": True, "monotonic_seconds": clock + 170})
        for label, clock in (("before-sft-reload", 32.), ("before-final-reload", 220.), ("after-final-inference", 440.)):
            write(label + "-memory.json", {"allocated_bytes": 0, "reserved_bytes": 0, "monotonic_seconds": clock,
                  "cleanup_operation": "torch._C._cuda_clearCublasWorkspaces", "workspace_clear_calls": 1,
                  "workspace_clear_succeeded": True, "before_allocated_bytes": 100, "before_reserved_bytes": 200,
                  "after_workspace_clear_allocated_bytes": 0, "after_workspace_clear_reserved_bytes": 200})
        summary = {"version": "ace-scientific-training-gpu-worker-v1", "execution_kind": "qwen_screen_singletons", "status": "worker_complete", "scope": "policy_train",
            "prepared_sha256": hashlib.sha256((json.dumps(self.prepared, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
            "training_plan_sha256": self.prepared["training_plan_sha256"],
            **{k: self.prepared[k] for k in ("collection_terminal_sha256", "collection_inventory_sha256", "collection_plan_sha256", "exposure_sha256", "tokenizer_sha256")},
            "engine_terminal_sha256": pin("training/terminal.json"), "reference_sha256": pin("training/reference.pt"),
            **{s + "_adapter_sha256": checkpoints[s]["files"]["adapter"] for s in ("initial", "sft", "final")},
            "base_signature": base, "projection_signature": projection, "reference_signature": reference,
            "first_gpu_work_sha256": pin("first-gpu-work.json"), "adapter_matrices": 112, "adapter_parameters": 1146880,
            "seed": 109, "visible_gpus": 1, "records": 160, "sft_steps": 160, "dpo_steps": d, "training_updates": total,
            "stored_tokens": plan["stored_tokens"], "training_input_tokens": sum(s["input_tokens"] for s in steps),
            "training_action_tokens": sum(s["action_tokens"] for s in steps), "sft_rows": 160, "dpo_rows": 2*d,
            "inference_training_updates": 0, "generation_count": 322, "legal_generation_count": 322,
            "stop_count": 64, "action_count": 258, "prepared_legal_commands": forwards, "reference_pair_count": d, "final_score_count": forwards,
            "reference_decoder_forwards": 2*d, "scoring_layout": "ordered_unpadded_singletons",
            "fresh_base_load_count": 3, "prior_adapter_restored": False, "sft_reload_scores_exact": True,
            "final_reload_scores_exact": True, "same_state_repeatability": True, "sft_reload_score_max_difference": 0.,
            "final_reload_score_max_difference": 0., "inference_signatures_unchanged": True, "base_unchanged": True,
            "projection_unchanged": True, "checkpoint_recompute_calls": recomputes, "cuda_step_ms_total": total,
            "cuda_generation_ms_total": 322., "cuda_first_forward_ms": 1., "training_wall_seconds": 8.,
            "environment_responses": 0, "scientific_responses": 0, "scientific_qualified": False,
            "whole_run_accepted": False, "outer_execution_qualified": False, "retry_allowed": False,
            "elapsed_seconds": 450., "process_cpu_seconds": 1., "model_load_seconds": 3., **memory,
            "total_device_bytes": 1000, "first_gpu_work_monotonic_seconds": 20., "training_start_monotonic_seconds": 21.,
            "training_complete_monotonic_seconds": 30., "inference_complete_monotonic_seconds": 430.}
        write("summary.json", summary)
        self._fixture_manifest(root)

    def test_complete_288_step_322_generation_saved_bundle(self):
        self._complete_fixture(self.work)
        checked=self.validate(self.work)
        self.assertEqual((checked['records'],checked['sft_steps'],checked['dpo_steps'],checked['training_updates']),(160,160,128,288))
        self.assertEqual((checked['generation_count'],checked['reference_pair_count'],checked['reference_decoder_forwards']),(322,128,256))
        self.assertEqual(checked['final_score_count'],1184)
        self.assertEqual(checked['checkpoint_recompute_calls'],11648)
        self.assertFalse(checked['tensor_files_deserialized']);self.assertFalse(checked['runtime_and_cleanup_qualified'])
        self.assertFalse(checked['whole_run_accepted']);self.assertFalse(checked['scientific_qualified'])
        self.assertEqual(checked['exposure_sha256'],self.prepared['exposure_sha256'])

    def test_exact_roster_partial_d_and_preserved_unselected_pairs(self):
        for d in (1,7,128):
            value=fabricated(d);self._fixture_write(self.work,f'prepared-{d}.json',value)
            original=canonical(value);plan=worker._plan(value,**pins(value))
            self.assertEqual(plan['checkpoint_steps'],[159,159+d]);self.assertEqual(len(plan['schedule']),160+d)
            self.assertEqual(sum(len(r['strict_pairs'])+len(r['tied_pairs']) for r in plan['records']),3584)
            self.assertFalse(plan['one_pass_dpo']);self.assertEqual(canonical(value),original)
            plan['records'][0]['prompt_ids'].append(10);self.assertEqual(canonical(value),original)

    def test_complete_minimum_d_keeps_all_322_generations(self):
        self.prepared=fabricated(1)
        self._complete_fixture(self.work)
        checked=self.validate(self.work)
        self.assertEqual((checked['training_updates'],checked['dpo_rows'],checked['reference_decoder_forwards']),(161,2,2))
        self.assertEqual((checked['generation_count'],checked['final_score_count']),(322,1184))
        self.assertEqual(checked['checkpoint_recompute_calls'],4536)
        self.assertTrue((self.work/'training/step-00160.adapter.pt').is_file())

    def test_saved_reader_exceeds_old_four_mib_without_tensor_load(self):
        value={'fabricated_padding':'x'*(5*1024**2)}
        self._fixture_write(self.work,'large.json',value)
        self.assertEqual(worker._read(self.work/'large.json'),value)
        self._fixture_write(self.work,'duplicate.json',b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'duplicate'):worker._read(self.work/'duplicate.json')
        self._fixture_write(self.work,'nonfinite.json',b'{"x":1e999}')
        with self.assertRaises(ValueError):worker._read(self.work/'nonfinite.json')
        (self.work/'linked.json').symlink_to(self.work/'large.json')
        with self.assertRaises(ValueError):worker._read(self.work/'linked.json')

    def test_rebound_d0_scope_roster_schedule_recipe_and_exposure_reject(self):
        changes=[lambda v:v['training_plan'].update(scope='artificial'),
            lambda v:v['training_plan'].update(execution_kind='qwen_gpu09'),
            lambda v:v['training_plan']['numeric_recipe'].update(seed=110),
            lambda v:v['training_plan']['records'].reverse(),
            lambda v:v['training_plan']['records'].pop(),
            lambda v:v['training_plan'].update(checkpoint_steps=[159,286]),
            lambda v:v['training_plan']['schedule'][-1].update(pair_index=999),
            lambda v:v['training_plan']['exposures'][worker.ROSTER[0]]['dpo_pairs'].__setitem__(0,2),
            lambda v:v['exposure']['counts'].update(retained_pairs=3583),
            lambda v:v['training_plan']['records'][0]['strict_pairs'].pop()]
        values=[fabricated(0)]
        for change in changes:
            v=copy.deepcopy(self.prepared);change(v);repin(v);values.append(v)
        for i,v in enumerate(values):
            self._fixture_write(self.work,f'rejected-{i}.json',v)
            with self.subTest(i=i),self.assertRaises((ValueError,IndexError)):
                worker._plan(v,**pins(v))

    def test_full_saved_tokens_no_truncation_eos_legal_membership(self):
        changes=[lambda r:r['prompt_ids'].extend([11]*8192),
            lambda r:r['commands'][r['supervised_command']].__setitem__(-1,0),
            lambda r:r['commands'].pop('{"stop":true}'),
            lambda r:r['token_metadata'].update(prompt_tokens=2),
            lambda r:r.update(tokenizer_sha256='f'*64)]
        for i,change in enumerate(changes):
            v=copy.deepcopy(self.prepared);change(v['training_plan']['records'][0]);repin(v)
            self._fixture_write(self.work,f'rejected-{i}.json',v)
            with self.subTest(i=i),self.assertRaises(ValueError):worker._plan(v,**pins(v))

    def test_saved_mutations_fail_with_rebound_inventory(self):
        self._complete_fixture(self.work)
        cases=[('training/reference-scores.json',lambda r:r.update(decoder_forwards=128)),
            ('training/step-00160.gradients.json',lambda r:r.update(checkpoint_recompute_delta=28)),
            ('training/step-00004.optimizer-attempt.json',lambda r:r.update(optimizer_step_started=False)),
            ('training/terminal.json',lambda r:r.update(completed_steps=287)),
            ('sft-reload-scores.json',lambda r:r['scores'].__setitem__(next(iter(r['scores'])),[0.,0.])),
            ('final-before-reload.json',lambda r:r.update(decoder_forwards=1183)),
            ('final-generation-160.json',lambda r:r['sequence_ids'].__setitem__(0,0)),
            ('before-final-reload-memory.json',lambda r:r.update(allocated_bytes=1)),
            ('final-reload.json',lambda r:r.update(tensor_versions_unchanged=False))]
        history=self.output/'mutation-originals';history.mkdir()
        manifestpath=self.work/'evidence-files.json';manifest_raw=manifestpath.read_bytes()
        for i,(name,change) in enumerate(cases):
            path=self.work/name;original=path.read_bytes();value=json.loads(original);change(value)
            changed=(canonical(value)+'\n').encode()
            self._fixture_write(history,f'{i}-before.json',original);self._fixture_write(history,f'{i}-after.json',changed)
            path.write_bytes(changed)
            manifest=json.loads(manifest_raw);manifest[name]=worker._sha(path);manifestpath.write_text(canonical(manifest)+'\n')
            try:
                with self.subTest(name=name),self.assertRaises(ValueError):self.validate(self.work)
            finally:path.write_bytes(original);manifestpath.write_bytes(manifest_raw)

    def test_no_extra_attempts_generations_or_checkpoint_corruption(self):
        self._complete_fixture(self.work)
        # Each failure is saved outside the worker root so the original remains reviewable.
        for i,name in enumerate(('failure.json','training/step-00288.complete.json','final-generation-161.json')):
            path=self.work/name;self._fixture_write(self.work,name,{'fabricated_extra':True})
            try:
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'failed|inventory'):self.validate(self.work)
            finally:path.rename(self.output/f'extra-{i}.json')
        path=self.work/'training/final.adapter.pt';original=path.read_bytes();path.write_bytes(original+b'corruption')
        with self.assertRaisesRegex(ValueError,'changed'):self.validate(self.work)

    def test_run_rejects_pin_before_heavy_import_and_preserves_failure(self):
        class Journal:
            output=self.work
            def save(j,name,value):self._fixture_write(self.work,name,value)
            def event(j,*args,**kwargs):raise AssertionError('no execution phase')
        class Observer:
            def observe(o,*args):raise AssertionError('no runtime observation')
        root=self.work/'artifacts'
        with self.assertRaisesRegex(ValueError,'independent prepared'):
            worker.run_training(self.prepared,'/never-load',root,Observer(),Journal(),prepared_sha256=None,plan_sha256=pins(self.prepared)['plan_sha256'])
        self.assertEqual(json.loads((root/'failure.json').read_bytes())['stage'],'prepared_validation')
        with self.assertRaisesRegex(ValueError,'no resume'):
            worker.run_training(self.prepared,'/never-load',root,Observer(),Journal(),**pins(self.prepared))
        with self.assertRaises(TypeError):worker.validate_outputs(root,self.prepared)
        self.assertFalse({'torch','transformers','numpy','llm_intervention_training_engine'}.intersection(sys.modules))

    def test_static_runner_delegates_once_and_replays_all_records(self):
        source=Path(worker.__file__).read_text();tree=ast.parse(source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_training')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
        fits=[n for n in calls if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='engine.fit']
        self.assertEqual(len(fits),1);self.assertEqual({k.arg:ast.unparse(k.value) for k in fits[0].keywords},
            {'model':'model','plan':'plan','plan_sha256':'plan_sha256','output':"root / 'training'"})
        text=ast.unparse(run)
        for expected in ('torch.manual_seed(109)','torch.cuda.manual_seed_all(109)',
            "targets=('q_proj', 'v_proj')",'rank=8, alpha=16.','engine.screen_numeric_recipe()',
            'sequence_log_probs_unpadded(current, batch, selected=True)',
            'enumerate([*records, records[0]])',"specs = pairs if checkpoint == 'sft' else final_specs",
            "observer.observe('after-requests')","sft_steps=160, dpo_steps=plan['dpo_steps']"):
            self.assertIn(expected,text)
        self.assertNotIn('AutoTokenizer',source);self.assertNotIn('realize_screen_worlds',source)
        # Heavy libraries remain inside run_training; the sole shared import is stdlib-only.
        self.assertFalse({'torch','transformers','numpy'}.intersection(sys.modules))
        for n in tree.body:
            if isinstance(n,ast.ImportFrom):self.assertIn(n.module,('__future__','pathlib','llm_intervention_screen_training_gpu_worker'))


if __name__=='__main__':unittest.main(verbosity=2)
