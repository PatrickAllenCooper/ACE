"""New CPU-phase coupling checks with fake training, tokenization and policies."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import llm_intervention_campaign_cpu_phase as phase
import llm_intervention_artificial_campaign as builder
from llm_intervention_contract import canonical,digest
from llm_intervention_task import unpack_envelope
from test_llm_intervention_artificial_campaign import fixture


class CPUPhaseTests(unittest.TestCase):
    def inputs(self,root):
        f=fixture(); exposure={'fixture':'exposure'}
        roster=('artificial/17/snapshot-000','artificial/17/snapshot-001',
                'artificial/4/snapshot-000','artificial/4/snapshot-001')
        records={rid:b'opaque-record-'+rid.encode() for rid in roster}
        return {'output':root/'cpu','campaign_output':root/'campaign',
            'accepted_training_inputs':{'prepared_raw':json.dumps({'exposure':exposure}).encode()},
            'collection_inventory':{'record_roster':list(roster),
                'record_pins':{rid:hashlib.sha256(raw).hexdigest() for rid,raw in records.items()},
                'supervision_receipt_pins':{rid:'a'*64 for rid in roster}},
            'records':records,'numeric_prefix':['/captured/numerical'],'ranker_prefix':['/captured/ranker'],
            'persistent_specs':{s['policy_id']:s for s in f['policy_specs_by_world'][9101][5:]},
            'services':f['services'],'numeric_provenance':dict.fromkeys(builder.NUMERIC_POLICIES,'b'*64),
            'outer_provenance_receipt_sha256':'c'*64,'tokenizer':object()}

    def fake_modules(self,events,*,token_fail=False):
        ranker=types.ModuleType('llm_intervention_ranker_training')
        ranker.numeric_recipe=lambda **kw:{'hidden':128,'artificial_control':kw['artificial']}
        def prepare(**kw):
            self.assertEqual(kw['scope'],'artificial');self.assertEqual(kw['checkpoint_steps'],(3,5))
            self.assertFalse(kw['recipe']['artificial_control'])
            return {'fixture':'ranker-plan'}
        ranker.prepare_plan=prepare
        def fit(**kw):
            events.append('fake_fit')
            kw['output'].mkdir()
            (kw['output']/'dpo.ranker.pt').write_bytes(b'fake-export-no-tensors')
            (kw['output']/'terminal.json').write_text('{}')
            return {'status':'complete','completed_steps':6,'sft_steps':4,'dpo_steps':2,
                'exports':{'dpo':{'path':'dpo.ranker.pt','sha256':hashlib.sha256(b'fake-export-no-tensors').hexdigest(),
                'training_receipt_sha256':'d'*64,'all_training_score_rows_equal':True}}}
        ranker.fit=fit
        torch=types.ModuleType('torch')
        torch.set_num_threads=lambda n:self.assertEqual(n,1)
        torch.use_deterministic_algorithms=lambda enabled,warn_only:self.assertTrue(enabled and not warn_only)
        language=types.ModuleType('llm_intervention_language_policy')
        def tokens(envelope,tokenizer,**kw):
            events.append('fake_tokens')
            self.assertEqual(kw,{'max_tokens':8192,'max_new_tokens':32})
            unpack_envelope(envelope)
            if token_fail: raise ValueError('intentional token fixture failure')
            return types.SimpleNamespace(prompt_ids=(1,),completion_ids=((2,3),)),{'prompt_tokens':1}
        language.prepare_task_tokens=tokens
        return {'llm_intervention_ranker_training':ranker,'torch':torch,'llm_intervention_language_policy':language}

    def exercise(self,root,*,token_fail=False,failed_policy=None):
        events=[];args=self.inputs(root);args['event']=lambda name,**kw:events.append(name)
        calls=[]
        def run(**kw):
            self.assertEqual(kw['phase'],'cpu')
            self.assertTrue((root/'cpu/campaign-bundle.json').exists())
            self.assertTrue((root/'cpu/tokenization.json').exists())
            self.assertEqual(events.count('fake_fit'),1)
            self.assertEqual(events.count('fake_tokens'),4)
            calls.append(kw['policy_id']);events.append('fake_policy')
            return {'policy_id':kw['policy_id'],'sha256':digest(kw['policy_id'])}
        def load(c,cp,path,pid,pin):
            self.assertEqual(pin,digest(pid))
            return {'errors':[], 'arms':[{'status':'failed' if pid==failed_policy else 'completed'}]*2},None
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=True), \
             patch.dict(sys.modules,self.fake_modules(events,token_fail=token_fail)), \
             patch('llm_intervention_campaign_training_lineage.admit_training',return_value={'exposure_sha256':digest({'fixture':'exposure'})}), \
             patch('llm_intervention_campaign.run_policy',side_effect=run), \
             patch('llm_intervention_campaign._load_attempt',side_effect=load), \
             patch('llm_intervention_campaign.finalize_campaign',side_effect=AssertionError('no private finalization')):
            if token_fail:
                with self.assertRaisesRegex(ValueError,'intentional token'):phase.run_cpu_phase(**args)
                self.assertFalse((root/'campaign').exists());self.assertEqual(calls,[])
                self.assertTrue((root/'cpu/failure.json').exists())
                self.assertFalse((root/'cpu/terminal.json').exists())
            else:
                terminal=phase.run_cpu_phase(**args)
                self.assertEqual(calls,list(builder.POLICY_IDS[:5]))
                self.assertLess(events.index('campaign_frozen_before_responses'),events.index('fake_policy'))
                self.assertEqual(terminal['gpu_services_started'],0)
                self.assertFalse(terminal['private_scores_computed'])
                self.assertFalse(terminal['scientific_qualified'])
                self.assertEqual(terminal['failed_policies'],[failed_policy] if failed_policy else [])
                self.assertEqual(terminal['status'],'cpu_phase_contains_failures' if failed_policy else 'cpu_phase_complete')
                before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                with self.assertRaises(FileExistsError):phase.run_cpu_phase(**args)
                self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_fixed_shape_probes_and_heterogeneous_affordability(self):
        bundle=builder.build_artificial_campaign(**fixture())
        for source in bundle['campaign']['sources']:
            rows=phase.shape_envelopes(source['plan'],plan_sha256=source['plan_sha256'])
            self.assertEqual(len(rows),2)
            for row,count in zip(rows,(2,5)):
                state,_,_,menu,_=unpack_envelope(row['envelope'])
                self.assertEqual((len(state['history']),state['remaining_units']),(count,6-count))
                self.assertEqual(row['environment_responses'],0)
                self.assertTrue(row['fixture'].startswith('fabricated_'))
            feasible=[a.action_id for a in menu if a.cost_units<=1]
            self.assertEqual('x_plus' in feasible,source['plan']['world']['world_id']==9101)

    def test_freeze_and_tokenization_precede_all_five_cpu_controls(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d))

    def test_token_failure_preserved_before_any_campaign_response(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),token_fail=True)

    def test_failed_arm_retained_without_gpu_admission_or_private_scoring(self):
        with tempfile.TemporaryDirectory() as d:self.exercise(Path(d),failed_policy='uniform_random')

    def test_gpu_assignment_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=self.inputs(root);args['event']=lambda *a,**k:None
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'},clear=True),self.assertRaises(ValueError):
                phase.run_cpu_phase(**args)
            self.assertFalse((root/'cpu').exists())


if __name__=='__main__':unittest.main()
