"""Fabricated metadata and opaque bytes only. No actual checkpoint is loaded."""
import copy
import json
import os
from pathlib import Path
import unittest

import llm_intervention_scientific_training_lineage as L


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class ScientificLineageTests(unittest.TestCase):
    def make(self, d=7):
        plan = {'scope':'policy_train', 'records':[{'record_id':r} for r in L.ROSTER],
            'execution_kind':'qwen_screen_singletons', 'sft_steps':160, 'dpo_steps':d,
            'checkpoint_steps':[159,159+d]}
        exposure = {'scope':'policy_train', 'record_roster':list(L.ROSTER),
            'counts':{'dpo_steps':d}, 'has_both_training_stages':True}
        p = {'version':'ace-complete-collection-training-plan-v1', 'training_plan':plan,
            'exposure':exposure, 'training_plan_sha256':L._semantic(plan), 'exposure_sha256':L._semantic(exposure)}
        t = {'status':'complete', 'execution_kind':'qwen_screen_singletons',
            'plan_sha256':L._semantic(plan), 'completed_steps':160+d, 'sft_steps':160,
            'dpo_steps':d, 'environment_responses':0}
        sft, final = b'opaque-fabricated-sft', b'opaque-fabricated-dpo'
        def cp(n, stage, adapter):
            return raw({'completed_steps':n, 'files':{'adapter':L._sha(adapter)},
                'attempt':{'index':n-1,'stage':stage,'optimizer_step_started':True,'optimizer_step_returned':True}})
        sc, fc = cp(160,'sft',sft), cp(160+d,'dpo',final)
        expected = {k:L._sha(k.encode()) for k in L.EXPECTED_PINS}
        expected.update(prepared_sha256=L._sha(raw(p)), training_plan_sha256=L._semantic(plan),
            exposure_sha256=L._semantic(exposure), job_id='987654321', source_commit='a'*40)
        a = {'schema':'ace-scientific-training-gpu-acceptance-v1',
            'status':'scientific_training_gpu_accepted', 'scope':'policy_train',
            'required_findings':0, 'scientific_utility_qualified':False, **expected,
            'engine_terminal_sha256':L._sha(raw(t)), 'sft_checkpoint_sha256':L._sha(sc),
            'final_checkpoint_sha256':L._sha(fc), 'sft_adapter_sha256':L._sha(sft),
            'final_adapter_sha256':L._sha(final), 'completed_steps':160+d,
            'sft_steps':160, 'dpo_steps':d, 'environment_responses':0}
        for k in ('training_gpu_qualified','checkpoint_reload_inference_qualified','complete_custody_verified',
                  'runtime_qualified','owned_process_cleanup_verified','scheduler_verified'):a[k]=True
        inputs = dict(acceptance_raw=raw(a), acceptance_sha256=L._sha(raw(a)), expected=expected,
            prepared_raw=raw(p), engine_terminal_raw=raw(t), sft_checkpoint_raw=sc,
            sft_adapter_raw=sft, final_checkpoint_raw=fc, final_adapter_raw=final)
        if os.environ.get('ACE_LINEAGE_TEST_EVIDENCE'):
            root=Path(os.environ['ACE_LINEAGE_TEST_EVIDENCE'])/self._testMethodName
            root.mkdir(exist_ok=True)
            n=len(list(root.iterdir())); dest=root/str(n);dest.mkdir()
            for name,value in inputs.items():
                (dest/name).write_bytes(value if type(value) is bytes else raw(value))
        return inputs

    def changed_acceptance(self, inputs, **changes):
        out=copy.deepcopy(inputs);a=json.loads(out['acceptance_raw']);a.update(changes)
        out['acceptance_raw']=raw(a);out['acceptance_sha256']=L._sha(raw(a));return out

    def test_roles_and_pin_domains_are_distinct(self):
        x=self.make();r=L.admit_training(**x)
        self.assertEqual(r['roles']['sft_llm']['treatment'],'sft')
        self.assertEqual(r['roles']['dpo_llm']['treatment'],'dpo')
        self.assertEqual(r['roles']['sft_llm']['completed_steps'],160)
        self.assertEqual(r['roles']['dpo_llm']['completed_steps'],167)
        for role in r['roles'].values():
            self.assertEqual(role['training_scope'],'policy_train')
            self.assertNotEqual(role['checkpoint_sha256'],r['acceptance_sha256'])
        self.assertFalse(r['development_execution_qualified'])
        self.assertFalse(r['scientific_utility_qualified'])
        self.assertFalse(r['tensor_files_deserialized'])
        self.assertEqual((r['new_training_updates'],r['environment_responses']),(0,0))

    def test_all_captured_byte_changes_are_rejected(self):
        x=self.make()
        for key in x:
            if key.endswith('_raw'):
                y=copy.deepcopy(x);y[key]+=b' '
                with self.subTest(key=key),self.assertRaises(ValueError):L.admit_training(**y)
        y=copy.deepcopy(x);y['sft_adapter_raw'],y['final_adapter_raw']=y['final_adapter_raw'],y['sft_adapter_raw']
        with self.assertRaises(ValueError):L.admit_training(**y)

    def test_independent_expected_lineage_cannot_be_replaced_by_repinning_acceptance(self):
        x=self.make()
        for key in x['expected']:
            with self.subTest(key=key),self.assertRaises(ValueError):
                L.admit_training(**self.changed_acceptance(x,**{key:'b'*64}))
        for bad in ({},dict(x['expected'],extra='x'),dict(x['expected'],job_id='0'),dict(x['expected'],source_commit='a'*39)):
            y=copy.deepcopy(x);y['expected']=bad
            with self.assertRaises(ValueError):L.admit_training(**y)

    def test_artificial_partial_or_unaccepted_results_rejected(self):
        x=self.make()
        changes={'schema':'ace-artificial-campaign-training-lineage-v1','status':'worker_complete',
            'scope':'artificial','required_findings':False,'scientific_utility_qualified':True,
            'environment_responses':1,'completed_steps':166,'sft_steps':True}
        changes.update({k:False for k in ('training_gpu_qualified','checkpoint_reload_inference_qualified',
            'complete_custody_verified','runtime_qualified','owned_process_cleanup_verified','scheduler_verified')})
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                L.admit_training(**self.changed_acceptance(x,**{key:value}))

    def test_dpo_range_and_fixed_roster(self):
        for d in (1,128):self.assertEqual(L.admit_training(**self.make(d))['roles']['dpo_llm']['completed_steps'],160+d)
        for d in (0,129):
            with self.assertRaises(ValueError):L.admit_training(**self.make(d))
        x=self.make();p=json.loads(x['prepared_raw']);p['training_plan']['records'].reverse()
        p['training_plan_sha256']=L._semantic(p['training_plan']);x['prepared_raw']=raw(p)
        x['expected'].update(prepared_sha256=L._sha(raw(p)),training_plan_sha256=p['training_plan_sha256'])
        x=self.changed_acceptance(x,**x['expected'])
        with self.assertRaises(ValueError):L.admit_training(**x)

    def test_rebound_checkpoint_requires_completed_correct_role(self):
        x=self.make()
        for field,value in (('completed_steps',166),('index',165),('stage','sft'),
                            ('optimizer_step_started',False),('optimizer_step_returned',False)):
            y=copy.deepcopy(x);cp=json.loads(y['final_checkpoint_raw'])
            if field=='completed_steps':cp[field]=value
            else:cp['attempt'][field]=value
            y['final_checkpoint_raw']=raw(cp)
            y=self.changed_acceptance(y,final_checkpoint_sha256=L._sha(raw(cp)))
            with self.subTest(field=field),self.assertRaises(ValueError):L.admit_training(**y)

    def test_rebound_engine_terminal_requires_complete_same_plan(self):
        x=self.make()
        for key,value in (('status','failed'),('environment_responses',True),('dpo_steps',6),
                          ('plan_sha256','f'*64),('execution_kind','other')):
            y=copy.deepcopy(x);t=json.loads(y['engine_terminal_raw']);t[key]=value
            y['engine_terminal_raw']=raw(t);y=self.changed_acceptance(y,engine_terminal_sha256=L._sha(raw(t)))
            with self.subTest(key=key),self.assertRaises(ValueError):L.admit_training(**y)

    def test_json_and_input_types_fail_closed(self):
        for value in (b'{"x":1,"x":2}',b'{"x":NaN}',b'[]'):
            with self.assertRaises(ValueError):L._decode(value)
        x=self.make();x['sft_adapter_raw']='not captured bytes'
        with self.assertRaises(ValueError):L.admit_training(**x)


if __name__=='__main__':unittest.main()
