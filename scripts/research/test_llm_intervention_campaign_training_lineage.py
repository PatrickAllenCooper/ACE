"""New pure role-admission fixtures. Fabricated pins are not actual qualification."""
import copy
import json
import unittest
from unittest.mock import patch

import llm_intervention_campaign_training_lineage as lineage


def raw(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


class LineageTests(unittest.TestCase):
    def make(self):
        plan = {'scope':'artificial','fixture':'only'}
        exposure = {'fixture':'matched'}
        prepared = {'training_plan':plan,'training_plan_sha256':lineage._semantic(plan),'exposure':exposure}
        terminal = {'status':'complete','completed_steps':6,'sft_steps':4,'dpo_steps':2,
                    'environment_responses':0,'plan_sha256':lineage._semantic(plan)}
        sft, final = b'opaque-sft-fixture', b'opaque-final-fixture'
        def checkpoint(count,stage,data):
            return raw({'completed_steps':count,'attempt':{'index':count-1,'stage':stage,
                'optimizer_step_started':True,'optimizer_step_returned':True},
                'files':{'adapter':lineage._sha(data)}})
        sc,fc = checkpoint(4,'sft',sft),checkpoint(6,'dpo',final)
        acceptance = {'status':'artificial_integrated_training_gpu_accepted','job_id':lineage.JOB_ID,
            'training_gpu_qualified':True,'checkpoint_reload_inference_qualified':True,
            'scientific_qualified':False,'required_remaining':0,'engine_terminal_sha256':lineage._sha(raw(terminal)),
            'training_plan_sha256':lineage._semantic(plan),'exposure_sha256':lineage._semantic(exposure),
            'sft_adapter_sha256':lineage._sha(sft),'final_adapter_sha256':lineage._sha(final)}
        inputs = {'acceptance_raw':raw(acceptance),'prepared_raw':raw(prepared),'engine_terminal_raw':raw(terminal),
            'sft_checkpoint_raw':sc,'final_checkpoint_raw':fc,'sft_adapter_raw':sft,'final_adapter_raw':final}
        patches = {'ACCEPTANCE_SHA256':lineage._sha(inputs['acceptance_raw']),
                   'PREPARED_SHA256':lineage._sha(inputs['prepared_raw']),
                   'CHECKPOINT_PINS':{'sft':lineage._sha(sc),'final':lineage._sha(fc)}}
        return inputs,patches

    def test_exact_artificial_roles_keep_checkpoint_and_acceptance_pins_distinct(self):
        inputs, pins = self.make()
        with patch.multiple(lineage,**pins):
            result = lineage.admit_training(**inputs)
        self.assertEqual(list(result['roles']),['sft_llm','dpo_llm'])
        for pid, stage, steps in (('sft_llm','sft',4),('dpo_llm','final',6)):
            r = result['roles'][pid]
            self.assertEqual(r['checkpoint_sha256'],pins['CHECKPOINT_PINS'][stage])
            self.assertNotEqual(r['checkpoint_sha256'],result['acceptance_sha256'])
            self.assertEqual(r['completed_steps'],steps)
            self.assertEqual(r['treatment'],'artificial_adapter')
            self.assertEqual(r['training_scope'],'artificial')
        self.assertFalse(result['scientific_qualified'])
        self.assertEqual(result['new_training_updates'],0)

    def test_swapped_roles_altered_receipts_and_unpinned_inputs_rejected(self):
        inputs,pins=self.make()
        for key in inputs:
            bad = dict(inputs); bad[key] += b' '
            with self.subTest(key=key), patch.multiple(lineage,**pins), self.assertRaises(ValueError):
                lineage.admit_training(**bad)
        bad = dict(inputs)
        bad['sft_adapter_raw'],bad['final_adapter_raw']=bad['final_adapter_raw'],bad['sft_adapter_raw']
        with patch.multiple(lineage,**pins), self.assertRaises(ValueError): lineage.admit_training(**bad)

    def test_fabricated_repin_does_not_bypass_scope_steps_or_completion_checks(self):
        for key,value in [('scientific_qualified',True),('training_gpu_qualified',False),
                          ('required_remaining',1),('job_id','other')]:
            inputs,pins=self.make(); a=json.loads(inputs['acceptance_raw']); a[key]=value
            inputs['acceptance_raw']=raw(a); pins['ACCEPTANCE_SHA256']=lineage._sha(inputs['acceptance_raw'])
            with self.subTest(key=key),patch.multiple(lineage,**pins),self.assertRaises(ValueError):
                lineage.admit_training(**inputs)
        inputs,pins=self.make(); c=json.loads(inputs['final_checkpoint_raw']); c['completed_steps']=5
        inputs['final_checkpoint_raw']=raw(c); pins['CHECKPOINT_PINS']['final']=lineage._sha(raw(c))
        with patch.multiple(lineage,**pins),self.assertRaises(ValueError): lineage.admit_training(**inputs)

    def test_nonbyte_and_duplicate_objects_rejected(self):
        inputs,pins=self.make(); inputs['sft_adapter_raw']='not bytes'
        with patch.multiple(lineage,**pins),self.assertRaises(ValueError): lineage.admit_training(**inputs)
        with self.assertRaises(ValueError): lineage._decode(b'{"a":1,"a":2}')


if __name__ == '__main__': unittest.main()
