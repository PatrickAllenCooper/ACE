"""Pure completion/freeze contract checks. No model imports or accelerator use."""
import copy
import unittest
from qualify_llm_intervention_task_gpu import SOURCES,MODELS,PACKAGES,ASSETS,RECIPE,validate_freeze,validate_terminal


def fixture():
    freeze={'schema':'ace-shared-task-gpu-v1','sources':dict.fromkeys(SOURCES,'a'*64),
            'model_files':dict.fromkeys(MODELS,'b'*64),'runtime':dict.fromkeys(PACKAGES,'test'),
            'assets':dict.fromkeys(ASSETS,'c'*64),'recipe':dict(RECIPE),'startup_seconds':420,'worker_seconds':840}
    terminal={'status':'complete','freeze_sha256':'d'*64,'source_pins':freeze['sources'],
              'sft_steps':3,'dpo_steps':3,'scientific_responses':0,'environment_responses':0,
              'generation_count':4,'cuda_step_ms_total':6.,'peak_gpu_allocated_bytes':100,
              'frozen_base_before':'a','frozen_base_after':'a','reference_before':'b','reference_after':'b',
              'adapter_before':'c','adapter_after':'d','selected_scoring_equivalence':True,
              'accepted_adapter_restored':True,'trained_adapter_reload_score_max_difference':0,
              'activation_checkpointing_used':False}
    steps=[{'stage':s,'step':i,'cuda_elapsed_ms':1.} for s in ('sft','dpo') for i in range(3)]
    return freeze,terminal,steps


class Contracts(unittest.TestCase):
    def test_positive(self):
        f,t,s=fixture();validate_freeze(f);validate_terminal(t,f,'d'*64,s)
    def test_changed_source_assets_recipe(self):
        f,_,_=fixture()
        for section in ('sources','model_files','assets'):
            bad=copy.deepcopy(f);bad[section].pop(next(iter(bad[section])))
            with self.assertRaises(ValueError):validate_freeze(bad)
        for key,value in (('activation_checkpointing',True),('sdpa_backend','automatic'),('max_tokens',16384)):
            bad=copy.deepcopy(f);bad['recipe'][key]=value
            with self.assertRaises(ValueError):validate_freeze(bad)
    def test_bad_pin_and_deadlines(self):
        f,_,_=fixture()
        bad=copy.deepcopy(f);bad['assets']['case0.json']='z'*64
        with self.assertRaises(ValueError):validate_freeze(bad)
        for key in ('startup_seconds','worker_seconds'):
            bad=copy.deepcopy(f);bad[key]+=1
            with self.assertRaises(ValueError):validate_freeze(bad)
    def test_terminal_rejects_fabricated_success(self):
        f,t,s=fixture()
        for key,value in (('environment_responses',1),('scientific_responses',1),('generation_count',3),
                          ('frozen_base_after','changed'),('reference_after','changed'),
                          ('adapter_after','c'),('trained_adapter_reload_score_max_difference',.001),
                          ('selected_scoring_equivalence',False),('accepted_adapter_restored',False),
                          ('activation_checkpointing_used',True)):
            bad={**t,key:value}
            with self.assertRaises(ValueError):validate_terminal(bad,f,'d'*64,s)
    def test_steps_complete_unique_finite(self):
        f,t,s=fixture()
        for bad in (s[:-1],s[:-1]+[s[0]],[{**x,'cuda_elapsed_ms':float('nan')} for x in s],
                    [{**x,'cuda_elapsed_ms':0} for x in s]):
            with self.assertRaises(ValueError):validate_terminal(t,f,'d'*64,bad)


class ComparisonCustody(unittest.TestCase):
    def test_success_and_both_failure_diagnostics(self):
        import json,tempfile
        from pathlib import Path
        import torch
        from qualify_llm_intervention_task_gpu import compare_and_record
        for name,other_loss,other_gradient,should_fail in [('pass',1.,1.,False),('loss',2.,1.,True),('gradient',1.,2.,True)]:
            with tempfile.TemporaryDirectory() as d:
                observations=[(1.,{'q.A':torch.tensor([1.])}),(other_loss,{'q.A':torch.tensor([other_gradient])})]
                if should_fail:
                    with self.assertRaises(ValueError):compare_and_record(Path(d),name,observations)
                else:compare_and_record(Path(d),name,observations)
                r=json.loads((Path(d)/('comparison-'+name+'.json')).read_bytes())
                self.assertEqual(r['loss_absolute_difference'],abs(1.-other_loss))
                self.assertEqual(r['tensor_details'][0]['max_absolute_difference'],abs(1.-other_gradient))
                self.assertEqual(r['tensor_details'][0]['name'],'q.A')

if __name__=='__main__':unittest.main(verbosity=2)
