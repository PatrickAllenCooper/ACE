import copy
import os
import math
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_prospective_batch import (expected_world_ids,intended_cells,validate_matrix,
    validate_descriptors,load_descriptors,allocation_plan,evaluate,training_variance,HISTORIES)
from runner_delivery_confirmation import write,read

DRAFT=Path(os.environ.get('ACE_DELIVERY_DRAFT','/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-draft-20261006'))
MANIFEST=Path(os.environ.get('ACE_DELIVERY_DRAFT_MANIFEST',str(DRAFT/'manifest.json')))


class ProspectiveBatchTests(unittest.TestCase):
    def test_full_membership_rejects_missing_duplicate_and_selected_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);write(out/'registration.json',{'fixture':'metadata only; no outcomes'})
            p={'worlds':expected_world_ids(),'source_hashes':{},'dependencies':{},
               'worker_hashes':{'delivery_prospective_models.py':'fixture'}}
            inputs={case+'/'+history:'training-only-sha' for case in p['worlds'] for history in HISTORIES}
            cells=intended_cells(out,p,inputs)
            self.assertEqual(len(cells),640)
            primary=[c for c in cells if c['init']==0 and c['arm'] in ('delivery','online','simpler')]
            self.assertEqual(len(primary),240)
            self.assertEqual(len({(c['case'],c['history'],c['arm'],c['init']) for c in cells}),640)
            validate_matrix(cells,out,p,inputs)
            altered=copy.deepcopy(cells);altered[0]['init']=2
            for bad in (cells[:-1],cells[:-1]+[cells[0]],altered):
                with self.assertRaises(ValueError):validate_matrix(bad,out,p,inputs)

    def test_world_identity_and_heldout_menus_cannot_change(self):
        manifest=read(MANIFEST);descriptors=load_descriptors(DRAFT,manifest)
        # This validates descriptor/action metadata, not a selected response.
        bad=copy.deepcopy(descriptors);case='5:00'
        bad[case][1]['balanced'][0]=bad[case][1]['evaluation'][0]
        with self.assertRaises(ValueError):validate_descriptors(manifest,bad)
        changed=copy.deepcopy(manifest);del changed['worlds']['30:19']
        with self.assertRaises(ValueError):validate_descriptors(changed,descriptors)

    def test_requested_allocations_include_rounding_and_remain_under_ceiling(self):
        p={'estimated_full_cpu_core_hours':20.,'strata':{'5':{'raw_full_fit_cpu_seconds':20*100.},
                                                       '30':{'raw_full_fit_cpu_seconds':20*1000.}}}
        r=allocation_plan(p)
        self.assertEqual(r['threads'],1)
        self.assertEqual(r['historical_failed_pilot_cpu_seconds'],126)
        self.assertTrue(all(s%900==0 for s in r['world_wall_seconds'].values()))
        self.assertLessEqual(r['total_requested_cpu_core_hours'],150)
        bad=copy.deepcopy(p);bad['estimated_full_cpu_core_hours']=151
        with self.assertRaises(ValueError):allocation_plan(bad)
        # A plausible fit projection cannot hide excessive rounded requests.
        bad=copy.deepcopy(p);bad['strata']['30']['raw_full_fit_cpu_seconds']=20*12000.
        with self.assertRaises(ValueError):allocation_plan(bad)

    def test_qualification_imports_frozen_project_from_unrelated_working_directory(self):
        from delivery_prospective_batch import phase_environment
        project=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            env=phase_environment({'project':str(project),'source':'unused in import-only check'},directory)
            # No model, selected outcome or simulator is loaded by this probe.
            result=subprocess.check_output([sys.executable,'-c',
                'import importlib.util; print(importlib.util.find_spec("experiments.large_scale_scm").origin)'],
                cwd=directory,env=env,text=True).strip()
            self.assertEqual(Path(result).resolve(),project/'experiments/large_scale_scm.py')

    def test_coefficient_roundoff_qualification_never_changes_frozen_descriptor(self):
        from delivery_prospective_batch import descriptor_runtime_parity
        frozen={'seed':17,'parents':{'R':[],'Y':['R']},'coefficients':{'a':.5},'noise_sd':.1,'source':'local'}
        before=copy.deepcopy(frozen);other=copy.deepcopy(frozen)
        other['coefficients']['a']=math.nextafter(.5,1.);other['source']='remote'
        differences=descriptor_runtime_parity(frozen,other)
        self.assertEqual(len(differences),1);self.assertEqual(frozen,before)
        cases=[]
        two=copy.deepcopy(other);two['coefficients']['a']=math.nextafter(two['coefficients']['a'],1.);cases.append(two)
        noise=copy.deepcopy(other);noise['noise_sd']=math.nextafter(.1,1.);cases.append(noise)
        graph=copy.deepcopy(other);graph['parents']['Y']=[];cases.append(graph)
        seed=copy.deepcopy(other);seed['seed']=18;cases.append(seed)
        nonfinite=copy.deepcopy(other);nonfinite['coefficients']['a']=float('nan');cases.append(nonfinite)
        for altered in cases:
            with self.assertRaises(ValueError):descriptor_runtime_parity(frozen,altered)

    def test_failure_blocks_test_generation_before_any_model_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);write(out/'stop_new_fits.json',{'retained_failure':True})
            with patch('delivery_prospective_batch.validate',return_value={}), \
                 patch('delivery_prospective_batch.validate_collection',return_value=[]), \
                 patch('delivery_prospective_batch.sealed_fits') as models, \
                 patch('delivery_prospective_batch.collect_history') as queries:
                with self.assertRaises(ValueError):evaluate(out)
                models.assert_not_called();queries.assert_not_called()
            self.assertFalse((out/'evaluation_started.json').exists())

    def test_nonpositive_or_nonfinite_training_normalizer_fails_before_fitting(self):
        for values in ([0.,0.],[1.,float('nan')],[1.,float('inf')]):
            data={'target':'Y','rows':[{'node_values':{'Y':v}} for v in values]}
            with self.assertRaises(ValueError):training_variance(data)
        self.assertAlmostEqual(training_variance({'target':'Y','rows':[
            {'node_values':{'Y':1.}},{'node_values':{'Y':3.}}]}),1.)

    def test_slurm_requests_only_cpu_and_respects_full_registered_membership(self):
        from delivery_prospective_slurm import scripts_for
        projection={'estimated_full_cpu_core_hours':20.,'strata':{
            '5':{'raw_full_fit_cpu_seconds':2000.},'30':{'raw_full_fit_cpu_seconds':20000.}}}
        resources=allocation_plan(projection)
        scripts=scripts_for('/scratch/alpine/paco0228/ACE/results/fixture','/usr/bin/python',
            '/project/scripts/research/delivery_prospective_batch.py',resources)
        self.assertEqual(set(scripts),{'qualification','collect','fit5','fit30','evaluate','audit'})
        for script in scripts.values():
            self.assertIn('--account=ucb736_asc1',script)
            self.assertIn('--cpus-per-task=1',script)
            self.assertNotIn('--gres',script)
        self.assertIn('--time=00:15:00',scripts['audit'])
        for name,size in (('fit5',5),('fit30',30)):
            self.assertIn('--array=0-19%2',scripts[name])
            self.assertIn(f"'{size}:%02d'",scripts[name])
        bad=copy.deepcopy(resources);bad['threads']=6
        with self.assertRaises(ValueError):scripts_for('/scratch/out','/usr/bin/python','/project/worker',bad)


if __name__=='__main__':unittest.main()
