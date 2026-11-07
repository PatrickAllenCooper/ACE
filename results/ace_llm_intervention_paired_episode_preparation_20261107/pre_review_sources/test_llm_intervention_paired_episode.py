"""New end-to-end supplied artificial-world tests, no pretrained models."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from llm_intervention_contract import Action, Node, canonical, digest
from llm_intervention_task import PredictionTask, TaskProcessPolicy
from llm_intervention_world import WorldDescriptor, LinearMechanism, ResponseAdapter
from llm_intervention_persistent_policy import PersistentPolicy
import llm_intervention_paired_episode as paired


WORKER = '''import json,sys
data=json.load(sys.stdin)
assert set(data)=={'version','state','task'}
assert 'mechanisms' not in data['state'] and 'private_evaluation' not in data
state=data['state']
kind=sys.argv[1]
if kind=='broken': print('not JSON')
elif kind=='stop': print('{"stop":true}')
elif kind=='costly': print(json.dumps({'action_id':'costly' if state['remaining_units']>=2 else 'cheap'}))
else: print('{"action_id":"observe"}')
'''


class PairedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ACE_PAIRED_TEST_OUTPUT'])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.path = self.root/self._testMethodName
        self.path.mkdir()
        self.worker = self.path/'worker.py'
        self.worker.write_text(WORKER)
        self.graph = (Node('X', (), 'unit', (-3., 3.)),)
        self.world = WorldDescriptor('artificial', 14, self.graph, (LinearMechanism('X', 2., (), 0.),))
        self.menu = (Action('observe','observe',(),('X',),1),
            Action('cheap','intervene',(('X',0.),),('X',),1),
            Action('costly','intervene',(('X',1.),),('X',),2))
        self.task = PredictionTask('X',(Action('q','observe',(),('X',),1),),(1.,),2.,.1)

    def make(self, ids, persistent=None):
        specs, policies = [], {}
        for pid in ids:
            command = [sys.executable,'-B',str(self.worker),pid]
            spec = {'policy_id':pid,'kind':'task_process','command':command,'timeout_seconds':5.,
                'identity_sha256':None,'policy_provenance_sha256':digest(['artificial',pid])}
            if persistent and pid == 'persistent':
                command, identity = persistent
                spec.update(kind='persistent',command=list(command),identity_sha256=digest(identity))
            else:
                policies[pid] = TaskProcessPolicy(tuple(command),5.,self.task)
            specs.append(spec)
        args = dict(world=self.world,world_sha256=self.world.sha256,task=self.task,menu=self.menu,
            noise_variances={'X':1.},prior_variance=1.,initial_actions=('observe',),acquisition_cap_units=3,
            initial_binding={'request_namespace':'shared','replicate_id':'initial'},
            measurement_bindings={pid:{'request_namespace':pid,'replicate_id':'measure'} for pid in ids},
            policy_specs=tuple(specs),objective='Artificial paired prediction task',domain_notes='Hand specified control.',
            outer_provenance_receipt_sha256='a'*64)
        plan = paired.prepare_plan(**args)
        paired._save(self.path/'input-plan.json',plan)
        return plan, policies, args

    def run_arm(self, plan, pid, policy, suffix=''):
        output=self.path/('arm-'+pid+suffix)
        paired.run_world_arm(plan=plan,plan_sha256=digest(plan),policy_id=pid,policy=policy,output=output)
        return {'path':str(output),'receipt_sha256':paired._sha((output/'arm-receipt.json').read_bytes()),'missing_reason':None}

    def finish(self, plan, arms, name='report', closed=None):
        return paired.finalize_paired_world(plan=plan,plan_sha256=digest(plan),arm_outputs=arms,
            persistent_closed={} if closed is None else closed,output=self.path/name)

    def test_actual_public_processes_to_hand_computed_private_costs(self):
        plan,policies,_=self.make(('inform','stop','costly','broken','missing'))
        with patch('llm_intervention_world.private_structural_means',side_effect=AssertionError('no private scoring during execution')), \
                patch('llm_intervention_cost_curves.private_structural_means',side_effect=AssertionError('no private reporting during execution')):
            arms={pid:self.run_arm(plan,pid,policies[pid]) for pid in ('inform','stop','costly','broken')}
        arms['missing']={'path':None,'receipt_sha256':None,'missing_reason':'fixture deliberately never started'}
        report=self.finish(plan,arms)
        by={x['policy_id']:x for x in report['rows']}
        self.assertEqual([by[k]['capped_acquisition_cost_units'] for k in plan_ids(plan)],[2,3,3,3,3])
        self.assertAlmostEqual(by['inform']['final_error'],.04)
        self.assertEqual(by['stop']['final_error'],.25)
        self.assertEqual(by['costly']['final_error'],.25)
        self.assertTrue(by['broken']['failed'])
        self.assertEqual(by['missing']['cell_status'],'unresolved')
        self.assertEqual(len(report['policy_pairs']),10)
        self.assertEqual(report['cell_count'],5)
        curve=json.loads((self.path/'report/arm-00/cost-report.json').read_bytes())['curve']
        expected=[.25,1/9,1/16,1/25]
        for p,e in zip(curve,expected):self.assertAlmostEqual(p['normalized_mse'],e)
        self.assertEqual([p['acquisition_cost_units'] for p in curve],[0,1,2,3])
        for pid in ('inform','stop','costly','broken'):
            original=json.loads((Path(arms[pid]['path'])/'return-000.json').read_bytes())
            self.assertEqual(original['response']['raw_json'],{'X':2.})

    def test_prepared_outcome_is_not_counted_as_returned(self):
        plan,policies,_=self.make(('inform','lost'))
        arms={'inform':self.run_arm(plan,'inform',policies['inform'])}
        original=ResponseAdapter.__call__
        def lost(adapter,action,request):
            value=original(adapter,action,request)
            if adapter._stream=='measurement':raise RuntimeError('injected after prepared outcome before return')
            return value
        with patch.object(ResponseAdapter,'__call__',lost):arms['lost']=self.run_arm(plan,'lost',policies['lost'])
        self.finish(plan,arms)
        report=json.loads((self.path/'report/arm-01/cost-report.json').read_bytes())
        self.assertEqual(report['charges']['unknown_returns'],1)
        self.assertEqual(report['charges']['known_returns'],1)
        self.assertEqual(report['charges']['actual_total_cost_units'],2)
        self.assertEqual(report['capped_cost_units'],3)
        self.assertFalse((Path(arms['lost']['path'])/'return-001.json').exists())
        self.assertIn('"event":"outcome"',(Path(arms['lost']['path'])/'measurement-world.jsonl').read_text())

    def test_invalid_return_and_partial_initial_failure_are_retained(self):
        plan,policies,_=self.make(('inform','invalid','partial'))
        arms={'inform':self.run_arm(plan,'inform',policies['inform'])}
        original=ResponseAdapter.__call__
        def invalid(adapter,action,request):
            value=original(adapter,action,request)
            return {'wrong':2.} if adapter._stream=='measurement' else value
        with patch.object(ResponseAdapter,'__call__',invalid):arms['invalid']=self.run_arm(plan,'invalid',policies['invalid'])
        def partial(adapter,action,request):raise RuntimeError('injected initial response unknown')
        with patch.object(ResponseAdapter,'__call__',partial):arms['partial']=self.run_arm(plan,'partial',policies['partial'])
        report=self.finish(plan,arms)
        invalid_report=json.loads((self.path/'report/arm-01/cost-report.json').read_bytes())
        partial_report=json.loads((self.path/'report/arm-02/cost-report.json').read_bytes())
        self.assertEqual(invalid_report['charges']['by_outcome']['returned_invalid']['attempts'],1)
        self.assertEqual(partial_report['charges']['unknown_returns'],1)
        self.assertIsNone(partial_report['final_error'])
        self.assertTrue(all(r['capped_acquisition_cost_units']==3 for r in report['rows'][1:]))

    def test_conflicting_initial_histories_never_select_an_authoritative_arm(self):
        plan,policies,_=self.make(('inform','different'))
        arms={'inform':self.run_arm(plan,'inform',policies['inform'])}
        original=ResponseAdapter.__call__
        def different(adapter,action,request):
            value=original(adapter,action,request)
            return {'X':3.} if adapter._stream=='initial' else value
        with patch.object(ResponseAdapter,'__call__',different):arms['different']=self.run_arm(plan,'different',policies['different'])
        with patch('llm_intervention_cost_curves.report_cost_curve',side_effect=AssertionError('no scoring conflicting pair')):
            report=self.finish(plan,arms)
        self.assertTrue(all(r['cell_status']=='unresolved' for r in report['rows']))
        self.assertTrue(all(r['capped_acquisition_cost_units']==3 for r in report['rows']))
        bad=copy.deepcopy(arms);bad['inform']['receipt_sha256']='0'*64
        other=self.finish(plan,bad,name='tampered-report')
        self.assertEqual(other['rows'][0]['cell_status'],'unresolved')

    def test_actual_persistent_cleanup_required_before_private_reporting(self):
        identity={'model_sha256':'1'*64,'source_sha256':'2'*64,'adapter_sha256':'3'*64,
            'numeric_recipe':{'artificial':True},'numeric_recipe_sha256':digest({'artificial':True})}
        worker=self.path/'persistent-worker.py'
        worker.write_text('import sys\nsys.path.insert(0,'+repr(str(Path(paired.__file__).parent))+')\n'
            'from llm_intervention_persistent_policy import serve_policy\n'
            'serve_policy(lambda: lambda envelope: \'{"action_id":"observe"}\', identity='+repr(identity)+')\n')
        command=(sys.executable,'-B',str(worker))
        plan,_,_=self.make(('persistent',),persistent=(command,identity))
        policy=PersistentPolicy(command,5.,expected_identity=identity,output=self.path/'service',startup_seconds=10.)
        try:arms={'persistent':self.run_arm(plan,'persistent',policy)}
        finally:policy.close()
        raw=(self.path/'service/closed.json').read_bytes()
        report=self.finish(plan,arms,closed={'persistent':{'raw':raw,'sha256':paired._sha(raw)}})
        self.assertEqual(report['rows'][0]['capped_acquisition_cost_units'],2)
        bad=json.loads(raw);bad['cleanup']['leader_returncode']=1;bad=paired._raw(bad)
        with patch('llm_intervention_cost_curves.report_cost_curve',side_effect=AssertionError('must not score failed service')):
            unresolved=self.finish(plan,arms,name='unclean-service',closed={'persistent':{'raw':bad,'sha256':paired._sha(bad)}})
        self.assertEqual(unresolved['rows'][0]['cell_status'],'unresolved')

    def test_invalid_plan_or_policy_rejects_before_responses(self):
        plan,policies,args=self.make(('inform','stop'))
        with patch.object(paired,'ResponseAdapter',side_effect=AssertionError('must not call environment')):
            with self.assertRaisesRegex(ValueError,'plan pin'):
                paired.run_world_arm(plan=plan,plan_sha256='0'*64,policy_id='inform',policy=policies['inform'],output=self.path/'absent')
            with self.assertRaisesRegex(ValueError,'policy differs'):
                paired.run_world_arm(plan=plan,plan_sha256=digest(plan),policy_id='inform',policy=policies['stop'],output=self.path/'absent')
            duplicate=copy.deepcopy(args);duplicate['measurement_bindings']['stop']=duplicate['measurement_bindings']['inform']
            with self.assertRaisesRegex(ValueError,'distinct'):paired.prepare_plan(**duplicate)
        self.assertFalse((self.path/'absent').exists())


def plan_ids(plan):
    return [s['policy_id'] for s in plan['policy_specs']]


if __name__=='__main__':unittest.main(verbosity=2)
