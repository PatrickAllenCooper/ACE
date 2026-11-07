"""NEW hand-written artificial campaign fixtures; never runs an older suite."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import llm_intervention_campaign as campaign
import llm_intervention_paired_episode as paired
from llm_intervention_contract import Action, Node, digest
from llm_intervention_persistent_policy import PersistentPolicy
from llm_intervention_task import PredictionTask
from llm_intervention_world import LinearMechanism, ResponseAdapter, WorldDescriptor

PROCESS = '''import json,sys
v=json.load(sys.stdin)
assert set(v)=={'version','task','state'}
assert 'mechanisms' not in v['state'] and 'private_evaluation' not in v
mode=sys.argv[1]
assert int(sys.argv[2]) in (51,52)
print('broken' if mode=='broken' else '{"stop":true}' if mode=='stop' else '{"action_id":"observe"}')
'''
SERVICE = '''import sys
from pathlib import Path
sys.path.insert(0, SOURCE)
from llm_intervention_persistent_policy import serve_policy
count=0
def initialize():
    with Path('initializations.txt').open('x') as f: f.write('once')
    return decide
def decide(envelope):
    global count
    assert set(envelope)=={'version','state','task'}
    assert 'mechanisms' not in envelope['state'] and 'private_evaluation' not in envelope
    count+=1
    if MODE=='fail' and count==2: raise RuntimeError('deliberate second-call transport failure')
    return '{"stop":true}' if MODE=='stop' else '{"action_id":"observe"}'
serve_policy(initialize,identity=IDENTITY,max_requests=64)
'''


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ACE_CAMPAIGN_TEST_OUTPUT'])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.path = self.root/self._testMethodName
        self.path.mkdir()
        self.responses = []
        original = ResponseAdapter.__call__
        def guarded(adapter, action, request):
            self.assertEqual(adapter._world.scope, 'artificial')
            self.assertIn(adapter._world.world_id, (51, 52))
            value = original(adapter, action, request)
            self.responses.append({'world_id': adapter._world.world_id, 'stream': adapter._stream, 'request_id': request, 'values': value})
            return value
        self.guard = patch.object(ResponseAdapter, '__call__', guarded)
        self.guard.start()
        self.addCleanup(self.guard.stop)
        draw = patch('llm_intervention_world.generate_world', side_effect=AssertionError('no generated worlds'))
        draw.start()
        self.addCleanup(draw.stop)

    def tearDown(self):
        paired._save(self.path/'artificial-response-record.json', {'responses': self.responses, 'count': len(self.responses),
            'scientific_worlds': 0, 'pretrained_models': 0, 'training_updates': 0, 'gpu_allocations': 0})

    def make(self, ids=('observe', 'stop', 'broken', 'missing', 'persistent'), mode='observe', scopes=('artificial','artificial')):
        worker = self.path/'process.py'
        if not worker.exists(): worker.write_text(PROCESS)
        identity = {'model_sha256': digest('no-model'), 'source_sha256': digest('tiny-fixture'),
            'adapter_sha256': digest('no-adapter'), 'numeric_recipe': {'artificial': True},
            'numeric_recipe_sha256': digest({'artificial': True})}
        service = self.path/('service-'+mode+'.py')
        if not service.exists():
            script = SERVICE.replace('SOURCE', repr(str(Path(campaign.__file__).parent))).replace('IDENTITY', repr(identity)).replace('MODE', repr(mode))
            if mode == 'startup_failure': script = 'raise RuntimeError("deliberate startup failure")\n'
            service.write_text(script)
        graph = (Node('X', (), 'unit', (-10.,10.)),)
        action = Action('observe', 'observe', (), ('X',), 1)
        sources = []
        for i, wid in enumerate((51,52)):
            world = WorldDescriptor(scopes[i], wid, graph, (LinearMechanism('X', float(3+3*i), (), 0.),))
            task = PredictionTask('X', (Action('q','observe',(),('X',),1),), (1.,), 3., .12 if i==0 else .2)
            specs = []
            for pid in ids:
                spec = {'policy_id': pid, 'kind': 'task_process', 'command': [sys.executable,'-B',str(worker),pid,str(wid)],
                    'timeout_seconds': 5., 'identity_sha256': None, 'policy_provenance_sha256': digest(['tiny',pid])}
                if pid == 'persistent':
                    spec.update(kind='persistent', command=[sys.executable,'-B',str(service)], identity_sha256=digest(identity))
                specs.append(spec)
            p = paired.prepare_plan(world=world,world_sha256=world.sha256,task=task,menu=(action,),noise_variances={'X':1.},
                prior_variance=1.,initial_actions=('observe',),acquisition_cap_units=2+i,
                initial_binding={'request_namespace':'paired-initial','replicate_id':'initial'},
                measurement_bindings={pid:{'request_namespace':pid,'replicate_id':'measurement'} for pid in ids},
                policy_specs=tuple(specs),objective='Hand-written tiny artificial campaign',domain_notes='No scientific admission.',
                outer_provenance_receipt_sha256=digest('artificial-provenance'))
            sources.append({'plan':p,'plan_sha256':digest(p),'strata':{'cost':'low' if i==0 else 'high','family':'single'}})
        services = {'persistent':{'identity':identity,'startup_seconds':10.,'cleanup_seconds':1.,'max_requests':5}} if 'persistent' in ids else {}
        return {'sources':sources,'strata_fields':['cost','family'],
            'policy_phases':{pid:'gpu' if pid=='persistent' else 'cpu' for pid in ids},'services':services}

    def setup_campaign(self, args=None):
        args = self.make() if args is None else args
        c = campaign.prepare_campaign(**args)
        self.binding = {'campaign':c,'campaign_sha256':digest(c),'source_plan_pins':campaign._source_pins(c['sources']), 'output':self.path/'campaign'}
        campaign.initialize_campaign(**self.binding)
        self.pins = {}
        return c

    def run_policy(self, pid, **kwargs):
        result = campaign.run_policy(**self.binding, policy_id=pid, phase=self.binding['campaign']['policy_phases'][pid], **kwargs)
        self.pins[pid] = result['sha256']
        return json.loads((campaign._attempt(self.binding['output'],self.binding['campaign'],pid)/'attempt.json').read_bytes())

    def finish(self):
        return campaign.finalize_campaign(**self.binding,attempt_pins=self.pins)

    def assert_no_private(self):
        self.assertFalse((self.binding['output']/'private').exists())
        self.assertFalse((self.binding['output']/'finalization-intent.json').exists())

    def test_complete_cross_product_phase_barrier_reuse_and_hand_costs(self):
        c = self.setup_campaign()
        with self.assertRaises(FileExistsError): campaign.initialize_campaign(**self.binding)
        with self.assertRaises(ValueError): campaign.run_policy(**self.binding,policy_id='persistent',phase='cpu')
        with self.assertRaises(FileNotFoundError): campaign.run_policy(**self.binding,policy_id='persistent',phase='gpu')
        with patch('llm_intervention_cost_curves.report_cost_curve',side_effect=AssertionError('no private scores during execution')):
            for pid in ('observe','stop','broken'): self.run_policy(pid)
            self.run_policy('missing',unstarted_reason='deliberately not admitted in this fixture')
            original_close = PersistentPolicy.close
            closed = []
            def close(p):
                closed.append(p.process.pid)
                return original_close(p)
            with patch.object(PersistentPolicy,'close',close): receipt = self.run_policy('persistent')
        self.assertEqual(len(closed),1)
        self.assertEqual([a['status'] for a in receipt['arms']],['completed','completed'])
        with self.assertRaises(FileExistsError): self.run_policy('persistent')
        service = self.binding['output']/'policy-04/service'
        done = json.loads((service/'closed.json').read_bytes())
        self.assertEqual(done['completed_requests'],5)
        self.assertEqual(done['last_request_id'],4)
        bindings = [json.loads(p.read_bytes()) for p in sorted(service.glob('task-binding-*.json'))]
        self.assertEqual(len(bindings),2)
        self.assertEqual({b['worker_pid'] for b in bindings},set(closed))
        self.assertNotEqual(bindings[0]['task_sha256'],bindings[1]['task_sha256'])
        summary = self.finish()
        self.assertEqual(summary['cell_count'],10)
        self.assertEqual(len(summary['policy_pairs']),10)
        self.assertEqual(summary['roster']['strata_fields'],['cost','family'])
        self.assertEqual([(s['field'],s['value'],s['world_ids']) for s in summary['strata']],
                         [('cost','low',[51]),('cost','high',[52]),('family','single',[51,52])])
        by = {s['policy_id']:s for s in summary['policy_summaries']}
        self.assertEqual(by['observe']['capped_acquisition_cost_vector'],[1,3])
        self.assertEqual(by['persistent']['capped_acquisition_cost_vector'],[1,3])
        self.assertEqual(by['stop']['capped_acquisition_cost_vector'],[2,3])
        self.assertEqual(by['broken']['failed'],2)
        self.assertEqual(by['missing']['unresolved'],2)
        self.assertEqual(by['missing']['capped_total_cost_vector'],[3,4])
        self.assertIsNone(by['missing']['all_world_mean_final_error'])
        self.assertAlmostEqual(by['observe']['final_error_vector'][0],1/16)
        self.assertAlmostEqual(by['observe']['final_error_vector'][1],4/25)
        self.assertEqual(len(self.responses),18)
        with self.assertRaises(FileExistsError): self.finish()
        self.assertFalse(summary['scientific_acceptance'])

    def test_exact_pins_scope_ids_roster_and_service_spec_rejections(self):
        args = self.make()
        good = campaign.prepare_campaign(**args)
        cases = []
        def change(label, mutate, repin=True):
            bad = copy.deepcopy(args); mutate(bad)
            if repin:
                for source in bad['sources']: source['plan_sha256']=digest(source['plan'])
            cases.append((label,bad))
        change('duplicate-world',lambda a:a['sources'].__setitem__(1,copy.deepcopy(a['sources'][0])))
        change('policy-order',lambda a:a['sources'][1]['plan']['policy_specs'].reverse())
        for key,value in [('command',['different-command']),('timeout_seconds',7.),('policy_provenance_sha256','b'*64),('identity_sha256','c'*64)]:
            change('persistent-'+key,lambda a,k=key,v=value:a['sources'][1]['plan']['policy_specs'][-1].__setitem__(k,v))
        change('task-timeout',lambda a:a['sources'][1]['plan']['policy_specs'][0].__setitem__('timeout_seconds',6.))
        change('plan-extra',lambda a:a['sources'][0]['plan'].__setitem__('extra',True))
        change('source-pin',lambda a:a['sources'][0].__setitem__('plan_sha256','0'*64),False)
        change('strata',lambda a:a['sources'][1]['strata'].__setitem__('outcome','selected'))
        change('capacity',lambda a:a['services']['persistent'].__setitem__('max_requests',4))
        change('identity',lambda a:a['services']['persistent']['identity'].__setitem__('model_sha256','d'*64))
        change('task-gpu',lambda a:a['policy_phases'].__setitem__('observe','gpu'))
        cases.append(('mixed-scope',self.make(scopes=('artificial','development'))))
        with patch.object(paired,'ResponseAdapter',side_effect=AssertionError('validation must not open environment')):
            for label,bad in cases:
                paired._save(self.path/(label+'.json'),bad)
                with self.subTest(label=label), self.assertRaises(ValueError): campaign.prepare_campaign(**bad)
            for label,pin,source_pins in [('campaign-pin','0'*64,campaign._source_pins(good['sources'])),
                                         ('inventory-missing',digest(good),[]),
                                         ('inventory-order',digest(good),list(reversed(campaign._source_pins(good['sources']))))]:
                with self.subTest(label=label), self.assertRaises(ValueError):
                    campaign.validate_campaign(campaign=good,campaign_sha256=pin,source_plan_pins=source_pins)
        self.assertEqual(self.responses,[])

    def test_startup_failure_keeps_entire_roster_unstarted(self):
        self.setup_campaign(self.make(ids=('persistent',),mode='startup_failure'))
        r = self.run_policy('persistent')
        self.assertEqual([a['status'] for a in r['arms']],['unstarted','unstarted'])
        self.assertTrue(r['errors'])
        summary = self.finish()
        self.assertEqual([r['cell_status'] for r in summary['rows']],['unresolved','unresolved'])
        self.assertEqual([r['capped_acquisition_cost_units'] for r in summary['rows']],[2,3])
        self.assertEqual(self.responses,[])

    def test_transport_failure_never_restarts_for_later_world(self):
        self.setup_campaign(self.make(ids=('persistent',),mode='fail'))
        r = self.run_policy('persistent')
        self.assertEqual([a['status'] for a in r['arms']],['failed','unstarted'])
        self.assertIn('no restart',r['arms'][1]['reason'])
        service = self.binding['output']/'policy-00/service'
        self.assertEqual((service/'initializations.txt').read_text(),'once')
        self.assertFalse((service/'closed.json').exists())
        failure = json.loads((service/'failure.json').read_bytes())
        self.assertFalse(failure['cleanup']['group_still_exists'])
        self.assertEqual(failure['completed_requests'],1)
        self.assertEqual(len(self.responses),2)
        summary = self.finish()
        self.assertTrue(all(r['cell_status']=='unresolved' for r in summary['rows']))
        self.assertEqual(summary['policy_summaries'][0]['capped_acquisition_cost_vector'],[2,3])

    def test_interrupt_preserves_partial_arm_and_closes_once(self):
        self.setup_campaign(self.make(ids=('persistent',)))
        original = ResponseAdapter.__call__
        def interrupt(adapter,action,request):
            if adapter._stream=='measurement': raise KeyboardInterrupt('new artificial interrupt')
            return original(adapter,action,request)
        original_close = PersistentPolicy.close
        calls = []
        def close(p):
            calls.append(p.process.pid)
            return original_close(p)
        with patch.object(ResponseAdapter,'__call__',interrupt), patch.object(PersistentPolicy,'close',close):
            with self.assertRaises(KeyboardInterrupt): self.run_policy('persistent')
        path = self.binding['output']/'policy-00'
        raw = (path/'attempt.json').read_bytes()
        self.pins['persistent'] = paired._sha(raw)
        r = json.loads(raw)
        self.assertEqual([a['status'] for a in r['arms']],['failed','unstarted'])
        self.assertEqual(len(calls),1)
        self.assertIsNotNone(r['arms'][0]['receipt_sha256'])
        summary = self.finish()
        self.assertTrue(summary['rows'][0]['failed'])
        self.assertEqual(summary['rows'][1]['cell_status'],'unresolved')
        self.assertEqual(len(self.responses),1)

    def test_complete_inventory_custody_and_all_closure_gates_precede_reporting(self):
        self.setup_campaign(self.make(ids=('missing','persistent'),mode='stop'))
        self.run_policy('missing',unstarted_reason='explicitly absent CPU arm')
        with patch.object(paired,'finalize_paired_world',side_effect=AssertionError('must gate before any report')):
            with self.assertRaises(ValueError): self.finish()
        self.assert_no_private()
        self.run_policy('persistent')
        root = self.binding['output']
        mutations = ['missing-arm','wrong-arm-id','receipt-pin','live-group','missing-closure','wrong-pid','launch-command','extra-evidence']
        for label in mutations:
            target = self.path/label
            shutil.copytree(root,target)
            path = target/'policy-01'
            r = json.loads((path/'attempt.json').read_bytes())
            if label=='missing-arm': r['arms'].pop()
            if label=='wrong-arm-id': r['arms'][0]['world_id']=52
            if label=='receipt-pin': r['arms'][0]['receipt_sha256']='0'*64
            if label in ('live-group','wrong-pid'):
                file=path/'service/closed.json'; closed=json.loads(file.read_bytes())
                closed['cleanup']['group_still_exists' if label=='live-group' else 'pgid'] = True if label=='live-group' else 999999
                file.write_bytes(paired._raw(closed)); r['service']['sha256']=paired._sha(file.read_bytes())
            if label=='missing-closure': r['service']={'status':'unconfirmed','file':None,'sha256':None}
            if label=='launch-command':
                file=path/'service/launch.json'; launch=json.loads(file.read_bytes());launch['command']=['wrong'];file.write_bytes(paired._raw(launch))
            if label!='extra-evidence': r['evidence_sha256']=campaign._inventory(path)
            else: (path/'unexpected').write_text('unadmitted')
            # Mutation copies are new negative fixtures; original evidence is intact.
            raw=paired._raw(r)
            (path/'attempt.json').write_bytes(raw);(path/'attempt.candidate.json').write_bytes(raw)
            pins=dict(self.pins,persistent=paired._sha(raw))
            binding=dict(self.binding,output=target)
            with self.subTest(label=label), patch.object(paired,'finalize_paired_world',side_effect=AssertionError('private gate bypass')):
                with self.assertRaises(ValueError): campaign.finalize_campaign(**binding,attempt_pins=pins)
            self.assertFalse((target/'private').exists())
        with patch.object(paired,'finalize_paired_world',side_effect=AssertionError('private gate bypass')):
            with self.assertRaises(ValueError): campaign.finalize_campaign(**self.binding,attempt_pins=dict(self.pins,persistent='0'*64))
        self.assert_no_private()
        result=self.finish()
        self.assertEqual(result['cell_count'],4)
        self.assertEqual(len(self.responses),2)

    def test_outer_arm_failure_continues_roster_without_retry(self):
        self.setup_campaign(self.make(ids=('observe',)))
        original=paired.run_world_arm
        calls=[]
        def fail_first(**kwargs):
            wid=kwargs['plan']['world']['world_id'];calls.append(wid)
            if wid==51: raise OSError('new artificial before-arm failure')
            return original(**kwargs)
        with patch.object(paired,'run_world_arm',fail_first): r=self.run_policy('observe')
        self.assertEqual(calls,[51,52])
        self.assertEqual([a['status'] for a in r['arms']],['failed','completed'])
        result=self.finish()
        self.assertEqual([row['cell_status'] for row in result['rows']],['unresolved','reported'])
        self.assertEqual(result['policy_summaries'][0]['capped_acquisition_cost_vector'],[2,3])
        self.assertEqual(len(self.responses),4)

    def test_journal_failure_still_closes_owned_service(self):
        self.setup_campaign(self.make(ids=('persistent',)))
        original=paired._raw
        def fail_event(value):
            if type(value) is dict and value.get('event')=='arm_start_intent': raise OSError('artificial journal failure')
            return original(value)
        with patch.object(paired,'_raw',fail_event): r=self.run_policy('persistent')
        self.assertEqual([a['status'] for a in r['arms']],['failed','unstarted'])
        closed=json.loads((self.binding['output']/'policy-00/service/closed.json').read_bytes())
        self.assertFalse(closed['cleanup']['group_still_exists'])
        self.assertEqual(closed['completed_requests'],0)
        self.assertTrue(r['errors'])
        self.assertEqual(self.responses,[])
        result=self.finish()
        self.assertTrue(all(row['cell_status']=='unresolved' for row in result['rows']))


if __name__=='__main__': unittest.main(verbosity=2)
