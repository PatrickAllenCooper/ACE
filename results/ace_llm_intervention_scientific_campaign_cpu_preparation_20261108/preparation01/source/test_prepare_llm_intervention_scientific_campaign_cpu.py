"""New fabricated orchestration controls; no real ranker, world or policy call."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import prepare_llm_intervention_scientific_campaign_cpu as W

REPO=Path(__file__).resolve().parents[2]
PROPOSAL=REPO/'results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
def raw(v):return (W.canonical(v)+'\n').encode()
def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(value if type(value) is bytes else raw(value))

def fixture(root):
    proposal=json.loads(PROPOSAL.read_bytes());d=7;exposure={'counts':{'dpo_steps':d},'record_pins':{},'supervision_receipt_pins':{}}
    records=[]
    for rid in W.TRAIN_ROSTER:
        text=W.canonical({'fixture_only':True,'record_id':rid})+'\n';rh=W.sha(text.encode());receipt=W.digest(['receipt',rid])
        records.append(dict(record_id=rid,raw_record_utf8=text));exposure['record_pins'][rid]=rh;exposure['supervision_receipt_pins'][rid]=receipt
    plan={'records':records,'outer_provenance_receipt_sha256':'1'*64}
    prepared={'training_plan':plan,'exposure':exposure}
    numerical={'scope':'policy_train','record_roster':list(W.TRAIN_ROSTER),'exposure':exposure,'exposure_sha256':W.digest(exposure),
        'checkpoint_steps':[159,166],'record_pins':exposure['record_pins'],'supervision_receipt_pins':exposure['supervision_receipt_pins'],
        'raw_records_utf8':{r['record_id']:r['raw_record_utf8'] for r in records},'outer_provenance_receipt_sha256':'1'*64,
        'scientific_provenance_qualified':False,'retry_allowed':False,'recipe':{'fabricated_fixed_recipe':True}}
    data={'proposal.json':PROPOSAL.read_bytes(),'scientific-freeze.json':b'fabricated combined freeze before draws',
        'prepared.json':raw(prepared),'numerical-plan.json':raw(numerical),'gpu-training-acceptance.json':b'fabricated independent GPU audit'}
    for n in ('engine-terminal.json','sft.checkpoint.json','final.checkpoint.json'):data[n]=raw({'fixture_only':True})
    for n in ('sft.adapter.pt','final.adapter.pt'):data[n]=b'opaque fabricated adapter '+n.encode()
    entries=[];world_pins={}
    for i,row in enumerate(proposal['training_worlds']+proposal['development_worlds']):
        payload={'scope':row['scope'],'world_id':row['world_id'],'fixture_only':True}
        e=dict(scope=row['scope'],world_id=row['world_id'],path=f'world-{i:03d}.json',status='complete',
            descriptor_sha256=W.digest(payload),file_sha256=W.sha(raw(payload)));entries.append(e)
        if i>=32:data[f'descriptor-{i-32:03d}.json']=raw(payload);world_pins[str(row['world_id'])]=W.digest(payload)
    freeze=W.sha(data['scientific-freeze.json']);namespace='2'*64
    terminal=dict(status='complete',worlds_generated=56,environment_responses=0,policy_updates=0,retry_allowed=False,
        proposal_sha256=W.PROPOSAL_SHA,source_pins=proposal['source_pins'],freeze_sha256=freeze,namespace_admission_sha256=namespace,entries=entries)
    data['realization-terminal.json']=raw(terminal)
    data['realization-acceptance.json']=raw(dict(status='accepted',required_findings=0,complete_custody_verified=True,
        realization_terminal_sha256=W.sha(data['realization-terminal.json']),freeze_sha256=freeze,namespace_admission_sha256=namespace))
    expected=dict(scientific_freeze_sha256=freeze,model_files_sha256=W.digest(W.B.MODEL_FILES),runtime_contract_sha256=W.digest({'fixture':True}),
        prepared_sha256=W.sha(data['prepared.json']),training_plan_sha256=W.digest(plan),exposure_sha256=W.digest(exposure))
    cpu=dict(schema='ace-scientific-training-preparation-acceptance-v1',status='accepted',scope='policy_train',required_findings=0,
        job_id='123456',cpu_packet_sha256='1'*64,prepared_sha256=expected['prepared_sha256'],plan_sha256=expected['training_plan_sha256'],
        freeze_sha256=freeze,namespace_admission_sha256=namespace,complete_custody_verified=True,
        runtime_qualified_for_this_cpu_execution=True,owned_process_cleanup_verified=True,training_execution_qualified=False,scientific_qualified=False,
        numerical_plan_file_sha256=W.sha(data['numerical-plan.json']),numerical_plan_sha256=W.digest(numerical),exposure_sha256=W.digest(exposure))
    data['cpu-preparation-acceptance.json']=raw(cpu);expected['cpu_preparation_acceptance_sha256']=W.sha(data['cpu-preparation-acceptance.json'])
    bindings=[dict(world_id=r['world_id'],policy_id=pid,binding=dict(world_id=r['world_id'],arm=pid,fixture_only=True),
        binding_sha256=W.digest(dict(world_id=r['world_id'],arm=pid,fixture_only=True))) for r in proposal['development_worlds'] for pid in W.NUMERIC_POLICIES]
    data['numeric-bindings.json']=raw(bindings)
    p=dict(schema=W.VERSION,scope='development',cpu_output=str(root/'cpu-output'),world_pins=world_pins,
        files={**{n:'a'*64 for n in W.SOURCE_NAMES},**{n:W.sha(v) for n,v in data.items()}},expected_training=expected,
        numerical_plan_sha256=W.digest(numerical),model_files=W.B.MODEL_FILES,runtime={'fixture':True},freeze_sha256=freeze,
        namespace_admission_sha256=namespace,protocol_source_pins=proposal['source_pins'],outer_provenance_receipt_sha256='3'*64,
        resources=dict(W.BASE_RESOURCES,wall_seconds=6000),deadlines=dict(startup=120,work=5880,hard=5990),
        numeric_timeout_seconds=30,numeric_work_seconds=28,source_commit='a'*40,persistent_specs={},services={},service_identity_pins={})
    for pid in W.LLM_POLICIES:
        identity={'fixture_service':pid};p['services'][pid]=dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=576)
        p['service_identity_pins'][pid]=W.digest(identity)
        p['persistent_specs'][pid]=dict(policy_id=pid,kind='persistent',command=['never-start',pid],timeout_seconds=125,
            identity_sha256=W.digest(identity),policy_provenance_sha256='b'*64)
    lineage=dict(exposure_sha256=W.digest(exposure),roles={})
    prefixes={wid:{kind:['captured-python',kind,'--world-id',wid,'--'] for kind in ('screen','ranker')} for wid in world_pins}
    ctx=dict(p=p,data=data,lineage=lineage);loc=W.paths(p);loc['artifacts'].mkdir(parents=True)
    return ctx,loc,prefixes,bindings


class Doubles:
    def __init__(self,test,ctx,loc,bindings,fail_policy=None,fit_fail=False):
        self.test,self.ctx,self.loc,self.bindings=test,ctx,loc,bindings;self.calls=[];self.fail_policy=fail_policy;self.fit_fail=fit_fail
    def numeric_recipe(self,**kw):
        self.test.assertEqual(kw,{'artificial':False});return {'fabricated_fixed_recipe':True}
    def numeric_policy_bindings(self,**kw):self.calls.append('bindings');return self.bindings
    def from_private_payload(self,payload,*,expected_sha256):
        self.test.assertEqual(W.digest(payload),expected_sha256);self.calls.append(('descriptor',payload['world_id']));return payload
    def fit(self,*,plan,plan_sha256,output):
        self.calls.append('fit');self.test.assertEqual(plan_sha256,self.ctx['p']['numerical_plan_sha256'])
        if self.fit_fail:raise RuntimeError('fabricated fit failure')
        self.test.assertEqual(plan,json.loads(self.ctx['data']['numerical-plan.json']))
        output.mkdir();training=raw({'fixture_only':True});put(output/'training_complete.json',training)
        checkpoint=b'opaque fabricated final ranker';put(output/'dpo.ranker.pt',checkpoint)
        result=dict(status='complete',sft_steps=160,dpo_steps=7,completed_steps=167,plan_sha256=plan_sha256,
            exposure_sha256=self.ctx['lineage']['exposure_sha256'],training_receipt_sha256=W.sha(training),
            exports={'dpo':dict(path='dpo.ranker.pt',sha256=W.sha(checkpoint),training_receipt_sha256=W.sha(training),all_training_score_rows_equal=True)})
        put(output/'terminal.json',result);return result
    def build_scientific_campaign(self,**kw):
        self.calls.append('build');self.test.assertEqual(len(kw['worlds_by_id']),24)
        specs=kw['policy_specs_by_world'];self.test.assertEqual(sum(map(len,specs.values())),192)
        self.test.assertTrue(all(v['max_requests']==576 for v in kw['services'].values()))
        self.test.assertEqual([s['policy_id'] for s in next(iter(specs.values()))],list(W.CPU_POLICIES+W.LLM_POLICIES))
        c=dict(sources=[{'plan':{'world':w}} for w in kw['worlds_by_id'].values()],policy_ids=list(W.CPU_POLICIES+W.LLM_POLICIES))
        return dict(campaign=c,campaign_sha256=W.digest(c),source_plan_pins=[])
    def initialize_campaign(self,**kw):
        self.calls.append('initialize');self.test.assertTrue((self.loc['phase']/'campaign-bundle.json').exists())
        kw['output'].mkdir()
    def run_policy(self,**kw):
        pid=kw['policy_id'];self.calls.append(pid);self.test.assertEqual(kw['phase'],'cpu');self.test.assertIn(pid,W.CPU_POLICIES)
        return dict(policy_id=pid,sha256=W.digest(pid))
    def _attempt(self,*args):return args[-1]
    def _load_attempt(self,c,h,path,pid,pin):
        return dict(errors=['fabricated failure'] if pid==self.fail_policy else [],arms=[dict(status='completed')]*24),None
    def event(self,name,**fields):self.calls.append(('event',name))
    def run(self):return W.run_phase(self.ctx,self.loc,self.prefixes,self.event,builder=self,ranker=self,campaign=self,world_type=self)


class ScientificCampaignGlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ['ACE_SCI_CAMPAIGN_CPU_TEST_OUTPUT']);cls.output.mkdir(exist_ok=False)
    def setUp(self):
        self.root=self.output/self._testMethodName;self.root.mkdir()
        self.ctx,self.loc,self.prefixes,self.bindings=fixture(self.root)
    def doubles(self,**kw):
        d=Doubles(self,self.ctx,self.loc,self.bindings,**kw);d.prefixes=self.prefixes;return d
    def acceptance(self):
        ctx=self.ctx
        class Lineage:
            def admit_training(inner,**kw):
                self.assertEqual(kw['expected'],ctx['p']['expected_training'])
                self.assertEqual(kw['acceptance_sha256'],ctx['p']['files']['gpu-training-acceptance.json'])
                return ctx['lineage']
        return W.accepted_inputs(ctx['p'],ctx['data'],Lineage())
    def rebind_cpu(self,value):
        data=self.ctx['data'];p=self.ctx['p'];data['cpu-preparation-acceptance.json']=raw(value)
        p['files']['cpu-preparation-acceptance.json']=W.sha(data['cpu-preparation-acceptance.json'])
        p['expected_training']['cpu_preparation_acceptance_sha256']=p['files']['cpu-preparation-acceptance.json']

    def test_exact_cpu_acceptance_schema_and_numerical_lineage(self):
        self.assertEqual(self.acceptance(),self.ctx['lineage'])
        original=copy.deepcopy(self.ctx)
        changes={'schema':'other','scope':'artificial','runtime_qualified_for_this_cpu_execution':False,
            'freeze_sha256':'0'*64,'namespace_admission_sha256':'0'*64,'numerical_plan_file_sha256':'0'*64,
            'numerical_plan_sha256':'0'*64,'exposure_sha256':'0'*64,'plan_sha256':'0'*64,'cpu_packet_sha256':'0'*64,
            'training_execution_qualified':True,'scientific_qualified':True,'required_findings':False,'job_id':'0'}
        for key,value in changes.items():
            self.ctx=copy.deepcopy(original);cpu=json.loads(self.ctx['data']['cpu-preparation-acceptance.json']);cpu[key]=value;self.rebind_cpu(cpu)
            put(self.root/(key+'.json'),cpu)
            with self.subTest(key=key),self.assertRaises(ValueError):self.acceptance()

    def test_complete_24_descriptors_original_byte_semantic_membership(self):
        self.acceptance();original=copy.deepcopy(self.ctx)
        for i,change in enumerate((lambda c:c['p']['world_pins'].pop(next(iter(c['p']['world_pins']))),
            lambda c:c['data'].__setitem__('descriptor-000.json',raw({'scope':'development','world_id':1})),
            lambda c:c['p'].update(namespace_admission_sha256='0'*64))):
            self.ctx=copy.deepcopy(original);change(self.ctx)
            with self.subTest(i=i),self.assertRaises(ValueError):self.acceptance()

    def test_once_fit_then_complete_contract_then_only_five_cpu_arms(self):
        d=self.doubles();result=d.run();calls=d.calls
        self.assertEqual(calls.count('fit'),1);self.assertEqual(len([c for c in calls if type(c)is tuple and c[0]=='descriptor']),24)
        self.assertLess(calls.index('fit'),calls.index('build'));self.assertLess(calls.index('build'),calls.index('initialize'))
        self.assertEqual([c for c in calls if c in W.CPU_POLICIES],list(W.CPU_POLICIES))
        self.assertEqual((result['numeric_training_updates'],result['completed_cpu_cells'],result['declared_cells']),(167,120,192))
        self.assertEqual(result['gpu_services_started'],0);self.assertFalse(result['private_scores_computed'])
        with self.assertRaises(FileExistsError):d.run()
        self.assertEqual(calls.count('fit'),1)

    def test_fixed_bindings_fail_before_fit_and_fit_failure_before_responses(self):
        self.ctx['data']['numeric-bindings.json']=raw(self.bindings[:-1]);d=self.doubles()
        with self.assertRaisesRegex(ValueError,'96'):d.run()
        self.assertNotIn('fit',d.calls);self.assertTrue((self.loc['phase']/'failure.json').exists())
        other=self.root/'fit-failure';other.mkdir();ctx,loc,prefixes,bindings=fixture(other)
        d=Doubles(self,ctx,loc,bindings,fit_fail=True);d.prefixes=prefixes
        with self.assertRaisesRegex(RuntimeError,'fabricated'):d.run()
        self.assertNotIn('build',d.calls);self.assertNotIn('initialize',d.calls);self.assertTrue((loc['phase']/'failure.json').exists())

    def test_failed_cpu_policy_retained_and_no_remaining_policy_or_gpu(self):
        d=self.doubles(fail_policy='menu_coverage')
        with self.assertRaisesRegex(ValueError,'failed/partial'):d.run()
        self.assertEqual([c for c in d.calls if c in W.CPU_POLICIES],['uniform_random','menu_coverage'])
        self.assertTrue((self.loc['phase']/'menu_coverage-attempt.json').exists())
        self.assertTrue((self.loc['phase']/'campaign-bundle.json').exists());self.assertTrue((self.loc['phase']/'failure.json').exists())
        self.assertFalse((self.loc['phase']/'terminal.json').exists())

    def test_numeric_specs_exact_96_suffixes_and_final_export(self):
        p=self.ctx['p'];export=dict(path='/fabricated/final/dpo.ranker.pt',sha256='a'*64,training_receipt_sha256='b'*64)
        specs=W.policy_specs(p,self.bindings,self.prefixes,export,'c'*64)
        self.assertEqual(len(specs),24)
        for wid,rows in specs.items():
            self.assertEqual(len(rows),8)
            for r in rows[:4]:
                W.numeric_tail('screen',r['command'][-4:],p,self.ctx['data'],wid)
            self.assertIn(export['path'],rows[4]['command']);self.assertEqual(rows[4]['policy_provenance_sha256'],'c'*64)
            self.assertTrue(all(r['kind']=='persistent' for r in rows[5:]))
        first=self.bindings[0];args=['--binding-json',W.canonical(first['binding']),'--binding-sha256',first['binding_sha256']]
        with self.assertRaises(ValueError):W.numeric_tail('screen',args,p,self.ctx['data'],list(specs)[1])
        self.assertTrue(all(s['max_requests']==576 for s in p['services'].values()))

    def test_saved_all_192_cell_design_namespace_and_commands(self):
        p,data=self.ctx['p'],self.ctx['data'];out=self.loc['phase'];out.mkdir()
        ranker={'exports':{'dpo':dict(path='dpo.ranker.pt',sha256='a'*64,training_receipt_sha256='b'*64)}}
        put(out/'ranker/terminal.json',ranker)
        export=dict(path=str(out/'ranker/dpo.ranker.pt'),sha256='a'*64,training_receipt_sha256='b'*64)
        specs=W.policy_specs(p,self.bindings,self.prefixes,export,W.sha(raw(ranker)))
        proposal=json.loads(data['proposal.json']);sources=[]
        binding=lambda value:{k:value[k] for k in ('request_namespace','replicate_id')}
        for i,r in enumerate(proposal['development_worlds']):
            public=r['public'];wid=r['world_id']
            plan=dict(world=json.loads(data[f'descriptor-{i:03d}.json']),world_sha256=p['world_pins'][str(wid)],
                policy_specs=specs[wid],initial_actions=public['initial_actions'],initial_cost_units=public['initial_cost_units'],
                acquisition_cap_units=24,task=public['task'],menu=public['menu'],learner=dict(graph=public['graph'],
                    noise_variances=public['learner']['noise_variances'],prior_variance=public['learner']['prior_variance']),
                initial_binding=binding(r['initial_binding']),measurement_bindings={pid:binding(e['measurement_binding'])
                    for pid,e in zip(W.CPU_POLICIES+W.LLM_POLICIES,r['development_episodes'])},objective=public['objective'],
                domain_notes=public['domain_notes'],outer_provenance_receipt_sha256=p['outer_provenance_receipt_sha256'],
                scientific_provenance_qualified=False,retry_allowed=False)
            sources.append(dict(plan=plan,plan_sha256=W.digest(plan),strata={k:r[k] for k in ('regime','graph_stratum','cost_stratum')}))
        campaign=dict(scope='development',sources=sources,policy_ids=list(W.CPU_POLICIES+W.LLM_POLICIES),
            policy_phases={pid:'cpu' if pid in W.CPU_POLICIES else 'gpu' for pid in W.CPU_POLICIES+W.LLM_POLICIES},services=p['services'])
        bundle=dict(scope='development',campaign=campaign,campaign_sha256=W.digest(campaign),proposal_sha256=W.PROPOSAL_SHA,
            numeric_policy_bindings=self.bindings,service_identity_pins=p['service_identity_pins'],source_pins=p['protocol_source_pins'],
            world_pins=[dict(world_id=r['world_id'],sha256=p['world_pins'][str(r['world_id'])]) for r in proposal['development_worlds']],
            worlds_generated=0,responses_generated=0,execution_authorized=False,scientific_protocol_frozen=False,
            scientific_provenance_qualified=False,source_qualified=False,runtime_qualified=False,scheduler_qualified=False,
            proposal_arm_mapping=dict(zip(W.CPU_POLICIES+W.LLM_POLICIES,list(W.NUMERIC_POLICIES)+['numerical_ranker','llm_frozen','llm_sft','llm_dpo'])),
            source_plan_pins=[dict(world_id=s['plan']['world']['world_id'],sha256=s['plan_sha256']) for s in sources])
        path=out/'campaign-bundle.json';put(path,bundle)
        self.assertEqual(W.validate_bundle(self.ctx,self.loc,self.prefixes,ranker),bundle)
        changes=[lambda v:v['campaign']['sources'][0]['plan']['task'].update(scale=999.),
            lambda v:v['campaign']['sources'][0]['plan']['initial_binding'].update(request_namespace='changed'),
            lambda v:v['campaign']['sources'][0]['plan']['policy_specs'][4]['command'].append('extra'),
            lambda v:v['campaign']['sources'][0]['strata'].update(regime='changed'),
            lambda v:v['campaign']['services']['sft_llm'].update(max_requests=577)]
        for i,change in enumerate(changes):
            bad=copy.deepcopy(bundle);change(bad)
            for s in bad['campaign']['sources']:s['plan_sha256']=W.digest(s['plan'])
            bad['source_plan_pins']=[dict(world_id=s['plan']['world']['world_id'],sha256=s['plan_sha256']) for s in bad['campaign']['sources']]
            bad['campaign_sha256']=W.digest(bad['campaign']);put(self.root/f'rejected-design-{i}.json',bad)
            path.write_bytes(raw(bad))
            with self.subTest(i=i),self.assertRaises(ValueError):W.validate_bundle(self.ctx,self.loc,self.prefixes,ranker)
        path.write_bytes(raw(bundle))

    def test_explicit_resources_admission_and_actual_allocation(self):
        p=self.ctx['p'];W.resource_contract(p)
        a=dict(schema=W.VERSION+'/admission',packet_sha256='a'*64,source_commit=p['source_commit'],freeze_sha256=p['freeze_sha256'],
            namespace_admission_sha256=p['namespace_admission_sha256'],stage='matched_ranker_and_development_cpu',output=p['cpu_output'],
            resources=p['resources'],deadlines=p['deadlines'],full_contract_frozen_before_first_scientific_draw=True,
            source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,independent_acceptances_verified=True,retry_allowed=False)
        W.admission_contract(p,a,'a'*64)
        with self.assertRaises(ValueError):W.admission_contract(p,dict(a,full_contract_frozen_before_first_scientific_draw=False),'a'*64)
        job='JobId=123 JobState=RUNNING UserId=paco0228(123) Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 Requeue=0 Restarts=0 ReqTRES=cpu=2,mem=7680M,node=1 AllocTRES=cpu=2,mem=7680M,node=1 TimeLimit=01:40:00'
        with patch.dict(os.environ,{'SLURM_JOB_GPUS':'','SLURM_STEP_GPUS':''}):
            W.allocation(job,'123',p)
            for text in (job.replace('01:40:00','00:15:00'),job.replace('mem=7680M,node=1','mem=7680M,node=1,gres/gpu=0'),job.replace('NumCPUs=2','NumCPUs=4')):
                with self.assertRaises(ValueError):W.allocation(text,'123',p)
        for value in (6001,True):
            bad=copy.deepcopy(p);bad['resources']['wall_seconds']=value
            with self.assertRaises(ValueError):W.resource_contract(bad)

    def test_finalizer_packet_shape_pure_freeze_and_cpu_pin_contract(self):
        p=copy.deepcopy(self.ctx['p'])
        p.update(source_names=sorted(W.SOURCE_NAMES),runtime_source='/fabricated/original-runtime',model_directory='/fabricated/future-model',
            archive=dict(path='/fabricated/runtime.tar',sha256='d'*64),manifest=dict(path='/fabricated/runtime.json',sha256='e'*64))
        p['files']={**{n:'a'*64 for n in W.INPUT_NAMES},**p['files'],**W.HELPER_PINS}
        for name,h in p['protocol_source_pins'].items():p['files'][Path(name).name]=h
        service=types.SimpleNamespace(RUNTIME=p['runtime'])
        diagnostic=types.SimpleNamespace(RUNTIME_SOURCE=p['runtime_source'],ARCHIVE_SHA='d'*64,MANIFEST_SHA='e'*64)
        self.assertEqual(set(p),W.PACKET_KEYS);W.packet_shape(p,service,diagnostic)
        for field in ('scientific-freeze.json','prepared.json','cpu-preparation-acceptance.json'):
            bad=copy.deepcopy(p);bad['files'][field]='0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):W.packet_shape(bad,service,diagnostic)
        bad=copy.deepcopy(p);bad['world_pins'].pop(next(iter(bad['world_pins'])))
        with self.assertRaises(ValueError):W.packet_shape(bad,service,diagnostic)
        self.assertFalse({'torch','numpy','transformers'}.intersection(sys.modules))

    def test_source_wiring_closure_no_generation_or_global_patching(self):
        tree=ast.parse(Path(W.__file__).read_bytes());text=ast.unparse(tree)
        self.assertFalse({'torch','numpy','transformers','llm_intervention_ranker_training','llm_intervention_world'}.intersection(sys.modules))
        self.assertNotIn('AutoTokenizer',text);self.assertNotIn('generate_world(',text);self.assertNotIn('realize_screen_worlds(',text)
        self.assertNotIn('setattr(B,',text);self.assertNotIn('finalize_campaign(',text)
        funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        phase=ast.unparse(funcs['run_phase']);self.assertEqual(phase.count('ranker.fit('),1)
        self.assertIn('world_type.from_private_payload(',phase)
        parent=ast.unparse(funcs['parent']);self.assertIn('ownership.supervise(',parent)
        self.assertLess(parent.index('ownership.supervise('),parent.index('validate_worker('))
        numeric=ast.unparse(funcs['numeric_child']);self.assertNotIn('descriptor-',numeric)
        self.assertNotIn('authenticate(args)',numeric)
        for name in W.SOURCE_NAMES:
            for node in ast.walk(ast.parse((REPO/'scripts/research'/name).read_bytes())):
                deps=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                for dep in deps:
                    if dep.startswith(W.B.PROJECT_PREFIXES):self.assertIn(dep+'.py',W.SOURCE_NAMES)


class ScientificCpuLifecycleTests(unittest.TestCase):
    """Only fabricated process/clock/signal boundaries; never start a worker."""
    @classmethod
    def setUpClass(cls):
        import llm_intervention_cpu_owned_supervision as ownership
        cls.O=ownership
        cls.output=Path(os.environ['ACE_SCI_CAMPAIGN_CPU_LIFECYCLE_OUTPUT']);cls.output.mkdir(exist_ok=False)
    def setUp(self):
        self.out=self.output/self._testMethodName;self.out.mkdir()
        self.now=1.;self.alive=True;self.kills=[];self.timers=[];self.observations=[]
        self.proc=types.SimpleNamespace(pid=1234,poll=lambda:0)
        self.owned=self.O.OwnedChild(30.)
        self.clock=patch.object(self.O.time,'monotonic',side_effect=lambda:self.now);self.clock.start();self.addCleanup(self.clock.stop)
        self.sleep=patch.object(self.O.time,'sleep',side_effect=self.advance);self.sleep.start();self.addCleanup(self.sleep.stop)
        self.group=patch.object(self.O.os,'killpg',side_effect=self.killpg);self.group.start();self.addCleanup(self.group.stop)
        self.handler=patch.object(self.O.signal,'signal',return_value=None);self.handler.start();self.addCleanup(self.handler.stop)
        self.timer=patch.object(self.O.signal,'setitimer',side_effect=lambda kind,seconds:self.timers.append((seconds,self.owned.retired)))
        self.timer.start();self.addCleanup(self.timer.stop)
        self.launch=patch.object(self.O.subprocess,'Popen',return_value=self.proc);self.launch.start();self.addCleanup(self.launch.stop)
        self.no_exit=patch.object(self.O.os,'_exit',side_effect=AssertionError('unexpected actual hard exit'))
        self.no_exit.start();self.addCleanup(self.no_exit.stop)
    def advance(self,seconds):self.now+=seconds
    def killpg(self,pid,sig):
        self.assertEqual(pid,1234);self.observations.append(sig)
        if not self.alive:raise ProcessLookupError('fabricated closed group')
        if sig:
            self.kills.append(sig);self.alive=False
    def marker(self,**changes):
        row=dict(packet_sha256='a'*64,pid=1234,monotonic=.5);row.update(changes)
        put(self.out/'worker/work-start.json',row)
    def supervise(self,cleanup):
        return self.O.supervise(['fabricated-only'],{},self.out,started=0.,startup_deadline=5.,work_deadline=20.,
            packet_sha256='a'*64,owned=self.owned,kill_owned=cleanup)
    def test_launch_publication_cleanup_and_resource_failures_keep_first_and_handle(self):
        real=self.O.save;first=OSError('first launch publication failure')
        def save(path,value):
            if path.name=='worker.started.json':raise first
            if path.name=='worker.resources.json':raise OSError('secondary resources failure')
            return real(path,value)
        def cleanup(proc):
            self.assertIs(proc,self.proc);raise RuntimeError('secondary cleanup failure')
        with patch.object(self.O,'save',side_effect=save):
            with self.assertRaises(OSError) as got:self.supervise(cleanup)
        self.assertIs(got.exception,first);self.assertIs(self.owned.process,self.proc)
        self.assertEqual([r['stage'] for r in self.owned.errors],['work','cleanup','resource_publication'])
        W.finish_owned(self.owned,self.out)
        self.assertTrue(self.owned.retired);self.assertIsNone(self.owned.process)
        self.assertEqual(self.timers[-1],(0,True));self.assertEqual(len(self.kills),1)
        before=list(self.observations);self.owned.kill();self.assertTrue(self.owned.confirm_closed())
        self.assertEqual(self.observations,before)
        receipt=json.loads((self.out/'worker.lifecycle-errors.json').read_bytes())
        self.assertEqual(receipt['errors'],self.owned.errors)
    def test_alive_group_holds_until_absolute_hard_exit(self):
        self.marker()
        cleanup=lambda p:dict(group_still_exists=True,errors=[],returncode=0,kill_sent=False)
        with self.assertRaisesRegex(ValueError,'natural CPU'):self.supervise(cleanup)
        self.assertIs(self.owned.process,self.proc);self.assertFalse(self.owned.retired)
        class HardExit(BaseException):pass
        def alive(pid,sig):
            self.assertEqual(pid,1234);self.observations.append(sig)
        self.now=29.99
        with patch.object(self.O.os,'killpg',side_effect=alive),patch.object(self.O.os,'_exit',side_effect=HardExit):
            with self.assertRaises(HardExit):W.finish_owned(self.owned,self.out)
        self.assertGreaterEqual(self.now,30.)
        self.assertFalse(self.owned.retired);self.assertFalse(any(seconds==0 for seconds,_ in self.timers))
        self.assertIn(self.O.signal.SIGKILL,self.observations)
    def test_parent_startup_marker_required_before_work_deadline(self):
        self.proc.poll=lambda:None
        def advance(seconds):self.now=5.01
        def cleanup(proc):
            proc.poll=lambda:0;self.alive=False
            return dict(group_still_exists=False,errors=[],returncode=0,kill_sent=False)
        with patch.object(self.O.time,'sleep',side_effect=advance):
            with self.assertRaisesRegex(ValueError,'startup/work'):self.supervise(cleanup)
        self.assertEqual(self.timers[0][0],4.)
        self.assertFalse(any(seconds==19. for seconds,_ in self.timers))
        self.assertTrue(self.owned.retired)
        W.finish_owned(self.owned,self.out)
    def test_marker_identity_and_original_startup_time_then_natural_retirement(self):
        for i,changes in enumerate((dict(pid=9),dict(monotonic=5.),dict(monotonic=float('nan')),dict(packet_sha256='b'*64))):
            path=self.out/f'invalid-marker-{i}.json'
            row=dict(packet_sha256='a'*64,pid=1234,monotonic=.5);row.update(changes)
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                self.O.startup_marker(path,packet_sha256='a'*64,pid=1234,started=0.,startup_deadline=5.)
        self.marker();self.alive=False
        def cleanup(proc):return dict(group_still_exists=False,errors=[],returncode=0,kill_sent=False)
        result=self.supervise(cleanup)
        self.assertTrue(result['startup_observed']);self.assertTrue(self.owned.retired)
        self.assertIn((19.,False),self.timers)
        W.finish_owned(self.owned,self.out)
        self.assertEqual(self.kills,[]);self.assertEqual(self.timers[-1],(0,True))
    def test_main_finally_retains_ownership_when_failure_publication_throws(self):
        ctx,loc,_,_=fixture(self.out/'fixture')
        p=ctx['p'];p.update(runtime_source='/fabricated/runtime',model_directory='/fabricated/model',
            archive={'path':'/fabricated/archive'},manifest={'path':'/fabricated/manifest'})
        # fixture created artifacts; use a separate main-owned exclusive output.
        p['cpu_output']=str(self.out/'main-output')
        first=RuntimeError('original parent failure')
        def parent(args,ctx):
            args.owned_child=self.owned;self.owned.launched=True;self.owned.process=self.proc
            raise first
        real=W.save
        def save(path,value):
            if path.name=='failure.json':raise OSError('failure publication denied')
            return real(path,value)
        argv=['fixture','--packet','/fabricated/packet.json','--packet-sha256','a'*64,
            '--source-commit','b'*40,'--admission','/fabricated/admission.json','--admission-sha256','c'*64]
        with patch.object(W.sys,'argv',argv),patch.object(W,'capture',return_value=raw(p)), \
             patch.object(W.B,'separate'),patch.object(W,'authenticate',return_value=ctx), \
             patch.object(W,'parent',side_effect=parent),patch.object(W,'save',side_effect=save):
            with self.assertRaises(RuntimeError) as got:W.main()
        self.assertIs(got.exception,first);self.assertTrue(self.owned.retired)
        self.assertEqual(self.timers[-1],(0,True));self.assertEqual(len(self.kills),1)
        receipt=json.loads((Path(p['cpu_output'])/'worker.lifecycle-errors.json').read_bytes())
        self.assertEqual(receipt['errors'][0]['stage'],'failure_publication')


if __name__=='__main__':unittest.main(verbosity=2)
