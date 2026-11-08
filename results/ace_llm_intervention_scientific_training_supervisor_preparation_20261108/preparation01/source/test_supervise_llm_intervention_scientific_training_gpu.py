"""New stdlib boundary tests only: fabricated CPU admission and staged-runtime doubles.

No real acceptance, runtime, tokenizer, tensor, scientific data or scheduler call.
Fixtures and rejections persist under ACE_SCIENTIFIC_GPU_TEST_OUTPUT. Independent
source pins are changed ONLY in isolated toy modules, never production files.
"""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from contextlib import ExitStack, redirect_stderr

SOURCE=Path(__file__).with_name('supervise_llm_intervention_scientific_training_gpu.py')

def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(raw(v)[:-1])
def load(path):
    m=types.ModuleType('isolated_scientific_supervisor');m.__file__=str(path)
    exec(compile(SOURCE.read_bytes(),str(path),'exec'),vars(m));return m


class Toy:
    def __init__(self,root):
        self.root=root;root.mkdir(parents=True,exist_ok=False)
        self.g=g=load(root/SOURCE.name)
        self.data={n:b'# invented inert source\n' for n in g.SOURCE_NAMES}
        self.data[g.ENTRY]=SOURCE.read_bytes()
        self.data[g.WORKER]=b'''def _plan(prepared,*,prepared_sha256,plan_sha256):
    if prepared['training_plan'].get('toy_gate') != 'valid':raise ValueError('pure worker plan rejected')
    return prepared['training_plan']
'''
        self.data['llm_intervention_runtime_archive.py']=b'# invented archive helper\n'
        self.cpu={'schema':'ace-scientific-training-preparation-cpu-v1','scope':'policy_train',
            'archive':{'path':'/toy/runtime','sha256':'a'*64},'manifest':{'path':'/toy/manifest','sha256':'b'*64},
            'runtime_source':'/toy/original','runtime':{'toy_accepted_runtime':'original'},
            'model_files':{'model.safetensors':'c'*64,'tokenizer.json':'d'*64},
            'model_directory':'/toy/tokenizer-only',
            'source_names':sorted(g.PREPARATION_SOURCES),
            'files':{n:sha(b) for n,b in self.data.items()},
            'collection':{'freeze_sha256':'e'*64,'files':{'terminal.json':'1'*64,'inventory.json':'2'*64,'plan.json':'3'*64}}}
        self.prepared={'training_plan':{'toy_gate':'valid','records':[{'prompt_ids':[1,2]} for _ in range(160)]},
            'tokenizer_sha256':digest(self.cpu['model_files']),'exposure_sha256':'4'*64,
            'collection_terminal_sha256':'1'*64,'collection_inventory_sha256':'2'*64,'collection_plan_sha256':'3'*64}
        self.packet={'schema':g.VERSION,'scope':'policy_train','source_commit':'f'*40,
            'source_names':sorted(g.SOURCE_NAMES),'files':{},'resources':g.RESOURCES.copy(),'deadlines':g.DEADLINES.copy(),
            **{k:copy.deepcopy(self.cpu[k]) for k in ('archive','manifest','runtime_source','runtime','model_files')},
            'model_directory':'/toy/full-base','plan_sha256':'0'*64,'freeze_sha256':'e'*64,'namespace_admission_sha256':'9'*64}
        self.acceptance={'schema':g.CPU_ACCEPTANCE_SCHEMA,'status':'accepted','scope':'policy_train','required_findings':0,
            'job_id':'123','complete_custody_verified':True,'runtime_qualified_for_this_cpu_execution':True,
            'owned_process_cleanup_verified':True,'training_execution_qualified':False,'scientific_qualified':False,
            'freeze_sha256':'e'*64,'namespace_admission_sha256':'9'*64,
            'numerical_plan_file_sha256':'5'*64,'numerical_plan_sha256':'6'*64,'exposure_sha256':'4'*64}
        self.admission={'schema':g.ADMISSION_SCHEMA,'status':'admitted','scope':'policy_train',
            'freeze_sha256':'e'*64,'namespace_admission_sha256':'9'*64,
            'resources':g.RESOURCES.copy(),'deadlines':g.DEADLINES.copy(),'required_findings':0,
            'source_runtime_scheduler_admitted':True,'scientific_protocol_frozen':True,'namespace_owned':True,
            'output':str(root.parent/(root.name+'-output')),'site_readiness_sha256':'8'*64}
        self.rebind()
        g.INDEPENDENT={n:sha(self.data[n]) for n in g.INDEPENDENT} # toy module ONLY
        self.executed=[];self.captured=[]
        original_module,original_capture=g.module,g.capture
        def mod(name,path,data):
            self.executed.append(Path(path).name)
            assert set(self.data)<=set(self.captured), 'executed before all captures'
            return original_module(name,path,data)
        def capture(path,pin,*args):
            result=original_capture(path,pin,*args);self.captured.append(Path(path).name);return result
        g.module,g.capture=mod,capture
        (root/'TOY-ONLY.json').write_bytes(raw({'actual_acceptance':False,'models':0,'scientific_worlds':0}))

    def rebind(self):
        self.data['cpu-packet.json']=raw(self.cpu)
        self.prepared['training_plan']['outer_provenance_receipt_sha256']=sha(self.data['cpu-packet.json'])
        self.prepared['training_plan_sha256']=digest(self.prepared['training_plan'])
        self.data['prepared.json']=raw(self.prepared)
        self.packet['plan_sha256']=self.prepared['training_plan_sha256']
        self.acceptance.update(cpu_packet_sha256=sha(self.data['cpu-packet.json']),
            prepared_sha256=sha(self.data['prepared.json']),plan_sha256=self.packet['plan_sha256'])
        self.data['cpu-acceptance.json']=raw(self.acceptance)
        self.flush()

    def flush(self):
        self.packet['files']={n:sha(b) for n,b in self.data.items()}
        self.admission.update(packet_sha256=sha(raw(self.packet)),cpu_acceptance_sha256=sha(self.data['cpu-acceptance.json']),
            prepared_sha256=sha(self.data['prepared.json']),plan_sha256=self.packet['plan_sha256'])
        for n,b in self.data.items():(self.root/n).write_bytes(b)
        (self.root/'packet.json').write_bytes(raw(self.packet));(self.root/'admission.json').write_bytes(raw(self.admission))

    def auth(self):
        try:return self.g.authenticate(self.root/'packet.json',sha(raw(self.packet)),self.root/'admission.json',sha(raw(self.admission)))
        except BaseException as e:
            (self.root/'rejection.json').write_bytes(raw({'type':type(e).__name__,'message':str(e),
                'executed':self.executed,'captured':self.captured}));raise


class ScientificSupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ['ACE_SCIENTIFIC_GPU_TEST_OUTPUT']);cls.output.mkdir(exist_ok=False)
    def toy(self,name):return Toy(self.output/self._testMethodName/name)
    def plain(self):
        root=self.output/self._testMethodName;root.mkdir();return load(SOURCE),root

    def test_unified_cpu_acceptance_requires_numeric_pins_and_exact_exposure(self):
        t=self.toy('unified_valid');t.auth()
        self.assertNotEqual(t.acceptance['numerical_plan_file_sha256'],t.acceptance['numerical_plan_sha256'])
        self.assertFalse(any(name.startswith('numerical') and name.endswith('.json') for name in t.captured))
        fields=('numerical_plan_file_sha256','numerical_plan_sha256','exposure_sha256')
        cases=[('missing_'+field,field,None) for field in fields]
        for field in fields:
            cases.extend((f'malformed_{field}_{i}',field,bad) for i,bad in enumerate(('', 'a'*63, 'g'*64, True)))
        cases.extend([('exposure_disagreement','exposure_sha256','7'*64),
                      ('legacy_runtime_name','runtime_qualified_for_this_cpu_execution',None),
                      ('extra_field','unexpected_numeric_override','a'*64)])
        for label,field,bad in cases:
            with self.subTest(case=label):
                t=self.toy(label)
                if bad is None:del t.acceptance[field]
                else:t.acceptance[field]=bad
                if label=='legacy_runtime_name':t.acceptance['runtime_qualified']=True
                t.data['cpu-acceptance.json']=raw(t.acceptance);t.flush()
                with self.assertRaisesRegex(ValueError,'CPU acceptance'):t.auth()
                self.assertEqual(t.executed,[])

    def test_authentication_capture_and_pure_plan_before_any_heavy_import(self):
        t=self.toy('valid');before=set(sys.modules);t.auth()
        self.assertEqual(set(t.data),set(t.captured)-{'packet.json','admission.json'})
        self.assertFalse({'torch','numpy','transformers'}&(set(sys.modules)-before))
        self.assertFalse(any(n.startswith('llm_intervention_') for n in set(sys.modules)-before))
        t=self.toy('pure_plan_reject');t.prepared['training_plan']['toy_gate']='bad';t.rebind()
        with self.assertRaisesRegex(ValueError,'pure worker plan'):t.auth()
        self.assertNotIn(Path(t.g.WORKER).stem,sys.modules)
        t=self.toy('corrupt_source');(t.root/t.g.WORKER).write_bytes(b'raise AssertionError("must not execute")')
        with self.assertRaisesRegex(ValueError,'pin differs'):t.auth()
        self.assertEqual(t.executed,[])

    def test_admission_and_cpu_acceptance_must_be_independent_complete_and_bound(self):
        for field,bad in [('status','pending'),('complete_custody_verified',False),('owned_process_cleanup_verified',False),
                          ('runtime_qualified_for_this_cpu_execution',False),('freeze_sha256','0'*64),
                          ('namespace_admission_sha256','0'*64),('scientific_qualified',True),('required_findings',True)]:
            with self.subTest(field=field):
                t=self.toy(field);t.acceptance[field]=bad;t.data['cpu-acceptance.json']=raw(t.acceptance);t.flush()
                with self.assertRaisesRegex(ValueError,'CPU acceptance'):t.auth()
                self.assertEqual(t.executed,[])
        for field in ('cpu_packet_sha256','prepared_sha256','plan_sha256'):
            t=self.toy(field);t.acceptance[field]='0'*64;t.data['cpu-acceptance.json']=raw(t.acceptance);t.flush()
            with self.assertRaisesRegex(ValueError,'CPU acceptance'):t.auth()
        t=self.toy('admission');t.admission['namespace_owned']=False;t.flush()
        with self.assertRaisesRegex(ValueError,'admission'):t.auth()
        t=self.toy('extra');t.acceptance['unreviewed_override']=True;t.data['cpu-acceptance.json']=raw(t.acceptance);t.flush()
        with self.assertRaisesRegex(ValueError,'CPU acceptance'):t.auth()

    def test_original_runtime_model_sources_and_collection_lineage_are_not_substitutable(self):
        for field in ('runtime','runtime_source','archive','manifest','model_files'):
            t=self.toy(field);t.packet[field]={'substitute':True};t.flush()
            with self.assertRaisesRegex(ValueError,'runtime/model'):t.auth()
        t=self.toy('source');name=sorted(t.g.PREPARATION_SOURCES)[0];t.cpu['files'][name]='0'*64;t.rebind()
        with self.assertRaisesRegex(ValueError,'shared source'):t.auth()
        t=self.toy('missing_source');t.cpu['source_names'].pop();t.rebind()
        with self.assertRaisesRegex(ValueError,'shared source'):t.auth()
        t=self.toy('lineage');t.prepared['collection_inventory_sha256']='0'*64;t.rebind()
        with self.assertRaisesRegex(ValueError,'collection lineage'):t.auth()
        t=self.toy('tokens');t.prepared['tokenizer_sha256']='0'*64;t.rebind()
        with self.assertRaisesRegex(ValueError,'model tokenizer'):t.auth()
        t=self.toy('prompt_bound');t.prepared['training_plan']['records'][0]['prompt_ids']=[1]*4745;t.rebind()
        with self.assertRaisesRegex(ValueError,'4744'):t.auth()

    def test_exact_source_membership_and_reviewed_worker_helper_pins(self):
        for label,mutation in [('missing',lambda t:t.data.pop('llm_intervention_gradient_probe.py')),
                               ('extra',lambda t:t.data.update({'unrelated.py':b'# nope'})),
                               ('worker',lambda t:t.data.update({t.g.WORKER:b'# replacement'})),
                               ('shape_helper',lambda t:t.data.update({'llm_intervention_screen_training_gpu_worker.py':b'# replacement'}))]:
            t=self.toy(label);mutation(t);t.flush()
            with self.assertRaises(ValueError):t.auth()
            self.assertEqual(t.executed,[])

    def test_slurm_actual_four_cpu_and_requested_typed_mig_are_both_required(self):
        g,out=self.plain()
        h=types.SimpleNamespace(scheduler_fields=lambda s:dict(pair.split('=',1) for pair in s.split()),
            memory_mib=lambda s:int(s[:-1])*(1024 if s.endswith('G') else 1))
        fields=dict(JobId='123',Account='ucb736_asc2',Partition='aa100',QOS='gpu-testing',JobState='RUNNING',
            NumNodes='1',NumTasks='1',TimeLimit='00:45:00',Requeue='0',Restarts='0',NumCPUs='4',**{'CPUs/Task':'4'},
            UserId='paco0228(99)',AllocTRES='cpu=4,mem=15G,node=1,gres/gpu:a100_3g.20gb=1',
            ReqTRES='cpu=4,mem=15360M,node=1,gres/gpu=1,gres/gpu:a100_3g.20gb=1')
        def check(f):return g.authenticate_allocation(' '.join(k+'='+v for k,v in f.items()),'123',g.RESOURCES,h)
        check(fields)
        for i,(key,value) in enumerate([('NumCPUs','8'),('QOS','gpu-normal'),('TimeLimit','01:00:00'),('Requeue','1'),
            ('AllocTRES',fields['AllocTRES'].replace('cpu=4','cpu=8')),
            ('AllocTRES',fields['AllocTRES'].replace('15G','12G')),
            ('ReqTRES',fields['ReqTRES'].replace('gres/gpu:a100_3g.20gb=1','gres/gpu:a100_3g.20gb=2')),
            ('ReqTRES',fields['ReqTRES']+',gres/gpu:a100=1')]):
            changed=dict(fields,**{key:value});(out/f'case-{i}.json').write_bytes(raw(changed))
            with self.assertRaises(ValueError):check(changed)

    def test_handoff_inherits_original_clock_and_rejects_parent_job_stage_changes(self):
        g,out=self.plain();args=types.SimpleNamespace(packet_sha256='a'*64,admission_sha256='b'*64,runtime_stage_sha256='c'*64)
        handoff=dict(vars(args),parent_pid=40,job_id='123',started=100.,deadlines=g.DEADLINES.copy())
        self.assertEqual(g.validate_handoff(handoff,args,250.,40,'123'),2720.)
        for key,value in [('started',300.),('parent_pid',41),('job_id','124'),('runtime_stage_sha256','0'*64),
                          ('deadlines',dict(g.DEADLINES,worker=2700))]:
            bad=dict(handoff,**{key:value});(out/(key+'.json')).write_bytes(raw(bad))
            with self.assertRaises(ValueError):g.validate_handoff(bad,args,250.,40,'123')
        with self.assertRaises(ValueError):g.validate_handoff(handoff,args,2720.,40,'123')
        handlers={};timers=[];kills=[]
        with patch.object(g.signal,'signal',side_effect=lambda s,h:handlers.update({s:h})),patch.object(g.signal,'setitimer',side_effect=lambda *v:timers.append(v)),patch.object(g.time,'monotonic',return_value=700.),patch.object(g.os,'killpg',side_effect=lambda *v:kills.append(v)),patch.object(g.os,'_exit',side_effect=SystemExit):
            g.arm_hard_deadline(100.,types.SimpleNamespace(pid=40))
            self.assertAlmostEqual(timers[-1][1],2070.)
            with self.assertRaises(SystemExit):handlers[g.signal.SIGALRM](None,None)
        self.assertEqual(kills,[(40,g.signal.SIGKILL)])

    def test_real_progress_180_seconds_is_separate_from_owned_memory_freshness(self):
        g,out=self.plain();sample={'returncode':0,'owned_rows':['40, GPU-toy, 100']}
        watch=g.ProgressWatch(0,('score-0',),40)
        for now in (60,120,180):watch.observe(now,sample,('score-0',))
        with self.assertRaisesRegex(TimeoutError,'180'):watch.observe(181,sample,())
        watch=g.ProgressWatch(0,(),40)
        for now in (80,160,240):watch.observe(now,sample,(str(now),))
        for row in ('41, GPU-toy, 100','40, GPU-toy, N/A','40, GPU-toy, nan','40, GPU-toy, 0'):
            self.assertFalse(g.positive_owned_memory({'returncode':0,'owned_rows':[row]},40))
        with self.assertRaisesRegex(TimeoutError,'uncertain'):
            watch.observe(331,{'returncode':1,'owned_rows':[]},('new-score',))
        (out/'phase-000.json').write_text('{');self.assertEqual(g.progress_records(out),())
        (out/'phase-001.json').write_bytes(raw({'phase':'score_record_complete'}))
        self.assertEqual(g.progress_records(out),('phase-001.json',))
        work=out/'first-gpu-work.json';work.write_bytes(raw({'cuda_interval_ms':0,'peak_gpu_allocated_bytes':10}))
        self.assertFalse(g.positive_work(work));work.write_bytes(raw({'cuda_interval_ms':1,'peak_gpu_allocated_bytes':10}))
        self.assertTrue(g.positive_work(work))

    def test_original_failure_survives_cleanup_failure_and_unclean_exit_cannot_pass(self):
        g,out=self.plain();helper=types.SimpleNamespace(terminate=lambda *a:(_ for _ in ()).throw(RuntimeError('secondary cleanup')))
        try:raise ValueError('original worker failure')
        except ValueError as first:g.preserve_failure(out,first,types.SimpleNamespace(pid=40),helper,100.)
        self.assertEqual(json.loads((out/'failure.json').read_bytes())['message'],'original worker failure')
        self.assertEqual(json.loads((out/'cleanup-failure.json').read_bytes())['message'],'secondary cleanup')
        clean=dict(leader_returncode=0,group_still_exists=False,term_sent=False,kill_sent=False,errors=[])
        g.clean_success(clean)
        for key,value in [('term_sent',True),('group_still_exists',True),('errors',['failed']),('leader_returncode',1)]:
            with self.assertRaises(ValueError):g.clean_success(dict(clean,**{key:value}))
        with self.assertRaises(FileExistsError):g.save(out/'failure.json',{'message':'replacement'})

    def test_parent_doubles_stage_then_owned_child_then_cleanup_and_saved_validation(self):
        for case in ('success','runtime_failure','allocation_failure'):
            t=self.toy(case);g=t.g;context=t.auth()
            out=Path(t.admission['output']);out.mkdir();scratch=t.root.parent/(case+'-scratch');scratch.mkdir()
            args=types.SimpleNamespace(output=str(out),started=100.,process_started=0.,packet=str(t.root/'packet.json'),
                packet_sha256=sha(raw(t.packet)),admission=str(t.root/'admission.json'),admission_sha256=sha(raw(t.admission)))
            order=[];clock=[101.];holder={}
            def stage(*a):
                order.append('stage')
                if case=='runtime_failure':raise ValueError('runtime archive rejected')
                return {'staged_prefix':str(scratch/'runtime')}
            d=types.SimpleNamespace(events=lambda *a:lambda *a,**kw:None,stage_runtime=stage,
                provenance_contract=lambda *a:{'toy':'provenance'},BOOTSTRAP='toy not executed')
            gpu=types.SimpleNamespace(scheduler_record=lambda *a:'fabricated scheduler',
                child_environment=lambda *a:{},install_child_environment=lambda e:{'toy':True},
                require_gpu_sample=lambda s,active:self.assertEqual(s['owned_rows'],[]))
            clean=dict(leader_returncode=0,group_still_exists=False,term_sent=False,kill_sent=False,errors=[])
            def cleanup(*a):order.append('cleanup');return clean
            def sample(pid):return {'returncode':0,'owned_rows':[] if holder['process'].calls>=3 else [f'{pid}, GPU-toy, 123']}
            helper=types.SimpleNamespace(terminate=cleanup,gpu_process_sample=sample,group_alive=lambda pid:False)
            def allocation(*a):
                order.append('allocation')
                if case=='allocation_failure':raise ValueError('allocation rejected')
                return {'toy':True}
            class Process:
                pid=40
                def __init__(self,command,**kw):
                    order.append('child');self.calls=0;holder['process']=self
                    self.command=command
                    artifact=out/'session/worker/artifacts';artifact.mkdir(parents=True)
                    g.save(artifact/'first-gpu-work.json',{'cuda_interval_ms':1,'peak_gpu_allocated_bytes':1})
                def poll(self):self.calls+=1;return 0 if self.calls>=3 else None
            def validate(*a):order.append('saved_validate');return {'toy':'full validator seam'}
            context=list(context);context[7:]=[d,gpu,helper]
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)}),patch.object(g,'authenticate_allocation',side_effect=allocation),patch.object(g,'arm_hard_deadline'),patch.object(g.signal,'signal'),patch.object(g.signal,'setitimer'),patch.object(g.subprocess,'Popen',side_effect=Process),patch.object(g.time,'monotonic',side_effect=lambda:clock[0]),patch.object(g.time,'sleep',side_effect=lambda n:clock.__setitem__(0,clock[0]+n)),patch.object(g,'validate_worker',side_effect=validate):
                if case=='success':g.parent(args,*context)
                else:
                    with self.assertRaisesRegex(ValueError,'rejected'):g.parent(args,*context)
            if case=='success':
                self.assertEqual(order,['allocation','stage','child','cleanup','saved_validate'])
                terminal=json.loads((out/'terminal.json').read_bytes());self.assertFalse(terminal['scientific_qualified'])
                handoff=json.loads((out/'handoff.json').read_bytes());self.assertEqual(handoff['started'],100.)
                self.assertIn('--handoff-sha256',holder['process'].command)
                self.assertNotIn('--worker-deadline',holder['process'].command)
            else:
                self.assertNotIn('child',order);self.assertTrue((out/'failure.json').is_file())
                self.assertFalse((out/'terminal.json').exists())

    def test_failed_cleanup_holds_main_watchdog_through_publication_error_until_live_group_exit(self):
        for case in ('cleanup_raises_then_group_exits','uncertain_group_until_hard','false_closed_receipt_until_hard'):
            with self.subTest(case=case):
                t=self.toy(case);g=t.g;context=list(t.auth());out=Path(t.admission['output'])
                scratch=t.root.parent/(case+'-scratch');scratch.mkdir()
                clock=[100.];state={'closed':False};timers=[];handlers={};kills=[];err=io.StringIO()
                process=types.SimpleNamespace(pid=40,poll=lambda:0 if state['closed'] else None)
                d=types.SimpleNamespace(events=lambda *a:lambda *a,**kw:None,
                    stage_runtime=lambda *a:{'staged_prefix':str(scratch/'runtime')},
                    provenance_contract=lambda *a:{'toy':True},BOOTSTRAP='not executed')
                gpu=types.SimpleNamespace(scheduler_record=lambda *a:'toy',child_environment=lambda *a:{},
                    install_child_environment=lambda *a:{'toy':True})
                def group_alive(pid):
                    self.assertEqual(pid,40)
                    if case=='uncertain_group_until_hard':raise OSError('group observation uncertain')
                    return not state['closed']
                def cleanup(*a):
                    if case=='cleanup_raises_then_group_exits':raise RuntimeError('secondary cleanup failure')
                    return {'group_still_exists':case!='false_closed_receipt_until_hard','leader_returncode':0}
                def sample(*a):raise ValueError('original monitor failure')
                helper=types.SimpleNamespace(terminate=cleanup,gpu_process_sample=sample,group_alive=group_alive)
                context[7:]=[d,gpu,helper]
                def sleep(seconds):
                    self.assertFalse(any(delay==0 for _,delay in timers),'watchdog cancelled before group exit')
                    if case=='cleanup_raises_then_group_exits':state['closed']=True;clock[0]+=seconds
                    else:clock[0]=2770.
                original_save=g.save
                def save(path,value):
                    if Path(path).name=='failure.json':raise OSError('failure publication failed')
                    return original_save(path,value)
                argv=[g.ENTRY,'--packet',str(t.root/'packet.json'),'--packet-sha256',sha(raw(t.packet)),
                    '--admission',str(t.root/'admission.json'),'--admission-sha256',sha(raw(t.admission)),
                    '--output',str(out)]
                with ExitStack() as stack:
                    for manager in (
                        patch.object(g.sys,'argv',argv),patch.object(g,'authenticate',return_value=tuple(context)),
                        patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)}),
                        patch.object(g,'authenticate_allocation',return_value={'toy':True}),
                        patch.object(g.subprocess,'Popen',return_value=process),patch.object(g,'save',side_effect=save),
                        patch.object(g.time,'monotonic',side_effect=lambda:clock[0]),patch.object(g.time,'sleep',side_effect=sleep),
                        patch.object(g.signal,'signal',side_effect=lambda s,h:handlers.update({s:h})),
                        patch.object(g.signal,'setitimer',side_effect=lambda *v:timers.append(v)),
                        patch.object(g.os,'killpg',side_effect=lambda *v:kills.append(v)),
                        patch.object(g.os,'_exit',side_effect=SystemExit(124)),redirect_stderr(err)):
                        stack.enter_context(manager)
                    if case=='cleanup_raises_then_group_exits':
                        with self.assertRaisesRegex(ValueError,'original monitor failure'):g.main()
                        self.assertEqual(timers[-1][1],0)
                        # Even a late saved handler has no authority over the retired PID.
                        with self.assertRaises(SystemExit):handlers[g.signal.SIGALRM](None,None)
                        self.assertEqual(kills,[])
                    else:
                        with self.assertRaises(SystemExit):g.main()
                        self.assertFalse(any(delay==0 for _,delay in timers))
                        self.assertEqual(kills,[(40,g.signal.SIGKILL)])
                self.assertIn('original monitor failure',err.getvalue())
                self.assertIn('failure publication failed',err.getvalue())
                self.assertFalse((out/'terminal.json').exists())
                (out/'captured-stderr.log').write_text(err.getvalue())
                (out/'watchdog-fixture.json').write_bytes(raw({'timers':timers,'owned_kills':kills,
                    'actual_group_closed':state['closed'],'processes_created':0}))

    def test_saved_worker_summary_disagreement_fails_without_tensor_reads(self):
        g,out=self.plain();summary={'toy':'saved'};checked={'summary_sha256':sha(raw(summary))}
        g.consistent_summary({'summary':summary},summary,raw(summary),checked)
        with self.assertRaisesRegex(ValueError,'outer summaries'):
            g.consistent_summary({'summary':{'changed':True}},summary,raw(summary),checked)
        with self.assertRaisesRegex(ValueError,'bytes'):
            g.consistent_summary({'summary':summary},summary,raw(summary)+b' ',checked)
        # Both run and parent validation must pass independently supplied pins.
        tree=ast.parse(SOURCE.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for name in ('run_training','validate_outputs'):
            found=[c for c in calls if (isinstance(c.func,ast.Name) and c.func.id==name) or (isinstance(c.func,ast.Attribute) and c.func.attr==name)]
            self.assertTrue(found)
            for call in found:self.assertTrue({'prepared_sha256','plan_sha256'}<={k.arg for k in call.keywords})
        self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr=='arm_final_deadline' for c in calls))


if __name__=='__main__':unittest.main(verbosity=2)
