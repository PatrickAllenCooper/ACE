"""New artificial supervisor/seam tests. No staging, Slurm, draws or collection.

Boundary doubles are deliberate: these tests qualify new glue only. Original
collector tests/reviews remain predecessor evidence, not repeated here.
"""
import ast
from contextlib import ExitStack
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).absolute().parent))
import supervise_llm_intervention_scientific_cpu as s
import collect_llm_intervention_scientific_training_cpu as c


def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(raw(v)[:-1])
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw(v))
def read(p):return json.loads(p.read_bytes())

def constant(name,field):
    tree=ast.parse(Path(s.__file__).with_name(name).read_bytes())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==field for t in n.targets)))


def packet_fixture(root,mode):
    sources=s.work_sources(mode)|{s.ENTRY}
    data={n:Path(s.__file__).with_name(n).read_bytes() for n in sources|set(s.HELPER_PINS)}
    data['proposal.json']=Path(s.__file__).with_name('proposal.json').read_bytes()
    service=types.SimpleNamespace(RUNTIME=constant(s.SERVICE,'RUNTIME'))
    d=types.SimpleNamespace(**{k:constant(s.DIAGNOSTIC,k) for k in ('ARCHIVE_SHA','MANIFEST_SHA','RUNTIME_SOURCE')})
    p=dict(schema=s.VERSION,mode=mode,source_commit='a'*40,output=str(root),
        files={},source_names=sorted(sources),protocol_source_pins={'scripts/research/'+n:sha(data[n]) for n in s.PROTOCOL_NAMES},
        archive=dict(path='/toy/archive.tar',sha256=d.ARCHIVE_SHA),manifest=dict(path='/toy/manifest.json',sha256=d.MANIFEST_SHA),
        runtime_source=d.RUNTIME_SOURCE,runtime=service.RUNTIME,resources=dict(s.BASE_RESOURCES,wall_seconds=300 if mode=='realize' else 900),
        deadlines=dict(s.REALIZE_DEADLINES) if mode=='realize' else dict(startup=120,work=850,cleanup=875,hard=890),
        freeze_sha256=digest('fake freeze'),namespace_admission_sha256=digest('fake namespace'))
    if mode=='collect':
        for i in range(32):data[f'descriptor-{i:03d}.json']=raw({'fabricated_input_only':i})
        data['realization-terminal.json']=raw({'fabricated_metadata_only':True})
        names={'proposal.json','realization-terminal.json'}|{f'descriptor-{i:03d}.json' for i in range(32)}
        inner=dict(schema='ace-scientific-training-collection-cpu-v1',files={n:sha(data[n]) for n in names},
            source_pins={'scripts/research/'+n:sha(data[n]) for n in s.work_sources(mode)},
            freeze_sha256=p['freeze_sha256'],namespace_admission_sha256=p['namespace_admission_sha256'],proposal_sha256=s.PROPOSAL_SHA)
        data['collection-packet.json']=raw(inner)
        data['collection-admission.json']=raw(dict(packet_sha256=sha(raw(inner)),output=str(root/'session/worker/task')))
        data['realization-acceptance.json']=raw(dict(status='accepted',required_findings=0,complete_custody_verified=True,
            realization_terminal_sha256=sha(data['realization-terminal.json']),freeze_sha256=p['freeze_sha256'],namespace_admission_sha256=p['namespace_admission_sha256']))
    p['files']={n:sha(b) for n,b in data.items()}
    a=dict(schema=s.VERSION+'/admission',mode=mode,status='admitted',packet_sha256=sha(raw(p)),output=str(root),
        freeze_sha256=p['freeze_sha256'],namespace_admission_sha256=p['namespace_admission_sha256'],
        scientific_protocol_frozen=True,namespace_owned=True,source_runtime_scheduler_qualified=True,accounting_admitted=True,
        resources=p['resources'],deadlines=p['deadlines'])
    return p,data,a,service,d


def slurm(mode):
    return ('JobId=123 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 '
        'TimeLimit='+('00:05:00' if mode=='realize' else '00:15:00')+' Requeue=0 Restarts=0 UserId=paco0228(42) '
        'AllocTRES=cpu=2,mem=7680M,node=1 ReqTRES=cpu=2,mem=7680M,node=1')


class SupervisorGlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['ACE_SCIENTIFIC_SUPERVISOR_TEST_OUTPUT']).resolve();cls.root.mkdir(exist_ok=False)
        save(cls.root/'ARTIFICIAL-ONLY.json',dict(actual_world_draws=0,actual_collection_calls=0,
            actual_runtime_stages=0,model_loads=0,remote_actions=0))

    def setUp(self):
        self.out=self.root/self._testMethodName;self.out.mkdir()

    def test_fixed_profiles_reject_acpu_memory_overage_auto_cpu_and_deadline_reset(self):
        for mode in ('realize','collect'):
            p,*_=packet_fixture(self.out/mode,mode)
            s.resource_contract(mode,p['resources'],p['deadlines']);s.allocation(slurm(mode),'123',mode)
            for field,value in [('memory_mib',8192),('cpus',3),('gpus',1),('requeue',True)]:
                r=dict(p['resources'],**{field:value})
                with self.subTest(mode=mode,field=field),self.assertRaises(ValueError):s.resource_contract(mode,r,p['deadlines'])
            for text in (slurm(mode).replace('NumCPUs=2','NumCPUs=3'),slurm(mode).replace('7680M','8192M'),
                         slurm(mode)+' JobId=123',slurm(mode).replace('node=1','node=1,gres/gpu=1')):
                with self.assertRaises(ValueError):s.allocation(text,'123',mode)
            for d in [dict(p['deadlines'],work=901),dict(p['deadlines'],cleanup=p['deadlines']['work']),
                      dict(p['deadlines'],hard=p['resources']['wall_seconds'])]:
                with self.assertRaises(ValueError):s.resource_contract(mode,p['resources'],d)
        ctx={'packet':{'deadlines':dict(startup=120,work=850)}}
        with patch.object(s.time,'monotonic',return_value=225),patch.object(s.signal,'setitimer') as timer:
            with self.assertRaisesRegex(ValueError,'startup'):s.start_work(self.out,ctx,100)
            timer.assert_not_called()

    def test_exact_mode_source_data_and_independent_admission(self):
        for mode in ('realize','collect'):
            root=self.out/mode;p,data,a,service,d=packet_fixture(root,mode)
            s.packet_shape(p,data,root,a,sha(raw(p)),service,d)
            save(self.out/(mode+'.packet.json'),p);save(self.out/(mode+'.admission.json'),a)
            for key,value in [('status','pending'),('namespace_owned',False),('scientific_protocol_frozen',False),
                              ('source_runtime_scheduler_qualified',False),('accounting_admitted',False),('packet_sha256','0'*64)]:
                bad=dict(a,**{key:value})
                with self.subTest(mode=mode,key=key),self.assertRaises(ValueError):s.packet_shape(p,data,root,bad,sha(raw(p)),service,d)
            changed=dict(data);changed['model.safetensors']=b'forbidden'
            with self.assertRaises(ValueError):s.packet_shape(p,changed,root,a,sha(raw(p)),service,d)
            changed=dict(data);changed.pop(next(iter(s.PROTOCOL_NAMES)))
            with self.assertRaises(ValueError):s.packet_shape(p,changed,root,a,sha(raw(p)),service,d)

    def test_collection_cannot_self_accept_realization_or_stage_development_payload(self):
        root=self.out/'collect';p,data,a,service,d=packet_fixture(root,'collect')
        for field,value in [('status','pending'),('required_findings',1),('complete_custody_verified',False),
                            ('realization_terminal_sha256','0'*64),('namespace_admission_sha256','0'*64)]:
            pd,dd=copy.deepcopy(p),dict(data);accepted=json.loads(dd['realization-acceptance.json']);accepted[field]=value
            dd['realization-acceptance.json']=raw(accepted);pd['files']['realization-acceptance.json']=sha(raw(accepted))
            ad=dict(a,packet_sha256=sha(raw(pd)))
            with self.subTest(field=field),self.assertRaises(ValueError):s.packet_shape(pd,dd,root,ad,sha(raw(pd)),service,d)
        pd,dd=copy.deepcopy(p),dict(data);dd['descriptor-032.json']=b'{}';pd['files']['descriptor-032.json']=sha(b'{}')
        with self.assertRaises(ValueError):s.packet_shape(pd,dd,root,dict(a,packet_sha256=sha(raw(pd))),sha(raw(pd)),service,d)

    def seam(self,label,*,operation_error=False,observer_error=None):
        out=self.out/label;events=[]
        sources={n:b'fixture_value = 1\n' for n in c.COLLECTION_SOURCES}
        p=dict(source_pins={n:sha(b) for n,b in sources.items()},roster_sha256='a'*64,world_pins={str(w):'b'*64 for w in c.WORLD_IDS},
            configuration_pins={str(w):'c'*64 for w in c.WORLD_IDS},proposal_sha256='d'*64)
        def bindings(*args):
            importlib.import_module('llm_intervention_contract')
            return {},{},{},[]
        def invoke(**kw):
            events.append('invoke')
            if operation_error:raise RuntimeError('original operation')
            return {}
        def observe(phase,loader):
            module=sys.modules['llm_intervention_contract']
            self.assertIs(module.__loader__,loader);self.assertIs(loader.loaded[module.__name__],module)
            self.assertIs(loader.sources,loader.source);events.append(phase)
            if observer_error==phase:raise OSError('observer '+phase)
        args=dict(packet_raw=b'{}',packet_sha256='0'*64,inputs={},admission_raw=b'{}',admission_sha256='0'*64,output=out,observation=observe)
        with patch.object(c,'_authenticate',return_value=(p,sources)),patch.object(c,'_bindings',bindings),\
             patch.object(c,'_invoke_collection',invoke),patch.object(c,'_verify_collection',return_value={}):
            try:result=c.run_collection(**args)
            except BaseException as exc:return out,events,exc
        return out,events,result

    def test_new_observer_seam_uses_live_source_before_and_after_operation(self):
        out,events,result=self.seam('success')
        self.assertEqual(events,['before_collection','invoke','after_collection'])
        self.assertEqual(result['status'],'collection_complete')
        self.assertNotIn('llm_intervention_contract',sys.modules)
        out,events,exc=self.seam('pre_rejection',observer_error='before_collection')
        self.assertIsInstance(exc,OSError);self.assertEqual(events,['before_collection'])
        self.assertFalse(read(out/'failure.json')['collection_invoked'])

    def test_exit_observer_failure_preserves_first_exception_and_blocks_false_success(self):
        out,events,exc=self.seam('double_failure',operation_error=True,observer_error='after_collection')
        self.assertIsInstance(exc,RuntimeError);self.assertEqual(str(exc),'original operation')
        self.assertEqual(read(out/'failure.json')['message'],'original operation')
        self.assertEqual(read(out/'observation-failure.json')['message'],'observer after_collection')
        self.assertFalse((out/'terminal.json').exists());self.assertNotIn('llm_intervention_contract',sys.modules)
        out,events,exc=self.seam('observer_only',observer_error='after_collection')
        self.assertIsInstance(exc,OSError);self.assertFalse((out/'terminal.json').exists())

    def test_saved_runtime_validation_uses_loaded_key_and_exact_numpy_source_closure(self):
        p,data,a,service,d=packet_fixture(self.out/'outer','collect');out=self.out/'worker';out.mkdir()
        ctx=dict(packet=p,data=data,packet_pin=sha(raw(p)),admission_pin=sha(raw(a)),
            diagnostic=types.SimpleNamespace(check_prefix=lambda *args,**kwargs:None))
        snapshot=dict(sys_path=['/fake/stage/lib'],packages={'numpy':{'loaded':False}},module_paths={})
        before=dict(snapshot=snapshot,captured_project_modules={},environment={},accelerators={})
        save(out/'runtime-before.json',before)
        expected={Path(n).stem:{'sha256':p['files'][n]} for n in s.work_sources('collect') if n!=s.COLLECT}
        after=copy.deepcopy(before);after['captured_project_modules']=expected
        after['snapshot'].update(packages={'numpy':{'loaded':True,'version':'2.2.5'}},module_paths={'numpy.core':'/fake/stage/numpy/core.so'})
        for phase in ('before_collection','after_collection'):save(out/('runtime-'+phase+'.json'),after)
        tree=dict(files={},bytes=0);save(out/'task-inventory.json',tree)
        terminal=dict(schema=s.VERSION,status='worker_complete',mode='collect',packet_sha256=ctx['packet_pin'],
            admission_sha256=ctx['admission_pin'],stage_sha256='e'*64,task_inventory_sha256=sha(raw(tree)),summary={'toy':True})
        save(out/'terminal.json',terminal)
        with patch.object(s,'inventory',return_value=tree),patch.object(s,'task_summary',return_value={'toy':True}):
            self.assertEqual(s.validate_worker(ctx,out,{'staged_prefix':'/fake/stage'},'e'*64),sha(raw(terminal)))
            bad=copy.deepcopy(after);bad['captured_project_modules'].pop('llm_intervention_world')
            (out/'runtime-after_collection.json').write_bytes(raw(bad))
            with self.assertRaisesRegex(ValueError,'runtime/source'):s.validate_worker(ctx,out,{'staged_prefix':'/fake/stage'},'e'*64)

    def test_parent_reuses_stage_bootstrap_and_absolute_owned_supervision_before_validation(self):
        p,data,a,service,d=packet_fixture(self.out/'outer','collect');out=Path(p['output']);out.mkdir()
        inputs=self.out/'inputs';inputs.mkdir();scratch=self.out/'scratch';scratch.mkdir()
        order=[]
        d.events=lambda *args:lambda phase,**kw:order.append(phase)
        d.stage_runtime=lambda p,helper,path,event:dict(staged_prefix=str(path))
        d.provenance_contract=lambda p:{'fake':True};d.BOOTSTRAP='FABRICATED BOOTSTRAP'
        ctx=dict(packet=p,data={**data,'llm_intervention_runtime_archive.py':b'TOY=True\n','probe.py':b'def kill_owned(p): raise AssertionError("no process")\n'},
            root=inputs,diagnostic=d,training=types.SimpleNamespace(cpu_environment=lambda p,e:dict(e)),packet_pin=sha(raw(p)),
            admission_pin=sha(raw(a)),packet_path=inputs/'packet.json',admission_path=inputs/'admission.json')
        args=types.SimpleNamespace(output=str(out),started=100.,process_started=1.)
        def supervise(command,env,session,origin,pin,kill):
            order.append('supervise');self.assertEqual(origin,100.);self.assertEqual(s.B.WORK,850)
            self.assertEqual(s.B.CLEANUP,875);self.assertEqual(command[:4],[str(scratch/'ace-scientific-collect-123/runtime/bin/python'),'-I','-B','-c'])
            self.assertIn('--handoff-sha256',command);return dict(group_still_exists=False,kill_sent=False,returncode=0,errors=[])
        def validate(*args):order.append('validate');return 'f'*64
        with ExitStack() as stack:
            for name in ('STARTUP','WORK','CLEANUP','HARD'):stack.enter_context(patch.object(s.B,name,getattr(s.B,name)))
            stack.enter_context(patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)}))
            stack.enter_context(patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=slurm('collect'),stderr='')))
            stack.enter_context(patch.object(s,'no_accelerators',return_value={}))
            timer=stack.enter_context(patch.object(s.signal,'setitimer'))
            stack.enter_context(patch.object(s.B,'supervise',supervise));stack.enter_context(patch.object(s,'validate_worker',validate))
            stack.enter_context(patch.object(s,'recheck',lambda ctx:None));stack.enter_context(patch.object(s.time,'monotonic',return_value=150.))
            s.parent(args,ctx)
            self.assertIn(unittest.mock.call(signal_itimer(),800.),timer.call_args_list)
        self.assertLess(order.index('supervise'),order.index('validate'))
        self.assertEqual(read(out/'session/handoff.json')['started_monotonic'],100.)
        self.assertEqual(read(out/'terminal.json')['status'],'complete')

    def test_realization_exit_observer_failure_does_not_mask_original(self):
        out=self.out/'outer/session/worker';out.mkdir(parents=True)
        p,data,a,service,d=packet_fixture(self.out/'outer','realize');stage={'staged_prefix':'/fake/stage'}
        save(out.parent.parent/'stage.json',stage)
        helper=types.ModuleType('llm_intervention_screen_realization')
        helper.realization_recipes=lambda *a,**kw:None
        def fail(**kw):raise RuntimeError('original realization failure')
        helper.realize_screen_worlds=fail
        loader=types.SimpleNamespace()
        ctx=dict(packet=p,data=data,root=self.out,training=types.SimpleNamespace(CapturedSources=lambda *args:loader))
        handoff=dict(started_monotonic=100.,stage_sha256=sha(raw(stage)))
        count=[]
        def observe(*args,**kwargs):
            count.append(kwargs['after'])
            if len(count)==3:raise OSError('exit observation failure')
            return {'fabricated':True}
        with patch.dict(sys.modules,{'llm_intervention_screen_realization':helper}),patch.object(s,'runtime_record',observe),\
             patch.object(s,'recheck',lambda ctx:None),patch.object(s,'start_work',lambda *args:None):
            with self.assertRaisesRegex(RuntimeError,'original realization'):s.child(types.SimpleNamespace(output=str(out)),ctx,handoff)
        self.assertEqual(read(out/'observation-failure.json')['message'],'exit observation failure')
        self.assertNotIn(loader,sys.meta_path)

    def test_immutable_task_inventory_and_publication_ambiguity(self):
        task=self.out/'task';task.mkdir();save(task/'original.json',{'fake':1})
        expected=s.inventory(task);self.assertEqual(len(expected['files']),1)
        (task/'linked').symlink_to(task/'original.json')
        with self.assertRaises(ValueError):s.inventory(task)
        out=self.out/'publish';out.mkdir();link=os.link
        def interrupted(src,dest):link(src,dest);raise KeyboardInterrupt('after link')
        with patch.object(s.os,'link',interrupted):
            try:s.publish(out,{'status':'complete'})
            except BaseException as exc:s.preserve_failure(out,exc)
        self.assertTrue((out/'terminal.json').exists());self.assertFalse((out/'failure.json').exists())
        self.assertTrue((out/'publication-interruption.json').exists())


def signal_itimer():
    return s.signal.ITIMER_REAL


if __name__=='__main__':unittest.main(verbosity=2)
