"""NEW stdlib boundary doubles; no real campaign, staging, models or responses.

Real source files are copied/hashes captured as inert input bytes. Module,
service, phase, runtime and scheduler APIs are explicit fabricated doubles.
These tests never qualify scientific inputs or replay a scientific trajectory.
"""
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

SOURCE=Path(__file__).with_name('qualify_llm_intervention_scientific_campaign_gpu.py')
REPO=Path(os.environ.get('ACE_RESEARCH_SOURCES','/Users/pat/code/ACE/scripts/research'))
def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def digest(v):return sha(raw(v)[:-1])
def load(path):
    m=types.ModuleType('isolated_scientific_gpu_outer');m.__file__=str(path)
    exec(compile(SOURCE.read_bytes(),str(path),'exec'),vars(m));return m


class Toy:
    def __init__(self,root):
        self.root=root;root.mkdir(parents=True)
        self.inputs=root/'gpu-inputs';self.inputs.mkdir();self.cpu_inputs=root/'cpu-inputs';self.cpu_inputs.mkdir()
        self.cpu_output=root/'cpu-output';self.phase=self.cpu_output/'worker/artifacts/cpu-phase';self.phase.mkdir(parents=True)
        self.campaign=self.cpu_output/'worker/artifacts/campaign';self.campaign.mkdir()
        self.output=root/'gpu-output';self.g=g=load(self.inputs/SOURCE.name);self.calls=[]
        # Static transitive source inventory, never execute upstream implementation.
        names={g.CPU_ENTRY,g.CPU_OWNED,'prepare_llm_intervention_campaign_cpu.py','prepare_llm_intervention_training_cpu.py',
            'llm_intervention_model_service.py',g.LINEAGE,'llm_intervention_scientific_campaign.py',
            'llm_intervention_service_runtime.py','qualify_llm_intervention_service_cpu.py',
            'llm_intervention_campaign_launch.py',g.DIAGNOSTIC}|g.EXTRA_SOURCES
        pending=list(names-{g.ENTRY})
        while pending:
            n=pending.pop()
            for node in ast.walk(ast.parse((REPO/n).read_bytes())):
                deps=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                for dep in deps:
                    if dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_')):
                        name=dep+'.py'
                        if name not in names:names.add(name);pending.append(name)
        self.data={n:(SOURCE.read_bytes() if n==g.ENTRY else (REPO/n).read_bytes()) for n in names}
        self.data['probe.py']=b'# inert toy packet probe, never executed\n'
        self.data['llm_intervention_runtime_archive.py']=(REPO/'llm_intervention_runtime_archive.py').read_bytes()
        self.service=types.SimpleNamespace(SOURCE_NAMES={'llm_intervention_model_service.py'},RUNTIME={'toy':'runtime'},
            digest=digest,identity=lambda f:{'toy-role':f['treatment']},validate_freeze=lambda f:f)
        self.d=types.SimpleNamespace(PROBE_SHA=sha(self.data['probe.py']),HELPER_SHA=sha(self.data['llm_intervention_runtime_archive.py']),
            ARCHIVE_SHA='a'*64,MANIFEST_SHA='b'*64,RUNTIME_SOURCE='/toy/immutable-runtime',BOOTSTRAP='TOY_NOT_EXECUTED')
        self.helper=types.SimpleNamespace(scheduler_fields=lambda s:dict(p.split('=',1) for p in s.split()),
            memory_mib=lambda s:int(s[:-1])*(1024 if s.endswith('G') else 1))
        self.gpu=types.SimpleNamespace();self.launch=types.SimpleNamespace(VERSION='toy-launch',BINDING_ENV='TOY_BINDING',validate_contract=lambda c:c)
        self.lineage={'roles':{p:{'treatment':'sft' if p=='sft_llm' else 'dpo','adapter_sha256':sha(p.encode()),
            'checkpoint_sha256':sha(raw({'checkpoint':p}))} for p in g.POLICIES[6:]}}
        self.expected={'scientific_freeze_sha256':sha(raw({'toy':'freeze'})),
            'cpu_preparation_acceptance_sha256':sha(raw({'toy':'cpu-preparation'})),
            'model_files_sha256':digest({'model.fake':'c'*64}),'runtime_contract_sha256':digest(self.service.RUNTIME)}
        self.data.update({'scientific-freeze.json':raw({'toy':'freeze'}),'cpu-preparation-acceptance.json':raw({'toy':'cpu-preparation'}),
            'prepared.json':raw({'exposure':{'counts':{'dpo_steps':128}}}),'gpu-training-acceptance.json':raw({'toy':'training accepted'}),
            'engine-terminal.json':raw({'toy':'engine'})})
        for p,stage in (('sft_llm','sft'),('dpo_llm','final')):
            self.data[stage+'.adapter.pt']=p.encode();self.data[stage+'.checkpoint.json']=raw({'checkpoint':p})
        self.old={'schema':'ace-scientific-campaign-cpu-v1','scope':'development','files':{},
            'source_names':sorted(names-g.EXTRA_SOURCES),'cpu_output':str(self.cpu_output),
            'freeze_sha256':self.expected['scientific_freeze_sha256'],'namespace_admission_sha256':'d'*64,
            'runtime':self.service.RUNTIME,'runtime_source':self.d.RUNTIME_SOURCE,
            'archive':{'path':'/toy/archive','sha256':'a'*64},'manifest':{'path':'/toy/manifest','sha256':'b'*64},
            'expected_training':self.expected,'model_directory':'/toy/base','model_files':{'model.fake':'c'*64}}
        settings={};specs=[];self.freezes={};self.contracts={}
        for p in g.POLICIES[5:]:
            stage='sft' if p=='sft_llm' else 'final';role=self.lineage['roles'].get(p)
            freeze={'scope':'development','limits':dict(startup_seconds=120,request_seconds=30,idle_seconds=30,total_seconds=900,max_requests=576),
                'sources':{n:sha(self.data[n]) for n in self.service.SOURCE_NAMES},'model_directory':'/toy/base','model_files':self.old['model_files'],
                'treatment':role['treatment'] if role else 'frozen','adapter':None if not role else dict(file=stage+'.adapter.pt',
                    sha256=role['adapter_sha256'],training_receipt=stage+'.checkpoint.json',training_receipt_sha256=role['checkpoint_sha256'],training_scope='policy_train')}
            name=p+'.freeze.json';self.data[name]=raw(freeze);self.freezes[p]=freeze
            contract=dict(policy_id=p,entry=str(self.cpu_inputs/'llm_intervention_model_service.py'),entry_sha256=sha(self.data['llm_intervention_model_service.py']),
                freeze=str(self.cpu_inputs/name),freeze_sha256=sha(self.data[name]),worker_output=str(self.output/'services'/p/'worker'),
                launch_output=str(self.output/'services'/p/'launch'),binding_path=str(self.output/'stage-binding.json'),stage_path=str(self.output/'stage.json'),
                archive_sha256='a'*64,manifest_sha256='b'*64,bootstrap=self.d.BOOTSTRAP,bootstrap_sha256=sha(self.d.BOOTSTRAP.encode()))
            self.contracts[p]=contract;self.data[p+'.launch.json']=raw(contract)
            cmd=[self.d.RUNTIME_SOURCE+'/bin/python','-I','-B','-c',self.d.BOOTSTRAP,str(self.cpu_inputs/'llm_intervention_campaign_launch.py'),
                sha(self.data['llm_intervention_campaign_launch.py']),'--contract',str(self.cpu_inputs/(p+'.launch.json')),'--contract-sha256',sha(self.data[p+'.launch.json'])]
            identity=self.service.identity(freeze);settings[p]=dict(identity=identity,startup_seconds=125,cleanup_seconds=5.,max_requests=576)
            specs.append(dict(policy_id=p,command=cmd,kind='persistent',timeout_seconds=35,identity_sha256=digest(identity),policy_provenance_sha256=sha(self.data[name])))
        sources=[{'plan':{'world':{'world_id':w},'initial_actions':['toy']*8,'initial_cost_units':8,'acquisition_cap_units':24,
            'policy_specs':copy.deepcopy(specs)},'plan_sha256':'e'*64} for w in g.WORLD_IDS]
        c={'scope':'development','policy_ids':list(g.POLICIES),'policy_phases':{p:'cpu' if i<5 else 'gpu' for i,p in enumerate(g.POLICIES)},'sources':sources,'services':settings}
        self.bundle=dict(scope='development',campaign=c,campaign_sha256=digest(c),source_plan_pins=[dict(world_id=w,sha256='e'*64) for w in g.WORLD_IDS])
        self.attempts={}
        for i,p in enumerate(g.POLICIES[:5]):
            folder=self.campaign/f'policy-{i:02d}';folder.mkdir();value=raw({'toy':p});self.attempts[p]=sha(value)
            for n in ('attempt.json','attempt.candidate.json'):(folder/n).write_bytes(value)
        for n in ('campaign.json','source-plan-pins.json','start.json'):(self.campaign/n).write_bytes(raw({'toy':n}))
        self.terminal=dict(schema=self.old['schema'],status='cpu_phase_complete',attempt_pins=self.attempts,numeric_training_updates=288,
            llm_training_updates=0,completed_cpu_cells=120,declared_cells=192,gpu_services_started=0,private_scores_computed=False,
            scientific_qualified=False,retry_allowed=False,campaign_sha256=self.bundle['campaign_sha256'])
        self.packet=dict(schema=g.VERSION,scope='development',files={},source_names=sorted(names),
            paths=dict(cpu_inputs=str(self.cpu_inputs),cpu_output=str(self.cpu_output),campaign=str(self.campaign),gpu_inputs=str(self.inputs),gpu_output=str(self.output)),
            resources={**g.RESOURCE_BASE,'wall_seconds':2700},limits=dict(session_reserve_seconds=120,session_cap_seconds=900,work_seconds=2640,final_seconds=2670),
            services={p:p+'.freeze.json' for p in g.POLICIES[5:]},contracts={p:p+'.launch.json' for p in g.POLICIES[5:]},
            lineage_files=g.LINEAGE_FILES.copy(),freeze_sha256=self.old['freeze_sha256'],namespace_admission_sha256='d'*64)
        self.acceptance=dict(schema='ace-scientific-campaign-cpu-acceptance-v1',status='development_cpu_phase_accepted',job_id='123',
            allocation=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,state='COMPLETED',exit_code='0:0'),
            cpu_output=str(self.cpu_output),campaign_root=str(self.campaign),cpu_output_excluded_directory='worker/artifacts/campaign',
            bundle_path='worker/artifacts/cpu-phase/campaign-bundle.json',phase_terminal_path='worker/artifacts/cpu-phase/terminal.json',
            attempt_pins=self.attempts,scientific_qualified=False)
        self.repin()
        def lineage_call(**kw):
            self.calls.append('training-lineage');assert kw['expected']==self.expected
            assert kw['acceptance_sha256']==sha(self.data['gpu-training-acceptance.json'])
            return self.lineage
        modules={'llm_intervention_model_service.py':self.service,g.GPU_HELPER:self.gpu,g.PROCESS_HELPER:self.helper,g.DIAGNOSTIC:self.d,
            g.LINEAGE:types.SimpleNamespace(admit_training=lineage_call),'llm_intervention_campaign_launch.py':self.launch,
            'llm_intervention_runtime_archive.py':types.SimpleNamespace()}
        def module(name,path,value):self.calls.append(Path(path).name);return modules[Path(path).name]
        g.module=module
        (root/'TOY-ONLY.json').write_bytes(raw({'actual_acceptance':False,'scientific_descriptors':0,'models':0}))

    def repin(self):
        self.old['files']={n:sha(b) for n,b in self.data.items() if n not in self.g.EXTRA_SOURCES and n not in ('cpu-packet.json','cpu-acceptance.json')}
        self.data['cpu-packet.json']=raw(self.old);self.packet['cpu_packet_sha256']=sha(self.data['cpu-packet.json'])
        (self.phase/'campaign-bundle.json').write_bytes(raw(self.bundle));(self.phase/'terminal.json').write_bytes(raw(self.terminal))
        self.acceptance.update(cpu_packet_sha256=self.packet['cpu_packet_sha256'],bundle_sha256=sha(raw(self.bundle)),phase_terminal_sha256=sha(raw(self.terminal)),
            cpu_output_files=self.g.inventory(self.cpu_output,exclude='worker/artifacts/campaign'),campaign_files=self.g.inventory(self.campaign))
        self.data['cpu-acceptance.json']=raw(self.acceptance);self.packet['cpu_acceptance_sha256']=sha(self.data['cpu-acceptance.json'])
        self.flush()

    def flush(self):
        self.packet['files']={n:sha(b) for n,b in self.data.items()}
        for n,b in self.data.items():(self.inputs/n).write_bytes(b)
        for n in self.old['files']:(self.cpu_inputs/n).write_bytes(self.data[n])
        (self.cpu_inputs/'packet.json').write_bytes(self.data['cpu-packet.json']);(self.inputs/'packet.json').write_bytes(raw(self.packet))
        self.admission=dict(schema=self.g.VERSION+'/admission',packet_sha256=sha(raw(self.packet)),scope='development',
            freeze_sha256=self.packet['freeze_sha256'],namespace_admission_sha256='d'*64,stage='development_gpu_campaign',output=str(self.output),
            resources=self.packet['resources'],limits=self.packet['limits'],full_contract_frozen_before_first_scientific_draw=True,
            source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,independent_acceptances_verified=True,retry_allowed=False,
            site_readiness_sha256='1'*64,accounting_sha256='2'*64)
        (self.root/'admission.json').write_bytes(raw(self.admission))

    def auth(self):
        try:return self.g.authenticate(self.inputs/'packet.json',sha(raw(self.packet)),self.root/'admission.json',sha(raw(self.admission)))
        except BaseException as e:
            (self.root/'rejection.json').write_bytes(raw({'type':type(e).__name__,'message':str(e),'toy_module_calls':self.calls}));raise


class ScientificCampaignOuterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.output=Path(os.environ['ACE_SCIENTIFIC_CAMPAIGN_GPU_TEST_OUTPUT']);cls.output.mkdir()
    def toy(self,name):return Toy(self.output/self._testMethodName/name)

    def test_reviewed_cpu_wrapper_and_owned_helper_are_required_in_original_closure(self):
        t=self.toy('latest_positive');t.auth()
        self.assertEqual(t.packet['files'][t.g.CPU_ENTRY],t.g.CPU_ENTRY_SHA)
        self.assertEqual(t.packet['files'][t.g.CPU_OWNED],t.g.CPU_OWNED_SHA)
        for name in (t.g.CPU_ENTRY,t.g.CPU_OWNED):
            bad=self.toy('changed_'+name);bad.data[name]=b'# stale CPU implementation\n';bad.repin()
            with self.assertRaisesRegex(ValueError,'helper pins'):bad.auth()
            self.assertEqual(bad.calls,[])
        missing=self.toy('helper_not_captured')
        missing.old['source_names'].remove(missing.g.CPU_OWNED);missing.repin()
        with self.assertRaisesRegex(ValueError,'CPU/freeze/namespace'):missing.auth()
        self.assertEqual(missing.calls,[])

    def test_full_authentication_and_inherited_stage_exec_with_576_service_contracts(self):
        t=self.toy('valid');g=t.g;ctx=t.auth();self.assertEqual(len(ctx['bundle']['campaign']['sources']),24)
        t.output.mkdir();scratch=t.root/'scratch';scratch.mkdir();calls=[]
        t.d.events=lambda *a:lambda *a,**kw:None;t.d.stage_runtime=lambda *a:{'staged_prefix':str(scratch/'runtime')}
        t.d.provenance_contract=lambda p:{'toy':str(p)}
        t.launch.derive_command=lambda *a,**kw:calls.append('derive')
        t.gpu.scheduler_record=lambda *a:'toy';t.gpu.child_environment=lambda *a:{};t.gpu.install_child_environment=lambda *a:{'toy':True}
        args=types.SimpleNamespace(output=str(t.output),packet=str(t.inputs/'packet.json'),packet_sha256=sha(raw(t.packet)),
            admission=str(t.root/'admission.json'),admission_sha256=sha(raw(t.admission)),started=100.,process_started=0.)
        class ExecHandoff(BaseException):pass
        def execve(binary,command,environment):
            self.assertIn('--admission-sha256',command);self.assertIn('--controller',command)
            calls.append('exec');raise ExecHandoff()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)}),patch.object(g,'allocation',return_value={'fields':{'JobId':'123'}}),patch.object(g.time,'monotonic',return_value=110.),patch.object(g.os,'execve',side_effect=execve):
            with self.assertRaises(ExecHandoff):g.stage_and_exec(args,ctx)
        self.assertEqual(calls,['derive','derive','derive','exec'])
        handoff=json.loads((t.output/'handoff.json').read_bytes());self.assertEqual(handoff['started_monotonic'],100.)
        command=json.loads((t.output/'controller-command.json').read_bytes())
        self.assertEqual(command['work_deadline_monotonic'],2740.)
        self.assertEqual(command['final_deadline_monotonic'],2770.)
        self.assertFalse((t.output/'terminal.json').exists())

    def test_protected_output_rejected_before_bundle_decode_and_output_claim(self):
        for label,destination in (('runtime','/toy/immutable-runtime/new-output'),('model','/toy/base/new-output'),
                                  ('archive','/toy/archive/new-output'),('manifest','/toy/manifest/new-output'),
                                  ('ancestor','/toy'),('equal','/toy/immutable-runtime'),('alias',None)):
            t=self.toy(label);g=t.g
            if destination is None:
                alias=t.root/'runtime-alias';alias.symlink_to('/toy/immutable-runtime',target_is_directory=True)
                (t.root/'intentional-symlink-fixture.json').write_bytes(raw({'path':str(alias),'target':'/toy/immutable-runtime',
                    'scope':'deliberate negative ancestry fixture; do not dereference during custody'}))
                destination=str(alias/'new-output')
            t.packet['paths']['gpu_output']=destination;t.flush();decoded=[];original=g.decode
            def decode(value):
                if value==raw(t.bundle):decoded.append(True)
                return original(value)
            with patch.object(g,'decode',side_effect=decode):
                with self.assertRaisesRegex(ValueError,'protected input'):t.auth()
            self.assertEqual(decoded,[]);self.assertEqual(t.calls,[])
            with self.assertRaisesRegex(ValueError,'protected input'):
                g.validated_output(t.inputs/'packet.json',sha(raw(t.packet)),destination)
            self.assertFalse(Path(destination).exists())
            if label=='runtime':
                argv=[g.ENTRY,'--packet',str(t.inputs/'packet.json'),'--packet-sha256',sha(raw(t.packet)),
                    '--admission',str(t.root/'admission.json'),'--admission-sha256',sha(raw(t.admission)),
                    '--output',destination]
                with patch.object(g.sys,'argv',argv):
                    with self.assertRaisesRegex(ValueError,'protected input'):g.main()
                self.assertFalse(Path(destination).exists())
        t=self.toy('positive_disjoint');g=t.g;t.auth()
        g.protect_inputs(t.old,t.output,t.root/'new-staging')
        for key in ('runtime_source','model_directory'):
            old=copy.deepcopy(t.old);old[key]='relative/path'
            with self.assertRaisesRegex(ValueError,'absolute protected'):g.protect_inputs(old,t.output)

    def test_protected_staging_rejected_before_mkdir_or_extraction(self):
        for field in ('runtime_source','model_directory','archive','manifest'):
            t=self.toy(field);g=t.g;ctx=t.auth();t.output.mkdir()
            scratch=t.root/'scratch';scratch.mkdir();local=scratch/'ace-scientific-campaign-gpu-123'
            ctx['old']=copy.deepcopy(ctx['old'])
            if field in ('archive','manifest'):ctx['old'][field]['path']=str(local/'protected-file')
            else:ctx['old'][field]=str(local)
            t.gpu.scheduler_record=lambda *a:'toy';t.d.stage_runtime=lambda *a:(_ for _ in ()).throw(AssertionError('extraction must not run'))
            args=types.SimpleNamespace(output=str(t.output))
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)}),patch.object(g,'allocation',return_value={'toy':True}):
                with self.assertRaisesRegex(ValueError,'protected input'):g.stage_and_exec(args,ctx)
            self.assertFalse(local.exists());self.assertFalse((t.output/'stage.json').exists())
            (t.root/'rejected-staging.json').write_bytes(raw({'protected_field':field,'candidate':str(local),'candidate_created':False}))

    def test_post_authentication_input_mutations_block_actual_controller_terminal(self):
        for mutation in ('added_gpu_source','gpu_cpu_copy','gpu_packet','admission','original_cpu_packet','original_cpu_member'):
            t=self.toy(mutation);g=t.g;ctx=t.auth();g.recheck_inputs(ctx);t.output.mkdir()
            g.save(t.output/'independent-provenance-sources.json',{'toy':True})
            args=types.SimpleNamespace(output=str(t.output),started=100.,process_started=0.,
                packet_sha256=sha(raw(t.packet)),admission_sha256=sha(raw(t.admission)))
            handoff={'stage_sha256':'3'*64,'binding_sha256':'4'*64};stage={'staged_prefix':'/toy/stage'};calls=[]
            class Journal:
                def __init__(self,out):self.output=out;out.mkdir(parents=True)
            t.service.Journal=Journal;t.service.read_pinned=lambda *a:None;t.service.decode=g.decode
            projects={Path(n).stem:{'sha256':t.packet['files'][n]} for n in t.packet['source_names'] if n!=g.ENTRY}
            class Observer:
                def __init__(self,*a,**kw):self.journal=a[7]
                def import_cpu_closure(self):
                    self.observe('after-service-imports');g.save(self.journal.output/'runtime-stage-binding.json',{'gpu_mapping_enabled':False})
                def observe(self,phase):g.save(self.journal.output/('runtime-'+phase+'.json'),{'captured_project_modules':projects})
            class Loader:
                def __init__(self,*a):pass
                def find_spec(self,*a):return None
            campaign=types.ModuleType('llm_intervention_campaign')
            campaign._root=lambda c,p,s,r:(c,Path(r));campaign._attempt=lambda root,c,p:root/('policy-%02d'%g.POLICIES.index(p))
            campaign._load_attempt=lambda *a:({'errors':[],'arms':[{'status':'completed'}]*24},None)
            def phase(**kw):
                calls.append('phase');self.assertEqual(kw['limits'],t.packet['limits']);self.assertEqual(kw['work_deadline'],2740.)
                self.assertEqual(kw['final_deadline'],2770.);self.assertEqual(kw['helper_sha256'],g.PROCESS_HELPER_SHA)
                self.assertEqual(set(kw['services']),set(g.POLICIES[5:]));pins={}
                kw['supervision_root'].mkdir()
                for i,pid in enumerate(g.POLICIES[5:],5):
                    folder=t.campaign/f'policy-{i:02d}';folder.mkdir();pins[pid]=g.save(folder/'attempt.json',{'toy':pid})
                    worker=Path(t.contracts[pid]['worker_output']);worker.mkdir(parents=True);g.save(worker/'terminal.json',{'toy':pid})
                    launch=Path(t.contracts[pid]['launch_output']);launch.mkdir()
                    g.save(launch/'execution.json',dict(stage_binding_sha256=handoff['binding_sha256'],contract_sha256=t.packet['files'][pid+'.launch.json']))
                result={'status':'gpu_phase_complete','attempt_pins':pins,'signals_restored':True}
                g.save(kw['supervision_root']/'terminal.json',result);return result
            def validate(**kw):
                calls.append('evidence:'+kw['policy_id']);self.assertEqual(kw['worker_evidence_sha256'],g.inventory(kw['worker_output_dir']))
                return {'operational_success':True,'toy':'saved evidence join'}
            def runtime(*a):
                calls.append('runtime')
                if calls.count('runtime')==3:
                    changed={'added_gpu_source':t.inputs/'llm_intervention_scientific_gpu_phase.py',
                        'gpu_cpu_copy':t.inputs/g.CPU_ENTRY,'gpu_packet':ctx['packet_path'],
                        'admission':ctx['admission_path'],'original_cpu_packet':t.cpu_inputs/'packet.json',
                        'original_cpu_member':t.cpu_inputs/g.CPU_ENTRY}[mutation]
                    changed.write_bytes(changed.read_bytes()+b'\n')
                    (t.root/'mutation.json').write_bytes(raw({'path':str(changed),'after':'third saved runtime validation'}))
                return {'toy':'runtime validated'}
            t.gpu.validate_runtime=runtime
            mods={}
            for name,attrs in {
                'llm_intervention_service_runtime':{'ServiceRuntime':Observer},'llm_intervention_campaign':vars(campaign),
                'llm_intervention_scientific_gpu_phase':{'run_gpu_phase':phase},
                'llm_intervention_scientific_service_evidence':{'validate_campaign_service':validate},
                'llm_intervention_contract':{'digest':digest},
                'qualify_llm_intervention_service_cpu':{'validate_runtime_records':lambda *a:calls.append('controller-runtime')}
            }.items():
                mod=types.ModuleType(name);vars(mod).update(attrs);mods[name]=mod
            with patch.dict(sys.modules,mods),patch.object(g,'CapturedSources',Loader),patch.object(g,'controller_handoff',return_value=(handoff,stage)),patch.object(g.signal,'setitimer'),patch.object(g.time,'monotonic',return_value=110.):
                with self.assertRaisesRegex(ValueError,'pin differs'):g.run_controller(args,ctx)
            self.assertEqual(calls,['phase','controller-runtime','evidence:frozen_llm','runtime',
                'evidence:sft_llm','runtime','evidence:dpo_llm','runtime'])
            self.assertFalse((t.output/'terminal.json').exists())
            self.assertTrue((t.output/'controller/phase-result.candidate.json').exists())
            self.assertEqual(g.inventory(t.cpu_output,exclude='worker/artifacts/campaign'),t.acceptance['cpu_output_files'])
            (t.root/'call-order.json').write_bytes(raw(calls))

    def test_controller_wires_scientific_phase_runtime_and_all_three_evidence_joins(self):
        for bad in (False,True):
            t=self.toy('validator_reject' if bad else 'complete');g=t.g;ctx=t.auth();t.output.mkdir()
            g.save(t.output/'independent-provenance-sources.json',{'toy':True})
            args=types.SimpleNamespace(output=str(t.output),started=100.,process_started=0.,
                packet_sha256=sha(raw(t.packet)),admission_sha256=sha(raw(t.admission)))
            handoff={'stage_sha256':'3'*64,'binding_sha256':'4'*64};stage={'staged_prefix':'/toy/stage'};calls=[]
            class Journal:
                def __init__(self,out):self.output=out;out.mkdir(parents=True)
            t.service.Journal=Journal;t.service.read_pinned=lambda *a:None;t.service.decode=g.decode
            projects={Path(n).stem:{'sha256':t.packet['files'][n]} for n in t.packet['source_names'] if n!=g.ENTRY}
            class Observer:
                def __init__(self,*a,**kw):self.journal=a[7]
                def import_cpu_closure(self):
                    self.observe('after-service-imports');g.save(self.journal.output/'runtime-stage-binding.json',{'gpu_mapping_enabled':False})
                def observe(self,phase):g.save(self.journal.output/('runtime-'+phase+'.json'),{'captured_project_modules':projects})
            class Loader:
                def __init__(self,*a):pass
                def find_spec(self,*a):return None
            campaign=types.ModuleType('llm_intervention_campaign')
            campaign._root=lambda c,p,s,r:(c,Path(r));campaign._attempt=lambda root,c,p:root/('policy-%02d'%g.POLICIES.index(p))
            campaign._load_attempt=lambda *a:({'errors':[],'arms':[{'status':'completed'}]*24},None)
            def phase(**kw):
                calls.append('phase');self.assertEqual(kw['limits'],t.packet['limits']);self.assertEqual(kw['work_deadline'],2740.)
                self.assertEqual(kw['final_deadline'],2770.);self.assertEqual(kw['helper_sha256'],g.PROCESS_HELPER_SHA)
                self.assertEqual(set(kw['services']),set(g.POLICIES[5:]));pins={}
                kw['supervision_root'].mkdir()
                for i,pid in enumerate(g.POLICIES[5:],5):
                    folder=t.campaign/f'policy-{i:02d}';folder.mkdir();pins[pid]=g.save(folder/'attempt.json',{'toy':pid})
                    worker=Path(t.contracts[pid]['worker_output']);worker.mkdir(parents=True);g.save(worker/'terminal.json',{'toy':pid})
                    launch=Path(t.contracts[pid]['launch_output']);launch.mkdir()
                    g.save(launch/'execution.json',dict(stage_binding_sha256=handoff['binding_sha256'],contract_sha256=t.packet['files'][pid+'.launch.json']))
                result={'status':'gpu_phase_complete','attempt_pins':pins,'signals_restored':True}
                g.save(kw['supervision_root']/'terminal.json',result);return result
            def validate(**kw):
                calls.append('evidence:'+kw['policy_id']);self.assertEqual(kw['worker_evidence_sha256'],g.inventory(kw['worker_output_dir']))
                return {'operational_success':not bad,'toy':'saved evidence join'}
            def runtime(*a):calls.append('runtime');return {'toy':'runtime validated'}
            t.gpu.validate_runtime=runtime
            mods={}
            for name,attrs in {
                'llm_intervention_service_runtime':{'ServiceRuntime':Observer},'llm_intervention_campaign':vars(campaign),
                'llm_intervention_scientific_gpu_phase':{'run_gpu_phase':phase},
                'llm_intervention_scientific_service_evidence':{'validate_campaign_service':validate},
                'llm_intervention_contract':{'digest':digest},
                'qualify_llm_intervention_service_cpu':{'validate_runtime_records':lambda *a:calls.append('controller-runtime')}
            }.items():
                mod=types.ModuleType(name);vars(mod).update(attrs);mods[name]=mod
            with patch.dict(sys.modules,mods),patch.object(g,'CapturedSources',Loader),patch.object(g,'controller_handoff',return_value=(handoff,stage)),patch.object(g.signal,'setitimer'),patch.object(g.time,'monotonic',return_value=110.):
                if bad:
                    with self.assertRaisesRegex(ValueError,'service did not complete'):g.run_controller(args,ctx)
                else:g.run_controller(args,ctx)
            self.assertEqual(calls[:2],['phase','controller-runtime'])
            if bad:self.assertFalse((t.output/'terminal.json').exists())
            else:
                self.assertEqual(calls[2:],['evidence:frozen_llm','runtime','evidence:sft_llm','runtime','evidence:dpo_llm','runtime'])
                terminal=g.decode(g.read(t.output/'terminal.json'))
                self.assertEqual(terminal['status'],'development_gpu_campaign_complete')
                self.assertEqual(set(terminal['combined_attempt_pins']),set(g.POLICIES))
                self.assertEqual(terminal['new_training_updates'],0);self.assertFalse(terminal['private_scores_computed'])
                self.assertFalse(terminal['scientific_qualified'])
            self.assertTrue((t.output/'controller/phase-result.candidate.json').exists())
            self.assertEqual(g.inventory(t.cpu_output,exclude='worker/artifacts/campaign'),t.acceptance['cpu_output_files'])
            (t.root/'call-order.json').write_bytes(raw(calls))

    def test_original_cpu_custody_and_authority_reject_before_private_bundle_decode(self):
        for case in ('custody_changed','cpu_status','cpu_failed_cell','wrong_world','wrong_checkpoint'):
            t=self.toy(case)
            if case=='custody_changed':(t.phase/'terminal.json').write_bytes(raw({'changed':True}))
            elif case=='cpu_status':
                t.acceptance['status']='artificial_cpu_phase_accepted';t.data['cpu-acceptance.json']=raw(t.acceptance)
                t.packet['cpu_acceptance_sha256']=sha(t.data['cpu-acceptance.json']);t.flush()
            elif case=='cpu_failed_cell':t.terminal['completed_cpu_cells']=119;t.repin()
            elif case=='wrong_world':t.bundle['campaign']['sources'][0]['plan']['world']['world_id']=0;t.repin()
            else:
                t.freezes['sft_llm']['adapter']['sha256']='0'*64;t.data['sft_llm.freeze.json']=raw(t.freezes['sft_llm']);t.repin()
            original=t.g.decode;decoded_bundle=[]
            def decode(b):
                if b==raw(t.bundle):decoded_bundle.append(True)
                return original(b)
            with patch.object(t.g,'decode',side_effect=decode):
                with self.assertRaises(ValueError):t.auth()
            if case in ('custody_changed','cpu_status','cpu_failed_cell'):self.assertFalse(decoded_bundle)

    def test_resources_admission_limits_and_exact_allocation(self):
        t=self.toy('base');g=t.g;g.resource_contract(t.packet);g.admission_contract(t.packet,t.admission,sha(raw(t.packet)))
        for section,key,bad in [('resources','max_allocated_cpus',8),('resources','wall_seconds',3601),('resources','memory_mib',12288),
            ('limits','final_seconds',2700),('limits','work_seconds',3570),('limits','session_reserve_seconds',119),('limits','session_cap_seconds',1201)]:
            p=copy.deepcopy(t.packet);p[section][key]=bad
            with self.assertRaises(ValueError):g.resource_contract(p)
        a=dict(t.admission,accounting_admitted=False)
        with self.assertRaises(ValueError):g.admission_contract(t.packet,a,sha(raw(t.packet)))
        fields=dict(JobId='123',Account='ucb736_asc2',Partition='aa100',QOS='gpu-testing',JobState='RUNNING',NumNodes='1',NumTasks='1',NumCPUs='4',
            **{'CPUs/Task':'4'},TimeLimit='00:45:00',Requeue='0',Restarts='0',UserId='paco0228(99)',
            AllocTRES='cpu=4,mem=15G,node=1,gres/gpu:a100_3g.20gb=1',ReqTRES='cpu=4,mem=15360M,node=1,gres/gpu:a100_3g.20gb=1')
        def check(f):return g.allocation(' '.join(k+'='+v for k,v in f.items()),'123',t.helper,t.packet['resources'])
        check(fields)
        for key,bad in [('NumCPUs','8'),('ReqTRES','cpu=4,mem=15G,node=1,gres/gpu:a100_3g.20gb=2'),('TimeLimit','00:10:00')]:
            with self.assertRaises(ValueError):check(dict(fields,**{key:bad}))
        (t.root/'negative-cases.json').write_bytes(raw({'resource_cases':7,'allocation_cases':3,'independent_accounting_rejected':True}))

    def test_preserved_cpu_sources_output_claim_and_exact_phase_validator_pins(self):
        t=self.toy('source_changed');(t.cpu_inputs/t.g.CPU_ENTRY).write_bytes(b'# replaced\n')
        with self.assertRaisesRegex(ValueError,'pin differs'):t.auth()
        self.assertEqual(t.calls,[])
        for case,name in [('phase','llm_intervention_scientific_gpu_phase.py'),('validator','llm_intervention_scientific_service_evidence.py')]:
            t=self.toy(case);t.data[name]=b'# substitute';t.flush()
            with self.assertRaisesRegex(ValueError,'helper pins'):t.auth()
        t=self.toy('output');g=t.g
        self.assertEqual(g.validated_output(t.inputs/'packet.json',sha(raw(t.packet)),t.output),t.output)
        with self.assertRaises(ValueError):g.validated_output(t.inputs/'packet.json',sha(raw(t.packet)),t.cpu_output)
        self.assertFalse(t.output.exists())
        t.admission['namespace_owned']=False;(t.root/'admission.json').write_bytes(raw(t.admission))
        with self.assertRaisesRegex(ValueError,'admission'):t.auth()
        self.assertEqual(t.calls,[])


if __name__=='__main__':unittest.main(verbosity=2)
