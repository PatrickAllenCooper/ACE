"""One independently admitted scientific GPU training allocation on saved CPU tokens.

No admission is synthesized here. The caller supplies packet and admission byte
pins; the latter binds an independent CPU acceptance, freeze, namespace, reviewed
source closure, resource/deadline profile, site readiness evidence and output.
The acceptance schema is an input contract for a FUTURE actual CPU result, not
an assertion that such an acceptance currently exists. No tokenization, world
access, resampling, retries, scheduler submission or scientific qualification.

Invoke through the existing pinned diagnostic BOOTSTRAP. Runtime/model checks
and the full stdlib saved-worker validator reuse the accepted shape supervisor.
This source and its artificial tests do not qualify an allocation or execution.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-scientific-training-gpu-supervisor-v1'
ENTRY = 'supervise_llm_intervention_scientific_training_gpu.py'
WORKER = 'llm_intervention_scientific_training_gpu_worker.py'
GPU = 'qualify_llm_intervention_model_service.py'
GPU_SHA = '76accdcd2c24bdc6253a1c131be191a6c6534b52bd5246caaaf66a1f64473e69'
SUPERVISOR = 'supervise_llm_intervention_checkpoint_gpu.py'
SUPERVISOR_SHA = 'ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec'
CPU = 'qualify_llm_intervention_service_cpu.py'
CPU_SHA = 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
DIAGNOSTIC_SHA = '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329'
SOURCE_NAMES = (
    'llm_intervention_action_scoring.py', 'llm_intervention_canonical_scoring.py',
    'llm_intervention_checkpointing.py', 'llm_intervention_contract.py',
    'llm_intervention_episode.py', 'llm_intervention_gpu_mappings.py',
    'llm_intervention_grammar.py', 'llm_intervention_language_policy.py',
    'llm_intervention_gradient_probe.py',
    'llm_intervention_learning.py', 'llm_intervention_model_service.py',
    'llm_intervention_numerical_policy.py', 'llm_intervention_persistent_policy.py',
    'llm_intervention_precision.py', 'llm_intervention_scm.py',
    'llm_intervention_service_runtime.py', 'llm_intervention_supervision.py',
    'llm_intervention_task.py', 'llm_intervention_training_engine.py',
    'llm_intervention_training_exposure.py', 'llm_intervention_training_plan.py',
    'qualify_llm_intervention_checkpoint_gpu.py', CPU, DIAGNOSTIC, GPU,
    SUPERVISOR, WORKER, ENTRY, 'llm_intervention_screen_training_gpu_worker.py',
)
PREPARATION_SOURCES = frozenset({
    'llm_intervention_contract.py', 'llm_intervention_task.py',
    'llm_intervention_numerical_policy.py', 'llm_intervention_scm.py',
    'llm_intervention_episode.py', 'llm_intervention_supervision.py',
    'llm_intervention_training_plan.py', 'llm_intervention_training_engine.py',
    'qualify_llm_intervention_checkpoint_gpu.py', 'llm_intervention_gradient_probe.py',
    'llm_intervention_learning.py', 'llm_intervention_action_scoring.py',
    'llm_intervention_checkpointing.py', 'llm_intervention_canonical_scoring.py',
    'llm_intervention_precision.py', 'llm_intervention_language_policy.py',
    'llm_intervention_grammar.py', 'llm_intervention_training_exposure.py',
})

EXTRA_DATA = {'prepared.json', 'cpu-packet.json', 'cpu-acceptance.json',
              'llm_intervention_runtime_archive.py'}
INDEPENDENT = {GPU:GPU_SHA, SUPERVISOR:SUPERVISOR_SHA, CPU:CPU_SHA, DIAGNOSTIC:DIAGNOSTIC_SHA,
    WORKER:'ab2c999a35b247e03df87dce9f19d771c50f84747381bfbc0ed980e9c832a878',
    'llm_intervention_screen_training_gpu_worker.py':'ecf9d7099d2a038fba01f2eddf93d93680fe675396a7d21d248ad81f2892c53d'}
RESOURCES = dict(account='ucb736_asc2', partition='aa100', qos='gpu-testing',
    requested_cpus=4, max_allocated_cpus=4, memory_mib=15360,
    wall_seconds=2700, gpus=1, gpu_type='a100_3g.20gb')
DEADLINES = dict(startup=240, worker=2620, soft=2630, cleanup=2645, hard=2670,
    progress=180, telemetry=90)
MAX_PREPARED = 256*1024**2
CPU_ACCEPTANCE_SCHEMA = 'ace-scientific-training-preparation-acceptance-v1'
ADMISSION_SCHEMA = 'ace-scientific-training-gpu-admission-v1'

def need(ok,message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path,maximum=32*1024**2):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'bounded regular input required')
        raw=f.read(maximum+1)
        after=os.fstat(f.fileno())
    need(len(raw)<=maximum and (before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_ino,after.st_size,after.st_mtime_ns),'input changed while reading')
    return raw


def capture(path,pin,maximum=8*1024**2):
    raw=read_bounded(path,maximum)
    need(type(pin) is str and len(pin)==64 and sha(raw)==pin,'independent input pin differs')
    return raw


def module(name,path,raw):
    m=types.ModuleType(name)
    m.__file__=str(path)
    exec(compile(raw,str(path),'exec'),vars(m))
    return m


def save(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return sha(raw)



def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def decode(raw):
    def pairs(items):
        out={}
        for key,value in items:
            need(key not in out,'duplicate JSON key')
            out[key]=value
        return out
    value=json.loads(raw,object_pairs_hook=pairs)
    canonical(value)
    return value


def exact(actual,expected):
    # JSON comparison also rejects bools where integer resources are required.
    return canonical(actual)==canonical(expected)


def packet_shape(packet):
    keys={'schema','files','source_names','resources','scope','source_commit',
          'archive','manifest','model_directory','model_files','runtime','runtime_source',
          'deadlines','plan_sha256','freeze_sha256','namespace_admission_sha256'}
    need(type(packet) is dict and set(packet)==keys and packet['schema']==VERSION
        and packet['scope']=='policy_train' and exact(packet['resources'],RESOURCES)
        and exact(packet['deadlines'],DEADLINES)
        and type(packet['source_commit']) is str and re.fullmatch('[0-9a-f]{40}',packet['source_commit'])
        and packet['source_names']==sorted(SOURCE_NAMES)
        and type(packet['files']) is dict and set(packet['files'])==set(SOURCE_NAMES)|EXTRA_DATA
        and all(pin(p) for p in packet['files'].values())
        and all(pin(packet[n]) for n in ('plan_sha256','freeze_sha256','namespace_admission_sha256')),
        'complete scientific packet contract differs')
    need(all(packet['files'][n]==p for n,p in INDEPENDENT.items()),'reviewed helper/worker source differs')


def validate_admission(admission,packet,packet_pin):
    expected={'schema':ADMISSION_SCHEMA,'status':'admitted','scope':'policy_train',
        'packet_sha256':packet_pin,'cpu_acceptance_sha256':packet['files']['cpu-acceptance.json'],
        'prepared_sha256':packet['files']['prepared.json'],'plan_sha256':packet['plan_sha256'],
        'freeze_sha256':packet['freeze_sha256'],'namespace_admission_sha256':packet['namespace_admission_sha256'],
        'resources':RESOURCES,'deadlines':DEADLINES,'required_findings':0,
        'source_runtime_scheduler_admitted':True,'scientific_protocol_frozen':True,'namespace_owned':True}
    need(type(admission) is dict and set(admission)==set(expected)|{'output','site_readiness_sha256'}
        and all(exact(admission.get(k),v) for k,v in expected.items())
        and pin(admission['site_readiness_sha256']) and type(admission['output']) is str
        and Path(admission['output']).is_absolute(),'independent scientific admission differs')


def validate_prepared(prepared,packet,data):
    cpu_packet=decode(data['cpu-packet.json'])
    need(cpu_packet['schema']=='ace-scientific-training-preparation-cpu-v1'
        and cpu_packet['scope']=='policy_train','original scientific CPU packet required')
    accepted=decode(data['cpu-acceptance.json'])
    expected={'schema':CPU_ACCEPTANCE_SCHEMA,'status':'accepted','scope':'policy_train','required_findings':0,
        'cpu_packet_sha256':packet['files']['cpu-packet.json'],
        'prepared_sha256':packet['files']['prepared.json'],'plan_sha256':packet['plan_sha256'],
        'freeze_sha256':packet['freeze_sha256'],'namespace_admission_sha256':packet['namespace_admission_sha256'],
        'complete_custody_verified':True,'runtime_qualified_for_this_cpu_execution':True,
        'owned_process_cleanup_verified':True,'training_execution_qualified':False,'scientific_qualified':False,
        'exposure_sha256':prepared['exposure_sha256']}
    # Numerical bytes/semantic identity are checked by the numerical consumer;
    # retain its required pins in the shared acceptance without loading its plan.
    numerical_pins={'numerical_plan_file_sha256','numerical_plan_sha256'}
    need(type(accepted) is dict and set(accepted)==set(expected)|{'job_id'}|numerical_pins
        and all(exact(accepted.get(k),v) for k,v in expected.items())
        and all(pin(accepted[n]) for n in numerical_pins|{'exposure_sha256'})
        and type(accepted['job_id']) is str and re.fullmatch('[1-9][0-9]*',accepted['job_id']),
        'independent actual scientific CPU acceptance differs')
    for key in ('archive','manifest','model_files','runtime','runtime_source'):
        need(exact(packet[key],cpu_packet[key]),'accepted CPU runtime/model input changed: '+key)
    # The CPU tokenizer-only directory intentionally differs from the GPU full model directory.
    cpu_sources=cpu_packet['source_names']
    need(type(cpu_sources) is list and len(cpu_sources)==len(set(cpu_sources))
        and PREPARATION_SOURCES<=set(cpu_sources)
        and all(cpu_packet['files'][n]==packet['files'][n] for n in set(cpu_sources)&set(SOURCE_NAMES)),
        'accepted CPU/training shared source closure differs')
    need(packet['files']['llm_intervention_runtime_archive.py']==cpu_packet['files']['llm_intervention_runtime_archive.py'],
        'accepted runtime archive source differs')
    need(cpu_packet['collection']['freeze_sha256']==packet['freeze_sha256'], 'collection freeze differs')
    need(data['prepared.json']==(canonical(prepared)+'\n').encode()
        and prepared['training_plan_sha256']==packet['plan_sha256']==sha(canonical(prepared['training_plan']).encode())
        and prepared['training_plan']['outer_provenance_receipt_sha256']==packet['files']['cpu-packet.json']
        and prepared['tokenizer_sha256']==sha(canonical(packet['model_files']).encode()),
        'original prepared bytes, plan or model tokenizer identity differs')
    for key,name in (('collection_terminal_sha256','terminal.json'),('collection_inventory_sha256','inventory.json'),
                     ('collection_plan_sha256','plan.json')):
        need(prepared[key]==cpu_packet['collection']['files'][name],'original collection lineage differs: '+key)
    records=prepared['training_plan']['records']
    need(len(records)==160 and all(0<len(r['prompt_ids'])<=4744 for r in records),
        'actual saved prompt exceeds measured 4744-token resource envelope; no truncation permitted')
    return cpu_packet


def authenticate(path,packet_pin,admission_path,admission_pin):
    root=Path(path).absolute().parent
    packet=decode(capture(path,packet_pin,256*1024))
    packet_shape(packet)
    admission=decode(capture(admission_path,admission_pin,128*1024))
    validate_admission(admission,packet,packet_pin)
    # Capture every input before executing even stdlib helper source.
    data={n:capture(root/n,p,MAX_PREPARED if n=='prepared.json' else 8*1024**2)
          for n,p in packet['files'].items()}
    prepared=decode(data['prepared.json'])
    validate_prepared(prepared,packet,data)
    need(Path(__file__).absolute()==root/ENTRY,'frozen entrypoint must be beside packet')
    service=module('pinned_scientific_service',root/'llm_intervention_model_service.py',data['llm_intervention_model_service.py'])
    cpu=module('pinned_scientific_cpu',root/CPU,data[CPU])
    d=module('pinned_scientific_diagnostic',root/DIAGNOSTIC,data[DIAGNOSTIC])
    gpu=module('pinned_scientific_gpu',root/GPU,data[GPU])
    helper=module('pinned_scientific_supervisor',root/SUPERVISOR,data[SUPERVISOR])
    # Full pure roster/exposure/recipe validation precedes staging and heavy imports.
    loader=CapturedSources({n:data[n] for n in packet['source_names']},root)
    sys.meta_path.insert(0,loader)
    try:
        worker=importlib.import_module(Path(WORKER).stem)
        worker._plan(prepared,prepared_sha256=packet['files']['prepared.json'],plan_sha256=packet['plan_sha256'])
    finally:
        sys.meta_path.remove(loader)
        for name,mod in loader.loaded.items():
            if sys.modules.get(name) is mod:del sys.modules[name]
    return root,packet,data,packet,prepared,service,cpu,d,gpu,helper


def authenticate_allocation(raw,job_id,resources,helper):
    need(exact(resources,RESOURCES) and re.fullmatch('[1-9][0-9]*',job_id),'fixed resources/job required')
    fields=helper.scheduler_fields(raw)
    expected={'JobId':job_id,'Account':'ucb736_asc2','Partition':'aa100','QOS':'gpu-testing',
        'JobState':'RUNNING','NumNodes':'1','NumTasks':'1','TimeLimit':'00:45:00',
        'Requeue':'0','Restarts':'0','NumCPUs':'4','CPUs/Task':'4'}
    need(all(fields.get(k)==v for k,v in expected.items()) and fields.get('UserId','').startswith('paco0228('),
        'scheduler identity/state/allocation differs')
    def tres(text):
        result={}
        for pair in text.split(','):
            need(pair.count('=')==1,'invalid resource evidence')
            key,value=pair.split('=');need(key not in result,'duplicate resource');result[key]=value
        need(result.get('cpu')=='4' and helper.memory_mib(result.get('mem',''))==15360
            and result.get('node')=='1' and result.get('gres/gpu:a100_3g.20gb')=='1'
            and result.get('gres/gpu','1')=='1'
            and not any(k.startswith('gres/') and k not in ('gres/gpu','gres/gpu:a100_3g.20gb') for k in result),
            'actual/requested CPU memory MIG resources differ')
        return result
    return {'fields':fields,'allocated_resources':tres(fields.get('AllocTRES','')),
            'requested_resources':tres(fields.get('ReqTRES',''))}


def positive_owned_memory(sample,pid):
    if sample.get('returncode')!=0 or type(sample.get('owned_rows')) is not list or len(sample['owned_rows'])!=1:
        return False
    row=sample['owned_rows'][0].split(',')
    if len(row)!=3 or row[0].strip()!=str(pid) or not row[1].strip():return False
    try:memory=float(row[2].strip())
    except ValueError:return False
    return math.isfinite(memory) and memory>0


class ProgressWatch:
    """Completed per-record score events count; telemetry is not utilization %."""
    def __init__(self,now,progress,pid):
        self.last_gpu=self.last_progress=now
        self.progress=set(progress);self.pid=pid
    def observe(self,now,sample,progress):
        if positive_owned_memory(sample,self.pid):self.last_gpu=now
        new=set(progress)
        if new-self.progress:self.last_progress=now
        self.progress.update(new)
        if now-self.last_gpu>DEADLINES['telemetry']:
            raise TimeoutError('no reliable positive owned GPU memory for 90 seconds; utilization uncertain')
        if now-self.last_progress>DEADLINES['progress']:
            raise TimeoutError('no completed training/inference operation for 180 seconds')
        return {'last_gpu_monotonic':self.last_gpu,'last_progress_monotonic':self.last_progress,
                'progress':sorted(self.progress),'progress_bound_seconds':180,'telemetry_bound_seconds':90}


def arm_hard_deadline(start,current):
    """Reserve absolute cleanup/reporting time; never call the old 580s alarm."""
    def final(sig,frame):
        try:
            if current is not None and current.pid is not None:
                try:os.killpg(current.pid,signal.SIGKILL)
                except OSError:pass  # Exit even if kill authority cannot be exercised.
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,final)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL,max(.001,start+DEADLINES['hard']-time.monotonic()))


class OwnedChild:
    """Live handle retained through top-level failure unwinding, never a PID receipt."""
    def __init__(self,start,helper):
        self.start,self.helper,self.process=start,helper,None

    @property
    def pid(self):
        return None if self.process is None else self.process.pid

    def confirm_closed(self):
        if self.process is None:return True
        try:
            exited=self.process.poll() is not None
            alive=self.helper.group_alive(self.process.pid)
        except BaseException:
            return False  # Failed observation is uncertainty, not closure.
        if exited and alive is False:
            self.process=None  # Retire kill authority only after actual group exit.
            return True
        return False

    def hold_until_closed(self):
        if self.confirm_closed():return
        arm_hard_deadline(self.start,self)
        while not self.confirm_closed():
            # Keep the controller alive even when cleanup/publication failed.
            # The alarm retains the ORIGINAL deadline; this is not another grace period.
            try:
                if time.monotonic()>=self.start+DEADLINES['hard']:
                    try:os.killpg(self.pid,signal.SIGKILL)
                    except OSError:pass
                    finally:os._exit(124)
                time.sleep(.1)
            except Exception:
                continue


def validate_handoff(handoff,args,now,parent_pid,job_id):
    need(set(handoff)=={'packet_sha256','admission_sha256','runtime_stage_sha256','parent_pid','job_id','started','deadlines'}
        and handoff['packet_sha256']==args.packet_sha256 and handoff['admission_sha256']==args.admission_sha256
        and handoff['runtime_stage_sha256']==args.runtime_stage_sha256
        and type(handoff['parent_pid']) is int and handoff['parent_pid']==parent_pid
        and handoff['job_id']==job_id and re.fullmatch('[1-9][0-9]*',job_id)
        and type(handoff['started']) in (int,float) and math.isfinite(handoff['started'])
        and 0<handoff['started']<=now and exact(handoff['deadlines'],DEADLINES)
        and now<handoff['started']+DEADLINES['worker'],'inherited absolute worker deadline/parent differs')
    return handoff['started']+DEADLINES['worker']

class CapturedSources(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self,captured,root):
        self.root,self.loaded=Path(root),{}
        self.sources={Path(n).stem:b for n,b in captured.items() if n!=ENTRY}
        need(not set(self.sources).intersection(sys.modules),'project imports preceded authentication')
    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/(fullname+'.py')))
        if fullname.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_')):
            raise ImportError('outside captured project closure: '+fullname)
        return None
    def create_module(self,spec):
        return None
    def exec_module(self,mod):
        mod.__file__=str(self.root/(mod.__name__+'.py'))
        exec(compile(self.sources[mod.__name__],mod.__file__,'exec'),vars(mod))
        self.loaded[mod.__name__]=mod


def child(args,root,packet,data,old,prepared,service,cpu,d,gpu,helper):
    journal=service.Journal(Path(args.output)/'artifacts')
    captured={n:data[n] for n in packet['source_names']}
    loader=CapturedSources(captured,root)
    sys.meta_path.insert(0,loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer=ServiceRuntime(old,captured,loader,root/ENTRY,globals(),args.runtime_stage,
            args.runtime_stage_sha256,journal,service.read_pinned,service.decode,allow_gpu_mappings=True)
        model=Path(old['model_directory'])
        need(not model.is_symlink() and model.is_dir() and {p.name for p in model.iterdir()}==service.MODEL_NAMES,
            'exact model directory required')
        models={n:service.read_pinned(model/n,p) for n,p in old['model_files'].items()}
        journal.save('authenticated.json',{'packet_sha256':args.packet_sha256,'models':models,
            'source_sha256':{n:packet['files'][n] for n in packet['source_names']},
            'prepared_sha256':packet['files']['prepared.json'],'cpu_acceptance_sha256':packet['files']['cpu-acceptance.json']})
        observer.import_cpu_closure()
        from llm_intervention_scientific_training_gpu_worker import run_training,validate_outputs
        summary=run_training(prepared,model,journal.output/'fit',observer,journal,prepared_sha256=packet['files']['prepared.json'],plan_sha256=packet['plan_sha256'])
        checked=validate_outputs(journal.output/'fit',prepared,prepared_sha256=packet['files']['prepared.json'],plan_sha256=packet['plan_sha256'])
        journal.save('terminal.json',{'schema':VERSION,'status':'scientific_training_complete',
            'packet_sha256':args.packet_sha256,'runtime_stage_sha256':args.runtime_stage_sha256,
            'summary':summary,'saved_validation':checked,'environment_responses':0,'scientific_qualified':False})
    except BaseException as exc:
        journal.save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        sys.meta_path.remove(loader)


def validate_worker(out,packet,packet_pin,data,old,prepared,service,cpu,d,stage,stage_pin,provenance):
    # The runtime validator imports the mapping module. Execute only the same
    # captured bytes, never an ambient project module from sys.path/sys.modules.
    loader=CapturedSources({n:data[n] for n in packet['source_names']},Path(__file__).absolute().parent)
    sys.meta_path.insert(0,loader)
    try:
        return _validate_worker(out,packet,packet_pin,data,old,prepared,service,cpu,d,stage,stage_pin,provenance)
    finally:
        sys.meta_path.remove(loader)
        for name,mod in loader.loaded.items():
            if sys.modules.get(name) is mod:del sys.modules[name]


def consistent_summary(terminal,outer_summary,fit_raw,checked):
    need(sha(fit_raw)==checked['summary_sha256'],'validated fit summary bytes differ')
    fit_summary=json.loads(fit_raw)
    need(terminal['summary']==outer_summary==fit_summary,'outer summaries differ from independently validated fit')


def _validate_worker(out,packet,packet_pin,data,old,prepared,service,cpu,d,stage,stage_pin,provenance):
    cache={}
    def read(name):
        if name not in cache:
            cache[name]=service.decode(read_bounded(out/name))
        return cache[name]
    need(not list(out.rglob('failure.json')),'failed worker cannot qualify')
    terminal=read('terminal.json')
    worker=module('pinned_training_worker',out/WORKER,data[WORKER])
    checked=worker.validate_outputs(out/'fit',prepared,prepared_sha256=packet['files']['prepared.json'],plan_sha256=packet['plan_sha256'])
    consistent_summary(terminal,read('worker-summary.json'),read_bounded(out/'fit/summary.json'),checked)
    need(terminal['schema']==VERSION and terminal['status']=='scientific_training_complete'
        and terminal['packet_sha256']==packet_pin and terminal['runtime_stage_sha256']==stage_pin
        and terminal['saved_validation']==checked and terminal['summary']==read('worker-summary.json')
        and terminal['environment_responses']==0 and terminal['scientific_qualified'] is False,'worker terminal differs')
    auth=read('authenticated.json')
    need(auth['packet_sha256']==packet_pin and auth['prepared_sha256']==packet['files']['prepared.json']
        and auth['cpu_acceptance_sha256']==packet['files']['cpu-acceptance.json']
        and auth['source_sha256']=={n:packet['files'][n] for n in packet['source_names']}
        and set(auth['models'])==service.MODEL_NAMES
        and all(r['sha256']==old['model_files'][n] and type(r['bytes']) is int and r['bytes']>0
            for n,r in auth['models'].items()),'authenticated source/model/token lineage differs')
    checks=module('pinned_training_runtime',out/'llm_intervention_service_runtime.py',data['llm_intervention_service_runtime.py'])
    need(read('runtime-stage-binding.json')['gpu_mapping_enabled'] is True,'GPU mapping contract absent')
    for phase in ('after-model-load','after-requests'):
        def mapped(name):
            if name!='runtime-after-cpu-tokenization.json':return read(name)
            return checks.checked_gpu_snapshot(read('runtime-'+phase+'.json'),
                read('gpu-mapping-inventory-'+phase+'.json'),read('gpu-mapping-validation-'+phase+'.json'),
                phase,Path(stage['staged_prefix']),d)
        cpu.validate_runtime_records(mapped,stage,stage_pin,service,d,checks,provenance)
    expected={Path(n).stem:packet['files'][n] for n in packet['source_names'] if n!=ENTRY}
    for phase in ('after-service-imports','after-model-load','after-requests'):
        projects=read('runtime-'+phase+'.json')['captured_project_modules']
        need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'}
            and all(projects[n]['sha256']==p for n,p in expected.items()),'full captured project import closure differs')
    phases=[read(p.name) for p in sorted(out.glob('phase-*.json'))]
    names=[r['phase'] for r in phases]
    required=('service_cpu_import_start','service_cpu_import_complete','training_start','training_complete')
    need(all(names.count(n)==1 for n in required) and [names.index(n) for n in required]==sorted(names.index(n) for n in required),
        'ordered training phases differ')
    checkpoints=[(i,r.get('checkpoint')) for i,r in enumerate(phases) if r['phase']=='checkpoint_inference_complete']
    need([r[1] for r in checkpoints]==['sft','final']
        and all(i>names.index('training_complete') for i,_ in checkpoints),'both checkpoint inference phases required')
    need(all(type(r['elapsed_seconds']) in (int,float) and math.isfinite(r['elapsed_seconds']) for r in phases)
        and [r['elapsed_seconds'] for r in phases]==sorted(r['elapsed_seconds'] for r in phases),'phase clocks differ')
    return {'worker_terminal_sha256':sha(read_bounded(out/'terminal.json')),'saved_validation':checked,
        'runtime_observations':['after-service-imports','after-model-load','after-requests']}


def progress_records(out):
    # Live partial writes are not accepted as progress. These names are not a
    # scientific result or terminal qualification. The final reporter is strict.
    rows=[]
    paths=list(out.glob('phase-*.json'))+list((out/'fit/training').glob('step-*.complete.json'))
    paths+=list((out/'fit').glob('*-generation-*.json'))
    for path in sorted(paths):
        try:
            row=json.loads(read_bounded(path,1024**2))
        except (FileNotFoundError,ValueError):continue
        if type(row) is dict:rows.append(str(path.relative_to(out)))
    return tuple(rows)


def positive_work(path):
    # A live reader can race an exclusive write. Terminal validation is strict.
    try:
        row=json.loads(read_bounded(path,64*1024))
    except (FileNotFoundError,ValueError):
        return False
    return (type(row.get('cuda_interval_ms')) in (int,float) and math.isfinite(row['cuda_interval_ms'])
        and row['cuda_interval_ms']>0 and type(row.get('peak_gpu_allocated_bytes')) is int
        and row['peak_gpu_allocated_bytes']>0)


def clean_success(cleanup):
    need(cleanup.get('leader_returncode')==0 and cleanup.get('group_still_exists') is False
        and cleanup.get('term_sent') is False and cleanup.get('kill_sent') is False
        and cleanup.get('errors')==[],'natural exit and clean owned group required')


def parent(args,root,packet,data,old,prepared,service,cpu,d,gpu,helper):
    out=Path(args.output).absolute()
    start,process_start=args.started,args.process_started
    current=None
    owned=args.owned_child=OwnedChild(start,helper)
    gpu_seen=False
    verified_at=None
    watch=None
    child_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    def interrupted(sig,frame):
        arm_hard_deadline(start,owned)
        raise TimeoutError('training supervisor signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.01,start+240-time.monotonic()))
    try:
        save(out/'packet.json',packet)
        job=os.environ.get('SLURM_JOB_ID','')
        scheduler=gpu.scheduler_record(out,job)
        save(out/'allocation.json',authenticate_allocation(scheduler,job,packet['resources'],helper))
        scratch=Path(os.environ.get('SLURM_SCRATCH',''))
        need(scratch.is_absolute(),'absolute node scratch required')
        scratch=scratch.resolve(strict=True)
        need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
        local=scratch/('ace-scientific-training-'+job)
        local.mkdir(mode=0o700,exist_ok=False)
        archive=module('training_archive',root/'llm_intervention_runtime_archive.py',data['llm_intervention_runtime_archive.py'])
        event=d.events(out,start,process_start,512*1024**2)
        event('stage_start')
        stage=d.stage_runtime(old,archive,local/'runtime',event)
        stage_pin=save(out/'stage.json',stage)
        event('stage_complete',staged_prefix=stage['staged_prefix'])
        prefix=Path(stage['staged_prefix'])
        provenance=d.provenance_contract(prefix)
        save(out/'independent-provenance-sources.json',provenance)
        environment=gpu.child_environment(prefix,os.environ)
        environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        save(out/'child-environment.json',gpu.install_child_environment(environment))
        session=out/'session';session.mkdir(mode=0o700,exist_ok=False)
        handoff_pin=save(out/'handoff.json',{'packet_sha256':args.packet_sha256,
            'admission_sha256':args.admission_sha256,'runtime_stage_sha256':stage_pin,
            'parent_pid':os.getpid(),'job_id':job,'started':start,'deadlines':DEADLINES})
        command=[str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(root/ENTRY),packet['files'][ENTRY],
            '--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,
            '--output',str(session/'worker'),'--worker','--runtime-stage',str(out/'stage.json'),
            '--runtime-stage-sha256',stage_pin,
            '--admission',str(Path(args.admission).absolute()),'--admission-sha256',args.admission_sha256,
            '--handoff',str(out/'handoff.json'),'--handoff-sha256',handoff_pin]
        save(out/'child-command.json',{'command':command,'first_gpu_deadline_from_start_seconds':240,
            'absolute_worker_deadline':start+DEADLINES['worker'],'absolute_supervisor_deadline':start+DEADLINES['hard']})
        with (session/'stdout.log').open('xb') as stdout,(session/'stderr.log').open('xb') as stderr:
            current=subprocess.Popen(command,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            owned.process=current
            save(out/'child-start.json',{'pid':current.pid,'elapsed_seconds':time.monotonic()-start})
            ordinal=0
            while current.poll() is None:
                sample=helper.gpu_process_sample(current.pid)
                save(out/f'gpu-sample-{ordinal:04d}.json',{'elapsed_seconds':time.monotonic()-start,**sample})
                ordinal+=1
                if positive_owned_memory(sample,current.pid):gpu_seen=True
                if verified_at is None and positive_owned_memory(sample,current.pid) and positive_work(session/'worker/artifacts/first-gpu-work.json'):
                    verified_at=time.monotonic()-start
                    need(verified_at<240,'first GPU work exceeded startup bound')
                    save(out/'first-verified-gpu-work.json',{'elapsed_seconds':verified_at,'owned_process_observed':True})
                    signal.setitimer(signal.ITIMER_REAL,max(.01,start+DEADLINES['soft']-time.monotonic()))
                    watch=ProgressWatch(time.monotonic(),progress_records(session/'worker/artifacts'),current.pid)
                if watch is not None:
                    evidence=watch.observe(time.monotonic(),sample,progress_records(session/'worker/artifacts'))
                    save(out/f'progress-{ordinal:04d}.json',evidence)
                if time.monotonic()>start+(DEADLINES['soft'] if verified_at is not None else DEADLINES['startup']):
                    raise TimeoutError('bounded training child deadline exceeded')
                time.sleep(1)
        arm_hard_deadline(start,owned)
        cleanup=helper.terminate(current,min(time.monotonic()+12,start+DEADLINES['cleanup']))
        save(out/'cleanup.json',cleanup)
        need(owned.confirm_closed(),'owned process group exit not confirmed')
        clean_success(cleanup)
        need(gpu_seen and verified_at is not None,'independent GPU work not verified during child')
        sample=helper.gpu_process_sample(current.pid)
        save(out/'post-exit-gpu.json',sample)
        gpu.require_gpu_sample(sample,active=False)
        validation=validate_worker(session/'worker/artifacts',packet,args.packet_sha256,data,old,prepared,
            service,cpu,d,stage,stage_pin,provenance)
        need(time.monotonic()<start+DEADLINES['hard'],'absolute supervisor deadline exceeded')
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out/'terminal.json',{'schema':VERSION,'status':'scientific_training_complete',
            'packet_sha256':args.packet_sha256,'admission_sha256':args.admission_sha256,'stage_sha256':stage_pin,'validation':validation,
            'first_verified_gpu_seconds':verified_at,'gpu_observation_count':ordinal,'cleanup':cleanup,
            'elapsed_seconds':time.monotonic()-start,'supervisor_cpu_seconds':time.process_time()-process_start,
            'child_and_helpers_cpu_seconds':usage.ru_utime+usage.ru_stime-child_usage.ru_utime-child_usage.ru_stime,
            'child_peak_rss_bytes':usage.ru_maxrss*1024,'environment_responses':0,'scientific_qualified':False,
            'external_scheduler_and_custody_acceptance_required':True})
    except BaseException as exc:
        arm_hard_deadline(start,owned)
        preserve_failure(out,exc,current,helper,start)
        raise



def preserve_failure(out,exc,current,helper,start):
    # Publish the original error before secondary cleanup/publication can fail.
    first={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    try:save(out/'failure.json',first)
    except BaseException:
        # Retain the first error on the already captured stderr if disk publication fails.
        try:
            print(json.dumps({'original_failure':first},sort_keys=True),file=sys.stderr,flush=True)
            traceback.print_exc(file=sys.stderr)
        except BaseException:pass
    if current is not None and not (out/'cleanup.json').exists():
        try:save(out/'cleanup.json',helper.terminate(current,min(time.monotonic()+12,start+DEADLINES['cleanup'])))
        except BaseException as secondary:
            try:save(out/'cleanup-failure.json',{'type':type(secondary).__name__,'message':str(secondary)})
            except BaseException:traceback.print_exc(file=sys.stderr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','packet-sha256','admission','admission-sha256','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--worker',action='store_true')
    for n in ('runtime-stage','runtime-stage-sha256','handoff','handoff-sha256'):p.add_argument('--'+n)
    args=p.parse_args()
    args.started,args.process_started=time.monotonic(),time.process_time()
    out=Path(args.output).absolute()
    # Resolve before publication, so aliases cannot overwrite the immutable input tree.
    inputs=Path(args.packet).resolve(strict=True).parent
    target=out.resolve()
    need(target!=inputs and inputs not in target.parents and target not in inputs.parents,
        'output must be disjoint from captured input tree')
    out.mkdir(mode=0o700,exist_ok=False)
    def interrupted(sig,frame):raise TimeoutError('scientific training authentication/worker signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        deadline=args.started+DEADLINES['startup']
        if args.worker:
            need(args.handoff and args.handoff_sha256 and args.runtime_stage and args.runtime_stage_sha256,
                 'independent parent handoff/stage required')
            handoff=decode(capture(args.handoff,args.handoff_sha256,128*1024))
            deadline=validate_handoff(handoff,args,time.monotonic(),os.getppid(),os.environ.get('SLURM_JOB_ID',''))
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
        context=authenticate(args.packet,args.packet_sha256,args.admission,args.admission_sha256)
        admission=decode(capture(args.admission,args.admission_sha256,128*1024))
        expected=Path(admission['output']).absolute()
        if args.worker:expected=expected/'session/worker'
        need(out==expected,'admitted exclusive output differs')
        if args.worker:child(args,*context)
        else:parent(args,*context)
    except BaseException as exc:
        if not (out/'failure.json').exists():
            try:save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            except BaseException:
                try:traceback.print_exc(file=sys.stderr)
                except BaseException:pass
        raise
    finally:
        owned=getattr(args,'owned_child',None)
        if owned is not None:owned.hold_until_closed()
        # Reached only without a child or after live leader/group closure.
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
