"""Supervise new full-shape unpadded SFT/DPO qualification on saved CPU tokens.

CPU preparation is independently pinned. No tokenizer, simulator, scientific
worlds or prior adapter is used. Full custody and Slurm acceptance stay external.
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
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-full-shape-singleton-training-gpu-v1'
ENTRY = 'qualify_llm_intervention_screen_training_gpu.py'
WORKER = 'llm_intervention_screen_training_gpu_worker.py'
GPU = 'qualify_llm_intervention_model_service.py'
GPU_SHA = '76accdcd2c24bdc6253a1c131be191a6c6534b52bd5246caaaf66a1f64473e69'
SUPERVISOR = 'supervise_llm_intervention_checkpoint_gpu.py'
SUPERVISOR_SHA = 'ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec'
CPU = 'qualify_llm_intervention_service_cpu.py'
CPU_SHA = 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
DIAGNOSTIC_SHA = '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329'
CAPACITY_PACKET_SHA = '0979cb607643b4b5d01fbf56c6e701a3e474d4d2bbda1e1c9f4e78e91251ccfa'
TOKENS_SHA = 'fb663a035eef20420fe461bc4dfa4fa6afa8462b38e32e1262c5661f666ff72c'
# Pins bind actual saved-token preparation. Distinct input review precedes launch.
CPU_ACCEPTANCE_SHA = '69446fb29e24c34cff070d2f9f699a5d1d8d3e9a478c9d438adc01be7a290e59'
PREPARED_SHA = '6ec85a1608d3e64c914b51b81605248be1a11dfbe879e0d463a9cc09b4da3712'
PLAN_SHA = '0807b705f2bad022630629629e99a36f519b7e0f9e980464ec9a7b3385c10ded'
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
    SUPERVISOR, WORKER, ENTRY,
)
EXTRA_DATA = {'prepared.json', 'cpu-acceptance.json', 'capacity-packet.json',
              'tokenization.json', 'llm_intervention_runtime_archive.py'}
INDEPENDENT = {GPU:GPU_SHA, SUPERVISOR:SUPERVISOR_SHA, CPU:CPU_SHA,
    DIAGNOSTIC:DIAGNOSTIC_SHA, 'cpu-acceptance.json':CPU_ACCEPTANCE_SHA,
    'prepared.json':PREPARED_SHA, 'capacity-packet.json':CAPACITY_PACKET_SHA,
    'tokenization.json':TOKENS_SHA}


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


def packet_shape(packet,gpu):
    fields = {'schema','files','source_names','resources','scope','source_commit',
              'archive','manifest','model_directory','model_files','runtime','runtime_source'}
    need(type(packet) is dict and set(packet) == fields and packet['schema'] == VERSION
         and packet['scope'] == 'artificial' and packet['resources'] == gpu.RESOURCES
         and type(packet['source_commit']) is str and len(packet['source_commit']) == 40
         and all(c in '0123456789abcdef' for c in packet['source_commit'])
         and packet['source_names'] == sorted(SOURCE_NAMES)
         and type(packet['files']) is dict and set(packet['files']) == set(SOURCE_NAMES)|EXTRA_DATA
         and all(type(p) is str and len(p) == 64 and all(c in '0123456789abcdef' for c in p)
                 for p in packet['files'].values()), 'complete fixed full-shape packet differs')
    need(all(type(p) is str and packet['files'][n] == p for n,p in INDEPENDENT.items()),
         'independent input/helper pins unresolved or changed')


def validate_prepared(prepared, packet, data, service):
    need(prepared['schema'] == 'ace-screen-training-shape-preparation-v1'
         and prepared['scope'] == 'artificial' and prepared['labels'] == 'fabricated only'
         and prepared['training_execution_qualified'] is prepared['scientific_qualified'] is False
         and prepared['environment_responses'] == prepared['tokenizer_calls'] == 0
         and prepared['capacity_acceptance_sha256'] == CPU_ACCEPTANCE_SHA
         and prepared['capacity_packet_sha256'] == CAPACITY_PACKET_SHA
         and prepared['tokenization_sha256'] == TOKENS_SHA
         and prepared['training_plan_sha256'] == PLAN_SHA == service.digest(prepared['training_plan']),
         'independent saved-token preparation/plan differs')
    need(all(packet['files'].get(n) == p for n,p in prepared['source_sha256'].items()),
         'preparation source differs from training source')
    rows = {r['fixture_id']:r for r in service.decode(data['tokenization.json'])['rows']}
    long_name = 'collider-uniform-noise0.25-long_spelling-last_all_actions'
    terminal_name = 'collider-uniform-noise0.25-long_spelling-full_history_terminal'
    plan = prepared['training_plan']
    need(len(plan['records']) == 4 and plan['execution_kind'] == 'qwen_screen_singletons',
         'four explicit artificial records required')
    for index, rec in enumerate(plan['records']):
        expected = rows[terminal_name if index == 2 else long_name]
        raw = rec['raw_record_utf8'].encode()
        record = service.decode(raw)
        need(rec['record_id'] == 'shape-'+str(index) and rec['record_sha256'] == sha(raw)
             and rec['scope'] == record['scope'] == 'artificial'
             and rec['prompt_ids'] == expected['prompt_ids']
             and rec['commands'] == expected['commands']
             and rec['token_metadata'] == expected['metadata']
             and service.digest(record['envelope']) == expected['metadata']['envelope_sha256']
             and rec['tokenizer_sha256'] == service.digest(packet['model_files'])
             and rec['pad_token_id'] == 151643 and rec['eos_token_id'] == 151645,
             'unchanged saved full prompt/action/metadata association required')
    return plan


def authenticate(path,pin):
    root=Path(path).absolute().parent
    raw=capture(path,pin,128*1024)
    gpu=module('pinned_shape_gpu',root/GPU,capture(root/GPU,GPU_SHA))
    packet=json.loads(raw)
    packet_shape(packet,gpu)
    data={n:capture(root/n,p) for n,p in packet['files'].items()}
    service=module('pinned_shape_service',root/'llm_intervention_model_service.py',data['llm_intervention_model_service.py'])
    cpu=module('pinned_shape_cpu',root/CPU,data[CPU])
    d=module('pinned_shape_diagnostic',root/DIAGNOSTIC,data[DIAGNOSTIC])
    helper=module('pinned_shape_supervisor',root/SUPERVISOR,data[SUPERVISOR])
    capacity=service.decode(data['capacity-packet.json'])
    for key in ('archive','manifest','model_directory','model_files','runtime','runtime_source'):
        need(packet[key] == capacity[key], 'accepted tokenizer/runtime/model input changed: '+key)
    accepted=service.decode(data['cpu-acceptance.json'])
    need(accepted.get('job_id') == '35391340' and accepted.get('status') == 'accepted'
         and accepted.get('required_findings') == 0 and accepted.get('capacity_qualified') is True
         and accepted.get('complete_custody_verified') is True
         and accepted.get('runtime_qualified_for_this_cpu_execution') is True
         and accepted.get('owned_process_cleanup_verified') is True
         and accepted.get('packet_sha256') == CAPACITY_PACKET_SHA
         and accepted.get('tokenization_sha256') == TOKENS_SHA
         and accepted.get('training_execution_qualified') is False
         and accepted.get('scientific_qualified') is False,
         'independent actual tokenizer acceptance differs')
    prepared=service.decode(data['prepared.json'])
    validate_prepared(prepared,packet,data,service)
    need(Path(__file__).absolute()==root/ENTRY,'frozen entrypoint must be beside packet')
    return root,packet,data,packet,prepared,service,cpu,d,gpu,helper


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
            'prepared_sha256':PREPARED_SHA,'cpu_acceptance_sha256':CPU_ACCEPTANCE_SHA})
        observer.import_cpu_closure()
        from llm_intervention_screen_training_gpu_worker import run_training,validate_outputs
        summary=run_training(prepared,model,journal.output/'fit',observer,journal,prepared_sha256=PREPARED_SHA,plan_sha256=PLAN_SHA)
        checked=validate_outputs(journal.output/'fit',prepared,prepared_sha256=PREPARED_SHA,plan_sha256=PLAN_SHA)
        journal.save('terminal.json',{'schema':VERSION,'status':'artificial_training_complete',
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
    checked=worker.validate_outputs(out/'fit',prepared,prepared_sha256=PREPARED_SHA,plan_sha256=PLAN_SHA)
    consistent_summary(terminal,read('worker-summary.json'),read_bounded(out/'fit/summary.json'),checked)
    need(terminal['schema']==VERSION and terminal['status']=='artificial_training_complete'
        and terminal['packet_sha256']==packet_pin and terminal['runtime_stage_sha256']==stage_pin
        and terminal['saved_validation']==checked and terminal['summary']==read('worker-summary.json')
        and terminal['environment_responses']==0 and terminal['scientific_qualified'] is False,'worker terminal differs')
    auth=read('authenticated.json')
    need(auth['packet_sha256']==packet_pin and auth['prepared_sha256']==PREPARED_SHA
        and auth['cpu_acceptance_sha256']==CPU_ACCEPTANCE_SHA
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


class ProgressWatch:
    """Bounded diagnosis after first GPU work, not a utilization percentage."""
    def __init__(self,now,progress):
        self.last_gpu=self.last_progress=now
        self.progress=progress
    def observe(self,now,sample,progress):
        if sample.get('returncode')==0 and sample.get('owned_rows'):
            self.last_gpu=now
        if progress!=self.progress:
            self.progress=progress
            self.last_progress=now
        if now-self.last_gpu>90:
            raise TimeoutError('no reliable owned GPU telemetry for 90 seconds; utilization uncertain')
        if now-self.last_progress>90:
            raise TimeoutError('no advancing training/inference phase or completed operation for 90 seconds')
        return {'last_gpu_monotonic':self.last_gpu,'last_progress_monotonic':self.last_progress,
                'progress':list(progress),'freshness_bound_seconds':90}


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
    gpu_seen=False
    verified_at=None
    watch=None
    child_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    def interrupted(sig,frame):
        gpu.arm_final_deadline(start,current)
        raise TimeoutError('training supervisor signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.01,start+240-time.monotonic()))
    try:
        save(out/'packet.json',packet)
        job=os.environ.get('SLURM_JOB_ID','')
        scheduler=gpu.scheduler_record(out,job)
        save(out/'allocation.json',gpu.authenticate_allocation(scheduler,job,packet['resources'],helper))
        scratch=Path(os.environ.get('SLURM_SCRATCH',''))
        need(scratch.is_absolute(),'absolute node scratch required')
        scratch=scratch.resolve(strict=True)
        need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
        local=scratch/('ace-screen-training-shape-'+job)
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
        command=[str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(root/ENTRY),packet['files'][ENTRY],
            '--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,
            '--output',str(session/'worker'),'--worker','--runtime-stage',str(out/'stage.json'),
            '--runtime-stage-sha256',stage_pin,'--worker-deadline',str(start+540)]
        save(out/'child-command.json',{'command':command,'first_gpu_deadline_from_start_seconds':240,
            'absolute_worker_deadline':start+540,'absolute_supervisor_deadline':start+580})
        with (session/'stdout.log').open('xb') as stdout,(session/'stderr.log').open('xb') as stderr:
            current=subprocess.Popen(command,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            save(out/'child-start.json',{'pid':current.pid,'elapsed_seconds':time.monotonic()-start})
            ordinal=0
            while current.poll() is None:
                sample=helper.gpu_process_sample(current.pid)
                save(out/f'gpu-sample-{ordinal:04d}.json',{'elapsed_seconds':time.monotonic()-start,**sample})
                ordinal+=1
                if sample.get('returncode')==0 and sample.get('owned_rows'):gpu_seen=True
                if verified_at is None and gpu_seen and positive_work(session/'worker/artifacts/first-gpu-work.json'):
                    verified_at=time.monotonic()-start
                    need(verified_at<240,'first GPU work exceeded startup bound')
                    save(out/'first-verified-gpu-work.json',{'elapsed_seconds':verified_at,'owned_process_observed':True})
                    signal.setitimer(signal.ITIMER_REAL,max(.01,start+550-time.monotonic()))
                    watch=ProgressWatch(time.monotonic(),progress_records(session/'worker/artifacts'))
                if watch is not None:
                    evidence=watch.observe(time.monotonic(),sample,progress_records(session/'worker/artifacts'))
                    save(out/f'progress-{ordinal:04d}.json',evidence)
                if time.monotonic()>start+(550 if verified_at is not None else 240):
                    raise TimeoutError('bounded training child deadline exceeded')
                time.sleep(1)
        gpu.arm_final_deadline(start,current)
        cleanup=helper.terminate(current,min(time.monotonic()+12,start+565))
        save(out/'cleanup.json',cleanup)
        clean_success(cleanup)
        need(gpu_seen and verified_at is not None,'independent GPU work not verified during child')
        sample=helper.gpu_process_sample(current.pid)
        save(out/'post-exit-gpu.json',sample)
        gpu.require_gpu_sample(sample,active=False)
        validation=validate_worker(session/'worker/artifacts',packet,args.packet_sha256,data,old,prepared,
            service,cpu,d,stage,stage_pin,provenance)
        need(time.monotonic()<start+580,'absolute supervisor deadline exceeded')
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out/'terminal.json',{'schema':VERSION,'status':'artificial_training_complete',
            'packet_sha256':args.packet_sha256,'stage_sha256':stage_pin,'validation':validation,
            'first_verified_gpu_seconds':verified_at,'gpu_observation_count':ordinal,'cleanup':cleanup,
            'elapsed_seconds':time.monotonic()-start,'supervisor_cpu_seconds':time.process_time()-process_start,
            'child_and_helpers_cpu_seconds':usage.ru_utime+usage.ru_stime-child_usage.ru_utime-child_usage.ru_stime,
            'child_peak_rss_bytes':usage.ru_maxrss*1024,'environment_responses':0,'scientific_qualified':False,
            'external_scheduler_and_custody_acceptance_required':True})
    except BaseException as exc:
        gpu.arm_final_deadline(start,current)
        if current is not None and not (out/'cleanup.json').exists():
            save(out/'cleanup.json',helper.terminate(current,min(time.monotonic()+12,start+565)))
        save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','packet-sha256','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--runtime-stage');p.add_argument('--runtime-stage-sha256');p.add_argument('--worker-deadline',type=float)
    args=p.parse_args()
    args.started,args.process_started=time.monotonic(),time.process_time()
    out=Path(args.output).absolute()
    out.mkdir(mode=0o700,exist_ok=False)  # A collision must never alter prior evidence.
    def interrupted(sig,frame):raise TimeoutError('training authentication/worker signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        if args.worker:
            need(args.worker_deadline is not None and math.isfinite(args.worker_deadline)
                and 0<args.worker_deadline-time.monotonic()<=540,'bounded absolute worker deadline required')
            deadline=args.worker_deadline
        else:deadline=args.started+240
        signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
        context=authenticate(args.packet,args.packet_sha256)
        if args.worker:
            need(args.runtime_stage and args.runtime_stage_sha256,'independent stage pin required')
            child(args,*context)
        else:parent(args,*context)
    except BaseException as exc:
        if not (out/'failure.json').exists():
            save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
