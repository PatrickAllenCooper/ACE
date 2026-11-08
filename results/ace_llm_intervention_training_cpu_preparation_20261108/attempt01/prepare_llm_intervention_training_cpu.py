"""One CPU-only preparation of the complete saved artificial training batch.

Authenticates captured source and all inputs before an isolated staged-runtime
child uses the real tokenizer. No model construction, CUDA API, or new response.
Scheduler completion and independent full-custody acceptance remain external.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-artificial-training-cpu-v1'
ENTRY = 'prepare_llm_intervention_training_cpu.py'
CPU = 'qualify_llm_intervention_service_cpu.py'
CPU_SHA = 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
# Keep this literal separate from the manifest, including the fixed saved batch.
DIAGNOSTIC_SHA = '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329'
ROSTER = ('artificial/17/snapshot-000', 'artificial/17/snapshot-001',
          'artificial/4/snapshot-000', 'artificial/4/snapshot-001')
COLLECTION_PINS = {
    'collection-terminal.json': '2585c7c404fcc8340d93d2708264838fc1323fd9fec8804a0fe524db56d59193',
    'collection-inventory.json': '6c11ad8adc37f706dab0822345efb7afc332ea3c179b9d480e6473e8307e02bf',
    'collection-plan.json': '987c04dbe8532dab68f187125c98b437a16aef0ea320221bb70a9551151f0d19'}
EXTRA_SOURCES = {ENTRY, CPU, 'llm_intervention_training_plan.py',
    'llm_intervention_training_engine.py', 'llm_intervention_training_exposure.py',
    'llm_intervention_supervision.py', 'llm_intervention_checkpointing.py',
    'qualify_llm_intervention_checkpoint_gpu.py'}
BATCH_FILES = set(COLLECTION_PINS) | {f'{kind}-{i}.json' for kind in ('record','receipt') for i in range(4)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(path, pin, maximum=8*1024**2):
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin) is not None, 'independent pin required')
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular file required')
        raw = f.read(maximum+1)
        after = os.fstat(f.fileno())
    need(len(raw) <= maximum and (before.st_ino,before.st_size,before.st_mtime_ns) ==
         (after.st_ino,after.st_size,after.st_mtime_ns) and sha(raw) == pin, 'input changed or pin differs')
    return raw


def module(name, path, raw):
    m = types.ModuleType(name)
    m.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), vars(m))
    return m


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha(raw)


def packet_shape(packet, service, cpu, diagnostic):
    need(type(packet) is dict and set(packet) == {'schema','files','source_names','scope','record_roster',
        'archive','manifest','runtime_source','resources','model_directory','model_files','runtime'}
        and packet['schema'] == VERSION and packet['scope'] == 'artificial'
        and packet['record_roster'] == list(ROSTER)
        and service.canonical(packet['resources']) == service.canonical(cpu.RESOURCES)
        and packet['runtime'] == service.RUNTIME, 'fixed CPU packet contract differs')
    sources = service.SOURCE_NAMES | EXTRA_SOURCES
    inputs = sources | BATCH_FILES | {'probe.py','llm_intervention_runtime_archive.py'}
    need(packet['source_names'] == sorted(sources) and type(packet['files']) is dict
        and set(packet['files']) == inputs and all(service.pin(x) for x in packet['files'].values()),
        'complete fixed source/input roster required')
    files = packet['files']
    need(files[CPU] == CPU_SHA and files[DIAGNOSTIC] == DIAGNOSTIC_SHA
        and files['probe.py'] == diagnostic.PROBE_SHA
        and files['llm_intervention_runtime_archive.py'] == diagnostic.HELPER_SHA
        and files['llm_intervention_gpu_mappings.py'] == service.RUNTIME['gpu_mapping_source_sha256']
        and all(files[n] == pin for n,pin in COLLECTION_PINS.items()), 'original input/helper pin differs')
    for key,pin in [('archive',diagnostic.ARCHIVE_SHA), ('manifest',diagnostic.MANIFEST_SHA)]:
        row = packet[key]
        need(type(row) is dict and set(row) == {'path','sha256'} and row['sha256'] == pin
            and type(row['path']) is str and Path(row['path']).is_absolute(), 'runtime archive input differs')
    need(packet['runtime_source'] == diagnostic.RUNTIME_SOURCE
        and type(packet['model_directory']) is str and Path(packet['model_directory']).is_absolute()
        and type(packet['model_files']) is dict and set(packet['model_files']) == service.MODEL_NAMES
        and all(service.pin(x) for x in packet['model_files'].values()), 'model/runtime identity differs')


def authenticate(packet_path, packet_pin):
    root = Path(packet_path).absolute().parent
    packet_raw = capture(packet_path, packet_pin, 128*1024)
    cpu = module('training_cpu_checks', root/CPU, capture(root/CPU, CPU_SHA))
    d = module('training_cpu_diagnostic', root/DIAGNOSTIC, capture(root/DIAGNOSTIC, DIAGNOSTIC_SHA))
    packet = d.decode(packet_raw)
    # Only plain known-looking filenames may be read before the full roster check.
    files = packet.get('files', {})
    need(type(files) is dict and len(files) <= 64 and all(type(n) is str and
        Path(n).name == n and n.endswith(('.py','.json')) for n in files), 'bounded plain input names required')
    data = {n:capture(root/n,pin) for n,pin in files.items()}
    service = module('training_cpu_service', root/'llm_intervention_model_service.py', data['llm_intervention_model_service.py'])
    packet_shape(packet,service,cpu,d)
    need(Path(__file__).absolute() == root/ENTRY, 'frozen entrypoint must be beside packet')
    inventory = service.decode(data['collection-inventory.json'])
    need(inventory['record_roster'] == list(ROSTER), 'original complete artificial roster differs')
    for i,rid in enumerate(ROSTER):
        need(files[f'record-{i}.json'] == inventory['record_pins'][rid]
            and files[f'receipt-{i}.json'] == inventory['supervision_receipt_pins'][rid], 'original row pin differs')
    return root,packet,data,service,cpu,d


class CapturedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, captured, root):
        self.root, self.loaded = Path(root), {}
        self.sources = {Path(n).stem:b for n,b in captured.items() if n != ENTRY}
        need(not set(self.sources).intersection(sys.modules), 'project imports preceded authentication')
    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/(fullname+'.py')))
        if fullname.startswith(('llm_intervention_','qualify_llm_intervention_')):
            raise ImportError('project module outside captured closure: '+fullname)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, mod):
        mod.__file__ = str(self.root/(mod.__name__+'.py'))
        exec(compile(self.sources[mod.__name__],mod.__file__,'exec'),vars(mod))
        self.loaded[mod.__name__] = mod


def cpu_environment(prefix, inherited):
    env = dict(inherited)
    for name in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD','MKL_SERVICE_FORCE_INTEL'):
        env.pop(name,None)
        os.unsetenv(name)  # Includes C variables absent from os.environ.
    env.update(LD_LIBRARY_PATH=str(prefix/'lib'), CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='GNU')
    os.environ.update(env)
    os.putenv('MKL_THREADING_LAYER','GNU')
    return env


def validate_prepared(result, packet, data, service, packet_pin):
    """Read-only artifact checks. No tokenizer, torch import, or model execution."""
    plan, exposure = result['training_plan'], result['exposure']
    need(result['version'] == 'ace-complete-collection-training-plan-v1'
        and result['training_plan_sha256'] == service.digest(plan)
        and result['exposure_sha256'] == service.digest(exposure)
        and result['tokenizer_sha256'] == service.digest(packet['model_files'])
        and result['new_environment_responses'] == result['training_updates'] == 0
        and result['outer_execution_qualified'] is result['scientific_qualified'] is False,
        'prepared output pins or scope differ')
    for field,name in [('terminal','collection-terminal.json'),('inventory','collection-inventory.json'),('plan','collection-plan.json')]:
        need(result['collection_'+field+'_sha256'] == packet['files'][name], 'prepared collection binding differs')
    need(plan['scope'] == 'artificial' and plan['execution_kind'] == 'qwen_gpu09'
        and plan['outer_provenance_receipt_sha256'] == packet_pin
        and plan['checkpoint_steps'] == [3,5]
        and plan['sft_steps'] == plan['sft_rows'] == 4 and plan['dpo_steps'] == 2 and plan['dpo_rows'] == 4
        and plan['declared_adapter_parameters'] == plan['max_trainable_parameters'] == 1146880
        and len(plan['expected_adapter_shapes']) == 112 and plan['schedule'] == exposure['schedule']
        and [r['record_id'] for r in plan['records']] == list(ROSTER), 'training plan roster/recipe/counts differ')
    expected_schedule = [{'stage':'sft','record_id':rid,'pair_index':None} for rid in ROSTER]
    expected_schedule += [{'stage':'dpo','record_id':ROSTER[0],'pair_index':0},
                          {'stage':'dpo','record_id':ROSTER[2],'pair_index':1}]
    need(plan['schedule'] == expected_schedule and exposure['counts']['tied_pairs'] == 2
        and exposure['counts']['unselected_pairs'] == 2 and exposure['counts']['records_without_strict_pairs'] == 2,
        'complete exposure differs')
    count = 0
    for i,rec in enumerate(plan['records']):
        rid = ROSTER[i]
        need(rec['raw_record_utf8'].encode() == data[f'record-{i}.json']
            and rec['record_sha256'] == packet['files'][f'record-{i}.json']
            and rec['supervision_receipt_sha256'] == packet['files'][f'receipt-{i}.json']
            and rec['tokenizer_sha256'] == result['tokenizer_sha256']
            and plan['record_pins'][rid] == service.digest(rec)
            and rec['eos_token_id'] == 151645 and rec['pad_token_id'] == 151643,
            'real-tokenizer prepared row identity differs')
        meta,prompt,commands = rec['token_metadata'],rec['prompt_ids'],rec['commands']
        need(meta['prompt_tokens'] == len(prompt) and prompt and len(prompt)+32 <= 8192
            and meta['max_tokens'] == 8192 and meta['max_new_tokens'] == 32
            and meta['enable_thinking'] is False and meta['legal_commands'] == len(commands)
            and all(type(t) is int and t>=0 for t in prompt)
            and all(ids and len(ids)<=32 and ids[-1]==151645 and all(type(t) is int and t>=0 for t in ids)
                    for ids in commands.values()), 'bounded real tokenization differs')
        count += len(prompt)+sum(map(len,commands.values()))
    need(plan['stored_tokens'] == count, 'stored token total differs')
    return {'records':4,'sft_steps':4,'dpo_steps':2,'stored_tokens':count,
            'training_plan_sha256':result['training_plan_sha256']}


def child(args, root, packet, data, service, cpu, diagnostic):
    journal = service.Journal(args.output)
    captured = {n:data[n] for n in packet['source_names']}
    loader = CapturedSources(captured,root)
    sys.meta_path.insert(0,loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer = ServiceRuntime(packet,captured,loader,root/ENTRY,globals(),args.runtime_stage,
            args.runtime_stage_sha256,journal,service.read_pinned,service.decode)
        model = Path(packet['model_directory'])
        need(not model.is_symlink() and model.is_dir() and {p.name for p in model.iterdir()} == service.MODEL_NAMES,
            'exact model directory membership required')
        models = {n:service.read_pinned(model/n,p) for n,p in packet['model_files'].items()}
        journal.save('authenticated.json',{'packet_sha256':args.packet_sha256,'models':models,
            'source_sha256':{n:packet['files'][n] for n in packet['source_names']}})
        observer.import_cpu_closure()
        from transformers import AutoTokenizer
        from llm_intervention_training_plan import prepare_complete_collection
        journal.event('cpu_tokenizer_load_start')
        tokenizer = AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        need(tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643,'original tokenizer IDs differ')
        result = prepare_complete_collection(
            terminal_raw=data['collection-terminal.json'],terminal_pin=packet['files']['collection-terminal.json'],
            inventory_raw=data['collection-inventory.json'],inventory_pin=packet['files']['collection-inventory.json'],
            collection_plan_raw=data['collection-plan.json'],collection_plan_pin=packet['files']['collection-plan.json'],
            records={rid:data[f'record-{i}.json'] for i,rid in enumerate(ROSTER)},
            receipts={rid:data[f'receipt-{i}.json'] for i,rid in enumerate(ROSTER)},
            expected_roster=ROSTER,expected_scope='artificial',tokenizer=tokenizer,
            tokenizer_pin=service.digest(packet['model_files']),outer_provenance_pin=args.packet_sha256,
            checkpoint_steps=(3,5))
        summary = validate_prepared(result,packet,data,service,args.packet_sha256)
        journal.save('prepared.json',result)
        journal.event('cpu_tokenization_complete',**summary)
        observer.observe('after-cpu-tokenization')
        journal.save('terminal.json',{'schema':VERSION,'status':'cpu_preparation_complete',
            'packet_sha256':args.packet_sha256,'runtime_stage_sha256':args.runtime_stage_sha256,
            'prepared_file_sha256':sha((journal.output/'prepared.json').read_bytes()),**summary,
            'model_loaded':False,'explicit_cuda_api_called':False,'new_training_updates':0,
            'environment_responses':0,'training_execution_qualified':False,'scientific_qualified':False})
    except BaseException as exc:
        journal.save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        sys.meta_path.remove(loader)


def validate_worker(out, packet, packet_pin, data, service, cpu, diagnostic, stage, stage_pin, provenance, checks):
    def read(name):
        raw = (out/name).read_bytes()
        need(len(raw)<=32*1024**2,'bounded worker receipt required')
        return service.decode(raw)
    need(not list(out.rglob('failure.json')), 'failed worker cannot qualify')
    terminal = read('terminal.json')
    summary = validate_prepared(read('prepared.json'),packet,data,service,packet_pin)
    need(terminal['schema']==VERSION and terminal['status']=='cpu_preparation_complete'
        and terminal['packet_sha256']==packet_pin and terminal['runtime_stage_sha256']==stage_pin
        and terminal['prepared_file_sha256']==sha((out/'prepared.json').read_bytes())
        and all(terminal.get(k)==v for k,v in summary.items())
        and all(terminal[k] is False for k in ('model_loaded','explicit_cuda_api_called','training_execution_qualified','scientific_qualified'))
        and terminal['new_training_updates']==terminal['environment_responses']==0,'terminal scope differs')
    authenticated = read('authenticated.json')
    need(authenticated['packet_sha256']==packet_pin
        and authenticated['source_sha256']=={n:packet['files'][n] for n in packet['source_names']}
        and set(authenticated['models'])==service.MODEL_NAMES
        and all(r['sha256']==packet['model_files'][n] and type(r['bytes']) is int and r['bytes']>0
                for n,r in authenticated['models'].items()),'input authentication differs')
    cpu.validate_runtime_records(read,stage,stage_pin,service,diagnostic,checks,provenance)
    need(read('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,'CPU device exception forbidden')
    expected = {Path(n).stem:packet['files'][n] for n in packet['source_names'] if n!=ENTRY}
    for phase in ('after-service-imports','after-cpu-tokenization'):
        projects = read('runtime-'+phase+'.json')['captured_project_modules']
        need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'}
            and all(projects[n]['sha256']==p for n,p in expected.items()),'complete captured import closure differs')
    return sha((out/'terminal.json').read_bytes())


def parent(args, root, packet, data, service, cpu, d):
    out = Path(args.output).absolute()
    out.mkdir(mode=0o700,exist_ok=False)
    start,process = time.monotonic(),time.process_time()
    try:
        save(out/'packet.json',packet)
        job = os.environ.get('SLURM_JOB_ID','')
        need(job.isdigit() and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'CPU Slurm job required')
        query = subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
        save(out/'allocation.json',{'stdout':query.stdout,'stderr':query.stderr,'returncode':query.returncode})
        need(query.returncode==0,'allocation query failed')
        cpu.allocation_fields(query.stdout,job)
        scratch = Path(os.environ.get('SLURM_SCRATCH',''))
        need(scratch.is_absolute(),'absolute node scratch required')
        scratch = scratch.resolve(strict=True)
        need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
        local = scratch/('ace-training-cpu-'+job)
        local.mkdir(mode=0o700,exist_ok=False)
        helper = module('training_archive',root/'llm_intervention_runtime_archive.py',data['llm_intervention_runtime_archive.py'])
        probe = module('training_cleanup',root/'probe.py',data['probe.py'])
        checks = module('training_runtime_checks',root/'llm_intervention_service_runtime.py',data['llm_intervention_service_runtime.py'])
        event = d.events(out,start,process,512*1024**2)
        event('stage_start')
        stage = d.stage_runtime(packet,helper,local/'runtime',event)
        stage_pin = save(out/'stage.json',stage)
        event('stage_complete',staged_prefix=stage['staged_prefix'])
        prefix = Path(stage['staged_prefix'])
        provenance = d.provenance_contract(prefix)
        save(out/'independent-provenance-sources.json',provenance)
        directory = out/'session'
        directory.mkdir(mode=0o700,exist_ok=False)
        command = [str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(root/ENTRY),packet['files'][ENTRY],
            '--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,
            '--output',str(directory/'worker'),'--worker','--runtime-stage',str(out/'stage.json'),
            '--runtime-stage-sha256',stage_pin]
        event('preparation_start')
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = d.run_child(command,cpu_environment(prefix,os.environ),directory,start+580,probe.kill_owned,
            lambda:validate_worker(directory/'worker',packet,args.packet_sha256,data,service,cpu,d,stage,stage_pin,provenance,checks))
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            cumulative_child_peak_rss_bytes=after.ru_maxrss*1024)
        event('preparation_complete',**result)
        need(time.monotonic()-start<580,'absolute CPU deadline exceeded')
        save(out/'terminal.json',{'schema':VERSION,'status':'cpu_preparation_complete',
            'packet_sha256':args.packet_sha256,'stage_sha256':stage_pin,'session':result,
            'elapsed_seconds':time.monotonic()-start,'supervisor_cpu_seconds':time.process_time()-process,
            'supervisor_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'model_weights_loaded':False,'new_training_updates':0,'environment_responses':0,
            'training_execution_qualified':False,'scientific_qualified':False})
    except BaseException as exc:
        save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('packet','packet-sha256','output'):
        parser.add_argument('--'+n,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--runtime-stage')
    parser.add_argument('--runtime-stage-sha256')
    args = parser.parse_args()
    def interrupted(sig, frame):
        raise InterruptedError('CPU preparation signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):
        signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,180 if args.worker else 580)
    try:
        context = authenticate(args.packet,args.packet_sha256)
        if args.worker:
            need(args.runtime_stage and args.runtime_stage_sha256,'independently pinned runtime stage required')
            child(args,*context)
        else:
            parent(args,*context)
    except BaseException as exc:
        out = Path(args.output).absolute()
        if not out.exists():
            out.mkdir(mode=0o700,exist_ok=False)
        if not (out/'failure.json').exists():
            save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),
                'traceback':traceback.format_exc(),'phase':'authentication_or_dispatch'})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()
