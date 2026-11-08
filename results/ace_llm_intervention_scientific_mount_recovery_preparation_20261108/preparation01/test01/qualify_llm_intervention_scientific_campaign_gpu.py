"""Bounded staging and inference-only execution of the supplied scientific campaign.

The original CPU packet and its independently accepted outputs are inputs. The
campaign commands stay fixed. This supervisor stages once and execs the same PID
into that interpreter before running the three sequential model services.
Caller supplies independently pinned packet and stage admission. No default
allocation, scientific acceptance, or budget reservation is manufactured. Original
CPU inputs, checkpoints and five attempts remain immutable. New source derives
from the accepted artificial outer without editing or patching its globals.
Independent Slurm completion, custody and final CPU reporting remain required.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import math
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

VERSION = 'ace-scientific-paired-campaign-gpu-v1'
ENTRY = 'qualify_llm_intervention_scientific_campaign_gpu.py'
CPU_ENTRY = 'prepare_llm_intervention_scientific_campaign_cpu.py'
CPU_ENTRY_SHA = '75a7179a8e27be4425b148c9bae706b81aaceb706c1d55a84ccf018e798fef82'
CPU_OWNED = 'llm_intervention_cpu_owned_supervision.py'
CPU_OWNED_SHA = '1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8'
GPU_HELPER = 'qualify_llm_intervention_model_service.py'
GPU_HELPER_SHA = '76accdcd2c24bdc6253a1c131be191a6c6534b52bd5246caaaf66a1f64473e69'
PROCESS_HELPER = 'supervise_llm_intervention_checkpoint_gpu.py'
PROCESS_HELPER_SHA = 'ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
DIAGNOSTIC_SHA = '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329'
EXTRA_SOURCES = {ENTRY, GPU_HELPER, PROCESS_HELPER,
    'llm_intervention_scientific_gpu_phase.py', 'llm_intervention_scientific_service_evidence.py'}
POLICIES = ('uniform_random','menu_coverage','local_information','target_risk',
            'learned_ranker','frozen_llm','sft_llm','dpo_llm')
RESOURCE_BASE = {'account':'ucb736_asc2','partition':'aa100','qos':'gpu-testing',
    'requested_cpus':4,'max_allocated_cpus':4,'memory_mib':15360,
    'gpus':1,'gpu_type':'a100_3g.20gb','requeue':False}

PHASE_SHA = 'c89e26d2393b97b6115dfd29969b2d9f7fbdf0ebd2dc1505ac604d15faf4d1d7'
EVIDENCE_SHA = 'd136efa7814435757e2c9d5026784a891f51ecafd72ea611c772dd0481d0b2b6'
WORLD_IDS = tuple(range(202611070100,202611070124))
LINEAGE = 'llm_intervention_scientific_training_lineage.py'
LINEAGE_SHA = 'f3b328203d5758418d0afd677513822e8d7f134acca10545aab2d3fda1d3fc35'
LINEAGE_FILES = dict(acceptance_raw='gpu-training-acceptance.json',prepared_raw='prepared.json',
    engine_terminal_raw='engine-terminal.json',sft_checkpoint_raw='sft.checkpoint.json',
    sft_adapter_raw='sft.adapter.pt',final_checkpoint_raw='final.checkpoint.json',final_adapter_raw='final.adapter.pt')
PATH_KEYS = {'cpu_inputs','cpu_output','campaign','gpu_inputs','gpu_output'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def decode(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            need(key not in result, 'duplicate input key')
            result[key] = item
        return result
    result = json.loads(value, object_pairs_hook=unique)
    raw(result)
    return result


def read(path, expected=None, maximum=256*1024**2):
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular input required')
        value = handle.read(maximum+1)
        after = os.fstat(handle.fileno())
    need(len(value) <= maximum and (before.st_ino,before.st_size,before.st_mtime_ns) ==
         (after.st_ino,after.st_size,after.st_mtime_ns), 'input changed during capture')
    if expected is not None:
        need(pin(expected) and sha(value) == expected, 'independent input pin differs: '+str(path))
    return value


def save(path, value):
    value = raw(value)
    with Path(path).open('xb') as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
    return sha(value)


def module(name, path, value):
    item = types.ModuleType(name)
    item.__file__ = str(path)
    exec(compile(value,str(path),'exec'),vars(item))
    return item


def absolute(path):
    return (type(path) is str and path.startswith('/') and '\0' not in path
            and str(Path(path)) == path and '..' not in Path(path).parts)


def protect_inputs(cpu_packet,*writable):
    """Reject writable ancestors/descendants of the pinned immutable input paths."""
    protected=(cpu_packet['runtime_source'],cpu_packet['model_directory'],
               cpu_packet['archive']['path'],cpu_packet['manifest']['path'])
    need(all(absolute(p) for p in protected),'absolute protected runtime/model/archive/manifest paths required')
    targets=[]
    for p in writable:
        need(absolute(str(p)),'absolute writable root required')
        targets.append(Path(p).resolve())
    for source in protected:
        original=Path(source).resolve()
        need(all(p!=original and p not in original.parents and original not in p.parents for p in targets),
             'writable output/staging overlaps protected input: '+source)


def inventory(root, *, exclude=None):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), 'ordinary custody root required')
    members = list(root.rglob('*'))
    need(len(members) <= 262144 and all(not p.is_symlink() and (p.is_file() or p.is_dir())
         for p in members), 'bounded ordinary custody membership required')
    excluded = root/exclude if exclude is not None else None
    files = sorted(p for p in members if p.is_file() and
                   (excluded is None or p!=excluded and excluded not in p.parents))
    need(sum(p.stat().st_size for p in files) <= 8*1024**3, 'CPU custody byte bound exceeded')
    return {str(p.relative_to(root)):sha(read(p)) for p in files}


def resource_contract(packet):
    r,limits=packet['resources'],packet['limits']
    need(type(r) is dict and set(r)==set(RESOURCE_BASE)|{'wall_seconds'}
         and all(raw(r[k])==raw(v) for k,v in RESOURCE_BASE.items())
         and type(r['wall_seconds']) is int and 300<=r['wall_seconds']<=3600,
         'explicit exact four-CPU/15GiB/one-MIG allocation required')
    need(type(limits) is dict and set(limits)=={'session_reserve_seconds','session_cap_seconds','work_seconds','final_seconds'}
         and all(type(v) is int for v in limits.values()), 'explicit integer phase limits required')
    reserve,cap,work,final=(limits[k] for k in ('session_reserve_seconds','session_cap_seconds','work_seconds','final_seconds'))
    need(120<=reserve<=cap<=1200 and 240<work<=3540 and work+20<=final<=3570
         and final<r['wall_seconds'] and 3*reserve+120<=work,
         'phase limits must reserve staging, three services and cleanup inside allocation')


def admission_contract(packet,value,packet_pin):
    expected=dict(schema=VERSION+'/admission',packet_sha256=packet_pin,scope='development',
        freeze_sha256=packet['freeze_sha256'],namespace_admission_sha256=packet['namespace_admission_sha256'],
        stage='development_gpu_campaign',output=packet['paths']['gpu_output'],resources=packet['resources'],
        limits=packet['limits'],full_contract_frozen_before_first_scientific_draw=True,
        source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,
        independent_acceptances_verified=True,retry_allowed=False)
    need(type(value) is dict and set(value)==set(expected)|{'site_readiness_sha256','accounting_sha256'}
         and all(raw(value[k])==raw(v) for k,v in expected.items())
         and pin(value['site_readiness_sha256']) and pin(value['accounting_sha256']),
         'independent stage/source/namespace/resource/accounting admission required')


def admit_cpu(acceptance,packet,cpu_packet,prepared):
    """Authenticate full original CPU custody before decoding its private bundle."""
    required={'schema','status','cpu_packet_sha256','job_id','allocation','cpu_output','campaign_root',
        'cpu_output_files','cpu_output_excluded_directory','campaign_files','bundle_path','bundle_sha256',
        'phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}
    need(type(acceptance) is dict and set(acceptance)==required
        and acceptance['schema']=='ace-scientific-campaign-cpu-acceptance-v1'
        and acceptance['status']=='development_cpu_phase_accepted' and acceptance['scientific_qualified'] is False
        and acceptance['cpu_packet_sha256']==packet['cpu_packet_sha256']
        and type(acceptance['job_id']) is str and re.fullmatch('[1-9][0-9]*',acceptance['job_id']),
        'independent scientific CPU campaign acceptance differs')
    need(acceptance['allocation']==dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',
        cpus=2,memory_mib=7680,gpus=0,state='COMPLETED',exit_code='0:0'),'accepted CPU allocation differs')
    paths=packet['paths'];excluded='worker/artifacts/campaign'
    need(acceptance['cpu_output']==paths['cpu_output']==cpu_packet['cpu_output']
        and acceptance['campaign_root']==paths['campaign']==str(Path(paths['cpu_output'])/excluded)
        and acceptance['cpu_output_excluded_directory']==excluded,'accepted CPU evidence roots differ')
    need(inventory(paths['cpu_output'],exclude=excluded)==acceptance['cpu_output_files'],
         'independent CPU output inventory differs')
    need(inventory(paths['campaign'])==acceptance['campaign_files'],'independent CPU campaign inventory differs')
    for field,wanted in (('bundle_path','worker/artifacts/cpu-phase/campaign-bundle.json'),
                         ('phase_terminal_path','worker/artifacts/cpu-phase/terminal.json')):
        need(acceptance[field]==wanted and pin(acceptance[field.replace('_path','_sha256')])
            and acceptance['cpu_output_files'].get(wanted)==acceptance[field.replace('_path','_sha256')],
            'original CPU bundle/phase path or inventory pin differs')
    need(set(acceptance['attempt_pins'])==set(POLICIES[:5]) and all(pin(h) for h in acceptance['attempt_pins'].values()),
         'all five CPU attempt pins required')
    for i,pid in enumerate(POLICIES[:5]):
        need(all(acceptance['campaign_files'].get(f'policy-{i:02d}/'+name)==acceptance['attempt_pins'][pid]
            for name in ('attempt.json','attempt.candidate.json')),'CPU attempt/candidate custody differs')
    base=Path(paths['cpu_output']);terminal=decode(read(base/acceptance['phase_terminal_path'],acceptance['phase_terminal_sha256']))
    d=prepared['exposure']['counts']['dpo_steps']
    need(type(d) is int and 1<=d<=128 and terminal['schema']=='ace-scientific-campaign-cpu-v1'
        and terminal['status']=='cpu_phase_complete' and terminal['attempt_pins']==acceptance['attempt_pins']
        and terminal['numeric_training_updates']==160+d and terminal['llm_training_updates']==0
        and terminal['completed_cpu_cells']==120 and terminal['declared_cells']==192
        and terminal['gpu_services_started']==0 and terminal['private_scores_computed'] is False
        and terminal['scientific_qualified'] is False and terminal['retry_allowed'] is False,
        'successful complete scientific CPU phase required')
    campaign_root=Path(paths['campaign'])
    need({p.name for p in campaign_root.iterdir()}=={'campaign.json','source-plan-pins.json','start.json'}|
        {f'policy-{i:02d}' for i in range(5)},'GPU attempt/finalization/retry already present')
    bundle=decode(read(base/acceptance['bundle_path'],acceptance['bundle_sha256']))
    c=bundle['campaign']
    need(c['policy_ids']==list(POLICIES) and c['scope']==bundle['scope']=='development'
        and len(c['sources'])==24 and [s['plan']['world']['world_id'] for s in c['sources']]==list(WORLD_IDS)
        and c['policy_phases']=={p:'cpu' if i<5 else 'gpu' for i,p in enumerate(POLICIES)}
        and terminal['campaign_sha256']==bundle['campaign_sha256']
        and all(len(s['plan']['initial_actions'])==8 and s['plan']['initial_cost_units']==8
                and s['plan']['acquisition_cap_units']==24 for s in c['sources']),
        'fixed development 24-world/192-cell campaign differs')
    return bundle


def allocation(raw_text,job,helper,resources):
    fields=helper.scheduler_fields(raw_text);wall=resources['wall_seconds']
    wanted={'JobId':job,'Account':'ucb736_asc2','Partition':'aa100','QOS':'gpu-testing',
        'JobState':'RUNNING','NumNodes':'1','NumTasks':'1','NumCPUs':'4','CPUs/Task':'4',
        'TimeLimit':f'{wall//3600:02d}:{wall%3600//60:02d}:{wall%60:02d}','Requeue':'0','Restarts':'0'}
    need(re.fullmatch('[1-9][0-9]*',job) and all(fields.get(k)==v for k,v in wanted.items())
        and fields.get('UserId','').startswith('paco0228('),'actual allocation identity differs')
    def tres(field):
        result={}
        for pair in fields.get(field,'').split(','):
            need(pair.count('=')==1,'malformed resource record');key,value=pair.split('=')
            need(key not in result,'duplicate resource key');result[key]=value
        need(result.get('node')=='1' and result.get('cpu')=='4'
            and helper.memory_mib(result.get('mem',''))==15360
            and result.get('gres/gpu:a100_3g.20gb')=='1' and result.get('gres/gpu','1')=='1'
            and not any(k.startswith('gres/') and k not in {'gres/gpu','gres/gpu:a100_3g.20gb'} for k in result),
            'actual/requested exact CPU memory MIG allocation differs')
        return result
    return dict(fields=fields,allocated_resources=tres('AllocTRES'),requested_resources=tres('ReqTRES'))


class CapturedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, captured, root):
        self.root,self.loaded = Path(root),{}
        self.sources = {Path(n).stem:b for n,b in captured.items() if n!=ENTRY}
        need(not set(self.sources).intersection(sys.modules), 'project imports preceded authentication')
    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/(fullname+'.py')))
        if fullname.startswith(('llm_intervention_','qualify_llm_intervention_',
                                'prepare_llm_intervention_','supervise_llm_intervention_')):
            raise ImportError('outside captured source closure: '+fullname)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, mod):
        mod.__file__ = str(self.root/(mod.__name__+'.py'))
        exec(compile(self.sources[mod.__name__],mod.__file__,'exec'),vars(mod))
        self.loaded[mod.__name__] = mod


def authenticate(path, expected, admission_path, admission_pin):
    root = Path(path).absolute().parent
    packet_raw=read(path,expected,256*1024)
    packet_pin=expected
    packet = decode(packet_raw)
    fields = {'schema','files','source_names','cpu_packet_sha256','cpu_acceptance_sha256',
              'paths','resources','services','contracts','lineage_files','scope','limits',
              'freeze_sha256','namespace_admission_sha256'}
    need(type(packet) is dict and set(packet)==fields and packet['schema']==VERSION
         and packet['scope']=='development', 'fixed scientific GPU packet required')
    resource_contract(packet)
    files = packet['files']
    need(type(files) is dict and len(files)<=256 and all(type(n) is str and
         Path(n).name==n and n.endswith(('.py','.json','.pt')) and pin(p) for n,p in files.items()),
         'bounded plain input roster required')
    need(files.get('cpu-packet.json')==packet['cpu_packet_sha256']
         and files.get('cpu-acceptance.json')==packet['cpu_acceptance_sha256']
         and files.get(GPU_HELPER)==GPU_HELPER_SHA and files.get(PROCESS_HELPER)==PROCESS_HELPER_SHA
         and files.get(DIAGNOSTIC)==DIAGNOSTIC_SHA
         and files.get(CPU_ENTRY)==CPU_ENTRY_SHA and files.get(CPU_OWNED)==CPU_OWNED_SHA
         and files.get('llm_intervention_scientific_gpu_phase.py')==PHASE_SHA
         and files.get('llm_intervention_scientific_service_evidence.py')==EVIDENCE_SHA, 'independent predecessor/helper pins differ')
    data={};total=0
    for n,p in files.items():
        data[n]=read(root/n,p);total+=len(data[n])
        need(total<=512*1024**2,'bounded total captured inputs required')
    old = decode(data['cpu-packet.json'])
    need(old['schema']=='ace-scientific-campaign-cpu-v1' and old['scope']=='development'
         and {CPU_ENTRY,CPU_OWNED}<=set(old['source_names'])
         and old['freeze_sha256']==packet['freeze_sha256']==files['scientific-freeze.json']
         and old['namespace_admission_sha256']==packet['namespace_admission_sha256']
         and pin(packet['namespace_admission_sha256']), 'original scientific CPU/freeze/namespace differs')
    expected_files = set(old['files'])|EXTRA_SOURCES|{'cpu-packet.json','cpu-acceptance.json'}
    need(set(files)==expected_files and all(files[n]==p for n,p in old['files'].items())
         and packet['source_names']==sorted(set(old['source_names'])|EXTRA_SOURCES)
         and all(n in files and n.endswith('.py') for n in packet['source_names']),
         'exact accepted CPU and added GPU source closure required')
    need(Path(__file__).absolute()==root/ENTRY, 'GPU entrypoint must be beside packet')
    paths = packet['paths']
    need(type(paths) is dict and set(paths)==PATH_KEYS
         and all(absolute(p) for p in paths.values()) and len(set(paths.values()))==5,
         'fixed separate absolute roots required')
    need(paths['cpu_output']==old['cpu_output'] and paths['campaign']==str(
         Path(old['cpu_output'])/'worker/artifacts/campaign'), 'accepted CPU layout differs')
    roots = [Path(paths[k]) for k in ('cpu_inputs','cpu_output','gpu_inputs','gpu_output')]
    need(root==Path(paths['gpu_inputs']) and all(not p.is_symlink() for p in roots)
         and len({p.resolve() for p in roots})==len(roots)
         and not any(a.resolve() in b.resolve().parents for a in roots for b in roots if a!=b), 'evidence roots overlap')
    protect_inputs(old,paths['gpu_output'])
    admission_raw=read(admission_path,admission_pin,1024**2)
    admission=decode(admission_raw)
    admission_contract(packet,admission,expected)
    need(read(Path(paths['cpu_inputs'])/'packet.json',packet['cpu_packet_sha256'])==data['cpu-packet.json'],
         'original CPU packet differs')
    # Service commands still point to the original CPU input directory. It must
    # retain the exact independently accepted bytes, including launch contracts.
    for name,value in old['files'].items():
        need(read(Path(paths['cpu_inputs'])/name,value)==data[name], 'original CPU input changed: '+name)
    service = module('pinned_campaign_service',root/'llm_intervention_model_service.py',data['llm_intervention_model_service.py'])
    gpu = module('pinned_campaign_gpu_helper',root/GPU_HELPER,data[GPU_HELPER])
    helper = module('pinned_campaign_process_helper',root/PROCESS_HELPER,data[PROCESS_HELPER])
    diagnostic = module('pinned_campaign_diagnostic',root/DIAGNOSTIC,data[DIAGNOSTIC])
    need(files['probe.py']==diagnostic.PROBE_SHA
         and files['llm_intervention_runtime_archive.py']==diagnostic.HELPER_SHA,
         'accepted archive/probe helpers differ')
    for key,value in (('archive',diagnostic.ARCHIVE_SHA),('manifest',diagnostic.MANIFEST_SHA)):
        need(old[key]['sha256']==value and absolute(old[key]['path']), 'accepted archive input differs')
    need(old['runtime_source']==diagnostic.RUNTIME_SOURCE and old['runtime']==service.RUNTIME,
         'original accepted runtime differs')
    acceptance = decode(data['cpu-acceptance.json'])
    bundle = admit_cpu(acceptance,packet,old,decode(data['prepared.json']))
    lineage_name = LINEAGE
    need(files[lineage_name]==LINEAGE_SHA,
         'reviewed training role admission differs')
    lineage_api = module('pinned_campaign_lineage',root/lineage_name,data[lineage_name])
    need(type(packet['lineage_files']) is dict and packet['lineage_files']==LINEAGE_FILES
         and all(n in old['files'] for n in packet['lineage_files'].values()), 'complete training-role inputs required')
    lineage = lineage_api.admit_training(**{k:data[n] for k,n in packet['lineage_files'].items()},
        acceptance_sha256=files['gpu-training-acceptance.json'],expected=old['expected_training'])
    need(old['expected_training']['scientific_freeze_sha256']==packet['freeze_sha256']
         and old['expected_training']['cpu_preparation_acceptance_sha256']==files['cpu-preparation-acceptance.json']
         and old['expected_training']['model_files_sha256']==service.digest(old['model_files'])
         and old['expected_training']['runtime_contract_sha256']==service.digest(old['runtime']),
         'independent training/base/runtime lineage differs')
    launch_name = 'llm_intervention_campaign_launch.py'
    need(files[launch_name]=='d3a97faef71f01910405e785b20fd83710ed4a3ef86027c052ba3e303f2ffc25',
         'reviewed launch shim differs')
    launch = module('pinned_campaign_launch',root/launch_name,data[launch_name])
    for section in ('services','contracts'):
        need(type(packet[section]) is dict and set(packet[section])==set(POLICIES[5:])
             and len(set(packet[section].values()))==3
             and all(n in old['files'] for n in packet[section].values()), 'three distinct service inputs required')
    freezes,contracts = {},{}
    gpu_root,original = Path(paths['gpu_output']),Path(paths['cpu_inputs'])
    for pid in POLICIES[5:]:
        freeze_name,contract_name = packet['services'][pid],packet['contracts'][pid]
        freeze = service.validate_freeze(decode(data[freeze_name]))
        contract = launch.validate_contract(decode(data[contract_name]))
        limits=freeze['limits']
        need(freeze['scope']=='development' and limits['max_requests']==576
             and limits['startup_seconds']<=120 and limits['total_seconds']<=packet['limits']['session_cap_seconds']
             and freeze['model_directory']==old['model_directory'] and freeze['model_files']==old['model_files']
             and freeze['sources']=={n:files[n] for n in service.SOURCE_NAMES}, 'fixed scientific service bounds/source differ')
        if pid=='frozen_llm':
            need(freeze['treatment']=='frozen' and freeze['adapter'] is None, 'frozen arm has an adapter')
        else:
            role,a = lineage['roles'][pid],freeze['adapter']
            need(freeze['treatment']==role['treatment'] and a['training_scope']=='policy_train'
                 and a['sha256']==role['adapter_sha256']
                 and a['training_receipt_sha256']==role['checkpoint_sha256']
                 and files[a['file']]==a['sha256'] and files[a['training_receipt']]==a['training_receipt_sha256'],
                 'accepted SFT/DPO checkpoint role differs')
        expected = {'policy_id':pid,'entry':str(original/'llm_intervention_model_service.py'),
            'entry_sha256':files['llm_intervention_model_service.py'],'freeze':str(original/freeze_name),
            'freeze_sha256':files[freeze_name],'worker_output':str(gpu_root/'services'/pid/'worker'),
            'launch_output':str(gpu_root/'services'/pid/'launch'),'binding_path':str(gpu_root/'stage-binding.json'),
            'stage_path':str(gpu_root/'stage.json'),'archive_sha256':diagnostic.ARCHIVE_SHA,
            'manifest_sha256':diagnostic.MANIFEST_SHA,'bootstrap':diagnostic.BOOTSTRAP,
            'bootstrap_sha256':sha(diagnostic.BOOTSTRAP.encode())}
        need(all(contract[k]==v for k,v in expected.items()), 'fixed service launch locations/pins differ')
        command = [str(Path(old['runtime_source'])/'bin/python'),'-I','-B','-c',diagnostic.BOOTSTRAP,str(original/launch_name),
            files[launch_name],'--contract',str(original/contract_name),'--contract-sha256',files[contract_name]]
        settings = bundle['campaign']['services'][pid]
        identity = service.identity(freeze)
        need(settings=={'identity':identity,'startup_seconds':limits['startup_seconds']+5,'cleanup_seconds':5.,'max_requests':576},
             'campaign service limits/identity differ')
        for source in bundle['campaign']['sources']:
            spec = next(s for s in source['plan']['policy_specs'] if s['policy_id']==pid)
            need(spec['command']==command and spec['kind']=='persistent' and spec['timeout_seconds']==limits['request_seconds']+5
                 and spec['identity_sha256']==service.digest(identity)
                 and spec['policy_provenance_sha256']==files[freeze_name], 'campaign command/provenance differs')
        freezes[pid],contracts[pid] = freeze,contract
    need(all(f['model_directory']==freezes['frozen_llm']['model_directory'] and
         f['model_files']==freezes['frozen_llm']['model_files'] for f in freezes.values()),
         'shared base-model identity differs')
    for name in packet['source_names']:
        for node in ast.walk(ast.parse(data[name])):
            deps=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                  [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            need(all(dep+'.py' in packet['source_names'] for dep in deps if dep.startswith(
                ('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_'))),
                'captured project import closure incomplete')
    return {'root':root,'packet':packet,'data':data,'old':old,'acceptance':acceptance,'bundle':bundle,
            'service':service,'gpu':gpu,'helper':helper,'diagnostic':diagnostic,
            'lineage':lineage,'launch':launch,'freezes':freezes,'contracts':contracts,'admission':admission,
            'packet_path':Path(path).absolute(),'packet_pin':packet_pin,'packet_raw':packet_raw,
            'admission_path':Path(admission_path).absolute(),'admission_pin':admission_pin,'admission_raw':admission_raw}


def recheck_inputs(ctx):
    """Before success, compare all authenticated disk inputs to captured bytes."""
    packet,data=ctx['packet'],ctx['data']
    need(read(ctx['packet_path'],ctx['packet_pin'],256*1024)==ctx['packet_raw'],'GPU packet changed')
    need(read(ctx['admission_path'],ctx['admission_pin'],1024**2)==ctx['admission_raw'],'GPU admission changed')
    for name,pin_value in packet['files'].items():
        need(read(ctx['root']/name,pin_value)==data[name],'GPU input changed: '+name)
    original=Path(packet['paths']['cpu_inputs'])
    need(read(original/'packet.json',packet['cpu_packet_sha256'])==data['cpu-packet.json'],'original CPU packet changed')
    for name,pin_value in ctx['old']['files'].items():
        need(read(original/name,pin_value)==data[name],'original CPU input changed: '+name)


def stage_and_exec(args, ctx):
    out = Path(args.output).absolute()
    packet,data,d,gpu,helper = (ctx[k] for k in ('packet','data','diagnostic','gpu','helper'))
    need(str(out)==packet['paths']['gpu_output'], 'fixed GPU output differs')
    job = os.environ.get('SLURM_JOB_ID','')
    scheduler = gpu.scheduler_record(out,job)
    allocation_pin = save(out/'allocation.json',allocation(scheduler,job,helper,packet['resources']))
    scratch = Path(os.environ.get('SLURM_SCRATCH',''))
    need(scratch.is_absolute(), 'absolute node scratch required')
    scratch = scratch.resolve(strict=True)
    need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(), 'owned node scratch required')
    local = scratch/('ace-scientific-campaign-gpu-'+job)
    protect_inputs(ctx['old'],out,local)
    local.mkdir(mode=0o700,exist_ok=False)
    archive = module('pinned_campaign_archive',ctx['root']/'llm_intervention_runtime_archive.py',
                     data['llm_intervention_runtime_archive.py'])
    event = d.events(out,args.started,args.process_started,512*1024**2)
    event('stage_start')
    stage = d.stage_runtime(ctx['old'],archive,local/'runtime',event)
    stage_pin = save(out/'stage.json',stage)
    prefix = Path(stage['staged_prefix'])
    event('stage_complete',staged_prefix=str(prefix))
    save(out/'independent-provenance-sources.json',d.provenance_contract(prefix))
    binding = {'schema':ctx['launch'].VERSION,'job_id':job,'stage_path':str(out/'stage.json'),
        'stage_sha256':stage_pin,'staged_prefix':str(prefix),
        'contracts':{pid:packet['files'][name] for pid,name in packet['contracts'].items()}}
    binding_pin = save(out/'stage-binding.json',binding)
    for pid,contract in ctx['contracts'].items():
        ctx['launch'].derive_command(contract,contract_sha256=binding['contracts'][pid],binding=binding,stage=stage)
        (out/'services'/pid).mkdir(mode=0o700,parents=True,exist_ok=False)
    environment = gpu.child_environment(prefix,os.environ)
    environment['ACE_CAMPAIGN_STAGE_BINDING_SHA256'] = binding_pin
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    save(out/'child-environment.json',gpu.install_child_environment(environment))
    handoff = {'schema':VERSION,'packet_sha256':args.packet_sha256,'pid':os.getpid(),'job_id':job,
        'started_monotonic':args.started,'started_process_cpu':args.process_started,
        'stage_sha256':stage_pin,'binding_sha256':binding_pin,'allocation_sha256':allocation_pin,
        'admission_sha256':args.admission_sha256,'limits':packet['limits'],
        'controller_start_elapsed_seconds':time.monotonic()-args.started}
    handoff_pin = save(out/'handoff.json',handoff)
    command = [str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(ctx['root']/ENTRY),
        packet['files'][ENTRY],'--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,
        '--output',str(out),'--controller','--handoff-sha256',handoff_pin,
        '--admission',str(Path(args.admission).absolute()),'--admission-sha256',args.admission_sha256]
    save(out/'controller-command.json',{'command':command,'same_pid_exec':True,
        'first_gpu_deadline_monotonic':args.started+240,'work_deadline_monotonic':args.started+packet['limits']['work_seconds'],
        'final_deadline_monotonic':args.started+packet['limits']['final_seconds']})
    need(time.monotonic()<args.started+120, 'staging left inadequate three-service time')
    os.execve(command[0],command,environment)


def controller_handoff(args, ctx):
    out = Path(args.output).absolute()
    need(str(out)==ctx['packet']['paths']['gpu_output'], 'fixed controller output differs')
    handoff = decode(read(out/'handoff.json',args.handoff_sha256,64*1024))
    need(handoff['schema']==VERSION and handoff['packet_sha256']==args.packet_sha256
         and handoff['admission_sha256']==args.admission_sha256
         and handoff['limits']==ctx['packet']['limits'] and handoff['pid']==os.getpid() and handoff['job_id']==os.environ.get('SLURM_JOB_ID')
         and all(type(handoff[k]) in (int,float) and math.isfinite(handoff[k]) for k in
                 ('started_monotonic','started_process_cpu'))
         and 0<=time.monotonic()-handoff['started_monotonic']<240,
         'same-process bounded controller handoff differs')
    args.started,args.process_started = handoff['started_monotonic'],handoff['started_process_cpu']
    stage = decode(read(out/'stage.json',handoff['stage_sha256']))
    need(str(Path(sys.prefix).resolve())==stage['staged_prefix'] and sys.flags.isolated==1
         and sys.dont_write_bytecode and list(sys.version_info[:3])==[3,10,19], 'actual staged Python differs')
    binding = decode(read(out/'stage-binding.json',handoff['binding_sha256']))
    need(os.environ.get(ctx['launch'].BINDING_ENV)==handoff['binding_sha256']
         and binding['job_id']==handoff['job_id'], 'inherited stage binding differs')
    alloc = decode(read(out/'allocation.json',handoff['allocation_sha256']))
    need(alloc['fields']['JobId']==handoff['job_id'], 'allocation handoff differs')
    for pid,contract in ctx['contracts'].items():
        ctx['launch'].derive_command(contract,contract_sha256=ctx['packet']['files'][ctx['packet']['contracts'][pid]],
                                    binding=binding,stage=stage)
    return handoff,stage


def run_controller(args, ctx):
    out,packet,data = Path(args.output).absolute(),ctx['packet'],ctx['data']
    handoff,stage = controller_handoff(args,ctx)
    signal.setitimer(signal.ITIMER_REAL,max(.01,args.started+240-time.monotonic()))
    directory = out/'controller'
    directory.mkdir(mode=0o700,exist_ok=False)
    captured = {n:data[n] for n in packet['source_names']}
    loader = CapturedSources(captured,ctx['root'])
    sys.meta_path.insert(0,loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        journal = ctx['service'].Journal(directory/'runtime')
        observer = ServiceRuntime({'runtime':ctx['service'].RUNTIME},captured,loader,ctx['root']/ENTRY,
            globals(),out/'stage.json',handoff['stage_sha256'],journal,ctx['service'].read_pinned,
            ctx['service'].decode)
        observer.import_cpu_closure()
        import llm_intervention_campaign as campaign
        from llm_intervention_scientific_gpu_phase import run_gpu_phase
        from llm_intervention_scientific_service_evidence import validate_campaign_service
        from llm_intervention_contract import digest
        bundle = ctx['bundle']
        c,campaign_root = campaign._root(bundle['campaign'],bundle['campaign_sha256'],
            bundle['source_plan_pins'],packet['paths']['campaign'])
        for pid,value in ctx['acceptance']['attempt_pins'].items():
            attempt,closed = campaign._load_attempt(c,bundle['campaign_sha256'],
                campaign._attempt(campaign_root,c,pid),pid,value)
            need(closed is None and not attempt['errors'] and
                 all(a['status']=='completed' for a in attempt['arms']), 'CPU attempts not wholly successful')
        save(directory/'admitted.json',{'cpu_acceptance_sha256':packet['cpu_acceptance_sha256'],
            'campaign_sha256':bundle['campaign_sha256'],'cpu_attempt_pins':ctx['acceptance']['attempt_pins'],
            'lineage':ctx['lineage'],'scientific_qualified':False})
        # This function owns the actual persistent process handles, deadlines,
        # telemetry and fallback cleanup. It never calls private finalization.
        transfer_phase_deadline(args.started,packet['limits'])
        result = run_gpu_phase(
            campaign=c,campaign_sha256=bundle['campaign_sha256'],source_plan_pins=bundle['source_plan_pins'],
            campaign_root=campaign_root,supervision_root=directory/'phase',
            services={pid:{'worker_output_dir':contract['worker_output'],
                'freeze_sha256':packet['files'][packet['services'][pid]],
                'identity_sha256':digest(ctx['service'].identity(ctx['freezes'][pid]))}
                for pid,contract in ctx['contracts'].items()},
            outer_started=args.started,work_deadline=args.started+packet['limits']['work_seconds'],final_deadline=args.started+packet['limits']['final_seconds'],
            helper=ctx['helper'],helper_sha256=PROCESS_HELPER_SHA,
            deadline_helper=ctx['gpu'],deadline_helper_sha256=GPU_HELPER_SHA,limits=packet['limits'])
        need(result['status']=='gpu_phase_complete', 'GPU phase did not return complete closure')
        save(directory/'phase-result.candidate.json',result)
        signal.setitimer(signal.ITIMER_REAL,max(.01,args.started+packet['limits']['final_seconds']-time.monotonic()))
        observer.observe('after-campaign-host')
        cpu_checks = sys.modules['qualify_llm_intervention_service_cpu']
        runtime_checks = sys.modules['llm_intervention_service_runtime']
        provenance = decode(read(out/'independent-provenance-sources.json'))
        def controller_read(name):
            if name=='runtime-after-cpu-tokenization.json': name='runtime-after-campaign-host.json'
            return decode(read(journal.output/name))
        cpu_checks.validate_runtime_records(controller_read,stage,handoff['stage_sha256'],ctx['service'],
            ctx['diagnostic'],runtime_checks,provenance)
        need(controller_read('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,
             'host controller cannot admit device mappings')
        expected = {Path(n).stem:packet['files'][n] for n in packet['source_names'] if n!=ENTRY}
        for phase in ('after-service-imports','after-campaign-host'):
            projects = decode(read(journal.output/('runtime-'+phase+'.json')))['captured_project_modules']
            need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'}
                 and all(projects[n]['sha256']==p for n,p in expected.items()), 'controller captured closure differs')
        validations = {}
        for pid in POLICIES[5:]:
            attempt_pin = result['attempt_pins'][pid]
            freeze = ctx['freezes'][pid]
            worker = Path(ctx['contracts'][pid]['worker_output'])
            inventory_now = inventory(worker)
            joined = validate_campaign_service(campaign=c,campaign_sha256=bundle['campaign_sha256'],
                source_plan_pins=bundle['source_plan_pins'],campaign_root=campaign_root,policy_id=pid,
                worker_output_dir=worker,freeze_sha256=packet['files'][packet['services'][pid]],
                identity_sha256=digest(ctx['service'].identity(freeze)),attempt_sha256=attempt_pin,
                worker_evidence_sha256=inventory_now)
            save(directory/(pid+'-evidence.json'),joined)
            need(joined['operational_success'], 'service did not complete successfully: '+pid)
            runtime = ctx['gpu'].validate_runtime(worker.parent,freeze,packet['files'],stage,
                handoff['stage_sha256'],cpu_checks,ctx['service'],ctx['diagnostic'],runtime_checks,provenance)
            launch = decode(read(Path(ctx['contracts'][pid]['launch_output'])/'execution.json'))
            need(launch['stage_binding_sha256']==handoff['binding_sha256']
                 and launch['contract_sha256']==packet['files'][packet['contracts'][pid]], 'executed shim binding differs')
            validations[pid] = {'attempt_sha256':attempt_pin,'evidence_sha256':sha(read(directory/(pid+'-evidence.json'))),
                'runtime':runtime,'worker_inventory':inventory_now}
        need(inventory(packet['paths']['cpu_output'],exclude='worker/artifacts/campaign')==ctx['acceptance']['cpu_output_files'],
             'CPU output changed during GPU phase')
        final_campaign=inventory(packet['paths']['campaign'])
        need({n:h for n,h in final_campaign.items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}}
             ==ctx['acceptance']['campaign_files'],'campaign changed outside three GPU attempts')
        recheck_inputs(ctx)
        need(time.monotonic()<args.started+packet['limits']['final_seconds'], 'absolute campaign supervisor deadline exceeded')
        save(out/'terminal.json',{'schema':VERSION,'status':'development_gpu_campaign_complete',
            'packet_sha256':args.packet_sha256,'cpu_acceptance_sha256':packet['cpu_acceptance_sha256'],
            'admission_sha256':args.admission_sha256,'cpu_packet_sha256':packet['cpu_packet_sha256'],
            'campaign_sha256':bundle['campaign_sha256'],'stage_sha256':handoff['stage_sha256'],
            'attempt_pins':result['attempt_pins'],'service_validation':validations,
            'combined_attempt_pins':{**ctx['acceptance']['attempt_pins'],**result['attempt_pins']},
            'training_lineage':ctx['lineage'],'limits':packet['limits'],'resources':packet['resources'],
            'new_training_updates':0,'private_scores_computed':False,'scientific_qualified':False,
            'response_accounting_source':'original charged campaign arm journals',
            'token_decoding_replayed':False,'cpu_finalization_and_external_acceptance_required':True,
            'elapsed_seconds':time.monotonic()-args.started,
            'supervisor_process_cpu_seconds':time.process_time()-args.process_started,
            'helpers_and_service_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime})
    finally:
        sys.meta_path.remove(loader)


def transfer_phase_deadline(started,limits):
    # The phase separately enforces first verified work by started+240.
    # Its inherited alarm must represent the remaining overall work deadline.
    need(time.monotonic()<started+limits['work_seconds'], 'absolute GPU work deadline exceeded')
    signal.setitimer(signal.ITIMER_REAL,max(.01,started+limits['work_seconds']-time.monotonic()))


def validated_output(packet_path, packet_pin, output):
    """Read the pinned destination before claiming any output or writing failure."""
    packet = decode(read(packet_path,packet_pin,256*1024))
    paths = packet.get('paths',{})
    need(packet.get('schema')==VERSION and type(paths) is dict
         and set(paths)==PATH_KEYS
         and all(absolute(p) for p in paths.values()), 'fixed output paths required')
    roots = [Path(paths[k]) for k in ('cpu_inputs','cpu_output','gpu_inputs','gpu_output')]
    need(all(not p.is_symlink() for p in roots)
         and all(a.resolve()!=b.resolve() and a.resolve() not in b.resolve().parents and b.resolve() not in a.resolve().parents
             for i,a in enumerate(roots) for b in roots[i+1:])
         and paths['campaign']==str(roots[1]/'worker/artifacts/campaign'),
         'separated fixed input and output locations required')
    need(Path(packet_path).absolute().parent==Path(paths['gpu_inputs']), 'fixed GPU input root differs')
    out = Path(output).absolute()
    need(str(out)==paths['gpu_output'], 'fixed GPU output differs before claim')
    cpu_pin=packet.get('cpu_packet_sha256')
    need(pin(cpu_pin) and packet.get('files',{}).get('cpu-packet.json')==cpu_pin,'independent original CPU packet required')
    old=decode(read(Path(packet_path).absolute().parent/'cpu-packet.json',cpu_pin))
    protect_inputs(old,out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet','packet-sha256','admission','admission-sha256','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--controller',action='store_true')
    parser.add_argument('--handoff-sha256')
    args = parser.parse_args()
    args.started,args.process_started = time.monotonic(),time.process_time()
    out = validated_output(args.packet,args.packet_sha256,args.output)
    owned = False
    if args.controller:
        need(pin(args.handoff_sha256) and out.is_dir() and not out.is_symlink(), 'pinned existing handoff required')
        early = decode(read(out/'handoff.json',args.handoff_sha256,64*1024))
        need(early['pid']==os.getpid() and early['job_id']==os.environ.get('SLURM_JOB_ID')
             and type(early['started_monotonic']) in (int,float)
             and math.isfinite(early['started_monotonic'])
             and 0<=time.monotonic()-early['started_monotonic']<240,
             'bounded original handoff clock required before authentication')
        args.started = early['started_monotonic']
        owned = True  # Independently pinned, same-process prior exclusive claim.
    def interrupted(sig,frame):
        raise TimeoutError('campaign supervisor signal '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.01,args.started+240-time.monotonic()))
    try:
        ctx = authenticate(args.packet,args.packet_sha256,args.admission,args.admission_sha256)
        if args.controller:
            run_controller(args,ctx)
        else:
            out.mkdir(mode=0o700,exist_ok=False)
            owned = True
            save(out/'packet.json',ctx['packet'])
            stage_and_exec(args,ctx)
    except BaseException as exc:
        if owned and not (out/'failure.json').exists():
            failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                     'retry_allowed':False,'scientific_qualified':False,
                     'response_accounting_source':'original charged journals, including failed attempts'}
            try:save(out/'failure.json',failure)
            except BaseException:
                try:print(json.dumps(failure),file=sys.stderr,flush=True)
                except BaseException:pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
