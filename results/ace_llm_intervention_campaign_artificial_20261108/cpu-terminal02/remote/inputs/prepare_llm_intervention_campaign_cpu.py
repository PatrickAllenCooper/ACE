"""Captured CPU preparation and five-policy execution for one artificial campaign.

Packet v1 has an exact flat files roster (INPUT_NAMES), SOURCE_NAMES, the original
runtime/model pins, three persistent_specs/services, outer provenance, and an
absolute cpu_output. authenticate(path, sha256) returns
(root, packet, data, service, cpu, diagnostic); it is stdlib-only, reads adapters
opaquely, and neither stages nor imports the numerical runtime. GPU successors
must independently accept the saved CPU execution and bind their own runtime.

Normal invocation claims cpu_output exclusively. --worker is the isolated child;
--numeric-entry screen|ranker is the bounded captured CLI used by its policies.
--validate-saved performs an independent, stdlib-only saved-evidence check. None
of these modes submits a job. No retry, private finalization, or scientific claim.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.util
import io
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

VERSION = 'ace-artificial-campaign-cpu-v1'
ENTRY = 'prepare_llm_intervention_campaign_cpu.py'
TRAINING = 'prepare_llm_intervention_training_cpu.py'
CPU = 'qualify_llm_intervention_service_cpu.py'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
SERVICE = 'llm_intervention_model_service.py'
LINEAGE = 'llm_intervention_campaign_training_lineage.py'
LAUNCH = 'llm_intervention_campaign_launch.py'
PROJECT_PREFIXES = ('llm_intervention_', 'qualify_llm_intervention_', 'prepare_llm_intervention_')
HELPER_PINS = {
    TRAINING: 'b2b1425bcd010d5ddc5deeda0dc3bd89817c9fe4cae6c4cc6465ec9f7f998f43',
    CPU: 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5',
    DIAGNOSTIC: '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329',
    SERVICE: 'a7735dd35f56d7653a5ff42520db7fd67b98395e62333cccce31583317f7134e',
    LINEAGE: '3f3796576a6d2ddac7465dc73a918b3001dd0830fc548352baa6a77a50358f06',
    LAUNCH: 'd3a97faef71f01910405e785b20fd83710ed4a3ef86027c052ba3e303f2ffc25',
    'probe.py': '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951',
    'llm_intervention_runtime_archive.py': '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e'}
SOURCE_NAMES = frozenset(ENTRY.split()) | frozenset(n+'.py' for n in '''
llm_intervention_action_scoring llm_intervention_artificial_campaign
llm_intervention_campaign llm_intervention_campaign_cpu_phase
llm_intervention_campaign_launch llm_intervention_campaign_training_lineage
llm_intervention_canonical_scoring llm_intervention_checkpointing
llm_intervention_contract llm_intervention_cost_curves llm_intervention_episode
llm_intervention_episode_seal llm_intervention_gradient_probe
llm_intervention_grammar llm_intervention_language_policy llm_intervention_learning
llm_intervention_model_service llm_intervention_numerical_policy
llm_intervention_paired_episode llm_intervention_persistent_policy
llm_intervention_policy_cli llm_intervention_precision llm_intervention_ranker
llm_intervention_ranker_training llm_intervention_replicated_branches
llm_intervention_replicated_feedback llm_intervention_scm llm_intervention_screen_policy
llm_intervention_screen_protocol llm_intervention_screen_reporting
llm_intervention_service_runtime llm_intervention_supervision llm_intervention_target_risk
llm_intervention_task llm_intervention_training_dataset llm_intervention_training_engine
llm_intervention_training_exposure llm_intervention_training_plan llm_intervention_world
prepare_llm_intervention_training_cpu qualify_llm_intervention_checkpoint_gpu
qualify_llm_intervention_service_cpu qualify_llm_intervention_runtime_metadata
llm_intervention_gpu_mappings'''.split())
NUMERIC_SOURCE_NAMES = {
    'screen': frozenset('llm_intervention_'+n+'.py' for n in '''contract episode
        numerical_policy replicated_branches replicated_feedback scm screen_policy
        screen_protocol supervision target_risk task training_dataset world'''.split()),
    'ranker': frozenset('llm_intervention_'+n+'.py' for n in '''contract episode learning
        numerical_policy policy_cli ranker scm target_risk task'''.split())}
NUMERIC_POLICIES = ('uniform_random','menu_coverage','local_information','target_risk')
CPU_POLICIES = NUMERIC_POLICIES + ('learned_ranker',)
LLM_POLICIES = ('frozen_llm','sft_llm','dpo_llm')
ROSTER = ('artificial/17/snapshot-000','artificial/17/snapshot-001',
          'artificial/4/snapshot-000','artificial/4/snapshot-001')
DATA_NAMES = frozenset(('collection-plan.json','collection-inventory.json','collection-terminal.json',
    'prepared.json','gpu03-acceptance.json','engine-terminal.json','sft.checkpoint.json',
    'final.checkpoint.json','sft.adapter.pt','final.adapter.pt')) | frozenset(
    f'{k}-{i}.json' for k in ('record','receipt') for i in range(4)) | frozenset(
    f'{p}.{k}.json' for p in LLM_POLICIES for k in ('freeze','launch'))
INPUT_NAMES = SOURCE_NAMES | DATA_NAMES | {'probe.py','llm_intervention_runtime_archive.py'}
RESOURCES = dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
                 memory_mib=7680,wall_seconds=600,gpus=0,requeue=False)
LIMITS = dict(startup_seconds=120,request_seconds=30,idle_seconds=30,total_seconds=180,max_requests=8)
MODEL_FILES = {
    'config.json':'660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
    'generation_config.json':'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'model.safetensors':'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b',
    'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
PACKET_KEYS = frozenset(('schema','scope','record_roster','files','source_names','archive','manifest',
    'runtime_source','runtime','model_directory','model_files','resources','persistent_specs',
    'services','outer_provenance_receipt_sha256','cpu_output'))


class CampaignInterrupted(BaseException):
    """Propagate through controllers that preserve ordinary policy failures."""


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return sha(canonical(value).encode())


def decode(raw):
    def unique(rows):
        result = {}
        for key,value in rows:
            need(key not in result,'duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(raw,object_pairs_hook=unique)
    canonical(result)
    return result


def read(path, maximum=32*1024**2):
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum,'bounded regular file required')
        raw = handle.read(maximum+1)
        after = os.fstat(handle.fileno())
    need(len(raw)<=maximum and (before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_ino,after.st_size,after.st_mtime_ns),'input changed during capture')
    return raw


def capture(path, expected, maximum=32*1024**2):
    need(pin(expected),'independent byte pin required')
    raw = read(path,maximum)
    need(sha(raw)==expected,'input pin differs: '+str(path))
    return raw


def save(path, value):
    raw = (canonical(value)+'\n').encode()
    with Path(path).open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return sha(raw)


def module(name, path, raw):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),vars(result))
    return result


def absolute(value):
    return (type(value) is str and value.startswith('/') and '\x00' not in value
            and str(Path(value))==value and '..' not in Path(value).parts)


def separate(paths):
    paths = list(map(Path,paths))
    need(all(a!=b and a not in b.parents and b not in a.parents
             for i,a in enumerate(paths) for b in paths[i+1:]),'evidence/input roots overlap')


def paths(packet):
    out = Path(packet['cpu_output'])
    artifacts = out/'worker/artifacts'
    return dict(output=out,artifacts=artifacts,phase=artifacts/'cpu-phase',
                campaign=artifacts/'campaign',numeric=artifacts/'numeric-calls')


def packet_shape(packet, service, diagnostic, training):
    need(type(packet) is dict and set(packet)==PACKET_KEYS and packet['schema']==VERSION
         and packet['scope']=='artificial' and packet['record_roster']==list(ROSTER)
         and canonical(packet['resources'])==canonical(RESOURCES),'fixed campaign CPU packet differs')
    need(packet['source_names']==sorted(SOURCE_NAMES) and type(packet['files']) is dict
         and set(packet['files'])==INPUT_NAMES and all(pin(p) for p in packet['files'].values()),
         'exact flat source/input roster required')
    need(all(packet['files'][n]==p for n,p in {**HELPER_PINS,**training.COLLECTION_PINS}.items())
         and packet['files']['llm_intervention_gpu_mappings.py']==service.RUNTIME['gpu_mapping_source_sha256'],
         'accepted helper/collection pins differ')
    need(packet['runtime']==service.RUNTIME and packet['runtime_source']==diagnostic.RUNTIME_SOURCE
         and packet['model_files']==MODEL_FILES and absolute(packet['model_directory'])
         and absolute(packet['cpu_output']) and pin(packet['outer_provenance_receipt_sha256']),
         'model/runtime/output/provenance identity differs')
    for key,expected in (('archive',diagnostic.ARCHIVE_SHA),('manifest',diagnostic.MANIFEST_SHA)):
        row = packet[key]
        need(type(row) is dict and set(row)=={'path','sha256'} and row['sha256']==expected
             and absolute(row['path']),'original archive/manifest binding differs')
    need(type(packet['persistent_specs']) is dict and set(packet['persistent_specs'])==set(LLM_POLICIES)
         and type(packet['services']) is dict and set(packet['services'])==set(LLM_POLICIES),
         'three fixed persistent roles required')


def training_inputs(data):
    return {arg:data[name] for arg,name in (
        ('acceptance_raw','gpu03-acceptance.json'),('prepared_raw','prepared.json'),
        ('engine_terminal_raw','engine-terminal.json'),('sft_checkpoint_raw','sft.checkpoint.json'),
        ('sft_adapter_raw','sft.adapter.pt'),('final_checkpoint_raw','final.checkpoint.json'),
        ('final_adapter_raw','final.adapter.pt'))}


def admit(root, packet, data, service, diagnostic):
    lineage = module('campaign_lineage',root/LINEAGE,data[LINEAGE]).admit_training(**training_inputs(data))
    launch = module('campaign_launch',root/LAUNCH,data[LAUNCH])
    contracts, evidence = [], []
    for role in LLM_POLICIES:
        freeze = service.validate_freeze(decode(data[role+'.freeze.json']))
        need(freeze['scope']=='artificial' and freeze['model_directory']==packet['model_directory'] and freeze['model_files']==packet['model_files']
             and freeze['sources']=={n:packet['files'][n] for n in service.SOURCE_NAMES}
             and canonical(freeze['limits'])==canonical(LIMITS),'freeze model/source/bounds differ')
        if role=='frozen_llm':
            need(freeze['treatment']=='frozen' and freeze['adapter'] is None,'frozen role differs')
        else:
            stage = 'sft' if role=='sft_llm' else 'final'
            accepted = lineage['roles'][role]
            need(freeze['treatment']=='artificial_adapter' and freeze['adapter']=={
                'file':stage+'.adapter.pt','sha256':accepted['adapter_sha256'],
                'training_receipt':stage+'.checkpoint.json',
                'training_receipt_sha256':accepted['checkpoint_sha256'],'training_scope':'artificial'},
                'service adapter/checkpoint role differs')
        identity = service.identity(freeze)
        need(packet['services'][role]==dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=8),
             'client identity/bounds differ')
        c = launch.validate_contract(decode(data[role+'.launch.json']))
        need(c['policy_id']==role and c['entry']==str(root/SERVICE)
             and c['entry_sha256']==packet['files'][SERVICE] and c['freeze']==str(root/(role+'.freeze.json'))
             and c['freeze_sha256']==packet['files'][role+'.freeze.json']
             and c['archive_sha256']==packet['archive']['sha256'] and c['manifest_sha256']==packet['manifest']['sha256']
             and c['bootstrap']==diagnostic.BOOTSTRAP,'stable launch contract differs')
        command = [str(Path(packet['runtime_source'])/'bin/python'),'-I','-B','-c',diagnostic.BOOTSTRAP,
            str(root/LAUNCH),packet['files'][LAUNCH],'--contract',str(root/(role+'.launch.json')),
            '--contract-sha256',packet['files'][role+'.launch.json']]
        need(packet['persistent_specs'][role]==dict(policy_id=role,kind='persistent',command=command,
            timeout_seconds=35,identity_sha256=digest(identity),
            policy_provenance_sha256=packet['files'][role+'.freeze.json']),
            'persistent argv/identity/freeze provenance differs')
        contracts.append(c); evidence.extend([c['worker_output'],c['launch_output']])
    need(len({(c['binding_path'],c['stage_path']) for c in contracts})==1,'shared future stage binding required')
    separate([root,packet['cpu_output'],packet['model_directory'],*evidence])
    for c in contracts:
        separate([root,packet['cpu_output'],c['binding_path'],c['stage_path']])
    return lineage


def authenticate(packet_path, packet_sha256):
    """Pure full-byte authentication; stable six-item API for the GPU successor."""
    root = Path(packet_path).absolute().parent
    packet = decode(capture(packet_path,packet_sha256,256*1024))
    need(type(packet) is dict and type(packet.get('files')) is dict and set(packet['files'])==INPUT_NAMES,
         'exact flat input roster required before reads')
    data = {n:capture(root/n,p,16*1024**2 if n.endswith('.pt') else 8*1024**2)
            for n,p in packet['files'].items()}
    need(all(sha(data[n])==p for n,p in HELPER_PINS.items()),'accepted helper bytes differ')
    service = module('campaign_service',root/SERVICE,data[SERVICE])
    cpu = module('campaign_cpu_checks',root/CPU,data[CPU])
    diagnostic = module('campaign_diagnostic',root/DIAGNOSTIC,data[DIAGNOSTIC])
    training = module('campaign_training_checks',root/TRAINING,data[TRAINING])
    packet_shape(packet,service,diagnostic,training)
    # CURC ancestors may be legitimate mount aliases. Trust parent directories;
    # retain lexical contract paths and reject only a symlink at the root leaf.
    need(Path(__file__).absolute()==root/ENTRY and absolute(str(root)) and not root.is_symlink(),
         'captured entry/root differs')
    inventory = decode(data['collection-inventory.json'])
    need(inventory['record_roster']==list(ROSTER),'complete original roster differs')
    for i,rid in enumerate(ROSTER):
        need(packet['files'][f'record-{i}.json']==inventory['record_pins'][rid]
             and packet['files'][f'receipt-{i}.json']==inventory['supervision_receipt_pins'][rid],
             'original record/receipt pin differs')
    admit(root,packet,data,service,diagnostic)
    return root,packet,data,service,cpu,diagnostic


class CapturedSources(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self, captured, root):
        self.root,self.loaded = Path(root),{}
        self.sources = {Path(n).stem:b for n,b in captured.items() if n!=ENTRY}
        need(not any(n.startswith(PROJECT_PREFIXES) for n in sys.modules),
             'cached project modules preceded captured authentication')
    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/(fullname+'.py')))
        if fullname.startswith(PROJECT_PREFIXES):
            raise ImportError('project module outside captured closure: '+fullname)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, mod):
        mod.__file__ = str(self.root/(mod.__name__+'.py'))
        exec(compile(self.sources[mod.__name__],mod.__file__,'exec'),vars(mod))
        self.loaded[mod.__name__] = mod


def numeric_binding(world, policy):
    namespace = digest(dict(protocol='ace-first-screen-proposal-20261107-v1',scope='artificial',
                            world_id=world,purpose='trajectory',arm=policy))
    return dict(version='ace-screen-public-numerical-policy-20261107-v1',
        proposal_version='ace-first-screen-proposal-20261107-v1',
        seed_prefix=['artificial',world,'policy',namespace,'behavior'],scope='artificial',
        world_id=world,request_namespace=namespace,arm=policy,role='behavior',
        initial_actions=['observe_all']*2,initial_cost_units=2,total_budget=6,planning_draws=32)


def numeric_prefix(root, packet, packet_path, packet_pin, prefix, stage_pin, deadline, kind, bootstrap):
    return [str(Path(prefix)/'bin/python'),'-I','-B','-c',bootstrap,str(root/ENTRY),packet['files'][ENTRY],
        '--packet',str(Path(packet_path).absolute()),'--packet-sha256',packet_pin,
        '--numeric-entry',kind,'--runtime-stage',str(paths(packet)['output']/'stage.json'),
        '--runtime-stage-sha256',stage_pin,'--worker-deadline',str(deadline),'--']


def numeric_authenticate(packet_path, packet_pin, kind):
    """No archive/model/adapter scan and no runtime imports on a decision path."""
    root = Path(packet_path).absolute().parent
    packet = decode(capture(packet_path,packet_pin,256*1024))
    need(type(packet) is dict and type(packet.get('files')) is dict and set(packet['files'])==INPUT_NAMES,
         'exact numeric input roster required')
    names = NUMERIC_SOURCE_NAMES[kind] | {ENTRY,TRAINING,DIAGNOSTIC,SERVICE}
    data = {n:capture(root/n,packet['files'][n],1024**2) for n in names}
    need(all(sha(data[n])==HELPER_PINS[n] for n in (TRAINING,DIAGNOSTIC,SERVICE)),
         'numeric helper pins differ')
    service = module('numeric_service_contract',root/SERVICE,data[SERVICE])
    diagnostic = module('numeric_diagnostic_contract',root/DIAGNOSTIC,data[DIAGNOSTIC])
    training = module('numeric_training_contract',root/TRAINING,data[TRAINING])
    packet_shape(packet,service,diagnostic,training)
    need(Path(__file__).absolute()==root/ENTRY and absolute(str(root)) and not root.is_symlink(),
         'numeric captured entry differs')
    return root,packet,{n:data[n] for n in NUMERIC_SOURCE_NAMES[kind] | {ENTRY}}


def numeric_tail(kind, argv, packet):
    """An exact accepted CLI, not a caller-selected executable or import."""
    if kind=='screen':
        need(len(argv)==4 and argv[::2]==['--binding-json','--binding-sha256'], 'screen argv differs')
        binding = decode(argv[1])
        need(binding.get('world_id') in (9101,9102) and binding.get('arm') in NUMERIC_POLICIES,
             'screen world/policy differs')
        expected = numeric_binding(binding['world_id'],binding['arm'])
        need(argv[1]==canonical(expected) and argv[3]==digest(expected),'fixed screen binding differs')
    else:
        ranker = paths(packet)['phase']/'ranker'
        terminal = decode(read(ranker/'terminal.json'))
        export = terminal['exports']['dpo']
        expected = ['--method','learned_ranker','--seed','109','--draws','32',
            '--checkpoint',str(ranker/'dpo.ranker.pt'),'--checkpoint-sha256',export['sha256'],
            '--training-receipt-sha256',terminal['training_receipt_sha256']]
        need(terminal['status']=='complete' and terminal['completed_steps']==6
             and export['path']=='dpo.ranker.pt' and export['all_training_score_rows_equal'] is True
             and export['training_receipt_sha256']==terminal['training_receipt_sha256']
             and argv==expected,'fixed final DPO ranker argv differs')
        capture(ranker/'training_complete.json',terminal['training_receipt_sha256'])
    return argv


def check_numeric_runtime(packet, stage, kind, loader=None):
    """Observe used Python/module paths; archive integrity belongs to the parent."""
    prefix = Path(stage['staged_prefix'])
    need(absolute(str(prefix)) and str(prefix)!=packet['runtime_source']
         and Path(sys.prefix).resolve()==prefix.resolve()
         and Path(sys.executable).resolve()==(prefix/'bin/python').resolve()
         and sys.version_info[:3]==(3,10,19) and sys.flags.isolated==1 and sys.dont_write_bytecode,
         'same isolated staged interpreter required')
    need(stage['archive_sha256']==packet['archive']['sha256']
         and stage['manifest_sha256']==packet['manifest']['sha256'],'numeric stage pins differ')
    controls = dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU',
        LD_LIBRARY_PATH=str(prefix/'lib'))
    need(all(os.environ.get(k)==v for k,v in controls.items()) and
         not any(os.environ.get(k) for k in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD',
             'MKL_SERVICE_FORCE_INTEL','SLURM_JOB_GPUS','SLURM_STEP_GPUS')),'numeric environment differs')
    need(os.environ.get('ACE_CAMPAIGN_CPU_PGID')==str(os.getpgrp()) and os.getpgrp()!=os.getpid(),
         'numeric child escaped owned worker group')
    need(all(p and (Path(p)==prefix or prefix in Path(p).parents) for p in sys.path),
         'numeric import search path escaped stage')
    projects,used,stdlib = {},{},{}
    required = {'numpy'} | ({'torch'} if kind=='ranker' else set())
    for name,mod in list(sys.modules.items()):
        if name.startswith(PROJECT_PREFIXES):
            need(loader is not None and loader.loaded.get(name) is mod and getattr(mod,'__loader__',None) is loader,
                 'unapproved/cached numeric project module')
            projects[name+'.py'] = sha(loader.sources[name])
        file = getattr(mod,'__file__',None)
        if name.split('.')[0] in sys.stdlib_module_names and file and not file.startswith('<'):
            need(prefix.resolve() in Path(file).resolve().parents,'used standard library escaped stage')
            stdlib[name] = str(Path(file).resolve())
        if name.split('.')[0] in {'numpy','torch'} and file:
            need(prefix.resolve() in Path(file).resolve().parents,'used numerical module escaped stage')
            used[name] = str(Path(file).resolve())
    if loader is None:
        need(not {'numpy','torch','scipy','transformers'}.intersection(sys.modules),'numeric imports preceded authentication')
    else:
        need(required<=set(sys.modules) and (kind=='ranker' or 'torch' not in sys.modules)
             and not {'transformers','tokenizers'}.intersection(sys.modules),'numeric import scope differs')
        for name in required:
            need(sys.modules[name].__version__==packet['runtime']['imported_versions'][name],
                 'actual used numerical version differs')
        need(any(n.startswith('numpy.') and p.endswith('.so') for n,p in used.items())
             and (kind!='ranker' or 'torch._C' in used),'actual numerical extension paths required')
    return dict(prefix=str(prefix),python_version='.'.join(map(str,sys.version_info[:3])),
        pid=os.getpid(),pgid=os.getpgrp(),
        environment=controls,sys_path=list(sys.path),used_module_paths=used,standard_library_paths=stdlib,source_sha256=projects,
        versions={n:sys.modules[n].__version__ for n in required if n in sys.modules},
        archive_rescanned=False,llm_model_or_adapter_loaded=False)


def numeric_child(args):
    started,cpu_started = time.monotonic(),time.process_time()
    root,packet,data = numeric_authenticate(args.packet,args.packet_sha256,args.numeric_entry)
    need(args.runtime_stage==str(paths(packet)['output']/'stage.json'),'numeric stage path differs')
    stage = decode(capture(args.runtime_stage,args.runtime_stage_sha256))
    need(0 < args.worker_deadline-time.monotonic() <= 480,'numeric absolute worker deadline differs')
    tail = args.cli[1:] if args.cli[:1]==['--'] else args.cli
    numeric_tail(args.numeric_entry,tail,packet)
    public = sys.stdin.read(2*1024**2+1)
    need(len(public.encode())<=2*1024**2,'bounded public numeric input required')
    # Identity includes policy/world argv and the full public adaptive state.
    call_id = digest(dict(kind=args.numeric_entry,argv=tail,input_sha256=sha(public.encode())))
    out = paths(packet)['numeric']/call_id
    out.mkdir(mode=0o700,exist_ok=False)
    try:
        need(len(list(out.parent.iterdir()))<=40,'maximum forty numerical decisions exceeded')
        before = check_numeric_runtime(packet,stage,args.numeric_entry)
        save(out/'before.json',before)
        (out/'input.utf8').write_bytes(public.encode())
        save(out/'command.json',dict(entry=args.numeric_entry,argv=tail,packet_sha256=args.packet_sha256,
            stage_sha256=args.runtime_stage_sha256,source_sha256={n:sha(b) for n,b in data.items()},
            worker_deadline=args.worker_deadline))
        loader = CapturedSources(data,root)
        sys.meta_path.insert(0,loader)
        old_in,old_argv = sys.stdin,sys.argv
        stdout,stderr = io.StringIO(),io.StringIO()
        captured_stdin = io.TextIOWrapper(io.BytesIO(public.encode('utf-8')),encoding='utf-8',newline='')
        try:
            sys.stdin,sys.argv = captured_stdin,[str(root/ENTRY),*tail]
            name = 'llm_intervention_screen_policy' if args.numeric_entry=='screen' else 'llm_intervention_policy_cli'
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                result = importlib.import_module(name).main()
            need((type(result) is int and result==0) if args.numeric_entry=='screen' else result is None,
                 'numeric CLI returned failure status: '+repr(result))
            after = check_numeric_runtime(packet,stage,args.numeric_entry,loader)
        finally:
            sys.stdin,sys.argv = old_in,old_argv
            captured_stdin.close()
            sys.meta_path.remove(loader)
            (out/'stdout.log').write_text(stdout.getvalue())
            (out/'stderr.log').write_text(stderr.getvalue())
        save(out/'after.json',after)
        need(time.monotonic()<args.worker_deadline and time.monotonic()-started<28,'numeric deadline exceeded')
        save(out/'terminal.json',dict(status='complete',call_id=call_id,packet_sha256=args.packet_sha256,
            input_sha256=sha(public.encode()),stdout_sha256=sha(stdout.getvalue().encode()),
            stderr_sha256=sha(stderr.getvalue().encode()),source_sha256={n:sha(b) for n,b in data.items()},
            elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,retry_allowed=False))
        sys.stderr.write(stderr.getvalue()); sys.stdout.write(stdout.getvalue())
    except BaseException as exc:
        save(out/'failure.json',failure(exc))
        raise


def model_observation(packet, service):
    model = Path(packet['model_directory'])
    need(not model.is_symlink() and {p.name for p in model.iterdir()}==set(MODEL_FILES),
         'exact model directory required')
    return {n:service.read_pinned(model/n,p) for n,p in packet['model_files'].items()}


def child(args, root, packet, data, service, cpu, diagnostic):
    loc = paths(packet)
    need(os.getpgrp()==os.getpid(),'CPU worker must lead its owned process group')
    os.environ['ACE_CAMPAIGN_CPU_PGID'] = str(os.getpid())
    journal = service.Journal(loc['artifacts'])
    captured = {n:data[n] for n in SOURCE_NAMES}
    loader = CapturedSources(captured,root)
    sys.meta_path.insert(0,loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer = ServiceRuntime(packet,captured,loader,root/ENTRY,globals(),args.runtime_stage,
            args.runtime_stage_sha256,journal,service.read_pinned,service.decode)
        authenticated = dict(packet_sha256=args.packet_sha256,files=packet['files'],
            models=model_observation(packet,service),roles=admit(root,packet,data,service,diagnostic))
        journal.save('authenticated-before.json',authenticated)
        observer.import_cpu_closure()
        from transformers import AutoTokenizer
        from llm_intervention_campaign_cpu_phase import run_cpu_phase
        journal.event('campaign_tokenizer_load_start')
        tokenizer = AutoTokenizer.from_pretrained(packet['model_directory'],local_files_only=True,trust_remote_code=False)
        need(tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643,'accepted tokenizer IDs differ')
        loc['numeric'].mkdir(mode=0o700,exist_ok=False)
        prefixes = {kind:numeric_prefix(root,packet,args.packet,args.packet_sha256,observer.prefix,
            args.runtime_stage_sha256,args.worker_deadline,kind,diagnostic.BOOTSTRAP) for kind in ('screen','ranker')}
        journal.save('numeric-prefixes.json',prefixes)
        result = run_cpu_phase(output=loc['phase'],campaign_output=loc['campaign'],
            accepted_training_inputs=training_inputs(data),collection_inventory=decode(data['collection-inventory.json']),
            records={rid:data[f'record-{i}.json'] for i,rid in enumerate(ROSTER)},
            numeric_prefix=prefixes['screen'],ranker_prefix=prefixes['ranker'],
            persistent_specs=packet['persistent_specs'],services=packet['services'],
            numeric_provenance={p:digest({n:packet['files'][n] for n in NUMERIC_SOURCE_NAMES['screen']}) for p in NUMERIC_POLICIES},
            outer_provenance_receipt_sha256=packet['outer_provenance_receipt_sha256'],tokenizer=tokenizer,event=journal.event)
        need(result['status']=='cpu_phase_complete','failed CPU policies forbid admission')
        observer.observe('after-cpu-phase')
        # Re-read every captured input and model byte after execution. No tensor decoding.
        need(all(capture(root/n,p)==data[n] for n,p in packet['files'].items()),'CPU inputs changed')
        need(decode(capture(args.packet,args.packet_sha256))==packet,'CPU packet changed')
        after = {**authenticated,'models':model_observation(packet,service)}
        need(after==authenticated,'model/input identity changed')
        journal.save('authenticated-after.json',after)
        need(time.monotonic()<args.worker_deadline,'absolute worker deadline exceeded')
        journal.save('terminal.json',dict(schema=VERSION,status='cpu_campaign_complete',
            packet_sha256=args.packet_sha256,runtime_stage_sha256=args.runtime_stage_sha256,
            phase_terminal_sha256=sha(read(loc['phase']/'terminal.json')),campaign_root=str(loc['campaign']),
            attempt_pins=result['attempt_pins'],numeric_training_updates=6,llm_training_updates=0,
            model_weights_loaded=False,gpu_services_started=0,private_scores_computed=False,
            scientific_qualified=False,retry_allowed=False,external_acceptance_required=True))
    except BaseException as exc:
        journal.save('failure.json',failure(exc))
        raise
    finally:
        sys.meta_path.remove(loader)


def inventory(root, exclude=()):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(),'regular evidence root required')
    result = {}
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root).as_posix()
        if any(relative==n or relative.startswith(n+'/') for n in exclude):
            continue
        need(not p.is_symlink(),'symlink in evidence inventory')
        if p.is_file():
            result[relative] = sha(read(p))
        else:
            need(p.is_dir(),'nonregular evidence object')
    return result


def validate_ranker(phase, packet, data, lineage):
    plan = decode(read(phase/'numeric-training-plan.json'))
    prepared = decode(data['prepared.json'])
    original = decode(data['collection-inventory.json'])
    need(plan['scope']=='artificial' and plan['record_roster']==list(ROSTER)
         and plan['record_pins']==original['record_pins']
         and plan['supervision_receipt_pins']==original['supervision_receipt_pins']
         and plan['raw_records_utf8']=={rid:data[f'record-{i}.json'].decode() for i,rid in enumerate(ROSTER)}
         and plan['exposure']==prepared['exposure'] and plan['exposure_sha256']==lineage['exposure_sha256']
         and plan['checkpoint_steps']==[3,5]
         and plan['outer_provenance_receipt_sha256']==packet['outer_provenance_receipt_sha256']
         and plan['retry_allowed'] is False and plan['scientific_provenance_qualified'] is False,
         'complete ranker exposure/role/provenance differs')
    recipe = dict(version='ace-matched-numerical-training-v1',artificial_control=False,hidden=128,seed=109,
        dtype='float32',device='cpu',optimizer='Adam',learning_rate=.001,beta=.1,betas=[.9,.999],
        eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False,fused=False,gradient_clipping=None,
        optimizer_state='continuous_across_sft_dpo',torch_threads=1,deterministic_algorithms=True,
        exposure='one_sft_per_record_then_at_most_one_strict_pair_per_record')
    need(plan['recipe']==recipe,'full six-update ranker recipe differs')
    ranker = phase/'ranker'
    terminal_raw = read(ranker/'terminal.json')
    terminal = decode(terminal_raw)
    training = decode(capture(ranker/'training_complete.json',terminal['training_receipt_sha256']))
    need(read(ranker/'terminal-candidate.json')==terminal_raw and terminal['status']=='complete'
         and terminal['completed_steps']==6 and terminal['sft_steps']==4 and terminal['dpo_steps']==2
         and terminal['plan_sha256']==digest(plan) and terminal['exposure_sha256']==lineage['exposure_sha256']
         and terminal['environment_responses']==0 and terminal['retry_allowed'] is False
         and terminal['scientific_provenance_qualified'] is False
         and training['completed_steps']==6 and training['plan_sha256']==digest(plan)
         and training['exposure_sha256']==lineage['exposure_sha256'] and len(training['steps'])==6,
         'completed ranker training/export evidence differs')
    for index,row in enumerate(training['steps']):
        expected = plan['exposure']['numeric_schedule'][index]
        need(decode(read(ranker/f'step-{index:04d}.complete.json'))==row
             and row['index']==index and all(row[k]==v for k,v in expected.items())
             and row['optimizer_started'] is True and row['optimizer_returned'] is True
             and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']), 'ranker step journal differs')
        gradients = decode(read(ranker/f'step-{index:04d}.gradients.json'))['parameters']
        need(gradients and all(g['present'] is True and g['finite'] is True for g in gradients),
             'ranker gradient evidence differs')
    need(set(terminal['exports'])=={'sft','dpo'},'both fresh ranker exports required')
    for role,export in terminal['exports'].items():
        need(export['path']==role+'.ranker.pt' and export['all_training_score_rows_equal'] is True
             and export['training_receipt_sha256']==terminal['training_receipt_sha256'],
             'ranker fresh export/reload binding differs')
        capture(ranker/export['path'],export['sha256'])
    need(set(training['checkpoints'])=={'initial','sft','final'},'complete ranker checkpoints required')
    for name,p in training['checkpoints'].items():
        capture(ranker/(name+'.state.pt'),p)
    reference = decode(read(ranker/'reference.json'))
    need(reference['completed_sft_steps']==4 and reference['parameter_signature']==training['reference_signature']
         and reference['sft_checkpoint_sha256']==training['checkpoints']['sft'],'fixed ranker reference differs')
    capture(ranker/'reference.pt',reference['state_sha256'])
    return terminal


def validate_bundle(bundle, packet, prefixes, ranker, ranker_pin):
    campaign = bundle['campaign']
    need(bundle['campaign_sha256']==digest(campaign) and campaign['scope']=='artificial'
         and campaign['policy_ids']==list(CPU_POLICIES+LLM_POLICIES)
         and campaign['policy_phases']=={p:'cpu' if p in CPU_POLICIES else 'gpu' for p in CPU_POLICIES+LLM_POLICIES}
         and campaign['services']==packet['services'] and len(campaign['sources'])==2,'campaign roster/binding differs')
    plans = []
    for wid,source in zip((9101,9102),campaign['sources']):
        plan = source['plan']; plans.append(plan)
        need(plan['world']['world_id']==wid and plan['world']['scope']=='artificial'
             and source['plan_sha256']==digest(plan) and plan['initial_actions']==['observe_all']*2
             and plan['acquisition_cap_units']==4
             and plan['outer_provenance_receipt_sha256']==packet['outer_provenance_receipt_sha256'],
             'fixed world/plan/provenance differs')
        expected = []
        for p in NUMERIC_POLICIES:
            binding = numeric_binding(wid,p)
            expected.append(dict(policy_id=p,kind='task_process',command=prefixes['screen']+[
                '--binding-json',canonical(binding),'--binding-sha256',digest(binding)],timeout_seconds=30.,
                identity_sha256=None,policy_provenance_sha256=digest({n:packet['files'][n] for n in NUMERIC_SOURCE_NAMES['screen']})))
        export = ranker['exports']['dpo']
        expected.append(dict(policy_id='learned_ranker',kind='task_process',command=prefixes['ranker']+[
            '--method','learned_ranker','--seed','109','--draws','32','--checkpoint',
            str(paths(packet)['phase']/'ranker/dpo.ranker.pt'),'--checkpoint-sha256',export['sha256'],
            '--training-receipt-sha256',export['training_receipt_sha256']],timeout_seconds=30.,
            identity_sha256=None,policy_provenance_sha256=ranker_pin))
        expected += [packet['persistent_specs'][p] for p in LLM_POLICIES]
        need(plan['policy_specs']==expected,'all eight frozen policy commands/roles differ')
    need(bundle['source_plan_pins']==[dict(world_id=p['world']['world_id'],sha256=digest(p)) for p in plans],
         'source plan pins differ')
    return plans


def validate_tokens(value, plans):
    need(value['scope']=='fabricated shape probes only' and value['environment_responses']==0
         and value['no_all_adaptive_prompt_bound_claim'] is True and len(value['rows'])==4,
         'four explicitly fake prompt probes required')
    for row,(plan,count) in zip(value['rows'],((p,c) for p in plans for c in (2,5))):
        envelope,state = row['envelope'],row['envelope']['state']
        graph = plan['learner']['graph']
        need(row['world_id']==plan['world']['world_id'] and row['fixture']==(
            'fabricated_initial_shape' if count==2 else 'fabricated_late_affordability_shape')
            and row['environment_responses']==0 and set(envelope)=={'version','state','task'}
            and envelope['version']=='ace-public-prediction-task-v1' and envelope['task']==plan['task']
            and row['envelope_sha256']==digest(envelope) and row['task_sha256']==digest(envelope['task'])
            and state['graph']==graph and state['menu']==plan['menu']
            and state['objective']==plan['objective'] and state['domain_notes']==plan['domain_notes']
            and state['requests']==state['returned_responses']==state['spent_units']==count
            and state['remaining_units']==6-count and state['terminal'] is False
            and state['history']==[dict(request_id=i,action_id='observe_all',kind='observe',targets={},
                values={n['name']:0. for n in graph},cost_units=1) for i in range(count)],
            'fabricated public envelope binding differs')
        expected_scm = []
        for n in graph:
            width = 1+len(n['parents'])
            expected_scm.append(dict(node=n['name'],terms=['intercept',*n['parents']],parameter_mean=[0.]*width,
                parameter_covariance=[[float(i==j) for j in range(width)] for i in range(width)],
                noise_variance=.25,eligible_rows=count))
        need(state['scm']==expected_scm,'fake summary shape differs')
        prompt,commands = row['prompt_ids'],row['completion_ids']
        need(prompt and row['prompt_tokens']==len(prompt) and len(prompt)+32<=8192
             and row['max_tokens']==8192 and row['max_new_tokens']==32 and row['enable_thinking'] is False
             and commands and row['legal_commands']==len(commands)
             and row['longest_command_tokens']==max(map(len,commands)) and pin(row['prompt_sha256'])
             and all(type(t) is int and t>=0 for t in prompt)
             and all(ids and len(ids)<=32 and ids[-1]==151645 and all(type(t) is int and t>=0 for t in ids)
                     for ids in commands),'bounded actual tokenization differs')


def validate_attempts(campaign_root, bundle, attempts):
    c = bundle['campaign']; policies = list(CPU_POLICIES)
    need(set(attempts)==set(policies),'exact five attempt pins required')
    need({p.name for p in campaign_root.iterdir()}=={'campaign.json','source-plan-pins.json','start.json'} |
         {f'policy-{i:02d}' for i in range(5)},'GPU attempts/private finalization/retry forbidden in CPU result')
    need(decode(read(campaign_root/'campaign.json'))==c
         and decode(read(campaign_root/'source-plan-pins.json'))==bundle['source_plan_pins'],
         'initial campaign root differs')
    processes = []
    for index,pid in enumerate(policies):
        directory = campaign_root/f'policy-{index:02d}'
        raw = capture(directory/'attempt.json',attempts[pid]); attempt = decode(raw)
        need(raw==read(directory/'attempt.candidate.json') and attempt['policy_id']==pid
             and attempt['campaign_sha256']==bundle['campaign_sha256'] and attempt['phase']=='cpu'
             and attempt['source_plan_pins']==bundle['source_plan_pins'] and attempt['errors']==[]
             and attempt['retry_allowed'] is False and attempt['private_scores_computed'] is False
             and attempt['service']==dict(status='not_started',file=None,sha256=None)
             and not (directory/'service').exists() and len(attempt['arms'])==2
             and inventory(directory,('attempt.json','attempt.candidate.json'))==attempt['evidence_sha256'],
             'CPU attempt identity/completion/custody differs')
        for i,(arm,source) in enumerate(zip(attempt['arms'],c['sources'])):
            need(arm['status']=='completed' and arm['reason'] is None and arm['path']==f'world-{i:02d}'
                 and arm['world_id']==source['plan']['world']['world_id'],'both CPU world arms must complete')
            path = directory/arm['path']
            arm_raw = capture(path/'arm-receipt.json',arm['receipt_sha256']); receipt = decode(arm_raw)
            need(read(path/'arm-receipt.candidate.json')==arm_raw and receipt['status']=='raw_ready'
                 and receipt['plan_sha256']==source['plan_sha256'] and receipt['policy_id']==pid
                 and receipt['new_training_updates']==0 and receipt['private_scores_computed'] is False
                 and receipt['retry_allowed'] is False and receipt['scientific_provenance_qualified'] is False
                 and inventory(path,('arm-receipt.json','arm-receipt.candidate.json'))==receipt['source_evidence_sha256'],
                 'complete arm custody/scope differs')
            diagnostics = decode(read(path/'execution-diagnostics.json'))
            need(diagnostics['episode_returned_cleanly'] is True and diagnostics['adapters_closed_cleanly'] is True,
                 'arm execution failed')
            command = source['plan']['policy_specs'][index]['command']
            for p in sorted(path.rglob('decision-*.process.json')):
                record = decode(read(p))
                need(record.get('returncode')==0 and record['command']==command,'numerical decision failed/argv differs')
                input_path = p.with_name(p.name.replace('.process.json','.input.json'))
                need(digest(decode(read(input_path)))==record['input_sha256'],'public process input pin differs')
                processes.append(record)
    need(10<=len(processes)<=40,'bounded complete numerical decision inventory required')
    return processes


def validate_numeric_calls(directory, processes, packet, packet_pin, stage, stage_pin, owned_pgid):
    need(directory.is_dir() and len(list(directory.iterdir()))==len(processes),'numeric call inventory differs')
    seen = set()
    for process in processes:
        command = process['command']; split = command.index('--')
        kind = command[command.index('--numeric-entry')+1]; argv = command[split+1:]
        cid = digest(dict(kind=kind,argv=argv,input_sha256=process['input_sha256']))
        need(cid not in seen,'numerical attempt repeated'); seen.add(cid)
        out = directory/cid
        need({p.name for p in out.iterdir()}=={'before.json','after.json','input.utf8','command.json',
             'stdout.log','stderr.log','terminal.json'},'incomplete numeric attempt')
        r = decode(read(out/'terminal.json')); launched = decode(read(out/'command.json'))
        expected = {n:packet['files'][n] for n in NUMERIC_SOURCE_NAMES[kind]|{ENTRY}}
        need(r['status']=='complete' and r['call_id']==cid and r['packet_sha256']==packet_pin
             and r['input_sha256']==process['input_sha256']==sha(read(out/'input.utf8'))
             and r['stdout_sha256']==sha(process['stdout'].encode())==sha(read(out/'stdout.log'))
             and r['stderr_sha256']==sha(process['stderr'].encode())==sha(read(out/'stderr.log'))
             and r['source_sha256']==expected and 0<=r['elapsed_seconds']<28 and r['cpu_seconds']>=0
             and r['retry_allowed'] is False and launched['entry']==kind and launched['argv']==argv
             and launched['source_sha256']==expected and launched['packet_sha256']==packet_pin
             and launched['stage_sha256']==stage_pin,'numeric call/response/captured source differs')
        before,after = decode(read(out/'before.json')),decode(read(out/'after.json'))
        required = {'numpy'} | ({'torch'} if kind=='ranker' else set())
        need(before['versions']=={} and before['used_module_paths']=={} and before['source_sha256']=={}
             and after['versions']=={n:packet['runtime']['imported_versions'][n] for n in required}
             and after['source_sha256'] and set(after['source_sha256'])<=NUMERIC_SOURCE_NAMES[kind]
             and all(expected[n]==p for n,p in after['source_sha256'].items()),'used numeric runtime/source differs')
        for observation in (before,after):
            need(observation['prefix']==stage['staged_prefix'] and observation['python_version']=='3.10.19'
                 and type(observation['pid']) is int and observation['pid']>0
                 and observation['pgid']==owned_pgid and observation['pid']!=owned_pgid
                 and observation['archive_rescanned'] is False and observation['llm_model_or_adapter_loaded'] is False
                 and all(Path(stage['staged_prefix']).resolve() in Path(p).parents
                         for p in [*observation['used_module_paths'].values(),*observation['standard_library_paths'].values()]),
                 'numeric stage observation differs')
        need(before['pid']==after['pid'],'numeric process identity changed')


def validate_worker(root, packet, packet_pin, data, service, cpu, diagnostic, stage, stage_pin, provenance,
                    *, output=None, deadline=None, packet_path=None):
    """Saved-byte checks only: no tensor decoding, tokenization, fitting or responses."""
    out = Path(output) if output is not None else paths(packet)['artifacts']
    def value(name):
        return decode(read(out/name))
    need(not list(out.rglob('failure.json')),'failed worker cannot qualify')
    lineage = admit(root,packet,data,service,diagnostic)
    before,after = value('authenticated-before.json'),value('authenticated-after.json')
    need(before==after and before['packet_sha256']==packet_pin and before['files']==packet['files']
         and before['roles']==lineage and set(before['models'])==set(MODEL_FILES)
         and all(r['sha256']==MODEL_FILES[n] and type(r['bytes']) is int and r['bytes']>0
                 for n,r in before['models'].items()),'source/input/model before-after identity differs')
    checks = module('campaign_runtime_checks',root/'llm_intervention_service_runtime.py',data['llm_intervention_service_runtime.py'])
    cpu.validate_runtime_records(lambda n:value('runtime-after-cpu-phase.json' if n=='runtime-after-cpu-tokenization.json' else n),
        stage,stage_pin,service,diagnostic,checks,provenance)
    need(value('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,'CPU GPU mapping exception forbidden')
    expected = {Path(n).stem:packet['files'][n] for n in SOURCE_NAMES if n!=ENTRY}
    for phase in ('after-service-imports','after-cpu-phase'):
        projects = value('runtime-'+phase+'.json')['captured_project_modules']
        need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'}
             and all(projects[n]['sha256']==p for n,p in expected.items()),'full captured import closure differs')
    phase = out/'cpu-phase'
    need(decode(read(phase/'training-lineage.json'))==lineage
         and decode(read(phase/'collection-inventory.json'))==decode(data['collection-inventory.json']),
         'saved complete collection/training admission differs')
    ranker = validate_ranker(phase,packet,data,lineage)
    bundle = decode(read(phase/'campaign-bundle.json')); prefixes = value('numeric-prefixes.json')
    need(deadline is not None and packet_path is not None,'independent launch/deadline binding required')
    need(prefixes=={kind:numeric_prefix(root,packet,packet_path,packet_pin,stage['staged_prefix'],stage_pin,
        deadline,kind,diagnostic.BOOTSTRAP) for kind in ('screen','ranker')},'captured numeric prefixes differ')
    plans = validate_bundle(bundle,packet,prefixes,ranker,sha(read(phase/'ranker/terminal.json')))
    validate_tokens(decode(read(phase/'tokenization.json')),plans)
    attempts = decode(read(phase/'attempt-pins.json'))
    processes = validate_attempts(out/'campaign',bundle,attempts)
    owned_pgid = decode(read(out.parent.parent/'worker.started.json'))['pgid']
    validate_numeric_calls(out/'numeric-calls',processes,packet,packet_pin,stage,stage_pin,owned_pgid)
    result = decode(read(phase/'terminal.json')); terminal = value('terminal.json')
    need(result['status']=='cpu_phase_complete' and result['failed_policies']==[]
         and result['attempt_pins']==attempts and result['campaign_sha256']==bundle['campaign_sha256']
         and result['numeric_training_updates']==6 and result['llm_training_updates']==result['gpu_services_started']==0
         and all(result[k] is False for k in ('private_scores_computed','scientific_qualified','retry_allowed'))
         and result['external_acceptance_required'] is True,'CPU phase result differs')
    need(terminal==dict(schema=VERSION,status='cpu_campaign_complete',packet_sha256=packet_pin,
        runtime_stage_sha256=stage_pin,phase_terminal_sha256=sha(read(phase/'terminal.json')),
        campaign_root=str(paths(packet)['campaign']),attempt_pins=attempts,numeric_training_updates=6,
        llm_training_updates=0,model_weights_loaded=False,gpu_services_started=0,private_scores_computed=False,
        scientific_qualified=False,retry_allowed=False,external_acceptance_required=True),'worker terminal differs')
    events = [value(p.name) for p in sorted(out.glob('phase-*.json'))]
    ordered = ['service_cpu_import_start','service_cpu_import_complete','campaign_tokenizer_load_start',
        'cpu_training_lineage_start','cpu_ranker_training_start','cpu_ranker_training_complete',
        'campaign_frozen_before_responses','campaign_tokenization_start','campaign_tokenization_complete']
    names = [r['phase'] for r in events]
    need(all(names.count(n)==1 for n in ordered) and [names.index(n) for n in ordered]==sorted(names.index(n) for n in ordered)
         and names.count('cpu_policy_start')==names.count('cpu_policy_complete')==5,'once-only phase order differs')
    starts = [r['policy_id'] for r in events if r['phase']=='cpu_policy_start']
    ends = [r['policy_id'] for r in events if r['phase']=='cpu_policy_complete']
    need(starts==ends==list(CPU_POLICIES) and names.index('cpu_policy_start')>names.index(ordered[-1]),
         'responses preceded complete command/token freeze')
    return dict(terminal_sha256=sha(read(out/'terminal.json')),campaign_sha256=bundle['campaign_sha256'],
        source_plan_pins=bundle['source_plan_pins'],attempt_pins=attempts,numeric_decisions=len(processes),
        roles=lineage['roles'],numeric_training_updates=6,completed_cpu_cells=10,
        gpu_cells_executed=0,private_scores_computed=False,scientific_qualified=False)


def allocation_fields(raw, job):
    fields = dict(re.findall(r'(\w+)=(\S+)',raw))
    expected = dict(JobId=job,JobState='RUNNING',Account='ucb736_asc2',Partition='acpu',QOS='cpu-normal',
        NumCPUs='2',NumTasks='1',NumNodes='1',TimeLimit='00:10:00',Requeue='0',Restarts='0')
    tres = dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    need(all(fields.get(k)==v for k,v in expected.items()) and tres.get('cpu')=='2'
         and tres.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in tres),
         'fixed two-CPU/7680MiB/600s/zero-GPU allocation differs')
    return fields


def failure(exc):
    return dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                retry_allowed=False,scientific_qualified=False)


def supervise_child(command, env, out, child_deadline, supervisor_deadline, kill_owned):
    """Absolute bounded wait. Cleanup precedes validation and any error receipt."""
    proc,cleanup = None,None
    started = time.monotonic()
    need(started<child_deadline<supervisor_deadline and supervisor_deadline-child_deadline>=5,
         'no time left for bounded child and owned cleanup')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        signal.setitimer(signal.ITIMER_REAL,max(.001,child_deadline-time.monotonic()))
        with (out/'worker.stdout.log').open('xb') as stdout,(out/'worker.stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            save(out/'worker.started.json',dict(command=command,pid=proc.pid,pgid=proc.pid,
                deadline_monotonic=child_deadline,supervisor_deadline_monotonic=supervisor_deadline,
                started_monotonic=started))
            remaining = child_deadline-time.monotonic()
            need(remaining>0,'child launch receipt consumed deadline')
            rc = proc.wait(timeout=remaining)
        need(rc==0 and time.monotonic()<child_deadline,'CPU child failed or exceeded absolute deadline')
    finally:
        # Ignore repeated external interrupts only in this bounded cleanup window.
        # A live descendant after a natural worker exit makes this attempt fail.
        signal.setitimer(signal.ITIMER_REAL,0)
        old = {s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
        cleanup_deadline = min(time.monotonic()+5,supervisor_deadline)
        signal.setitimer(signal.ITIMER_REAL,max(.001,cleanup_deadline-time.monotonic()))
        try:
            if proc is not None:
                cleanup = kill_owned(proc)
                save(out/'worker.cleanup.json',cleanup)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            save(out/'worker.resources.json',dict(elapsed_seconds=time.monotonic()-started,
                child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                cumulative_child_peak_rss_bytes=after.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))
            need(time.monotonic()<cleanup_deadline,'owned cleanup exceeded five seconds')
        finally:
            for s,handler in old.items():
                signal.signal(s,handler)
            signal.setitimer(signal.ITIMER_REAL,max(.001,supervisor_deadline-time.monotonic()))
    need(cleanup is not None and cleanup['group_still_exists'] is False and not cleanup['errors']
         and cleanup['returncode']==0 and cleanup['kill_sent'] is False,
         'natural CPU exit with no remaining owned group required')
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(returncode=rc,cleanup=cleanup,elapsed_including_cleanup_seconds=time.monotonic()-started,
        child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        cumulative_child_peak_rss_bytes=after.ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def parent(args, root, packet, data, service, cpu, diagnostic):
    out = paths(packet)['output']
    started,process_started = args.started,args.process_started
    deadline = started+480
    save(out/'packet.json',packet)
    save(out/'start.json',dict(schema=VERSION,packet_path=str(Path(args.packet).absolute()),
        packet_sha256=args.packet_sha256,started_monotonic=started,worker_deadline=deadline,
        supervisor_deadline=started+580,resources=RESOURCES,retry_allowed=False))
    job = os.environ.get('SLURM_JOB_ID','')
    need(job.isdigit() and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),
         'CPU Slurm job without GPU allocation required')
    query = subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
    save(out/'allocation.json',dict(stdout=query.stdout,stderr=query.stderr,returncode=query.returncode))
    need(query.returncode==0,'CPU allocation query failed')
    allocation_fields(query.stdout,job)
    scratch = Path(os.environ.get('SLURM_SCRATCH',''))
    need(scratch.is_absolute(),'absolute node scratch required')
    scratch = scratch.resolve(strict=True)
    need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
    local = scratch/('ace-campaign-cpu-'+job)
    local.mkdir(mode=0o700,exist_ok=False)
    archive = module('campaign_archive',root/'llm_intervention_runtime_archive.py',data['llm_intervention_runtime_archive.py'])
    probe = module('campaign_cleanup',root/'probe.py',data['probe.py'])
    training = module('campaign_environment',root/TRAINING,data[TRAINING])
    event = diagnostic.events(out,started,process_started,512*1024**2)
    event('stage_start')
    stage = diagnostic.stage_runtime(packet,archive,local/'runtime',event)
    stage_pin = save(out/'stage.json',stage)
    event('stage_complete',staged_prefix=stage['staged_prefix'])
    prefix = Path(stage['staged_prefix'])
    provenance = diagnostic.provenance_contract(prefix)
    save(out/'independent-provenance-sources.json',provenance)
    command = [str(prefix/'bin/python'),'-I','-B','-c',diagnostic.BOOTSTRAP,str(root/ENTRY),packet['files'][ENTRY],
        '--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,'--worker',
        '--runtime-stage',str(out/'stage.json'),'--runtime-stage-sha256',stage_pin,'--worker-deadline',str(deadline)]
    session = supervise_child(command,training.cpu_environment(prefix,os.environ),out,deadline,started+580,probe.kill_owned)
    summary = validate_worker(root,packet,args.packet_sha256,data,service,cpu,diagnostic,stage,stage_pin,provenance,
        deadline=deadline,packet_path=args.packet)
    need(time.monotonic()<started+580,'absolute supervisor deadline exceeded')
    save(out/'supervisor.json',dict(schema=VERSION,status='cpu_campaign_complete',job_id=job,
        packet_sha256=args.packet_sha256,stage_sha256=stage_pin,session=session,validation=summary,
        elapsed_seconds=time.monotonic()-started,supervisor_cpu_seconds=time.process_time()-process_started,
        retry_allowed=False,scientific_qualified=False,independent_acceptance_required=True))


def validate_saved(args, root, packet, data, service, cpu, diagnostic):
    out = paths(packet)['output']
    need(not (out/'failure.json').exists(),'failed supervisor cannot qualify')
    supervisor = decode(read(out/'supervisor.json'))
    start = decode(read(out/'start.json')); launch = decode(read(out/'worker.started.json'))
    stage = decode(capture(out/'stage.json',supervisor['stage_sha256']))
    need(decode(read(out/'packet.json'))==packet and start['packet_sha256']==args.packet_sha256
         and start['packet_path']==str(Path(args.packet).absolute()) and start['resources']==RESOURCES
         and start['worker_deadline']==start['started_monotonic']+480
         and start['supervisor_deadline']==start['started_monotonic']+580
         and launch['deadline_monotonic']==start['worker_deadline']
         and launch['supervisor_deadline_monotonic']==start['supervisor_deadline']
         and launch['pid']==launch['pgid'] and type(launch['pid']) is int and launch['pid']>0,
         'saved absolute launch/resource contract differs')
    expected_command = [str(Path(stage['staged_prefix'])/'bin/python'),'-I','-B','-c',diagnostic.BOOTSTRAP,
        str(root/ENTRY),packet['files'][ENTRY],'--packet',start['packet_path'],'--packet-sha256',args.packet_sha256,
        '--worker','--runtime-stage',str(out/'stage.json'),'--runtime-stage-sha256',supervisor['stage_sha256'],
        '--worker-deadline',str(start['worker_deadline'])]
    need(launch['command']==expected_command,'actual captured worker argv differs')
    allocated = decode(read(out/'allocation.json'))
    need(allocated['returncode']==0,'saved allocation query failed')
    allocation_fields(allocated['stdout'],supervisor['job_id'])
    session = supervisor['session']; cleanup = decode(read(out/'worker.cleanup.json'))
    usage = decode(read(out/'worker.resources.json'))
    need(supervisor['schema']==VERSION and supervisor['status']=='cpu_campaign_complete'
         and supervisor['packet_sha256']==args.packet_sha256 and 0<supervisor['elapsed_seconds']<580
         and supervisor['retry_allowed'] is False and supervisor['scientific_qualified'] is False
         and supervisor['independent_acceptance_required'] is True and session['returncode']==0
         and session['cleanup']==cleanup and cleanup['pid']==cleanup['pgid']==launch['pid']
         and cleanup['kill_sent'] is False and cleanup['group_still_exists'] is False
         and cleanup['returncode']==0 and cleanup['errors']==[]
         and usage['child_and_cleanup_cpu_seconds']>=0 and usage['cumulative_child_peak_rss_bytes']>=0
         and session['elapsed_including_cleanup_seconds']<=start['worker_deadline']-launch['started_monotonic']+5,
         'successful bounded owned-group closure differs')
    summary = validate_worker(root,packet,args.packet_sha256,data,service,cpu,diagnostic,stage,
        supervisor['stage_sha256'],decode(read(out/'independent-provenance-sources.json')),
        deadline=start['worker_deadline'],packet_path=start['packet_path'])
    need(summary==supervisor['validation'],'independent saved validation differs')
    # The campaign is deliberately a separate inventory: a GPU successor adds
    # new policy attempts there without mutating the closed CPU-only inventory.
    receipt = dict(schema=VERSION,status='saved_cpu_campaign_validated',packet_sha256=args.packet_sha256,
        supervisor_sha256=sha(read(out/'supervisor.json')),cpu_output=str(out),campaign_root=str(paths(packet)['campaign']),
        cpu_output_files=inventory(out,('saved-validation','worker/artifacts/campaign')),
        initial_campaign_files=inventory(paths(packet)['campaign']),**summary,
        archive_execution_independently_accepted=False,external_acceptance_required=True)
    save(out/'saved-validation/validation.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True)
    parser.add_argument('--packet-sha256',required=True)
    parser.add_argument('--output',help='optional assertion of packet cpu_output; never overrides it')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--worker',action='store_true')
    modes.add_argument('--numeric-entry',choices=('screen','ranker'))
    modes.add_argument('--validate-saved',action='store_true')
    parser.add_argument('--runtime-stage')
    parser.add_argument('--runtime-stage-sha256')
    parser.add_argument('--worker-deadline',type=float)
    parser.add_argument('cli',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.started,args.process_started = time.monotonic(),time.process_time()
    def interrupted(sig, frame):
        raise CampaignInterrupted('campaign CPU signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):
        signal.signal(sig,interrupted)
    limit = 28 if args.numeric_entry else (480 if args.worker else 580)
    if args.worker or args.numeric_entry:
        need(args.worker_deadline is not None and 0<args.worker_deadline-args.started<=480
             and args.runtime_stage and pin(args.runtime_stage_sha256),'absolute worker/stage binding required')
        limit = min(limit,args.worker_deadline-args.started)
    signal.setitimer(signal.ITIMER_REAL,limit)
    claimed = None
    try:
        if args.numeric_entry:
            need(args.output is None,'numeric output cannot be overridden')
            numeric_child(args)
            return
        need(not args.cli,'arbitrary CPU commands forbidden')
        # Claim only the byte-pinned, predetermined output. A failure after this
        # claim is retained; an existing attempt is never opened for writing.
        packet = decode(capture(args.packet,args.packet_sha256,256*1024))
        need(type(packet) is dict and absolute(packet.get('cpu_output')),'fixed absolute CPU output required')
        out = Path(packet['cpu_output'])
        need(args.output is None or args.output==str(out),'output differs from pinned packet')
        root = Path(args.packet).absolute().parent
        separate([out,root])
        for key in ('model_directory','runtime_source'):
            need(absolute(packet.get(key)),'absolute input/runtime path required')
            separate([out,packet[key]])
        for key in ('archive','manifest'):
            need(type(packet.get(key)) is dict and absolute(packet[key].get('path')),'absolute archive input required')
            separate([out,packet[key]['path']])
        claimed = out/'saved-validation' if args.validate_saved else (out/'worker' if args.worker else out)
        candidate = claimed
        claimed = None
        candidate.mkdir(mode=0o700,exist_ok=False)
        claimed = candidate
        context = authenticate(args.packet,args.packet_sha256)
        if args.worker:
            need(args.runtime_stage==str(out/'stage.json'),'worker stage path differs')
            child(args,*context)
        elif args.validate_saved:
            validate_saved(args,*context)
        else:
            need(not args.runtime_stage and not args.runtime_stage_sha256 and args.worker_deadline is None,
                 'parent cannot accept a supplied stage or deadline')
            parent(args,*context)
    except BaseException as exc:
        if claimed is not None and not (claimed/'failure.json').exists():
            save(claimed/'failure.json',failure(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
