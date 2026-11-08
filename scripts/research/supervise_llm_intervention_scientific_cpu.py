"""Two fixed scientific CPU stages: realize OR collect. Never auto-chain them.

CLI uses the existing pinned diagnostic BOOTSTRAP and independent entry/packet/
admission pins. Parent owns the scientific freeze, namespace/controller locks,
ledger, actual-result acceptance and once-only submission. No tokenizer or model
files are admitted. Resource profile is exactly 2 CPU / 7680 MiB / zero GPU.
Realization seals 56 descriptors; collection stages only 32 training payloads.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import types

ENTRY = 'supervise_llm_intervention_scientific_cpu.py'
VERSION = 'ace-scientific-cpu-supervisor-v1'
FINALIZER = 'finalize_llm_intervention_campaign_cpu.py'
TRAINING = 'prepare_llm_intervention_training_cpu.py'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
SERVICE = 'llm_intervention_model_service.py'
CHECKS = 'llm_intervention_service_runtime.py'
COLLECT = 'collect_llm_intervention_scientific_training_cpu.py'
REALIZE = 'llm_intervention_screen_realization.py'
OWNED = 'llm_intervention_cpu_owned_supervision.py'
HELPER_PINS = {
    OWNED: '1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8',
    FINALIZER: '88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f',
    TRAINING: 'b2b1425bcd010d5ddc5deeda0dc3bd89817c9fe4cae6c4cc6465ec9f7f998f43',
    DIAGNOSTIC: '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329',
    SERVICE: 'a7735dd35f56d7653a5ff42520db7fd67b98395e62333cccce31583317f7134e',
    CHECKS: '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0',
    'llm_intervention_runtime_archive.py': '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e',
    'probe.py': '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951',
}
PROTOCOL_NAMES = frozenset('llm_intervention_'+n+'.py' for n in (
    'contract','episode','numerical_policy','replicated_branches','replicated_feedback',
    'scm','screen_protocol','supervision','target_risk','task','training_dataset','world'))
PROPOSAL_BYTE_SHA = '8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SHA = '585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
BASE_RESOURCES = dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
                      memory_mib=7680,gpus=0,requeue=False)
REALIZE_DEADLINES = dict(startup=120,work=260,cleanup=280,hard=290)
PACKET_KEYS = {'schema','mode','source_commit','output','files','source_names','protocol_source_pins',
    'archive','manifest','runtime_source','runtime','resources','deadlines','freeze_sha256','namespace_admission_sha256'}


def _base():
    path = Path(__file__).with_name(FINALIZER)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        raw = f.read(1024**2+1)
    if len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=HELPER_PINS[FINALIZER]:
        raise ValueError('independent supervisor helper pin differs')
    m=types.ModuleType('_scientific_supervisor_base');m.__file__=str(path)
    exec(compile(raw,str(path),'exec'),vars(m))
    return m


B = _base()


def _output_path_binding(path, logical_root, physical_root):
    """Admit identity or one observed CURC mount alias, without rewriting paths."""
    path, logical_root, physical_root = map(Path, (path, logical_root, physical_root))
    B.need(path.is_absolute() and '..' not in path.parts, 'absolute normalized output required')
    resolved = path.resolve()
    result = dict(logical=str(path), resolved=str(resolved), mount_alias=False)
    if resolved == path:
        return result
    B.need(logical_root in path.parents, 'unrecognized output path alias')
    suffix = path.relative_to(logical_root)
    B.need(resolved == physical_root/suffix
           and logical_root.resolve(strict=True) == physical_root
           and physical_root.resolve(strict=True) == physical_root
           and logical_root.is_dir() and physical_root.is_dir(), 'CURC mount alias differs')
    left, right = logical_root.stat(), physical_root.stat()
    B.need((left.st_dev, left.st_ino) == (right.st_dev, right.st_ino), 'mount roots differ')
    current = logical_root
    for component in suffix.parts:
        current = current/component
        B.need(not current.is_symlink(), 'output descendant symlink forbidden')
    result.update(mount_alias=True, root_device=left.st_dev, root_inode=left.st_ino)
    return result


def output_path_binding(path):
    return _output_path_binding(path, '/scratch/alpine', '/gpfs/alpine1/scratch')


def work_sources(mode):
    B.need(mode in ('realize','collect'),'exactly one fixed stage required')
    return PROTOCOL_NAMES | ({REALIZE} if mode=='realize' else {COLLECT,'llm_intervention_training_collection.py'})


def input_names(mode):
    names={'proposal.json'}
    if mode=='collect':
        names |= {'collection-packet.json','collection-admission.json','realization-terminal.json','realization-acceptance.json'}
        names |= {f'descriptor-{i:03d}.json' for i in range(32)}
    return names


def resource_contract(mode, resources, deadlines):
    B.need(mode in ('realize','collect') and B.raw(resources)==B.raw(dict(BASE_RESOURCES,wall_seconds=300 if mode=='realize' else 900)),
           'exact 2 CPU/7680 MiB/zero GPU stage profile required')
    B.need(type(deadlines) is dict and set(deadlines)=={'startup','work','cleanup','hard'}
        and all(type(v) is int for v in deadlines.values())
        and 0<deadlines['startup']<deadlines['work']<deadlines['cleanup']<deadlines['hard']<resources['wall_seconds']
        and deadlines['cleanup']-deadlines['work']>=5 and deadlines['hard']-deadlines['cleanup']>=5,
        'explicit ordered absolute deadlines with cleanup reserve required')
    B.need(mode!='realize' or deadlines==REALIZE_DEADLINES,'fixed realization deadlines required')


def allocation(text, job, mode):
    rows=re.findall(r'([^\s=]+)=(\S+)',text);fields=dict(rows)
    B.need(len(fields)==len(rows) and re.fullmatch('[1-9][0-9]*',job or ''),'unique scheduler identity required')
    expected=dict(JobId=job,JobState='RUNNING',Account='ucb736_asc2',Partition='acpu',QOS='cpu-normal',
        NumCPUs='2',NumTasks='1',NumNodes='1',TimeLimit='00:05:00' if mode=='realize' else '00:15:00',Requeue='0',Restarts='0')
    B.need(mode in ('realize','collect') and all(fields.get(k)==v for k,v in expected.items())
        and fields.get('UserId','').startswith('paco0228('),'actual allocation identity differs')
    for name in ('AllocTRES','ReqTRES'):
        pairs=[x.split('=') for x in fields.get(name,'').split(',')]
        B.need(all(len(x)==2 for x in pairs),'malformed resources')
        tres=dict(pairs)
        B.need(len(tres)==len(pairs) and tres.get('cpu')=='2' and tres.get('node')=='1'
            and tres.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in tres),
            'requested/allocated CPU/memory/accelerators differ')
    return fields


def no_accelerators():
    keys=('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS','CUDA_VISIBLE_DEVICES',
          'NVIDIA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')
    environment={k:os.environ.get(k,'') for k in keys}
    devices=sorted({p for pattern in ('/dev/nvidia[0-9]*','/dev/kfd','/dev/dri/renderD*','/dev/accel/accel*') for p in glob.glob(pattern)})
    B.need(not devices and not any(environment.values()),'CPU-only allocation/environment/devices required')
    return dict(environment=environment,devices=devices)


def packet_shape(p, data, output, admission, packet_pin, service, diagnostic):
    B.need(type(p) is dict and set(p)==PACKET_KEYS and p['schema']==VERSION
        and re.fullmatch('[0-9a-f]{40}',p['source_commit']),'exact scientific stage packet required')
    resource_contract(p['mode'],p['resources'],p['deadlines'])
    sources=work_sources(p['mode']) | {ENTRY}
    B.need(p['source_names']==sorted(sources) and set(p['files'])==sources|set(HELPER_PINS)|input_names(p['mode'])
        and set(data)==set(p['files']) and all(B.pin(h) and B.sha(data[n])==h for n,h in p['files'].items())
        and all(p['files'][n]==h for n,h in HELPER_PINS.items()),'complete exact source/input/helper pins required')
    B.need(p['files']['proposal.json']==PROPOSAL_BYTE_SHA
        and B.sha(B.raw(B.decode(data['proposal.json']))[:-1])==PROPOSAL_SHA
        and p['protocol_source_pins']=={'scripts/research/'+n:p['files'][n] for n in PROTOCOL_NAMES}
        and B.decode(data['proposal.json'])['source_pins']==p['protocol_source_pins'], 'original proposal/source pins differ')
    B.need(p['runtime']==service.RUNTIME and p['runtime_source']==diagnostic.RUNTIME_SOURCE,'accepted runtime identity differs')
    for key,pin in (('archive',diagnostic.ARCHIVE_SHA),('manifest',diagnostic.MANIFEST_SHA)):
        B.need(type(p[key]) is dict and set(p[key])=={'path','sha256'} and p[key]['sha256']==pin
            and B.absolute(p[key]['path']),'original runtime archive/manifest required')
    B.need(p['output']==str(output) and B.absolute(p['output'])
        and B.pin(p['freeze_sha256']) and B.pin(p['namespace_admission_sha256']),'exact output/freeze/namespace required')
    expected=dict(schema=VERSION+'/admission',mode=p['mode'],status='admitted',packet_sha256=packet_pin,
        output=p['output'],freeze_sha256=p['freeze_sha256'],namespace_admission_sha256=p['namespace_admission_sha256'],
        scientific_protocol_frozen=True,namespace_owned=True,source_runtime_scheduler_qualified=True,
        accounting_admitted=True,resources=p['resources'],deadlines=p['deadlines'])
    B.need(B.raw(admission)==B.raw(expected),'independent stage-specific caller admission differs')
    if p['mode']=='collect':
        inner=B.decode(data['collection-packet.json']);a=B.decode(data['collection-admission.json'])
        inner_inputs={'proposal.json','realization-terminal.json'}|{f'descriptor-{i:03d}.json' for i in range(32)}
        B.need(inner['schema']=='ace-scientific-training-collection-cpu-v1'
            and inner['files']=={n:p['files'][n] for n in inner_inputs}
            and inner['source_pins']=={'scripts/research/'+n:p['files'][n] for n in work_sources('collect')}
            and inner['freeze_sha256']==p['freeze_sha256'] and inner['namespace_admission_sha256']==p['namespace_admission_sha256']
            and inner['proposal_sha256']==PROPOSAL_SHA
            and a['packet_sha256']==p['files']['collection-packet.json']
            and a['output']==str(output/'session/worker/task'), 'original inner collection packet/admission differs')
        accepted=B.decode(data['realization-acceptance.json'])
        B.need(accepted.get('status')=='accepted' and type(accepted.get('required_findings')) is int
            and accepted['required_findings']==0 and accepted.get('complete_custody_verified') is True
            and accepted.get('realization_terminal_sha256')==p['files']['realization-terminal.json']
            and accepted.get('freeze_sha256')==p['freeze_sha256']
            and accepted.get('namespace_admission_sha256')==p['namespace_admission_sha256'],
            'independent complete realization acceptance required')


def authenticate(args):
    root=Path(args.packet).absolute().parent
    p=B.decode(B.read(args.packet,args.packet_sha256,2*1024**2))
    B.need(type(p.get('files')) is dict and len(p['files'])<80
        and all(type(n) is str and Path(n).name==n and B.pin(h) for n,h in p['files'].items()),'bounded flat file roster required')
    data={n:B.read(root/n,h,8*1024**2) for n,h in p['files'].items()}
    B.need(sum(map(len,data.values()))<=32*1024**2 and all(p['files'].get(n)==h for n,h in HELPER_PINS.items()),'bounded fixed helper closure required')
    helpers={role:B.module('_scientific_'+role,root/name,data[name]) for role,name in
        [('training',TRAINING),('diagnostic',DIAGNOSTIC),('service',SERVICE),('checks',CHECKS)]}
    outer=Path(p['output'])
    a_raw=B.read(args.admission,args.admission_sha256,256*1024)
    packet_shape(p,data,outer,B.decode(a_raw),args.packet_sha256,helpers['service'],helpers['diagnostic'])
    expected=outer/'session/worker' if args.worker else outer
    B.need(str(expected)==args.output and args.source_commit==p['source_commit']
        and Path(__file__).absolute()==root/ENTRY and not root.is_symlink(),'captured entry/output/commit differs')
    B.need(B.absolute(args.admission) and Path(args.admission).absolute()==root/'admission.json','separate pinned admission path required')
    return dict(root=root,packet=p,data=data,admission_raw=a_raw,packet_path=Path(args.packet),
                admission_path=Path(args.admission),packet_pin=args.packet_sha256,admission_pin=args.admission_sha256,**helpers)


def recheck(ctx):
    p=ctx['packet']
    B.read(ctx['packet_path'],ctx['packet_pin'],2*1024**2)
    B.need(B.read(ctx['admission_path'],ctx['admission_pin'],256*1024)==ctx['admission_raw'],'admission changed')
    for name,pin in p['files'].items():
        B.need(B.read(ctx['root']/name,pin,8*1024**2)==ctx['data'][name],'captured input changed')


def publish(out, value):
    B.save(out/'terminal-candidate.json',value)
    os.link(out/'terminal-candidate.json',out/'terminal.json')


def preserve_failure(out, exc):
    try:
        name='publication-interruption.json' if os.path.lexists(out/'terminal.json') else 'failure.json'
        if not os.path.lexists(out/name):
            B.save(out/name,dict(type=type(exc).__name__,message=str(exc),retry_allowed=False,
                accounting_authority='task/ original descriptors or collection journals; never infer zero charges'))
    except BaseException:
        pass


def runtime_record(ctx, loader, stage, *, after):
    p,d,checks=ctx['packet'],ctx['diagnostic'],ctx['checks'];prefix=Path(stage['staged_prefix'])
    B.need(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.version_info[:3]==(3,10,19)
        and os.getpgrp()==os.getpid(),'isolated owned staged worker required')
    checks.check_stage(stage,p['runtime'],prefix)
    controls=dict(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(prefix/'lib'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU')
    B.need(all(os.environ.get(k)==v for k,v in controls.items()) and not any(os.environ.get(k) for k in
        ('PYTHONHOME','PYTHONPATH','LD_PRELOAD','MKL_SERVICE_FORCE_INTEL')),'fixed CPU environment differs')
    forbidden={'torch','transformers','tokenizers','scipy','safetensors'}
    B.need(not any(n.split('.')[0] in forbidden for n in sys.modules),'only NumPy numerical imports admitted')
    captured={n:ctx['data'][n] for n in p['source_names']}
    modules,projects=checks.runtime_modules(sys.modules,loader,captured,ctx['root']/ENTRY,globals())
    snapshot=d.runtime_snapshot(prefix,p['runtime_source'],modules=modules)
    d.check_prefix(prefix,snapshot)
    B.need(not d.inside(prefix,p['runtime_source']) and not snapshot['generated_provenance']['present'],'original/generated runtime forbidden')
    if after:
        expected={Path(n).stem:p['files'][n] for n in work_sources(p['mode']) if n!=COLLECT}
        B.need(set(projects)==set(expected) and all(projects[n]['sha256']==h for n,h in expected.items()),'exact live captured source closure differs')
        B.need(snapshot['packages']['numpy']['version']==p['runtime']['imported_versions']['numpy']
            and any(n.startswith('numpy.') and path.endswith('.so') for n,path in snapshot['module_paths'].items()),'actual pinned NumPy extension required')
    else:
        B.need('numpy' not in sys.modules and not projects,'numerical/project imports preceded runtime admission')
    return dict(snapshot=snapshot,captured_project_modules=projects,environment=controls,accelerators=no_accelerators())


def inventory(root):
    result={};total=0
    for path in sorted(root.rglob('*')):
        B.need(not path.is_symlink(),'output symlink forbidden')
        if path.is_dir():continue
        value=B.read(path,maximum=64*1024**2);total+=len(value)
        B.need(total<=2*1024**3 and len(result)<65536,'bounded complete output required')
        result[str(path.relative_to(root))]=B.sha(value)
    return dict(files=result,bytes=total)


def task_summary(ctx, task):
    p=ctx['packet'];terminal=B.decode(B.read(task/'terminal.json'))
    B.need(not (task/'failure.json').exists() and not (task/'publication-interruption.json').exists(),'failed/ambiguous task cannot qualify')
    if p['mode']=='realize':
        B.need(terminal['version']=='ace-first-screen-descriptor-realization-v1' and terminal['status']=='complete'
            and terminal['proposal_sha256']==PROPOSAL_SHA and terminal['source_pins']==p['protocol_source_pins']
            and terminal['freeze_sha256']==p['freeze_sha256'] and terminal['namespace_admission_sha256']==p['namespace_admission_sha256']
            and terminal['worlds_generated']==56 and terminal['environment_responses']==terminal['policy_updates']==0,
            'complete realization terminal differs')
        proposal=B.decode(ctx['data']['proposal.json']);rows=proposal['training_worlds']+proposal['development_worlds']
        B.need(len(terminal['entries'])==56,'all 56 descriptor entries required')
        for i,(entry,row) in enumerate(zip(terminal['entries'],rows)):
            B.need(entry['world_id']==row['world_id'] and entry['scope']==row['scope']
                and entry['status']=='complete' and entry['path']==f'world-{i:03d}.json','descriptor order/identity differs')
            value=B.decode(B.read(task/entry['path'],entry['file_sha256']))
            B.need(B.sha(B.raw(value)[:-1])==entry['descriptor_sha256'] and value['world_id']==row['world_id']
                and value['scope']==row['scope'],'sealed descriptor bytes/identity differ')
        return dict(worlds_generated=56,environment_response_attempts=0,environment_cost_units=0)
    B.need(terminal['schema']=='ace-scientific-training-collection-cpu-v1' and terminal['status']=='collection_complete'
        and terminal['packet_sha256']==p['files']['collection-packet.json']
        and terminal['admission_sha256']==p['files']['collection-admission.json']
        and terminal['record_count']==160 and terminal['retained_pair_count']==3584
        and terminal['accounting']['total_reserved_cost_units']==10240
        and terminal['accounting']['unknown_return_attempts']==0,'complete collection handoff differs')
    for field,name in [('terminal','terminal.json'),('inventory','inventory.json'),('plan','plan.json')]:
        B.read(task/'collection'/name,terminal['collection_'+field+'_sha256'])
    return dict(worlds_generated=0,record_count=160,retained_pair_count=3584,accounting=terminal['accounting'])


def start_work(out, ctx, origin):
    B.need(time.monotonic()<origin+ctx['packet']['deadlines']['startup'],'absolute startup deadline exhausted')
    B.save(out/'work-start.json',dict(packet_sha256=ctx['packet_pin'],pid=os.getpid(),monotonic=time.monotonic()))
    signal.setitimer(signal.ITIMER_REAL,max(.001,origin+ctx['packet']['deadlines']['work']-time.monotonic()))


def child(args, ctx, handoff):
    out=Path(args.output);p=ctx['packet'];origin=handoff['started_monotonic']
    stage=B.decode(B.read(out.parent.parent/'stage.json',handoff['stage_sha256']))
    sources={n:ctx['data'][n] for n in p['source_names'] if n!=ENTRY and n!=COLLECT}
    loader=ctx['training'].CapturedSources(sources,ctx['root'])
    B.save(out/'runtime-before.json',runtime_record(ctx,loader,stage,after=False))
    recheck(ctx)
    if p['mode']=='collect':
        collection=B.module('_admitted_collection',ctx['root']/COLLECT,ctx['data'][COLLECT])
        def observe(phase, live_loader):
            B.save(out/('runtime-'+phase+'.json'),runtime_record(ctx,live_loader,stage,after=True))
            if phase=='before_collection':start_work(out,ctx,origin)
        collection.run_collection(packet_raw=ctx['data']['collection-packet.json'],packet_sha256=p['files']['collection-packet.json'],
            inputs={n:ctx['data'][n] for n in collection.INPUT_NAMES},admission_raw=ctx['data']['collection-admission.json'],
            admission_sha256=p['files']['collection-admission.json'],output=out/'task',observation=observe)
    else:
        sys.meta_path.insert(0,loader)
        original=None
        try:
            import llm_intervention_screen_realization as realization
            # The helper performs exact protocol/source checks before any draw.
            realization.realization_recipes(B.decode(ctx['data']['proposal.json']),proposal_sha256=PROPOSAL_SHA,
                source_pins=p['protocol_source_pins'])
            B.save(out/'runtime-before_realization.json',runtime_record(ctx,loader,stage,after=True))
            start_work(out,ctx,origin)
            realization.realize_screen_worlds(proposal=B.decode(ctx['data']['proposal.json']),proposal_sha256=PROPOSAL_SHA,
                source_pins=p['protocol_source_pins'],output=out/'task',freeze_sha256=p['freeze_sha256'],
                namespace_admission_sha256=p['namespace_admission_sha256'])
        except BaseException as exc:
            original=exc;raise
        finally:
            try:
                B.save(out/'runtime-after_realization.json',runtime_record(ctx,loader,stage,after=True))
            except BaseException as exc:
                if original is None:raise
                try:
                    B.save(out/'observation-failure.json',dict(type=type(exc).__name__,message=str(exc),
                        original_type=type(original).__name__,original_message=str(original)))
                except BaseException:
                    pass
            finally:
                sys.meta_path.remove(loader)
    recheck(ctx)
    summary=task_summary(ctx,out/'task')
    tree=inventory(out/'task');tree_pin=B.save(out/'task-inventory.json',tree)
    publish(out,dict(schema=VERSION,status='worker_complete',mode=p['mode'],packet_sha256=ctx['packet_pin'],
        admission_sha256=ctx['admission_pin'],stage_sha256=handoff['stage_sha256'],task_inventory_sha256=tree_pin,
        summary=summary,training_updates=0,model_loads=0,tokenizer_calls=0,independent_acceptance_required=True))


def validate_worker(ctx, out, stage, stage_pin):
    p=ctx['packet'];terminal=B.decode(B.read(out/'terminal.json'))
    B.need(not (out/'failure.json').exists() and not (out/'publication-interruption.json').exists()
        and terminal['schema']==VERSION and terminal['status']=='worker_complete' and terminal['mode']==p['mode']
        and terminal['packet_sha256']==ctx['packet_pin'] and terminal['admission_sha256']==ctx['admission_pin']
        and terminal['stage_sha256']==stage_pin,'successful original worker binding required')
    before=B.decode(B.read(out/'runtime-before.json'));d=ctx['diagnostic'];prefix=Path(stage['staged_prefix'])
    d.check_prefix(prefix,before['snapshot'])
    B.need(not before['captured_project_modules'] and not before['snapshot']['packages']['numpy']['loaded'],'preimport observation differs')
    expected={Path(n).stem:p['files'][n] for n in work_sources(p['mode']) if n!=COLLECT}
    for phase in ('before_'+('collection' if p['mode']=='collect' else 'realization'),
                  'after_'+('collection' if p['mode']=='collect' else 'realization')):
        row=B.decode(B.read(out/('runtime-'+phase+'.json')));snap=row['snapshot']
        d.check_prefix(prefix,snap,baseline=before['snapshot']['sys_path'])
        B.need(set(row['captured_project_modules'])==set(expected)
            and all(row['captured_project_modules'][n]['sha256']==h for n,h in expected.items())
            and snap['packages']['numpy']['version']==p['runtime']['imported_versions']['numpy']
            and any(n.startswith('numpy.') and path.endswith('.so') for n,path in snap['module_paths'].items())
            and row['environment']==before['environment'] and row['accelerators']==before['accelerators'], 'saved actual runtime/source differs')
    tree=B.decode(B.read(out/'task-inventory.json',terminal['task_inventory_sha256']))
    B.need(tree==inventory(out/'task') and terminal['summary']==task_summary(ctx,out/'task'),'complete original task output changed')
    return B.sha(B.read(out/'terminal.json'))


def parent(args, ctx):
    p,d,out=ctx['packet'],ctx['diagnostic'],Path(args.output)
    ownership=B.module('_scientific_cpu_owned',ctx['root']/OWNED,ctx['data'][OWNED])
    args.owned_child=ownership.OwnedChild(args.started+p['deadlines']['hard'])
    job=os.environ.get('SLURM_JOB_ID','')
    q=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
    B.save(out/'allocation.json',dict(stdout=q.stdout,stderr=q.stderr,returncode=q.returncode))
    B.need(q.returncode==0,'allocation query failed');allocation(q.stdout,job,p['mode'])
    B.save(out/'accelerators.json',no_accelerators())
    scratch=Path(os.environ.get('SLURM_SCRATCH',''))
    B.need(scratch.is_absolute(),'absolute scratch required');scratch=scratch.resolve(strict=True)
    B.need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
    local=scratch/('ace-scientific-'+p['mode']+'-'+job)
    B.separate([local,out,ctx['root'],p['runtime_source'],p['archive']['path'],p['manifest']['path']])
    local.mkdir(mode=0o700,exist_ok=False)
    archive=B.module('_scientific_archive',ctx['root']/'llm_intervention_runtime_archive.py',ctx['data']['llm_intervention_runtime_archive.py'])
    probe=B.module('_scientific_cleanup',ctx['root']/'probe.py',ctx['data']['probe.py'])
    event=d.events(out,args.started,args.process_started,512*1024**2)
    event('stage_start');stage=d.stage_runtime(p,archive,local/'runtime',event)
    stage_pin=B.save(out/'stage.json',stage);event('stage_complete')
    B.save(out/'independent-provenance-sources.json',d.provenance_contract(Path(stage['staged_prefix'])))
    session=out/'session';session.mkdir(mode=0o700,exist_ok=False)
    handoff=dict(schema=VERSION,packet_sha256=ctx['packet_pin'],stage_sha256=stage_pin,
        started_monotonic=args.started,parent_pid=os.getpid(),job_id=job)
    handoff_pin=B.save(session/'handoff.json',handoff)
    command=[str(Path(stage['staged_prefix'])/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(ctx['root']/ENTRY),p['files'][ENTRY],
        '--packet',str(ctx['packet_path']),'--packet-sha256',ctx['packet_pin'],'--source-commit',p['source_commit'],
        '--admission',str(ctx['admission_path']),'--admission-sha256',ctx['admission_pin'],
        '--output',str(session/'worker'),'--worker','--handoff-sha256',handoff_pin]
    session_result=ownership.supervise(command,ctx['training'].cpu_environment(Path(stage['staged_prefix']),os.environ),
        session,started=args.started,startup_deadline=args.started+p['deadlines']['startup'],
        work_deadline=args.started+p['deadlines']['work'],packet_sha256=ctx['packet_pin'],
        owned=args.owned_child,kill_owned=probe.kill_owned)
    cleanup=session_result['cleanup']
    terminal_pin=validate_worker(ctx,session/'worker',stage,stage_pin)
    recheck(ctx);B.need(time.monotonic()<args.started+p['deadlines']['hard'],'absolute hard deadline exceeded')
    publish(out,dict(schema=VERSION,status='complete',mode=p['mode'],packet_sha256=ctx['packet_pin'],
        admission_sha256=ctx['admission_pin'],stage_sha256=stage_pin,worker_terminal_sha256=terminal_pin,
        job_id=job,cleanup=cleanup,elapsed_seconds=time.monotonic()-args.started,
        scientific_qualified=False,independent_acceptance_required=True,retry_allowed=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','packet-sha256','source-commit','admission','admission-sha256','output'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--handoff-sha256')
    args=ap.parse_args();args.started,args.process_started=time.monotonic(),time.process_time();args.owned_child=None
    def interrupted(sig,frame):raise InterruptedError('scientific CPU stage interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,120)
    out=Path(args.output).absolute();claimed=False
    try:
        p=B.decode(B.read(args.packet,args.packet_sha256,2*1024**2))
        resource_contract(p['mode'],p['resources'],p['deadlines'])
        path_bindings = [output_path_binding(out), output_path_binding(p['output'])]
        B.separate([Path(p['output']),Path(args.packet).absolute().parent,p['runtime_source'],p['archive']['path'],p['manifest']['path']])
        handoff=None
        if args.worker:
            B.need(out==Path(p['output'])/'session/worker' and B.pin(args.handoff_sha256),'original worker handoff required')
            handoff=B.decode(B.read(out.parent/'handoff.json',args.handoff_sha256,65536))
            origin=handoff['started_monotonic']
            B.need(handoff['schema']==VERSION and handoff['packet_sha256']==args.packet_sha256
                and handoff['parent_pid']==os.getppid() and handoff['job_id']==os.environ.get('SLURM_JOB_ID')
                and os.getpgrp()==os.getpid() and type(origin) in (int,float) and math.isfinite(origin)
                and 0<=args.started-origin<p['deadlines']['startup'],'owned absolute-clock handoff differs')
            args.started=origin
        else:B.need(out==Path(p['output']),'parent output differs')
        signal.setitimer(signal.ITIMER_REAL,max(.001,args.started+p['deadlines']['startup']-time.monotonic()))
        out.mkdir(mode=0o700,exist_ok=False);claimed=True
        B.save(out/'start.json',dict(schema=VERSION,packet_sha256=args.packet_sha256,resources=p['resources'],
            deadlines={k:args.started+v for k,v in p['deadlines'].items()},retry_allowed=False,
            output_path_bindings=path_bindings))
        ctx=authenticate(args)
        if args.worker:child(args,ctx,handoff)
        else:parent(args,ctx)
    except BaseException as exc:
        if claimed:
            try:preserve_failure(out,exc)
            except BaseException as publication:
                if args.owned_child is not None:args.owned_child.error('failure_publication',publication)
                try:print('Failure publication:',repr(publication),'Original:',repr(exc),file=sys.stderr)
                except BaseException:pass
        raise
    finally:
        if args.owned_child is not None:
            args.owned_child.hold_until_closed()
            if args.owned_child.errors:
                try:print('Lifecycle diagnostics:',args.owned_child.errors,file=sys.stderr)
                except BaseException:pass
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
