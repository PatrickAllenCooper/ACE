"""Build explicit pre-draw recovery02 realization packet only after independent freeze adoption.

No scientific imports, draws, environment calls, runtime staging or submission.
All output directories are exclusive; an interrupted build is retained, not resumed.
See handoff.md for the narrow caller-owned contract projection.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import fcntl

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL = REPO/'results/ace_llm_intervention_scientific_realization_20261108/launch02'
EXT = BASE/'ACE-LLM-intervention-scientific-realization-20261108-02'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_REGISTRY = BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY = str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace-recovery02.json')
OLD_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01'
OLD_LOCAL_REGISTRY = BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
OLD_REMOTE_REGISTRY = str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace.json')
OLD_CLAIM_SHA = '58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
OLD_FREEZE_SHA = '57a3e6dd27ded5cec007be235b799a2bec711fc2c5630ff16fef87d74273cd28'
FAILURE_SHA = '5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d'
NO_DRAW_SHA = '5fb94f9d07da3acd25c979250471037e8981066e0b201716f8c32d0b681fb9ef'
FAILED_JOB = '35402725'
SOURCE_CHANGES = {'scripts/research/'+n for n in (
    'supervise_llm_intervention_scientific_cpu.py','collect_llm_intervention_scientific_training_cpu.py',
    'prepare_llm_intervention_scientific_campaign_cpu.py','qualify_llm_intervention_scientific_campaign_gpu.py')}
RECOVERY_INPUTS = ('predecessor-freeze','failure-acceptance','no-draw-proof','failure-inventory','failure-readback','recovery-authority')
HELPER = '/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
ENTRY = 'supervise_llm_intervention_scientific_cpu.py'
ENTRY_SHA = 'e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8'
VERSION = 'ace-scientific-cpu-supervisor-v1'
BASELINE = REPO/'results/ace_llm_intervention_scientific_realization_20261108/failure-acceptance01/accounting.json'
BASELINE_SHA = '63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'
PROPOSAL = 'results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
PROPOSAL_SHA = '8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SEMANTIC = '585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
PROBE = 'results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/probe.py'
PACKED = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification/packed/'
RESOURCES = dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
                 memory_mib=7680,gpus=0,requeue=False,wall_seconds=300)
DEADLINES = dict(startup=120,work=260,cleanup=280,hard=290)
WORLD_IDS = dict(policy_train=list(range(202611070000,202611070032)),
                 development=list(range(202611070100,202611070124)))


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()
def raw(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pin(value): return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def decode(data):
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate JSON key');result[k]=v
        return result
    def bad(value): raise ValueError('nonfinite JSON '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)


def read(path,expected=None):
    path=Path(path)
    need(not path.is_symlink(),'symlink input file: '+str(path))
    # macOS /tmp is a system directory alias; canonicalize its parent before opening.
    path=path.resolve(strict=True)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        info=os.fstat(f.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular independent file required')
        data=f.read(32*1024**2+1)
    need(len(data)<=32*1024**2,'bounded input required')
    if expected is not None: need(pin(expected) and sha(data)==expected,'input pin differs: '+str(path))
    return data


def write(path,data):
    with Path(path).open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def committed(revision,path):
    need(re.fullmatch('[0-9a-f]{40}',revision) is not None,'full source/packet Git commit required')
    need(subprocess.check_output(['git','--no-replace-objects','cat-file','-t',revision],cwd=REPO).strip()==b'commit',
         'source object must be a commit')
    return subprocess.check_output(['git','--no-replace-objects','show',revision+':'+path],cwd=REPO)


def literal(source,name):
    definitions={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):definitions.setdefault(target.id,[]).append(node.value)
    active=set()
    def reference(key):
        need(key not in active and len(definitions.get(key,[]))==1,'unique acyclic constant required: '+key)
        active.add(key)
        try:return resolve(definitions[key][0])
        finally:active.remove(key)
    def resolve(node):
        if isinstance(node,ast.Constant) and type(node.value) in (str,int,bool,type(None)):
            return node.value
        if isinstance(node,ast.Name):return reference(node.id)
        if isinstance(node,(ast.List,ast.Tuple,ast.Set)):
            values=[resolve(v) for v in node.elts]
            return list(values) if isinstance(node,ast.List) else tuple(values) if isinstance(node,ast.Tuple) else set(values)
        if isinstance(node,ast.Dict):
            result={}
            for k,v in zip(node.keys,node.values):
                need(k is not None,'dictionary expansion forbidden');key=resolve(k)
                need(key not in result,'duplicate static dictionary key');result[key]=resolve(v)
            return result
        raise ValueError('nonliteral constant expression forbidden: '+type(node).__name__)
    return reference(name)


def validate_authority(freeze,review_raw,adoption,namespace,freeze_pin,review_pin,namespace_pin,revision,source_pins,recovery):
    # This is a projection of the full caller-owned freeze, not a replacement freeze.
    need(type(freeze) is dict and freeze.get('schema')=='ace-first-trained-intervention-screen-contract-v1'
         and freeze.get('source_commit')==revision and type(freeze.get('source_pins')) is dict,
         'original combined contract/source map required')
    need(all(freeze['source_pins'].get(n)==h for n,h in source_pins.items()),'combined-contract source pins differ')
    need(freeze.get('stage_closure',{}).get('source_pins')==freeze['source_pins'],
         'independent complete stage-closure map differs')
    need(freeze['stage_closure']['stages']['realize']['input_source_files']==
         {Path(n).name:h for n,h in source_pins.items()},'exact realization stage source roster differs')
    need(freeze.get('original_proposal')==dict(path=PROPOSAL,sha256=PROPOSAL_SHA,semantic_sha256=PROPOSAL_SEMANTIC)
         and freeze.get('training_world_ids')==WORLD_IDS['policy_train']
         and freeze.get('development_world_ids')==WORLD_IDS['development'],'original proposal/schedule differs')
    expected_stage=dict(stage='realize',entry=ENTRY,entry_sha256=ENTRY_SHA,cpus=2,memory_mib=7680,gpus=0,
        gpu_type=None,wall_seconds=300,deadlines=DEADLINES,account='ucb736_asc2',partition='acpu',qos='cpu-normal',
        nodes=1,tasks=1,requeue=False,max_restarts=0,remote_root=ROOT,
        repository_root=str(LOCAL.parent.relative_to(REPO)),exclusive_external_root=str(EXT))
    stages=freeze.get('stages',[])
    need(type(stages) is list and raw([s for s in stages if s.get('stage')=='realize'])==raw([expected_stage]),
         'combined realization stage differs')
    need(freeze.get('admission_accounting',{}).get('baseline_sha256')==BASELINE_SHA
         and freeze.get('independent_contract_adoption_required') is True
         and freeze.get('namespace_admission_required') is True
         and freeze.get('independent_actual_acceptance_between_stages') is True
         and type(freeze.get('scientific_worlds_generated')) is int
         and freeze['scientific_worlds_generated']==0 and freeze.get('no_automatic_retry') is True,
         'combined first-draw/admission contract differs')
    expected_namespace=dict(schema='ace-scientific-screen-namespace-descriptor-v1',status='predeclared',
        freeze_sha256=freeze_pin,proposal_sha256=PROPOSAL_SHA,proposal_semantic_sha256=PROPOSAL_SEMANTIC,
        world_ids=WORLD_IDS,namespaces_from='exact original proposal bindings',
        local_registry=str(LOCAL_REGISTRY),remote_registry=REMOTE_REGISTRY,realization_root=ROOT,
        retry_allowed=False,replacement_worlds=False,recovery_authority_sha256=recovery['authority_pin'],
        original_namespace_claim_sha256=OLD_CLAIM_SHA,failed_job_id=FAILED_JOB)
    need(raw(namespace)==raw(expected_namespace),'caller namespace descriptor differs')
    expected_adoption=dict(schema='ace-scientific-contract-adoption-v1',status='adopted',
        freeze_sha256=freeze_pin,review_sha256=review_pin,required_findings=0,adopted_before_first_draw=True)
    need(raw(adoption)==raw(expected_adoption) and review_raw.strip(),'independent freeze review/adoption differs')


def validate_recovery(supplied,freeze,freeze_pin,revision,baseline):
    """Consume accepted failure/no-draw evidence; never rerun the audit or infer retry authority."""
    need(set(RECOVERY_INPUTS)<=set(supplied),'explicit independently pinned recovery inputs required')
    need(sha(supplied['predecessor-freeze'])==OLD_FREEZE_SHA and sha(supplied['failure-acceptance'])==FAILURE_SHA
         and sha(supplied['no-draw-proof'])==NO_DRAW_SHA,'independent accepted failure/predecessor pins differ')
    previous=decode(supplied['predecessor-freeze']);accepted=decode(supplied['failure-acceptance']);audit=decode(supplied['no-draw-proof'])
    need(accepted['schema']=='ace-scientific-realization-failure-acceptance-v1'
         and accepted['status']=='accepted_failed_before_draw' and accepted['job_id']==FAILED_JOB
         and accepted['failure_accepted'] is True and accepted['realization_accepted'] is False
         and type(accepted['required_findings']) is int and accepted['required_findings']==0
         and accepted['complete_custody_verified'] is True and accepted['two_remote_inventories_matched'] is True
         and accepted['original_inputs_and_claims_verified'] is True and accepted['scheduler_terminal_verified'] is True
         and accepted['source_control_flow_and_traceback_verified'] is True
         and accepted['scientific_worlds_generated']==accepted['environment_responses']==0
         and all(accepted[n] is False for n in ('runtime_staging_started','owned_worker_spawned','realizer_called','output_directory_claimed'))
         and accepted['namespace_claim_sha256']==OLD_CLAIM_SHA and accepted['freeze_sha256']==OLD_FREEZE_SHA
         and accepted['audit_sha256']==NO_DRAW_SHA,'independent pre-draw failure acceptance differs')
    need(audit['status']=='failure_custody_and_pre_draw_boundary_verified' and audit['job_id']==FAILED_JOB
         and audit['scientific_worlds_generated']==audit['environment_responses']==0
         and audit['realizer_call_reached'] is False and audit['source_commit']==accepted['source_commit'],
         'independent source-bound no-draw proof differs')
    need(baseline['failure_acceptance_sha256']==FAILURE_SHA and baseline['namespace_claim_preserved_sha256']==OLD_CLAIM_SHA
         and baseline['terminal_jobs']==39 and baseline['submission_journals']==80
         and baseline['closed_cpu_seconds']==11048 and baseline['closed_gpu_slice_seconds']==1601
         and baseline['outstanding_reservation']==0 and baseline['automatic_retry_allowed'] is False,
         'accepted failed attempt must remain in complete baseline')
    files=freeze['source_pins'];old=previous['source_pins']
    need(set(files)==set(old) and len(files)==67 and {n for n in files if files[n]!=old[n]}==SOURCE_CHANGES,
         'exact four reviewed infrastructure deltas within unchanged67-source roster required')
    need(files['scripts/research/'+ENTRY]==ENTRY_SHA and
         files['scripts/research/collect_llm_intervention_scientific_training_cpu.py']==
         '2bc872349f9fbfbafe1564b751cde899705f487f3bf822cce5518e0865bd9a4d',
         'independently specified CPU/collection path corrections differ')
    mutable={'utc','source_commit','source_pins','stage_closure','component_reviews','stages','admission_accounting',
             'operational_recovery','infrastructure_amendment'}
    need(set(freeze)-set(previous)<=mutable and all(raw(freeze.get(k))==raw(v) for k,v in previous.items() if k not in mutable),
         'scientific contract/design/32+24/training/reporting fields changed')
    old_stages={r['stage']:r for r in previous['stages']};new_stages={r['stage']:r for r in freeze['stages']}
    need(len(new_stages)==len(freeze['stages']) and set(new_stages)==set(old_stages),'stage roster changed')
    for key,row in old_stages.items():
        expected=dict(row,entry_sha256=files['scripts/research/'+row['entry']])
        if key=='realize':expected.update(remote_root=ROOT,exclusive_external_root=str(EXT))
        need(raw(new_stages[key])==raw(expected),'non-infrastructure stage/resource change: '+key)
    a=freeze['admission_accounting']
    need(a['baseline_sha256']==BASELINE_SHA and a['closed_cpu_seconds']==11048 and a['closed_gpu_seconds']==1601
         and a['prep_cpu_allowance']==3600 and a['new_reservations']==0
         and a['max_projected_cpu_seconds']==11048+3600+sum(s['cpus']*s['wall_seconds'] for s in freeze['stages'])
         and a['max_projected_gpu_seconds']==1601+sum(s['gpus']*s['wall_seconds'] for s in freeze['stages'])
         and a['max_projected_cpu_seconds']<=57600 and a['max_projected_gpu_seconds']<=7200,
         'successor accounting must retain actual failed cost and unchanged finite stages')
    authority=decode(supplied['recovery-authority']);authority_pin=sha(supplied['recovery-authority'])
    expected=dict(schema='ace-scientific-realization-recovery-authority-v1',status='authorized',recovery_id='realization02',
        failed_job_id=FAILED_JOB,failure_acceptance_sha256=FAILURE_SHA,no_draw_proof_sha256=NO_DRAW_SHA,
        predecessor_freeze_sha256=OLD_FREEZE_SHA,successor_freeze_sha256=freeze_pin,source_commit=revision,
        original_namespace_claim_sha256=OLD_CLAIM_SHA,accounting_sha256=BASELINE_SHA,
        original_root=OLD_ROOT,recovery_root=ROOT,local_recovery_registry=str(LOCAL_REGISTRY),remote_recovery_registry=REMOTE_REGISTRY,
        one_submission_only=True,automatic_retry_allowed=False,replacement_worlds_allowed=False)
    need(raw(authority)==raw(expected),'explicit one-attempt recovery authority required; failure eligibility is insufficient')
    observations=[]
    for key,receipt_name in (('failure-inventory','inventory_transport.json'),('failure-readback','inventory_after_transport.json')):
        need(sha(supplied[key])==accepted['evidence_pins'][receipt_name],'accepted failure inventory transport pin differs')
        transport=decode(supplied[key]);need(transport['returncode']==0,'accepted failure transport failed')
        observed=decode(transport['stdout'])
        need(observed['terminal_gate'] is True and observed['root_present'] is True and observed['namespace_present'] is True,
             'complete terminal original raw custody required')
        observations.append(observed)
    before,after=observations
    need(before['files']==after['files'] and before['directories']==after['directories']
         and len(before['files'])==accepted['raw_files']==41
         and sum(r['bytes'] for r in before['files'].values())==accepted['raw_bytes']
         and before['files']['namespace.json']['sha256']==OLD_CLAIM_SHA,'complete authoritative41-file failed roster differs')
    need(all(n=='namespace.json' or n.startswith('launch/') and '..' not in Path(n).parts for n in before['files']),
         'unsafe failed custody path')
    return dict(authority_pin=authority_pin,failed_files=before['files'],failed_directories=before['directories'],
                acceptance=accepted,original_claim_sha256=OLD_CLAIM_SHA)


def local_failed_custody(root,recovery):
    """All raw files, not a prefix/sample; read only the already accepted failed custody."""
    root=Path(root).resolve(strict=True);accepted=recovery['acceptance']
    need(root in {Path(accepted[k]).resolve(strict=True) for k in ('repository_terminal_root','external_terminal_root')},
         'accepted failed custody location required')
    files={};directories=[]
    for path in sorted((root/'remote').rglob('*')):
        need(not path.is_symlink(),'failed custody alias forbidden')
        name=path.relative_to(root/'remote').as_posix()
        if path.is_dir():directories.append(name);continue
        blob=read(path);files[name]=dict(bytes=len(blob),sha256=sha(blob))
    need(files==recovery['failed_files'] and sorted(directories)==sorted(recovery['failed_directories']),
         'all original failed raw bytes/directories required')
    return read(root/'remote/namespace.json',OLD_CLAIM_SHA)


def original_claim_remote_probe():
    """Future BUILD invocation only: read physical original remote registry, never mutate it."""
    code='PATH='+repr(OLD_REMOTE_REGISTRY)+'\nEXPECTED='+repr(OLD_CLAIM_SHA)+'\n'+r'''
from pathlib import Path
import hashlib,json,os,stat
p=Path(PATH)
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
 s=os.fstat(f.fileno())
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('ordinary original claim required')
 raw=f.read()
 h=hashlib.sha256(raw).hexdigest()
 if h!=EXPECTED:raise ValueError('original remote registry changed')
print(json.dumps(dict(path=PATH,sha256=h,bytes=len(raw),mutation=False)))
'''
    try:
        result=subprocess.run(['bash',HELPER,'run','python3 -'],input=code,capture_output=True,text=True,timeout=45)
        return dict(python_source=code,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,utc=utc())
    except subprocess.TimeoutExpired as exc:
        txt=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else v
        return dict(python_source=code,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr),utc=utc())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-commit',required=True)
    for key in ('freeze','freeze-review','adoption','namespace-admission',*RECOVERY_INPUTS):
        ap.add_argument('--'+key,required=True);ap.add_argument('--'+key+'-sha256',required=True)
    ap.add_argument('--failure-custody',required=True)
    args=ap.parse_args();rev=args.source_commit
    supplied={k:read(getattr(args,k.replace('-','_')),getattr(args,k.replace('-','_')+'_sha256'))
              for k in ('freeze','freeze-review','adoption','namespace-admission',*RECOVERY_INPUTS)}
    baseline_raw=read(BASELINE,BASELINE_SHA);baseline=decode(baseline_raw)
    need(baseline['terminal_jobs']==39 and baseline['submission_journals']==80
         and baseline['closed_cpu_seconds']==11048 and baseline['closed_gpu_slice_seconds']==1601
         and baseline['preparation_allowance_cpu_seconds']==3600 and baseline['outstanding_reservation']==0,
         'exact existing baseline required')
    source=committed(rev,'scripts/research/'+ENTRY);need(sha(source)==ENTRY_SHA,'reviewed entry differs')
    helpers=literal(source,'HELPER_PINS')
    proposal_raw=committed(rev,PROPOSAL);need(sha(proposal_raw)==PROPOSAL_SHA,'original proposal differs')
    proposal=decode(proposal_raw)
    need(sha(json.dumps(proposal,sort_keys=True,separators=(',',':'),allow_nan=False).encode())==PROPOSAL_SEMANTIC,
         'original proposal semantic pin differs')
    need([r['world_id'] for r in proposal['training_worlds']]==WORLD_IDS['policy_train']
         and [r['world_id'] for r in proposal['development_worlds']]==WORLD_IDS['development'],'original32/24 schedule differs')
    protocol={Path(n).name for n in proposal['source_pins']}
    freeze=decode(supplied['freeze'])
    recovery=validate_recovery(supplied,freeze,args.freeze_sha256,rev,baseline)
    original_claim=local_failed_custody(args.failure_custody,recovery)
    need(read(OLD_LOCAL_REGISTRY,OLD_CLAIM_SHA)==original_claim,'physical original local registry differs')
    # Use the independently pinned AST projection of work_sources('realize'); no project callable execution.
    stage_closure=freeze['stage_closure']['stages']['realize']
    work=stage_closure['work_sources']
    need(type(work) is list and work==sorted(protocol|{literal(source,'REALIZE')}),
         'reviewed work_sources(realize) projection differs')
    sources=set(work)|{ENTRY}
    need(stage_closure['source_names']==sorted(sources) and stage_closure['helper_pins']==helpers,
         'exact source_names/helper projection differs')
    origins={n:('scripts/research/'+n if n!='probe.py' else PROBE) for n in sources|set(helpers)}
    data={n:committed(rev,path) for n,path in origins.items()}
    need(all(sha(data[n])==h for n,h in helpers.items()),'accepted fixed helper pins differ')
    need(all(sha(data[Path(n).name])==h for n,h in proposal['source_pins'].items()),'original protocol source differs')
    source_pins={origins[n]:sha(b) for n,b in data.items()}
    validate_authority(freeze,supplied['freeze-review'],decode(supplied['adoption']),
        decode(supplied['namespace-admission']),args.freeze_sha256,args.freeze_review_sha256,
        args.namespace_admission_sha256,rev,source_pins,recovery)
    for path,h in freeze['source_pins'].items():
        need(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts and pin(h),
             'repo-relative full closure source pins required')
        need(sha(committed(rev,path))==h,'combined full metadata closure differs: '+path)
    data['proposal.json']=proposal_raw
    diagnostic=data[literal(source,'DIAGNOSTIC')];service=data[literal(source,'SERVICE')]
    packet=dict(schema=VERSION,mode='realize',source_commit=rev,output=ROOT+'/realization',
        files={n:sha(b) for n,b in sorted(data.items())},source_names=sorted(sources),protocol_source_pins=proposal['source_pins'],
        archive=dict(path=PACKED+'runtime.tar',sha256=literal(diagnostic,'ARCHIVE_SHA')),
        manifest=dict(path=PACKED+'manifest.json',sha256=literal(diagnostic,'MANIFEST_SHA')),
        runtime_source=literal(diagnostic,'RUNTIME_SOURCE'),runtime=literal(service,'RUNTIME'),
        resources=RESOURCES,deadlines=DEADLINES,freeze_sha256=args.freeze_sha256,
        namespace_admission_sha256=args.namespace_admission_sha256)
    need(freeze['runtime']==packet['runtime'] and freeze['runtime_source']==packet['runtime_source']
         and freeze['runtime_archive_sha256']==packet['archive']['sha256']
         and freeze['runtime_manifest_sha256']==packet['manifest']['sha256']
         and freeze['runtime_archive_path']==packet['archive']['path']
         and freeze['runtime_manifest_path']==packet['manifest']['path'],'combined runtime contract differs')
    packet_raw=raw(packet)
    admission=dict(schema=VERSION+'/admission',mode='realize',status='admitted',packet_sha256=sha(packet_raw),
        output=packet['output'],freeze_sha256=args.freeze_sha256,namespace_admission_sha256=args.namespace_admission_sha256,
        scientific_protocol_frozen=True,namespace_owned=True,source_runtime_scheduler_qualified=True,
        accounting_admitted=True,resources=RESOURCES,deadlines=DEADLINES)
    # Admission is a frozen conditional payload: launcher must claim both registries before making it runnable.
    files={'inputs/'+n:b for n,b in data.items()}
    files.update({'inputs/packet.json':packet_raw,'inputs/admission.json':raw(admission),
        'freeze.json':supplied['freeze'],'freeze-review.md':supplied['freeze-review'],'adoption.json':supplied['adoption'],
        'namespace-admission.json':supplied['namespace-admission'],'accepted-accounting.json':baseline_raw})
    for name in RECOVERY_INPUTS:files[name+'.json']=supplied[name]
    files['original-namespace-claim.json']=original_claim
    files['failed-raw-inventory.json']=raw(dict(root=OLD_ROOT,registry=OLD_REMOTE_REGISTRY,
        files=recovery['failed_files'],directories=recovery['failed_directories']))
    for name in ('build_packet.py','launch_once.py','monitor_once.py','pull_terminal.py'):
        files[name]=read(Path(__file__).resolve().with_name(name))
    command=[packet['runtime_source']+'/bin/python','-I','-B','-c',literal(diagnostic,'BOOTSTRAP'),
        ROOT+'/inputs/'+ENTRY,ENTRY_SHA,'--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(packet_raw),
        '--source-commit',rev,'--admission',ROOT+'/inputs/admission.json','--admission-sha256',sha(raw(admission)),
        '--output',packet['output']]
    files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_realize02',
        '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M',
        '#SBATCH --time=00:05:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
        '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
        "export CUDA_VISIBLE_DEVICES=''",'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
        'exec '+shlex.join(command),''])).encode()
    outer=dict(schema='ace-scientific-realization-launch-v1',source_commit=rev,remote_root=ROOT,
        resources=RESOURCES,deadlines=DEADLINES,source_pins=source_pins,
        new_input_files={n:sha(b) for n,b in sorted(files.items())},baseline_sha256=BASELINE_SHA,
        admission=dict(closed_cpu_seconds=11048,closed_gpu_slice_seconds=1601,preparation_allowance_cpu_seconds=3600,
            candidate_cpu_reservation=600,candidate_gpu_reservation=0,total_cpu_seconds=15248,total_gpu_seconds=1601,
            ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,prior_jobs=39,prior_journals=80),
        recovery=dict(authority_sha256=recovery['authority_pin'],failed_job_id=FAILED_JOB,
            failure_acceptance_sha256=FAILURE_SHA,no_draw_proof_sha256=NO_DRAW_SHA,
            original_namespace_claim_sha256=OLD_CLAIM_SHA,original_root=OLD_ROOT,
            original_local_registry=str(OLD_LOCAL_REGISTRY),original_remote_registry=OLD_REMOTE_REGISTRY,
            automatic_retry_allowed=False,one_submission_only=True),
        namespace_claimed=False,submitted=False,qualification='concrete recovery packet still requires independent review',
        sizing='Measured 2CPU/7680MiB profile; 300s bounded staging+56 descriptor realization, no memory-headroom claim.')
    need(not any(os.path.lexists(p) for p in (LOCAL,EXT,LOCAL_REGISTRY)),'existing packet/namespace; no adoption or retry')
    created=[];locks=[]
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=BASE/name;need(path==path.resolve(),'coordinator lock alias')
            handle=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        need(read(OLD_LOCAL_REGISTRY,OLD_CLAIM_SHA)==original_claim,'original local claim changed before build')
        for root in (LOCAL,EXT):
            root.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(root)
            (root/'inputs').mkdir(mode=0o700)
        probe=original_claim_remote_probe()
        # Persist every remote-read outcome, including partial timeout streams, before inspecting it.
        probe_raw=raw(probe)
        for root in created:write(root/'original-claim-build-readback.json',probe_raw)
        need(probe['returncode']==0,'original remote claim build readback failed; preserve attempt, no retry')
        observed=decode(probe['stdout'])
        need(observed==dict(path=OLD_REMOTE_REGISTRY,sha256=OLD_CLAIM_SHA,bytes=len(original_claim),mutation=False)
             and read(OLD_LOCAL_REGISTRY,OLD_CLAIM_SHA)==original_claim,'original physical claim readback differs')
        outer['new_input_files']['original-claim-build-readback.json']=sha(probe_raw)
        files['packet.json']=raw(outer)
        for root in created:
            for name,b in files.items(): write(root/name,b)
        for name,b in files.items():
            need(read(LOCAL/name)==read(EXT/name)==b,'two-copy packet readback differs')
        need(read(LOCAL/'original-claim-build-readback.json')==read(EXT/'original-claim-build-readback.json')==probe_raw,
             'two-copy original registry readback evidence differs')
        print(json.dumps(dict(local=str(LOCAL),external=str(EXT),packet_sha256=sha(files['packet.json']),
                              stage_packet_sha256=sha(packet_raw),namespace_claimed=False,submitted=False),indent=2))
    except BaseException as exc:
        for root in created:
            try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
            except BaseException:pass
        raise
    finally:
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


if __name__=='__main__': main()
