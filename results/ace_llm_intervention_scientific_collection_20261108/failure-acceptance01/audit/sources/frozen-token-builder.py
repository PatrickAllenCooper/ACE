"""AUTHORING ONLY: future first scientific token preparation packet.
Requires actual independently accepted realization/collection and actual41/84 baseline.
No project imports, tokenizer, model, source execution, remote operation or submission.
"""
from pathlib import Path
import argparse, ast, datetime, fcntl, hashlib, json, os, re, shlex, stat, subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
FREEZE_SHA='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='prepare_llm_intervention_scientific_training_cpu.py'
ENTRY_SHA='47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3'
VERSION='ace-scientific-training-preparation-cpu-v1'
OUTER_VERSION='ace-scientific-tokenization-launch-v1'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
LOCAL=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
COLLECTION_TREE=COLLECTION_ROOT+'/collection/session/worker/task/collection'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
FIXED_PINS={
 'freeze':FREEZE_SHA,
 'freeze-review':'088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416',
 'adoption':'ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07',
 'namespace-admission':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
 'recovery-authority':'ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50',
 'failure-baseline':'63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'}
METADATA=tuple(FIXED_PINS)+('namespace-claim','realization-acceptance','realization-disposition',
 'realization-packet','realization-result','realization-baseline','collection-acceptance',
 'collection-disposition','collection-packet','collection-result','baseline')
PROPOSAL='results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
PROPOSAL_SHA='8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SEMANTIC='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
PROBE='results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/probe.py'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,gpus=0,requeue=False,wall_seconds=600)
DEADLINES=dict(startup=120,work=550,cleanup=570,hard=590)
TOKENIZER_NAMES={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json'}
WORLD_IDS=list(range(202611070000,202611070032))


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


def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def normalized(rows):
    result=[]
    for r in rows:
        need(len(r)>=8 and not any(r[8:]) and r[0].isdigit() and r[4].isdigit() and r[5].isdigit(),'malformed baseline row')
        pairs=[x.split('=',1) for x in r[7].split(',') if x]
        need(all(len(x)==2 for x in pairs) and len({x[0] for x in pairs})==len(pairs),'malformed baseline TRES')
        result.append([r[0],r[1].strip(),r[2].strip(),r[3].strip(),int(r[4]),int(r[5]),r[6].strip(),sorted(pairs)])
    need(len({r[0] for r in result})==len(result),'duplicate baseline job')
    return sorted(result,key=lambda r:int(r[0]))


def relative(name):
    return (type(name) is str and bool(name) and not Path(name).is_absolute()
            and all(p not in ('','.','..') for p in name.split('/')))


def fingerprint(path):
    """Opaque custody bytes only. Never decode records or follow member symlinks."""
    path=Path(path);need(not path.is_symlink(),'linked custody member')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as handle:
        before=os.fstat(handle.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=64*1024**2,'bounded regular custody member')
        digest=hashlib.sha256()
        for chunk in iter(lambda:handle.read(1024**2),b''):digest.update(chunk)
        after=os.fstat(handle.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(before)==identity(after)==identity(path.lstat()),'custody member changed')
    return dict(sha256=digest.hexdigest(),bytes=after.st_size)


def verify_collection_tree(root,expected):
    root=Path(root);need(root.is_dir() and not root.is_symlink(),'ordinary accepted collection custody required')
    files={};total=0
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(),'collection member symlink forbidden')
        if path.is_dir():continue
        row=fingerprint(path);total+=row['bytes'];files[path.relative_to(root).as_posix()]=row['sha256']
        need(len(files)<=65536 and total<=2*1024**3,'complete collection custody exceeds worker bounds')
    need(files==expected,'complete collection custody differs; no subset/filtering')
    return dict(files=len(files),bytes=total,files_sha256=digest(files))


def transition(m,pins,kind,prior_key,next_key,count,wall):
    """Join an already accepted terminal CPU stage to its independently supplied baseline."""
    a=m[kind+'-acceptance'];d=m[kind+'-disposition'];result=m[kind+'-result'];packet=m[kind+'-packet']
    old=m[prior_key];base=m[next_key];job=result.get('job_id')
    root=REALIZATION_ROOT if kind=='realization' else COLLECTION_ROOT
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job!='35402725'
         and result.get('returncode')==0 and result.get('ambiguous') is False and pin(result.get('intent_sha256')),
         'independently pinned successful '+kind+' submission required')
    need(a.get('status')=='accepted' and type(a.get('required_findings')) is int and a['required_findings']==0
         and a.get('complete_custody_verified') is True and a.get('job_id')==job
         and a.get('disposition_sha256')==pins[kind+'-disposition']
         and a.get(kind+'_packet_sha256')==pins[kind+'-packet'], 'independent actual '+kind+' acceptance linkage required')
    need(packet['remote_root']==root and packet['source_commit']==SOURCE_COMMIT
         and packet['new_input_files']['freeze.json']==FREEZE_SHA
         and packet['new_input_files']['namespace-admission.json']==pins['namespace-admission'],
         'accepted stage frozen source/namespace differs')
    need(d['job_id']==job and d['packet_sha256']==pins[kind+'-packet']
         and d['submission_result_sha256']==pins[kind+'-result'] and d['remote_root']==root
         and d['custody_bytes_verified'] is True and d['current_lineage_verified'] is True
         and d['errors']==[] and d['input_pin_mismatches']==[] and d['exact_prior_accounting_matched'] is True
         and d['exact_prior_and_current_journals_matched'] is True
         and d['observed_related_jobs']==count and d['observed_submission_journals']==2*count+2,
         'complete independently accepted terminal custody required')
    allocation=d['allocation'];elapsed=allocation['elapsed_seconds']
    need(allocation['resource_contract_matched'] is True and allocation['errors']==[]
         and allocation['scheduler_state']=='COMPLETED' and allocation['scheduler_exit']=='0:0'
         and allocation['allocated_cpus']==2 and type(elapsed) is int and 0<=elapsed<=wall
         and allocation['allocated_cpu_seconds']==2*elapsed
         and allocation['gpu_count']==allocation['gpu_slice_seconds']==0,
         'actual upstream bounded CPU/no-GPU allocation required')
    need(old['terminal_jobs']==count-1 and old['submission_journals']==2*count
         and base['terminal_jobs']==count and base['submission_journals']==2*count+2
         and old['outstanding_reservation']==base['outstanding_reservation']==0
         and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
         and base['input_pins']['predecessor']==pins[prior_key]
         and base['input_pins']['acceptance']==pins[kind+'-acceptance']
         and base['input_pins']['disposition']==pins[kind+'-disposition'], 'independent next baseline links/counts differ')
    prior=[r for r in base['related_accounting'] if r[0]!=job];current=[r for r in base['related_accounting'] if r[0]==job]
    need(len(current)==1 and len(prior)==count-1 and normalized(prior)==normalized(old['related_accounting']),
         'historical rows changed or duplicate/missing new job')
    row=normalized(current)[0]
    job_name='ace_llm_scientific_realize02' if kind=='realization' else 'ace_llm_scientific_collect01'
    need(row[1]==job_name and row[2:4]==['COMPLETED','0:0'] and row[4:7]==[2,elapsed,'ucb736_asc2']
         and row[7]==sorted(p.split('=',1) for p in allocation['alloc_tres'].split(',')), 'actual row/disposition differs')
    tres=dict(row[7]);need(tres.get('cpu')=='2' and tres.get('node')=='1'
         and tres.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in tres),
         'upstream exact CPU/memory/no-accelerator resource required')
    journals={**old['journal_pins'],root+'/submission.intent.json':result['intent_sha256'],
              root+'/submission.result.json':pins[kind+'-result']}
    need(len(journals)==2*count+2 and base['journal_pins']==journals,'complete prior plus actual journal map differs')
    for path,value in ((root+'/packet.json',pins[kind+'-packet']),
                       (root+'/submission.intent.json',result['intent_sha256']),
                       (root+'/submission.result.json',pins[kind+'-result'])):
        rel='launch/'+Path(path).name
        need(d['remote_files'][rel]['sha256']==value,'original raw packet/journal pin differs')
    need(base['closed_cpu_seconds']==d['closed_cpu_seconds']==old['closed_cpu_seconds']+2*elapsed
         and base['closed_gpu_slice_seconds']==d['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']
         and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600
         and sum(int(r[4])*int(r[5]) for r in base['related_accounting'])==base['closed_cpu_seconds'],
         'actual cost arithmetic differs; no inferred success costs')
    return job


def admit(supplied):
    need(set(supplied)==set(METADATA),'all independently pinned authority/evidence inputs required')
    pins={n:sha(v) for n,v in supplied.items()}
    need(all(pins[n]==h for n,h in FIXED_PINS.items()),'adopted candidate03 metadata differs')
    m={n:decode(v) for n,v in supplied.items() if n!='freeze-review'};f=m['freeze'];ns=m['namespace-admission']
    need(f['source_commit']==SOURCE_COMMIT and f['stage_closure']['source_pins']==f['source_pins'],'adopted source closure differs')
    expected=dict(stage='tokenize',entry=ENTRY,entry_sha256=ENTRY_SHA,cpus=3,memory_mib=11520,gpus=0,
        gpu_type=None,wall_seconds=600,deadlines=DEADLINES,account='ucb736_asc2',partition='acpu',qos='cpu-normal',
        nodes=1,tasks=1,requeue=False,max_restarts=0,remote_root=ROOT,
        repository_root=str(LOCAL.parent.relative_to(REPO)),exclusive_external_root=str(EXT))
    need([s for s in f['stages'] if s['stage']=='tokenize']==[expected],'frozen token stage differs')
    need(m['adoption']==dict(schema='ace-scientific-contract-adoption-v1',status='adopted',freeze_sha256=FREEZE_SHA,
        review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True),'independent adoption required')
    claim=m['namespace-claim'];authority=m['recovery-authority']
    need(ns['freeze_sha256']==claim['freeze_sha256']==FREEZE_SHA and ns['realization_root']==REALIZATION_ROOT
         and claim['namespace_admission_sha256']==pins['namespace-admission'] and claim['realization_root']==REALIZATION_ROOT
         and claim['packet_sha256']==pins['realization-packet'] and claim['world_ids']==ns['world_ids']
         and claim['local_registry']==ns['local_registry']==str(LOCAL_REGISTRY)
         and claim['remote_registry']==ns['remote_registry']==REMOTE_REGISTRY
         and claim['retry_allowed'] is False and claim['replacement_worlds'] is False
         and claim['recovery_authority_sha256']==pins['recovery-authority']
         and claim['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA
         and claim['failed_job_id']==authority['failed_job_id']=='35402725', 'original accepted recovery namespace differs')
    realization_job=transition(m,pins,'realization','failure-baseline','realization-baseline',40,300)
    collection_job=transition(m,pins,'collection','realization-baseline','baseline',41,900)
    need(realization_job!=collection_job,'distinct completed upstream jobs required')
    a=m['realization-acceptance'];d=m['realization-disposition']
    need(a['freeze_sha256']==FREEZE_SHA and a['namespace_admission_sha256']==pins['namespace-admission']
         and a['namespace_claim_sha256']==pins['namespace-claim']
         and d['namespace_local_sha256']==d['namespace_remote_sha256']==pins['namespace-claim']
         and d['original_failed_custody_verified'] is True and d['original_failed_raw_files']==41
         and d['original_namespace_local_sha256']==d['original_namespace_remote_sha256']==ORIGINAL_CLAIM_SHA,
         'accepted realization and preserved failed-attempt lineage required')
    need(d['remote_files']['launch/realization/session/worker/task/terminal.json']['sha256']==a['realization_terminal_sha256'],
         'accepted complete realization terminal binding differs')
    acceptance=m['collection-acceptance'];c=acceptance['collection']
    need(acceptance['schema']=='ace-scientific-collection-acceptance-v1' and acceptance['scope']=='policy_train'
         and acceptance['accounting_complete'] is True,'complete independently accepted policy_train collection required')
    need(set(c)=={'root','files','freeze_sha256','realization_acceptance_sha256','world_pins','configuration_pins','source_pins','namespaces'}
         and c['root']==COLLECTION_TREE and c['freeze_sha256']==FREEZE_SHA
         and c['realization_acceptance_sha256']==pins['realization-acceptance'],'exact worker collection binding differs')
    keys={str(w) for w in WORLD_IDS}
    need(set(c['world_pins'])==set(c['configuration_pins'])==set(c['namespaces'])==keys
         and all(pin(v) for field in ('world_pins','configuration_pins','source_pins') for v in c[field].values()),
         'complete32-world source/descriptor/configuration pin maps required')
    need(type(c['files']) is dict and 0<len(c['files'])<=65536
         and all(relative(n) and pin(v) for n,v in c['files'].items())
         and {'terminal.json','inventory.json','plan.json','intent.json'}<=set(c['files']), 'complete accepted raw collection map required')
    prefix='launch/collection/session/worker/task/collection/'
    remote=m['collection-disposition']['remote_files']
    observed={n[len(prefix):]:v['sha256'] for n,v in remote.items() if n.startswith(prefix)}
    need(observed==c['files'] and sum(v['bytes'] for n,v in remote.items() if n.startswith(prefix))<=2*1024**3,
         'collection acceptance map must equal complete original remote subtree')
    cp=m['collection-packet'];need(cp['new_input_files']['realization-acceptance.json']==pins['realization-acceptance']
         and cp['new_input_files']['namespace-claim.json']==pins['namespace-claim'], 'collection parent lineage differs')
    base=m['baseline'];need(base['closed_cpu_seconds']+3600+1800<=57600 and base['closed_gpu_slice_seconds']<=7200,
         'actual remaining budget cannot admit3CPU times600 seconds')
    return dict(metadata=m,pins=pins,job=realization_job,collection_job=collection_job)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:
        ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--collection-custody',required=True,help='local accepted raw policy_train collection subtree, hashed only')
    ap.add_argument('--tokenizer-directory',required=True,help='local six-file public tokenizer snapshot; weights forbidden')
    a=ap.parse_args();locks=[];created=[]
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=BASE/name;need(path==path.resolve(),'controller lock alias')
            handle=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'existing build; preserve and never retry')
        supplied={n:read(getattr(a,n.replace('-','_')),getattr(a,n.replace('-','_')+'_sha256')) for n in METADATA}
        ctx=admit(supplied);m=ctx['metadata'];pins=ctx['pins'];f=m['freeze'];c=m['collection-acceptance']['collection']
        need(read(LOCAL_REGISTRY,pins['namespace-claim'])==supplied['namespace-claim'],'physical recovery registry changed')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        custody=verify_collection_tree(a.collection_custody,c['files'])
        source=committed(SOURCE_COMMIT,'scripts/research/'+ENTRY);need(sha(source)==ENTRY_SHA,'reviewed entry changed')
        closure=f['stage_closure']['stages']['token'];helpers=literal(source,'HELPER_PINS')
        need(closure['helper_pins']==helpers and closure['entrypoint']==ENTRY and closure['entrypoint_sha256']==ENTRY_SHA,'token closure differs')
        origins={n:PROBE if n=='probe.py' else 'scripts/research/'+n for n in closure['input_source_files']}
        data={n:committed(SOURCE_COMMIT,path) for n,path in origins.items()}
        need(len(data)==32 and {n:sha(v) for n,v in data.items()}==closure['input_source_files']
             and all(sha(data[n])==h for n,h in helpers.items()),'exact32-source closure required')
        for n,path in origins.items():need(sha(data[n])==f['source_pins'][path],'adopted source pin differs')
        proposal_raw=committed(SOURCE_COMMIT,PROPOSAL);need(sha(proposal_raw)==PROPOSAL_SHA,'original proposal changed')
        proposal=decode(proposal_raw);need(digest(proposal)==PROPOSAL_SEMANTIC,'proposal semantics changed')
        need(c['namespaces']=={str(w['world_id']):w['training']['request_namespace'] for w in proposal['training_worlds']}
             and c['source_pins']=={**proposal['source_pins'],'scripts/research/llm_intervention_training_collection.py':
                 f['source_pins']['scripts/research/llm_intervention_training_collection.py']},'original complete collection source/namespace differs')
        token_root=Path(a.tokenizer_directory)
        need(token_root.is_dir() and not token_root.is_symlink() and {p.name for p in token_root.iterdir()}==TOKENIZER_NAMES,
             'exact six public tokenizer files required; never provide weights')
        tokens={n:read(token_root/n,f['model_files'][n]) for n in TOKENIZER_NAMES}
        data.update({'proposal.json':proposal_raw,'collection-acceptance.json':supplied['collection-acceptance']})
        need(len(data)==34 and all(len(v)<=16*1024**2 for v in data.values()),'bounded exact worker input files')
        packet=dict(schema=VERSION,scope='policy_train',source_commit=SOURCE_COMMIT,files={n:sha(v) for n,v in sorted(data.items())},
            source_names=closure['source_names'],protocol_source_pins=proposal['source_pins'],collection=c,
            archive=dict(path=f['runtime_archive_path'],sha256=f['runtime_archive_sha256']),
            manifest=dict(path=f['runtime_manifest_path'],sha256=f['runtime_manifest_sha256']),runtime_source=f['runtime_source'],
            runtime=f['runtime'],model_directory=ROOT+'/tokenizer',model_files=f['model_files'],
            tokenizer_files={n:f['model_files'][n] for n in sorted(TOKENIZER_NAMES)},resources=RESOURCES,deadlines=DEADLINES)
        need(set(packet)==set(literal(source,'PACKET_KEYS')),'exact accepted preparer packet contract required')
        diagnostic=data['qualify_llm_intervention_runtime_metadata.py'];packet_raw=raw(packet)
        command=[packet['runtime_source']+'/bin/python','-I','-B','-c',literal(diagnostic,'BOOTSTRAP'),ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
            '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(packet_raw),'--source-commit',SOURCE_COMMIT,'--output',ROOT+'/preparation']
        files={'inputs/'+n:v for n,v in data.items()};files['inputs/packet.json']=packet_raw
        files.update({'tokenizer/'+n:v for n,v in tokens.items()})
        files.update({n+('.md' if n=='freeze-review' else '.json'):v for n,v in supplied.items()})
        for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_token01',
            '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=3','#SBATCH --mem=11520M',
            '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
            '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
            "export CUDA_VISIBLE_DEVICES=''",'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
            'exec '+shlex.join(command),''])).encode()
        base=m['baseline']
        outer=dict(schema=OUTER_VERSION,source_commit=SOURCE_COMMIT,local_root=str(LOCAL),external_root=str(EXT),remote_root=ROOT,
            resources=RESOURCES,deadlines=DEADLINES,source_pins={origins[n]:sha(v) for n,v in data.items() if n in origins},
            new_input_files={n:sha(v) for n,v in sorted(files.items())},baseline_sha256=pins['baseline'],
            realization_job_id=ctx['job'],collection_job_id=ctx['collection_job'],namespace_claim_sha256=pins['namespace-claim'],
            collection_acceptance_sha256=pins['collection-acceptance'],collection_custody=custody,
            admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],
                preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=1800,candidate_gpu_reservation=0,
                total_cpu_seconds=base['closed_cpu_seconds']+5400,total_gpu_seconds=base['closed_gpu_slice_seconds'],
                ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,prior_jobs=41,prior_journals=84),
            namespace_claimed=False,submitted=False,qualification='independent concrete packet review required')
        files['packet.json']=raw(outer)
        for dest in (LOCAL,EXT):
            dest.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(dest)
            for sub in ('inputs','tokenizer'):(dest/sub).mkdir(mode=0o700)
            for n,v in files.items():write(dest/n,v)
        for n,v in files.items():need(read(LOCAL/n)==read(EXT/n)==v,'two-copy readback differs')
        print(raw(dict(local=str(LOCAL),external=str(EXT),packet_sha256=sha(files['packet.json']),
            stage_packet_sha256=sha(packet_raw),submitted=False,namespace_claimed=False)).decode())
    except BaseException as exc:
        for dest in created:
            try:write(dest/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
            except BaseException:pass
        raise
    finally:
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


if __name__=='__main__':main()


