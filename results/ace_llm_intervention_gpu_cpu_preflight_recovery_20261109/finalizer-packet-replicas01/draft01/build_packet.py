"""Prospective scientific finalizer coordinator, byte-only complete GPU custody admission.
No project imports or descriptor/outcome/token/checkpoint decoding. Future accepted
stage metadata and independent pins are mandatory. Building never submits.
"""
from pathlib import Path
import argparse,ast,contextlib,datetime,fcntl,hashlib,json,os,re,shlex,stat,subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
GPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
GPU_OUTPUT_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'
OLD_GPU_ENTRY='qualify_llm_intervention_campaign_gpu.py'
FROZEN_ENTRY_SHA='28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a'
ENTRY_SHA='8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f'
VERSION='ace-scientific-campaign-finalizer-phase-clock-v3'
OUTER_VERSION='ace-scientific-finalization-phase-clock-launch-v3'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
 memory_mib=7680,wall_seconds=4200,gpus=0,requeue=False)
DEADLINES=dict(preparation=1500,report_startup=120,report_startup_cap=1620,work=3320,cleanup=3350,hard=4190)
FROZEN_DEADLINES=dict(startup=120,work=1140,cleanup=1170,hard=1190)
GPU_RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
GPU_LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)
RAW_ROOTS=dict(control=GPU_ROOT,launch=GPU_OUTPUT_ROOT,cpu=CPU_ROOT,training=TRAIN_ROOT,token=TOKEN_ROOT,common=COLLECTION_ROOT,
 failed01='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01')
GPU_ACCEPTANCE_KEYS={'schema','status','scope','gpu_packet_sha256','cpu_packet_sha256','cpu_acceptance_sha256',
 'job_id','allocation','gpu_output','campaign_root','gpu_output_files','campaign_files','bundle_sha256',
 'attempt_pins','accepted_checks','terminal_pins','scientific_qualified'}
POLICIES=('uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm')
FIXED={'freeze':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d',
 'freeze-review':'088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416',
 'adoption':'ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07',
 'namespace-admission':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
 'recovery-authority':'ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50',
 'operational-amendment':'fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855',
 'operational-adoption-review':'db2a672a4f0222a06a46d1e61ebc5bdc22b869640f570e19d36a0f699a1ddbaf',
 'collection-failure-acceptance':'00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee',
 'no-response-proof':'59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998',
 'collection-failure-baseline':'eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b'}
STAGES=(('collection',COLLECTION_ROOT,42,86,2,0,900),('token',TOKEN_ROOT,44,90,3,0,600),
 ('training',TRAIN_ROOT,45,92,4,1,2700),('cpu',CPU_ROOT,46,94,2,0,6000),('gpu',GPU_ROOT,48,98,4,1,2880))
TEXT_METADATA={'freeze-review','operational-adoption-review','site-readiness','cpu-review','gpu-review'}
def metadata_name(n):return n+('.md' if n in TEXT_METADATA else '.json')
METADATA=tuple(FIXED)+('namespace-claim','site-readiness','cpu-review','cpu-packet','cpu-phase-terminal','gpu-review','gpu-packet','gpu-inventory-after','gpu-custody-inventory')+tuple(
 prefix+'-'+kind for prefix,*_ in STAGES for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
CPU_ACCEPTANCE_KEYS={'schema','status','cpu_packet_sha256','job_id','allocation','cpu_output','campaign_root',
 'cpu_output_files','cpu_output_excluded_directory','campaign_files','bundle_path','bundle_sha256',
 'phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}
FREEZE_SHA=FIXED['freeze']
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
FAILED_TOKEN_JOB='35415334'
FAILED_COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
TOKEN_ENTRY='prepare_llm_intervention_scientific_training_cpu.py'
TOKEN_ENTRY_SHA='47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3'
TOKEN_RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,wall_seconds=600,gpus=0,requeue=False)
TOKEN_ORIGINAL_DEADLINES=dict(startup=120,work=550,cleanup=570,hard=590)
TOKEN_DEADLINES=dict(startup=300,work=550,cleanup=570,hard=590)
FAILED_TOKEN_PACKET_SHA='a831737c6bc24a3b38269e36fa347ed3934c7803e7e18b0738881ac0b8fc323d'
FAILED_TOKEN_STAGE_SHA='101d88f3ff15ed322273d63ad281de05afd921754102a907f7232ac000801e7b'
FAILED_TOKEN_RESULT_SHA='39f045ab33f30d07126745bbc27edf41776b4ae96284f166da93256abc479f94'
COLLECTION_FAILURE_DISPOSITION_SHA='fd3a52be3054c1dd8d0d2b324dea3e64bdf8bc3676c28e7367cc0bbf06a3c685'
TEXT_METADATA |= {'token-operational-review','token-operational-adoption-review','token-operational-guide'}
TOKEN_METADATA=('collection-failure-disposition', 'failed-token-packet', 'failed-token-stage', 'failed-token-result', 'token-failure-acceptance', 'token-no-learnedwork-proof', 'token-failure-disposition', 'token-failure-inventory', 'token-failure-readback', 'token-failure-custody-inventory', 'token-failure-baseline', 'token-operational-amendment', 'token-operational-review', 'token-operational-adoption-review', 'token-operational-guide', 'token-diagnosis')
METADATA+=TOKEN_METADATA+('token-packet',)
RAW_ROOTS['failed-token01']=FAILED_TOKEN_ROOT


# Explicit operational successor. Original scientific freeze and entry remain preserved.
OLDROOT=GPU_OUTPUT_ROOT
FAILED_GPU_JOB='35498218'
ORIGINAL_GPU_ENTRY_SHA='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
SCIENTIFIC_GPU_ENTRY='qualify_llm_intervention_scientific_campaign_gpu.py'
PREFLIGHT='llm_intervention_cpu_preflight_receipt.py'
HISTORY='llm_intervention_cpu_history_admission.py'
FAILED_GPU_PACKET_SHA='436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'
FAILED_GPU_SEAL_SHA='8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0'
FAILED_GPU_DISPOSITION_SHA='545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507'
RECOVERY_AUTHORITY_SHA='b3630d400649ecd186e35232b24e9020e5bced7ad616379c7eaf5eef84498603'
RECOVERY_FILES={'authority':'gpu-recovery-authority.json','context':'cpu-history-expected.json',
 'receipt':'cpu-history-preflight.json','read_set':'cpu-history-readset.json'}
RECOVERY_METADATA=('gpu-failure-acceptance','gpu-failure-baseline','gpu-failure-packet','gpu-failure-result',
 'gpu-failure-disposition','gpu-failure-seal','gpu-failure-inventory','gpu-failure-inventory-after',
 'gpu-recovery-authority','gpu-recovery-review','operational-source-manifest','operational-source-review',
 'cpu-history-expected','cpu-history-preflight','cpu-history-readset','preflight-review')
TEXT_METADATA|={'gpu-recovery-review','operational-source-review','preflight-review','finalizer-source-review'}
METADATA+=RECOVERY_METADATA+('gpu-recovery-acceptance','finalizer-source-manifest','finalizer-source-review')
METADATA+=('finalizer-clock-adoption','finalizer-timing-evidence','finalizer-timing-review')
TEXT_METADATA|={'finalizer-timing-evidence','finalizer-timing-review'}



FULL_MAP_METADATA = frozenset(('cpu-inventory', 'cpu-disposition', 'cpu-acceptance', 'gpu-inventory', 'gpu-inventory-after', 'gpu-custody-inventory', 'gpu-disposition', 'gpu-acceptance'))

FULL_MAP_METADATA |= frozenset(('gpu-failure-acceptance','gpu-failure-disposition',
    'gpu-failure-inventory','gpu-failure-inventory-after','gpu-failure-seal','cpu-history-expected'))

def metadata_maximum(name):
    return 128*1024**2 if name in FULL_MAP_METADATA else 32*1024**2


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


def read(path,expected=None,maximum=32*1024**2):
    path=Path(path)
    need(not path.is_symlink(),'symlink input file: '+str(path))
    # macOS /tmp is a system directory alias; canonicalize its parent before opening.
    path=path.resolve(strict=True)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        info=os.fstat(f.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular independent file required')
        data=f.read(maximum+1)
        after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(info)==identity(after)==identity(path.lstat()),'input changed during capture')
    need(len(data)<=maximum,'bounded input required')
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
        if isinstance(node,ast.Constant) and type(node.value) in (str,int,float,bool,type(None)):
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
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            if node.func.id=='dict' and not node.args and all(k.arg for k in node.keywords):
                return {k.arg:resolve(k.value) for k in node.keywords}
            if node.func.id=='frozenset' and len(node.args)==1 and not node.keywords:return frozenset(resolve(node.args[0]))
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



def safe_relative(n):
    need(type(n) is str and n and not Path(n).is_absolute() and all(p not in ('','.','..') for p in n.split('/')),'safe relative member required')
    return n

def hash_file(path):
    path=Path(path);need(not path.is_symlink(),'symlink custody forbidden')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_nlink==1,'independent regular custody file required')
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(before)==identity(after)==identity(path.lstat()),'custody changed during read')
    return dict(bytes=after.st_size,sha256=h.hexdigest())

def inventory(root):
    root=Path(root);need(root.is_dir() and not root.is_symlink(),'ordinary complete raw custody root required')
    members=sorted(root.rglob('*'));need(len(members)<=262144,'bounded custody membership required')
    result={}
    for p in members:
        need(not p.is_symlink(),'custody symlink forbidden')
        if p.is_dir():continue
        result[str(p.relative_to(root))]=hash_file(p)
    return result

def cost(rows):
    cpu=gpu=0
    for r in normalized(rows):
        t=dict(r[7]);need(t or r[4]==r[5]==0,'missing actual TRES; unknown cost')
        g={k:v for k,v in t.items() if k.startswith('gres/')}
        need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()),'unknown accelerator charge')
        typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
        n=int(g['gres/gpu']) if 'gres/gpu' in g else typed
        need(not ('gres/gpu' in g and any(k.startswith('gres/gpu:') for k in g)) or n==typed,'inconsistent GPU charge')
        cpu+=r[4]*r[5];gpu+=n*r[5]
    return cpu,gpu

def failed_token_transition(m,pins):
    """Consume independently accepted failure; never infer acceptance from the traceback alone."""
    a=m['token-failure-acceptance'];proof=m['token-no-learnedwork-proof']
    d=m['token-failure-disposition'];base=m['token-failure-baseline'];old=m['collection-baseline']
    packet=m['failed-token-packet'];stage=m['failed-token-stage'];result=m['failed-token-result']
    need(pins['failed-token-packet']==FAILED_TOKEN_PACKET_SHA and pins['failed-token-stage']==FAILED_TOKEN_STAGE_SHA
         and pins['failed-token-result']==FAILED_TOKEN_RESULT_SHA,'actual original TOKEN01 input/result pins differ')
    need(a['schema']=='ace-scientific-tokenization-failure-acceptance-v1'
         and a['status']=='accepted_failed_before_token_worker' and a['failure_accepted'] is True
         and a['tokenization_accepted'] is False and a['complete_custody_verified'] is True
         and type(a['required_findings']) is int and a['required_findings']==0
         and a['job_id']==FAILED_TOKEN_JOB and a['packet_sha256']==pins['failed-token-packet']
         and a['disposition_sha256']==pins['token-failure-disposition']
         and a['freeze_sha256']==FREEZE_SHA and a['source_commit']==SOURCE_COMMIT
         and a['source_control_flow_and_traceback_verified'] is True
         and a['two_remote_inventories_matched'] is True and a['owned_worker_spawned'] is False,
         'independently accepted actual pre-token-worker failure required')
    need(proof['schema']=='ace-scientific-tokenization-no-learnedwork-proof-v1'
         and proof['status']=='verified_failed_before_token_worker' and proof['failed_job_id']==FAILED_TOKEN_JOB
         and proof['failure_acceptance_sha256']==pins['token-failure-acceptance']
         and proof['accounting_sha256']==pins['token-failure-baseline']
         and proof['packet_sha256']==pins['failed-token-packet']
         and proof['disposition_sha256']==pins['token-failure-disposition']
         and proof['freeze_sha256']==FREEZE_SHA and proof['source_commit']==SOURCE_COMMIT
         and proof['collection_acceptance_sha256']==pins['collection-acceptance']
         and proof['proof']['owned_worker_not_spawned'] is True
         and proof['proof']['token_preparation_not_invoked'] is True,
         'actual independent positive no-learnedwork boundary proof required')
    zeros=('scientific_worlds_generated','environment_response_attempts','environment_responses',
           'unknown_return_attempts','tokenizer_calls','model_loads','inference_calls','training_updates','policy_updates')
    need(all(type(v[k]) is int and v[k]==0 for v in (a,proof) for k in zeros),
         'missing/unknown learned or response work cannot authorize this recovery')
    need(packet['remote_root']==FAILED_TOKEN_ROOT and packet['source_commit']==SOURCE_COMMIT
         and packet['resources']==TOKEN_RESOURCES and packet['deadlines']==TOKEN_ORIGINAL_DEADLINES
         and packet['baseline_sha256']==pins['collection-baseline']
         and packet['new_input_files']['inputs/packet.json']==pins['failed-token-stage']
         and packet['new_input_files']['collection-acceptance.json']==pins['collection-acceptance']
         and packet['new_input_files']['namespace-claim.json']==pins['namespace-claim']
         and packet['new_input_files']['freeze.json']==FREEZE_SHA,'original failed TOKEN01 packet lineage differs')
    need(stage['source_commit']==SOURCE_COMMIT and stage['resources']==TOKEN_RESOURCES
         and stage['deadlines']==TOKEN_ORIGINAL_DEADLINES and stage['model_directory']==FAILED_TOKEN_ROOT+'/tokenizer'
         and stage['collection']==m['collection-acceptance']['collection']
         and stage['files'][TOKEN_ENTRY]==TOKEN_ENTRY_SHA and stage['source_names']==m['freeze']['stage_closure']['stages']['token']['source_names'],
         'original failed inner scientific/tokenizer/source contract differs')
    need(result['job_id']==FAILED_TOKEN_JOB and result['returncode']==0 and result['ambiguous'] is False
         and pin(result['intent_sha256']),'actual unambiguous original failed-token submission required')
    first=decode(m['token-failure-inventory']['stdout']);last=decode(m['token-failure-readback']['stdout'])
    need(m['token-failure-inventory']['returncode']==m['token-failure-readback']['returncode']==0
         and first['terminal_gate'] is last['terminal_gate'] is True
         and not first.get('observation_error') and not last.get('observation_error'), 'complete original token transport observations required')
    need(all(literal(m[n]['python_source'],'ROOT')==FAILED_TOKEN_ROOT and literal(m[n]['python_source'],'JOB')==FAILED_TOKEN_JOB
             for n in ('token-failure-inventory','token-failure-readback')), 'failed token captured inventory root/job differs')
    need(all(first[k]==last[k] for k in ('files','directories','original_hardlink_groups','journal_pins','parent_accounting','upstream_pins','upstream_errors'))
         and normalized(first['related_accounting'])==normalized(last['related_accounting'])
         and first['files']==d['remote_files'] and len(first['files'])==d['raw_files']
         and sum(v['bytes'] for v in first['files'].values())==d['raw_bytes'], 'complete failed token custody differs')
    need(d['job_id']==FAILED_TOKEN_JOB and d['remote_root']==FAILED_TOKEN_ROOT
         and d['packet_sha256']==pins['failed-token-packet'] and d['submission_result_sha256']==pins['failed-token-result']
         and d['status']=='terminal_failed_outputs_preserved' and d['custody_bytes_verified'] is d['current_lineage_verified'] is True
         and d['errors']==d['input_pin_mismatches']==[] and d['acceptance'] is d['tokenization_accepted'] is False,
         'actual immutable failed TOKEN disposition required')
    for name,field in [('token-failure-inventory','inventory_transport.json'),('token-failure-readback','inventory_after_transport.json'),
                       ('token-failure-custody-inventory','custody-inventory.json')]:
        need(a['evidence_pins'][field]==pins[name],'independent failure evidence pin differs')
    custody=m['token-failure-custody-inventory']
    need(custody['identical'] is True and len(custody['copies'])==2,'two complete original token custody copies required')
    for view in custody['copies'].values():
        need(view['files']['disposition.json']['sha256']==pins['token-failure-disposition']
             and view['files']['inventory_transport.json']['sha256']==pins['token-failure-inventory']
             and view['files']['inventory_after_transport.json']['sha256']==pins['token-failure-readback']
             and {n[len('remote/'):]:v for n,v in view['files'].items() if n.startswith('remote/')}==first['files'],
             'independently pinned full capsule differs from original raw inventory')
    for n,h in {**packet['new_input_files'],'packet.json':pins['failed-token-packet'],
                'submission.intent.json':result['intent_sha256'],'submission.result.json':pins['failed-token-result']}.items():
        need(first['files']['launch/'+n]['sha256']==h,'failed token original input/journal bytes differ')
    need('launch/preparation/failure.json' in first['files']
         and not any(n.startswith('launch/preparation/session/worker/') for n in first['files']),
         'independently audited pre-worker custody boundary differs')
    rows=base['related_accounting'];prior=[r for r in rows if r[0]!=FAILED_TOKEN_JOB];current=[r for r in rows if r[0]==FAILED_TOKEN_JOB]
    need(base['input_pins']['predecessor']==pins['collection-baseline'] and base['input_pins']['acceptance']==pins['token-failure-acceptance']
         and base['input_pins']['disposition']==pins['token-failure-disposition']
         and len(rows)==base['terminal_jobs']==43 and base['submission_journals']==88
         and old['terminal_jobs']==42 and old['submission_journals']==86
         and len(current)==1 and normalized(prior)==normalized(old['related_accounting'])
         and normalized(current)==normalized([first['parent_accounting']])
         and normalized(rows)==normalized(first['related_accounting']), 'actual42/86 to failed43/88 extension required')
    row=normalized(current)[0];tres=dict(row[7])
    need(row[1]=='ace_llm_scientific_token01' and row[2:4]==['FAILED','1:0'] and row[4]==3
         and 0<row[5]<=600 and row[6]=='ucb736_asc2' and tres.get('cpu')=='3'
         and tres.get('node')=='1' and tres.get('mem') in ('11520M','11.25G')
         and not any(k.startswith('gres/') for k in tres), 'actual bounded failed3CPU allocation required')
    journals={**old['journal_pins'],FAILED_TOKEN_ROOT+'/submission.intent.json':result['intent_sha256'],
              FAILED_TOKEN_ROOT+'/submission.result.json':pins['failed-token-result']}
    need(len(journals)==88 and journals==base['journal_pins']==first['journal_pins'], 'all original journals plus failed token pair required')
    need(base['closed_cpu_seconds']==d['closed_cpu_seconds']==old['closed_cpu_seconds']+3*row[5]
         and base['closed_gpu_slice_seconds']==d['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']
         and sum(int(r[4])*int(r[5]) for r in rows)==base['closed_cpu_seconds']
         and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600
         and base['outstanding_reservation']==0 and base['preparation_allowance_cpu_seconds']==3600,
         'actual failed token charge must remain fully included; no default cost')
    preserved={}
    for label,root in [('launch',FAILED_TOKEN_ROOT),('common',COLLECTION_ROOT),('failed01',FAILED_COLLECTION_ROOT)]:
        files={n[len(label)+1:]:v['sha256'] for n,v in first['files'].items() if n.startswith(label+'/')}
        need(files and all(safe_relative('launch/'+n) and pin(h) for n,h in files.items()),'complete bounded failed/prior raw root required')
        preserved[root]=files
    for label,metadata in [('common','collection-disposition'),('failed01','collection-failure-disposition')]:
        want={n[len('launch/'):]:v['sha256'] for n,v in m[metadata]['remote_files'].items() if n.startswith('launch/')}
        root=COLLECTION_ROOT if label=='common' else FAILED_COLLECTION_ROOT
        need(preserved[root]==want,'failed-token capsule must preserve whole original collection02/01 roots')
    return dict(files=first['files'],directories=first['directories'],preserved_roots=preserved)

def token_operational_binding(supplied,m,pins):
    expected=dict(schema='ace-scientific-tokenization-operational-amendment-v1',status='adopted',stage='tokenize',recovery_id='token02',
        freeze_sha256=FREEZE_SHA,source_commit=SOURCE_COMMIT,entry_sha256=TOKEN_ENTRY_SHA,
        failed_job_id=FAILED_TOKEN_JOB,failed_packet_sha256=pins['failed-token-packet'],failed_stage_packet_sha256=pins['failed-token-stage'],
        failed_root=FAILED_TOKEN_ROOT,recovery_root=TOKEN_ROOT,
        failure_acceptance_sha256=pins['token-failure-acceptance'],no_learnedwork_proof_sha256=pins['token-no-learnedwork-proof'],
        failure_disposition_sha256=pins['token-failure-disposition'],failure_inventory_sha256=pins['token-failure-inventory'],
        failure_readback_sha256=pins['token-failure-readback'],failure_custody_inventory_sha256=pins['token-failure-custody-inventory'],
        baseline_sha256=pins['token-failure-baseline'],collection_acceptance_sha256=pins['collection-acceptance'],
        collection_baseline_sha256=pins['collection-baseline'],namespace_admission_sha256=pins['namespace-admission'],namespace_claim_sha256=pins['namespace-claim'],
        original_namespace_claim_sha256=ORIGINAL_CLAIM_SHA,original_deadlines=TOKEN_ORIGINAL_DEADLINES,deadlines=TOKEN_DEADLINES,resources=TOKEN_RESOURCES,
        review_sha256=pins['token-operational-review'],guide_sha256=pins['token-operational-guide'],diagnosis_sha256=pins['token-diagnosis'],
        required_findings=0,scientific_inputs_unchanged=True,other_stages_unchanged=True,one_submission_only=True,
        automatic_retry_allowed=False,replacement_worlds_allowed=False)
    need(raw(m['token-operational-amendment'])==raw(expected)
         and supplied['token-operational-review'].strip() and supplied['token-operational-adoption-review'].strip()
         and supplied['token-operational-guide'].strip(), 'exact independently pinned token-only operational amendment/reviews required')
    need(pins['token-diagnosis']=='d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63',
         'independent specific ValueError diagnosis required; do not infer SIGALRM or IO cause')
    return dict(amendment_sha256=pins['token-operational-amendment'],adoption_review_sha256=pins['token-operational-adoption-review'],
        failure_acceptance_sha256=pins['token-failure-acceptance'],no_learnedwork_proof_sha256=pins['token-no-learnedwork-proof'],
        failure_baseline_sha256=pins['token-failure-baseline'],original_packet_sha256=pins['failed-token-packet'],
        original_stage_packet_sha256=pins['failed-token-stage'])

def token_science_equality(old,new):
    before=dict(old);after=dict(new)
    need(before.pop('deadlines')==TOKEN_ORIGINAL_DEADLINES and after.pop('deadlines')==TOKEN_DEADLINES
         and before.pop('model_directory')==FAILED_TOKEN_ROOT+'/tokenizer'
         and after.pop('model_directory')==TOKEN_ROOT+'/tokenizer','only explicit startup/root changes permitted')
    need(raw(before)==raw(after),'complete child scientific/source/data/runtime/numerical contract must remain unchanged')
    return dict(original_stage_sha256=sha(raw(old)),new_stage_sha256=sha(raw(new)),nonoperational_semantic_sha256=digest(before),
        changed_fields=['deadlines.startup','model_directory'],scientific_inputs_unchanged=True)

def recovery_joins(m,pins,recovery):
    """Bind unchanged producer bytes; do not relabel an old success as a new attempt."""
    for stage in ('token','training','cpu','gpu'):
        outer=m[stage+'-launch-packet'];captured=outer['new_input_files']
        need(outer['token_recovery']==recovery,'exact upstream TOKEN recovery object differs: '+stage)
        if stage!='token':
            need(outer['failed_token_root']==FAILED_TOKEN_ROOT and outer['failed_token_job_id']==FAILED_TOKEN_JOB,
                 'upstream original failed TOKEN identity differs: '+stage)
        for name in TOKEN_METADATA:
            if stage=='token' and name=='collection-failure-disposition':continue
            need(captured[metadata_name(name)]==pins[name],'upstream recovery bytes differ: '+stage+'/'+name)
    # CPU coordinator follows the independently reviewed training consumer aliases.
    for stage,fields in {
        'training':{'cpu-packet.json':'token-packet'},
        'cpu':{'collection-baseline.json':'collection-baseline','collection-acceptance.json':'collection-acceptance',
               'collection-disposition.json':'collection-disposition','cpu-packet.json':'token-packet',
               'cpu-launch-packet.json':'token-launch-packet','cpu-result.json':'token-result',
               'cpu-disposition.json':'token-disposition','cpu-inventory.json':'token-inventory'},
        'gpu':{'token-packet.json':'token-packet'}
    }.items():
        for name,key in fields.items():
            need(m[stage+'-launch-packet']['new_input_files'][name]==pins[key], 'captured upstream metadata alias differs: '+stage+'/'+name)


def observed_root(observation,packet_pin):
    """Unique whole root by original packet bytes, independent of collector label."""
    files=observation['files'];directories=observation['directories']
    labels=[n.split('/')[0] for n,v in files.items() if n.count('/')==1 and n.endswith('/packet.json') and v['sha256']==packet_pin]
    need(len(labels)==1 and labels[0] in directories,'unique complete original root required')
    prefix=labels[0]+'/'
    return dict(files={n[len(prefix):]:v for n,v in files.items() if n.startswith(prefix)},
                directories=sorted(n[len(prefix):] for n in directories if n.startswith(prefix)))


def retained_root_history(m,pins,observations):
    failure=decode(m['token-failure-inventory']['stdout'])
    originals={'collection':(pins['collection-launch-packet'],observed_root(observations['collection'],pins['collection-launch-packet'])),
        'failed-token':(pins['failed-token-packet'],observed_root(failure,pins['failed-token-packet']))}
    failed_collection_pin=m['collection-failure-disposition']['packet_sha256']
    originals['failed-collection']=(failed_collection_pin,observed_root(failure,failed_collection_pin))
    for stage in ('token','training','cpu','gpu'):
        obs=observations[stage]
        for previous,(packet_pin,expected) in originals.items():
            actual=observed_root(obs,packet_pin)
            if previous=='cpu' and stage=='gpu':
                # Only the already-declared GPU arm append is permitted in the common campaign.
                additions=tuple('cpu/worker/artifacts/campaign/policy-'+i for i in ('05','06','07'))
                permitted=lambda n:any(n==p or n.startswith(p+'/') for p in additions)
                actual=dict(files={n:v for n,v in actual['files'].items() if not permitted(n)},
                            directories=[n for n in actual['directories'] if not permitted(n)])
            need(actual==expected,'complete immutable upstream files/directories differ: '+stage+'/'+previous)
        originals[stage]=(pins[stage+'-launch-packet'],observed_root(obs,pins[stage+'-launch-packet']))


def history(m,pins):
    previous=m['collection-failure-baseline'];previous_pin=pins['collection-failure-baseline'];jobs=[];observations={}
    need(previous['terminal_jobs']==41 and previous['submission_journals']==84,'accepted failed41/84 boundary required')
    for prefix,root,count,journals,cpus,gpus,wall in STAGES:
        if prefix=='token':
            failed_token_transition(m,pins)
            previous=m['token-failure-baseline'];previous_pin=pins['token-failure-baseline']
            jobs.append(FAILED_TOKEN_JOB)
        if prefix=='gpu':
            observations['failed-gpu-root']=failed_gpu_history(m,pins)
            previous=m['gpu-failure-baseline'];previous_pin=pins['gpu-failure-baseline']
            jobs.append(FAILED_GPU_JOB)
        a,p,r,d,t,base=(m[prefix+'-'+kind] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
        job=r['job_id'];need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job not in jobs+['35407428'],'distinct actual job required')
        jobs.append(job)
        need(r['returncode']==0 and r['ambiguous'] is False and pin(r['intent_sha256']) and a['job_id']==job,'accepted original submission differs')
        need(p['remote_root']==root and p['source_commit']==SOURCE_COMMIT and p['baseline_sha256']==previous_pin,'original stage source/root/predecessor differs')
        schemas=dict(collection='ace-scientific-collection-launch-v1',token='ace-scientific-tokenization-launch-v1',
                     training='ace-scientific-training-gpu-launch-v1',cpu='ace-scientific-cpu-campaign-launch-v1',gpu='ace-scientific-campaign-gpu-recovery-launch-v2')
        request=p['resources'];memory=dict(collection=7680,token=11520,training=15360,cpu=7680,gpu=15360)[prefix]
        need(p['schema']==schemas[prefix] and request['wall_seconds']==wall and request['memory_mib']==memory
             and request['gpus']==gpus and request['account']=='ucb736_asc2'
             and request['partition']==('aa100' if gpus else 'acpu') and request['qos']==('gpu-testing' if gpus else 'cpu-normal')
             and request.get('cpus',request.get('requested_cpus'))==cpus,'original stage profile differs')
        need(d['job_id']==job and d['remote_root']==root and d['packet_sha256']==pins[prefix+'-launch-packet']
             and d['submission_result_sha256']==pins[prefix+'-result'] and d['custody_bytes_verified'] is d['current_lineage_verified'] is True
             and d['errors']==d['input_pin_mismatches']==[] and d['exact_prior_accounting_matched'] is True
             and d['exact_prior_and_current_journals_matched'] is True and d['observed_related_jobs']==count
             and d['observed_submission_journals']==journals,'independently accepted complete original stage custody required')
        need(t['returncode']==0 and literal(t['python_source'],'ROOT')==root and literal(t['python_source'],'JOB')==job,'original inventory transport identity differs')
        obs=decode(t['stdout']);observations[prefix]=obs
        need(obs['terminal_gate'] is True and obs['root_present'] is True and not obs.get('observation_error')
             and obs['parent_accounting'][0]==job and len(obs['files'])==d['raw_files']
             and sum(v['bytes'] for v in obs['files'].values())==d['raw_bytes'],'complete original raw inventory differs')
        for n,v in obs['files'].items():
            safe_relative(n);need(set(v)=={'bytes','sha256'} and type(v['bytes']) is int and v['bytes']>=0 and pin(v['sha256']),'safe byte inventory required')
        for n,h in {**p['new_input_files'],'packet.json':pins[prefix+'-launch-packet'],
                    'submission.intent.json':r['intent_sha256'],'submission.result.json':pins[prefix+'-result']}.items():
            safe_relative(n);need(pin(h) and obs['files'][('control/' if prefix=='gpu' else 'launch/')+n]['sha256']==h,'original stage inputs/journals differ')
        need(base['terminal_jobs']==count and base['submission_journals']==journals and base['outstanding_reservation']==0
             and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
             and base['input_pins']['predecessor']==previous_pin and base['input_pins']['acceptance']==pins[prefix+'-acceptance']
             and base['input_pins']['disposition']==pins[prefix+'-disposition'],'independent accepted baseline extension differs')
        prior=[r for r in base['related_accounting'] if r[0]!=job];current=[r for r in base['related_accounting'] if r[0]==job]
        need(len(prior)==count-1 and len(current)==1 and normalized(prior)==normalized(previous['related_accounting'])
             and normalized(current)==normalized([obs['parent_accounting']])
             and normalized(base['related_accounting'])==normalized(obs['related_accounting']),'exact complete row history differs')
        row=normalized(current)[0];need(row[2:4]==['COMPLETED','0:0'] and row[4]==cpus and 0<=row[5]<=wall
             and row[6]=='ucb736_asc2' and cost(current)==(cpus*row[5],gpus*row[5]),'accepted bounded allocation differs')
        jp={**previous['journal_pins'],root+'/submission.intent.json':r['intent_sha256'],root+'/submission.result.json':pins[prefix+'-result']}
        need(len(jp)==journals and jp==base['journal_pins']==obs['journal_pins'],'exact predecessor journals differ')
        before=cost(previous['related_accounting']);after=cost(base['related_accounting'])
        need(before==(previous['closed_cpu_seconds'],previous['closed_gpu_slice_seconds'])
             and after==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])
             and after==(before[0]+cpus*row[5],before[1]+gpus*row[5])
             and base['cpu_admission_seconds']==after[0]+3600,'actual full-chain charges differ')
        previous,previous_pin=base,pins[prefix+'-baseline']
    return observations

def authenticate(supplied):
    need(set(supplied)==set(METADATA),'complete independent metadata inputs required')
    pins={n:sha(v) for n,v in supplied.items()};need(all(pins[n]==h for n,h in FIXED.items()),'fixed original authority bytes differ')
    need(all(supplied[n].strip() for n in TEXT_METADATA),'independent review/readiness evidence required')
    m={n:decode(v) for n,v in supplied.items() if n not in TEXT_METADATA};f=m['freeze']
    need(m['adoption']==dict(schema='ace-scientific-contract-adoption-v1',status='adopted',freeze_sha256=pins['freeze'],
         review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True),'original independent freeze adoption required')
    need(f['source_commit']==SOURCE_COMMIT and f['stage_closure']['source_pins']==f['source_pins'] and len(f['source_pins'])==67,'original full frozen source authority required')
    stages={s['stage']:s for s in f['stages']};s=stages['finalize']
    need(s['entry']==ENTRY and s['entry_sha256']==FROZEN_ENTRY_SHA and s['remote_root']==ROOT and s['deadlines']==FROZEN_DEADLINES
         and s['cpus']==2 and s['memory_mib']==7680 and s['wall_seconds']==1200 and s['gpus']==0 and s['gpu_type'] is None
         and s['account']=='ucb736_asc2' and s['partition']=='acpu' and s['qos']=='cpu-normal' and s['requeue'] is False,'frozen scientific finalizer profile differs')
    claim=m['namespace-claim'];am=m['operational-amendment']
    need(claim['freeze_sha256']==pins['freeze'] and claim['namespace_admission_sha256']==pins['namespace-admission']
         and claim['world_ids']==m['namespace-admission']['world_ids'] and claim['local_registry']==str(LOCAL_REGISTRY) and claim['remote_registry']==REMOTE_REGISTRY
         and claim['recovery_authority_sha256']==pins['recovery-authority'] and claim['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA
         and claim['claim_kind']=='explicit_pre_draw_recovery02' and claim['replacement_worlds'] is claim['retry_allowed'] is False,
         'preserved recovery claim authority differs')
    need(am['freeze_sha256']==pins['freeze'] and am['source_commit']==SOURCE_COMMIT and am['recovery_root']==COLLECTION_ROOT
         and am['namespace_claim_sha256']==pins['namespace-claim'] and am['baseline_sha256']==pins['collection-failure-baseline']
         and am['failure_acceptance_sha256']==pins['collection-failure-acceptance'] and am['no_response_proof_sha256']==pins['no-response-proof']
         and am['scientific_inputs_unchanged'] is am['other_stages_unchanged'] is True,'collection-only amendment lineage differs')
    need(pins['collection-failure-disposition']==COLLECTION_FAILURE_DISPOSITION_SHA,'original collection failure disposition differs')
    token_recovery=token_operational_binding(supplied,m,pins)
    token_science_equality(m['failed-token-stage'],m['token-packet'])
    need(pins['token-packet']==m['token-acceptance']['cpu_packet_sha256'],'accepted TOKEN02 inner pin differs')
    recovery_joins(m,pins,token_recovery)
    observations=history(m,pins)
    retained_root_history(m,pins,observations)
    collection=m['collection-acceptance'];token=m['token-acceptance'];train=m['training-acceptance'];cpu=m['cpu-acceptance'];old=m['cpu-packet']
    need(collection['status']=='accepted' and type(collection['required_findings']) is int and collection['required_findings']==0
         and collection['complete_custody_verified'] is True and collection['collection_packet_sha256']==pins['collection-launch-packet']
         and collection['disposition_sha256']==pins['collection-disposition'],'accepted full collection required')
    need(token['schema']=='ace-scientific-training-preparation-acceptance-v1' and token['status']=='accepted' and token['scope']=='policy_train'
         and type(token['required_findings']) is int and token['required_findings']==0
         and token['complete_custody_verified'] is token['runtime_qualified_for_this_cpu_execution'] is token['owned_process_cleanup_verified'] is True
         and token['training_execution_qualified'] is token['scientific_qualified'] is False
         and token['freeze_sha256']==pins['freeze'] and token['namespace_admission_sha256']==pins['namespace-admission'],'accepted token preparation required')
    need(token['cpu_packet_sha256']==m['token-launch-packet']['new_input_files']['inputs/packet.json'],
         'accepted token inner packet differs from original submission')
    need(train['schema']=='ace-scientific-training-gpu-acceptance-v1' and train['status']=='scientific_training_gpu_accepted'
         and train['scope']=='policy_train' and type(train['required_findings']) is int and train['required_findings']==0
         and train['scientific_utility_qualified'] is False and all(train[k] is True for k in ('training_gpu_qualified',
          'checkpoint_reload_inference_qualified','complete_custody_verified','runtime_qualified','owned_process_cleanup_verified','scheduler_verified')),
         'accepted actual training required')
    expected=old['expected_training']
    expected_pins={'prepared_sha256','training_plan_sha256','exposure_sha256','scientific_freeze_sha256',
                  'collection_acceptance_sha256','cpu_preparation_acceptance_sha256','model_files_sha256','runtime_contract_sha256'}
    need(set(expected)==expected_pins|{'job_id','source_commit'} and all(pin(expected[k]) for k in expected_pins),
         'complete independently accepted CPU expected-training interface required')
    need(expected['source_commit']==SOURCE_COMMIT and expected['job_id']==train['job_id'] and all(train[k]==v for k,v in expected.items())
         and expected['scientific_freeze_sha256']==pins['freeze'] and expected['collection_acceptance_sha256']==pins['collection-acceptance']
         and expected['cpu_preparation_acceptance_sha256']==pins['token-acceptance']
         and expected['prepared_sha256']==token['prepared_sha256'] and expected['training_plan_sha256']==token['plan_sha256']
         and expected['exposure_sha256']==token['exposure_sha256'] and expected['model_files_sha256']==digest(f['model_files'])
         and expected['runtime_contract_sha256']==digest(f['runtime']),'matched prepared/plan/exposure/base/runtime training lineage differs')
    need(set(cpu)==CPU_ACCEPTANCE_KEYS and cpu['schema']=='ace-scientific-campaign-cpu-acceptance-v1'
         and cpu['status']=='development_cpu_phase_accepted' and cpu['scientific_qualified'] is False
         and cpu['cpu_packet_sha256']==pins['cpu-packet']
         and cpu['allocation']==dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,state='COMPLETED',exit_code='0:0'),
         'exact independently accepted CPU campaign required')
    need(old['schema']=='ace-scientific-campaign-cpu-v1' and old['scope']=='development' and old['source_commit']==SOURCE_COMMIT
         and old['freeze_sha256']==pins['freeze'] and old['namespace_admission_sha256']==pins['namespace-admission']
         and old['files']['scientific-freeze.json']==pins['freeze'] and old['files']['gpu-training-acceptance.json']==pins['training-acceptance']
         and old['files']['cpu-preparation-acceptance.json']==pins['token-acceptance']
         and old['files']['prepared.json']==token['prepared_sha256'] and old['files']['numerical-plan.json']==token['numerical_plan_file_sha256']
         and old['numerical_plan_sha256']==token['numerical_plan_sha256'],'original CPU packet/input plan differs')
    need(old['runtime']==f['runtime'] and old['runtime_source']==f['runtime_source'] and old['model_directory']==f['model_directory']
         and old['model_files']==f['model_files'] and old['archive']==dict(path=f['runtime_archive_path'],sha256=f['runtime_archive_sha256'])
         and old['manifest']==dict(path=f['runtime_manifest_path'],sha256=f['runtime_manifest_sha256']),'original CPU model/runtime differs')
    need(set(old['world_pins'])=={str(i) for i in range(202611070100,202611070124)} and all(pin(h) for h in old['world_pins'].values()),'exact24 accepted development descriptors required')
    # Join nested original upstream acceptances and baseline bytes, not claims copied into a new history.
    for prefix,required in [('token',{'collection-acceptance.json':pins['collection-acceptance']}),
       ('training',{'cpu-acceptance.json':pins['token-acceptance'],'collection-acceptance.json':pins['collection-acceptance']}),
       ('cpu',{'gpu-training-acceptance.json':pins['training-acceptance'],'cpu-preparation-acceptance.json':pins['token-acceptance']})]:
        recorded=m[prefix+'-launch-packet']['new_input_files']
        for name,h in required.items():need(recorded[name]==h,'original upstream acceptance bytes differ: '+prefix+'/'+name)
    joins={
        'token':{**{'collection-'+k+'.json':'collection-'+('launch-packet' if k=='packet' else k)
                         for k in ('packet','result','disposition')},'baseline.json':'collection-baseline'},
        'training':{'cpu-launch-packet.json':'token-launch-packet','cpu-result.json':'token-result',
                    'cpu-disposition.json':'token-disposition','cpu-inventory.json':'token-inventory',
                    'baseline.json':'token-baseline','previous-baseline.json':'collection-baseline',
                    'collection-inventory.json':'collection-inventory'},
        'cpu':{'training-packet.json':'training-launch-packet','training-result.json':'training-result',
               'training-disposition.json':'training-disposition','training-inventory.json':'training-inventory',
               'baseline.json':'training-baseline','previous-baseline.json':'token-baseline'}}
    for prefix,fields in joins.items():
        recorded=m[prefix+'-launch-packet']['new_input_files']
        for name,key in fields.items():need(recorded[name]==pins[key],'original captured predecessor metadata differs: '+prefix+'/'+name)
    for prefix,*_ in STAGES:
        recorded=m[prefix+'-launch-packet']['new_input_files']
        for name in FIXED:
            # The exact amendment adoption review followed collection02 authoring;
            # its original packet captures the prior operational design review instead.
            if prefix=='collection' and name=='operational-adoption-review':
                need(recorded['operational-review.md']=='2d472a124d471c3ebe8fdcc9aaea2bdcec5a949486432ec26127886ca65952cc',
                     'original collection operational design review differs')
                continue
            # collection02 calls its failed predecessor baseline failure-baseline.json.
            file='failure-baseline.json' if prefix=='collection' and name=='collection-failure-baseline' else metadata_name(name)
            need(recorded[file]==pins[name],'original immutable authority differs: '+prefix+'/'+file)
    base=m['gpu-baseline'];remaining=[stages['finalize']]
    gpu_authority(m,pins)
    recovery=recovery_authority(supplied,m,pins,observations)
    # The GPU producer captures every predecessor metadata argument byte unchanged.
    for name in tuple(FIXED)+TOKEN_METADATA+('token-packet','namespace-claim','cpu-review','cpu-packet','cpu-phase-terminal')+tuple(
        prefix+'-'+kind for prefix,*_ in STAGES[:-1] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline')):
        need(m['gpu-launch-packet']['new_input_files'][metadata_name(name)]==pins[name], 'original GPU lineage capture differs: '+name)
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
         and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,'full remaining frozen caps exceed budget')
    clock_admission(m,pins)
    return dict(metadata=m,pins=pins,observations=observations,token_recovery=token_recovery,gpu_recovery=recovery)

def gpu_authority(m,pins):
    """Public acceptance/packet joins only. No attempt or report payload decoding."""
    g=m['gpu-acceptance'];gp=m['gpu-packet'];cpu=m['cpu-acceptance'];old=m['cpu-packet']
    need(set(g)==GPU_ACCEPTANCE_KEYS and g['schema']=='ace-scientific-campaign-gpu-acceptance-v1'
         and g['status']=='development_gpu_phase_accepted' and g['scope']=='development'
         and g['scientific_qualified'] is False and g['job_id']==m['gpu-result']['job_id'], 'independent scientific GPU acceptance required')
    need(all(g[k]==pins[n] for k,n in [('gpu_packet_sha256','gpu-packet'),('cpu_packet_sha256','cpu-packet'),
         ('cpu_acceptance_sha256','cpu-acceptance')]), 'original CPU/GPU acceptance join differs')
    need(g['accepted_checks']==dict(scheduler=True,runtime=True,cleanup=True,custody=True)
         and all(type(v) is bool for v in g['accepted_checks'].values()), 'all independent GPU checks required')
    need(g['allocation']==dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
         allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',state='COMPLETED',exit_code='0:0'),
         'actual frozen GPU allocation differs')
    need(gp['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2' and gp['scope']=='development'
         and gp['resources']==GPU_RESOURCES and gp['limits']==GPU_LIMITS
         and gp['freeze_sha256']==pins['freeze'] and gp['namespace_admission_sha256']==pins['namespace-admission']
         and gp['cpu_packet_sha256']==pins['cpu-packet'] and gp['cpu_acceptance_sha256']==pins['cpu-acceptance'],
         'original frozen scientific GPU input contract differs')
    need(m['gpu-launch-packet']['new_input_files']['inputs/packet.json']==pins['gpu-packet']
         and m['gpu-launch-packet']['gpu_packet_sha256']==pins['gpu-packet'], 'original GPU inner packet differs')
    paths=dict(cpu_inputs=CPU_ROOT+'/inputs',cpu_output=CPU_ROOT+'/cpu',campaign=CPU_ROOT+'/cpu/worker/artifacts/campaign',
               gpu_inputs=GPU_ROOT+'/inputs',gpu_output=GPU_OUTPUT_ROOT+'/gpu')
    need(gp['paths']==paths and g['gpu_output']==paths['gpu_output']
         and g['campaign_root']==cpu['campaign_root']==paths['campaign'] and cpu['cpu_output']==old['cpu_output']==paths['cpu_output']
         and cpu['cpu_output_excluded_directory']=='worker/artifacts/campaign', 'exact inherited evidence roots required')
    need(all(gp['files'].get(n)==h for n,h in old['files'].items())
         and gp['files']['cpu-packet.json']==pins['cpu-packet'] and gp['files']['cpu-acceptance.json']==pins['cpu-acceptance'],
         'prior input/acceptance bytes changed')
    need(g['bundle_sha256']==cpu['bundle_sha256'] and set(g['attempt_pins'])==set(POLICIES)
         and set(cpu['attempt_pins'])==set(POLICIES[:5]) and all(pin(h) for h in g['attempt_pins'].values())
         and all(g['attempt_pins'][k]==v for k,v in cpu['attempt_pins'].items()), 'all eight accepted original attempts required')
    files=g['campaign_files']
    need(type(files) is dict and files and {n:h for n,h in files.items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}}==cpu['campaign_files'],
         'CPU bytes changed or unexpected campaign append')
    need({n.split('/')[0] for n in files}=={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}' for i in range(8)},
         'complete campaign required; previous finalization forbidden')
    for i,pid in enumerate(POLICIES):
        need(all(files.get(f'policy-{i:02d}/'+n)==g['attempt_pins'][pid] for n in ('attempt.json','attempt.candidate.json')),
             'original accepted attempt/candidate differs')
    need(set(g['terminal_pins'])=={'terminal.json','controller/phase/terminal.json'}
         and all(pin(h) and g['gpu_output_files'].get(n)==h for n,h in g['terminal_pins'].items()), 'accepted full GPU terminal pins required')
    for k in ('bundle','phase_terminal'):
        path=cpu[k+'_path'];safe_relative(path)
        need(cpu['cpu_output_files'].get(path)==cpu[k+'_sha256'], 'accepted CPU terminal/bundle missing')
    need(cpu['phase_terminal_sha256']==pins['cpu-phase-terminal'], 'accepted CPU phase metadata differs')
    term=m['cpu-phase-terminal'];d=m['training-acceptance']['dpo_steps']
    need(type(d) is int and 1<=d<=128 and term['schema']=='ace-scientific-campaign-cpu-v1' and term['status']=='cpu_phase_complete'
         and term['numeric_training_updates']==160+d and term['llm_training_updates']==0 and term['completed_cpu_cells']==120
         and term['declared_cells']==192 and term['gpu_services_started']==0 and term['attempt_pins']==cpu['attempt_pins']
         and term['private_scores_computed'] is term['scientific_qualified'] is term['retry_allowed'] is False, 'complete matched CPU phase required')


def tree(root):
    """Complete local opaque inventory, including empty directories."""
    root=Path(root);files=inventory(root)
    dirs=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir())
    return dict(files=files,directories=dirs)


def accepted_rosters(ctx):
    m=ctx['metadata'];g=m['gpu-acceptance'];c=m['cpu-acceptance'];obs=ctx['observations']['gpu'];files=obs['files']
    need(set(obs['directories'])==set(m['gpu-disposition']['remote_directories']) and files==m['gpu-disposition']['remote_files'],
         'original full GPU disposition/inventory differs')
    for label in RAW_ROOTS:need(label in obs['directories'], 'missing full upstream root: '+label)
    for name,expected in [('namespace.json',ctx['pins']['namespace-claim']),('original-namespace.json',ORIGINAL_CLAIM_SHA)]:
        need(files[name]['sha256']==expected, 'original namespace custody differs')
    def sub(prefix):return {n[len(prefix):]:v['sha256'] for n,v in files.items() if n.startswith(prefix)}
    cpu=sub('cpu/cpu/');campaign='worker/artifacts/campaign/'
    need({n:h for n,h in cpu.items() if not n.startswith(campaign)}==c['cpu_output_files']
         and sub('cpu/cpu/'+campaign)==g['campaign_files'] and sub('launch/gpu/')==g['gpu_output_files'],
         'exact complete independent CPU/GPU/campaign inventories required')
    cp='cpu/cpu/'+campaign
    tops={n[len(cp):].split('/')[0] for n in list(files)+obs['directories'] if n.startswith(cp)}
    need(tops=={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}' for i in range(8)},
         'unexpected/empty campaign directory; no previous private finalization')
    rosters={}
    for label,root in RAW_ROOTS.items():
        prefix=label+'/'
        rosters[root]=dict(files=sub(prefix),directories=sorted(n[len(prefix):] for n in obs['directories'] if n.startswith(prefix)))
    need(set(files)=={label+'/'+n for label,root in RAW_ROOTS.items() for n in rosters[root]['files']}|{'namespace.json','original-namespace.json'},
         'unexpected raw top-level member')
    return rosters


def nested_failure_capsule(ctx,inventory):
    """Complete original archive/receipts as well as raw root must survive upstream custody."""
    seal=ctx['metadata']['token-failure-custody-inventory'];pin_value=ctx['pins']['token-failure-custody-inventory']
    views=list(seal['copies'].values())
    need(seal['identical'] is True and len(views)==2 and views[0]==views[1],'original failed capsule copy maps differ')
    matches=[n for n,v in inventory['files'].items() if Path(n).name=='custody-inventory.json' and v['sha256']==pin_value]
    need(matches,'complete original failed TOKEN capsule, archive and receipts must remain in upstream custody')
    for name in matches:
        prefix=name[:-len('custody-inventory.json')]
        observed=dict(files={n[len(prefix):]:v for n,v in inventory['files'].items() if n.startswith(prefix) and n!=name},
                      directories=sorted(n[len(prefix):] for n in inventory['directories'] if n.startswith(prefix) and n!=prefix.rstrip('/')))
        need(observed==views[0],'nested original failure capsule bytes/directories differ')


def admit_gpu_bytes(ctx,capsule,mirror):
    """Authenticate both sealed complete copies; inputs alone never suffice."""
    m,pins=ctx['metadata'],ctx['pins'];seal=m['gpu-custody-inventory'];obs=ctx['observations']['gpu']
    capsule=Path(capsule).resolve(strict=True);mirror=Path(mirror).resolve(strict=True)
    need(capsule!=mirror and capsule not in mirror.parents and mirror not in capsule.parents, 'independent separate custody copies required')
    need(seal['identical'] is True and seal['acceptance'] is False and seal['excludes']=='custody-inventory.json itself'
         and set(seal['copies'])=={str(capsule),str(mirror)}, 'original full two-copy custody seal required')
    after_transport=m['gpu-inventory-after']
    need(after_transport['returncode']==0 and literal(after_transport['python_source'],'ROOT')==GPU_ROOT
         and literal(after_transport['python_source'],'JOB')==m['gpu-result']['job_id'], 'original second inventory identity differs')
    after=decode(after_transport['stdout'])
    for k in ('files','directories','related_accounting','journal_pins','parent_accounting','terminal_gate'):
        if k=='directories':
            need(type(after[k]) is list and type(obs[k]) is list
                 and len(set(after[k]))==len(after[k]) and len(set(obs[k]))==len(obs[k])
                 and sorted(after[k])==sorted(obs[k]), 'remote directories changed before/after')
        else:need(after[k]==obs[k], 'remote custody changed before/after: '+k)
    need(after['terminal_gate'] is True and not after.get('observation_error'), 'conclusive second terminal observation required')
    gate=ctx['compact_inputs']
    need(gate.capsules==(capsule,mirror),'bound compact capsule identities differ')
    need(sha(gate.supplied['storage-seal.json'])==pins['gpu-custody-inventory'] and
         sha(gate.supplied['storage-after.json'])==pins['gpu-inventory-after'],'original independently pinned selected custody metadata differs')
    recheck_storage(ctx)
    ctx['upstream_rosters']=accepted_rosters(ctx)
    gp=m['gpu-packet'];files=obs['files'];data={}
    need(files['control/inputs/packet.json']['sha256']==pins['gpu-packet'], 'raw GPU inner packet differs')
    for n,h in gp['files'].items():
        need(Path(n).name==n and n.endswith(('.py','.json','.pt')) and pin(h), 'flat original GPU file required')
        need(files['control/inputs/'+n]['sha256']==h, 'raw GPU input differs')
        data[n]=ctx['packet_replicas'].capture('control/inputs/'+n,h,256*1024**2)
    upstream={root+'/'+n:h for root,roster in ctx['upstream_rosters'].items() for n,h in roster['files'].items()}
    for prefix,root,*_ in STAGES:
        outer=m[prefix+'-launch-packet'];result=m[prefix+'-result']
        for n,h in {**outer['new_input_files'],'packet.json':pins[prefix+'-launch-packet'],
                    'submission.intent.json':result['intent_sha256'],'submission.result.json':pins[prefix+'-result']}.items():
            need(upstream.get(root+'/'+n)==h, 'complete original input/journal bytes differ: '+root+'/'+n)
    need(upstream[CPU_ROOT+'/inputs/packet.json']==pins['cpu-packet'] and all(upstream[CPU_ROOT+'/inputs/'+n]==h for n,h in m['cpu-packet']['files'].items()),
         'original CPU inputs no longer match accepted bytes')
    ctx['upstream_pins']=upstream
    ctx['custody_paths']=(capsule,mirror)
    return data


def make_packet(ctx,gpu_data,supplied):
    m,pins=ctx['metadata'],ctx['pins'];f=m['freeze'];gp=m['gpu-packet']
    closure=f['stage_closure']['stages']['finalizer'];prior=f['stage_closure']['stages']['GPUcampaign']
    authority=m['gpu-recovery-authority'];manifest=m['finalizer-source-manifest']
    need(set(gpu_data)==set(gp['files']) and all(sha(v)==gp['files'][n] for n,v in gpu_data.items()), 'unchanged full GPU inputs required')
    need(gp['source_names']==sorted(authority['operational_source_files'])
         and all(gp['files'][n]==h for n,h in authority['operational_source_files'].items())
         and set(gp['source_names'])==set(prior['source_names'])|{PREFLIGHT,HISTORY}
         and all(gp['files'][n]==h for n,h in prior['input_source_files'].items() if n!=SCIENTIFIC_GPU_ENTRY),
         'only the three independently authorized GPU operational source exceptions are permitted')
    data=dict(gpu_data);captured={}
    for origin,h in f['source_pins'].items():
        safe_relative(origin);value=committed(SOURCE_COMMIT,origin)
        need(sha(value)==h,'full frozen67-source Git authority differs');captured[origin]=value
    for n,h in closure['input_source_files'].items():
        found=[v for origin,v in captured.items() if Path(origin).name==n and sha(v)==h]
        need(len(found)==1,'unique frozen source origin required')
        if n in data:
            need(n==SCIENTIFIC_GPU_ENTRY or data[n]==found[0],'unexpected inherited source replacement')
        else:data[n]=found[0]
    need(sha(data[ENTRY])==FROZEN_ENTRY_SHA and closure['entrypoint_sha256']==FROZEN_ENTRY_SHA,
         'preserved original finalizer source differs')
    record=manifest['entry'];entry=committed(manifest['source_commit'],record['repository_path'])
    need(sha(entry)==record['sha256']==ENTRY_SHA,'independently pinned committed finalizer successor differs')
    data[ENTRY]=entry
    need(set(data)==set(gpu_data)|{ENTRY,OLD_GPU_ENTRY},'exact operational finalizer source closure required')
    data.update({'gpu-packet.json':supplied['gpu-packet'],'gpu-acceptance.json':supplied['gpu-acceptance'],
        'gpu-recovery-acceptance.json':supplied['gpu-recovery-acceptance'],
        'gpu-success-baseline.json':supplied['gpu-baseline'],
        'gpu-failure-acceptance.json':supplied['gpu-failure-acceptance'],
        'gpu-failure-baseline.json':supplied['gpu-failure-baseline'],
        'finalizer-clock-adoption.json':supplied['finalizer-clock-adoption']})
    need(len(data)<=256 and sum(map(len,data.values()))<=768*1024**2
         and all(len(v)<=256*1024**2 for v in data.values()),'frozen finalizer input limits exceeded')
    need(literal(entry,'RESOURCES')==RESOURCES and literal(entry,'VERSION')==VERSION
         and literal(entry,'GPU_ACCEPTANCE_KEYS')==GPU_ACCEPTANCE_KEYS,'exact finalizer operational contract differs')
    for n,h in closure['helper_pins'].items():need(sha(data[n])==h,'finalizer helper pin differs')
    bootstrap=literal(data['qualify_llm_intervention_runtime_metadata.py'],'BOOTSTRAP')
    paths={**gp['paths'],'output':ROOT+'/finalization'}
    packet=dict(schema=VERSION,scope='development',files={n:sha(v) for n,v in sorted(data.items())},
        source_names=sorted(set(gp['source_names'])|{ENTRY,OLD_GPU_ENTRY}),paths=paths,resources=RESOURCES,
        cpu_packet_sha256=pins['cpu-packet'],gpu_packet_sha256=pins['gpu-packet'],
        cpu_acceptance_sha256=pins['cpu-acceptance'],gpu_acceptance_sha256=pins['gpu-acceptance'],
        gpu_recovery_acceptance_sha256=pins['gpu-recovery-acceptance'],source_commit=SOURCE_COMMIT,
        operational_source_commit=manifest['source_commit'],freeze_sha256=pins['freeze'],
        namespace_admission_sha256=pins['namespace-admission'],clock_adoption_sha256=pins['finalizer-clock-adoption'])
    need(set(packet)==literal(entry,'PACKET_KEYS') and len(raw(packet))<=256*1024,'exact production packet keys/bound required')
    return packet,data,bootstrap



def clock_admission(metadata, pins):
    """No default/adoption is manufactured by this prospective coordinator."""
    a=metadata['finalizer-clock-adoption'];manifest=metadata['finalizer-source-manifest']
    need(a['schema']=='ace-finalizer-clock-adoption-v1' and a['status']=='adopted'
         and a['scope']=='operational_finalization' and a['remote_root']==ROOT
         and a['freeze_sha256']==pins['freeze'] and a['source_commit']==manifest['source_commit']
         and a['source_manifest_sha256']==pins['finalizer-source-manifest']
         and a['source_review_sha256']==pins['finalizer-source-review']
         and a['entry_sha256']==ENTRY_SHA and a['predecessor_entry_sha256']=='b4d18f4be2320069ffc5f2aed2060efcfb8d49bfcc6e8b00f8099b91598f8745'
         and raw(a['original_deadlines'])==raw(FROZEN_DEADLINES) and a['original_wall_seconds']==1200
         and raw(a['resources'])==raw(RESOURCES) and raw(a['deadlines'])==raw(DEADLINES)
         and a['cpu_acceptance_sha256']==pins['cpu-acceptance']
         and a['gpu_acceptance_sha256']==pins['gpu-acceptance']
         and a['gpu_recovery_acceptance_sha256']==pins['gpu-recovery-acceptance']
         and a['baseline_sha256']==pins['gpu-baseline']
         and a['timing_evidence_sha256']==pins['finalizer-timing-evidence']
         and a['timing_review_sha256']==pins['finalizer-timing-review']
         and a['full_path_timing_supported'] is True
         and type(a['required_findings']) is int and a['required_findings']==0
         and a['scientific_inputs_unchanged'] is True
         and a['scientific_qualified'] is False and a['execution_qualified'] is False,
         'explicit independently reviewed timing-supported clock adoption required')


def budget(base):
    need(base['terminal_jobs']==48 and base['submission_journals']==98 and base['outstanding_reservation']==0,
         'actual released48/98 baseline required')
    cpu,gpu=cost(base['related_accounting'])
    need(len(normalized(base['related_accounting']))==48 and len(base['journal_pins'])==98
         and (cpu,gpu)==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])
         and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_admission_seconds']==cpu+3600,
         'actual complete baseline charge differs')
    need(cpu+3600+8400<=57600 and gpu<=7200, 'bounded finalizer allocation exceeds full caps')
    return dict(closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,preparation_allowance_cpu_seconds=3600,
        candidate_cpu_reservation=8400,candidate_gpu_reservation=0,total_cpu_seconds=cpu+12000,total_gpu_seconds=gpu,
        ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200)


def job_script(ctx,stage,bootstrap):
    command=[ctx['metadata']['cpu-packet']['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
        '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(raw(stage)),'--output',ROOT+'/finalization']
    return ('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_finalization01',
        '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M','#SBATCH --time=01:10:00','#SBATCH --no-requeue',
        '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
        'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL','exec '+shlex.join(command),''])).encode()


@contextlib.contextmanager
def coordinator_locks():
    handles=[]
    try:
        for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            p=BASE/n;need(p==p.resolve(),'lock alias forbidden')
            f=os.fdopen(os.open(p,os.O_RDWR|os.O_NOFOLLOW),'r+');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        for f in reversed(handles):fcntl.flock(f,fcntl.LOCK_UN);f.close()



# Explicit outer storage adapter; never added to frozen scientific child inputs.
COMPACT_ADAPTER_SHA='a5443df94bad0a58e744efe742903867b725348ae72dfeae20c797f461c007e1'
CURC_HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

def compact_adapter():
    import types
    path=Path(__file__).resolve().with_name('compact_inputs.py')
    source=read(path,COMPACT_ADAPTER_SHA)
    mod=types.ModuleType('captured_compact_inputs');mod.__file__=str(path)
    exec(compile(source,str(path),'exec'),vars(mod))
    return mod


def storage_members():
    c=compact_adapter()
    return ('storage-config.json',*c.STORAGE_FILES.values(),'compact_inputs.py')


def start_storage_operations(ctx, root, reference_roots, identity):
    """Register durable caller evidence before the first archive verification."""
    c=compact_adapter();roots=tuple(Path(p).resolve(strict=True) for p in reference_roots)
    def publish(name,value):
        errors=[]
        for target in roots:
            try:write(target/name,raw(value))
            except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
        need(not errors,'storage operation reference publication failed: '+repr(errors))
    identity={**identity,'coordinator_sha256':sha(read(Path(__file__))),
              'source_commit':SOURCE_COMMIT,'reference_roots':[str(p) for p in roots]}
    ctx['storage_operation_evidence']=c.OperationEvidence(root,identity,publish)
    return ctx['storage_operation_evidence']


def configure_storage(ctx,supplied,capsule,mirror):
    c=compact_adapter()
    gate=ctx['storage_operation_evidence'].run('binding',lambda:c.bind(ctx,supplied,predecessor='gpu',capsule=capsule,mirror=mirror,
        expected_roots={**RAW_ROOTS,'namespace.json':REMOTE_REGISTRY,'original-namespace.json':ORIGINAL_REMOTE_REGISTRY},committed=committed,helper_path=CURC_HELPER))
    ctx['storage_supplied']=supplied
    return gate


def recheck_storage(ctx):
    compact_adapter().recheck_capsules(ctx['compact_inputs'])
    ctx['storage_operation_evidence'].run('packet-replica-recheck',ctx['packet_replicas'].recheck)


def configure_packet_replicas(ctx,roots):
    """Bind explicit original packet copies after the unchanged full storage gate."""
    gate=ctx['compact_inputs'];pins=ctx['pins'];c=compact_adapter()
    replicas=ctx['storage_operation_evidence'].run('packet-replica-binding',lambda:c.PacketReplicas(
        roots=roots,outer_sha256=pins['gpu-launch-packet'],inner_sha256=pins['gpu-packet'],
        after_raw=ctx['storage_supplied']['storage-after.json'],after_sha256=pins['gpu-inventory-after'],
        control_root=GPU_ROOT,protected_roots=[*gate.capsules,gate.staging]),details={'roots':list(roots)})
    ctx['packet_replicas']=replicas
    return replicas


def storage_packet_files(ctx):
    return {**ctx['storage_supplied'],
        'storage-preparation-intent.json':raw(ctx['storage_operation_evidence'].intent_reference),
        'storage-preparation-reference.json':raw(ctx['storage_operation_evidence'].last_reference),
        'compact_inputs.py':read(Path(__file__).resolve().with_name('compact_inputs.py'),COMPACT_ADAPTER_SHA)}


def storage_packet_binding(ctx):
    gate=ctx['compact_inputs']
    return dict(mode='curc_projects_full_backup_with_compact_local_evidence',
        predecessor_remote_custody_sha256=gate.receipt_sha256,
        storage_config_sha256=sha(ctx['storage_supplied']['storage-config.json']),
        backend_sha256=gate.helper_sha256,compact_adapter_sha256=COMPACT_ADAPTER_SHA,
        original_ancestor_references=gate.ancestry,complete_local_raw_copy=False,
        input_replica_binding=ctx['packet_replicas'].binding())


def configure_packet_storage(ctx,local,pins,packet):
    c=compact_adapter();supplied=c.stored_config(local,pins)
    gate=configure_storage(ctx,supplied,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])
    configure_packet_replicas(ctx,packet['storage_binding']['input_replica_binding']['roots'])
    need(packet['storage_binding']==storage_packet_binding(ctx),'independent compact storage packet binding differs')
    return gate


def exact_inventory(files,directories):
    need(type(files) is dict and files and type(directories) is list,'full inventory maps required')
    need(all(type(n) is str for n in directories) and len(set(directories))==len(directories),'unique inventory directories required')
    for n,v in files.items():
        safe_relative(n);need(type(v) is dict and set(v)=={'bytes','sha256'} and type(v['bytes']) is int
             and v['bytes']>=0 and pin(v['sha256']),'exact file identity required')
    for n in directories:safe_relative(n)
    need(not set(files)&set(directories),'file/directory overlap')
    for n in list(files)+directories:
        need(all(str(p) in directories for p in Path(n).parents if str(p)!='.'),'inventory missing directory')
    return dict(files=files,directories=sorted(directories))

def failed_gpu_history(m,pins):
    """Original nonzero collector is preserved; independent reconciliation is authority."""
    a=m['gpu-failure-acceptance'];base=m['gpu-failure-baseline'];prior=m['cpu-baseline']
    d=m['gpu-failure-disposition'];seal=m['gpu-failure-seal'];p=m['gpu-failure-packet'];r=m['gpu-failure-result']
    need(pins['gpu-failure-packet']==FAILED_GPU_PACKET_SHA and pins['gpu-failure-seal']==FAILED_GPU_SEAL_SHA
         and pins['gpu-failure-disposition']==FAILED_GPU_DISPOSITION_SHA,'original failed GPU capsule differs')
    authority=m['gpu-recovery-authority']
    required=dict(schema='ace-scientific-campaign-gpu-failure-acceptance-v1',status='accepted_failed_prepolicy_attempt',
        job_id=FAILED_GPU_JOB,required_findings=0,acceptance=True,complete_CURC_backup_verified_from_saved_records=True,
        all_selected_local_files_verified_in_both_copies=True,complete_local_raw_copy=False,
        new_policy_attempts=0,new_scientific_responses=0,new_training_updates=0,new_world_realizations=0,
        original_collector_returncode=1,retroactive_collector_success=False,originals_changed=False,
        outstanding_reservation=0,related_terminal_jobs=47,submission_journals=96,
        scheduler_state='FAILED',scheduler_exit_code='1:0',source_commit=SOURCE_COMMIT)
    need(set(required)<=set(a) and raw({k:a[k] for k in required})==raw(required),
         'independent accepted failed prepolicy attempt/no-work reconciliation required')
    need(authority['failed_gpu01_acceptance_sha256']==pins['gpu-failure-acceptance'], 'failure acceptance pin differs')
    for name,key in {'original_disposition':'gpu-failure-disposition','original_failure_seal':'gpu-failure-seal',
        'original_submission_result':'gpu-failure-result','outer_packet':'gpu-failure-packet',
        'predecessor_acceptance':'cpu-acceptance','predecessor_baseline':'cpu-baseline'}.items():
        need(a['input_pins'][name]['sha256']==pins[key],'actual reconciliation original pin differs: '+name)
    need(a['input_pins']['submitted_entry']['sha256']==ORIGINAL_GPU_ENTRY_SHA
         and all(pin(a[k]) for k in ('no_policy_work_proof_sha256','resource_reconciliation_sha256','source_boundary_sha256')),
         'actual reconciliation source/no-work/resource evidence pins required')
    need(d['status']=='terminal_custody_with_binding_disagreements' and d['job_id']==FAILED_GPU_JOB
         and d['remote_root']==OLDROOT and d['packet_sha256']==pins['gpu-failure-packet']
         and d['submission_result_sha256']==pins['gpu-failure-result'] and d['custody_bytes_verified'] is True
         and d['current_lineage_verified'] is False and d['input_pin_mismatches']==[]
         and d['errors']==['actual/requested GPU identity/resource/no-requeue/wall contract differs or unavailable']
         and d['allocation']['resource_contract_matched'] is False,'retain sole original collector disagreement verbatim')
    need(p['schema']=='ace-scientific-campaign-gpu-launch-v1' and p['source_commit']==SOURCE_COMMIT
         and p['remote_root']==OLDROOT and p['resources']==GPU_RESOURCES and p['limits']==GPU_LIMITS
         and p['baseline_sha256']==pins['cpu-baseline'] and p['cpu_acceptance_sha256']==pins['cpu-acceptance']
         and p['cpu_packet_sha256']==pins['cpu-packet'],'original failed GPU submission lineage differs')
    need(r['job_id']==FAILED_GPU_JOB and r['returncode']==0 and r['ambiguous'] is False and pin(r['intent_sha256']),
         'original once-submitted failed job differs')
    observations=[]
    for key in ('gpu-failure-inventory','gpu-failure-inventory-after'):
        t=m[key];need(t['returncode']==0 and literal(t['python_source'],'ROOT')==OLDROOT
             and literal(t['python_source'],'JOB')==FAILED_GPU_JOB,'complete original observation required')
        o=decode(t['stdout']);need(o['terminal_gate'] is True and o['root_present'] is True
             and not o.get('observation_error'),'failed GPU complete observation required')
        observations.append(o)
    before,after=observations
    need(before['original_hardlink_groups']==after['original_hardlink_groups'], 'original failure alias metadata changed')
    full=exact_inventory(before['files'],before['directories'])
    need(full==exact_inventory(after['files'],after['directories'])
         and full==exact_inventory(d['remote_files'],d['remote_directories']), 'failed original before/after/full custody differs')
    copies=seal['copies'];need(seal['identical'] is True and len(copies)==2 and len(set(copies))==2,'two original failed seals required')
    views=list(copies.values());need(views[0]==views[1],'original failed seal copies differ')
    for n,key in {'disposition.json':'gpu-failure-disposition','inventory_transport.json':'gpu-failure-inventory',
                  'inventory_after_transport.json':'gpu-failure-inventory-after'}.items():
        need(views[0]['files'][n]['sha256']==pins[key],'original failed seal metadata differs')
    for n,h in {**p['new_input_files'],'packet.json':pins['gpu-failure-packet'],
                'submission.intent.json':r['intent_sha256'],'submission.result.json':pins['gpu-failure-result']}.items():
        need(full['files']['launch/'+safe_relative(n)]['sha256']==h,'original failed source/input/journal differs')
    old=exact_inventory({n[7:]:v for n,v in full['files'].items() if n.startswith('launch/')},
                        [n[7:] for n in full['directories'] if n.startswith('launch/')])
    need(not any(n=='gpu' or n.startswith('gpu/') for n in list(old['files'])+old['directories']),
         'old output/phase already exists; no retry')
    # Failure observation must retain the exact accepted CPU tree, including empty directories.
    original=m['cpu-inventory'];cpu_before=decode(original['stdout'])
    for new_label,old_label in [('cpu','launch'),('training','training'),('token','cpu'),
                                 ('failed-token01','failed-token01'),('common','common'),('failed01','failed01')]:
        want={n[len(old_label)+1:]:v for n,v in cpu_before['files'].items() if n.startswith(old_label+'/')}
        got={n[len(new_label)+1:]:v for n,v in full['files'].items() if n.startswith(new_label+'/')}
        wd=sorted(n[len(old_label)+1:] for n in cpu_before['directories'] if n.startswith(old_label+'/'))
        gd=sorted(n[len(new_label)+1:] for n in full['directories'] if n.startswith(new_label+'/'))
        need(want and want==got and wd==gd,'failed GPU changed accepted ancestor: '+new_label)
    need(base['terminal_jobs']==47 and base['submission_journals']==96 and base['outstanding_reservation']==0
         and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
         and base['input_pins']['predecessor']==pins['cpu-baseline']
         and base['input_pins']['acceptance']==pins['gpu-failure-acceptance']
         and base['input_pins']['disposition']==pins['gpu-failure-disposition'],'actual failed47/96 baseline required')
    rows=base['related_accounting'];oldrows=[v for v in rows if v[0]!=FAILED_GPU_JOB];current=[v for v in rows if v[0]==FAILED_GPU_JOB]
    need(len(oldrows)==46 and len(current)==1 and normalized(oldrows)==normalized(prior['related_accounting'])
         and normalized(rows)==normalized(before['related_accounting'])==normalized(after['related_accounting'])
         and normalized(current)==normalized([before['parent_accounting']])==normalized([after['parent_accounting']]),
         'exact failed job append differs')
    row=normalized(current)[0];need(row[2:7]==['FAILED','1:0',4,245,'ucb736_asc2'] and cost(current)==(980,245),
         'observed failed GPU allocation differs')
    expected={**prior['journal_pins'],OLDROOT+'/submission.intent.json':r['intent_sha256'],
              OLDROOT+'/submission.result.json':pins['gpu-failure-result']}
    need(len(expected)==96 and expected==base['journal_pins']==before['journal_pins']==after['journal_pins'],
         'complete96 failed-history journal pins differ')
    oldcost=cost(prior['related_accounting']);newcost=cost(rows)
    need(newcost==(a['closed_cpu_seconds'],a['closed_gpu_slice_seconds'])
         and a['cpu_admission_seconds']==newcost[0]+3600, 'reconciliation actual costs differ from baseline')
    need(newcost==(oldcost[0]+980,oldcost[1]+245)==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])
         and base['cpu_admission_seconds']==newcost[0]+3600,'actual failed charge extension differs')
    need(newcost[0]+17520<=57600 and newcost[1]+2880<=7200,'remaining frozen caps exceed actual budget')
    return old

def recovery_authority(supplied,m,pins,observations):
    """Join the actual recovery, unchanged17-key acceptance, and full historical custody."""
    a=m['gpu-recovery-authority'];gp=m['gpu-packet'];outer=m['gpu-launch-packet']
    c=m['cpu-history-expected']['execution_context'];g=m['gpu-acceptance'];s=m['gpu-recovery-acceptance']
    need(pins['gpu-recovery-authority']==RECOVERY_AUTHORITY_SHA
         and a['schema']=='ace-gpu-cpu-history-recovery-authority-v1' and a['status']=='operationally_adopted'
         and a['historical_postcustody_required'] is a['unchanged_scientific_inputs'] is True
         and a['control_root']==GPU_ROOT and a['failed_gpu01_root']==GPU_OUTPUT_ROOT
         and a['allowed_new_output']==GPU_OUTPUT_ROOT+'/gpu'
         and a['original_source_commit']==SOURCE_COMMIT and a['original_entry_sha256']==ORIGINAL_GPU_ENTRY_SHA
         and a['original_gpu01_packet_sha256']==pins['gpu-failure-packet']
         and a['original_gpu01_job_id']==FAILED_GPU_JOB
         and a['cpu_acceptance_sha256']==pins['cpu-acceptance']
         and a['namespace_sha256']==pins['namespace-claim'] and a['original_namespace_sha256']==ORIGINAL_CLAIM_SHA,
         'exact adopted operational authority required')
    need(a['baseline']==c['baseline']==dict(sha256=pins['gpu-failure-baseline'],terminal_jobs=47,submission_journals=96)
         and c['control_root']==GPU_ROOT and c['gpu_output']==GPU_OUTPUT_ROOT+'/gpu'
         and c['failed_gpu01_root']==GPU_OUTPUT_ROOT
         and c['cpu_roots']==dict(cpu_output=CPU_ROOT+'/cpu',campaign=CPU_ROOT+'/cpu/worker/artifacts/campaign'),
         'original failed baseline/context roots differ')
    need(gp['recovery_files']==RECOVERY_FILES
         and all(gp['files'][n]==pins[n[:-5]] for n in RECOVERY_FILES.values())
         and gp['source_names']==sorted(a['operational_source_files']) and len(gp['source_names'])==56
         and {n:gp['files'][n] for n in gp['source_names']}==a['operational_source_files']
         and c['source_files_sha256']==digest(a['operational_source_files'])
         and c['model_files_sha256']==digest(m['freeze']['model_files'])
         and c['resources_sha256']==digest(GPU_RESOURCES) and c['limits_sha256']==digest(GPU_LIMITS),
         'complete recovery source/model/resource/preflight context differs')
    for name in RECOVERY_METADATA:
        need(outer['new_input_files'][metadata_name(name)]==pins[name], 'actual GPU02 captured recovery metadata differs: '+name)
    manifest=m['operational-source-manifest']
    need(set(manifest)=={'schema','source_commit','files'} and manifest['schema']=='ace-gpu02-operational-sources-v1'
         and set(manifest['files'])=={SCIENTIFIC_GPU_ENTRY,PREFLIGHT,HISTORY}
         and outer['operational_source_commit']==manifest['source_commit'], 'three original GPU02 operational sources required')
    captured={}
    for n,v in manifest['files'].items():
        need(set(v)=={'repository_path','sha256'} and v['sha256']==gp['files'][n], 'original operational source pin differs')
        safe_relative(v['repository_path']);captured[n]=committed(manifest['source_commit'],v['repository_path'])
        need(sha(captured[n])==v['sha256'],'committed original GPU02 operational source differs')
    lineage=c['lineage']
    need(lineage['cpu_acceptance_sha256']==pins['cpu-acceptance']
         and lineage['failed_gpu01_acceptance_sha256']==pins['gpu-failure-acceptance']
         and lineage['recovery_authority_sha256']==pins['gpu-recovery-authority']
         and lineage['checker_sha256']==gp['files'][PREFLIGHT]
         and lineage['source_sha256']==a['historical_verifier_sha256']
         and lineage['namespace_sha256']==pins['namespace-claim'] and lineage['original_namespace_sha256']==ORIGINAL_CLAIM_SHA
         and c['proof_pins']==a['historical_proof_pins']==dict(custody_sha256=pins['gpu-failure-seal'],
             before_observation_sha256=pins['gpu-failure-inventory'],after_observation_sha256=pins['gpu-failure-inventory-after'],
             read_set_artifact_sha256=pins['cpu-history-readset']), 'original preflight provenance differs')
    # This is the existing pure supplied-bytes checker. No scientific producer is imported.
    import types
    checker=types.ModuleType('captured_finalizer_preflight')
    exec(compile(captured[PREFLIGHT],PREFLIGHT,'exec'),vars(checker))
    scoped=checker.validate_receipt(supplied['cpu-history-preflight'],pins['cpu-history-preflight'],
                                    supplied['cpu-history-expected'],pins['cpu-history-expected'])
    need(scoped['full_external_postcustody_required'] is True,'original preflight boundary changed')
    old=observations['failed-gpu-root'];obs=observations['gpu'];files=obs['files']
    permitted=lambda n:n=='gpu' or n.startswith('gpu/')
    current=exact_inventory({n[7:]:v for n,v in files.items() if n.startswith('launch/')},
                           [n[7:] for n in obs['directories'] if n.startswith('launch/')])
    preserved=dict(files={n:v for n,v in current['files'].items() if not permitted(n)},
                   directories=[n for n in current['directories'] if not permitted(n)])
    need(preserved==old,'failed GPU01 root may gain only the explicitly authorized gpu output')
    marker='controller/historical-postcustody-required.json'
    required=dict(schema='ace-scientific-campaign-gpu-recovery-acceptance-v1',status='accepted',scope='development',
        required_findings=0,acceptance=True,scientific_qualified=False,historical_postcustody_verified=True,originals_preserved=True,
        job_id=g['job_id'],gpu_acceptance_sha256=pins['gpu-acceptance'],gpu_packet_sha256=pins['gpu-packet'],
        gpu_launch_packet_sha256=pins['gpu-launch-packet'],cpu_acceptance_sha256=pins['cpu-acceptance'],
        authority_sha256=pins['gpu-recovery-authority'],preflight_context_sha256=pins['cpu-history-expected'],
        preflight_receipt_sha256=pins['cpu-history-preflight'],readset_sha256=pins['cpu-history-readset'],
        failure_acceptance_sha256=pins['gpu-failure-acceptance'],failure_baseline_sha256=pins['gpu-failure-baseline'],
        baseline_sha256=pins['gpu-baseline'],source_files_sha256=c['source_files_sha256'],
        gpu_terminal_sha256=g['terminal_pins']['terminal.json'],phase_terminal_sha256=g['terminal_pins']['controller/phase/terminal.json'],
        postcustody_marker_sha256=g['gpu_output_files'].get(marker),
        before_observation_sha256=pins['gpu-inventory'],after_observation_sha256=pins['gpu-inventory-after'],
        custody_sha256=pins['gpu-custody-inventory'],control_root=GPU_ROOT,output_root=GPU_OUTPUT_ROOT+'/gpu',
        raw_root_labels=dict(control=GPU_ROOT,launch=GPU_OUTPUT_ROOT),cpu_campaign_root=g['campaign_root'],
        original_source_commit=SOURCE_COMMIT,original_gpu_entry_sha256=ORIGINAL_GPU_ENTRY_SHA,
        operational_source_commit=manifest['source_commit'],failed_gpu01_job_id=FAILED_GPU_JOB,
        archive_receipt_sha256=m['gpu-disposition']['remote_custody_receipt_sha256'],
        failed_archive_receipt_sha256=m['gpu-failure-disposition']['remote_custody_receipt_sha256'],
        evidence_pins={n:pins[n] for n in ('gpu-review','gpu-disposition','gpu-inventory','gpu-inventory-after',
            'gpu-custody-inventory','gpu-recovery-review','preflight-review','operational-source-manifest',
            'operational-source-review','gpu-failure-disposition','gpu-failure-seal',
            'gpu-failure-inventory','gpu-failure-inventory-after')})
    need(type(s) is dict and set(required)<=set(s) and raw({k:s[k] for k in required})==raw(required)
         and pin(required['postcustody_marker_sha256'])
         and files['launch/gpu/'+marker]['sha256']==required['postcustody_marker_sha256'],
         'independent actual GPU02 full-postcustody acceptance and exact evidence pins required')
    need(pin(s['archive_receipt_sha256']) and pin(s['failed_archive_receipt_sha256'])
         and list(m['gpu-failure-seal']['copies'].values())[0]['files']['remote-custody.json']['sha256']==s['failed_archive_receipt_sha256'],
         'original/current full archive receipt pins required')
    fm=m['finalizer-source-manifest']
    need(type(fm) is dict and set(fm)=={'schema','source_commit','entry'}
         and fm['schema']=='ace-finalizer-gpu02-operational-source-v1'
         and type(fm['source_commit']) is str and re.fullmatch('[0-9a-f]{40}',fm['source_commit'])
         and set(fm['entry'])=={'repository_path','sha256'} and fm['entry']['sha256']==ENTRY_SHA,
         'independent finalizer source manifest required')
    safe_relative(fm['entry']['repository_path'])
    need(sha(committed(fm['source_commit'],fm['entry']['repository_path']))==ENTRY_SHA,
         'independent committed finalizer successor required')
    return dict(authority_sha256=pins['gpu-recovery-authority'],acceptance_sha256=pins['gpu-recovery-acceptance'],
        failed_job_id=FAILED_GPU_JOB,failed_root=GPU_OUTPUT_ROOT,failure_acceptance_sha256=pins['gpu-failure-acceptance'],
        failure_baseline_sha256=pins['gpu-failure-baseline'],failure_seal_sha256=pins['gpu-failure-seal'],
        control_root=GPU_ROOT,output_root=GPU_OUTPUT_ROOT+'/gpu',
        preflight_receipt_sha256=pins['cpu-history-preflight'],execution_context_sha256=m['cpu-history-expected']['execution_context_sha256'],
        full_historical_postcustody_verified=True,original_collector_status=m['gpu-failure-disposition']['status'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--gpu-terminal-custody',required=True,help='original complete repository terminal capsule')
    ap.add_argument('--gpu-terminal-mirror',required=True,help='original complete independent external terminal capsule')
    ap.add_argument('--gpu-packet-replica',required=True,help='original GPU02 repository packet root')
    ap.add_argument('--gpu-packet-replica-mirror',required=True,help='original GPU02 independent external packet root')
    ap.add_argument('--storage-config',required=True)
    ap.add_argument('--storage-config-sha256',required=True)
    args=ap.parse_args()
    with coordinator_locks():
        supplied={n:read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256'),metadata_maximum(n)) for n in METADATA}
        ctx=authenticate(supplied)
        need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'], 'existing recovery claim changed')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        storage=compact_adapter().capture_config(args.storage_config,args.storage_config_sha256)
        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'exclusive new launch destinations required before storage use')
        preparations=(EXT.with_name(EXT.name+'-storage-build-preparation'),LOCAL.parent/'storage-build-preparation01')
        need(not any(os.path.lexists(p) for p in preparations),'existing storage preparation; never retry')
        need(LOCAL.parent==LOCAL.parent.resolve(strict=False),'repository stage parent alias forbidden')
        LOCAL.parent.mkdir(mode=0o700,exist_ok=True)
        need(LOCAL.parent.is_dir() and not LOCAL.parent.is_symlink()
             and LOCAL.parent==LOCAL.parent.resolve(strict=True),'ordinary repository stage parent required')
        for path in preparations:path.mkdir(mode=0o700,exist_ok=False)
        start_storage_operations(ctx,preparations[0]/'records',preparations,
            dict(phase='build',storage_config_sha256=args.storage_config_sha256,
                 intended_packet_roots=[str(EXT),str(LOCAL)]))
        configure_storage(ctx,storage,args.gpu_terminal_custody,args.gpu_terminal_mirror)
        configure_packet_replicas(ctx,[args.gpu_packet_replica,args.gpu_packet_replica_mirror])
        gpu_data=admit_gpu_bytes(ctx,args.gpu_terminal_custody,args.gpu_terminal_mirror)
        stage,data,bootstrap=make_packet(ctx,gpu_data,supplied);stage_raw=raw(stage)
        files={'inputs/'+n:v for n,v in data.items()};files['inputs/packet.json']=stage_raw
        for n,v in supplied.items():files[metadata_name(n)]=v
        files.update(storage_packet_files(ctx))
        for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        files['job.sbatch']=job_script(ctx,stage,bootstrap)
        outer=dict(schema=OUTER_VERSION,source_commit=SOURCE_COMMIT,remote_root=ROOT,resources=RESOURCES,deadlines=DEADLINES,
            clock_adoption_sha256=ctx['pins']['finalizer-clock-adoption'],
            finalizer_packet_sha256=sha(stage_raw),gpu_packet_sha256=ctx['pins']['gpu-packet'],gpu_acceptance_sha256=ctx['pins']['gpu-acceptance'],
            gpu_job_id=ctx['metadata']['gpu-acceptance']['job_id'],token_recovery=ctx['token_recovery'],
            gpu_recovery=ctx['gpu_recovery'],operational_source_commit=stage['operational_source_commit'],
            gpu_recovery_acceptance_sha256=ctx['pins']['gpu-recovery-acceptance'],
            failed_token_root=FAILED_TOKEN_ROOT,failed_token_job_id=FAILED_TOKEN_JOB,baseline_sha256=ctx['pins']['gpu-baseline'],
            operational_amendment_sha256=ctx['pins']['operational-amendment'],namespace_claim_sha256=ctx['pins']['namespace-claim'],
            new_input_files={n:sha(v) for n,v in sorted(files.items())},original_input_pins=ctx['upstream_pins'],
            upstream_rosters=ctx['upstream_rosters'],paths=stage['paths'],gpu_terminal_custody=str(ctx['custody_paths'][0]),
            gpu_terminal_mirror=str(ctx['custody_paths'][1]),storage_binding=storage_packet_binding(ctx),prior_jobs=48,prior_journals=98,worlds=24,cells=192,comparisons=28,
            policy_ids=list(POLICIES),admission=budget(ctx['metadata']['gpu-baseline']),submitted=False,
            new_training_updates=0,new_responses=0,new_worlds=0,scientific_qualified=False,execution_qualified=False,
            sizing='Operational CPU reporting cap4200s2CPU7.5GiB requires separately adopted measured clock support. Private saved-response reporting only in allocated worker. '
                   'Future terminal accounting must preserve all48 prior rows/98journals and extend49/100 with actual charges, including failures.')
        files['packet.json']=raw(outer)
        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)), 'exclusive new launch destinations required')
        created=[]
        try:
            for root in (EXT,LOCAL):
                root.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(root);(root/'inputs').mkdir(mode=0o700)
                for n,v in files.items():write(root/n,v)
            for n,v in files.items():need(read(LOCAL/n,maximum=256*1024**2)==read(EXT/n,maximum=256*1024**2)==v, 'two-copy packet differs')
            recheck_storage(ctx)
            need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'], 'claim changed during build')
            read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
            print(raw(dict(storage_operations=ctx['storage_operation_evidence'].last_reference,packet_sha256=sha(files['packet.json']),finalizer_packet_sha256=sha(stage_raw),submitted=False,scientific_qualified=False)).decode())
        except BaseException as exc:
            for root in created:
                try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
                except BaseException:pass
            raise

if __name__=='__main__':main()
