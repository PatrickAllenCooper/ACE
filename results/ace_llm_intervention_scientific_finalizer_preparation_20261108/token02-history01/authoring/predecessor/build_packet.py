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
GPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'
OLD_GPU_ENTRY='qualify_llm_intervention_campaign_gpu.py'
ENTRY_SHA='28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a'
VERSION='ace-scientific-campaign-finalizer-cpu-v1'
OUTER_VERSION='ace-scientific-finalization-launch-v1'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
 memory_mib=7680,wall_seconds=1200,gpus=0,requeue=False)
DEADLINES=dict(startup=120,work=1140,cleanup=1170,hard=1190)
GPU_RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
GPU_LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)
RAW_ROOTS=dict(launch=GPU_ROOT,cpu=CPU_ROOT,training=TRAIN_ROOT,token=TOKEN_ROOT,common=COLLECTION_ROOT,
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
STAGES=(('collection',COLLECTION_ROOT,42,86,2,0,900),('token',TOKEN_ROOT,43,88,3,0,600),
 ('training',TRAIN_ROOT,44,90,4,1,2700),('cpu',CPU_ROOT,45,92,2,0,6000),('gpu',GPU_ROOT,46,94,4,1,2880))
TEXT_METADATA={'freeze-review','operational-adoption-review','site-readiness','cpu-review','gpu-review'}
def metadata_name(n):return n+('.md' if n in TEXT_METADATA else '.json')
METADATA=tuple(FIXED)+('namespace-claim','site-readiness','cpu-review','cpu-packet','cpu-phase-terminal','gpu-review','gpu-packet','gpu-inventory-after','gpu-custody-inventory')+tuple(
 prefix+'-'+kind for prefix,*_ in STAGES for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
CPU_ACCEPTANCE_KEYS={'schema','status','cpu_packet_sha256','job_id','allocation','cpu_output','campaign_root',
 'cpu_output_files','cpu_output_excluded_directory','campaign_files','bundle_path','bundle_sha256',
 'phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}
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

def history(m,pins):
    previous=m['collection-failure-baseline'];previous_pin=pins['collection-failure-baseline'];jobs=[];observations={}
    need(previous['terminal_jobs']==41 and previous['submission_journals']==84,'accepted failed41/84 boundary required')
    for prefix,root,count,journals,cpus,gpus,wall in STAGES:
        a,p,r,d,t,base=(m[prefix+'-'+kind] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
        job=r['job_id'];need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job not in jobs+['35407428'],'distinct actual job required')
        jobs.append(job)
        need(r['returncode']==0 and r['ambiguous'] is False and pin(r['intent_sha256']) and a['job_id']==job,'accepted original submission differs')
        need(p['remote_root']==root and p['source_commit']==SOURCE_COMMIT and p['baseline_sha256']==previous_pin,'original stage source/root/predecessor differs')
        schemas=dict(collection='ace-scientific-collection-launch-v1',token='ace-scientific-tokenization-launch-v1',
                     training='ace-scientific-training-gpu-launch-v1',cpu='ace-scientific-cpu-campaign-launch-v1',gpu='ace-scientific-campaign-gpu-launch-v1')
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
            safe_relative(n);need(pin(h) and obs['files']['launch/'+n]['sha256']==h,'original stage inputs/journals differ')
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
    need(s['entry']==ENTRY and s['entry_sha256']==ENTRY_SHA and s['remote_root']==ROOT and s['deadlines']==DEADLINES
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
    observations=history(m,pins)
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
    # The GPU producer captures every predecessor metadata argument byte unchanged.
    for name in tuple(FIXED)+('namespace-claim','cpu-review','cpu-packet','cpu-phase-terminal')+tuple(
        prefix+'-'+kind for prefix,*_ in STAGES[:-1] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline')):
        need(m['gpu-launch-packet']['new_input_files'][metadata_name(name)]==pins[name], 'original GPU lineage capture differs: '+name)
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
         and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,'full remaining frozen caps exceed budget')
    return dict(metadata=m,pins=pins,observations=observations)

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
    need(gp['schema']=='ace-scientific-paired-campaign-gpu-v1' and gp['scope']=='development'
         and gp['resources']==GPU_RESOURCES and gp['limits']==GPU_LIMITS
         and gp['freeze_sha256']==pins['freeze'] and gp['namespace_admission_sha256']==pins['namespace-admission']
         and gp['cpu_packet_sha256']==pins['cpu-packet'] and gp['cpu_acceptance_sha256']==pins['cpu-acceptance'],
         'original frozen scientific GPU input contract differs')
    need(m['gpu-launch-packet']['new_input_files']['inputs/packet.json']==pins['gpu-packet']
         and m['gpu-launch-packet']['gpu_packet_sha256']==pins['gpu-packet'], 'original GPU inner packet differs')
    paths=dict(cpu_inputs=CPU_ROOT+'/inputs',cpu_output=CPU_ROOT+'/cpu',campaign=CPU_ROOT+'/cpu/worker/artifacts/campaign',
               gpu_inputs=GPU_ROOT+'/inputs',gpu_output=GPU_ROOT+'/gpu')
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
        need(after[k]==obs[k], 'remote custody changed before/after: '+k)
    need(after['terminal_gate'] is True and not after.get('observation_error'), 'conclusive second terminal observation required')
    copies=[]
    for root in (capsule,mirror):
        value=tree(root);read(root/'custody-inventory.json',pins['gpu-custody-inventory'])
        del value['files']['custody-inventory.json']
        need(value==seal['copies'][str(root)], 'full original capsule seal mismatch')
        for name,key in [('disposition.json','gpu-disposition'),('inventory_transport.json','gpu-inventory'),
                         ('inventory_after_transport.json','gpu-inventory-after')]:read(root/name,pins[key])
        need(tree(root/'remote')==dict(files=obs['files'],directories=sorted(obs['directories'])), 'complete raw files/directories differ')
        copies.append(value)
    need(copies[0]==copies[1], 'complete independent capsule bytes differ')
    for n in copies[0]['files']:
        need(not os.path.samestat((capsule/n).stat(),(mirror/n).stat()), 'custody mirrors alias the same inode')
    ctx['upstream_rosters']=accepted_rosters(ctx)
    gp=m['gpu-packet'];files=obs['files'];data={}
    need(files['launch/inputs/packet.json']['sha256']==pins['gpu-packet'], 'raw GPU inner packet differs')
    for n,h in gp['files'].items():
        need(Path(n).name==n and n.endswith(('.py','.json','.pt')) and pin(h), 'flat original GPU file required')
        need(files['launch/inputs/'+n]['sha256']==h, 'raw GPU input differs')
        data[n]=read(capsule/'remote/launch/inputs'/n,h,256*1024**2)
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
    need(set(gpu_data)==set(gp['files']) and all(sha(v)==gp['files'][n] for n,v in gpu_data.items()), 'unchanged full GPU inputs required')
    need(gp['source_names']==prior['source_names'] and all(gp['files'][n]==h for n,h in prior['input_source_files'].items()), 'frozen GPU closure differs')
    data=dict(gpu_data);captured={}
    for origin,h in f['source_pins'].items():
        safe_relative(origin);value=committed(SOURCE_COMMIT,origin);need(sha(value)==h, 'full frozen67-source Git authority differs');captured[origin]=value
    for n,h in closure['input_source_files'].items():
        found=[v for origin,v in captured.items() if Path(origin).name==n and sha(v)==h]
        need(len(found)==1, 'unique frozen source origin required')
        need(n not in data or data[n]==found[0], 'finalizer must not replace inherited source bytes');data[n]=found[0]
    need(set(data)==set(gpu_data)|{ENTRY,OLD_GPU_ENTRY} and closure['source_names']==sorted(set(gp['source_names'])|{ENTRY,OLD_GPU_ENTRY}),
         'exact scientific finalizer source closure required')
    data.update({'gpu-packet.json':supplied['gpu-packet'],'gpu-acceptance.json':supplied['gpu-acceptance']})
    need(len(data)<=256 and sum(map(len,data.values()))<=768*1024**2 and all(len(v)<=256*1024**2 for v in data.values()), 'frozen finalizer input limits exceeded')
    entry=data[ENTRY]
    need(sha(entry)==ENTRY_SHA and literal(entry,'RESOURCES')==RESOURCES and literal(entry,'VERSION')==VERSION
         and literal(entry,'GPU_ACCEPTANCE_KEYS')==GPU_ACCEPTANCE_KEYS and closure['entrypoint_sha256']==ENTRY_SHA,
         'exact production finalizer contract differs')
    for n,h in closure['helper_pins'].items():need(sha(data[n])==h, 'finalizer helper pin differs')
    bootstrap=literal(data['qualify_llm_intervention_runtime_metadata.py'],'BOOTSTRAP')
    paths={**gp['paths'],'output':ROOT+'/finalization'}
    packet=dict(schema=VERSION,scope='development',files={n:sha(v) for n,v in sorted(data.items())},source_names=closure['source_names'],
        paths=paths,resources=RESOURCES,cpu_packet_sha256=pins['cpu-packet'],gpu_packet_sha256=pins['gpu-packet'],
        cpu_acceptance_sha256=pins['cpu-acceptance'],gpu_acceptance_sha256=pins['gpu-acceptance'],source_commit=SOURCE_COMMIT,
        freeze_sha256=pins['freeze'],namespace_admission_sha256=pins['namespace-admission'])
    need(set(packet)==literal(entry,'PACKET_KEYS') and len(raw(packet))<=256*1024, 'exact production packet keys/bound required')
    return packet,data,bootstrap


def budget(base):
    need(base['terminal_jobs']==46 and base['submission_journals']==94 and base['outstanding_reservation']==0,
         'actual released46/94 baseline required')
    cpu,gpu=cost(base['related_accounting'])
    need(len(normalized(base['related_accounting']))==46 and len(base['journal_pins'])==94
         and (cpu,gpu)==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])
         and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_admission_seconds']==cpu+3600,
         'actual complete baseline charge differs')
    need(cpu+3600+2400<=57600 and gpu<=7200, 'bounded finalizer allocation exceeds full caps')
    return dict(closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,preparation_allowance_cpu_seconds=3600,
        candidate_cpu_reservation=2400,candidate_gpu_reservation=0,total_cpu_seconds=cpu+6000,total_gpu_seconds=gpu,
        ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200)


def job_script(ctx,stage,bootstrap):
    command=[ctx['metadata']['cpu-packet']['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
        '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(raw(stage)),'--output',ROOT+'/finalization']
    return ('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_finalization01',
        '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M','#SBATCH --time=00:20:00','#SBATCH --no-requeue',
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--gpu-terminal-custody',required=True,help='original complete repository terminal capsule')
    ap.add_argument('--gpu-terminal-mirror',required=True,help='original complete independent external terminal capsule')
    args=ap.parse_args()
    with coordinator_locks():
        supplied={n:read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256'),256*1024**2) for n in METADATA}
        ctx=authenticate(supplied)
        need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'], 'existing recovery claim changed')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        gpu_data=admit_gpu_bytes(ctx,args.gpu_terminal_custody,args.gpu_terminal_mirror)
        stage,data,bootstrap=make_packet(ctx,gpu_data,supplied);stage_raw=raw(stage)
        files={'inputs/'+n:v for n,v in data.items()};files['inputs/packet.json']=stage_raw
        for n,v in supplied.items():files[metadata_name(n)]=v
        for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        files['job.sbatch']=job_script(ctx,stage,bootstrap)
        outer=dict(schema=OUTER_VERSION,source_commit=SOURCE_COMMIT,remote_root=ROOT,resources=RESOURCES,deadlines=DEADLINES,
            finalizer_packet_sha256=sha(stage_raw),gpu_packet_sha256=ctx['pins']['gpu-packet'],gpu_acceptance_sha256=ctx['pins']['gpu-acceptance'],
            gpu_job_id=ctx['metadata']['gpu-acceptance']['job_id'],baseline_sha256=ctx['pins']['gpu-baseline'],
            operational_amendment_sha256=ctx['pins']['operational-amendment'],namespace_claim_sha256=ctx['pins']['namespace-claim'],
            new_input_files={n:sha(v) for n,v in sorted(files.items())},original_input_pins=ctx['upstream_pins'],
            upstream_rosters=ctx['upstream_rosters'],paths=stage['paths'],gpu_terminal_custody=str(ctx['custody_paths'][0]),
            gpu_terminal_mirror=str(ctx['custody_paths'][1]),prior_jobs=46,prior_journals=94,worlds=24,cells=192,comparisons=28,
            policy_ids=list(POLICIES),admission=budget(ctx['metadata']['gpu-baseline']),submitted=False,
            new_training_updates=0,new_responses=0,new_worlds=0,scientific_qualified=False,execution_qualified=False,
            sizing='Frozen CPU reporting cap1200s2CPU7.5GiB. Private saved-response reporting only in allocated worker. '
                   'Future terminal accounting must preserve all46 prior rows/94journals and extend47/96 with actual charges, including failures.')
        files['packet.json']=raw(outer)
        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)), 'exclusive new launch destinations required')
        created=[]
        try:
            for root in (EXT,LOCAL):
                root.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(root);(root/'inputs').mkdir(mode=0o700)
                for n,v in files.items():write(root/n,v)
            for n,v in files.items():need(read(LOCAL/n,maximum=256*1024**2)==read(EXT/n,maximum=256*1024**2)==v, 'two-copy packet differs')
            for root in ctx['custody_paths']:
                now=tree(root);del now['files']['custody-inventory.json']
                need(now==ctx['metadata']['gpu-custody-inventory']['copies'][str(root)], 'original full custody changed during capture')
            need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'], 'claim changed during build')
            read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
            print(raw(dict(packet_sha256=sha(files['packet.json']),finalizer_packet_sha256=sha(stage_raw),submitted=False,scientific_qualified=False)).decode())
        except BaseException as exc:
            for root in created:
                try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
                except BaseException:pass
            raise

if __name__=='__main__':main()
