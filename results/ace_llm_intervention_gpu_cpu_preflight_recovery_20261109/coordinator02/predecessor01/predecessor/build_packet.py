"""Prospective GPU development campaign coordinator, byte-only CPU custody admission.
No project imports or descriptor/outcome/token/checkpoint decoding. Future accepted
stage metadata and independent pins are mandatory. Building never submits.
"""
from pathlib import Path
import types
import argparse,ast,contextlib,datetime,fcntl,hashlib,json,os,re,shlex,stat,subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
FAILED_TOKEN_JOB='35415334'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='qualify_llm_intervention_scientific_campaign_gpu.py'
ENTRY_SHA='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
VERSION='ace-scientific-paired-campaign-gpu-v1'
OUTER_VERSION='ace-scientific-campaign-gpu-launch-v1'
RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)
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
 ('training',TRAIN_ROOT,45,92,4,1,2700),('cpu',CPU_ROOT,46,94,2,0,6000))
TEXT_METADATA={'freeze-review','operational-adoption-review','site-readiness','cpu-review'}
def metadata_name(n):return n+('.md' if n in TEXT_METADATA else '.json')
METADATA=tuple(FIXED)+('namespace-claim','site-readiness','cpu-review','cpu-packet','cpu-phase-terminal')+tuple(
 prefix+'-'+kind for prefix,*_ in STAGES for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
CPU_ACCEPTANCE_KEYS={'schema','status','cpu_packet_sha256','job_id','allocation','cpu_output','campaign_root',
 'cpu_output_files','cpu_output_excluded_directory','campaign_files','bundle_path','bundle_sha256',
 'phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}

# Standalone outer metadata validator, exact captured CPU history source; never child code.
CPU_HISTORY_SHA='3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'
TOKEN_METADATA=('collection-failure-disposition','failed-token-packet','failed-token-stage','failed-token-result',
 'token-failure-acceptance','token-no-learnedwork-proof','token-failure-disposition','token-failure-inventory',
 'token-failure-readback','token-failure-custody-inventory','token-failure-baseline','token-operational-amendment',
 'token-operational-review','token-operational-adoption-review','token-operational-guide','token-diagnosis')
TEXT_METADATA |= {'token-operational-review','token-operational-adoption-review','token-operational-guide'}
METADATA+=TOKEN_METADATA+('token-packet',)
HISTORY_ALIASES={'cpu-packet':'token-packet','cpu-launch-packet':'token-launch-packet','cpu-result':'token-result',
 'cpu-disposition':'token-disposition','cpu-inventory':'token-inventory','cpu-preparation-acceptance':'token-acceptance',
 'training-packet':'training-launch-packet','gpu-training-acceptance':'training-acceptance',
 'previous-baseline':'token-baseline','baseline':'training-baseline'}


def load_cpu_history():
    path=Path(__file__).resolve().with_name('cpu_history_contract.py');source=read(path,CPU_HISTORY_SHA)
    module=types.ModuleType('captured_cpu_history');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),vars(module))
    return module


def cpu_history_view(m,pins):
    values=dict(m);hashes=dict(pins)
    for target,source in HISTORY_ALIASES.items():values[target]=m[source];hashes[target]=pins[source]
    values['expected-training']=m['cpu-packet']['expected_training']
    return values,hashes


FULL_MAP_METADATA = frozenset(('cpu-inventory', 'cpu-disposition', 'cpu-acceptance', 'gpu-inventory', 'gpu-inventory-after', 'gpu-custody-inventory', 'gpu-disposition', 'gpu-acceptance'))

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

def history(m,pins):
    previous=m['collection-failure-baseline'];previous_pin=pins['collection-failure-baseline'];jobs=[];observations={}
    need(previous['terminal_jobs']==41 and previous['submission_journals']==84,'accepted failed41/84 boundary required')
    for prefix,root,count,journals,cpus,gpus,wall in STAGES:
        if prefix=='token':
            hm,hp=cpu_history_view(m,pins);load_cpu_history().failed_token_transition(hm,hp)
            previous=m['token-failure-baseline'];previous_pin=pins['token-failure-baseline']
        a,p,r,d,t,base=(m[prefix+'-'+kind] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
        job=r['job_id'];need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job not in jobs+['35407428'],'distinct actual job required')
        jobs.append(job)
        need(r['returncode']==0 and r['ambiguous'] is False and pin(r['intent_sha256']) and a['job_id']==job,'accepted original submission differs')
        need(p['remote_root']==root and p['source_commit']==SOURCE_COMMIT and p['baseline_sha256']==previous_pin,'original stage source/root/predecessor differs')
        schemas=dict(collection='ace-scientific-collection-launch-v1',token='ace-scientific-tokenization-launch-v1',
                     training='ace-scientific-training-gpu-launch-v1',cpu='ace-scientific-cpu-campaign-launch-v1')
        request=p['resources'];memory=dict(collection=7680,token=11520,training=15360,cpu=7680)[prefix]
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

def token_recovery_history(supplied,m,pins,observations):
    """Re-use the captured metadata validator; bind its result through the CPU campaign."""
    helper=load_cpu_history();hm,hp=cpu_history_view(m,pins)
    need(pins['collection-failure-disposition']==helper.FIXED['collection-failure-disposition'],
         'independent original collection failure disposition differs')
    bound=helper.campaign_token_history(supplied,hm,hp,observations['training']['files'])
    outer=m['cpu-launch-packet'];recovery=bound['token_recovery']
    need(outer['token_recovery']==recovery and outer['failed_token_root']==FAILED_TOKEN_ROOT
         and outer['failed_token_job_id']==FAILED_TOKEN_JOB and outer['preserved_directories']==bound['preserved_directories'],
         'original CPU campaign nested TOKEN02 authority differs')
    for name in TOKEN_METADATA:
        need(outer['new_input_files'][metadata_name(name)]==pins[name],'CPU original recovery capture differs: '+name)
    for member,key in {'cpu-packet.json':'token-packet','cpu-launch-packet.json':'token-launch-packet',
        'cpu-result.json':'token-result','cpu-disposition.json':'token-disposition','cpu-inventory.json':'token-inventory',
        'collection-baseline.json':'collection-baseline','collection-acceptance.json':'collection-acceptance',
        'collection-disposition.json':'collection-disposition'}.items():
        need(outer['new_input_files'][member]==pins[key],'CPU original history capture differs: '+member)
    need(all(outer['original_input_pins'].get(path)==h for path,h in bound['upstream_pins'].items()),
         'CPU original full failed/token/collection byte pins differ')
    # CPU custody must contain all original stage roots, not merely their summary counts.
    cpu_files=observations['cpu']['files'];cpu_dirs=observations['cpu']['directories']
    failed=decode(m['token-failure-inventory']['stdout'])
    candidates=[n.split('/')[0] for n,v in cpu_files.items() if n.count('/')==1 and n.endswith('/packet.json')
                and v['sha256']==pins['failed-token-packet']]
    need(len(candidates)==1,'unique failed TOKEN01 root required in CPU custody')
    mapping=[('training',TRAIN_ROOT,observations['training'],'launch'),('cpu',TOKEN_ROOT,observations['token'],'launch'),
        (candidates[0],FAILED_TOKEN_ROOT,failed,'launch'),('common',COLLECTION_ROOT,observations['collection'],'launch'),
        ('failed01',helper.FAILED_COLLECTION_ROOT,failed,'failed01')]
    upstream={};directories={}
    for label,root,original,old_label in mapping:
        want={n[len(old_label)+1:]:v for n,v in original['files'].items() if n.startswith(old_label+'/')}
        got={n[len(label)+1:]:v for n,v in cpu_files.items() if n.startswith(label+'/')}
        want_dirs=sorted(n[len(old_label)+1:] for n in original['directories'] if n.startswith(old_label+'/'))
        got_dirs=sorted(n[len(label)+1:] for n in cpu_dirs if n.startswith(label+'/'))
        need(want and want==got and want_dirs==got_dirs,'complete CPU upstream root changed: '+label)
        for n,v in want.items():safe_relative(n);need(pin(v['sha256']),'invalid original history pin');upstream[root+'/'+n]=v['sha256']
        directories[root]=want_dirs
    return dict(token_recovery=recovery,history_upstream_pins=upstream,history_directories=directories)


def authenticate(supplied):
    need(set(supplied)==set(METADATA),'complete independent metadata inputs required')
    pins={n:sha(v) for n,v in supplied.items()};need(all(pins[n]==h for n,h in FIXED.items()),'fixed original authority bytes differ')
    need(all(supplied[n].strip() for n in TEXT_METADATA),'independent review/readiness evidence required')
    m={n:decode(v) for n,v in supplied.items() if n not in TEXT_METADATA};f=m['freeze']
    need(m['adoption']==dict(schema='ace-scientific-contract-adoption-v1',status='adopted',freeze_sha256=pins['freeze'],
         review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True),'original independent freeze adoption required')
    need(f['source_commit']==SOURCE_COMMIT and f['stage_closure']['source_pins']==f['source_pins'] and len(f['source_pins'])==67,'original full frozen source authority required')
    stages={s['stage']:s for s in f['stages']};s=stages['gpu_campaign']
    need(s['entry']==ENTRY and s['entry_sha256']==ENTRY_SHA and s['remote_root']==ROOT and s['deadlines']==LIMITS
         and s['cpus']==4 and s['memory_mib']==15360 and s['wall_seconds']==2880 and s['gpus']==1 and s['gpu_type']=='a100_3g.20gb'
         and s['account']=='ucb736_asc2' and s['partition']=='aa100' and s['qos']=='gpu-testing' and s['requeue'] is False,'frozen GPU campaign profile differs')
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
    base=m['cpu-baseline'];remaining=[stages['gpu_campaign'],stages['finalize']]
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
         and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,'full remaining frozen caps exceed budget')
    recovery=token_recovery_history(supplied,m,pins,observations)
    return dict(metadata=m,pins=pins,observations=observations,**recovery)

def admit_cpu_bytes(ctx,raw_root):
    m,pins=ctx['metadata'],ctx['pins'];old=m['cpu-packet'];a=m['cpu-acceptance'];files=ctx['observations']['cpu']['files']
    gate=ctx['compact_inputs'];need(Path(raw_root).resolve(strict=True) in gate.local_roots,'bound selected CPU view required')
    observed=gate.full['files'];need(observed==files,'complete remote CPU raw inventory differs')
    need(files['launch/inputs/packet.json']['sha256']==pins['cpu-packet']
         and m['cpu-launch-packet']['new_input_files']['inputs/packet.json']==pins['cpu-packet'],'original CPU inner packet differs')
    need(old['cpu_output']==a['cpu_output']==CPU_ROOT+'/cpu'
         and a['campaign_root']==CPU_ROOT+'/cpu/worker/artifacts/campaign'
         and a['cpu_output_excluded_directory']=='worker/artifacts/campaign','original CPU output roots differ')
    cpu_files={n.removeprefix('launch/cpu/'):v['sha256'] for n,v in files.items() if n.startswith('launch/cpu/')}
    campaign_prefix='worker/artifacts/campaign/'
    need({n:v for n,v in cpu_files.items() if not n.startswith(campaign_prefix)}==a['cpu_output_files']
         and {n.removeprefix(campaign_prefix):v for n,v in cpu_files.items() if n.startswith(campaign_prefix)}==a['campaign_files'],
         'complete accepted CPU output/campaign inventories differ')
    for key,path in [('bundle','worker/artifacts/cpu-phase/campaign-bundle.json'),('phase_terminal','worker/artifacts/cpu-phase/terminal.json')]:
        need(a[key+'_path']==path and pin(a[key+'_sha256']) and a['cpu_output_files'][path]==a[key+'_sha256'],'original CPU phase path/pin differs')
    need(a['phase_terminal_sha256']==pins['cpu-phase-terminal'],'independent actual CPU phase terminal differs')
    terminal=m['cpu-phase-terminal'];d=m['training-acceptance']['dpo_steps']
    need(type(d) is int and 1<=d<=128 and terminal['schema']=='ace-scientific-campaign-cpu-v1'
         and terminal['status']=='cpu_phase_complete' and terminal['attempt_pins']==a['attempt_pins']
         and terminal['numeric_training_updates']==160+d and terminal['llm_training_updates']==0
         and terminal['completed_cpu_cells']==120 and terminal['declared_cells']==192 and terminal['gpu_services_started']==0
         and terminal['private_scores_computed'] is terminal['scientific_qualified'] is terminal['retry_allowed'] is False,
         'accepted complete matched CPU phase required')
    need(set(a['attempt_pins'])==set(POLICIES[:5]),'all5 accepted CPU attempts required')
    for i,pid in enumerate(POLICIES[:5]):
        h=a['attempt_pins'][pid];need(pin(h) and all(a['campaign_files'][f'policy-{i:02d}/'+n]==h for n in ('attempt.json','attempt.candidate.json')),'CPU attempts changed')
    roots={n.split('/',1)[0] for n in a['campaign_files']}
    need(roots=={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}' for i in range(5)},'GPU attempts/finalization already exist')
    data={}
    for n,h in old['files'].items():
        need(Path(n).name==n and pin(h),'plain accepted CPU input required')
        need(files['launch/inputs/'+n]['sha256']==h,'CPU input/original inventory differs')
        data[n]=gate.capture('launch/inputs/'+n,h,256*1024**2)
    upstream={CPU_ROOT+'/'+n.removeprefix('launch/'):v['sha256'] for n,v in files.items() if n.startswith('launch/')}
    recheck_storage(ctx)
    for prefix,root,*_ in STAGES:
        outer=m[prefix+'-launch-packet'];result=m[prefix+'-result']
        for name,h in {**outer['new_input_files'],'packet.json':pins[prefix+'-launch-packet'],'submission.intent.json':result['intent_sha256'],'submission.result.json':pins[prefix+'-result']}.items():
            path=root+'/'+name;need(path not in upstream or upstream[path]==h,'historical original input pin collision');upstream[path]=h
    for path,h in ctx['history_upstream_pins'].items():
        need(path not in upstream or upstream[path]==h,'original failed/history pin collision');upstream[path]=h
    ctx['upstream_pins']=upstream
    return data

def make_packet(ctx,cpu_data,supplied):
    m,pins=ctx['metadata'],ctx['pins'];f=m['freeze'];old=m['cpu-packet']
    closure=f['stage_closure']['stages']['GPUcampaign'];cpu_closure=f['stage_closure']['stages']['CPUcampaign']
    need(set(cpu_data)==set(old['files']) and all(sha(v)==old['files'][n] for n,v in cpu_data.items()),'exact unchanged CPU bytes required')
    need(old['source_names']==cpu_closure['source_names'] and all(old['files'][n]==h for n,h in cpu_closure['input_source_files'].items()),'same frozen CPU source closure required')
    data=dict(cpu_data);captured={}
    for origin,h in f['source_pins'].items():
        safe_relative(origin);value=committed(SOURCE_COMMIT,origin);need(sha(value)==h,'full67-source Git closure differs');captured[origin]=value
    for n,h in closure['input_source_files'].items():
        candidates=[v for p,v in captured.items() if Path(p).name==n and sha(v)==h]
        need(len(candidates)==1,'unique frozen basename/origin required: '+n)
        value=candidates[0];need(n not in data or data[n]==value,'new GPU source would replace original CPU bytes');data[n]=value
    entry=data[ENTRY]
    need(sha(entry)==ENTRY_SHA and closure['entrypoint']==ENTRY and closure['entrypoint_sha256']==ENTRY_SHA,'frozen GPU entry differs')
    extra=literal(entry,'EXTRA_SOURCES')
    need(set(data)==set(old['files'])|extra and closure['source_names']==sorted(set(old['source_names'])|extra), 'exact inherited CPU plus five GPU additions required')
    data.update({'cpu-packet.json':supplied['cpu-packet'],'cpu-acceptance.json':supplied['cpu-acceptance']})
    need(len(data)<=256 and sum(map(len,data.values()))<=512*1024**2,'bounded full GPU input capture required')
    diagnostic=data['qualify_llm_intervention_runtime_metadata.py'];service=data['llm_intervention_model_service.py']
    bootstrap=literal(diagnostic,'BOOTSTRAP')
    need(literal(service,'RUNTIME')==old['runtime'] and literal(diagnostic,'RUNTIME_SOURCE')==old['runtime_source'],'original qualified runtime constants differ')
    paths=dict(cpu_inputs=CPU_ROOT+'/inputs',cpu_output=CPU_ROOT+'/cpu',campaign=CPU_ROOT+'/cpu/worker/artifacts/campaign',
               gpu_inputs=ROOT+'/inputs',gpu_output=ROOT+'/gpu')
    need(m['cpu-launch-packet']['future_gpu_output']==paths['gpu_output'],'CPU frozen prospective GPU output differs')
    for pid in POLICIES[5:]:
        freeze=decode(data[pid+'.freeze.json']);contract=decode(data[pid+'.launch.json']);adapter=None
        if pid!='frozen_llm':
            role='sft' if pid=='sft_llm' else 'final';accepted=m['training-acceptance']
            adapter=dict(file=role+'.adapter.pt',sha256=accepted[role+'_adapter_sha256'],training_receipt=role+'.checkpoint.json',
                         training_receipt_sha256=accepted[role+'_checkpoint_sha256'],training_scope='policy_train')
            need(sha(data[adapter['file']])==adapter['sha256'] and sha(data[adapter['training_receipt']])==adapter['training_receipt_sha256'],'opaque accepted checkpoint bytes differ')
        need(freeze['scope']=='development' and freeze['model_directory']==f['model_directory'] and freeze['model_files']==f['model_files']
             and freeze['runtime']==f['runtime'] and freeze['recipe']==literal(service,'RECIPE')
             and freeze['adapter']==adapter and freeze['treatment']==('frozen' if adapter is None else 'sft' if pid=='sft_llm' else 'dpo')
             and freeze['sources']=={n:sha(data[n]) for n in literal(service,'SOURCE_NAMES')}
             and freeze['limits']==dict(startup_seconds=120,request_seconds=30,idle_seconds=30,total_seconds=1200,max_requests=576),
             'unchanged fixed three-service treatment/recipe/request limits required')
        wanted=dict(policy_id=pid,entry=paths['cpu_inputs']+'/llm_intervention_model_service.py',entry_sha256=sha(service),
            freeze=paths['cpu_inputs']+'/'+pid+'.freeze.json',freeze_sha256=sha(data[pid+'.freeze.json']),
            worker_output=paths['gpu_output']+'/services/'+pid+'/worker',launch_output=paths['gpu_output']+'/services/'+pid+'/launch',
            binding_path=paths['gpu_output']+'/stage-binding.json',stage_path=paths['gpu_output']+'/stage.json',
            archive_sha256=f['runtime_archive_sha256'],manifest_sha256=f['runtime_manifest_sha256'],bootstrap=bootstrap,bootstrap_sha256=sha(bootstrap.encode()))
        need(all(contract[k]==v for k,v in wanted.items()),'original service endpoint/launch contract differs; never rewrite')
        identity=dict(model_sha256=digest(f['model_files']),source_sha256=digest(freeze['sources']),
            adapter_sha256=adapter['sha256'] if adapter else digest(dict(treatment='frozen',adapter=None)),
            numeric_recipe=freeze['recipe'],numeric_recipe_sha256=digest(freeze['recipe']))
        command=[f['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,paths['cpu_inputs']+'/llm_intervention_campaign_launch.py',
                 sha(data['llm_intervention_campaign_launch.py']),'--contract',paths['cpu_inputs']+'/'+pid+'.launch.json','--contract-sha256',sha(data[pid+'.launch.json'])]
        need(old['persistent_specs'][pid]==dict(policy_id=pid,kind='persistent',command=command,timeout_seconds=35,
             identity_sha256=digest(identity),policy_provenance_sha256=sha(data[pid+'.freeze.json']))
             and old['services'][pid]==dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=576)
             and old['service_identity_pins'][pid]==digest(identity),'original persistent service identity/command differs')
    for n in closure['source_names']:
        for node in ast.walk(ast.parse(data[n])):
            deps=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            need(all(dep+'.py' in closure['source_names'] for dep in deps if dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_'))),'captured static import closure incomplete')
    packet=dict(schema=VERSION,files={n:sha(v) for n,v in sorted(data.items())},source_names=closure['source_names'],
        cpu_packet_sha256=pins['cpu-packet'],cpu_acceptance_sha256=pins['cpu-acceptance'],paths=paths,resources=RESOURCES,
        services={p:p+'.freeze.json' for p in POLICIES[5:]},contracts={p:p+'.launch.json' for p in POLICIES[5:]},
        lineage_files=literal(entry,'LINEAGE_FILES'),scope='development',limits=LIMITS,freeze_sha256=pins['freeze'],namespace_admission_sha256=pins['namespace-admission'])
    return packet,data,bootstrap

def admission(ctx,stage):
    return dict(schema=VERSION+'/admission',packet_sha256=sha(raw(stage)),scope='development',freeze_sha256=ctx['pins']['freeze'],
        namespace_admission_sha256=ctx['pins']['namespace-admission'],stage='development_gpu_campaign',output=stage['paths']['gpu_output'],
        resources=RESOURCES,limits=LIMITS,full_contract_frozen_before_first_scientific_draw=True,source_runtime_scheduler_qualified=True,
        namespace_owned=True,accounting_admitted=True,independent_acceptances_verified=True,retry_allowed=False,
        site_readiness_sha256=ctx['pins']['site-readiness'],accounting_sha256=ctx['pins']['cpu-baseline'])

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
COMPACT_ADAPTER_SHA='a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245'
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
    gate=ctx['storage_operation_evidence'].run('binding',lambda:c.bind(ctx,supplied,predecessor='cpu',capsule=capsule,mirror=mirror,
        expected_roots=dict(launch=CPU_ROOT,training=TRAIN_ROOT,cpu=TOKEN_ROOT,**{'failed-token01':FAILED_TOKEN_ROOT},common=COLLECTION_ROOT,failed01=load_cpu_history().FAILED_COLLECTION_ROOT,**{'namespace.json':REMOTE_REGISTRY,'original-namespace.json':ORIGINAL_REMOTE_REGISTRY}),committed=committed,helper_path=CURC_HELPER))
    ctx['storage_supplied']=supplied
    return gate


def recheck_storage(ctx):
    compact_adapter().recheck_capsules(ctx['compact_inputs'])


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
        original_ancestor_references=gate.ancestry,complete_local_raw_copy=False)


def configure_packet_storage(ctx,local,pins,packet):
    c=compact_adapter();supplied=c.stored_config(local,pins)
    gate=configure_storage(ctx,supplied,Path(packet['cpu_remote_custody']).parent,packet['cpu_terminal_mirror'])
    need(packet['storage_binding']==storage_packet_binding(ctx),'independent compact storage packet binding differs')
    return gate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--cpu-remote-custody',required=True,help='selected accepted CPU terminal remote/ view, opaque bytes only')
    ap.add_argument('--cpu-terminal-mirror',required=True,help='independent selected CPU terminal capsule')
    ap.add_argument('--storage-config',required=True)
    ap.add_argument('--storage-config-sha256',required=True)
    args=ap.parse_args()
    with coordinator_locks():
        supplied={n:read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256'),metadata_maximum(n)) for n in METADATA}
        ctx=authenticate(supplied)
        need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'existing recovery claim changed')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        custody=Path(args.cpu_remote_custody).resolve(strict=True)
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
        configure_storage(ctx,storage,custody.parent,args.cpu_terminal_mirror)
        cpu_data=admit_cpu_bytes(ctx,custody)
        stage,data,bootstrap=make_packet(ctx,cpu_data,supplied);stage_raw=raw(stage);gate=admission(ctx,stage)
        files={'inputs/'+n:v for n,v in data.items()};files.update({'inputs/packet.json':stage_raw,'inputs/admission.json':raw(gate)})
        for n,v in supplied.items():files[metadata_name(n)]=v
        files.update(storage_packet_files(ctx))
        for n in ('build_packet.py','launch_once.py','monitor_once.py','cpu_history_contract.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        command=[ctx['metadata']['cpu-packet']['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
            '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(stage_raw),'--admission',ROOT+'/inputs/admission.json',
            '--admission-sha256',sha(raw(gate)),'--output',ROOT+'/gpu']
        files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_campaign_gpu01',
            '#SBATCH --account=ucb736_asc2','#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=4','#SBATCH --mem=15360M','#SBATCH --gres=gpu:a100_3g.20gb:1','#SBATCH --time=00:48:00','#SBATCH --no-requeue',
            '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8',
            'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL','exec '+shlex.join(command),''])).encode()
        base=ctx['metadata']['cpu-baseline']
        outer=dict(schema=OUTER_VERSION,source_commit=SOURCE_COMMIT,remote_root=ROOT,resources=RESOURCES,limits=LIMITS,
            gpu_packet_sha256=sha(stage_raw),cpu_packet_sha256=ctx['pins']['cpu-packet'],cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],
            cpu_job_id=ctx['metadata']['cpu-acceptance']['job_id'],baseline_sha256=ctx['pins']['cpu-baseline'],
            operational_amendment_sha256=ctx['pins']['operational-amendment'],namespace_claim_sha256=ctx['pins']['namespace-claim'],
            new_input_files={n:sha(v) for n,v in sorted(files.items())},original_cpu_inputs=ctx['upstream_pins'],
            token_recovery=ctx['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,failed_token_job_id=FAILED_TOKEN_JOB,
            history_directories=ctx['history_directories'],
            paths=stage['paths'],cpu_remote_custody=str(custody),cpu_terminal_mirror=str(Path(args.cpu_terminal_mirror).resolve(strict=True)),
            storage_binding=storage_packet_binding(ctx),prior_jobs=46,prior_journals=94,worlds=24,cells=192,policy_ids=list(POLICIES),gpu_policy_ids=list(POLICIES[5:]),
            admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],
                preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=11520,candidate_gpu_reservation=2880,
                total_cpu_seconds=base['closed_cpu_seconds']+15120,total_gpu_seconds=base['closed_gpu_slice_seconds']+2880,
                remaining_finalizer_cpu_cap=2400,total_with_finalizer_cpu_seconds=base['closed_cpu_seconds']+17520,
                ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200),
            submitted=False,new_training_updates=0,new_worlds=0,scientific_qualified=False,execution_qualified=False,
            sizing='Frozen one20GB MIG,4CPU15GiB,2880s; measured artificial timings motivate the bound, not a capacity guarantee. '
              'First GPU work240s; service startup120s; freshness90s; late-service reserve600s,cap1200s,work2820/final2860. '
              'Retained2929s timing sensitivity exceeds this cap; final40s validation margin remains uncertain. No adaptive truncation or fit.')
        files['packet.json']=raw(outer)
        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'exclusive new launch destinations required')
        created=[]
        try:
            for root in (EXT,LOCAL):
                root.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(root);(root/'inputs').mkdir(mode=0o700)
                for n,v in files.items():write(root/n,v)
            for n,v in files.items():need(read(LOCAL/n,maximum=256*1024**2)==read(EXT/n,maximum=256*1024**2)==v,'two-copy packet differs')
            recheck_storage(ctx)
            need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'claim changed during build')
            read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
            print(raw(dict(storage_operations=ctx['storage_operation_evidence'].last_reference,packet_sha256=sha(files['packet.json']),gpu_packet_sha256=sha(stage_raw),admission_sha256=sha(raw(gate)),submitted=False,scientific_qualified=False)).decode())
        except BaseException as exc:
            for root in created:
                try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
                except BaseException:pass
            raise

if __name__=='__main__':main()
