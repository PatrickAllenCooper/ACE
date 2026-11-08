"""PROPOSED operational TOKEN02 recovery; actual reviewed failure/amendment inputs mandatory.
Requires actual independently accepted realization/collection and actual42/86 baseline after accepted failed41/84.
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
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
FAILED_TOKEN_JOB='35415334'
FAILED_TOKEN_PACKET_SHA='a831737c6bc24a3b38269e36fa347ed3934c7803e7e18b0738881ac0b8fc323d'
FAILED_TOKEN_STAGE_SHA='101d88f3ff15ed322273d63ad281de05afd921754102a907f7232ac000801e7b'
FAILED_TOKEN_RESULT_SHA='39f045ab33f30d07126745bbc27edf41776b4ae96284f166da93256abc479f94'
LOCAL=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/launch02'
EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-02'
ORIGINAL_EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
FAILED_COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
FAILED_JOB='35407428'
COLLECTION_ENTRY_SHA='e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8'
COLLECTION_RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,requeue=False,wall_seconds=900)
COLLECTION_ORIGINAL_DEADLINES=dict(startup=120,work=850,cleanup=875,hard=890)
COLLECTION_DEADLINES=dict(startup=300,work=850,cleanup=875,hard=890)
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
FIXED_PINS.update({
 'realization-acceptance':'5e28e1aa16512eca3230154526dab65df65330f6920fcc094548c320736a65c1',
 'realization-baseline':'69cc97784637b0ec522da6e00e00a1f08bdfabe9712e2104a43010ee19f2035f',
 'collection-failure-acceptance':'00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee',
 'no-response-proof':'59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998',
 'collection-failure-baseline':'eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b',
 'failure-disposition':'fd3a52be3054c1dd8d0d2b324dea3e64bdf8bc3676c28e7367cc0bbf06a3c685',
 'failure-inventory':'ab556daa4bb0bd7b3433b168b21e07d0ef8fdbe3590c79c317a9eb747400b454',
 'failure-readback':'6b0aaa590d3f62e4196643f2b0ca0a5b5015e4214820ec7f9c47ff10a51fe202',
 'failure-custody-inventory':'0b0ab5a287cd58a31eea828e25ac7ccfa599900287d665da93ea2bbac2054b4d',
 'failed-collection-packet':'aaf0f21110c0906ba0a6a0ba23faa9e0b5ffc6098f1802682901d47e28302961',
 'failed-collection-result':'92333af96e9f30d321b4f047e2ffcf3b9dce0f9351eb5979b83a027c27b74cbb',
 'operational-amendment':'fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855',
 'operational-review':'2d472a124d471c3ebe8fdcc9aaea2bdcec5a949486432ec26127886ca65952cc',
 'operational-guide':'b6a1290b8c3f4dc9511858aec56836c055c93fc86d6d5950e1958bf78668d47f',
 'diagnosis':'61dc39510dfa1d50787ba451f90b180f275e057fadf670422f55729d50b28e24',
 'operational-adoption-review':'db2a672a4f0222a06a46d1e61ebc5bdc22b869640f570e19d36a0f699a1ddbaf'})
TEXT_METADATA={'freeze-review','operational-review','operational-guide','operational-adoption-review'}
def metadata_name(name):return name+('.md' if name in TEXT_METADATA else '.json')
BASE_METADATA=tuple(FIXED_PINS)+('namespace-claim','realization-disposition','realization-packet','realization-result',
 'collection-acceptance','collection-disposition','collection-packet','collection-result','baseline',
 'failed-supervisor-packet','failed-inner-admission','failed-supervisor-admission',
 'collection-supervisor-packet','collection-inner-admission','collection-supervisor-admission',
 'collection-science-packet')

TOKEN_RECOVERY_METADATA=('failed-token-packet','failed-token-stage','failed-token-result',
 'token-failure-acceptance','token-no-learnedwork-proof','token-failure-baseline','token-failure-disposition',
 'token-failure-inventory','token-failure-readback','token-failure-custody-inventory',
 'token-operational-amendment','token-operational-review','token-operational-adoption-review','token-operational-guide','token-diagnosis')
METADATA=BASE_METADATA+TOKEN_RECOVERY_METADATA
TEXT_METADATA.update({'token-operational-review','token-operational-adoption-review','token-operational-guide'})

PROPOSAL='results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
PROPOSAL_SHA='8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SEMANTIC='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
PROBE='results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/probe.py'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,gpus=0,requeue=False,wall_seconds=600)
ORIGINAL_DEADLINES=dict(startup=120,work=550,cleanup=570,hard=590)
DEADLINES=dict(startup=300,work=550,cleanup=570,hard=590)
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
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job not in ('35402725',FAILED_JOB)
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
    job_name='ace_llm_scientific_realize02' if kind=='realization' else 'ace_llm_scientific_collect02'
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


def failed_collection_transition(m,pins):
    """Consume closed failure evidence unchanged; preserve every predecessor row/journal."""
    a=m['collection-failure-acceptance'];proof=m['no-response-proof'];d=m['failure-disposition']
    old=m['realization-baseline'];base=m['collection-failure-baseline'];result=m['failed-collection-result']
    packet=m['failed-collection-packet']
    need(a['status']=='accepted_failed_before_response' and a['failure_accepted'] is True
         and a['collection_accepted'] is False and a['complete_custody_verified'] is True
         and type(a['required_findings']) is int and a['required_findings']==0 and a['job_id']==FAILED_JOB
         and a['packet_sha256']==pins['failed-collection-packet'] and a['disposition_sha256']==pins['failure-disposition']
         and a['freeze_sha256']==FREEZE_SHA and a['source_commit']==SOURCE_COMMIT
         and a['source_control_flow_and_traceback_verified'] is True and a['two_remote_inventories_matched'] is True,
         'exact independently accepted failed collection required')
    need(proof['status']=='verified_failed_before_collection_worker' and proof['failed_job_id']==FAILED_JOB
         and proof['failure_acceptance_sha256']==pins['collection-failure-acceptance']
         and proof['accounting_sha256']==pins['collection-failure-baseline']
         and proof['packet_sha256']==pins['failed-collection-packet'] and proof['disposition_sha256']==pins['failure-disposition']
         and proof['freeze_sha256']==FREEZE_SHA and proof['source_commit']==SOURCE_COMMIT
         and proof['realization_acceptance_sha256']==pins['realization-acceptance']
         and all(proof['proof'][k] is True for k in ('owned_worker_not_spawned','collection_not_invoked',
             'worker_session_not_created','synchronous_stage_did_not_return')),'actual positive no-response proof required')
    zeros=('environment_response_attempts','environment_responses','unknown_return_attempts','scientific_worlds_generated',
           'records_created','pairs_created','scientific_response_cost_units','training_updates')
    need(all(type(v[k]) is int and v[k]==0 for v in (a,proof) for k in zeros)
         and all(type(proof[k]) is int and proof[k]==0 for k in ('model_loads','tokenizer_calls','policy_updates'))
         and a['owned_worker_spawned'] is False and a['collection_invoked'] is False,
         'missing/unknown response charges cannot authorize recovery')
    need(all(v['runtime_qualified_for_this_execution'] is False and v['node_local_partial_runtime_cleanup_verified'] is False
             for v in (a,proof)), 'do not upgrade incomplete runtime/cleanup evidence')
    first=decode(m['failure-inventory']['stdout']);last=decode(m['failure-readback']['stdout'])
    need(m['failure-inventory']['returncode']==m['failure-readback']['returncode']==0
         and first['terminal_gate'] is last['terminal_gate'] is True
         and not first.get('observation_error') and not last.get('observation_error'), 'original failed inventories required')
    need(all(first[k]==last[k] for k in ('files','directories','journal_pins','parent_accounting','upstream_pins','upstream_errors'))
         and normalized(first['related_accounting'])==normalized(last['related_accounting'])
         and first['files']==d['remote_files'] and len(first['files'])==146
         and sum(v['bytes'] for v in first['files'].values())==2336700, 'complete original failed custody changed')
    for name,field in [('failure-inventory','inventory_transport.json'),('failure-readback','inventory_after_transport.json'),
                       ('failure-custody-inventory','custody-inventory.json')]:
        need(a['evidence_pins'][field]==pins[name],'failure acceptance evidence pin differs')
    need(d['job_id']==FAILED_JOB and d['remote_root']==FAILED_COLLECTION_ROOT and d['status']=='terminal_failed_outputs_preserved'
         and d['custody_bytes_verified'] is d['current_lineage_verified'] is True
         and d['errors']==d['input_pin_mismatches']==[] and d['collection_accepted'] is d['acceptance'] is False,
         'original failed disposition required, never successful collection')
    need(packet['remote_root']==FAILED_COLLECTION_ROOT and packet['source_commit']==SOURCE_COMMIT
         and packet['new_input_files']['freeze.json']==FREEZE_SHA,'original failed packet binding differs')
    need(result['job_id']==FAILED_JOB and result['returncode']==0 and result['ambiguous'] is False
         and d['submission_result_sha256']==pins['failed-collection-result'], 'original failed-job submit result required')
    for n,h in [('packet.json',pins['failed-collection-packet']),('submission.intent.json',result['intent_sha256']),
                ('submission.result.json',pins['failed-collection-result'])]:
        need(first['files']['launch/'+n]['sha256']==h,'failed raw packet/journal binding differs')
    need(base['input_pins']['predecessor']==pins['realization-baseline']
         and base['input_pins']['acceptance']==pins['collection-failure-acceptance']
         and base['input_pins']['disposition']==pins['failure-disposition'],'failure baseline provenance differs')
    rows=base['related_accounting'];prior=[r for r in rows if r[0]!=FAILED_JOB];new=[r for r in rows if r[0]==FAILED_JOB]
    need(normalized(prior)==normalized(old['related_accounting']) and len(new)==1
         and normalized(new)==normalized([first['parent_accounting']])
         and normalized(rows)==normalized(first['related_accounting']), 'complete40 predecessor rows plus actual failure required')
    row=normalized(new)[0];need(row[1:7]==['ace_llm_scientific_collect01','FAILED','1:0',2,126,'ucb736_asc2']
         and dict(row[7])==dict(billing='2',cpu='2',mem='7.50G',node='1'),'exact failed allocation differs')
    journals={**old['journal_pins'],FAILED_COLLECTION_ROOT+'/submission.intent.json':result['intent_sha256'],
              FAILED_COLLECTION_ROOT+'/submission.result.json':pins['failed-collection-result']}
    need(journals==base['journal_pins']==first['journal_pins'] and len(journals)==84
         and len(rows)==base['terminal_jobs']==41 and base['submission_journals']==84
         and old['terminal_jobs']==40 and old['submission_journals']==82,'complete failed transition differs')
    need(base['closed_cpu_seconds']==d['closed_cpu_seconds']==old['closed_cpu_seconds']+252==11386
         and base['closed_gpu_slice_seconds']==d['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']==1601
         and sum(int(r[4])*int(r[5]) for r in rows)==11386 and base['cpu_admission_seconds']==14986
         and base['outstanding_reservation']==0 and base['preparation_allowance_cpu_seconds']==3600,
         'actual failed allocation remains fully charged')
    return first['files']


def operational_binding(supplied,m,pins):
    """Exact pinned collection-only amendment; no authority inferred from mutable fields."""
    expected=dict(schema='ace-scientific-collection-operational-amendment-v1',status='adopted',stage='collect',recovery_id='collection02',
        freeze_sha256=FREEZE_SHA,source_commit=SOURCE_COMMIT,entry_sha256=COLLECTION_ENTRY_SHA,
        failed_job_id=FAILED_JOB,failed_packet_sha256=pins['failed-collection-packet'],failed_root=FAILED_COLLECTION_ROOT,
        recovery_root=COLLECTION_ROOT,failure_acceptance_sha256=pins['collection-failure-acceptance'],
        no_response_proof_sha256=pins['no-response-proof'],failure_disposition_sha256=pins['failure-disposition'],
        failure_inventory_sha256=pins['failure-inventory'],failure_readback_sha256=pins['failure-readback'],
        failure_custody_inventory_sha256=pins['failure-custody-inventory'],baseline_sha256=pins['collection-failure-baseline'],
        realization_acceptance_sha256=pins['realization-acceptance'],realization_baseline_sha256=pins['realization-baseline'],
        namespace_admission_sha256=pins['namespace-admission'],namespace_claim_sha256=pins['namespace-claim'],
        original_namespace_claim_sha256=ORIGINAL_CLAIM_SHA,original_deadlines=COLLECTION_ORIGINAL_DEADLINES,
        deadlines=COLLECTION_DEADLINES,resources=COLLECTION_RESOURCES,review_sha256=pins['operational-review'],
        guide_sha256=pins['operational-guide'],diagnosis_sha256=pins['diagnosis'],required_findings=0,
        scientific_inputs_unchanged=True,other_stages_unchanged=True,one_submission_only=True,
        automatic_retry_allowed=False,replacement_worlds_allowed=False)
    need(raw(m['operational-amendment'])==raw(expected) and supplied['operational-review'].strip()
         and supplied['operational-adoption-review'].strip(),'exact adopted amendment and independently pinned reviews required')


def collection_science_equality(m,pins):
    """Compare both concrete producer interfaces; no descriptors decoded or algorithms invoked."""
    old=m['failed-collection-packet'];new=m['collection-packet']
    before=m['failed-supervisor-packet'];after=m['collection-supervisor-packet']
    before_a=m['failed-inner-admission'];after_a=m['collection-inner-admission']
    before_o=m['failed-supervisor-admission'];after_o=m['collection-supervisor-admission']
    for packet,prefix in ((old,'failed'),(new,'collection')):
        names={'supervisor-packet':'inputs/packet.json','inner-admission':'inputs/collection-admission.json',
               'supervisor-admission':'inputs/admission.json'}
        for role,path in names.items():need(packet['new_input_files'][path]==pins[prefix+'-'+role],'independent producer metadata pin differs')
        need(packet['new_input_files']['inputs/collection-packet.json']==pins['collection-science-packet'],
             'original inner scientific packet bytes must remain identical')
    need(before['output']==FAILED_COLLECTION_ROOT+'/collection' and after['output']==COLLECTION_ROOT+'/collection'
         and before['deadlines']==COLLECTION_ORIGINAL_DEADLINES and after['deadlines']==COLLECTION_DEADLINES
         and before['resources']==after['resources']==COLLECTION_RESOURCES,'only declared collection operational change allowed')
    need(all(old['new_input_files']['inputs/'+n]==h for n,h in before['files'].items())
         and all(new['new_input_files']['inputs/'+n]==h for n,h in after['files'].items()), 'complete outer/worker file-map join differs')
    need(set(before['files'])==set(after['files']) and len(before['files'])==60
         and all(after['files'][n]==v for n,v in before['files'].items() if n!='collection-admission.json'),
         'all original sources/training/proposal/realization inputs unchanged')
    need(before['files']['collection-admission.json']==pins['failed-inner-admission']
         and after['files']['collection-admission.json']==pins['collection-inner-admission'],'inner admission file binding')
    project=lambda p:{k:v for k,v in p.items() if k not in ('output','deadlines','files')}
    need(project(before)==project(after),'nonoperational supervisor fields changed')
    need(after_a=={**before_a,'output':COLLECTION_TREE.removesuffix('/collection')}
         and before_a['output']==FAILED_COLLECTION_ROOT+'/collection/session/worker/task','inner admission changes only output')
    need(after_o=={**before_o,'packet_sha256':pins['collection-supervisor-packet'],
        'output':COLLECTION_ROOT+'/collection','deadlines':COLLECTION_DEADLINES}
         and before_o['packet_sha256']==pins['failed-supervisor-packet'],'outer admission changes only output/deadline/packet pin')
    need(new['resources']==old['resources']==COLLECTION_RESOURCES and new['deadlines']==COLLECTION_DEADLINES
         and old['deadlines']==COLLECTION_ORIGINAL_DEADLINES and new['source_pins']==old['source_pins']
         and new['baseline_sha256']==pins['collection-failure-baseline']
         and new['operational_amendment_sha256']==pins['operational-amendment'],'collection02 actual resource/source/amendment binding')
    for key in ('planned_records','planned_pairs','max_response_attempts','max_reserved_cost_units','development_payloads_staged'):
        need(new[key]==old[key],'scientific schedule or privacy bound changed: '+key)
    aliases={'collection-failure-acceptance':'collection-failure-acceptance','no-response-proof':'no-response-proof',
        'failure-baseline':'collection-failure-baseline','failure-disposition':'failure-disposition',
        'failure-inventory':'failure-inventory','failure-readback':'failure-readback',
        'failure-custody-inventory':'failure-custody-inventory','operational-amendment':'operational-amendment',
        'operational-review':'operational-review','operational-guide':'operational-guide','diagnosis':'diagnosis'}
    for producer,consumer in aliases.items():need(new['new_input_files'][metadata_name(producer)]==pins[consumer],
        'collection02 must carry exact original operational authority/evidence')
    science=m['collection-science-packet'];c=m['collection-acceptance']['collection']
    need(science['world_roster']==WORLD_IDS and science['scope']=='policy_train' and science['freeze_sha256']==FREEZE_SHA
         and science['world_pins']==c['world_pins'] and science['configuration_pins']==c['configuration_pins'],
         'accepted collection binds the unchanged complete scientific input')
    return dict(original_science_packet_sha256=pins['collection-science-packet'],
        immutable_worker_input_pins={n:h for n,h in before['files'].items() if n!='collection-admission.json'},
        supervisor_nonoperational_sha256=digest(project(before)),scientific_inputs_unchanged=True,
        token_worker_sha256=ENTRY_SHA,token_resources=RESOURCES,token_deadlines=DEADLINES)


def admit_original(supplied):
    need(set(supplied)==set(BASE_METADATA),'all independently pinned authority/evidence inputs required')
    pins={n:sha(v) for n,v in supplied.items()}
    need(all(pins[n]==h for n,h in FIXED_PINS.items()),'adopted candidate03 metadata differs')
    m={n:decode(v) for n,v in supplied.items() if n not in TEXT_METADATA};f=m['freeze'];ns=m['namespace-admission']
    need(f['source_commit']==SOURCE_COMMIT and f['stage_closure']['source_pins']==f['source_pins'],'adopted source closure differs')
    expected=dict(stage='tokenize',entry=ENTRY,entry_sha256=ENTRY_SHA,cpus=3,memory_mib=11520,gpus=0,
        gpu_type=None,wall_seconds=600,deadlines=ORIGINAL_DEADLINES,account='ucb736_asc2',partition='acpu',qos='cpu-normal',
        nodes=1,tasks=1,requeue=False,max_restarts=0,remote_root=FAILED_TOKEN_ROOT,
        repository_root=str(LOCAL.parent.relative_to(REPO)),exclusive_external_root=str(ORIGINAL_EXT))
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
    failed_files=failed_collection_transition(m,pins)
    operational_binding(supplied,m,pins)
    collection_job=transition(m,pins,'collection','collection-failure-baseline','baseline',42,900)
    equality=collection_science_equality(m,pins)
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
    return dict(metadata=m,pins=pins,job=realization_job,collection_job=collection_job,failed_files=failed_files,science_equality=equality)


def failed_token_transition(m,pins):
    """Consume independently accepted failure; never infer acceptance from the traceback alone."""
    a=m['token-failure-acceptance'];proof=m['token-no-learnedwork-proof']
    d=m['token-failure-disposition'];base=m['token-failure-baseline'];old=m['baseline']
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
         and packet['resources']==RESOURCES and packet['deadlines']==ORIGINAL_DEADLINES
         and packet['baseline_sha256']==pins['baseline']
         and packet['new_input_files']['inputs/packet.json']==pins['failed-token-stage']
         and packet['new_input_files']['collection-acceptance.json']==pins['collection-acceptance']
         and packet['new_input_files']['namespace-claim.json']==pins['namespace-claim']
         and packet['new_input_files']['freeze.json']==FREEZE_SHA,'original failed TOKEN01 packet lineage differs')
    need(stage['source_commit']==SOURCE_COMMIT and stage['resources']==RESOURCES
         and stage['deadlines']==ORIGINAL_DEADLINES and stage['model_directory']==FAILED_TOKEN_ROOT+'/tokenizer'
         and stage['collection']==m['collection-acceptance']['collection']
         and stage['files'][ENTRY]==ENTRY_SHA and len(stage['files'])==34 and len(packet['source_pins'])==32
         and stage['source_names']==m['freeze']['stage_closure']['stages']['token']['source_names'],
         'original failed inner scientific/tokenizer/source contract differs')
    need(result['job_id']==FAILED_TOKEN_JOB and result['returncode']==0 and result['ambiguous'] is False
         and pin(result['intent_sha256']),'actual unambiguous original failed-token submission required')
    first=decode(m['token-failure-inventory']['stdout']);last=decode(m['token-failure-readback']['stdout'])
    need(m['token-failure-inventory']['returncode']==m['token-failure-readback']['returncode']==0
         and first['terminal_gate'] is last['terminal_gate'] is True
         and not first.get('observation_error') and not last.get('observation_error'), 'complete original token transport observations required')
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
    need(base['input_pins']['predecessor']==pins['baseline'] and base['input_pins']['acceptance']==pins['token-failure-acceptance']
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
    preserved={};preserved_directories={}
    for label,root in [('launch',FAILED_TOKEN_ROOT),('common',COLLECTION_ROOT),('failed01',FAILED_COLLECTION_ROOT)]:
        files={n[len(label)+1:]:v['sha256'] for n,v in first['files'].items() if n.startswith(label+'/')}
        need(files and all(relative(n) and pin(h) for n,h in files.items()),'complete bounded failed/prior raw root required')
        preserved[root]=files
        preserved_directories[root]=sorted(n[len(label)+1:] for n in first['directories'] if n.startswith(label+'/'))
    for label,metadata in [('common','collection-disposition'),('failed01','failure-disposition')]:
        want={n[len('launch/'):]:v['sha256'] for n,v in m[metadata]['remote_files'].items() if n.startswith('launch/')}
        root=COLLECTION_ROOT if label=='common' else FAILED_COLLECTION_ROOT
        need(preserved[root]==want,'failed-token capsule must preserve whole original collection02/01 roots')
    return dict(files=first['files'],directories=first['directories'],preserved_roots=preserved,preserved_directories=preserved_directories)


def token_operational_binding(supplied,m,pins):
    expected=dict(schema='ace-scientific-tokenization-operational-amendment-v1',status='adopted',stage='tokenize',recovery_id='token02',
        freeze_sha256=FREEZE_SHA,source_commit=SOURCE_COMMIT,entry_sha256=ENTRY_SHA,
        failed_job_id=FAILED_TOKEN_JOB,failed_packet_sha256=pins['failed-token-packet'],failed_stage_packet_sha256=pins['failed-token-stage'],
        failed_root=FAILED_TOKEN_ROOT,recovery_root=ROOT,
        failure_acceptance_sha256=pins['token-failure-acceptance'],no_learnedwork_proof_sha256=pins['token-no-learnedwork-proof'],
        failure_disposition_sha256=pins['token-failure-disposition'],failure_inventory_sha256=pins['token-failure-inventory'],
        failure_readback_sha256=pins['token-failure-readback'],failure_custody_inventory_sha256=pins['token-failure-custody-inventory'],
        baseline_sha256=pins['token-failure-baseline'],collection_acceptance_sha256=pins['collection-acceptance'],
        collection_baseline_sha256=pins['baseline'],namespace_admission_sha256=pins['namespace-admission'],namespace_claim_sha256=pins['namespace-claim'],
        original_namespace_claim_sha256=ORIGINAL_CLAIM_SHA,original_deadlines=ORIGINAL_DEADLINES,deadlines=DEADLINES,resources=RESOURCES,
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
    need(before.pop('deadlines')==ORIGINAL_DEADLINES and after.pop('deadlines')==DEADLINES
         and before.pop('model_directory')==FAILED_TOKEN_ROOT+'/tokenizer'
         and after.pop('model_directory')==ROOT+'/tokenizer','only explicit startup/root changes permitted')
    need(raw(before)==raw(after),'complete child scientific/source/data/runtime/numerical contract must remain unchanged')
    return dict(original_stage_sha256=sha(raw(old)),new_stage_sha256=sha(raw(new)),nonoperational_semantic_sha256=digest(before),
        changed_fields=['deadlines.startup','model_directory'],scientific_inputs_unchanged=True)


def admit(supplied):
    need(set(supplied)==set(METADATA),'complete original inputs plus explicit token failure/amendment inputs required')
    ctx=admit_original({n:supplied[n] for n in BASE_METADATA})
    pins={n:sha(v) for n,v in supplied.items()};m={n:decode(v) for n,v in supplied.items() if n not in TEXT_METADATA}
    failure=failed_token_transition(m,pins);authority=token_operational_binding(supplied,m,pins)
    base=m['token-failure-baseline'];stages={s['stage']:s for s in m['freeze']['stages']}
    remaining=[stages[n] for n in ('tokenize','train','cpu_campaign','gpu_campaign','finalize')]
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
         and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,
         'actual failed charge plus all remaining frozen caps exceed budget')
    return dict(**{k:v for k,v in ctx.items() if k not in ('metadata','pins')},metadata=m,pins=pins,
        failed_token=failure,token_recovery=authority)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:
        ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--failed-token-remote-custody',required=True,help='complete independently audited TOKEN01 terminal remote/ tree; byte-only')
    ap.add_argument('--failed-remote-custody',required=True,help='original failed terminal01/remote; opaque complete byte check')
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
        verify_collection_tree(a.failed_remote_custody,{n:v['sha256'] for n,v in ctx['failed_files'].items()})
        failed_token_tree=str(Path(a.failed_token_remote_custody).resolve(strict=True))
        failed_token_readback=verify_collection_tree(failed_token_tree,{n:v['sha256'] for n,v in ctx['failed_token']['files'].items()})
        need(sorted(str(q.relative_to(Path(failed_token_tree))) for q in Path(failed_token_tree).rglob('*') if q.is_dir())
             ==sorted(ctx['failed_token']['directories']),'complete failed token directory roster differs')
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
        recovery_equality=token_science_equality(m['failed-token-stage'],packet)
        diagnostic=data['qualify_llm_intervention_runtime_metadata.py'];packet_raw=raw(packet)
        command=[packet['runtime_source']+'/bin/python','-I','-B','-c',literal(diagnostic,'BOOTSTRAP'),ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
            '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(packet_raw),'--source-commit',SOURCE_COMMIT,'--output',ROOT+'/preparation']
        files={'inputs/'+n:v for n,v in data.items()};files['inputs/packet.json']=packet_raw
        files.update({'tokenizer/'+n:v for n,v in tokens.items()})
        files.update({metadata_name(n):v for n,v in supplied.items()})
        for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        # Outer transport tooling only; child files/source_names and scientific freeze are unchanged.
        for n,h in (('stream_custody.py','fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a'),
                    ('stream_custody_review.md','5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371')):
            files[n]=read(Path(__file__).resolve().with_name(n),h)
        files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_token02',
            '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=3','#SBATCH --mem=11520M',
            '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
            '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
            "export CUDA_VISIBLE_DEVICES=''",'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
            'exec '+shlex.join(command),''])).encode()
        base=m['token-failure-baseline']
        outer=dict(schema=OUTER_VERSION,source_commit=SOURCE_COMMIT,local_root=str(LOCAL),external_root=str(EXT),remote_root=ROOT,
            resources=RESOURCES,deadlines=DEADLINES,source_pins={origins[n]:sha(v) for n,v in data.items() if n in origins},
            new_input_files={n:sha(v) for n,v in sorted(files.items())},baseline_sha256=pins['token-failure-baseline'],
            token_recovery=ctx['token_recovery'],token_science_equality=recovery_equality,failed_token_custody=failed_token_tree,failed_token_custody_readback=failed_token_readback,
            realization_job_id=ctx['job'],collection_job_id=ctx['collection_job'],namespace_claim_sha256=pins['namespace-claim'],
            collection_acceptance_sha256=pins['collection-acceptance'],collection_custody=custody,
            operational_amendment_sha256=pins['operational-amendment'],operational_adoption_review_sha256=pins['operational-adoption-review'],
            collection_failure_acceptance_sha256=pins['collection-failure-acceptance'],no_response_proof_sha256=pins['no-response-proof'],
            science_equality=ctx['science_equality'],
            admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],
                preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=1800,candidate_gpu_reservation=0,
                total_cpu_seconds=base['closed_cpu_seconds']+5400,total_gpu_seconds=base['closed_gpu_slice_seconds'],
                ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,prior_jobs=base['terminal_jobs'],prior_journals=base['submission_journals']),
            namespace_claimed=False,submitted=False,qualification='independent concrete packet review required')
        files['packet.json']=raw(outer)
        for dest in (LOCAL,EXT):
            dest.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(dest)
            for sub in ('inputs','tokenizer'):(dest/sub).mkdir(mode=0o700)
            for n,v in files.items():write(dest/n,v)
        for n,v in files.items():need(read(LOCAL/n)==read(EXT/n)==v,'two-copy readback differs')
        verify_collection_tree(failed_token_tree,{n:v['sha256'] for n,v in ctx['failed_token']['files'].items()})
        need(sorted(str(q.relative_to(Path(failed_token_tree))) for q in Path(failed_token_tree).rglob('*') if q.is_dir())
             ==sorted(ctx['failed_token']['directories']),'failed token directories changed during build')
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


