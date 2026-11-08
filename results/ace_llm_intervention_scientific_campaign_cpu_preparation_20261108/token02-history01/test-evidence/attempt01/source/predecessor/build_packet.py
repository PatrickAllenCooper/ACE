"""Prospective scientific CPU campaign coordinator; no project-module execution.
The future accepted training/CPU/realization chain must precede opaque input capture.
"""
from pathlib import Path
import argparse,ast,contextlib,datetime,fcntl,hashlib,json,os,re,shlex,stat,subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-scientific-campaign-cpu-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='prepare_llm_intervention_scientific_campaign_cpu.py'
ENTRY_SHA='ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893'
VERSION='ace-scientific-campaign-cpu-v1'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,requeue=False,wall_seconds=6000)
DEADLINES=dict(startup=180,work=5940,hard=5980)
POLICIES=('uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm')
LLM=POLICIES[5:]
FIXED={'freeze':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d',
 'freeze-review':'088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416',
 'adoption':'ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07',
 'namespace-admission':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
 'recovery-authority':'ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50'}
FIXED.update({
 'operational-amendment':'fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855',
 'operational-adoption-review':'db2a672a4f0222a06a46d1e61ebc5bdc22b869640f570e19d36a0f699a1ddbaf',
 'collection-failure-acceptance':'00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee',
 'no-response-proof':'59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998',
 'collection-failure-baseline':'eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b'})
FIXED['realization-acceptance']='5e28e1aa16512eca3230154526dab65df65330f6920fcc094548c320736a65c1'
TEXT_METADATA={'freeze-review','operational-adoption-review','training-review','site-readiness'}
def metadata_name(n):return n+('.md' if n in TEXT_METADATA else '.json')
METADATA=tuple(FIXED)+('namespace-claim','realization-terminal','realization-inventory',
 'gpu-training-acceptance','training-packet','training-result','training-disposition','training-inventory','training-review',
 'cpu-preparation-acceptance','expected-training','baseline','previous-baseline','site-readiness')
ARTIFACTS=('prepared.json','numerical-plan.json','engine-terminal.json','sft.checkpoint.json','final.checkpoint.json','sft.adapter.pt','final.adapter.pt')
ARTIFACT_PATHS={'prepared.json':'launch/inputs/prepared.json','numerical-plan.json':'cpu/preparation/session/worker/artifacts/numerical-plan.json',
 'engine-terminal.json':'launch/training/session/worker/artifacts/fit/training/terminal.json',
 **{n:'launch/training/session/worker/artifacts/fit/training/'+n for n in ('sft.checkpoint.json','final.checkpoint.json','sft.adapter.pt','final.adapter.pt')}}

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

def relative(name):
    need(type(name) is str and name.startswith('launch/') and not Path(name).is_absolute()
         and all(v not in ('','.','..') for v in name.split('/')),'safe original CPU inventory path required')
    return name


def gpu_cost(rows):
    cpu=gpu=0
    for r in normalized(rows):
        t=dict(r[7]);g={k:v for k,v in t.items() if k.startswith('gres/')}
        need(t or r[4]==r[5]==0,'unknown allocation charges')
        need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()),'unknown accelerator charge')
        typed=[int(v) for k,v in g.items() if k.startswith('gres/gpu:')];n=int(g['gres/gpu']) if 'gres/gpu' in g else sum(typed)
        need(not typed or 'gres/gpu' not in g or n==sum(typed),'GPU charge disagreement')
        cpu+=r[4]*r[5];gpu+=n*r[5]
    return cpu,gpu


def admit(supplied):
    need(set(supplied)==set(METADATA),'exact explicit authority/metadata inputs required')
    pins={n:sha(v) for n,v in supplied.items()};need(all(pins[n]==h for n,h in FIXED.items()),'frozen/adopted authority differs')
    m={n:decode(v) for n,v in supplied.items() if n not in TEXT_METADATA}
    need(all(supplied[n].strip() for n in TEXT_METADATA),'independent review/readiness text required')
    f=m['freeze'];stages={s['stage']:s for s in f['stages']};stage=stages['cpu_campaign']
    need(f['source_commit']==SOURCE_COMMIT and stage['entry']==ENTRY and stage['entry_sha256']==ENTRY_SHA
        and stage['remote_root']==ROOT and stage['repository_root']==str(LOCAL.parent.relative_to(REPO))
        and stage['exclusive_external_root']==str(EXT) and stage['deadlines']==DEADLINES
        and all(stage[k]==v for k,v in RESOURCES.items()) and f['numeric_internal_seconds']==50
        and f['numeric_timeout_seconds']==55 and f['policies']==list(POLICIES),'unchanged scientific CPU stage required')
    need(m['adoption']==dict(schema='ace-scientific-contract-adoption-v1',status='adopted',freeze_sha256=pins['freeze'],
        review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True),'independent original adoption required')
    claim=m['namespace-claim'];ns=m['namespace-admission'];authority=m['operational-amendment']
    need(claim['freeze_sha256']==pins['freeze'] and claim['namespace_admission_sha256']==pins['namespace-admission']
        and claim['world_ids']==ns['world_ids'] and claim['local_registry']==str(LOCAL_REGISTRY) and claim['remote_registry']==REMOTE_REGISTRY
        and claim['recovery_authority_sha256']==pins['recovery-authority'] and claim['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA
        and claim['retry_allowed'] is False and claim['replacement_worlds'] is False
        and authority['freeze_sha256']==pins['freeze'] and authority['source_commit']==SOURCE_COMMIT
        and authority['namespace_claim_sha256']==pins['namespace-claim'] and authority['other_stages_unchanged'] is True,
        'existing namespace and separate collection-only amendment required')
    a=m['gpu-training-acceptance'];cpu=m['cpu-preparation-acceptance'];expected=m['expected-training'];tp=m['training-packet']
    ep={'prepared_sha256','training_plan_sha256','exposure_sha256','scientific_freeze_sha256','collection_acceptance_sha256',
        'cpu_preparation_acceptance_sha256','model_files_sha256','runtime_contract_sha256'}
    need(set(expected)==ep|{'job_id','source_commit'} and all(pin(expected[k]) for k in ep)
        and expected['source_commit']==SOURCE_COMMIT and expected['scientific_freeze_sha256']==pins['freeze']
        and expected['cpu_preparation_acceptance_sha256']==pins['cpu-preparation-acceptance']
        and expected['model_files_sha256']==digest(f['model_files']) and expected['runtime_contract_sha256']==digest(f['runtime']),
        'independently supplied complete expected training lineage required')
    need(a['schema']=='ace-scientific-training-gpu-acceptance-v1' and a['status']=='scientific_training_gpu_accepted'
        and a['scope']=='policy_train' and type(a['required_findings']) is int and a['required_findings']==0
        and a['scientific_utility_qualified'] is False and all(a[k]==v for k,v in expected.items())
        and all(a[k] is True for k in ('training_gpu_qualified','checkpoint_reload_inference_qualified','complete_custody_verified',
            'runtime_qualified','owned_process_cleanup_verified','scheduler_verified')),
        'actual independent complete GPU training acceptance required before any private capture')
    need(cpu['schema']=='ace-scientific-training-preparation-acceptance-v1' and cpu['status']=='accepted' and cpu['scope']=='policy_train'
        and cpu['required_findings']==0 and cpu['complete_custody_verified'] is True
        and cpu['runtime_qualified_for_this_cpu_execution'] is cpu['owned_process_cleanup_verified'] is True
        and cpu['training_execution_qualified'] is cpu['scientific_qualified'] is False
        and cpu['prepared_sha256']==expected['prepared_sha256'] and cpu['plan_sha256']==expected['training_plan_sha256']
        and cpu['exposure_sha256']==expected['exposure_sha256'] and cpu['freeze_sha256']==pins['freeze']
        and cpu['namespace_admission_sha256']==pins['namespace-admission']
        and pin(cpu['numerical_plan_sha256']) and pin(cpu['numerical_plan_file_sha256']), 'actual accepted matched numerical/token plan required')
    job=a['job_id'];result=m['training-result'];disp=m['training-disposition'];base=m['baseline'];prior=m['previous-baseline']
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job==result['job_id'] and job!=cpu['job_id']
        and result['returncode']==0 and result['ambiguous'] is False and pin(result['intent_sha256']), 'actual distinct unambiguous training job required')
    need(tp['schema']=='ace-scientific-training-gpu-launch-v1' and tp['source_commit']==SOURCE_COMMIT and tp['remote_root']==TRAIN_ROOT
        and tp['baseline_sha256']==pins['previous-baseline'] and tp['cpu_acceptance_sha256']==pins['cpu-preparation-acceptance']
        and tp['cpu_job_id']==cpu['job_id'] and tp['prepared_sha256']==expected['prepared_sha256'] and tp['plan_sha256']==expected['training_plan_sha256'],
        'original accepted training packet/CPU chain differs')
    for n in FIXED:
        if n=='realization-acceptance':continue
        need(tp['new_input_files'][metadata_name(n)]==pins[n],'training original authority differs: '+n)
    need(tp['new_input_files']['namespace-claim.json']==pins['namespace-claim']
        and tp['new_input_files']['collection-acceptance.json']==expected['collection_acceptance_sha256'], 'training original collection lineage differs')
    need(disp['job_id']==job and disp['remote_root']==TRAIN_ROOT and disp['packet_sha256']==pins['training-packet']
        and disp['submission_result_sha256']==pins['training-result'] and disp['custody_bytes_verified'] is disp['current_lineage_verified'] is True
        and disp['errors']==disp['input_pin_mismatches']==[] and disp['exact_prior_accounting_matched'] is True
        and disp['exact_prior_and_current_journals_matched'] is True and disp['observed_related_jobs']==44
        and disp['observed_submission_journals']==90,'actual full GPU training terminal custody required')
    transport=m['training-inventory'];need(transport['returncode']==0,'successful original training inventory required')
    inv=decode(transport['stdout']);files=inv['files']
    need(inv['terminal_gate'] is True and inv['root_present'] is True and not inv.get('observation_error')
        and inv['parent_accounting'][0]==job and literal(transport['python_source'],'ROOT')==TRAIN_ROOT
        and literal(transport['python_source'],'JOB')==job and len(files)==disp['raw_files']
        and sum(v['bytes'] for v in files.values())==disp['raw_bytes'], 'original complete training inventory/root/job differs')
    originals={**tp['new_input_files'],'packet.json':pins['training-packet'],'submission.intent.json':result['intent_sha256'],
        'submission.result.json':pins['training-result']}
    for n,h in originals.items():relative('launch/'+n);need(files['launch/'+n]['sha256']==h,'training original input/journal pin differs')
    need(prior['terminal_jobs']==43 and prior['submission_journals']==88 and base['terminal_jobs']==44 and base['submission_journals']==90
        and base['outstanding_reservation']==prior['outstanding_reservation']==0 and base['preparation_allowance_cpu_seconds']==3600
        and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
        and base['input_pins']['predecessor']==pins['previous-baseline'] and base['input_pins']['acceptance']==pins['gpu-training-acceptance']
        and base['input_pins']['disposition']==pins['training-disposition'],'independent actual44/90 baseline extension required')
    before=[r for r in base['related_accounting'] if r[0]!=job];current=[r for r in base['related_accounting'] if r[0]==job]
    need(len(before)==43 and len(current)==1 and normalized(before)==normalized(prior['related_accounting'])
        and normalized(current)==normalized([inv['parent_accounting']]) and normalized(base['related_accounting'])==normalized(inv['related_accounting']),
        'all original rows including failures must remain')
    row=normalized(current)[0];alloc=disp['allocation']
    need(row[2:4]==['COMPLETED','0:0'] and row[4]==4 and 0<=row[5]<=2700
        and alloc['resource_contract_matched'] is True and alloc['errors']==[] and alloc['allocated_cpus']==4
        and alloc['elapsed_seconds']==row[5] and alloc['gpu_count']==1 and alloc['gpu_slice_seconds']==row[5]
        and alloc['allocated_cpu_seconds']==4*row[5],'actual accepted bounded training resources differ')
    journals={**prior['journal_pins'],TRAIN_ROOT+'/submission.intent.json':result['intent_sha256'],TRAIN_ROOT+'/submission.result.json':pins['training-result']}
    need(len(prior['journal_pins'])==88 and len(journals)==90 and base['journal_pins']==inv['journal_pins']==journals,
        'all88 historical plus actual training journals required')
    need(gpu_cost(prior['related_accounting'])==(prior['closed_cpu_seconds'],prior['closed_gpu_slice_seconds'])
        and gpu_cost(base['related_accounting'])==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])
        and base['closed_cpu_seconds']==disp['closed_cpu_seconds']==prior['closed_cpu_seconds']+4*row[5]
        and base['closed_gpu_slice_seconds']==disp['closed_gpu_slice_seconds']==prior['closed_gpu_slice_seconds']+row[5]
        and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600,'actual totals only; unknown costs reject')
    remaining=f['stages'][next(i for i,s in enumerate(f['stages']) if s['stage']=='cpu_campaign'):]
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
        and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,'remaining frozen cap closure exceeds budget')
    ra=m['realization-acceptance'];rt=m['realization-terminal'];ri=m['realization-inventory']
    need(ra['status']=='accepted' and ra['required_findings']==0 and ra['complete_custody_verified'] is True
        and ra['realization_terminal_sha256']==pins['realization-terminal'] and ra['freeze_sha256']==pins['freeze']
        and ra['namespace_admission_sha256']==pins['namespace-admission'] and ri['returncode']==0,'original accepted realization required')
    rf=decode(ri['stdout']);need(rf['terminal_gate'] is True and not rf.get('observation_error')
        and literal(ri['python_source'],'ROOT')==REALIZATION_ROOT and literal(ri['python_source'],'JOB')==ra['job_id']
        and rf['files']['launch/realization/session/worker/task/terminal.json']['sha256']==pins['realization-terminal'], 'original realization inventory identity differs')
    need(rt['status']=='complete' and rt['worlds_generated']==56 and rt['environment_responses']==rt['policy_updates']==0
        and rt['freeze_sha256']==pins['freeze'] and rt['namespace_admission_sha256']==pins['namespace-admission']
        and len(rt['entries'])==56 and rt['retry_allowed'] is False,'all56 original realization metadata required')
    world_pins={};descriptor_files={}
    for i,e in enumerate(rt['entries'][32:]):
        need(e['world_id']==202611070100+i and e['scope']=='development' and e['status']=='complete'
            and e['path']==f'world-{32+i:03d}.json' and pin(e['descriptor_sha256']) and pin(e['file_sha256'])
            and rf['files']['launch/realization/session/worker/task/'+e['path']]['sha256']==e['file_sha256'],'exact24 original descriptors required')
        world_pins[str(e['world_id'])]=e['descriptor_sha256'];descriptor_files[f'descriptor-{i:03d}.json']=(e['path'],e['file_sha256'])
    artifact_pins={'prepared.json':expected['prepared_sha256'],'numerical-plan.json':cpu['numerical_plan_file_sha256'],
        'engine-terminal.json':a['engine_terminal_sha256'],**{stage+'.'+kind: a[stage+'_'+('adapter' if kind=='adapter.pt' else 'checkpoint')+'_sha256']
         for stage in ('sft','final') for kind in ('adapter.pt','checkpoint.json')}}
    for n,h in artifact_pins.items():need(pin(h) and files[ARTIFACT_PATHS[n]]['sha256']==h,'accepted original training artifact pin differs: '+n)
    upstream_pins={TRAIN_ROOT+'/'+n:h for n,h in originals.items()}
    for n,h in artifact_pins.items():
        rel=ARTIFACT_PATHS[n];upstream_pins[(TRAIN_ROOT if rel.startswith('launch/') else TOKEN_ROOT)+'/'+rel.split('/',1)[1]]=h
    upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/terminal.json']=pins['realization-terminal']
    for path,h in descriptor_files.values():upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/'+path]=h
    return dict(metadata=m,pins=pins,world_pins=world_pins,descriptor_files=descriptor_files,artifact_pins=artifact_pins,upstream_pins=upstream_pins)


def make_packet(ctx,artifacts,descriptors):
    m,pins=ctx['metadata'],ctx['pins'];f=m['freeze'];closure=f['stage_closure']['stages']['CPUcampaign']
    need(set(artifacts)==set(ARTIFACTS) and set(descriptors)==set(ctx['descriptor_files']),'exact supplied opaque artifact roster')
    need(all(sha(v)==ctx['artifact_pins'][n] for n,v in artifacts.items())
        and all(sha(v)==ctx['descriptor_files'][n][1] for n,v in descriptors.items()),'original opaque input bytes differ')
    data={};origins={}
    for name,h in closure['input_source_files'].items():
        candidates=[n for n,v in f['source_pins'].items() if Path(n).name==name and v==h]
        need(len(candidates)==1,'unique original source origin required: '+name);origin=candidates[0]
        value=committed(SOURCE_COMMIT,origin);need(sha(value)==h,'captured full-commit source differs: '+name)
        data[name]=value;origins[name]=origin
    need(sha(data[ENTRY])==ENTRY_SHA and closure['entrypoint']==ENTRY and closure['entrypoint_sha256']==ENTRY_SHA
        and len(closure['source_names'])==49 and len(data)==51,'exact frozen CPUcampaign source closure differs')
    need(f['stage_closure']['source_pins']==f['source_pins'] and len(f['source_pins'])==67,'full frozen source closure differs')
    for n,h in f['source_pins'].items():need(sha(committed(SOURCE_COMMIT,n))==h,'original67-source full commit differs: '+n)
    proposal_raw=committed(SOURCE_COMMIT,f['original_proposal']['path'])
    need(sha(proposal_raw)==f['original_proposal']['sha256'],'original proposal bytes differ')
    proposal=decode(proposal_raw);need(digest(proposal)==f['original_proposal']['semantic_sha256'],'original proposal semantics differ')
    need([w['world_id'] for w in proposal['development_worlds']]==list(range(202611070100,202611070124)), 'unchanged24 ordered worlds required')
    bindings=[]
    for w in proposal['development_worlds']:
        public=w['public']
        need(public['initial_actions']==['observe_all']*8 and public['initial_cost_units']==8 and public['total_budget']==32,'frozen charged initial/budget differs')
        need([e['arm'] for e in w['development_episodes']]==list(POLICIES[:4])+['numerical_ranker','llm_frozen','llm_sft','llm_dpo'],'all8 original arms required')
        for e in w['development_episodes'][:4]:
            binding=dict(version=literal(data['llm_intervention_screen_policy.py'],'VERSION'),proposal_version=proposal['version'],
                seed_prefix=e['policy_seed_prefix'],scope=w['scope'],world_id=w['world_id'],
                request_namespace=e['measurement_binding']['request_namespace'],arm=e['arm'],role='behavior',
                initial_actions=public['initial_actions'],initial_cost_units=8,total_budget=32,planning_draws=32)
            need(binding['seed_prefix']==['development',w['world_id'],'policy',binding['request_namespace'],'behavior'], 'public arm-specific namespace differs')
            bindings.append(dict(world_id=w['world_id'],policy_id=e['arm'],binding=binding,binding_sha256=digest(binding)))
    need(len(bindings)==96,'all96 numerical bindings required')
    data.update(artifacts);data.update(descriptors)
    data.update({'proposal.json':proposal_raw,'scientific-freeze.json':raw(f),
        'realization-terminal.json':ctx['supplied']['realization-terminal'],'realization-acceptance.json':ctx['supplied']['realization-acceptance'],
        'gpu-training-acceptance.json':ctx['supplied']['gpu-training-acceptance'],
        'cpu-preparation-acceptance.json':ctx['supplied']['cpu-preparation-acceptance'],'numeric-bindings.json':raw(bindings)})
    # Preserve original freeze bytes, not a reserialization with the same meaning.
    data['scientific-freeze.json']=ctx['supplied']['freeze']
    service_source=data['llm_intervention_model_service.py'];service_names=literal(service_source,'SOURCE_NAMES')
    diagnostic=data['qualify_llm_intervention_runtime_metadata.py'];bootstrap=literal(diagnostic,'BOOTSTRAP')
    need(literal(service_source,'RUNTIME')==f['runtime'] and literal(diagnostic,'RUNTIME_SOURCE')==f['runtime_source'], 'qualified runtime identity differs')
    stages={s['stage']:s for s in f['stages']};gpuout=stages['gpu_campaign']['remote_root']+'/gpu'
    specs={};services={};identities={};inputroot=ROOT+'/inputs'
    # Prospective service metadata only; services never start in the CPU allocation.
    limits=dict(startup_seconds=120,request_seconds=30,idle_seconds=30,
        total_seconds=stages['gpu_campaign']['deadlines']['session_cap_seconds'],max_requests=576)
    for role in LLM:
        stage='sft' if role=='sft_llm' else 'final';adapter=None
        if role!='frozen_llm':adapter=dict(file=stage+'.adapter.pt',sha256=ctx['artifact_pins'][stage+'.adapter.pt'],
            training_receipt=stage+'.checkpoint.json',training_receipt_sha256=ctx['artifact_pins'][stage+'.checkpoint.json'],training_scope='policy_train')
        freeze=dict(schema=literal(service_source,'VERSION'),scope='development',model_directory=f['model_directory'],model_files=f['model_files'],
            sources={n:sha(data[n]) for n in service_names},runtime=f['runtime'],python_version=[3,10,19],recipe=literal(service_source,'RECIPE'),
            treatment='frozen' if adapter is None else 'sft' if role=='sft_llm' else 'dpo',adapter=adapter,limits=limits)
        data[role+'.freeze.json']=raw(freeze)
        contract=dict(schema='ace-campaign-staged-launch-v1',policy_id=role,entry=inputroot+'/llm_intervention_model_service.py',
            entry_sha256=sha(data['llm_intervention_model_service.py']),freeze=inputroot+'/'+role+'.freeze.json',freeze_sha256=sha(data[role+'.freeze.json']),
            worker_output=gpuout+'/services/'+role+'/worker',launch_output=gpuout+'/services/'+role+'/launch',
            binding_path=gpuout+'/stage-binding.json',stage_path=gpuout+'/stage.json',archive_sha256=f['runtime_archive_sha256'],
            manifest_sha256=f['runtime_manifest_sha256'],bootstrap=bootstrap,bootstrap_sha256=sha(bootstrap.encode()))
        data[role+'.launch.json']=raw(contract)
        identity=dict(model_sha256=digest(f['model_files']),source_sha256=digest(freeze['sources']),
            adapter_sha256=adapter['sha256'] if adapter else digest(dict(treatment='frozen',adapter=None)),
            numeric_recipe=freeze['recipe'],numeric_recipe_sha256=digest(freeze['recipe']))
        identities[role]=digest(identity);services[role]=dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=576)
        command=[f['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,inputroot+'/llm_intervention_campaign_launch.py',
            sha(data['llm_intervention_campaign_launch.py']),'--contract',inputroot+'/'+role+'.launch.json','--contract-sha256',sha(data[role+'.launch.json'])]
        specs[role]=dict(policy_id=role,kind='persistent',command=command,timeout_seconds=35,identity_sha256=digest(identity),policy_provenance_sha256=sha(data[role+'.freeze.json']))
    provenance=dict(schema='ace-scientific-cpu-campaign-input-lineage-v1',source_commit=SOURCE_COMMIT,authority_pins=pins,
        worlds=ctx['world_pins'],policy_ids=list(POLICIES),cpu_policy_ids=list(POLICIES[:5]),expected_training=m['expected-training'],
        opaque_artifact_pins=ctx['artifact_pins'],cpu_output=ROOT+'/cpu',future_gpu_output=gpuout,scientific_utility_qualified=False)
    p=dict(schema=VERSION,scope='development',source_commit=SOURCE_COMMIT,files={n:sha(v) for n,v in sorted(data.items())},
        source_names=closure['source_names'],cpu_output=ROOT+'/cpu',archive=dict(path=f['runtime_archive_path'],sha256=f['runtime_archive_sha256']),
        manifest=dict(path=f['runtime_manifest_path'],sha256=f['runtime_manifest_sha256']),runtime_source=f['runtime_source'],runtime=f['runtime'],
        model_directory=f['model_directory'],model_files=f['model_files'],resources=RESOURCES,deadlines=DEADLINES,
        numeric_timeout_seconds=55,numeric_work_seconds=50,protocol_source_pins=proposal['source_pins'],freeze_sha256=pins['freeze'],
        namespace_admission_sha256=pins['namespace-admission'],world_pins=ctx['world_pins'],expected_training=m['expected-training'],
        numerical_plan_sha256=m['cpu-preparation-acceptance']['numerical_plan_sha256'],persistent_specs=specs,services=services,service_identity_pins=identities,
        outer_provenance_receipt_sha256=sha(raw(provenance)))
    need(set(p)==literal(data[ENTRY],'PACKET_KEYS') and all(p['files'][n]==h for n,h in closure['helper_pins'].items()),'exact production packet/helper contract differs')
    need(len(data)==95 and all(len(v)<=(16*1024**2 if n.endswith('.pt') else 256*1024**2) for n,v in data.items())
        and sum(map(len,data.values()))<=512*1024**2,'exact bounded captured input roster required')
    return p,data,bootstrap,provenance


@contextlib.contextmanager
def coordinator_locks():
    handles=[]
    try:
        for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=BASE/n;need(path==path.resolve(),'lock alias forbidden')
            f=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        for f in reversed(handles):fcntl.flock(f,fcntl.LOCK_UN);f.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in METADATA:ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--training-remote-custody',required=True);ap.add_argument('--realization-task',required=True)
    args=ap.parse_args()
    with coordinator_locks():
        supplied={n:read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256'),256*1024**2) for n in METADATA}
        ctx=admit(supplied);ctx['supplied']=supplied
        need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'existing namespace differs')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        training=Path(args.training_remote_custody).resolve(strict=True);task=Path(args.realization_task).resolve(strict=True)
        artifacts={n:read(training/ARTIFACT_PATHS[n],h,16*1024**2 if n.endswith('.pt') else 256*1024**2) for n,h in ctx['artifact_pins'].items()}
        descriptors={n:read(task/path,h) for n,(path,h) in ctx['descriptor_files'].items()}
        stage,data,bootstrap,provenance=make_packet(ctx,artifacts,descriptors);stage_raw=raw(stage)
        admission=dict(schema=VERSION+'/admission',packet_sha256=sha(stage_raw),source_commit=SOURCE_COMMIT,freeze_sha256=ctx['pins']['freeze'],
            namespace_admission_sha256=ctx['pins']['namespace-admission'],stage='matched_ranker_and_development_cpu',output=stage['cpu_output'],resources=RESOURCES,deadlines=DEADLINES,
            full_contract_frozen_before_first_scientific_draw=True,source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,
            independent_acceptances_verified=True,retry_allowed=False)
        files={'inputs/'+n:v for n,v in data.items()};files.update({'inputs/packet.json':stage_raw,'inputs/admission.json':raw(admission),'input-lineage.json':raw(provenance)})
        for n,v in supplied.items():files[metadata_name(n)]=v
        for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
        command=[stage['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
            '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(stage_raw),'--source-commit',SOURCE_COMMIT,
            '--admission',ROOT+'/inputs/admission.json','--admission-sha256',sha(raw(admission))]
        files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_campaign_cpu01',
            '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M','#SBATCH --time=01:40:00','#SBATCH --no-requeue',
            '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
            "export CUDA_VISIBLE_DEVICES=''",'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL','exec '+shlex.join(command),''])).encode()
        base=ctx['metadata']['baseline'];outer=dict(schema='ace-scientific-cpu-campaign-launch-v1',source_commit=SOURCE_COMMIT,remote_root=ROOT,
            local_root=str(LOCAL),external_root=str(EXT),resources=RESOURCES,deadlines=DEADLINES,new_input_files={n:sha(v) for n,v in sorted(files.items())},
            baseline_sha256=ctx['pins']['baseline'],training_job_id=ctx['metadata']['gpu-training-acceptance']['job_id'],
            training_acceptance_sha256=ctx['pins']['gpu-training-acceptance'],namespace_claim_sha256=ctx['pins']['namespace-claim'],
            operational_amendment_sha256=ctx['pins']['operational-amendment'],original_input_pins=ctx['upstream_pins'],
            cpu_output=stage['cpu_output'],future_gpu_output=provenance['future_gpu_output'],worlds=24,cells=192,policy_ids=list(POLICIES),cpu_policy_ids=list(POLICIES[:5]),
            admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],preparation_allowance_cpu_seconds=3600,
                candidate_cpu_reservation=12000,candidate_gpu_reservation=0,total_cpu_seconds=base['closed_cpu_seconds']+15600,total_gpu_seconds=base['closed_gpu_slice_seconds'],
                ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,prior_jobs=44,prior_journals=90),submitted=False,scientific_qualified=False)
        files['packet.json']=raw(outer);need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'exclusive packet destinations required; never retry')
        claimed=[]
        try:
            for root in (EXT,LOCAL):
                root.mkdir(parents=True,mode=0o700,exist_ok=False);claimed.append(root);(root/'inputs').mkdir(mode=0o700)
                for n,v in files.items():write(root/n,v)
            for n,v in files.items():need(read(LOCAL/n,maximum=256*1024**2)==read(EXT/n,maximum=256*1024**2)==v,'two complete packet copies differ')
            need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'namespace changed during build')
            read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
            print(raw(dict(packet_sha256=sha(files['packet.json']),stage_sha256=sha(stage_raw),submitted=False)).decode())
        except BaseException as exc:
            for root in claimed:
                try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
                except BaseException:pass
            raise


if __name__=='__main__':main()
