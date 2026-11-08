"""First scientific GPU training packet, accepted saved CPU tokens only.
Authoring is not execution/admission. All future acceptance, inventory and baseline pins are explicit.
No project imports, tokenizer invocation, model load or scientific environment calls.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,os,re,shlex,stat,subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_scientific_training_gpu_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-scientific-training-gpu-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ENTRY='supervise_llm_intervention_scientific_training_gpu.py'
ENTRY_SHA='55be5ba74fa16f3e05d5632a1d0ff4ac1b0a3187507849e82d6594810f3545fd'
WORKER='llm_intervention_scientific_training_gpu_worker.py'
WORKER_SHA='ab2c999a35b247e03df87dce9f19d771c50f84747381bfbc0ed980e9c832a878'
VERSION='ace-scientific-training-gpu-supervisor-v1'
RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2700,gpus=1,gpu_type='a100_3g.20gb')
DEADLINES=dict(startup=240,worker=2620,soft=2630,cleanup=2645,hard=2670,progress=180,telemetry=90)
# Reference bindings only: the collection failure may require a separately reviewed successor.
# No successor is silently adopted. Rebind these/source closure only after explicit reviewed pins.
FIXED={'freeze':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d',
 'freeze-review':'088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416',
 'adoption':'ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07',
 'namespace-admission':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
 'recovery-authority':'ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50'}
METADATA=tuple(FIXED)+('namespace-claim','cpu-packet','cpu-acceptance','cpu-inventory','cpu-disposition',
 'cpu-launch-packet','cpu-result','cpu-terminal','cpu-report','collection-acceptance','baseline','previous-baseline','site-readiness')
PATH_KEYS=('cpu-packet-relative','cpu-prepared-relative','cpu-terminal-relative')
ROSTER=[f'policy_train/{wid}/snapshot-{i:03d}' for wid in range(202611070000,202611070032) for i in range(5)]


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


def authenticate(supplied,prepared_raw,prepared_pin,plan_pin,paths):
    need(set(supplied)==set(METADATA) and set(paths)==set(PATH_KEYS),'complete explicit caller bindings required')
    pins={n:sha(v) for n,v in supplied.items()}
    need(all(pins[n]==h for n,h in FIXED.items()),'adopted freeze/source/namespace metadata differs')
    m={n:decode(v) for n,v in supplied.items() if n not in ('freeze-review','cpu-report','site-readiness')}
    f=m['freeze'];cpu=m['cpu-packet'];a=m['cpu-acceptance'];base=m['baseline'];old=m['previous-baseline']
    need(f['source_commit']==SOURCE_COMMIT and supplied['cpu-report'].strip() and supplied['site-readiness'].strip(),
         'independent CPU review and site-readiness evidence required')
    need(raw(m['adoption'])==raw(dict(schema='ace-scientific-contract-adoption-v1',status='adopted',
        freeze_sha256=pins['freeze'],review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True)),
        'exact independent adoption required')
    stages={s['stage']:s for s in f['stages']};stage=stages['train']
    need(raw(stage)==raw(dict(stage='train',entry=ENTRY,entry_sha256=ENTRY_SHA,cpus=4,memory_mib=15360,gpus=1,
        gpu_type='a100_3g.20gb',wall_seconds=2700,deadlines=DEADLINES,account='ucb736_asc2',partition='aa100',qos='gpu-testing',
        nodes=1,tasks=1,requeue=False,max_restarts=0,remote_root=ROOT,
        repository_root=str(LOCAL.parent.relative_to(REPO)),exclusive_external_root=str(EXT)))
        and stages['tokenize']['remote_root']==CPU_ROOT,'fixed train/tokenization stage contract differs')
    claim=m['namespace-claim'];ns=m['namespace-admission'];auth=m['recovery-authority']
    need(claim['freeze_sha256']==pins['freeze'] and claim['namespace_admission_sha256']==pins['namespace-admission']
        and claim['world_ids']==ns['world_ids'] and claim['local_registry']==str(LOCAL_REGISTRY)
        and claim['remote_registry']==REMOTE_REGISTRY and claim['recovery_authority_sha256']==pins['recovery-authority']
        and claim['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA
        and claim['retry_allowed'] is False and claim['replacement_worlds'] is False
        and auth['successor_freeze_sha256']==pins['freeze'],'existing accepted recovery namespace required')
    expected=dict(schema='ace-scientific-training-preparation-acceptance-v1',status='accepted',scope='policy_train',
        required_findings=0,cpu_packet_sha256=pins['cpu-packet'],prepared_sha256=prepared_pin,plan_sha256=plan_pin,
        freeze_sha256=pins['freeze'],namespace_admission_sha256=pins['namespace-admission'],
        complete_custody_verified=True,runtime_qualified_for_this_cpu_execution=True,owned_process_cleanup_verified=True,
        training_execution_qualified=False,scientific_qualified=False)
    extra={'job_id','numerical_plan_file_sha256','numerical_plan_sha256','exposure_sha256'}
    need(set(a)==set(expected)|extra and all(raw(a[k])==raw(v) for k,v in expected.items())
        and all(pin(a[k]) for k in extra-{'job_id'}) and type(a['job_id']) is str and re.fullmatch('[1-9][0-9]*',a['job_id']),
        'exact independently accepted scientific CPU preparation required')
    need(pin(plan_pin) and pin(prepared_pin) and sha(prepared_raw)==prepared_pin,'independent original prepared/plan pins required')
    need(cpu['schema']=='ace-scientific-training-preparation-cpu-v1' and cpu['scope']=='policy_train'
        and cpu['source_commit']==SOURCE_COMMIT,'original accepted CPU packet differs')
    for key in ('runtime','runtime_source','model_files'):
        need(raw(cpu[key])==raw(f[key]),'frozen CPU/GPU runtime/model identity differs: '+key)
    for key in ('archive','manifest'):
        need(cpu[key]==dict(path=f['runtime_'+key+'_path'],sha256=f['runtime_'+key+'_sha256']),
             'CPU staged runtime input differs: '+key)
    collection=m['collection-acceptance']
    need(pins['collection-acceptance']==cpu['files']['collection-acceptance.json']
        and collection['schema']=='ace-scientific-collection-acceptance-v1' and collection['status']=='accepted'
        and collection['required_findings']==0 and collection['scope']=='policy_train'
        and collection['complete_custody_verified'] is True and collection['accounting_complete'] is True
        and collection['collection']==cpu['collection'] and cpu['collection']['freeze_sha256']==pins['freeze'],
        'complete original accepted collection binding differs')
    # Metadata only until independent CPU acceptance has been authenticated above.
    prepared=decode(prepared_raw);plan=prepared['training_plan'];exposure=prepared['exposure'];d=plan['dpo_steps']
    need(prepared_raw==(json.dumps(prepared,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        and digest(plan)==prepared['training_plan_sha256']==plan_pin and digest(exposure)==prepared['exposure_sha256']==a['exposure_sha256']
        and plan['outer_provenance_receipt_sha256']==pins['cpu-packet'] and prepared['tokenizer_sha256']==digest(f['model_files']),
        'original prepared canonical bytes, plan/exposure/tokenizer identity differ')
    need(type(d) is int and 1<=d<=128 and plan['scope']=='policy_train' and plan['execution_kind']=='qwen_screen_singletons'
        and plan['sft_steps']==160 and plan['sft_rows']==160 and plan['dpo_rows']==2*d and plan['one_pass_sft'] is True
        and plan['checkpoint_steps']==[159,159+d] and plan['numeric_recipe']['seed']==109
        and plan['numeric_recipe']['scoring_path']=='canonical_selected_unpadded'
        and len(plan['records'])==160 and [r['record_id'] for r in plan['records']]==ROSTER
        and all(type(r['prompt_ids']) is list and 0<len(r['prompt_ids'])<=4744 for r in plan['records'])
        and exposure['counts']['retained_pairs']==3584 and exposure['counts']['dpo_steps']==d,
        'saved scientific160SFT+D1..128/singleton/no-truncation envelope differs')
    for key,name in (('collection_terminal_sha256','terminal.json'),('collection_inventory_sha256','inventory.json'),('collection_plan_sha256','plan.json')):
        need(prepared[key]==cpu['collection']['files'][name],'original collection lineage differs')
    result=m['cpu-result'];outer=m['cpu-launch-packet'];disp=m['cpu-disposition'];terminal=m['cpu-terminal']
    job=a['job_id']
    need(result['job_id']==job and result['returncode']==0 and result['ambiguous'] is False
        and outer['remote_root']==CPU_ROOT and outer['source_commit']==SOURCE_COMMIT
        and disp['job_id']==job and disp['remote_root']==CPU_ROOT and disp['packet_sha256']==pins['cpu-launch-packet']
        and disp['submission_result_sha256']==pins['cpu-result'] and disp['custody_bytes_verified'] is True
        and disp['current_lineage_verified'] is True and disp['errors']==[]
        and disp['exact_prior_accounting_matched'] is True and disp['exact_prior_and_current_journals_matched'] is True
        and disp['observed_related_jobs']==base['terminal_jobs']
        and disp['observed_submission_journals']==base['submission_journals'],
        'complete actual tokenization terminal custody/job required')
    transport=m['cpu-inventory'];need(transport['returncode']==0,'successful independently pinned CPU inventory required')
    observed=decode(transport['stdout']);files=observed['files']
    need(observed['terminal_gate'] is True and observed['root_present'] is True and not observed.get('observation_error')
        and observed['parent_accounting'][0]==job and len(files)==disp['raw_files']
        and sum(v['bytes'] for v in files.values())==disp['raw_bytes'],'complete CPU inventory terminal/count/byte joins differ')
    assigned={n:[] for n in ('ROOT','JOB')}
    for node in ast.parse(transport['python_source']).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in assigned:
                    need(isinstance(node.value,ast.Constant),'literal captured CPU root/job required');assigned[target.id].append(node.value.value)
    need(assigned==dict(ROOT=[CPU_ROOT],JOB=[job]),'captured CPU inventory root/job differs')
    for n in paths:relative(paths[n])
    expected_files={paths['cpu-prepared-relative']:prepared_pin,paths['cpu-packet-relative']:pins['cpu-packet'],
        paths['cpu-terminal-relative']:pins['cpu-terminal'],'launch/packet.json':pins['cpu-launch-packet'],
        'launch/submission.intent.json':result['intent_sha256'],'launch/submission.result.json':pins['cpu-result']}
    need(len(expected_files)==6 and paths['cpu-prepared-relative'].endswith('/session/worker/artifacts/llm-prepared.json')
        and paths['cpu-terminal-relative']==str(Path(paths['cpu-prepared-relative']).with_name('terminal.json')),
        'original tokenization worker artifact paths required')
    cpu_dir=Path(paths['cpu-packet-relative']).parent
    need(outer['new_input_files'][paths['cpu-packet-relative'].removeprefix('launch/')]==pins['cpu-packet'],
        'original CPU inner/outer packet join differs')
    for n,h in cpu['files'].items():
        need(Path(n).name==n and pin(h),'safe CPU input roster required');expected_files[str(cpu_dir/n)]=h
    need(all(files[n]['sha256']==h for n,h in expected_files.items()),'CPU prepared/terminal/source/input/journal inventory pins differ')
    need(terminal['schema']=='ace-scientific-training-preparation-cpu-v1' and terminal['status']=='cpu_preparation_complete'
        and terminal['packet_sha256']==pins['cpu-packet'] and terminal['llm_prepared_file_sha256']==prepared_pin
        and terminal['training_plan_sha256']==plan_pin and terminal['exposure_sha256']==a['exposure_sha256']
        and terminal['numerical_plan_file_sha256']==a['numerical_plan_file_sha256']
        and terminal['numerical_plan_sha256']==a['numerical_plan_sha256']
        and terminal['records']==160 and terminal['retained_pairs']==3584 and terminal['sft_steps']==160 and terminal['dpo_steps']==d,
        'accepted CPU worker report pins/counts differ')
    for flag in ('model_weights_loaded','explicit_cuda_api_called','training_execution_qualified','scientific_qualified','retry_allowed'):
        need(terminal[flag] is False,'CPU preparation flag differs: '+flag)
    need(all(terminal[n]==0 for n in ('inference_calls','environment_responses','training_updates')),'CPU preparation performed forbidden work')
    need(type(old['terminal_jobs']) is int and old['terminal_jobs']>0
        and type(old['submission_journals']) is int and old['submission_journals']>0
        and base['terminal_jobs']==old['terminal_jobs']+1 and base['submission_journals']==old['submission_journals']+2
        and base['outstanding_reservation']==old['outstanding_reservation']==0
        and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
        and base['input_pins'].get('predecessor')==pins['previous-baseline']
        and base['input_pins'].get('acceptance')==pins['cpu-acceptance']
        and base['input_pins'].get('disposition')==pins['cpu-disposition'],'actual accepted baseline successor required, never synthesize')
    prior=[r for r in base['related_accounting'] if r[0]!=job];new=[r for r in base['related_accounting'] if r[0]==job]
    need(len(prior)==old['terminal_jobs'] and len(new)==1 and normalized(prior)==normalized(old['related_accounting'])
        and normalized(new)==normalized([observed['parent_accounting']]),'all actual prior rows and new CPU job must remain')
    allocation=disp['allocation'];row=normalized(new)[0]
    need(row[2:4]==['COMPLETED','0:0'] and allocation['resource_contract_matched'] is True and allocation['errors']==[]
        and allocation['allocated_cpus']==row[4]==stages['tokenize']['cpus']
        and allocation['elapsed_seconds']==row[5] and 0<=row[5]<=stages['tokenize']['wall_seconds']
        and allocation['gpu_count']==allocation['gpu_slice_seconds']==0
        and allocation['allocated_cpu_seconds']==row[4]*row[5],'actual CPU-only preparation allocation differs')
    journals={**old['journal_pins'],CPU_ROOT+'/submission.intent.json':result['intent_sha256'],CPU_ROOT+'/submission.result.json':pins['cpu-result']}
    need(len(old['journal_pins'])==old['submission_journals'] and base['journal_pins']==journals
        and len(journals)==base['submission_journals'] and observed['journal_pins']==journals,
        'complete original journal pins differ')
    need(base['closed_cpu_seconds']==disp['closed_cpu_seconds']==old['closed_cpu_seconds']+row[4]*row[5]
        and base['closed_gpu_slice_seconds']==disp['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']
        and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600,'actual CPU/GPU costs differ')
    remaining=f['stages'][next(i for i,s in enumerate(f['stages']) if s['stage']=='train'):]
    need(base['closed_cpu_seconds']+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining)<=57600
        and base['closed_gpu_slice_seconds']+sum(s['gpus']*s['wall_seconds'] for s in remaining)<=7200,
        'actual costs plus frozen remaining stage caps exceed budget')
    return dict(metadata=m,pins=pins,prepared=prepared,dpo_steps=d,
        upstream_pins={CPU_ROOT+'/'+n.removeprefix('launch/'):h for n,h in expected_files.items()})
def make_packet(ctx,prepared_raw,prepared_pin,plan_pin,supplied):
    f=ctx['metadata']['freeze'];cpu=ctx['metadata']['cpu-packet'];closure=f['stage_closure']['stages']['train']
    source=committed(SOURCE_COMMIT,'scripts/research/'+ENTRY);need(sha(source)==ENTRY_SHA,'reviewed scientific training entry differs')
    names=literal(source,'SOURCE_NAMES');extra=literal(source,'EXTRA_DATA')
    need(sorted(names)==closure['source_names'] and len(names)==29 and len(set(names))==29
        and raw(literal(source,'RESOURCES'))==raw(RESOURCES) and raw(literal(source,'DEADLINES'))==raw(DEADLINES),
        'exact frozen29-source training/resource/deadline interface differs')
    need(set(extra)=={'prepared.json','cpu-packet.json','cpu-acceptance.json','llm_intervention_runtime_archive.py'},
        'exact scientific saved-input interface differs')
    data={n:committed(SOURCE_COMMIT,'scripts/research/'+n) for n in set(names)|{'llm_intervention_runtime_archive.py'}}
    need(closure['input_source_files']=={n:sha(v) for n,v in data.items()}
        and sha(data[WORKER])==WORKER_SHA and all(sha(data[n])==h for n,h in literal(source,'INDEPENDENT').items()),
        'complete reviewed scientific source/helper closure differs')
    need(f['stage_closure']['source_pins']==f['source_pins'] and len(f['source_pins'])==67,'exact full frozen closure required')
    for n,h in f['source_pins'].items():
        need(not Path(n).is_absolute() and '..' not in Path(n).parts and pin(h),'safe frozen source path required')
        need(sha(committed(SOURCE_COMMIT,n))==h,'full captured source metadata differs: '+n)
    prep=f['stage_closure']['stages']['tokenize']
    need(cpu['source_names']==prep['source_names'] and all(cpu['files'][n]==h for n,h in prep['input_source_files'].items()),
        'accepted CPU source closure differs from same freeze')
    need(set(literal(source,'PREPARATION_SOURCES'))==set(closure['prepared_artifact_source_roster'])
        and all(cpu['files'][n]==sha(v) for n,v in data.items() if n in cpu['files']),
        'shared CPU/GPU source bytes differ')
    data.update({'prepared.json':prepared_raw,'cpu-packet.json':supplied['cpu-packet'],'cpu-acceptance.json':supplied['cpu-acceptance']})
    need(set(data)==set(names)|set(extra) and all(len(v)<= (256*1024**2 if n=='prepared.json' else 8*1024**2) for n,v in data.items()),
        'bounded exact supervisor input membership required')
    stage=dict(schema=VERSION,scope='policy_train',source_commit=SOURCE_COMMIT,
        files={n:sha(v) for n,v in sorted(data.items())},source_names=sorted(names),resources=RESOURCES,deadlines=DEADLINES,
        archive=cpu['archive'],manifest=cpu['manifest'],model_directory=f['model_directory'],model_files=f['model_files'],
        runtime=f['runtime'],runtime_source=f['runtime_source'],plan_sha256=plan_pin,
        freeze_sha256=ctx['pins']['freeze'],namespace_admission_sha256=ctx['pins']['namespace-admission'])
    need(stage['runtime']==literal(data['llm_intervention_model_service.py'],'RUNTIME')
        and stage['runtime_source']==literal(data['qualify_llm_intervention_runtime_metadata.py'],'RUNTIME_SOURCE'),
        'captured runtime constants differ')
    return stage,data,literal(data['qualify_llm_intervention_runtime_metadata.py'],'BOOTSTRAP')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in (*METADATA,'prepared'):
        ap.add_argument('--'+n,required=True);ap.add_argument('--'+n+'-sha256',required=True)
    ap.add_argument('--plan-sha256',required=True)
    for n in PATH_KEYS:ap.add_argument('--'+n,required=True)
    args=ap.parse_args();supplied={n:read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256')) for n in METADATA}
    prepared_raw=read(args.prepared,args.prepared_sha256,256*1024**2)
    paths={n:getattr(args,n.replace('-','_')) for n in PATH_KEYS}
    ctx=authenticate(supplied,prepared_raw,args.prepared_sha256,args.plan_sha256,paths)
    need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'existing local namespace changed')
    read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
    stage,data,bootstrap=make_packet(ctx,prepared_raw,args.prepared_sha256,args.plan_sha256,supplied)
    stage_raw=raw(stage);output=ROOT+'/training'
    # Frozen conditional admission: caller-owned accepted CPU/freeze/readiness pins; the launcher must
    # independently recheck live site/accounting/source/model-input state before making it runnable.
    admission=dict(schema='ace-scientific-training-gpu-admission-v1',status='admitted',scope='policy_train',
        packet_sha256=sha(stage_raw),cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],prepared_sha256=args.prepared_sha256,
        plan_sha256=args.plan_sha256,freeze_sha256=ctx['pins']['freeze'],namespace_admission_sha256=ctx['pins']['namespace-admission'],
        resources=RESOURCES,deadlines=DEADLINES,required_findings=0,source_runtime_scheduler_admitted=True,
        scientific_protocol_frozen=True,namespace_owned=True,output=output,site_readiness_sha256=ctx['pins']['site-readiness'])
    files={'inputs/'+n:v for n,v in data.items()}
    files.update({'inputs/training.json':stage_raw,'inputs/admission.json':raw(admission),'cpu-paths.json':raw(paths)})
    for n,v in supplied.items():files[n+('.md' if n in ('freeze-review','cpu-report') else '.json')]=v
    for n in ('build_packet.py','launch_once.py','monitor_once.py'):files[n]=read(Path(__file__).resolve().with_name(n))
    command=[stage['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,
        '--packet',ROOT+'/inputs/training.json','--packet-sha256',sha(stage_raw),'--admission',ROOT+'/inputs/admission.json',
        '--admission-sha256',sha(raw(admission)),'--output',output]
    files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_train01',
        '#SBATCH --account=ucb736_asc2','#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=15360M',
        '#SBATCH --gres=gpu:a100_3g.20gb:1','#SBATCH --time=00:45:00','#SBATCH --no-requeue',
        '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL','exec '+shlex.join(command),''])).encode()
    base=ctx['metadata']['baseline']
    outer=dict(schema='ace-scientific-training-gpu-launch-v1',source_commit=SOURCE_COMMIT,remote_root=ROOT,
        resources=RESOURCES,deadlines=DEADLINES,new_input_files={n:sha(v) for n,v in sorted(files.items())},
        cpu_job_id=ctx['metadata']['cpu-acceptance']['job_id'],cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],
        prepared_sha256=args.prepared_sha256,plan_sha256=args.plan_sha256,baseline_sha256=ctx['pins']['baseline'],
        namespace_claim_sha256=ctx['pins']['namespace-claim'],original_cpu_inputs=ctx['upstream_pins'],
        admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],
            preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=10800,candidate_gpu_reservation=2700,
            total_cpu_seconds=base['closed_cpu_seconds']+14400,total_gpu_seconds=base['closed_gpu_slice_seconds']+2700,
            ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,
            prior_jobs=base['terminal_jobs'],prior_journals=base['submission_journals']),
        fresh_adapter_required=True,prior_adapter_restored=False,seed=109,sft_steps=160,dpo_steps=ctx['dpo_steps'],
        tokenizer_calls_on_gpu=0,new_environment_responses=0,downloads=False,installs=False,submitted=False,training_qualified=False,
        sizing='One measured20GB A100 slice, exact4CPU15360MiB; all saved prompts<=4744. Fixed2700sec bound, no scaling/throughput claim. '
               'Two independent GPU work signals by240sec; progress180/telemetry90; inherited absolute deadlines and owned cleanup.')
    files['packet.json']=raw(outer);need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'exclusive packet destinations required; no retry')
    created=[]
    try:
        for root in (EXT,LOCAL):
            root.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(root);(root/'inputs').mkdir(mode=0o700)
            for n,v in files.items():write(root/n,v)
        for n,v in files.items():need(read(LOCAL/n,maximum=256*1024**2)==read(EXT/n,maximum=256*1024**2)==v,'complete two-copy packet differs')
        need(read(LOCAL_REGISTRY,ctx['pins']['namespace-claim'])==supplied['namespace-claim'],'existing claim changed during build')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        print(raw(dict(packet_sha256=sha(files['packet.json']),training_sha256=sha(stage_raw),admission_sha256=sha(raw(admission)),
                       submitted=False,training_qualified=False)).decode())
    except BaseException as exc:
        for root in created:
            try:write(root/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
            except BaseException:pass
        raise


if __name__=='__main__':main()
