"""Build one training-only collection packet from independently accepted realization.
No project callable execution, world generation, development payload reads or submission.
"""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, os, re, shlex, stat, subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
# Adopted metadata is fixed; actual realization acceptance/claim/baseline are required CLI inputs.
# Runtime locations below are assigned only after authentication in admit().
LOCAL=EXT=ROOT=REALIZATION_ROOT=LOCAL_REGISTRY=REMOTE_REGISTRY=None
FREEZE_SHA=SOURCE_COMMIT=ENTRY_SHA=PRIOR_BASELINE_SHA=None
EXPECTED_FREEZE_SHA='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
EXPECTED_SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
EXPECTED_ENTRY_SHA='e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8'
EXPECTED_COLLECTION_SHA='2bc872349f9fbfbafe1564b751cde899705f487f3bf822cce5518e0865bd9a4d'
EXPECTED_METADATA_PINS={
    'freeze':EXPECTED_FREEZE_SHA,
    'freeze-review':'088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416',
    'adoption':'ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07',
    'namespace-admission':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
    'recovery-authority':'ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50',
    'predecessor-baseline':'63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'}
ORIGINAL_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
ORIGINAL_LOCAL_REGISTRY=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
ORIGINAL_REMOTE_REGISTRY='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json'
ENTRY='supervise_llm_intervention_scientific_cpu.py'
VERSION='ace-scientific-cpu-supervisor-v1'
COLLECT='collect_llm_intervention_scientific_training_cpu.py'
INNER_VERSION='ace-scientific-training-collection-cpu-v1'
PROPOSAL='results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
PROPOSAL_SHA='8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SEMANTIC='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
PROBE='results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/probe.py'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,requeue=False,wall_seconds=900)
DEADLINES=dict(startup=120,work=850,cleanup=875,hard=890)
WORLD_IDS=list(range(202611070000,202611070032))
METADATA=('freeze','freeze-review','adoption','namespace-admission','namespace-claim','recovery-authority',
          'realization-acceptance','realization-disposition','realization-inventory','baseline','predecessor-baseline',
          'realization-packet','realization-result')


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


def realization_files(transport_raw,transport_pin,accepted,disposition,result,result_pin,root):
    """Authenticated saved inventory metadata only; no remote/program/payload execution."""
    need(pin(transport_pin) and sha(transport_raw)==transport_pin
         and accepted.get('inventory_transport_sha256')==transport_pin,
         'independent realization acceptance must bind exact inventory transport bytes')
    transport=decode(transport_raw)
    need(type(transport.get('returncode')) is int and transport['returncode']==0
         and type(transport.get('stdout')) is str,'successful saved realization inventory transport required')
    observation=decode(transport['stdout']);job=result['job_id']
    # The existing transport stores ROOT/JOB in its captured Python source, not stdout.
    # Parse only their literal assignments; never execute the captured controller.
    assignments={name:[] for name in ('ROOT','JOB')}
    for node in ast.parse(transport['python_source']).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in assignments:
                    need(isinstance(node.value,ast.Constant) and type(node.value.value) is str,
                         'literal inventory root/job binding required')
                    assignments[target.id].append(node.value.value)
    need(assignments==dict(ROOT=[root],JOB=[job]) and disposition['remote_root']==root
         and disposition['job_id']==accepted['job_id']==job,'saved inventory root/job binding differs')
    need(observation.get('terminal_gate') is True and observation.get('root_present') is True
         and not observation.get('observation_error'),'complete terminal inventory required')
    row=observation['parent_accounting']
    need(normalized([row])[0][:4]==[job,'ace_llm_scientific_realize02','COMPLETED','0:0'],
         'inventory actual terminal job differs')
    files=observation['files']
    need(type(files) is dict and all(type(n) is str and n and not Path(n).is_absolute()
         and '..' not in Path(n).parts and type(v) is dict and set(v)=={'bytes','sha256'}
         and type(v['bytes']) is int and v['bytes']>=0 and pin(v['sha256']) for n,v in files.items()),
         'complete safe file-byte inventory required')
    need(type(disposition['raw_files']) is int and type(disposition['raw_bytes']) is int
         and len(files)==disposition['raw_files'] and sum(v['bytes'] for v in files.values())==disposition['raw_bytes'],
         'inventory count/bytes differ from independently pinned disposition')
    need(disposition['output_root']==root+'/realization'
         and files['launch/realization/session/worker/task/terminal.json']['sha256']==accepted['realization_terminal_sha256']
         and files['launch/submission.intent.json']['sha256']==result['intent_sha256']
         and files['launch/submission.result.json']['sha256']==result_pin,
         'accepted realization terminal/journal inventory joins differ')
    return files


def admit(supplied):
    """Metadata only: independently accepted realization, existing namespace, actual next baseline."""
    need(set(supplied)==set(METADATA),'exact caller metadata required')
    pins={n:sha(v) for n,v in supplied.items()}
    need(all(pins[n]==h for n,h in EXPECTED_METADATA_PINS.items()),'exact adopted candidate03/recovery02 metadata required')
    m={n:decode(v) for n,v in supplied.items() if n!='freeze-review'}
    f=m['freeze'];a=m['realization-acceptance'];d=m['realization-disposition'];base=m['baseline']
    old=m['predecessor-baseline']
    ns=m['namespace-admission'];claim=m['namespace-claim'];rp=m['realization-packet'];rr=m['realization-result']
    need(f['schema']=='ace-first-trained-intervention-screen-contract-v1'
         and f['source_commit']==EXPECTED_SOURCE_COMMIT,'exact adopted source commit required')
    global LOCAL,EXT,ROOT,REALIZATION_ROOT,LOCAL_REGISTRY,REMOTE_REGISTRY,FREEZE_SHA,SOURCE_COMMIT,ENTRY_SHA,PRIOR_BASELINE_SHA
    FREEZE_SHA=pins['freeze'];SOURCE_COMMIT=f['source_commit'];PRIOR_BASELINE_SHA=pins['predecessor-baseline']
    stages={s['stage']:s for s in f['stages']}
    need(len(stages)==len(f['stages']),'unique frozen stages required')
    collect_stage=stages['collect'];realize_stage=stages['realize'];ENTRY_SHA=collect_stage['entry_sha256']
    ROOT=collect_stage['remote_root'];REALIZATION_ROOT=realize_stage['remote_root']
    LOCAL=REPO/collect_stage['repository_root']/'launch01';EXT=Path(collect_stage['exclusive_external_root'])
    LOCAL_REGISTRY=Path(ns['local_registry']);REMOTE_REGISTRY=ns['remote_registry']
    scratch=Path(ROOT).parent
    need(all(type(v) is str and Path(v).is_absolute() and '..' not in Path(v).parts for v in (ROOT,REALIZATION_ROOT,REMOTE_REGISTRY))
         and Path(ROOT).parent==Path(REALIZATION_ROOT).parent==Path(REMOTE_REGISTRY).parent==scratch
         and scratch.name=='results' and scratch.parent.name=='ACE'
         and ROOT!=REALIZATION_ROOT and LOCAL.is_relative_to(REPO/'results') and '..' not in LOCAL.parts
         and not Path(collect_stage['repository_root']).is_absolute() and EXT.parent==BASE
         and LOCAL_REGISTRY.parent==BASE and ENTRY_SHA==EXPECTED_ENTRY_SHA
         and collect_stage['entry']==ENTRY and realize_stage['entry']==ENTRY,
         'explicit bounded amended roots/entry required')
    need(f['admission_accounting']['baseline_sha256']==PRIOR_BASELINE_SHA,
         'amended freeze must retain independently accepted predecessor including failed attempts')
    need(f['source_pins']['scripts/research/'+ENTRY]==EXPECTED_ENTRY_SHA
         and f['source_pins']['scripts/research/'+COLLECT]==EXPECTED_COLLECTION_SHA,
         'exact reviewed CPU entry/collection path corrections required')
    expected_stage=dict(stage='collect',entry=ENTRY,entry_sha256=ENTRY_SHA,cpus=2,memory_mib=7680,gpus=0,
        gpu_type=None,wall_seconds=900,deadlines=DEADLINES,account='ucb736_asc2',partition='acpu',qos='cpu-normal',
        nodes=1,tasks=1,requeue=False,max_restarts=0,remote_root=ROOT,
        repository_root=str(LOCAL.parent.relative_to(REPO)),exclusive_external_root=str(EXT))
    need(raw([s for s in f['stages'] if s['stage']=='collect'])==raw([expected_stage]),'frozen collection stage differs')
    need(f['original_proposal']==dict(path=PROPOSAL,sha256=PROPOSAL_SHA,semantic_sha256=PROPOSAL_SEMANTIC)
         and f['training_world_ids']==WORLD_IDS and f['development_world_ids']==list(range(202611070100,202611070124)),
         'original proposal/32+24 roster differs')
    need(raw(m['adoption'])==raw(dict(schema='ace-scientific-contract-adoption-v1',status='adopted',
        freeze_sha256=FREEZE_SHA,review_sha256=pins['freeze-review'],required_findings=0,adopted_before_first_draw=True))
        and supplied['freeze-review'].strip(),'independent full-contract adoption required')
    need(ns['freeze_sha256']==FREEZE_SHA and ns['proposal_sha256']==PROPOSAL_SHA
         and ns['proposal_semantic_sha256']==PROPOSAL_SEMANTIC and ns['realization_root']==REALIZATION_ROOT
         and ns['world_ids']==dict(policy_train=WORLD_IDS,development=f['development_world_ids'])
         and ns['local_registry']==str(LOCAL_REGISTRY) and ns['remote_registry']==REMOTE_REGISTRY,
         'original namespace descriptor differs')
    authority=m['recovery-authority']
    need(authority['schema']=='ace-scientific-realization-recovery-authority-v1'
         and authority['successor_freeze_sha256']==FREEZE_SHA and authority['source_commit']==SOURCE_COMMIT
         and authority['recovery_root']==REALIZATION_ROOT and authority['accounting_sha256']==PRIOR_BASELINE_SHA
         and authority['local_recovery_registry']==str(LOCAL_REGISTRY)
         and authority['remote_recovery_registry']==REMOTE_REGISTRY
         and authority['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA
         and authority['one_submission_only'] is True and authority['automatic_retry_allowed'] is False
         and authority['replacement_worlds_allowed'] is False,'adopted recovery authority linkage differs')
    recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority'],
        original_namespace_claim_sha256=ORIGINAL_CLAIM_SHA,failed_job_id=authority['failed_job_id'])
    need(all(ns.get(k)==v for k,v in recovery_bindings.items()),'recovery namespace descriptor lineage differs')
    need(claim['schema']=='ace-scientific-screen-namespace-claim-v1'
         and claim['freeze_sha256']==FREEZE_SHA and claim['namespace_admission_sha256']==pins['namespace-admission']
         and claim['packet_sha256']==pins['realization-packet'] and claim['realization_root']==REALIZATION_ROOT
         and claim['world_ids']==ns['world_ids'] and claim['local_registry']==str(LOCAL_REGISTRY)
         and claim['remote_registry']==REMOTE_REGISTRY and claim['retry_allowed'] is False
         and claim['replacement_worlds'] is False,'independently accepted original namespace claim required')
    need(claim.get('claim_kind')=='explicit_pre_draw_recovery02'
         and all(claim.get(k)==v for k,v in recovery_bindings.items())
         and claim.get('failure_acceptance_sha256')==authority['failure_acceptance_sha256']
         and claim.get('no_draw_proof_sha256')==authority['no_draw_proof_sha256'],
         'accepted recovery02 claim must preserve original claim/failure/no-draw authority')
    need(rp['schema']=='ace-scientific-realization-launch-v1' and rp['remote_root']==REALIZATION_ROOT
         and rp['source_commit']==SOURCE_COMMIT and rp['new_input_files']['freeze.json']==FREEZE_SHA
         and rp['new_input_files']['namespace-admission.json']==pins['namespace-admission'],
         'original realization launch packet differs')
    need(rp['new_input_files'].get('recovery-authority.json')==pins['recovery-authority']
         and rp['recovery']['authority_sha256']==pins['recovery-authority']
         and rp['recovery']['original_namespace_claim_sha256']==ORIGINAL_CLAIM_SHA,
         'accepted realization packet recovery authority differs')
    job=rr.get('job_id')
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) is not None and rr.get('returncode')==0
         and job!=authority['failed_job_id']
         and rr.get('ambiguous') is False and pin(rr.get('intent_sha256')),'unambiguous new realization job/result required')
    need(a.get('status')=='accepted' and type(a.get('required_findings')) is int and a['required_findings']==0
         and a.get('complete_custody_verified') is True and a.get('freeze_sha256')==FREEZE_SHA
         and a.get('namespace_admission_sha256')==pins['namespace-admission'] and a.get('job_id')==job
         and a.get('disposition_sha256')==pins['realization-disposition']
         and a.get('namespace_claim_sha256')==pins['namespace-claim']
         and a.get('realization_packet_sha256')==pins['realization-packet']
         and pin(a.get('realization_terminal_sha256')),'independent actual realization acceptance required')
    need(d['schema']=='ace-scientific-realization-terminal-custody-v1' and d['job_id']==job
         and d['packet_sha256']==pins['realization-packet'] and d['submission_result_sha256']==pins['realization-result']
         and d['custody_bytes_verified'] is True and d['current_lineage_verified'] is True and d['errors']==[]
         and d['input_pin_mismatches']==[] and d['exact_prior_accounting_matched'] is True
         and d['exact_prior_and_current_journals_matched'] is True
         and d['observed_related_jobs']==old['terminal_jobs']+1
         and d['observed_submission_journals']==old['submission_journals']+2
         and d['namespace_local_sha256']==d['namespace_remote_sha256']==pins['namespace-claim']
         and d['allocation']['resource_contract_matched'] is True and d['allocation']['errors']==[]
         and d['allocation']['scheduler_state']=='COMPLETED' and d['allocation']['scheduler_exit']=='0:0',
         'complete actual terminal custody/Slurm/namespace required')
    need(d.get('original_failed_custody_verified') is True and d.get('original_failed_raw_files')==41
         and d.get('original_namespace_local_sha256')==d.get('original_namespace_remote_sha256')==ORIGINAL_CLAIM_SHA,
         'accepted recovery must retain complete failed01 custody and both original claims')
    files=realization_files(supplied['realization-inventory'],pins['realization-inventory'],a,d,rr,
                            pins['realization-result'],REALIZATION_ROOT)
    need(base['terminal_jobs']==40 and old['terminal_jobs']==39
         and base['submission_journals']==82 and old['submission_journals']==80
         and base['outstanding_reservation']==0 and old['outstanding_reservation']==0
         and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600
         and base['gpu_cap_seconds']==7200 and base['input_pins'].get('predecessor')==PRIOR_BASELINE_SHA
         and base['input_pins'].get('acceptance')==pins['realization-acceptance']
         and base['input_pins'].get('disposition')==pins['realization-disposition'],
         'supplied accepted next baseline required; never synthesize it')
    prior=[r for r in base['related_accounting'] if r[0]!=job];new=[r for r in base['related_accounting'] if r[0]==job]
    need(len(prior)==old['terminal_jobs'] and len(new)==1 and normalized(prior)==normalized(old['related_accounting']),
         'all original rows including failed attempts plus actual successful realization required')
    actual=normalized(new)[0];allocation=d['allocation']
    need(actual[1].startswith('ace_llm') and actual[2:4]==['COMPLETED','0:0'] and actual[4]==allocation['allocated_cpus']
         and actual[5]==allocation['elapsed_seconds'] and actual[6]=='ucb736_asc2'
         and actual[7]==sorted(x.split('=',1) for x in allocation['alloc_tres'].split(',')),
         'actual realization allocation row differs from disposition')
    need(allocation['allocated_cpus']==2 and type(allocation['elapsed_seconds']) is int
         and 0<=allocation['elapsed_seconds']<=300
         and allocation['allocated_cpu_seconds']==2*allocation['elapsed_seconds']
         and allocation['gpu_count']==allocation['gpu_slice_seconds']==0,
         'actual realization costs must be measured2CPU/zeroGPU within300sec')
    expected_journals={**old['journal_pins'],REALIZATION_ROOT+'/submission.intent.json':rr['intent_sha256'],
                       REALIZATION_ROOT+'/submission.result.json':pins['realization-result']}
    need(base['journal_pins']==expected_journals and len(expected_journals)==base['submission_journals'],
         'all original and accepted-realization journal pins required')
    need(base['closed_cpu_seconds']==d['closed_cpu_seconds']==old['closed_cpu_seconds']+allocation['allocated_cpu_seconds']
         and base['closed_gpu_slice_seconds']==d['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']
         and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600,
         'next baseline must use actual realization charges')
    need(base['closed_cpu_seconds']+3600+1800<=57600 and base['closed_gpu_slice_seconds']<=7200,
         'actual remaining budget cannot admit this collection')
    return dict(metadata=m,pins=pins,job=job,realization_files=files)


def configuration(row):
    """Plain JSON projection of the pinned assembly_recipe/configuration_payload, no calls/imports."""
    public=row['public'];training=row['training']
    return dict(execution_authorized=False,runtime_bindings_pending=['world','expected_world_sha256','learner','output'],
        learner_parameters=dict(graph=public['graph'],noise_variances=public['learner']['noise_variances'],prior_variance=1.0),
        assembly_parameters=dict(scope='policy_train',graph=public['graph'],menu=public['menu'],task=public['task'],
            expected_task_sha256=public['task_sha256'],initial_actions=public['initial_actions'],total_budget=32,
            behavior_method=training['behavior_method'],snapshots=[{k:s[k] for k in
                ('acquisition_cost_units','teacher_method','candidates')} for s in training['snapshots']],
            replicate_ids=training['replicate_ids'],planning_draws=training['planning_draws'],tie_multiplier=training['tie_multiplier'],
            minimum_gain_gap=training['minimum_gain_gap'],structural_scores=training['structural_scores'],
            objective=public['objective'],domain_notes=public['domain_notes'],request_namespace=training['request_namespace']))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in METADATA:
        ap.add_argument('--'+key,required=True);ap.add_argument('--'+key+'-sha256',required=True)
    ap.add_argument('--realization-task',required=True,help='accepted custody task directory; only terminal and world000..031 are opened')
    a=ap.parse_args()
    supplied={k:read(getattr(a,k.replace('-','_')),getattr(a,k.replace('-','_')+'_sha256')) for k in METADATA}
    ctx=admit(supplied);m=ctx['metadata'];pins=ctx['pins'];f=m['freeze'];accepted=m['realization-acceptance']
    need(read(LOCAL_REGISTRY,pins['namespace-claim'])==supplied['namespace-claim'],'existing local claim differs; never create/adopt')
    read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
    source=committed(SOURCE_COMMIT,'scripts/research/'+ENTRY);need(sha(source)==ENTRY_SHA,'reviewed CPU entry differs')
    helpers=literal(source,'HELPER_PINS');closure=f['stage_closure']['stages']['collect']
    need(f['stage_closure']['source_pins']==f['source_pins'] and closure['helper_pins']==helpers
         and closure['entrypoint']==ENTRY and closure['entrypoint_sha256']==ENTRY_SHA,'frozen stage closure differs')
    proposal_raw=committed(SOURCE_COMMIT,PROPOSAL);need(sha(proposal_raw)==PROPOSAL_SHA,'original proposal bytes differ')
    proposal=decode(proposal_raw);need(digest(proposal)==PROPOSAL_SEMANTIC,'original proposal semantics differ')
    protocol={Path(n).name for n in proposal['source_pins']}
    work=protocol|{COLLECT,'llm_intervention_training_collection.py'};sources=work|{ENTRY}
    need(closure['work_sources']==sorted(work) and closure['source_names']==sorted(sources),'exact collect work/source roster differs')
    origins={n:PROBE if n=='probe.py' else 'scripts/research/'+n for n in sources|set(helpers)}
    data={n:committed(SOURCE_COMMIT,path) for n,path in origins.items()}
    need(len(data)==23 and {n:sha(b) for n,b in data.items()}==closure['input_source_files']
         and sha(data[COLLECT])==EXPECTED_COLLECTION_SHA
         and all(sha(data[n])==h for n,h in helpers.items()),'exact23-source helper closure required')
    for path,h in f['source_pins'].items():
        need(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts and pin(h),'relative source pin required')
        need(sha(committed(SOURCE_COMMIT,path))==h,'committed full-contract metadata source differs: '+path)
    task=Path(a.realization_task);terminal_raw=read(task/'terminal.json',accepted['realization_terminal_sha256'])
    terminal=decode(terminal_raw)
    need(terminal['status']=='complete' and terminal['version']=='ace-first-screen-descriptor-realization-v1'
         and terminal['freeze_sha256']==FREEZE_SHA and terminal['namespace_admission_sha256']==pins['namespace-admission']
         and terminal['proposal_sha256']==PROPOSAL_SEMANTIC and terminal['source_pins']==proposal['source_pins']
         and terminal['worlds_generated']==56 and terminal['environment_responses']==terminal['policy_updates']==0
         and terminal['retry_allowed'] is False,'accepted complete realization metadata differs')
    entries=terminal['entries'];rows=proposal['training_worlds']+proposal['development_worlds']
    need(len(entries)==len(rows)==56,'full56 descriptor metadata required')
    for i,(entry,row) in enumerate(zip(entries,rows)):
        need(entry['world_id']==row['world_id'] and entry['scope']==row['scope'] and entry['status']=='complete'
             and entry['path']==f'world-{i:03d}.json' and pin(entry['file_sha256']) and pin(entry['descriptor_sha256']),
             'exact56 identity/order pins required')
    inputs={'proposal.json':proposal_raw,'realization-terminal.json':terminal_raw};world_pins={}
    for i,wid in enumerate(WORLD_IDS):
        entry=entries[i];blob=read(task/f'world-{i:03d}.json',entry['file_sha256'])
        need(ctx['realization_files'][f'launch/realization/session/worker/task/world-{i:03d}.json']['sha256']==sha(blob),
             'original accepted training descriptor custody differs')
        value=decode(blob)
        need(value['scope']=='policy_train' and type(value['world_id']) is int and value['world_id']==wid
             and digest(value)==entry['descriptor_sha256'],'training descriptor identity/semantic pin differs')
        inputs[f'descriptor-{i:03d}.json']=blob;world_pins[str(wid)]=entry['descriptor_sha256']
    # No glob/rglob/copytree of task: world032..055 are never opened or staged.
    configurations={str(r['world_id']):configuration(r) for r in proposal['training_worlds']}
    snapshots=[s for c in configurations.values() for s in c['assembly_parameters']['snapshots']]
    need(len(snapshots)==160 and sum(len(s['candidates'])*(len(s['candidates'])-1)//2 for s in snapshots)==3584,
         'all160 records/all3584 comparisons must remain planned')
    inner=dict(schema=INNER_VERSION,scope='policy_train',files={n:sha(b) for n,b in inputs.items()},
        source_pins={'scripts/research/'+n:sha(data[n]) for n in work},proposal_sha256=PROPOSAL_SEMANTIC,
        world_roster=WORLD_IDS,roster_sha256=digest(WORLD_IDS),world_pins=world_pins,
        configuration_pins={wid:digest(c) for wid,c in configurations.items()},freeze_sha256=FREEZE_SHA,
        namespace_admission_sha256=pins['namespace-admission'])
    inner_raw=raw(inner);output=ROOT+'/collection'
    inner_admission=dict(schema=INNER_VERSION+'/admission',stage='policy_train_collection',status='admitted',
        packet_sha256=sha(inner_raw),output=output+'/session/worker/task',freeze_sha256=FREEZE_SHA,
        namespace_admission_sha256=pins['namespace-admission'],realization_terminal_sha256=sha(terminal_raw),
        source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,
        max_response_attempts=9216,max_reserved_cost_units=10240,gpus=0)
    data.update(inputs);data.update({'realization-acceptance.json':supplied['realization-acceptance'],
        'collection-packet.json':inner_raw,'collection-admission.json':raw(inner_admission)})
    need(len(data)==60 and max(map(len,data.values()))<=8*1024**2 and sum(map(len,data.values()))<=32*1024**2,
         'exact bounded training-only supervisor inputs required')
    diagnostic=data[literal(source,'DIAGNOSTIC')];service=data[literal(source,'SERVICE')]
    need(f['runtime']==literal(service,'RUNTIME') and f['runtime_source']==literal(diagnostic,'RUNTIME_SOURCE')
         and f['runtime_archive_sha256']==literal(diagnostic,'ARCHIVE_SHA')
         and f['runtime_manifest_sha256']==literal(diagnostic,'MANIFEST_SHA'),'frozen runtime differs')
    stage=dict(schema=VERSION,mode='collect',source_commit=SOURCE_COMMIT,output=output,
        files={n:sha(b) for n,b in sorted(data.items())},source_names=sorted(sources),protocol_source_pins=proposal['source_pins'],
        archive=dict(path=f['runtime_archive_path'],sha256=f['runtime_archive_sha256']),
        manifest=dict(path=f['runtime_manifest_path'],sha256=f['runtime_manifest_sha256']),
        runtime=f['runtime'],runtime_source=f['runtime_source'],resources=RESOURCES,deadlines=DEADLINES,
        freeze_sha256=FREEZE_SHA,namespace_admission_sha256=pins['namespace-admission'])
    stage_raw=raw(stage)
    admission=dict(schema=VERSION+'/admission',mode='collect',status='admitted',packet_sha256=sha(stage_raw),
        output=output,freeze_sha256=FREEZE_SHA,namespace_admission_sha256=pins['namespace-admission'],
        scientific_protocol_frozen=True,namespace_owned=True,source_runtime_scheduler_qualified=True,
        accounting_admitted=True,resources=RESOURCES,deadlines=DEADLINES)
    files={'inputs/'+n:b for n,b in data.items()}
    files.update({'inputs/packet.json':stage_raw,'inputs/admission.json':raw(admission)})
    for name,blob in supplied.items():files[name+('.md' if name=='freeze-review' else '.json')]=blob
    for name in ('build_packet.py','launch_once.py','monitor_once.py'):files[name]=read(Path(__file__).resolve().with_name(name))
    command=[stage['runtime_source']+'/bin/python','-I','-B','-c',literal(diagnostic,'BOOTSTRAP'),
        ROOT+'/inputs/'+ENTRY,ENTRY_SHA,'--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(stage_raw),
        '--source-commit',SOURCE_COMMIT,'--admission',ROOT+'/inputs/admission.json','--admission-sha256',sha(raw(admission)),
        '--output',output]
    files['job.sbatch']=('\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_scientific_collect01',
        '#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M',
        '#SBATCH --time=00:15:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
        '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','umask 077',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
        "export CUDA_VISIBLE_DEVICES=''",'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
        'exec '+shlex.join(command),''])).encode()
    base=m['baseline']
    outer=dict(schema='ace-scientific-collection-launch-v1',source_commit=SOURCE_COMMIT,remote_root=ROOT,
        local_root=str(LOCAL),external_root=str(EXT),
        resources=RESOURCES,deadlines=DEADLINES,source_pins={origins[n]:sha(data[n]) for n in origins},
        new_input_files={n:sha(b) for n,b in sorted(files.items())},baseline_sha256=pins['baseline'],
        realization_job_id=ctx['job'],namespace_claim_sha256=pins['namespace-claim'],
        admission=dict(closed_cpu_seconds=base['closed_cpu_seconds'],closed_gpu_slice_seconds=base['closed_gpu_slice_seconds'],
            preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=1800,candidate_gpu_reservation=0,
            total_cpu_seconds=base['closed_cpu_seconds']+5400,total_gpu_seconds=base['closed_gpu_slice_seconds'],
            ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,
            prior_jobs=base['terminal_jobs'],prior_journals=base['submission_journals']),
        submitted=False,new_namespace_claimed=False,realization_repeated=False,development_payloads_staged=0,
        planned_records=160,planned_pairs=3584,max_response_attempts=9216,max_reserved_cost_units=10240)
    files['packet.json']=raw(outer)
    need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'existing collection packet; no retry')
    created=[]
    try:
        need(read(LOCAL_REGISTRY,pins['namespace-claim'])==supplied['namespace-claim'],'existing recovery claim changed before publication')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        for dest in (EXT,LOCAL):
            dest.mkdir(parents=True,mode=0o700,exist_ok=False);created.append(dest);(dest/'inputs').mkdir(mode=0o700)
            for name,blob in files.items():write(dest/name,blob)
        for name,blob in files.items():need(read(LOCAL/name)==read(EXT/name)==blob,'complete two-copy packet differs')
        need(read(LOCAL_REGISTRY,pins['namespace-claim'])==supplied['namespace-claim'],'existing recovery claim changed during publication')
        read(ORIGINAL_LOCAL_REGISTRY,ORIGINAL_CLAIM_SHA)
        print(raw(dict(packet_sha256=sha(files['packet.json']),stage_packet_sha256=sha(stage_raw),
                       inner_collection_sha256=sha(inner_raw),submitted=False)).decode())
    except BaseException as exc:
        for dest in created:
            try:write(dest/'build-failure.json',raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False)))
            except BaseException:pass
        raise


if __name__=='__main__':main()
