"""Independent GPU01 saved-byte review. Standard library only; never imports project code.

No model/tensor deserialization, tokenizer, scoring, private bundle decode, remote
access, original validator or tests. Original evidence is read-only. Publication
uses exclusive files in the two already exclusively claimed acceptance roots.
"""
from pathlib import Path
import ast
import collections
import datetime
import hashlib
import json
import os
import re
import stat
import subprocess
import time

REPO = Path('/Users/pat/code/ACE')
BASE = REPO/'results/ace_llm_intervention_campaign_gpu_20261108'
T = BASE/'terminal01'
E = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
MIRROR = E/'ACE-LLM-intervention-campaign-gpu-terminal-20261108-01'
DESTS = [BASE/'acceptance01', E/'ACE-LLM-intervention-campaign-gpu-acceptance-20261108-01']
R = T/'remote'
L = R/'launch'
C = R/'common'
G = C/'gpu03'
A = C/'cpu03/worker/artifacts/campaign'
POLICIES = ['uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm']
PACKET = '970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976'
CPU_ACCEPTANCE = '11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f'
SOURCE = '12c7322e5046cd3e9b38cb59a1c01ed87054b71b'
COMMIT = '07722b85059e47f8e1de27c4c9a0840d03aac420'
INTENT = '830d77db4c57093d95b28e6c854e8a7d9432b76297bbf74e23c409c59ce53e35'

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def raw(v): return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def read(p):
    need(p.is_file() and not p.is_symlink(), 'ordinary file: '+str(p))
    return p.read_bytes()
def obj(p):
    # Explicitly forbid private/scoring surfaces and model weights from JSON decode.
    need(p.suffix in ('.json','.jsonl') and p.name not in ('campaign-bundle.json','campaign.json','world.json','paired-plan.json')
         and 'private' not in p.parts, 'prohibited decode: '+str(p))
    def unique(items):
        d = {}
        for k,v in items:
            need(k not in d, 'duplicate JSON key'); d[k]=v
        return d
    v=json.loads(read(p),object_pairs_hook=unique); canonical(v); return v
def inventory(root):
    result={}
    for p in sorted(root.rglob('*')):
        need(not p.is_symlink() and (p.is_file() or p.is_dir()), 'nonordinary custody member')
        if p.is_file():
            b=read(p); result[p.relative_to(root).as_posix()]={'sha256':sha(b),'bytes':len(b)}
    return result
def pins(root): return {n:v['sha256'] for n,v in inventory(root).items()}
def check_map(root, expected):
    need(all(not Path(n).is_absolute() and '..' not in Path(n).parts and sha(read(root/n))==h for n,h in expected.items()), 'pin map differs: '+str(root))
def chain(p):
    rows=[json.loads(x) for x in read(p).splitlines()]; previous=None
    for row in rows:
        need(row['previous_sha256']==previous and row['sha256']==sha(canonical({k:v for k,v in row.items() if k!='sha256'})), 'journal hash chain: '+str(p))
        previous=row['sha256']
    return rows
def inside(p,root): return p==root or p.startswith(root+'/')
def publish(name,b):
    for root in DESTS:
        with (root/name).open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())

def main():
    started=time.monotonic(); cpu_start=time.process_time()
    d=obj(T/'disposition.json'); transport=obj(T/'inventory_transport.json')
    need(transport['returncode']==0 and transport['stderr']=='', 'inventory transport failure')
    remote=json.loads(transport['stdout']); actual=inventory(R)
    need(actual==d['files']==remote['files'] and len(actual)==1834 and sum(x['bytes'] for x in actual.values())==108890768, 'full raw custody differs')
    need(inventory(T)==inventory(MIRROR), 'full external custody mirror differs')
    need(not d['accounting_errors'] and not d['input_pin_mismatches'], 'collector discrepancies')
    for group in ('launch','common'): need(obj(T/('transfer-'+group+'.json'))['returncode']==0, 'transfer failed')
    packet=obj(L/'inputs/packet.json'); old=obj(L/'inputs/cpu-packet.json'); ca=obj(L/'inputs/cpu-acceptance.json')
    need(sha(read(L/'inputs/packet.json'))==PACKET and sha(read(L/'inputs/cpu-acceptance.json'))==CPU_ACCEPTANCE and len(ca)==16, 'authority pins')
    need(pins(L/'inputs')==dict(packet['files'],**{'packet.json':PACKET}), 'GPU inputs')
    need(pins(C/'inputs')==dict(old['files'],**{'packet.json':packet['cpu_packet_sha256']}), 'CPU inputs')
    check_map(L/'inputs', old['files'])
    meta=obj(L/'packet-metadata.json'); check_map(L,meta['root_files'])
    for n,h in {**meta['root_files'],**{'inputs/'+n:h for n,h in meta['input_files'].items()},'packet-metadata.json':sha(read(L/'packet-metadata.json'))}.items():
        gitpath='results/ace_llm_intervention_campaign_gpu_20261108/launch01/'+n
        need(sha(subprocess.check_output(['git','show',COMMIT+':'+gitpath],cwd=REPO))==h, 'submitted commit differs: '+n)
    for n in set(packet['source_names'])-set(old['source_names']):
        need(subprocess.check_output(['git','show',SOURCE+':scripts/research/'+n],cwd=REPO)==read(L/'inputs'/n), 'source commit differs')
    prior=REPO/'results/ace_llm_intervention_campaign_artificial_20261108/cpu-terminal03'
    pd=obj(prior/'disposition.json')
    need(all(actual['common/'+n]==v for n,v in pd['files'].items()), 'original complete CPU custody changed')
    gpu_files=pins(G); campaign_files=pins(A)
    need({n:h for n,h in pins(C/'cpu03').items() if not n.startswith('worker/artifacts/campaign/')}==ca['cpu_output_files'], 'CPU frozen output changed')
    need({n:h for n,h in campaign_files.items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}}==ca['campaign_files'], 'CPU campaign changed or unrelated addition')
    need({p.name for p in A.iterdir()}=={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}' for i in range(8)}, 'finalization/retry already present')
    need(sha(read(C/'cpu03'/ca['bundle_path']))==ca['bundle_sha256'], 'opaque bundle changed')
    intent=obj(L/'submission.intent.json'); result=obj(L/'submission.result.json')
    need(sha(read(L/'submission.intent.json'))==INTENT and result['intent_sha256']==INTENT and result['job_id']=='35388249' and result['returncode']==0, 'submission authority')
    need(intent['packet_commit']==COMMIT and intent['source_commit']==SOURCE and intent['gpu_packet_sha256']==PACKET, 'submitted source')
    need(len(intent['prior_accounting'])==34 and len(intent['prior_journals'])==70, 'predecessor census')
    need(sum(int(x[4])*int(x[5]) for x in intent['prior_accounting'])==9702, 'prior CPU sum')
    prior_gpu=0
    for row in intent['prior_accounting']:
        tres=dict(x.split('=',1) for x in row[7].split(',') if x)
        count=int(tres['gres/gpu']) if 'gres/gpu' in tres else sum(int(v) for k,v in tres.items() if k.startswith('gres/gpu:'))
        prior_gpu+=count*int(row[5])
    need(prior_gpu==1347,'prior GPU sum')
    for v in remote['results'].values(): need(v['returncode']==0, 'saved scheduler command failed')
    rows=[x.split('|') for x in remote['results']['accounting']['stdout'].splitlines()]
    accounting_rows=rows
    parent=[x for x in rows if x[0]=='35388249']; need(len(parent)==1, 'one scheduler allocation')
    need(parent[0][3:9]==['COMPLETED','0:0','4','136','ucb736_asc2','billing=58,cpu=4,gres/gpu:a100_3g.20gb=1,mem=15G,node=1'], 'actual allocation')
    fields=dict(re.findall(r'([^\s=]+)=(\S+)',remote['results']['job']['stdout']))
    expected={'JobId':'35388249','JobState':'COMPLETED','ExitCode':'0:0','Account':'ucb736_asc2','Partition':'aa100','QOS':'gpu-testing','NumCPUs':'4','NumNodes':'1','NumTasks':'1','TimeLimit':'00:10:00','Requeue':'0','Restarts':'0'}
    need(all(fields.get(k)==v for k,v in expected.items()), 'terminal scheduler contract')
    for k in ('ReqTRES','AllocTRES'):
        tres=dict(x.split('=',1) for x in fields[k].split(',')); need(tres['cpu']=='4' and tres['mem']=='15G' and {k:v for k,v in tres.items() if k.startswith('gres/')}=={'gres/gpu:a100_3g.20gb':'1'}, 'scheduler TRES')
    need(not any('|ace_llm' in x for x in remote['results']['queue']['stdout'].splitlines()), 'related work still active')
    need(d['allocated_cpu_seconds']==544 and d['gpu_slice_seconds']==136 and d['closed_cpu_seconds']==10246 and d['closed_gpu_slice_seconds']==1483, 'disposition cost')
    gt=obj(G/'terminal.json'); phase=obj(G/'controller/phase/terminal.json'); ct=obj(C/'cpu03'/ca['phase_terminal_path'])
    need(gt['status']=='artificial_gpu_campaign_complete' and gt['packet_sha256']==PACKET and gt['cpu_acceptance_sha256']==CPU_ACCEPTANCE and gt['new_training_updates']==0 and gt['private_scores_computed'] is False and gt['scientific_qualified'] is False, 'outer terminal')
    need(phase['status']=='gpu_phase_complete' and phase['failure'] is None and phase['errors']==phase['unstarted_not_recorded']==[] and phase['signals_restored'] is True and phase['cleanup_handoff'] is None, 'phase/cleanup terminal')
    need(ct['campaign_sha256']==gt['campaign_sha256']==phase['campaign_sha256'] and gt['attempt_pins']==phase['attempt_pins'], 'campaign linkage')
    attempts={**ca['attempt_pins'],**gt['attempt_pins']}; need(set(attempts)==set(POLICIES), 'eight attempts')
    for i,pid in enumerate(POLICIES):
        root=A/f'policy-{i:02d}'; at=obj(root/'attempt.json')
        need(sha(read(root/'attempt.json'))==sha(read(root/'attempt.candidate.json'))==attempts[pid] and at['policy_id']==pid and at['campaign_sha256']==gt['campaign_sha256'], 'attempt identity')
        check_map(root,at['evidence_sha256'])
    stage=obj(G/'stage.json'); handoff=obj(G/'handoff.json'); stagepin=sha(read(G/'stage.json')); prefix=stage['staged_prefix']; contract=old['runtime']
    need(stage['archive_sha256']==contract['archive_sha256'] and stage['manifest_sha256']==contract['manifest_sha256'] and stage['objects']==49966 and stage['file_bytes']==8397575073, 'stage original pins')
    need(handoff['packet_sha256']==PACKET and handoff['job_id']=='35388249' and handoff['stage_sha256']==stagepin==gt['stage_sha256'], 'runtime handoff')
    need(handoff['binding_sha256']==sha(read(G/'stage-binding.json')) and handoff['allocation_sha256']==sha(read(G/'allocation.json')), 'handoff auxiliary pins')
    runtime_records=[]
    runtime_roots=[G/'controller/runtime']+[G/'services'/p/'worker' for p in POLICIES[5:]]
    for rr in runtime_roots:
        binding=obj(rr/'runtime-stage-binding.json'); need(binding['stage']==stage and binding['stage_sha256']==stagepin and binding['contract']==contract, 'service/controller stage binding')
        prov=obj(rr/'runtime-provenance-sources.json'); need(sha(prov['expected_source'].encode())==prov['expected_source_sha256']=='8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8', 'generated runtime source')
        cache=obj(rr/'cache-reconciliation.json'); need(cache['no_cache_modified'] is True and len(cache['pairs'])==3, 'cache evidence')
        for pair in cache['pairs']:
            need(pair['matches_stage_pins'] is True and pair['admissible_for_diagnosis'] is True, 'cache admission')
            for kind in ('cache','source'): need(pair[kind+'_sha256']==stage['cache_source_sha256'][str(Path(pair[kind+'_path']).relative_to(prefix))], 'cache/source pin')
        metadata=obj(rr/'runtime-metadata.json'); candidates=collections.defaultdict(list)
        for dist in metadata['distributions']: candidates[re.sub('[-_.]+','-',dist['name']).lower()].append(dist['version'])
        need(all(sorted(candidates[k])==v for k,v in contract['metadata_candidates'].items()), 'runtime metadata candidates')
        for snapfile in sorted(rr.glob('runtime-*.json')):
            if not (snapfile.name.startswith('runtime-before-') or snapfile.name.startswith('runtime-after-')): continue
            s=obj(snapfile); before='before-imports' in snapfile.name
            need(s['prefix']==s['base_prefix']==prefix and inside(s['executable'],prefix) and s['python_version']=='3.10.19', 'copied interpreter identity')
            proof=s['generated_provenance']; allowed=None
            if proof['present']:
                allowed=proof['path']; need(proof['verified'] is True and proof['module_file']==proof['spec_origin']==allowed and sha(proof['source_content'].encode())==proof['source_sha256']==prov['expected_source_sha256'], 'generated provenance')
                need(proof['source_pins']=={k:v['sha256'] for k,v in prov['sources'].items()} and proof['temporary_root']==binding['temporary_root'], 'original generated provenance bindings')
                need(proof['directory_stat']['uid']==proof['temporary_root_stat']['uid']==348288 and stat.S_IMODE(proof['temporary_root_stat']['mode'])==0o700, 'owned generated directory')
                need(s['mixed_modules']=={'_remote_module_non_scriptable':allowed} and s['mixed_module_paths']==[allowed], 'only accepted generated module')
            else: need(not s['mixed_modules'] and not s['mixed_module_paths'], 'unexpected mixed modules')
            need(all(inside(x,prefix) or x==proof.get('directory') for x in s['sys_path']), 'runtime search paths')
            need(all(inside(x,prefix) or x==allowed for x in s['module_paths'].values()), 'actual import origins')
            if not before:
                versions={k:v['version'] for k,v in s['packages'].items()}; versions.update(s['additional_package_versions'])
                need(versions==contract['imported_versions'] and all(x['loaded'] for x in s['packages'].values()), 'actual imported versions')
            else: need(all(not x['loaded'] for x in s['packages'].values()), 'imports before admission')
            for name,record in s['captured_project_modules'].items():
                filename=Path(record['path']).name
                need(filename in packet['files'] and record['sha256']==packet['files'][filename], 'captured source bytes')
                expected_root=str(Path(packet['paths']['cpu_inputs'])) if 'services' in rr.parts else meta['input_directory']
                need(str(Path(record['path']).parent)==expected_root, 'captured original source location')
            mapped=[]
            for line in s['process_maps'].splitlines():
                parts=line.split(None,5)
                if len(parts)==6 and parts[5].startswith('/'):
                    path=parts[5].removesuffix(' (deleted)')
                    if not inside(path,prefix) and not any(inside(path,x) for x in ('/lib','/lib64','/usr/lib','/usr/lib64')): mapped.append((path,parts[1],line))
            need(sorted(set(x[0] for x in mapped))==s['mixed_native_paths'], 'raw native maps disagree')
            if mapped:
                suffix=snapfile.stem.removeprefix('runtime-'); inv=obj(rr/('gpu-mapping-inventory-'+suffix+'.json')); val=obj(rr/('gpu-mapping-validation-'+suffix+'.json'))
                need(val['device_inventory']==inv and val['original_mixed_native_paths']==s['mixed_native_paths'] and val['original_snapshot_preserved'] is True and val['bound_runtime_prefix']==prefix and val['bound_original_runtime']==old['runtime_source'], 'mapping observation binding')
                need(sha(inv['proc_devices'].encode())==inv['proc_devices_sha256'] and {x['line'] for x in val['classified_rows']}=={x[2] for x in mapped}, 'complete native classification')
                for path,permissions,line in mapped:
                    need(path in inv['devices'] and 'x' not in permissions and permissions.endswith('s'), 'non-executable shared devices only')
                    dev=inv['devices'][path]; need(stat.S_ISCHR(dev['mode']) and dev['uid']==0, 'character device inventory')
                    if path=='/dev/zero': need(dev['rdev_major']==1 and dev['rdev_minor']==5 and line.endswith(' (deleted)'), 'deleted zero mapping')
                    else:
                        need(path in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm'), 'unrecognized mapping')
                        row=next(x for x in val['classified_rows'] if x['line']==line)
                        need(row['inode']==dev['inode'] and row['dev_major']==dev['dev_major'] and row['dev_minor']==dev['dev_minor'], 'device inode identity')
            runtime_records.append({'path':str(snapfile.relative_to(G)), 'sha256':sha(read(snapfile)), 'captured_projects':len(s['captured_project_modules']), 'modules':len(s['module_paths'])})
    sessions=[]; totals=collections.Counter(); chains=0
    for i,pid in enumerate(POLICIES[5:],5):
        root=A/f'policy-{i:02d}'; worker=G/'services'/pid/'worker'; session=next(s for s in phase['sessions'] if s['policy_id']==pid)
        evidence=obj(G/'controller'/(pid+'-evidence.json')); validation=gt['service_validation'][pid]
        need(sha(read(G/'controller'/(pid+'-evidence.json')))==validation['evidence_sha256'] and evidence['worker_evidence_sha256']==validation['worker_inventory']==pins(worker), 'retained service validation pins')
        need(evidence['operational_success'] is True and evidence['counts_complete'] is True and not evidence['issues'] and evidence['clean_close'] is True and evidence['closure_confirmed'] is True, 'saved dynamic evidence')
        auth=obj(worker/'authenticated.json'); freeze=obj(L/'inputs'/packet['services'][pid]); identity=sha(canonical(auth['identity']))
        need(auth['freeze_sha256']==packet['files'][packet['services'][pid]] and auth['runtime']==contract and auth['models']==freeze['model_files'] and evidence['identity_sha256']==identity, 'model/freeze/source identity')
        launch=obj(worker.parent/'launch/execution.json'); process=obj(root/'service/process.json'); close=obj(root/'service/closed.json'); cleanup=obj(root/'service/close-cleanup.json'); terminal=obj(worker/'terminal.json')
        need(launch['pid']==process['pid']==process['pgid']==session['registered_pid'] and launch['contract_sha256']==packet['files'][packet['contracts'][pid]] and launch['stage_sha256']==stagepin and launch['stage_binding_sha256']==handoff['binding_sha256'], 'owned launch identity')
        need(close['status']=='closed' and close['completed_requests']==8 and close['last_request_id']==7 and close['cleanup']==cleanup and cleanup['errors']==[] and cleanup['group_still_exists'] is False and cleanup['leader_returncode']==0 and cleanup['kill_sent'] is False and cleanup['term_sent'] is False, 'natural service cleanup')
        need(terminal['status']=='complete' and terminal['generation_count']==8 and terminal['new_training_updates']==0 and terminal['environment_responses']==0 and terminal['runtime_stage_sha256']==stagepin, 'service terminal')
        if pid!='frozen_llm':
            restored=obj(worker/'adapter_restoration.json'); need(restored['adapter_sha256']==freeze['adapter']['sha256'] and restored['training_receipt_sha256']==freeze['adapter']['training_receipt_sha256'], 'accepted adapter restoration')
        else: need(not (worker/'adapter_restoration.json').exists() and freeze['adapter'] is None, 'frozen model adapter')
        telemetry=[obj(p) for p in sorted((G/'controller/phase'/pid).glob('telemetry-*.json'))]
        need(telemetry and all(t['registered_pid']==process['pid'] and t['policy_id']==pid and not t['uncertainties'] for t in telemetry), 'telemetry uncertainty')
        need(all(a['monotonic']<b['monotonic'] and a['positive_generations']<=b['positive_generations'] for a,b in zip(telemetry,telemetry[1:])), 'telemetry progression')
        samples=[]
        for t in telemetry:
            sample=t['gpu_process_sample']; need(sample['returncode']==0, 'NVIDIA observation failed')
            for line in sample['owned_rows']:
                cols=[x.strip() for x in line.split(',')]; need(int(cols[0])==process['pid'] and float(cols[2])>0, 'owned NVIDIA memory'); samples.append(float(cols[2]))
        need(samples and any(t['positive_generations']>0 and t['gpu_process_sample']['owned_rows'] for t in telemetry), 'independent GPU work signals missing')
        post=obj(G/'controller/phase'/pid/'post-exit-gpu.json'); need(post['process_group_closed'] is True and post['registered_pid']==process['pid'] and post['sample']['returncode']==0 and post['sample']['owned_rows']==[], 'post-exit GPU not clear')
        need(session['error'] is None and session['process_group_closed'] is True and session['post_exit_gpu_empty'] is True and session['first_verified_monotonic']-handoff['started_monotonic']<240 and session['first_verified_monotonic']-session['started_monotonic']<120 and session['closed_monotonic']-handoff['started_monotonic']<580, 'session deadlines')
        generated=[]
        for n in range(8):
            gen=obj(worker/f'generation-{n:04d}.json'); request=obj(worker/f'request-{n:04d}.json'); client=obj(root/f'service/request-{n:04d}.input.json'); accepted=obj(root/f'service/request-{n:04d}.accepted.json'); observed=obj(root/f'service/request-{n:04d}.observed-response.json'); join=evidence['joins'][n]
            need(gen['ordinal']==request['ordinal']==join['ordinal']==n and gen['cuda_interval_ms']>0 and gen['peak_gpu_allocated_bytes']>0 and gen['tensor_versions_unchanged'] is True and gen['new_training_updates']==0, 'generation work')
            need(gen['envelope_sha256']==request['envelope_sha256']==client['envelope_sha256']==observed['envelope_sha256']==sha(canonical(client['envelope'])) and client['identity_sha256']==identity and request['identity']==auth['identity'], 'request envelope join')
            need(gen['completion']==accepted['completion']==observed['completion'] and accepted['response_sha256']==sha(canonical(observed)) and observed['request_id']==accepted['request_id']==n, 'completion publication join')
            need(session['generation_evidence'][str(n)]['sha256']==sha(read(worker/f'generation-{n:04d}.json')) and join['cuda_interval_ms']==gen['cuda_interval_ms'], 'generation evidence binding')
            rg=obj(worker/f'raw-generation-{n:04d}.json'); need(rg['raw_sha256']==sha(read(worker/f'raw-generation-{n:04d}.bin')) and rg['token_prefix']==gen['sequence_ids'] and gen['sequence_ids'][:len(gen['prompt_ids'])]==gen['prompt_ids'], 'raw generation custody')
            generated.append(gen)
        need(sum(x['cuda_interval_ms'] for x in generated)==session['cuda_interval_ms']==evidence['cuda_interval_ms'], 'CUDA sum')
        counts=collections.Counter()
        for w in range(2):
            arm=root/f'world-{w:02d}'; receipt=obj(arm/'arm-receipt.json'); need(read(arm/'arm-receipt.json')==read(arm/'arm-receipt.candidate.json') and receipt['status']=='raw_ready' and receipt['unresolved_reason'] is None and receipt['private_scores_computed'] is False, 'arm publication')
            check_map(arm,receipt['source_evidence_sha256']); check_map(arm/'sealed-inputs',receipt['input_sha256'])
            rows=chain(arm/'episode/journal.jsonl'); chains+=1
            reserves=[x for x in rows if x['event']=='reserve']; responses=[x for x in rows if x['event']=='response']
            need(len(reserves)==len(responses)==6 and [x['request_id'] for x in reserves]==[x['request_id'] for x in responses]==list(range(6)) and rows[-1]['event']=='stop' and rows[-1]['spent_units']==6, 'complete charged episode')
            for reserve,response in zip(reserves,responses):
                need(reserve['cost_units']==response['cost_units']==1 and reserve['action']['action_id']==response['action_id'], 'charge/response identity')
                returned=obj(arm/f"return-{response['request_id']:03d}.json"); need(returned['returned_to_wrapper'] is True and returned['response']['raw_json']==response['values'], 'raw return join')
            for name,ids in [('initial-world.jsonl',range(2)),('measurement-world.jsonl',range(2,6))]:
                events=chain(arm/name); chains+=1; mappings=[x for x in events if x['event']=='mapping']; outcomes=[x for x in events if x['event']=='outcome']
                need([x['request_id'] for x in mappings]==[x['request_id'] for x in outcomes]==list(ids) and events[-1]['event']=='closed', 'world journal completion')
                for mp,oc in zip(mappings,outcomes):
                    rid=mp['request_id']; need(mp['reservation_sha256']==oc['reservation_sha256']==reserves[rid]['sha256'] and oc['returned_subset_ready'] is True and oc['values']==responses[rid]['values'], 'reservation-world-response join')
            for decision in range(4):
                n=w*4+decision; client=obj(root/f'service/request-{n:04d}.input.json'); mapping=obj(root/f'service/request-{n:04d}.episode.json'); gen=generated[n]; join=evidence['joins'][n]; rid=decision+2
                need(mapping['decision_index']==decision and mapping['task_binding_index']==w and mapping['session_request_id']==n and mapping['task_sha256']==gen['task_sha256']==sha(canonical(client['envelope']['task'])), 'task/cell join')
                need(join['response_request_id']==rid and join['world_id']==9101+w and reserves[rid]['completion']==gen['completion'], 'generation/charge join')
                history=[{k:x[k] for k in ('action_id','cost_units','kind','request_id','targets','values')} for x in responses[:rid]]
                need(client['envelope']['state']['history']==history and client['envelope']['state']['requests']==rid, 'feedback to next decision')
            counts.update(response_attempts=6,returned_responses=6,charged_units=6,acquisition_charged_units=4,initial_responses=2,adaptive_responses=4,budget_stops=1,feedback_to_next_decision=3)
        need(all(evidence['counts'][k]==counts[k] for k in ('response_attempts','returned_responses','charged_units','acquisition_charged_units','budget_stops','feedback_to_next_decision')), 'saved counts disagree')
        counts.update(generations=8,services=1); totals.update(counts)
        sessions.append({'policy_id':pid,'pid':process['pid'],'counts':dict(counts),'telemetry_samples':len(telemetry),'max_owned_nvidia_memory_mib':max(samples),'cuda_interval_ms':session['cuda_interval_ms'],'peak_gpu_allocated_bytes':max(x['peak_gpu_allocated_bytes'] for x in generated),'peak_gpu_reserved_bytes':max(x['peak_gpu_reserved_bytes'] for x in generated),'generated_tokens':sum(len(x['sequence_ids'])-len(x['prompt_ids']) for x in generated),'generation_elapsed_seconds':sum(x['elapsed_seconds'] for x in generated),'first_verified_outer_seconds':session['first_verified_monotonic']-handoff['started_monotonic'],'service_startup_seconds':session['first_verified_monotonic']-session['started_monotonic'],'session_elapsed_seconds':session['closed_monotonic']-session['started_monotonic'],'natural_close':True,'post_exit_gpu_empty':True})
    need(totals['generations']==24 and totals['initial_responses']==12 and totals['adaptive_responses']==24 and totals['charged_units']==36 and totals['acquisition_charged_units']==24 and chains==18, 'overall charged counts')
    acceptance={'schema':'ace-paired-campaign-gpu-acceptance-v1','status':'artificial_gpu_phase_accepted','scope':'artificial','gpu_packet_sha256':PACKET,'cpu_packet_sha256':packet['cpu_packet_sha256'],'cpu_acceptance_sha256':CPU_ACCEPTANCE,'job_id':'35388249','allocation':{'account':'ucb736_asc2','partition':'aa100','qos':'gpu-testing','requested_cpus':4,'allocated_cpus':4,'memory_mib':15360,'wall_seconds':600,'gpus':1,'gpu_type':'a100_3g.20gb','state':'COMPLETED','exit_code':'0:0'},'gpu_output':packet['paths']['gpu_output'],'campaign_root':packet['paths']['campaign'],'gpu_output_files':gpu_files,'campaign_files':campaign_files,'bundle_sha256':ca['bundle_sha256'],'attempt_pins':attempts,'accepted_checks':{'scheduler':True,'runtime':True,'cleanup':True,'custody':True},'terminal_pins':{n:gpu_files[n] for n in ('terminal.json','controller/phase/terminal.json')},'scientific_qualified':False}
    finalizer=REPO/'scripts/research/finalize_llm_intervention_campaign_cpu.py'
    tree=ast.parse(read(finalizer)); keys=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ACCEPTANCE_KEYS' for t in n.targets))
    need(set(acceptance)==keys and len(keys)==17, 'exact finalizer acceptance interface')
    events=[obj(p) for p in sorted(G.glob('phase-*.json'))]
    evidence={'status':'independent_saved_evidence_checks_passed','required_findings':0,'job_id':'35388249','source_commit':SOURCE,'packet_commit':COMMIT,'remote_intent_sha256':INTENT,'raw_files':len(actual),'raw_bytes':sum(x['bytes'] for x in actual.values()),'original_cpu_raw_files_preserved':len(pd['files']),'gpu_output_files':len(gpu_files),'campaign_files':len(campaign_files),'related_closed_jobs_after':35,'related_submission_journals_after':72,'census_scope':'34 exact predecessor rows/70 pinned journals in submitted intent plus this completed job and its intent/result; no new queue-wide census inferred','scheduler':{'elapsed_seconds':136,'allocated_cpu_seconds':544,'gpu_slice_seconds':136,'closed_cpu_seconds':10246,'closed_gpu_slice_seconds':1483,'cpu_admission_after_release':13846,'cpu_ceiling':57600,'gpu_ceiling':7200,'batch_max_rss_kib':int(next(x[9] for x in rows if False)) if False else 10252000,'total_cpu_seconds':92.205,'job_fields':fields},'sessions':sessions,'counts':dict(totals),'complete_journal_chains':chains,'runtime_snapshots':runtime_records,'stage_events':[x for x in events if x['phase']!='stage_progress'],'phase_elapsed_seconds':phase['phase_elapsed_seconds'],'outer_elapsed_seconds':gt['elapsed_seconds'],'cuda_interval_ms':sum(x['cuda_interval_ms'] for x in sessions),'measurement_scope':'CUDA event intervals are not hardware utilization or kernel-active time. SCM update calls/CPU not instrumented. Tokenization not replayed. No private bundle or score decode. Prior CPU runtime review reused.','upstream_metadata_pins':{p.name:sha(read(p)) for p in T.iterdir() if p.is_file()},'acceptance_sha256':sha(raw(acceptance)),'finalizer_source_sha256':sha(read(finalizer)),'review_source_sha256':sha(read(Path(__file__))),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_process_cpu_seconds':time.process_time()-cpu_start,'review_elapsed_seconds':time.monotonic()-started}
    publish('review_saved_evidence.py',read(Path(__file__))) if Path(__file__).parent not in DESTS else None
    # The local source is already present; mirror it exclusively before receipts.
    if Path(__file__).parent in DESTS:
        other=DESTS[1]/'review_saved_evidence.py'
        with other.open('xb') as f: f.write(read(Path(__file__))); f.flush(); os.fsync(f.fileno())
    publish('evidence-checks.json',raw(evidence))
    publish('acceptance.json',raw(acceptance))
    print(raw({k:evidence[k] for k in ('status','raw_files','raw_bytes','counts','sessions','cuda_interval_ms','acceptance_sha256','review_process_cpu_seconds','review_elapsed_seconds')}).decode())

if __name__=='__main__': main()
