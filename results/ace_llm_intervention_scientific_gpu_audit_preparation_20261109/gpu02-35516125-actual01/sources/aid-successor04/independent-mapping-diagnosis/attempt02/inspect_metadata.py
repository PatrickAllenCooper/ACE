"""One-off saved metadata diagnosis. No project imports, calls, or private decoding."""
import hashlib
import json
from pathlib import Path
import re
import traceback

OUT = Path(__file__).parent
R = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
A = BASE/'ACE-LLM-gpu02-35516125-postcustody-01'
CPU = R/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02'
pins = {}
checks = 0
def need(ok, label):
    global checks
    checks += 1
    if not ok: raise ValueError(label)
def sha(b): return hashlib.sha256(b).hexdigest()
def read(path, expected=None):
    b = Path(path).read_bytes()
    if expected is not None: need(sha(b) == expected, 'pin '+str(path))
    pins[str(path)] = dict(bytes=len(b), sha256=sha(b))
    return b
def dump(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
def top(raw):
    # Split canonical JSON at top-level punctuation without deserializing nested values.
    raw=raw.strip(); need(raw[:1]==b'{' and raw[-1:]==b'}','object required')
    chunks=[]; start=1; depth=0; quoted=False; escape=False
    for i in range(1,len(raw)-1):
        x=raw[i]
        if quoted:
            if escape: escape=False
            elif x==92: escape=True
            elif x==34: quoted=False
        elif x==34: quoted=True
        elif x in (123,91): depth+=1
        elif x in (125,93): depth-=1
        elif x==44 and depth==0: chunks.append(raw[start:i]); start=i+1
    need(not quoted and depth==0,'balanced object')
    chunks.append(raw[start:-1]); result={}
    for chunk in chunks:
        k,v=chunk.split(b':',1); key=json.loads(k)
        need(key not in result,'duplicate top field'); result[key]=v
    return result
def metadata(raw, keys):
    fields=top(raw)
    return {k:json.loads(fields[k]) for k in keys},fields

try:
    c=json.loads(read(A/'inputs/config.json','952dcdcb39f54330374552400118c040adb1a6f2fba0cb9777136f55379379a2'))
    def artifact(name):
        p=c['artifacts'][name];return json.loads(read(p['path'],p['sha256']))
    current=artifact('remote_custody'); full=current['full_inventory']['files']
    prior=artifact('cpu_remote_custody')['full_inventory']['files']
    p=artifact('inner'); accepted=artifact('cpu_acceptance'); claim=artifact('namespace_claim')
    def member(name):
        row=full[name]
        if name in c['fetched']:
            item=c['fetched'][name];need(item['sha256']==row['sha256'],'batch/full pin join '+name)
            path=Path(item['path'])
        else: path=Path(c['capsules'][1])/'remote'/name
        raw=read(path,row['sha256']);need(len(raw)==row['bytes'],'size '+name);return raw
    def j(name): return json.loads(member(name))
    old=j('control/inputs/cpu-packet.json')
    need(p['cpu_packet_sha256']==accepted['cpu_packet_sha256']==full['control/inputs/cpu-packet.json']['sha256'],'CPU packet join')
    need(p['paths']['cpu_output']==old['cpu_output']==accepted['cpu_output'],'logical CPU output')
    need(p['paths']['campaign']==accepted['campaign_root']==old['cpu_output']+'/worker/artifacts/campaign','logical campaign path')
    starts=[]
    for relative in ['start.json','worker/start.json']:
        raw=read(CPU/'remote/launch/cpu'/relative,'02ff21af91dec84b82bc0e46411c479eeb27cf694c01dccbad676fb6e15527f1')
        ext=BASE/'ACE-LLM-intervention-scientific-campaign-cpu-terminal-20261108-02'/'remote/launch/cpu'/relative
        need(read(ext)==raw,'independent CPU copy')
        row=dict(bytes=len(raw),sha256=sha(raw))
        need(prior['launch/cpu/'+relative]==full['cpu/cpu/'+relative]==row,'both full inventories/start pin')
        need(accepted['cpu_output_files'][relative]==sha(raw),'independent CPU acceptance/start')
        start=json.loads(raw);need(start['packet_sha256']==p['cpu_packet_sha256'],'start/CPU packet')
        starts.append(start)
    need(starts[0]==starts[1],'parent/worker path proof identical')
    binding=starts[0]['output_path_binding']
    need(binding['logical']==old['cpu_output'] and binding['mount_alias'] is True,'recorded logical binding')
    need(binding['resolved']=='/gpfs/alpine1/scratch/'+binding['logical'].removeprefix('/scratch/alpine/')
         and binding['logical'].startswith('/scratch/alpine/')
         and type(binding['root_device']) is int and type(binding['root_inode']) is int,'one exact recorded mount alias')
    resolved=binding['resolved']+'/worker/artifacts/campaign'
    source_pins={}
    for name in ['llm_intervention_persistent_policy.py','llm_intervention_scientific_service_evidence.py','llm_intervention_campaign.py','llm_intervention_paired_episode.py','llm_intervention_scientific_gpu_phase.py']:
        raw=member('launch/inputs/'+name)
        need(sha(raw)==p['files'][name],'current packet source pin '+name)
        need(read(R/'scripts/research'/name)==raw,'read source exact captured bytes '+name)
        source_pins[name]=sha(raw)
    name='prepare_llm_intervention_scientific_campaign_cpu.py'
    raw=read(CPU/'remote/launch/inputs'/name,old['files'][name])
    need(sha(raw)==full['cpu/inputs/'+name]['sha256']==prior['launch/inputs/'+name]['sha256'],'CPU path guard source lineage')
    need(read(R/'scripts/research'/name)==raw,'current CPU source equals actual capture')
    source_pins[name]=sha(raw)
    terminal=j('launch/gpu/terminal.json')
    planpins=j('cpu/cpu/worker/artifacts/campaign/source-plan-pins.json')
    worlds=claim['world_ids']['development']
    need([x['world_id'] for x in planpins]==worlds and len(worlds)==24,'ordered namespace source plans')
    C='cpu/cpu/worker/artifacts/campaign/'
    policies=['frozen_llm','sft_llm','dpo_llm']; summaries=[]; mapping_rows=[]
    for pi,policy in enumerate(policies,5):
        root=C+f'policy-{pi:02d}/';client=root+'service/'
        worker='launch/gpu/services/'+policy+'/worker/'
        attempt=j(root+'attempt.json');evidence=j('launch/gpu/controller/'+policy+'-evidence.json')
        validation=terminal['service_validation'][policy]
        need(validation['evidence_sha256']==full['launch/gpu/controller/'+policy+'-evidence.json']['sha256'],'terminal validator evidence pin')
        need(validation['attempt_sha256']==evidence['attempt_sha256']==full[root+'attempt.json']['sha256'],'attempt validator pin')
        ip=old['service_identity_pins'][policy]
        need(evidence['identity_sha256']==ip and evidence['recorded_campaign_root']==resolved,'recorded root and identity')
        need(len(attempt['arms'])==24,'24 original arms')
        offset=0;prior_task=None
        for wi,arm in enumerate(attempt['arms']):
            worldroot=root+f'world-{wi:02d}/'
            need(arm['world_id']==worlds[wi] and arm['path']==f'world-{wi:02d}','arm world namespace')
            receipt=j(worldroot+'arm-receipt.json')
            need(arm['receipt_sha256']==full[worldroot+'arm-receipt.json']['sha256'],'arm receipt pin')
            need(receipt['policy_id']==policy and receipt['plan_sha256']==planpins[wi]['sha256'],'arm policy/plan')
            sess=receipt['session'];taskbinding_raw=member(client+f'task-binding-{wi:04d}.json')
            tb,tf=metadata(taskbinding_raw,['binding_index','last_completed_session_request_id','model_reloaded','previous_task_sha256','task_sha256','worker_pid'])
            need(sha(tf['task'])==tb['task_sha256'],'opaque task digest')
            need(tb['binding_index']==wi and tb['last_completed_session_request_id']==offset-1
                 and tb['previous_task_sha256']==prior_task and tb['model_reloaded'] is False,'task binding lineage')
            task=tb['task_sha256'];prior_task=task
            need(sess['count_before']==offset and sess['request_before']==offset-1 and sess['identity_sha256']==ip
                 and sess['pid']==tb['worker_pid'] and sess['kind']=='persistent','session start')
            end=sess['count_after'];need(sess['request_after']==end-1 and type(end) is int and end>=offset,'session end')
            for ordinal in range(offset,end):
                n=client+f'request-{ordinal:04d}.episode.json';mapping=j(n)
                expected=dict(episode_directory=resolved+f'/policy-{pi:02d}/world-{wi:02d}/episode',
                    decision_index=ordinal-offset,session_request_id=ordinal,task_sha256=task,
                    task_binding_index=wi,identity_sha256=ip,status='planned_before_transmission',scope='client_only_episode_to_session_mapping')
                need(mapping==expected,'exact mapping/'+n)
                join=evidence['joins'][ordinal]
                need(join['ordinal']==join['session_request_id']==ordinal and join['world_id']==worlds[wi]
                    and join['decision_index']==ordinal-offset and join['task_sha256']==task and join['accepted'] is True,'validator decision join')
                request_raw=member(client+f'request-{ordinal:04d}.input.json')
                request,rf=metadata(request_raw,['type','request_id','identity_sha256','envelope_sha256'])
                need(request==dict(type='request',request_id=ordinal,identity_sha256=ip,envelope_sha256=join['envelope_sha256']),'input RPC/identity/envelope')
                need(sha(rf['envelope'])==request['envelope_sha256'],'opaque full envelope digest')
                ef=top(rf['envelope']);need(sha(ef['task'])==task,'opaque request task join')
                worker_raw=member(worker+f'request-{ordinal:04d}.json')
                wr,wf=metadata(worker_raw,['type','request_id','identity_sha256','envelope_sha256'])
                need(wr==request and wf['envelope']==rf['envelope'],'client/worker exact opaque request bytes')
                generation,gf=metadata(member(worker+f'generation-{ordinal:04d}.json'),['ordinal','task_sha256','envelope_sha256'])
                need(generation==dict(ordinal=ordinal,task_sha256=task,envelope_sha256=request['envelope_sha256']),'generation ordinal/task/envelope')
                wire_raw=member(client+f'request-{ordinal:04d}.observed-response.json')
                wire,wfields=metadata(wire_raw,['type','request_id','identity_sha256','envelope_sha256'])
                need(wire==dict(type='completion',request_id=ordinal,identity_sha256=ip,envelope_sha256=request['envelope_sha256']),'observed response RPC join')
                accepted_row,af=metadata(member(client+f'request-{ordinal:04d}.accepted.json'),['request_id','response_sha256'])
                need(accepted_row==dict(request_id=ordinal,response_sha256=sha(wire_raw.strip())),'accepted wire pin')
                need(af['completion']==wfields['completion']==gf['completion'],'opaque completion equality')
                mapping_rows.append(dict(member=n,sha256=full[n]['sha256'],policy_id=policy,world_id=worlds[wi],world_index=wi,decision_index=ordinal-offset,session_request_id=ordinal))
            offset=end
        names={n for n in full if n.startswith(client) and re.fullmatch(r'request-\d{4}\.episode\.json',n[len(client):])}
        need(names=={client+f'request-{i:04d}.episode.json' for i in range(offset)},'complete mapping member roster')
        need(len(evidence['joins'])==offset,'complete validator join roster')
        summaries.append(dict(policy_id=policy,requests=offset,recorded_campaign_root=resolved,validator_sha256=full['launch/gpu/controller/'+policy+'-evidence.json']['sha256']))
    report=json.loads(read(A/'audit-attempt03/report.json','27a512367ba45499ca7b6339c086682a84c0129a11eca4292217b5a897c45466'))
    need(report['status']!='passed','retained failed attempt')
    facts=dict(status='narrow_saved_metadata_joins_match',scientific_acceptance=False,
        checks=checks,output_path_binding=binding,mapping_count=len(mapping_rows),policies=summaries,
        source_pins=source_pins,mapping_rows=mapping_rows,failures=[],
        scope='Independent path/session/RPC diagnosis only; task/state/completion payloads compared opaquely, not decoded. No numerical/scientific or project execution; no full acceptance.')
    dump(OUT/'facts.json',facts)
    print(json.dumps({k:v for k,v in facts.items() if k not in ['mapping_rows','source_pins']},sort_keys=True))
except BaseException:
    with (OUT/'failure.txt').open('x') as f:f.write(traceback.format_exc())
    raise
finally:
    dump(OUT/'read-pins.json',pins)
