"""Independent local metadata/hash/source-static review. Never imports project code."""
import ast
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

WORK = Path(__file__).parent
REPO = Path('/Users/pat/code/ACE')
ROOT = REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
TERM = ROOT/'terminal01'
EXT = BASE/'ACE-LLM-intervention-scientific-campaign-gpu-terminal-20261108-01'
LAUNCH = ROOT/'launch01'
PINS = {}
CHECKS = []
START = time.monotonic()
CPU_START = time.process_time()

def check(ok, label, **details):
    CHECKS.append(dict(check=label, passed=bool(ok), **details))
    if not ok:
        (WORK/'failed-checks.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
        raise ValueError(label + ': ' + json.dumps(details))

def sha(b): return hashlib.sha256(b).hexdigest()
def canon(d): return (json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def pairs(items):
    d={}
    for k,v in items:
        if k in d: raise ValueError('duplicate JSON key '+k)
        d[k]=v
    return d
def decode(b): return json.loads(b, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def ident(s): return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
def read(p, contents=True):
    p=Path(p).absolute()
    if p.is_symlink(): raise ValueError('symlink '+str(p))
    h=hashlib.sha256(); blocks=[]
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode): raise ValueError('nonregular '+str(p))
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
            if contents: blocks.append(b)
        after=os.fstat(f.fileno())
    if ident(before)!=ident(after) or ident(after)!=ident(p.lstat()): raise ValueError('changed during read '+str(p))
    entry=dict(bytes=after.st_size,sha256=h.hexdigest())
    if str(p) in PINS and PINS[str(p)]!=entry: raise ValueError('changed since earlier read '+str(p))
    PINS[str(p)]=entry
    return b''.join(blocks) if contents else entry
def j(p): return decode(read(p))
def inventory(root, exclude=()):
    files={}; dirs=[]
    for p in sorted(Path(root).rglob('*')):
        n=p.relative_to(root).as_posix()
        if p.is_symlink(): raise ValueError('symlink '+str(p))
        if p.is_dir(): dirs.append(n)
        elif n not in exclude: files[n]=read(p,False)
    return dict(files=files,directories=dirs)
def normalized(d): return dict(files=d['files'], directories=sorted(d['directories']))
def prefix_files(d,prefix): return {n[len(prefix):]:v for n,v in d.items() if n.startswith(prefix)}
def prefix_dirs(d,prefix): return sorted(n[len(prefix):] for n in d if n.startswith(prefix))
def hash_map(d): return {k:v['sha256'] for k,v in d.items()}

def main():
    anchors={
      'terminal01/custody-failure-inventory.json':'8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0',
      'terminal01/disposition.json':'545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507',
      'terminal01/inventory_transport.json':'f9c7dc2ecf1e0c91990e2c7d787abea37187fa520675ad580a1a7f5359f8d9d3',
      'terminal01/inventory_after_transport.json':'4303d7d4ef775ac133dde12b3d2ac875b7c31ed2382b45abd0c6cc6a55a188ac',
      'terminal01/remote-custody.json':'5224bf51110996d0a72f334dc976db7644f24d99e7500b99730ddf00eff19214',
      'preliminary-failure-review01/review.md':'45932dbb87627c21aed845716aada2d0db323c0fef26d2deb5a39b4802383bc0',
      'failure-observation01/stdout.json':'cce561b31d38d34e8a2f0e2f626d741bd845decd5c8d31817558993f20a65aa7',
      'launch01/cpu-baseline.json':'c21f43194ff61b30c0322acff99c20e71158d980a00a34a1685b87eac7f27125',
      'launch01/packet.json':'436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7',
      'launch01/inputs/qualify_llm_intervention_scientific_campaign_gpu.py':'2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'}
    for n,h in anchors.items(): check(read(ROOT/n,False)['sha256']==h,'user anchor '+n)
    seal=j(TERM/'custody-failure-inventory.json')
    check(set(seal['copies'])=={str(TERM),str(EXT)},'seal names exact two copies')
    views={}
    for root in (TERM,EXT):
        check(read(root/'custody-failure-inventory.json',False)==PINS[str(TERM/'custody-failure-inventory.json')],'failure seals identical '+str(root))
        views[str(root)]=inventory(root,('custody-failure-inventory.json',))
        check(views[str(root)]==seal['copies'][str(root)],'every sealed file/directory verified '+str(root),files=len(views[str(root)]['files']))
    check(views[str(TERM)]==views[str(EXT)],'terminal copies identical')
    print('VERIFIED terminal copies',len(views[str(TERM)]['files']),flush=True)
    for name in ('local-launch','external-launch'):
        ref=j(TERM/'coordinator'/f'{name}-reference.json')
        check(inventory(ref['path'])==ref['inventory'],'entire immutable coordinator '+name,files=len(ref['inventory']['files']))
    for folder in ('preliminary-failure-review01','failure-observation01','collection-command01','launch-command01','monitor01','watch01','packet-review01'):
        inventory(ROOT/folder)
    obs_ext=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-failure-observation-20261109-01'
    for name in ('stdout.json','remote_source.py','intent.json','result.json'):
        check(read(obs_ext/name,False)==read(ROOT/'failure-observation01'/name,False),'failure observation two copies '+name)
    for path,entry in j(ROOT/'preliminary-failure-review01/source-pins.json').items():
        check(read(path,False)==entry,'preliminary source pin '+path)

    bt=j(TERM/'inventory_transport.json'); at=j(TERM/'inventory_after_transport.json')
    before=decode(bt['stdout']); after=decode(at['stdout'])
    check(bt['returncode']==at['returncode']==0 and 'observation_error' not in before and 'observation_error' not in after,'complete successful before/after observations')
    check(bt['python_source']==at['python_source'],'same before/after observation source')
    for key in ('files','directories','journal_pins','upstream_pins','upstream_errors','original_hardlink_groups','related_accounting','parent_accounting'):
        check(before[key]==after[key],'before/after exact '+key)
    check(before['upstream_errors']=={},'zero upstream read errors')
    full=normalized(before)
    check(len(full['files'])==72557 and sum(v['bytes'] for v in full['files'].values())==1366843790,'full file and byte count')
    receipt=j(TERM/'remote-custody.json'); disp=j(TERM/'disposition.json')
    check(receipt==j(TERM/'backup-operations/receipt.json'),'backup receipt copies')
    check(full==receipt['full_inventory']==dict(files=disp['remote_files'],directories=disp['remote_directories']),'full maps independently equal')
    check(receipt['original_hardlink_groups']==before['original_hardlink_groups']==disp['original_hardlink_groups'],'hardlink alias metadata retained')
    invsha=sha(canon(full))
    check(all(receipt[k]==invsha for k in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256')),'all backup inventory digests recomputed',sha256=invsha)
    req=j(TERM/'backup-operations/request.json')
    check(req['observation']==before and sha(canon(before))==receipt['source_observation_sha256'],'backup original observation binding')
    check(req['roots']==receipt['roots'],'backup source root bindings')
    transport=j(TERM/'backup-operations/transport.json')
    check(transport['returncode']==0 and transport['remote_outcome_ambiguous'] is False,'backup transport successful unambiguous')
    check(receipt['full_backup_verified'] is True and receipt['full_local_copy'] is False,'full CURC backup and selected local scope')
    check(read(TERM/'remote_custody.py',False)['sha256']==receipt['helper_sha256']==read(TERM/'backup-operations/helper.py',False)['sha256'],'captured backup helper')
    framed=read(TERM/'backup-operations/request.bin'); line,framed=framed.split(b'\n',1); n=int(line)
    check(sha(framed[:n])==receipt['helper_sha256'] and decode(framed[n:])==req,'framed backup request source and metadata binding')
    local=receipt['local_inventory']; omitted=receipt['omitted_files']
    check(not (set(local['files'])&set(omitted)) and {**local['files'],**omitted}==full['files'],'selected/omitted exact partition')
    check(len(local['files'])==182 and sum(v['bytes'] for v in local['files'].values())==285130684,'local selected counts')
    check(len(omitted)==72375 and sum(v['bytes'] for v in omitted.values())==receipt['omitted_bytes']==1081713106,'omitted counts')
    for root in (TERM,EXT):
        view=views[str(root)]
        actual=dict(files=prefix_files(view['files'],'remote/'),directories=prefix_dirs(view['directories'],'remote/'))
        check(actual==local,'all 182 selected local files '+str(root))
    tar_unpadded=1024+512*len(full['directories'])+sum(512+((v['bytes']+511)//512)*512 for v in full['files'].values())
    check(receipt['backup_archive_bytes']==((tar_unpadded+10239)//10240)*10240,'USTAR exact archive size including 10240-byte record padding')
    print('VERIFIED complete maps and saved backup binding',flush=True)

    packet=j(LAUNCH/'packet.json'); intent=j(LAUNCH/'submission.intent.json'); result=j(LAUNCH/'submission.result.json')
    rintent=j(TERM/'remote/launch/submission.intent.json')
    baseline=j(LAUNCH/'cpu-baseline.json'); cpu=j(LAUNCH/'cpu-acceptance.json'); cpu_disp=j(LAUNCH/'cpu-disposition.json')
    cpu_before=decode(j(LAUNCH/'cpu-inventory.json')['stdout']); cpu_after=decode(j(LAUNCH/'storage-after.json')['stdout'])
    check(normalized(cpu_before)==normalized(cpu_after)==dict(files=cpu_disp['remote_files'],directories=cpu_disp['remote_directories']),'accepted CPU complete before/after/disposition maps')
    cpu_receipt=j(LAUNCH/'storage-receipt.json')
    labels={k:next(n for n,p in receipt['roots'].items() if p==v) for k,v in cpu_receipt['roots'].items()}
    def remap(n):
        label,sep,rest=n.partition('/')
        return labels[label]+sep+rest
    mapped={remap(n):v for n,v in cpu_disp['remote_files'].items()}
    mapped_dirs=sorted(remap(n) for n in cpu_disp['remote_directories'])
    check(len(mapped)==len(cpu_disp['remote_files']) and cpu_receipt['full_inventory']==normalized(cpu_before),'predecessor absolute roots yield lossless label mapping',labels=labels)
    actual_prior={n:v for n,v in full['files'].items() if not n.startswith('launch/')}
    actual_prior_dirs=sorted(n for n in full['directories'] if n!='launch' and not n.startswith('launch/'))
    check(mapped==actual_prior and mapped_dirs==actual_prior_dirs,'every predecessor raw path/digest/size/directory unchanged',files=len(mapped))
    check(before['upstream_pins']==after['upstream_pins']==packet['original_cpu_inputs']==disp['original_input_pins'],'all 72372 original input pins exact',pins=len(before['upstream_pins']))
    for label,expected in disp['preserved_upstream_roots'].items():
        check(prefix_files(full['files'],label+'/')==expected and prefix_dirs(full['directories'],label+'/')==disp['preserved_upstream_directories'][label],'full upstream no additions/deletions/changes '+label,files=len(expected))
    abs_files={str(Path(receipt['roots'][n.split('/')[0]])/n.split('/',1)[1]):v for n,v in full['files'].items() if '/' in n}
    check(all(abs_files.get(n,{}).get('sha256')==h for n,h in packet['original_cpu_inputs'].items()),'absolute upstream pins match actual full inventory')
    cpu_out=hash_map(prefix_files(full['files'],'cpu/cpu/'))
    campaign=prefix_files(cpu_out,'worker/artifacts/campaign/')
    noncampaign={n:v for n,v in cpu_out.items() if not n.startswith('worker/artifacts/campaign/')}
    check(noncampaign==cpu['cpu_output_files'],'all accepted noncampaign CPU files unchanged',files=len(noncampaign))
    check(campaign==cpu['campaign_files'],'all accepted campaign CPU files unchanged',files=len(campaign))
    terminal=j(LAUNCH/'cpu-phase-terminal.json')
    check(read(LAUNCH/'cpu-phase-terminal.json',False)['sha256']==cpu['phase_terminal_sha256']==cpu_out[cpu['phase_terminal_path']],'accepted CPU phase terminal exact')
    check(terminal['completed_cpu_cells']==120 and terminal['declared_cells']==192 and terminal['gpu_services_started']==0 and terminal['private_scores_computed'] is False,'CPU120 preserved public metadata')
    check(terminal['attempt_pins']==cpu['attempt_pins'],'CPU five attempt pins preserved')
    for i,pid in enumerate(('uniform_random','menu_coverage','local_information','target_risk','learned_ranker')):
        check(all(campaign[f'policy-{i:02d}/{f}']==cpu['attempt_pins'][pid] for f in ('attempt.json','attempt.candidate.json')),'CPU attempt and candidate '+pid)
    forbidden=[n for n in list(full['files'])+full['directories'] if n=='launch/gpu' or n.startswith('launch/gpu/') or any(p in ('policy-05','policy-06','policy-07') for p in n.split('/'))]
    check(not forbidden,'no GPU output or policy05..07 path anywhere in full maps',matches=forbidden)
    check(set(rintent)-set(intent)=={'fresh_readiness','remote_utc'} and all(rintent[k]==v for k,v in intent.items()),'remote intent exact local extension')
    check(rintent['fresh_readiness']['related_accounting']==baseline['related_accounting'] and rintent['fresh_readiness']['journal_pins']==baseline['journal_pins'],'submission fresh history exact baseline')
    check(result['job_id']=='35498218' and result['ambiguous'] is False and result['returncode']==0 and result['stdout']=='35498218\n','one unambiguous successful submission')
    check(result['intent_sha256']==read(TERM/'remote/launch/submission.intent.json',False)['sha256'],'result binds exact remote intent')
    check(intent['packet_commit']=='41bda8e1d9e925eca68f1aa6b16f4d616d1cf7db' and intent['packet_sha256']==sha(read(LAUNCH/'packet.json')),'immutable packet identity')
    check(intent['pins']=={**packet['new_input_files'],'packet.json':intent['packet_sha256']},'intent pins exact packet closure')
    for n,h in intent['pins'].items():
        check(read(LAUNCH/n,False)['sha256']==full['files']['launch/'+n]['sha256']==h,'immutable launch member '+n)
    for local_key,remote_name in (('local_namespace','namespace.json'),('original_local_namespace','original-namespace.json')):
        ci=j(TERM/'collector-intent.json'); expected=full['files'][remote_name]
        check(read(ci[local_key],False)==expected,'unchanged namespace local/remote '+remote_name)
    expected_launch=set(intent['pins'])|{'submission.intent.json','submission.result.json','readiness.json','logs/35498218.err','logs/35498218.out'}
    check(read(LAUNCH/'readiness.json',False)==full['files']['launch/readiness.json'],'pre-submission readiness metadata unchanged')
    check(set(prefix_files(full['files'],'launch/'))==expected_launch,'entire remote launch has only inputs, readiness, journals and logs',files=len(expected_launch))
    check(prefix_dirs(full['directories'],'launch/')==['inputs','logs'],'no new output directories')
    print('VERIFIED CPU120, all ancestor/input maps, no new policy paths',flush=True)

    job='35498218'; rows=before['related_accounting']; parent=before['parent_accounting']; old=baseline['related_accounting']
    check(len(rows)==47 and len({r[0] for r in rows})==47 and [r for r in rows if r[0]!=job]==old and [r for r in rows if r[0]==job]==[parent],'47 jobs exact unchanged 46 plus failed parent')
    check(all(r[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','NODE_FAIL'} for r in rows),'all related jobs terminal')
    expected_journals={**baseline['journal_pins'],packet['remote_root']+'/submission.intent.json':result['intent_sha256'],packet['remote_root']+'/submission.result.json':read(LAUNCH/'submission.result.json',False)['sha256']}
    check(len(expected_journals)==96 and before['journal_pins']==expected_journals,'96 journals exact unchanged94 plus current2')
    for snapshot in (before,after):
        rawrows=[r.split('|') for r in snapshot['results']['related_accounting']['stdout'].splitlines() if '|ace_llm' in r]
        check(rawrows==rows,'saved raw sacct census parsed equals 47 rows')
        q=snapshot['results']['queue']
        check(q['returncode']==0 and all(not r.split('|')[1].lower().startswith('ace') and r.split('|')[0]!=job for r in q['stdout'].splitlines() if r),'saved queue no active ACE job')
    def tres(s):
        parts=[v.split('=',1) for v in s.split(',') if v];d=dict(parts)
        if len(parts)!=len(d): raise ValueError('duplicate TRES')
        return d
    def gpu_count(d):
        typed=sum(int(v) for k,v in d.items() if k.startswith('gres/gpu:'))
        if 'gres/gpu' in d:
            check(not typed or int(d['gres/gpu'])==typed,'aggregate/typed GPU consistency')
            return int(d['gres/gpu'])
        return typed
    def cost(rs): return (sum(int(r[4])*int(r[5]) for r in rs),sum(gpu_count(tres(r[7]))*int(r[5]) for r in rs))
    check(cost(old)==(29261,2505)==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']),'predecessor totals recomputed')
    check(cost([parent])==(980,245) and cost(rows)==(30241,2750),'actual allocation charges recomputed')
    check(baseline['preparation_allowance_cpu_seconds']==3600 and baseline['outstanding_reservation']==0,'single unchanged preparation allowance')
    obs=j(ROOT/'failure-observation01/stdout.json'); commands=obs['commands']; sc=commands['details']
    check(sc['returncode']==0 and sc['argv']==['scontrol','show','job',job,'-o'],'saved successful scontrol exact job')
    fields=dict(re.findall(r'([^\s=]+)=(\S+)',sc['stdout']))
    expected_fields={'JobId':job,'JobName':parent[1],'Account':'ucb736_asc2','QOS':'gpu-testing','Partition':'aa100','JobState':'FAILED','ExitCode':'1:0','Requeue':'0','Restarts':'0','NumNodes':'1','NumCPUs':'4','NumTasks':'1','CPUs/Task':'4','RunTime':'00:04:05','TimeLimit':'00:48:00','Command':packet['remote_root']+'/job.sbatch','StdErr':packet['remote_root']+'/logs/'+job+'.err','StdOut':packet['remote_root']+'/logs/'+job+'.out'}
    check(all(fields.get(k)==v for k,v in expected_fields.items()),'actual scontrol resource/no-requeue/job/log identity')
    allocated=tres(parent[7]); requested=tres(fields['ReqTRES'])
    check(allocated==requested==tres(fields['AllocTRES']) and allocated=={'billing':'58','cpu':'4','gres/gpu:a100_3g.20gb':'1','mem':'15G','node':'1'},'actual and requested TRES exact')
    detailrow=before['results']['details']['stdout'].strip().split('|')
    check(detailrow[:4]==[job,'aa100','gpu-testing','48'] and detailrow[5]=='paco0228' and tres(detailrow[4])==allocated,'saved detailed sacct request')
    check(before['results']['details']==after['results']['details'],'detailed sacct stable')
    steptext=before['results']['steps']['stdout']; steps=[r.split('|') for r in steptext.splitlines()]
    check(steptext==after['results']['steps']['stdout']==commands['accounting']['stdout'] and steps[0][:8]==parent and {r[0] for r in steps}=={job,job+'.batch',job+'.extern'},'job/step rows independently agree')
    check(int(parent[5])==int((datetime.datetime.fromisoformat(fields['EndTime'])-datetime.datetime.fromisoformat(fields['StartTime'])).total_seconds())==245,'actual start/end lifecycle elapsed')
    check(fields['SubmitTime']<=fields['StartTime']<fields['EndTime'],'actual submitted/started/ended lifecycle')
    check(all(x['results']['job']['returncode']==1 and 'Invalid job id' in x['results']['job']['stderr'] for x in (before,after)),'later scontrol unavailable, not contradictory')
    error='actual/requested GPU identity/resource/no-requeue/wall contract differs or unavailable'
    check(disp['errors']==disp['allocation']['errors']==[error] and disp['allocation']['allocation_fields']=={} and disp['allocation']['allocation_source_sha256'] is None,'only original collector resource interface disagreement')
    check(disp['allocation']['allocated_cpu_seconds']==980 and disp['allocation']['gpu_slice_seconds']==245 and disp['closed_cpu_seconds']==30241 and disp['closed_gpu_slice_seconds']==2750 and disp['cpu_admission_after_release']==33841,'original disposition actual charges agree')
    check(j(ROOT/'collection-command01/result.json')['returncode']==1 and j(TERM/'collector_failure.json')['acceptance'] is False,'original collector failure retained')
    for n,item in obs['files'].items():
        b=read(TERM/'remote/launch'/n)
        check(len(b)==item['bytes'] and sha(b)==item['sha256'] and b==item['text'].encode(),'observation raw log/journal binding '+n)
    check(obs['gpu_output_exists'] is False,'early observation GPU output absent')
    err=obs['files']['logs/'+job+'.err']['text']
    check('TimeoutError: campaign supervisor signal 14' in err and all('line '+str(n) in err for n in (677,346,198,151,98,673)),'exact failure traceback boundary')
    check('--controller' not in read(LAUNCH/'job.sbatch').decode(),'submitted invocation is initial authentication, no controller')
    print('VERIFIED resource disagreement and exact 47/96 accounting',flush=True)

    summary=dict(schema='ace-gpu01-independent-metadata-review-checks-v1',required_findings=0,checks=CHECKS,
        full_files=72557,full_bytes=1366843790,selected_local_files=182,selected_local_bytes=285130684,
        terminal_capsule_files_per_copy=len(views[str(TERM)]['files'])+1,
        full_inventory_sha256=invsha,predecessor_full_files=len(mapped),cpu_output_files=len(noncampaign),cpu_campaign_files=len(campaign),
        resource_fields=fields,allocated_tres=allocated,requested_tres=requested,parent_accounting=parent,
        related_accounting=rows,journal_pins=expected_journals,original_collector_result=j(ROOT/'collection-command01/result.json'),
        original_collector_process_cpu_seconds=disp['collector_process_cpu_seconds'],original_collector_wall_seconds=disp['collector_wall_seconds'],
        review_process_cpu_seconds=time.process_time()-CPU_START,review_wall_seconds=time.monotonic()-START,
        observation_times={'before_observation_utc':before['utc'],'before_transport_complete_utc':bt['utc'],'after_observation_utc':after['utc'],'after_transport_complete_utc':at['utc']},
        remote_actions=0,original_source_executions=0,models_imported=0,tests_run=0,scored_outcomes_decoded=False)
    (WORK/'checks.json').write_bytes(canon(summary));(WORK/'source-pins.json').write_bytes(canon(PINS))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','related_accounting','journal_pins','resource_fields')},indent=2))

if __name__=='__main__': main()
