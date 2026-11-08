#!/usr/bin/env python3
"""Prepared exclusive CPU custody: --job-id ID --packet-sha256 PIN.

Only the new CPU root is inventoried/transferred. Older submission journals are
read solely for 37-job/76-journal accounting. Two remote inventories and exact
external/repository copies retain every input, log and worker result. Failed or
partial custody blocks blind reruns. No tokenization, inference, job mutation or
independent acceptance. capacity_rejected with exit 0 is retained as such.
"""
from pathlib import Path
import argparse, collections, datetime, fcntl, hashlib, json, os, re, shlex, shutil, stat, subprocess, time

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH = REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/launch01'
LAUNCH_EXT = BASE/'ACE-LLM-intervention-screen-prompt-cpu-20261108-01'
LOCAL = LAUNCH.parent/'terminal01'
EXT = BASE/'ACE-LLM-intervention-screen-prompt-cpu-terminal-20261108-01'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01'
HELPER = '/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
MAX_BYTES, MAX_FILES = 512*1024**2, 4096

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(value): return hashlib.sha256(value).hexdigest()
def raw(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def text(value): return value.decode(errors='replace') if isinstance(value,bytes) else value
def read(path, cap=32*1024**2):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as handle:
        before=os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular input required')
        value=handle.read(cap+1);after=os.fstat(handle.fileno())
    need(len(value)<=cap and (before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_ino,after.st_size,after.st_mtime_ns),'input changed during capture')
    return value
def inventory(root, deadline):
    need(root.is_dir() and not root.is_symlink(),'ordinary custody root required')
    result={};size=0
    for path in sorted(root.rglob('*')):
        need(time.monotonic()<deadline and not path.is_symlink(),'custody deadline or symlink')
        if path.is_dir():continue
        value=read(path,64*1024**2);size+=len(value)
        need(size<=MAX_BYTES and len(result)<MAX_FILES,'complete custody exceeds fixed bound')
        result[path.relative_to(root).as_posix()]=dict(bytes=len(value),sha256=sha(value))
    return result
def tres(value):
    pairs=[item.split('=',1) for item in value.split(',') if item]
    need(all(len(p)==2 and p[0] and p[1] for p in pairs) and len(dict(pairs))==len(pairs),'malformed resource inventory')
    return dict(pairs)
def gpu_count(resources, cpus, elapsed):
    if not resources:return 0 if cpus==elapsed==0 else None
    values={k:v for k,v in resources.items() if k.startswith('gres/')}
    if any(not v.isdigit() or not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in values.items()):return None
    typed=sum(int(v) for k,v in values.items() if k.startswith('gres/gpu:'))
    generic=int(values['gres/gpu']) if 'gres/gpu' in values else None
    if generic is not None and any(k.startswith('gres/gpu:') for k in values) and generic!=typed:return None
    return generic if generic is not None else typed

REMOTE = r'''
from pathlib import Path
import datetime,hashlib,json,os,stat,subprocess,time
root=Path(ROOT);deadline=time.monotonic()+150
def need(ok,message):
 if not ok:raise ValueError(message)
def read(path,cap):
 need(time.monotonic()<deadline,'remote inventory deadline exceeded')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as h:
  before=os.fstat(h.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular raw file required')
  value=h.read(cap+1);after=os.fstat(h.fileno())
 need(len(value)<=cap and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'raw file changed')
 return value
v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={})
commands={
 'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],
 'details':['sacct','-X','-j',JOB,'-n','-P','--format=Partition,QOS,TimelimitRaw,ReqTRES%180,User'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'census':['sacct','-X','-u','paco0228','-S','2026-11-02','-n','-P','--format=JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180'],
 'job':['scontrol','show','job','-o',JOB]}
for name,command in commands.items():
 p=subprocess.run(command,capture_output=True,text=True,timeout=min(20,max(.1,deadline-time.monotonic())))
 v['results'][name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
need(all(v['results'][k]['returncode']==0 for k in ('accounting','queue','census')),'terminal scheduler observation failed')
rows=[line.split('|') for line in v['results']['accounting']['stdout'].splitlines() if line.startswith(JOB+'|')]
need(len(rows)==1 and len(rows[0])>=9 and rows[0][3].split()[0] in
 ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'),'job not conclusively terminal')
need(not any(line.split('|')[0]==JOB or '|ace_llm' in line for line in v['results']['queue']['stdout'].splitlines()),'related job active; custody not frozen')
v['parent_accounting']=rows[0]
v['related_accounting']=sorted([line.split('|')[:8] for line in v['results']['census']['stdout'].splitlines()
 if '|ace_llm' in line],key=lambda row:int(row[0]))
need(root.is_dir() and not root.is_symlink(),'retained CPU root required')
need(hashlib.sha256(read(root/'inputs/packet.json',256*1024)).hexdigest()==PIN,'remote packet pin differs')
size=0
for path in sorted(root.rglob('*')):
 need(not path.is_symlink(),'raw symlink forbidden')
 if path.is_dir():continue
 value=read(path,64*1024**2);size+=len(value)
 need(size<=512*1024**2 and len(v['files'])<4096,'complete raw custody exceeds fixed bound')
 v['files']['launch/'+path.relative_to(root).as_posix()]=dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
# Journal hashes only; no old campaign tree is inventoried or transferred.
journals={}
for folder in root.parent.glob('ace_llm_intervention*'):
 for path in folder.glob('submission.*.json'):
  need(len(journals)<128,'journal census exceeds bound')
  journals[str(path)]=hashlib.sha256(read(path,4*1024**2)).hexdigest()
v['submission_journals']=journals
print(json.dumps(v))
'''

def token_summary(root):
    path=root/'cpu01/worker/artifacts/tokenization.json'
    if not path.exists():return dict(present=False),[]
    value=read(path);data=json.loads(value);rows=data['rows']
    need(type(rows) is list and len(rows)<=120,'bounded token-result roster required')
    prompts=[len(r['prompt_ids']) for r in rows if type(r.get('prompt_ids')) is list]
    commands=[len(ids) for r in rows if type(r.get('commands')) is dict for ids in r['commands'].values() if type(ids) is list]
    return dict(present=True,tokenization_sha256=sha(value),records=len(rows),
        status_counts=dict(collections.Counter(str(r.get('status','unknown')) for r in rows)),
        prompts_with_ids=len(prompts),command_sequences=len(commands),max_prompt_tokens=max(prompts,default=None),
        max_command_plus_eos_tokens=max(commands,default=None),
        max_prompt_plus_generation_reserve=max(prompts)+32 if prompts else None),rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True);parser.add_argument('--packet-sha256',required=True)
    args=parser.parse_args()
    need(re.fullmatch('[1-9][0-9]*',args.job_id) and re.fullmatch('[0-9a-f]{64}',args.packet_sha256),'explicit job ID and packet pin required')
    for name in ('packet-metadata.json','submission.result.json','inputs/packet.json'):
        need(read(LAUNCH/name)==read(LAUNCH_EXT/name),'local/external launch binding differs')
    meta_raw=read(LAUNCH/'packet-metadata.json');meta=json.loads(meta_raw)
    result=json.loads(read(LAUNCH/'submission.result.json'));packet=json.loads(read(LAUNCH/'inputs/packet.json'))
    need(meta['remote_root']==ROOT and meta['output']==ROOT+'/cpu01' and packet['output']==ROOT+'/cpu01'
        and meta['screen_packet_sha256']==args.packet_sha256==sha(read(LAUNCH/'inputs/packet.json'))
        and result.get('returncode')==0 and result.get('job_id')==args.job_id,'submission/packet unresolved or differs')
    admission=meta['admission']
    need(admission['closed_cpu_seconds']==10368 and admission['closed_gpu_slice_seconds']==1483
        and admission['cumulative_preparation_cpu_allowance']==3600 and admission['new_cpu_reservation']==600
        and admission['new_gpu_slice_reservation']==0 and admission['cpu_admission']==14568
        and admission['gpu_slice_admission']==1483 and len(meta['prior_accounting'])==36
        and len(meta['prior_journals'])==74,'original admitted ledger differs')
    source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,JOB=args.job_id,PIN=args.packet_sha256).items())+'\n'+REMOTE
    locks=[];claimed=[];started=time.monotonic();process=time.process_time();deadline=started+900
    def save(name,value):
        for root in claimed:
            with (root/name).open('xb') as h:h.write(raw(value));h.flush();os.fsync(h.fileno())
    def observe(name):
        try:
            p=subprocess.run(['bash',HELPER,'run',shlex.join(['python3','-I','-B','-'])],input=source,
                env=dict(os.environ,CURC_SOCKET='/tmp/defab-curc.sock'),capture_output=True,text=True,
                timeout=min(180,max(.1,deadline-time.monotonic())))
            transport=dict(source=source,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=utc())
        except subprocess.TimeoutExpired as exc:
            save(name,dict(source=source,returncode=None,stdout=text(exc.stdout),stderr=text(exc.stderr),timed_out=True,utc=utc()));raise
        save(name,transport);need(p.returncode==0,'terminal observation failed; retained, no blind retry')
        return json.loads(p.stdout)
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            h=open(BASE/name,'r+')
            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:h.close();raise
            locks.append(h)
        for dest in (EXT,LOCAL):need(not dest.exists() and not dest.is_symlink(),'terminal01 already claimed; no blind retry')
        for dest in (EXT,LOCAL):
            dest.mkdir();claimed.append(dest);(dest/'remote').mkdir()
            with (dest/'pull_terminal.py').open('xb') as h:h.write(Path(__file__).read_bytes())
        save('collector-start.json',dict(utc=utc(),job_id=args.job_id,packet_sha256=args.packet_sha256,
            script_sha256=sha(Path(__file__).read_bytes()),remote_root=ROOT,deadline_seconds=900,
            held_locks=[h.name for h in locks],retry_allowed=False))
        before=observe('inventory_transport.json');size=sum(v['bytes'] for v in before['files'].values())
        for parent in (BASE,LOCAL.parent):
            fs=os.statvfs(parent);need(fs.f_bavail*fs.f_frsize>size*2+2*1024**3,'insufficient complete two-copy custody space')
        try:
            p=subprocess.run(['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','ControlMaster=no','-o','BatchMode=yes',
                '-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no','-o','PubkeyAuthentication=no',
                'paco0228@login.rc.colorado.edu:'+ROOT,str(EXT/'remote/launch')],capture_output=True,text=True,
                timeout=min(300,max(.1,deadline-time.monotonic())))
            save('transfer-launch.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=utc()))
        except subprocess.TimeoutExpired as exc:
            save('transfer-launch.json',dict(returncode=None,stdout=text(exc.stdout),stderr=text(exc.stderr),timed_out=True,utc=utc()));raise
        need(p.returncode==0,'partial transfer retained; no blind retry')
        actual=inventory(EXT/'remote',deadline)
        need(actual==before['files'],'complete external raw membership/bytes differ from first inventory')
        after=observe('inventory_after_transport.json')
        need(after['files']==actual and after['parent_accounting'][:9]==before['parent_accounting'][:9]
            and after['related_accounting']==before['related_accounting']
            and after['submission_journals']==before['submission_journals'],'remote evidence changed during transfer')
        shutil.copytree(EXT/'remote/launch',LOCAL/'remote/launch')
        need(inventory(LOCAL/'remote',deadline)==actual,'complete repository raw copy differs')

        retained=EXT/'remote/launch';errors=[]
        r=before['parent_accounting'];need(r[5].isdigit() and r[6].isdigit(),'unknown actual CPU accounting')
        cpus,elapsed=int(r[5]),int(r[6]);allocated=tres(r[8]);gpu=gpu_count(allocated,cpus,elapsed)
        details=before['results']['details'];dr=[line.split('|')[:5] for line in details['stdout'].splitlines() if line.strip()]
        requested_ok=False
        if details['returncode']==0 and len(dr)==1 and dr[0][:3]==['acpu','cpu-normal','5'] and dr[0][4]=='paco0228':
            req=tres(dr[0][3]);requested_ok=(req.get('cpu')=='2' and req.get('node')=='1'
                and req.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in req))
        allocation_path=retained/'cpu01/allocation.json';no_requeue=False
        if allocation_path.exists():
            allocation=json.loads(read(allocation_path));fields=dict(re.findall(r'([^\s=]+)=(\S+)',allocation.get('stdout','')))
            no_requeue=(allocation.get('returncode')==0 and fields.get('JobId')==args.job_id
                        and fields.get('Requeue')=='0' and fields.get('Restarts')=='0')
        resource_match=(requested_ok and no_requeue and r[2]=='ace_llm_screen_prompt_cpu01' and r[7]=='ucb736_asc2'
            and cpus==2 and allocated.get('cpu')=='2' and allocated.get('node')=='1'
            and allocated.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in allocated)
            and gpu==0 and 0<=elapsed<=300)
        if not resource_match:errors.append('requested/actual CPU allocation, no-requeue evidence or resource cap differs')
        if gpu is None:errors.append('GPU allocation accounting uncertain; no zero-use inference')
        intent_raw=read(retained/'submission.intent.json');remote_result_raw=read(retained/'submission.result.json')
        intent=json.loads(intent_raw);remote_result=json.loads(remote_result_raw)
        need(remote_result==result and result['intent_sha256']==sha(intent_raw)
            and intent['screen_packet_sha256']==args.packet_sha256 and intent['remote_root']==ROOT
            and intent['source_commit']==meta['source_commit'] and intent['prior_accounting']==meta['prior_accounting']
            and intent['prior_journals']==meta['prior_journals'],'submission lineage differs')
        expected_rows=sorted([row[:8] for row in meta['prior_accounting']]+[[r[0],*r[2:9]]],key=lambda row:int(row[0]))
        expected_journals={**meta['prior_journals'],ROOT+'/submission.intent.json':sha(intent_raw),
                           ROOT+'/submission.result.json':sha(remote_result_raw)}
        if len(before['related_accounting'])!=37 or before['related_accounting']!=expected_rows:errors.append('37-job extension differs')
        if len(before['submission_journals'])!=76 or before['submission_journals']!=expected_journals:errors.append('76-journal extension differs')
        cost=cpus*elapsed;gcost=None if gpu is None else gpu*elapsed
        ledger_cpu=sum(int(row[4])*int(row[5]) for row in before['related_accounting'])
        ledger_gpu_counts=[gpu_count(tres(row[7]),int(row[4]),int(row[5])) for row in before['related_accounting']]
        ledger_gpu=None if any(n is None for n in ledger_gpu_counts) else sum(n*int(row[5]) for n,row in zip(ledger_gpu_counts,before['related_accounting']))
        if ledger_cpu!=10368+cost or ledger_gpu is None or gcost is None or ledger_gpu!=1483+gcost:errors.append('actual cumulative scheduler cost differs or uncertain')
        expected={'launch/inputs/'+n:p for n,p in meta['input_files'].items()}
        expected.update({'launch/'+n:p for n,p in meta['root_files'].items()});expected['launch/packet-metadata.json']=sha(meta_raw)
        mismatches=[n for n,p in expected.items() if actual.get(n,{}).get('sha256')!=p]
        input_names={n.removeprefix('launch/inputs/') for n in actual if n.startswith('launch/inputs/')}
        if input_names!=set(meta['input_files']):mismatches.append('exact inputs directory roster')
        summary,records=token_summary(retained)
        parents={}
        for name in ('cpu01/terminal.json','cpu01/worker/artifacts/terminal.json','cpu01/failure.json',
                     'cpu01/worker/failure.json','cpu01/worker.cleanup.json','cpu01/worker.resources.json'):
            path=retained/name
            if path.exists():parents[name]=json.loads(read(path,256*1024))
        parent=parents.get('cpu01/terminal.json',{});worker=parents.get('cpu01/worker/artifacts/terminal.json',{})
        cleanup=parents.get('cpu01/worker.cleanup.json',{})
        terminal_binding=(parent.get('packet_sha256')==worker.get('packet_sha256')==args.packet_sha256
            and parent.get('worker_terminal_sha256')==actual.get('launch/cpu01/worker/artifacts/terminal.json',{}).get('sha256')
            and worker.get('tokenization_sha256')==summary.get('tokenization_sha256'))
        counts=summary.get('status_counts',{})
        count_binding=(summary.get('records')==worker.get('fixtures')==120 and worker.get('fits')==counts.get('fits',0)
            and worker.get('overflows')==counts.get('overflow',0) and worker.get('failures')==counts.get('failed',0)
            and sum(counts.get(k,0) for k in ('fits','overflow','failed'))==120)
        scheduler_ok=r[3]=='COMPLETED' and r[4]=='0:0'
        clean=(cleanup.get('returncode')==0 and cleanup.get('group_still_exists') is False
               and cleanup.get('kill_sent') is False and cleanup.get('errors')==[])
        complete=(scheduler_ok and terminal_binding and count_binding and clean
            and not any(name.endswith('failure.json') for name in parents))
        outcome='execution_failed_or_incomplete'
        if complete:
            outcome='fixture_failures' if counts.get('failed',0) else 'context_overflow' if counts.get('overflow',0) else 'all_fixtures_fit_reported'
            expected_fit=outcome=='all_fixtures_fit_reported'
            if parent.get('status')!=('capacity_measured' if expected_fit else 'capacity_rejected') or parent.get('capacity_qualified') is not expected_fit or worker.get('capacity_qualified') is not expected_fit:
                outcome='inconsistent_capacity_report';errors.append('parent/worker capacity classification differs')
        save('accounting-extension.json',dict(prior_jobs=36,terminal_jobs=len(before['related_accounting']),
            prior_journals=74,submission_journals=len(before['submission_journals']),prior_closed_cpu_seconds=10368,
            prior_closed_gpu_slice_seconds=1483,new_cpu_seconds=cost,new_gpu_slice_seconds=gcost,
            related_accounting=before['related_accounting'],journal_pins=before['submission_journals'],errors=errors))
        disposition=dict(utc=utc(),status='complete_custody_independent_review_pending',job_id=args.job_id,
            packet_sha256=args.packet_sha256,scheduler_state=r[3],scheduler_exit=r[4],account=r[7],
            allocated_cpus=cpus,elapsed_seconds=elapsed,gpu_count=gpu,alloc_tres=r[8],resource_contract_matched=resource_match,
            allocated_cpu_seconds=cost,gpu_slice_seconds=gcost,closed_cpu_seconds=10368+cost,
            closed_gpu_slice_seconds=None if gcost is None else 1483+gcost,cumulative_preparation_cpu_allowance=3600,
            cpu_admission_before_run=14568,gpu_admission_before_run=1483,cpu_reservation_released=600,
            cpu_admission_after_release=10368+cost+3600,outstanding_cpu_reservation=0,
            cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,terminal_jobs=len(before['related_accounting']),
            submission_journals=len(before['submission_journals']),accounting_errors=errors,input_pin_mismatches=mismatches,
            capacity_outcome=outcome,reported_parent_status=parent.get('status'),token_summary=summary,
            terminal_record_pins={n:actual['launch/'+n]['sha256'] for n in parents},
            raw_files=len(actual),raw_bytes=size,files=actual,raw_copies_verified=2,remote_inventories_matched=True,
            scientific_qualified=False,capacity_qualified=False,independent_acceptance_required=True,retry_allowed=False,
            collector_elapsed_seconds=time.monotonic()-started,collector_process_cpu_seconds=time.process_time()-process,
            cost_scope='Collector process only; excludes transfer child processes and prior preparation, not whole-cycle CPU.')
        need(time.monotonic()<deadline,'collector deadline exceeded')
        save('disposition.json',disposition)
        print(json.dumps({k:v for k,v in disposition.items() if k!='files'},sort_keys=True,indent=2))
    except BaseException as exc:
        if claimed:save('collector_failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),retry_allowed=False))
        raise
    finally:
        for h in reversed(locks):fcntl.flock(h,fcntl.LOCK_UN);h.close()

if __name__ == '__main__': main()
