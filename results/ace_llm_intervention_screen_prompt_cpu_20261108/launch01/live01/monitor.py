#!/usr/bin/env python3
"""Prepared read-only CPU snapshot: --job-id ID --packet-sha256 PIN.

Call early/periodically; never submits, cancels, reconnects or retries. The
120/260/280/290 clocks are observations for parent diagnosis. Snapshot receipts
and this script's bytes are retained in both launch copies. No token IDs printed.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shlex, subprocess

LOCAL = Path('/Users/pat/code/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108/launch01')
EXT = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-prompt-cpu-20261108-01')
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01'
HELPER = '/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(value): return hashlib.sha256(value).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def text(value): return value.decode(errors='replace') if isinstance(value, bytes) else value

REMOTE = r'''
from pathlib import Path
import collections,datetime,hashlib,json,os,stat,subprocess,time
root=Path(ROOT);out=root/'cpu01';deadline=time.monotonic()+70
def need(ok,message):
 if not ok:raise ValueError(message)
def read(path,cap):
 need(time.monotonic()<deadline,'snapshot deadline exceeded')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as h:
  before=os.fstat(h.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular snapshot required')
  value=h.read(cap+1);after=os.fstat(h.fileno())
 need(len(value)<=cap and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'live file changed during read')
 return value
need(root.is_dir() and not root.is_symlink(),'retained launch root missing')
need(hashlib.sha256(read(root/'inputs/packet.json',256*1024)).hexdigest()==PIN,'remote packet differs')
result=json.loads(read(root/'submission.result.json',128*1024))
intent=read(root/'submission.intent.json',4*1024**2)
need(result.get('returncode')==0 and result.get('job_id')==JOB and result.get('intent_sha256')==hashlib.sha256(intent).hexdigest(),
 'remote submission identity differs')
v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,packet_sha256=PIN,
 files={},scientific_qualified=False,independent_acceptance_required=True)
for label,command in {
 'queue':['squeue','-j',JOB,'-h','-o','%i|%j|%T|%R|%a'],
 'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,AllocTRES%180,MaxRSS,TotalCPU'],
 'job':['scontrol','show','job','-o',JOB]}.items():
 p=subprocess.run(command,capture_output=True,text=True,timeout=min(15,max(.1,deadline-time.monotonic())))
 v[label]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
fields=('schema','status','packet_sha256','stage_sha256','source_commit','started_utc','started_monotonic',
 'monotonic','pid','phase','utc','elapsed_seconds','process_cpu_seconds','fixtures','fits','overflows','failures',
 'capacity_qualified','returncode','group_still_exists','kill_sent','child_and_cleanup_cpu_seconds',
 'cumulative_child_peak_rss_bytes','type','message')
names=['start.json','handoff.json','worker.started.json','worker/work-start.json','worker/artifacts/terminal.json',
 'terminal.json','failure.json','worker/failure.json','worker.cleanup.json','worker.resources.json']
for directory in (out,out/'worker/artifacts'):
 names.extend(str(p.relative_to(out)) for p in sorted(directory.glob('phase-*.json'))[-3:])
decoded={}
for name in names:
 p=out/name
 if not p.exists() and not p.is_symlink():continue
 try:
  value=read(p,256*1024);obj=json.loads(value);decoded[name]=obj
  selected={k:x[:600] if isinstance(x,str) else x for k,x in obj.items() if k in fields and type(x) in (str,int,float,bool,type(None))}
  v['files'][name]=dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),fields=selected)
 except Exception as exc:v['files'][name]=dict(read_error=type(exc).__name__+': '+str(exc)[:300],live_read_uncertain=True)
start=decoded.get('start.json')
if start:
 need(start['packet_sha256']==PIN,'entry packet binding differs')
 age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(start['started_utc'])).total_seconds()
 v.update(entry_age_wall_seconds=age,startup_overdue_without_work_marker=age>=120 and 'worker/work-start.json' not in decoded,
  work_deadline_passed=age>=260,cleanup_deadline_passed=age>=280,hard_deadline_passed=age>=290,
  clock_note='Cross-node UTC observation; investigate clock/telemetry uncertainty. Never cancel or retry automatically.')
token_path=out/'worker/artifacts/tokenization.json'
if token_path.exists():
 try:
  token_raw=read(token_path,32*1024**2);records=json.loads(token_raw)['rows']
  need(type(records) is list and len(records)<=120,'bounded fixture roster required')
  prompt=[len(r['prompt_ids']) for r in records if type(r.get('prompt_ids')) is list]
  commands=[len(ids) for r in records if type(r.get('commands')) is dict for ids in r['commands'].values() if type(ids) is list]
  counts=collections.Counter(str(r.get('status','unknown')) for r in records)
  v['token_summary']=dict(records=len(records),status_counts=dict(counts),prompts_with_ids=len(prompt),
   command_sequences=len(commands),max_prompt_tokens=max(prompt,default=None),
   max_command_plus_eos_tokens=max(commands,default=None),
   max_prompt_plus_generation_reserve=max(prompt)+32 if prompt else None,
   tokenization_file_sha256=hashlib.sha256(token_raw).hexdigest())
 except Exception as exc:v['token_summary']=dict(read_error=type(exc).__name__+': '+str(exc)[:300],live_read_uncertain=True)
parent=decoded.get('terminal.json',{})
v['reported_capacity_status']=parent.get('status')
v['capacity_note']='capacity_rejected may accompany scheduler exit 0; overflow and fixture/execution failures stay distinct. Not independent acceptance.'
print(json.dumps(v))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--packet-sha256', required=True)
    args = parser.parse_args()
    need(re.fullmatch('[1-9][0-9]*', args.job_id) and re.fullmatch('[0-9a-f]{64}', args.packet_sha256), 'explicit job ID and pin required')
    for name in ('packet-metadata.json', 'submission.result.json', 'inputs/packet.json'):
        need(not (LOCAL/name).is_symlink() and not (EXT/name).is_symlink()
             and (LOCAL/name).read_bytes() == (EXT/name).read_bytes(), 'launch custody copies differ')
    meta = json.loads((LOCAL/'packet-metadata.json').read_bytes())
    result = json.loads((LOCAL/'submission.result.json').read_bytes())
    need(meta['remote_root'] == ROOT and meta['output'] == ROOT+'/cpu01'
         and meta['screen_packet_sha256'] == args.packet_sha256 == sha((LOCAL/'inputs/packet.json').read_bytes())
         and result.get('returncode') == 0 and result.get('job_id') == args.job_id, 'packet/submission unresolved or differs')
    source = '\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT, JOB=args.job_id, PIN=args.packet_sha256).items())+'\n'+REMOTE
    index = 1
    while any((p/f'live{index:02d}').exists() or (p/f'live{index:02d}').is_symlink() for p in (EXT, LOCAL)): index += 1
    claimed = []
    try:
        for root in (EXT, LOCAL):
            folder = root/f'live{index:02d}'; folder.mkdir(); claimed.append(folder)
            (folder/'monitor.py').write_bytes(Path(__file__).read_bytes())
        try:
            p = subprocess.run(['bash', HELPER, 'run', shlex.join(['python3','-I','-B','-'])],
                env=dict(os.environ, CURC_SOCKET='/tmp/defab-curc.sock'), input=source,
                capture_output=True, text=True, timeout=90)
            receipt = dict(source=source, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                           local_utc=utc(), script_sha256=sha(Path(__file__).read_bytes()))
        except subprocess.TimeoutExpired as exc:
            receipt = dict(source=source, returncode=None, stdout=text(exc.stdout), stderr=text(exc.stderr),
                           local_utc=utc(), timed_out=True, script_sha256=sha(Path(__file__).read_bytes()))
        value = (json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        for folder in claimed:
            with (folder/'snapshot.json').open('xb') as h: h.write(value); h.flush(); os.fsync(h.fileno())
        need(receipt['returncode'] == 0, 'snapshot failed; evidence retained, no automatic job action')
        print(receipt['stdout'])
    except BaseException as exc:
        for folder in claimed:
            with (folder/'failure.json').open('x') as h: json.dump(dict(type=type(exc).__name__,message=str(exc)),h)
        raise

if __name__ == '__main__': main()
