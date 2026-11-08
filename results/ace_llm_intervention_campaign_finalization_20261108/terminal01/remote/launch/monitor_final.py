#!/usr/bin/env python3
"""Future read-only CPU final01 snapshot; never submits, cancels or finalizes.

Call early and periodically after launch. Startup is bounded at 120s, work at
260s, cleanup at 280s and publication at 290s from the finalizer's entry. Missing
markers or overdue startup require parent diagnosis, not automatic cancellation
or retry. A snapshot is not terminal custody or independent acceptance.
"""
from pathlib import Path
import datetime, hashlib, json, subprocess

LOCAL=Path('/Users/pat/code/ACE/results/ace_llm_intervention_campaign_finalization_20261108/launch01')
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-20261108-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
COMMON='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

def main():
 meta=json.loads((LOCAL/'packet-metadata.json').read_bytes())
 if meta['remote_root']!=ROOT or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=meta['root_files']['monitor_final.py']:
  raise ValueError('captured monitor/root differs')
 result=json.loads((LOCAL/'submission.result.json').read_bytes());job=result['job_id']
 if result.get('returncode')!=0 or not isinstance(job,str) or not job.isdigit():raise ValueError('submission unresolved; reconcile first')
 source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,COMMON=COMMON,JOB=job,PIN=meta['finalizer_packet_sha256']).items())+r'''
from pathlib import Path
import json,subprocess,datetime
root=Path(ROOT);out=root/'finalization';campaign=Path(COMMON)/'cpu03/worker/artifacts/campaign'
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':JOB,'files':{},'finalization_qualified':False}
for label,args in {'queue':['squeue','-j',JOB,'-h','-o','%i|%j|%T|%R|%a'],
 'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,AllocTRES%180,MaxRSS,TotalCPU'],
 'job':['scontrol','show','job','-o',JOB]}.items():
 p=subprocess.run(args,capture_output=True,text=True,timeout=15)
 v[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
# Explicit phase evidence; never decode the private campaign bundle or scores.
for name in ('start.json','stage.json','handoff.json','worker.started.json','worker/work-start.json',
 'worker/terminal.json','worker/failure.json','worker.cleanup.json','worker.resources.json','failure.json','attempt.json'):
 p=out/name
 if p.is_file() and not p.is_symlink() and p.stat().st_size<=128*1024:v['files'][name]=p.read_text()
start=v['files'].get('start.json');work=v['files'].get('worker/work-start.json')
if start:
 s=json.loads(start)
 if s['packet_sha256']!=PIN:raise ValueError('worker packet differs')
 age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(s['started_utc'])).total_seconds()
 v['entry_age_wall_seconds']=age
 v['startup_overdue_without_work_marker']=age>=120 and work is None
 v['age_note']='Cross-node UTC observation; investigate clock/telemetry uncertainty before diagnosis.'
v['campaign_publication_present']=(campaign/'finalization.json').is_file()
v['publication_note']='Original campaign/finalization.json is authoritative if published, including after an outer failure; never retry blindly.'
print(json.dumps(v))
'''
 p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=60)
 v=dict(source=source,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 i=1
 while any((d/f'live{i:02d}.json').exists() for d in (LOCAL,EXT)):i+=1
 for d in (EXT,LOCAL):
  with (d/f'live{i:02d}.json').open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 print(p.stdout if p.returncode==0 else p.stderr)
 if p.returncode:raise RuntimeError('snapshot failed; evidence retained, no automatic job action')

if __name__=='__main__':main()
