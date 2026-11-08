"""Explicitly amended Option A only: original case + eleven original seeds.

One process-group watchdog owns the entire stage, including all case workers.
Default invocation validates; --run requires the separately frozen amendment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import runner_delivery_confirmation as guard


def admission(seconds,n=11,ceiling=7500):
    return 1.2*n*seconds+300<=ceiling


def validate(out):
    out=Path(out);f=guard.read(out/'option_a.json')
    if not f['approved'] or f['remaining_seeds']!=f['original_seeds'][1:] or len(f['original_seeds'])!=12:
        raise ValueError('explicit Option A/unchanged seed order required')
    if f['prior_charged_reserved_seconds']+f['stage_seconds']>f['aggregate_seconds']:
        raise ValueError('aggregate envelope exceeded')
    if (f['stage_seconds'],f['aggregate_seconds'],f['threads'],f['rss_bytes'])!=(7500,11100,6,8589934592):
        raise ValueError('amended limits changed')
    if guard.sha(__file__)!=f['continuation_sha256'] or guard.sha(out/'adapter.py')!=f['adapter_sha256'] or guard.sha(guard.__file__)!=f['adapter_sha256']:
        raise ValueError('adapter identity changed')
    if guard.sha(out/'approval.json')!=f['approval_sha256']:
        raise ValueError('approval receipt changed')
    original=Path(f['original_output'])
    if guard.inventory(original/'cases')!=f['original_case_hashes']:
        raise ValueError('original immutable history changed')
    if (original/'scores.json').exists() or (original/'sealed.json').exists():
        raise ValueError('original case no longer outcome-blind')
    reg,_=guard.validate_protocol(out/'protocol',out/'source',check_git=False)
    if reg['seeds']!=f['original_seeds'] or guard.sha(out/'protocol/registration.json')!=f['registration_sha256']:
        raise ValueError('original registration changed')
    if guard.inventory(out/'source')!=f['source_inventory']:
        raise ValueError('source snapshot changed')
    return f,reg


def worker(out):
    out=Path(out);f,reg=validate(out)
    if {p.name for p in (out/'cases').iterdir()}!={str(f['original_seeds'][0])}:
        raise ValueError('duplicate/previously started continuation')
    retained=guard.inventory(out/'cases')
    if retained!=f['original_case_hashes']:raise ValueError('retained copy differs')
    guard.approval_check(out/'approval.json',out/'protocol')
    total=4803;reports=[]
    for index,seed in enumerate(f['remaining_seeds']):
        guard.write(out/'progress.json',{'stage':'case','seed':seed,'prior_calls':total,'at':guard.utc()})
        start=time.monotonic()
        with (out/f'case-{seed}.log').open('w') as log:
            # Inherit the stage worker PGID; outer supervisor owns all descendants.
            child=subprocess.run([sys.executable,str(out/'adapter.py'),'--case-worker',str(out),str(seed),str(62436-total)],stdout=log,stderr=subprocess.STDOUT)
        journal=out/'cases'/str(seed)/'calls.jsonl'
        total+=guard.journal_count(journal) if journal.exists() else 0
        elapsed=time.monotonic()-start
        reports.append({'seed':seed,'seconds':elapsed,'exit_code':child.returncode,'aggregate_calls':total});guard.write(out/'case_reports.json',reports)
        if child.returncode!=0:raise RuntimeError('case failure; no retry')
        if total>62436:raise ValueError('aggregate call ceiling')
        if index==0:
            passed=admission(elapsed)
            guard.write(out/'timing_admission.json',{'first_new_case_seconds':elapsed,'projected_seconds':1.2*11*elapsed+300,'stage_cap_seconds':7500,'passed':passed,'at':guard.utc()})
            if not passed:
                guard.write(out/'stop.json',{'status':'timing_gate_failed','calls':total,'at':guard.utc()});return
    if guard.inventory(Path(f['original_output'])/'cases')!=f['original_case_hashes']:
        raise ValueError('historical case changed')
    first=str(f['original_seeds'][0])
    if {first+'/'+k:v for k,v in guard.inventory(out/'cases'/first).items()}!=f['original_case_hashes']:
        raise ValueError('retained working copy changed')
    manifest=guard.seal_cases(out,reg)
    guard.write(out/'sealed.json',manifest)
    guard.write(out/'progress.json',{'stage':'all12_sealed_evaluation','calls':total,'at':guard.utc()})
    subprocess.run([sys.executable,str(out/'adapter.py'),'--evaluate-worker',str(out)],check=True)
    guard.write(out/'completed.json',{'status':'complete','calls':total,'at':guard.utc()})


def main():
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--worker',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:worker(out);return
    f,_=validate(out)
    if not a.run:print('validated; not launched');return
    if (out/'execution.json').exists() or (out/'progress.json').exists():raise ValueError('no duplicate/resume')
    (out/'launch-claim').mkdir(exist_ok=False) # Atomic duplicate-start refusal.
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','USE_LLM':'false','ACE_OBS_LOG':'on','OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','PYTHONDONTWRITEBYTECODE':'1'}
    start=time.monotonic()
    report=guard.supervise([sys.executable,str(Path(__file__).resolve()),str(out),'--worker'],start+7500,8589934592,out/'stage.log',sampler=lambda _:guard.process_rss(os.getpid()),interval=.5,env=env)
    report['elapsed_seconds']=time.monotonic()-start
    report['prior_charged_reserved_seconds']=f['prior_charged_reserved_seconds']
    report['aggregate_charged_seconds']=f['prior_charged_reserved_seconds']+report['elapsed_seconds']
    if (out/'stop.json').exists():report['status']=guard.read(out/'stop.json')['status']
    if report['status']!='complete' and (out/'scores.json').exists():
        (out/'scores.json').rename(out/'INCOMPLETE-scores.json')
    guard.write(out/'execution.json',report);print(report['status'])

if __name__=='__main__':main()
