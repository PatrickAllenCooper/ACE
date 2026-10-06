"""Explicit final-history amendment. Validate by default; one bounded stage."""
import argparse, os, subprocess, sys, time
from pathlib import Path
import runner_delivery_confirmation as g

def validate(out):
    out=Path(out); a=g.read(out/'amendment.json'); old=Path(a['prior_output'])
    if (a['stage_seconds'],a['aggregate_seconds'],a['threads'],a['rss_bytes'],a['call_cap'],a['seed'])!=(2700,13800,6,8589934592,62436,884825602):raise ValueError('limits')
    if not a['approved'] or a['preparation_reserved_seconds']!=300 or a['prior_seconds']+2700>13800:raise ValueError('approval/accounting')
    if g.sha(__file__)!=a['script_sha256'] or g.sha(g.__file__)!=a['adapter_sha256'] or g.sha(out/'adapter.py')!=a['adapter_sha256']:raise ValueError('code identity')
    if g.sha(old/'execution.json')!=a['prior_execution_sha256'] or g.inventory(old/'cases')!=a['prior_cases_inventory']:raise ValueError('prior custody')
    if g.read(old/'execution.json')['status']!='time_limit':raise ValueError('prior not terminal')
    reg,_=g.validate_protocol(out/'protocol',out/'source',check_git=False)
    g.approval_check(out/'approval.json',out/'protocol')
    if g.sha(out/'approval.json')!=a['approval_sha256'] or g.inventory(out/'source')!=a['source_inventory']:raise ValueError('frozen inputs')
    if g.inventory(out/'cases')!=a['retained_inventory'] and not(out/'launch-claim').exists():raise ValueError('retained cases')
    for seed in reg['seeds'][:-1]:
        if g.inventory(out/'cases'/str(seed))!=a['retained_case_hashes'][str(seed)]:raise ValueError('retained history changed')
    if reg['seeds'][-1]!=a['seed'] or len(reg['seeds'])!=12:raise ValueError('seed order')
    if g.journal_count(old/'cases'/str(a['seed'])/'calls.jsonl')!=3326 or a['prior_calls']!=56159:raise ValueError('discarded attempts')
    ps=subprocess.check_output(['ps','-axo','pid,ppid,pgid'],text=True)
    if any(any(v in ('61498','61499') for v in line.split()) for line in ps.splitlines()[1:]):raise ValueError('prior workers alive')
    return a,reg

def worker(out):
    out=Path(out);a,reg=validate(out)
    if (out/'cases'/str(a['seed'])).exists():raise ValueError('no repeat')
    g.write(out/'progress.json',{'stage':'final_case','seed':a['seed'],'at':g.utc(),'prior_calls':a['prior_calls']})
    with (out/'case.log').open('w') as log:
        subprocess.run([sys.executable,str(out/'adapter.py'),'--case-worker',str(out),str(a['seed']),str(min(5203,a['call_cap']-a['prior_calls']))],stdout=log,stderr=subprocess.STDOUT,check=True)
    calls=a['prior_calls']+g.journal_count(out/'cases'/str(a['seed'])/'calls.jsonl')
    if calls>a['call_cap']:raise ValueError('call cap')
    validate(out)
    g.write(out/'sealed.json',g.seal_cases(out,reg))
    g.write(out/'progress.json',{'stage':'all12_sealed_evaluation','aggregate_calls_including_discarded':calls,'at':g.utc()})
    subprocess.run([sys.executable,str(out/'adapter.py'),'--evaluate-worker',str(out)],check=True)
    g.write(out/'completed.json',{'at':g.utc(),'aggregate_calls_including_discarded':calls})

def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--run',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args();out=Path(args.out)
    if args.worker:worker(out);return
    a,_=validate(out)
    if not args.run:print('validated; not launched');return
    if any((out/n).exists() for n in ['progress.json','execution.json','sealed.json','scores.json']):raise ValueError('no duplicate/resume')
    (out/'launch-claim').mkdir(exist_ok=False)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','USE_LLM':'false','ACE_OBS_LOG':'on','OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','PYTHONDONTWRITEBYTECODE':'1'}
    g.write(out/'start.json',{'at':g.utc(),'supervisor_pid':os.getpid(),'stage_seconds':2700})
    start=time.monotonic()
    report=g.supervise([sys.executable,str(Path(__file__).resolve()),str(out),'--worker'],start+2400,8589934592,out/'stage.log',sampler=lambda _:g.process_rss(os.getpid()),interval=.5,env=env)
    report.update(elapsed_seconds=time.monotonic()-start+300,preparation_reserved_seconds=300,prior_charged_reserved_seconds=a['prior_seconds'])
    report['aggregate_charged_seconds']=a['prior_seconds']+report['elapsed_seconds']
    journal=out/'cases'/str(a['seed'])/'calls.jsonl';report['aggregate_calls']=a['prior_calls']+(g.journal_count(journal) if journal.exists() else 0)
    if report['status']!='complete' and (out/'scores.json').exists():(out/'scores.json').rename(out/'INCOMPLETE-scores.json')
    g.write(out/'execution.json',report);print(report['status'])
if __name__=='__main__':main()
