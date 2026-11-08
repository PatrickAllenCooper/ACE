#!/usr/bin/env python3
"""Exclusive local CPU attempt, whole-process deadline and terminal cell ledger."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time
import traceback

LIMITS = {'pilot':900, 'fixture':120}
STOP_UNIX = 1791583200  # 2026-10-09T22:00:00Z
VARIANTS=('null','coefficient_M','missing_M','missing_Y')
METHODS = ('grammar32','grammar24','pfn24','terminal24','mechanism24','prechange24')
SOURCES = ('scripts/research/foundation_mismatch_pilot.py','scripts/research/supervise_foundation_mismatch.py','scripts/research/foundation_mixture_selection.py','scripts/research/foundation_component_pilot.py','scripts/research/summarize_foundation_mismatch.py','docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md')


def write(path, obj):
    raw = (json.dumps(obj, indent=2, allow_nan=False)+'\n').encode()
    tmp = Path(str(path)+'.pending')
    with tmp.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(tmp, path)  # exclusive publication, never overwrite a previous receipt
    tmp.unlink()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def captured(path, pin):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:
        raise ValueError('captured bytes pin mismatch: '+str(path))
    return json.loads(raw)


def deadline_check(deadline,cancellation):
    if cancellation:raise InterruptedError('controller signal '+str(cancellation[0]))
    remaining=min(deadline-time.monotonic(),STOP_UNIX-time.time())
    if remaining<=0:raise TimeoutError('preflight or launch deadline exhausted')
    return remaining


def bounded_git(command,deadline,cancellation):
    deadline_check(deadline,cancellation)
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:
        while True:
            remaining=deadline_check(deadline,cancellation)
            try:
                out,err=child.communicate(timeout=min(.1,remaining))
                if child.returncode:raise subprocess.CalledProcessError(child.returncode,command,out,err)
                return out
            except subprocess.TimeoutExpired:pass
    finally:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.communicate()


def validate_stage(freeze, mode,deadline=None,cancellation=None):
    deadline=time.monotonic()+LIMITS[mode] if deadline is None else deadline
    cancellation=[] if cancellation is None else cancellation
    deadline_check(deadline,cancellation)
    if freeze.get('schema')!='ace-mismatch-freeze-v1' or freeze.get('mode')!=mode:raise ValueError('stage/schema mismatch')
    if set(freeze['sources'])!=set(SOURCES):raise ValueError('source closure mismatch')
    if freeze.get('limit_seconds')!=LIMITS[mode] or freeze.get('stop_unix')!=STOP_UNIX:raise ValueError('resource freeze mismatch')
    if time.time()+LIMITS[mode]>STOP_UNIX:raise ValueError('stage cannot finish within approved cycle')
    revision=freeze.get('source_revision','')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('committed source required')
    for rel,pin in freeze['sources'].items():
        data=bounded_git(['git','--no-replace-objects','-C',freeze['repository'],'show',revision+':'+rel],deadline,cancellation)
        deadline_check(deadline,cancellation)
        if hashlib.sha256(data).hexdigest()!=pin or sha(Path(freeze['repository'])/rel)!=pin:raise ValueError('committed/current source mismatch '+rel)
    if mode=='pilot':
        bind=freeze['fixture']
        t=captured(bind['terminal_path'],bind['terminal_sha256']);f=captured(bind['freeze_path'],bind['freeze_sha256'])
        if t['status']!='complete' or t['mode']!='fixture' or t['freeze_sha256']!=bind['freeze_sha256'] or any(r['status']!='complete' for r in t['cells']):raise ValueError('qualified artificial fixture required')
        for key in ('sources','dependencies','python','python_version','checkpoint','checkpoint_sha256'):
            if f[key]!=freeze[key]:raise ValueError('fixture provenance mismatch '+key)
        if t['child_cpu_s'] is None or t['child_cpu_s']>=120 or t['elapsed_s']>=120:raise ValueError('fixture resource qualification failed')
        if t['peak_child_rss_bytes'] is None or t['peak_child_rss_bytes']>6*1024**3:raise ValueError('fixture memory sizing failed')


def supervise(args):
    start, own_cpu = time.monotonic(), time.process_time()
    args.output.mkdir(exist_ok=False)
    plan = [{'seed': seed, 'variant':v,'method': method} for seed in (range(92000,92006) if args.mode=='pilot' else (123456,)) for v in VARIANTS for method in METHODS]
    write(args.output/'plan.json', {'mode': args.mode, 'cells': plan, 'at_unix': time.time()})
    old_handlers={}; cancellation=[]
    def cancelled(signum, frame):
        cancellation.append(signum)  # defer across process ownership and wait4 publication
    for signum in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
        old_handlers[signum]=signal.signal(signum,cancelled)
    child = None; usage = None; error = None; reason = 'preflight_failed'; code = None
    limit = LIMITS[args.mode]
    try:
        freeze = captured(args.freeze,args.freeze_sha256)
        validate_stage(freeze,args.mode,start+limit,cancellation)
        if sha(__file__) != freeze['sources']['scripts/research/supervise_foundation_mismatch.py'] or sha(args.worker) != freeze['sources']['scripts/research/foundation_mismatch_pilot.py']:
            raise ValueError('launcher/worker source mismatch')
        if freeze['schema'] != 'ace-mismatch-freeze-v1':
            raise ValueError('freeze schema')
        if str(args.python.absolute())!=freeze['python']:raise ValueError('interpreter path mismatch')
        if str(args.output.resolve())!=freeze['output'] or str(args.lock.resolve())!=freeze['lock']:raise ValueError('frozen output/controller lock mismatch')
        # No newer run may silently inherit another attempt's budget.
        write(args.output/'launch.json', {'freeze_sha256': args.freeze_sha256, 'limit_wall_s':limit,
              'limit_process_cpu_s':limit,'cpu_threads':1,'gpu':False,'python':str(args.python),
              'scope':'one owned process, Python process spawning rejected by worker; startup included'})
        env = dict(os.environ, PYTHONHASHSEED='0', ACE_MISMATCH_SUPERVISED='1')
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[k]='1'
        def child_limit():
            resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(limit)),max(1,int(limit))+1))
        command=[str(args.python),str(args.worker),'--output',str(args.output),'--freeze',str(args.freeze),
                 '--mode',args.mode,'--freeze-sha256',args.freeze_sha256]
        deadline_check(start+limit,cancellation)
        with (args.output/'stdout.log').open('xb') as out, (args.output/'stderr.log').open('xb') as err:
            deadline_check(start+limit,cancellation)
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True,preexec_fn=child_limit)
            write(args.output/'child.json',{'pid':child.pid,'at_unix':time.time(),'command':command})
            while True:
                pid,status,usage_now=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    usage=usage_now; code=os.waitstatus_to_exitcode(status); child.returncode=code
                    reason='cancelled' if cancellation else ('exited' if code == 0 else 'child_failed'); break
                if cancellation or time.monotonic()-start>=limit or time.time()>=STOP_UNIX:
                    reason='cancelled' if cancellation else 'wall_timeout'; os.killpg(child.pid,signal.SIGKILL)
                    _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code;break
                time.sleep(.1)
    except BaseException:
        error=traceback.format_exc()
    finally:
        for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
        if child is not None:
            # Also remove unexpected surviving members of this attempt's process group.
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            if child.returncode is None:
                _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
        cells=[]
        for item in plan:
            d=args.output/str(item['seed'])/item['variant']; result=d/(item['method']+'.json')
            started=d/(item['method']+'.started.json')
            row=dict(item,status='unattempted')
            if result.exists():
                try: row=json.loads(result.read_text())
                except Exception: row.update(status='invalid_record',error=traceback.format_exc())
            elif started.exists(): row['status']='interrupted'
            cells.append(row)
        if cancellation:
            reason='cancelled';error=error or ('controller signal '+str(cancellation[0]))
        complete_path=args.output/'complete.json'
        passed=False
        if code==0 and reason=='exited' and error is None:
            try:
                completion=json.loads(complete_path.read_text())
                if completion['mode']!=args.mode or completion['planned_cells']!=len(plan) or completion['cells']!=cells:
                    raise ValueError('completion and terminal ledger disagree')
                for expected,actual in zip(plan,cells):
                    if any(actual.get(k)!=v for k,v in expected.items()) or actual['status'] not in ('complete','failed'):
                        raise ValueError('invalid final cell identity/disposition')
                n=6 if args.mode=='pilot' else 1
                if completion['training_responses_total']!=n*160 or completion['private_responses_total']!=n*3072:raise ValueError('response counts')
                if args.mode=='fixture' and any(r['status']!='complete' for r in cells):raise ValueError('artificial fixture cells failed')
                passed=True
            except Exception:
                error=traceback.format_exc();reason='invalid_completion'

        write(args.output/'terminal.json',{'status':'complete' if passed else 'failed','reason':reason,
              'mode':args.mode,'freeze_sha256':args.freeze_sha256,
              'exit_code':code,'error':error,'cells':cells,'planned_cells':len(plan),
              'elapsed_s':time.monotonic()-start,'supervisor_process_cpu_s':time.process_time()-own_cpu,
              'child_cpu_s':None if usage is None else usage.ru_utime+usage.ru_stime,
              'peak_child_rss_bytes':None if usage is None else usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
              'platform':sys.platform,'accounting_scope':'wait4 child lifetime including imports and waited descendants; supervisor separate',
              'gpu_seconds':0,'new_accepted_study_responses':0})
    for signum, handler in old_handlers.items():signal.signal(signum,handler)
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser()
    for name in ('output','python','worker','freeze','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--mode',choices=('pilot','fixture'),required=True)
    a=p.parse_args()
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return supervise(a)

if __name__=='__main__':
    sys.exit(main())
