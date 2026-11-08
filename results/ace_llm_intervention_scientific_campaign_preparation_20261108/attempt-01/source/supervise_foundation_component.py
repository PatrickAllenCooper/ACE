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

LIMITS = {'pilot':1800, 'tabpfn-smoke':120, 'language-smoke':120}
METHODS = ('polynomial', 'extra_trees', 'tabpfn_v2', 'grammar', 'language')


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


def validate_stage(freeze, mode):
    if mode not in freeze.get('allowed_modes', []):
        raise ValueError('mode not authorized by freeze')
    if mode != 'pilot':
        if freeze.get('stage') != 'compatibility':raise ValueError('compatibility stage required')
        return
    if freeze.get('stage') != 'pilot':raise ValueError('scientific stage required')
    revision=freeze.get('source_revision','')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('committed source revision required')
    for file,key in [('scripts/research/foundation_component_pilot.py','source_sha256'),
                     ('scripts/research/supervise_foundation_component.py','supervisor_sha256'),
                     ('docs/development/guidance/ace_foundation_component_pilot_2026-10-09.md','protocol_sha256'),
                     ('scripts/research/summarize_foundation_component.py','reporter_sha256')]:
        data=subprocess.check_output(['git','--no-replace-objects','-C',freeze['repository'],'show',revision+':'+file])
        if hashlib.sha256(data).hexdigest()!=freeze[key]:raise ValueError('committed source mismatch')
    if set(freeze.get('smokes',{})) != {'tabpfn-smoke','language-smoke'}:
        raise ValueError('both smoke receipts required')
    for name,binding in freeze['smokes'].items():
        t=captured(binding['terminal_path'],binding['terminal_sha256'])
        f=captured(binding['freeze_path'],binding['freeze_sha256'])
        if t['status']!='complete' or t['mode']!=name or t['freeze_sha256']!=binding['freeze_sha256']:
            raise ValueError('unsuccessful or wrong compatibility receipt')
        for key in ('source_sha256','supervisor_sha256','protocol_sha256','checkpoint_sha256','language_files','dependencies'):
            if f[key]!=freeze[key]:raise ValueError('smoke provenance mismatch: '+key)


def supervise(args):
    start, own_cpu = time.monotonic(), time.process_time()
    args.output.mkdir(exist_ok=False)
    plan = ([{'seed': seed, 'method': method} for seed in range(91000,91006) for method in METHODS]
            if args.mode == 'pilot' else [{'seed': 'smoke', 'method': args.mode}])
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
        validate_stage(freeze,args.mode)
        if sha(__file__) != freeze['supervisor_sha256'] or sha(args.worker) != freeze['source_sha256']:
            raise ValueError('launcher/worker source mismatch')
        if freeze['schema'] != 'ace-component-freeze-v1':
            raise ValueError('freeze schema')
        # No newer run may silently inherit another attempt's budget.
        write(args.output/'launch.json', {'freeze_sha256': args.freeze_sha256, 'limit_wall_s':limit,
              'limit_process_cpu_s':limit,'cpu_threads':1,'gpu':False,'python':str(args.python),
              'scope':'one owned process, Python process spawning rejected by worker; startup included'})
        env = dict(os.environ, PYTHONHASHSEED='0', ACE_COMPONENT_SUPERVISED='1')
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[k]='1'
        def child_limit():
            resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(limit)),max(1,int(limit))+1))
        command=[str(args.python),str(args.worker),'--output',str(args.output),'--freeze',str(args.freeze),
                 '--checkpoint',str(args.checkpoint),'--language-model',str(args.language_model),
                 '--mode',args.mode,'--freeze-sha256',args.freeze_sha256]
        if cancellation:raise InterruptedError('controller signal '+str(cancellation[0]))
        with (args.output/'stdout.log').open('xb') as out, (args.output/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True,preexec_fn=child_limit)
            write(args.output/'child.json',{'pid':child.pid,'at_unix':time.time(),'command':command})
            while True:
                pid,status,usage_now=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    usage=usage_now; code=os.waitstatus_to_exitcode(status); child.returncode=code
                    reason='cancelled' if cancellation else ('exited' if code == 0 else 'child_failed'); break
                if cancellation or time.monotonic()-start>=limit:
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
            d=args.output/str(item['seed']); result=d/(item['method']+'.json')
            started=d/(item['method']+'.started.json')
            row=dict(item,status='unattempted')
            if result.exists():
                try: row=json.loads(result.read_text())
                except Exception: row.update(status='invalid_record',error=traceback.format_exc())
            elif started.exists(): row['status']='interrupted'
            cells.append(row)
        if cancellation:
            reason='cancelled';error=error or ('controller signal '+str(cancellation[0]))
        complete_path=args.output/('complete.json' if args.mode=='pilot' else 'smoke.json')
        passed=False
        if code==0 and reason=='exited' and error is None:
            try:
                completion=json.loads(complete_path.read_text())
                if args.mode=='pilot':
                    if completion['planned_cells']!=len(plan) or completion['cells']!=cells:
                        raise ValueError('completion and terminal ledger disagree')
                    if any(r['status'] not in ('complete','failed') for r in cells):
                        raise ValueError('invalid final cell disposition')
                elif completion.get('status')!='complete' or cells[0]!=completion:
                    raise ValueError('smoke completion and cell disagree')
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
    for name in ('output','python','worker','freeze','checkpoint','language-model','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--mode',choices=('pilot','tabpfn-smoke','language-smoke'),required=True)
    a=p.parse_args()
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return supervise(a)

if __name__=='__main__':
    sys.exit(main())
