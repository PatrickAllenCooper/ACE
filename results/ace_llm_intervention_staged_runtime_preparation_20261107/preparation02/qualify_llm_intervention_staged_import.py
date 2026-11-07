"""Bounded CPU packing, node-local staging and original-sequence imports.

No packages are installed. The original probe remains byte-identical and is
called once in the copied interpreter. No weights or environment responses.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import types


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def pin_module(path, expected, name):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('independent source pin differs: '+str(path))
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(raw,str(path),'exec'),m.__dict__)
    return m


def allocation(out):
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise ValueError('CPU Slurm job required')
    p=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
    write(out/'allocation.json',dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode))
    f=dict(re.findall(r'(\w+)=(\S+)',p.stdout));t=f.get('AllocTRES','').split(',')
    if (p.returncode or f.get('JobId')!=job or f.get('Account')!='ucb736_asc2'
        or f.get('Partition')!='acpu' or f.get('QOS')!='cpu-normal' or f.get('NumCPUs')!='1'
        or f.get('NumTasks')!='1' or f.get('NumNodes')!='1' or f.get('TimeLimit')!='00:30:00'
        or f.get('Requeue')!='0' or 'cpu=1' not in t or 'mem=3G' not in t
        or any('gres/' in x for x in t)):
        raise ValueError('actual allocation differs from fixed CPU-only envelope')
    return job


def copied_worker(args, probe):
    prefix=Path(args.staged_prefix).resolve(strict=True)
    if Path(sys.prefix).resolve()!=prefix or not Path(sys.executable).resolve().is_relative_to(prefix):
        raise ValueError('copied interpreter did not adopt intended prefix')
    # Add origin capture just before the original successful receipt. Import order
    # and original source/model/runtime authentication remain unchanged.
    original_write=probe.write
    def observed_write(path,value):
        if Path(path).name=='terminal.json':
            origins={}
            mixed=[]
            for name,m in sorted(sys.modules.items()):
                f=getattr(m,'__file__',None)
                if not f or f.startswith('<'):continue
                p=Path(f).resolve()
                origins[name]=str(p)
                if p.is_relative_to(Path(args.runtime_source).resolve()):mixed.append(str(p))
                if ('site-packages' in p.parts or 'lib-dynload' in p.parts) and not p.is_relative_to(prefix):
                    mixed.append(str(p))
            maps=Path('/proc/self/maps').read_text()
            native_mixed=[line for line in maps.splitlines() if args.runtime_source+'/' in line]
            with (Path(args.output)/'process_maps.txt').open('x') as f:f.write(maps)
            original_write(Path(args.output)/'runtime_origins.json',dict(
                prefix=str(prefix),executable=sys.executable,sys_prefix=sys.prefix,
                sys_path=sys.path,module_paths=origins,mixed_original_paths=sorted(set(mixed)),
                native_mixed_original_paths=native_mixed,scope='loaded module origins and process maps, not universal dynamic dependency closure'))
            if mixed or native_mixed:
                raise ValueError('copied runtime still uses original installed paths')
        original_write(path,value)
    probe.write=observed_write
    args.mode='matched_import_order'
    return probe.worker(args)



def run_bounded_child(command, env, out, overall_deadline, kill_owned):
    start=time.monotonic()
    if overall_deadline-start < 200:
        raise TimeoutError('insufficient preparation time left for fixed startup and cleanup')
    deadline=start+185
    proc=None
    cleanup=None
    # The timer covers process creation and receipt fsync, not only wait().
    signal.setitimer(signal.ITIMER_REAL, min(deadline,overall_deadline)-start)
    try:
        with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            write(out/'worker.started.json',dict(command=command,pid=proc.pid,pgid=proc.pid,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outer_deadline_monotonic=deadline))
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('startup receipt consumed original child deadline')
            rc=proc.wait(timeout=remaining)
        elapsed=time.monotonic()-start
    finally:
        # Only bounded group cleanup runs while the phase timer is disabled.
        # No filesystem writes precede termination of the owned group.
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if proc is not None:cleanup=kill_owned(proc)
        finally:
            signal.setitimer(signal.ITIMER_REAL,max(.001,overall_deadline-time.monotonic()))
        if proc is not None:write(out/'worker.cleanup.json',cleanup)
    return rc,elapsed,cleanup


def main():
    p=argparse.ArgumentParser()
    for n in ('helper','helper-sha256','probe','probe-sha256','service','service-sha256','freeze','freeze-sha256','output'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--runtime-source',default='/projects/paco0228/miniconda3/envs/ace')
    p.add_argument('--staged-prefix')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    probe=pin_module(args.probe,args.probe_sha256,'ace_original_import_probe')
    if args.worker:return copied_worker(args,probe)
    out=Path(args.output);out.mkdir(exist_ok=False)
    start=time.monotonic();cpu=time.process_time();serial=0
    def event(phase,**extra):
        nonlocal serial
        v=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,**extra)
        write(out/f'phase-{serial:05d}.json',v);serial+=1
    def interrupt(signum,frame):
        raise InterruptedError('CPU preparation interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGALRM,interrupt);signal.alarm(1740)
    try:
        job=allocation(out)
        runtime=Path(args.runtime_source).resolve(strict=True)
        if runtime!=Path('/projects/paco0228/miniconda3/envs/ace'):
            raise ValueError('unexpected existing runtime source')
        scratch_env=os.environ.get('SLURM_SCRATCH','')
        if not scratch_env or not Path(scratch_env).is_absolute():raise ValueError('node-local SLURM_SCRATCH required')
        scratch=Path(scratch_env).resolve(strict=True)
        if not scratch.is_dir() or scratch.stat().st_uid!=os.getuid():raise ValueError('node-local scratch owner differs')
        local=scratch/('ace-runtime-'+job);local.mkdir(mode=0o700,exist_ok=False)
        helper=pin_module(args.helper,args.helper_sha256,'ace_runtime_archive')
        event('pack_start',source=str(runtime),node_scratch=str(scratch),free_bytes=os.statvfs(scratch).f_bavail*os.statvfs(scratch).f_frsize)
        manifest=helper.pack(runtime,out/'packed',progress=lambda v:event('packing_progress',detail=v))
        manifest_path=out/'packed/manifest.json';mh=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        event('pack_complete',manifest_sha256=mh,archive_bytes=manifest['archive_bytes'],file_bytes=manifest['file_bytes'],objects=manifest['objects'])
        if os.statvfs(local).f_bavail*os.statvfs(local).f_frsize < manifest['file_bytes']+1024**3:
            raise ValueError('insufficient node-local capacity')
        staged=local/'runtime'
        stage=helper.stage(out/'packed/runtime.tar',manifest_path,mh,staged,progress=lambda v:event('staging_progress',detail=v))
        write(out/'stage.json',stage)
        event('stage_complete',staged_prefix=str(staged))
        # CPU preparation is charged separately. Only this fresh process models
        # service import startup under the explicitly staged runtime layout.
        interpreter=staged/'bin/python'
        command=[str(interpreter),'-B','-X','importtime',str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--staged-prefix',str(staged)]
        childargs=command.index('--output')+1;command[childargs]=str(out/'original_sequence')
        env=dict(os.environ)
        for k in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD'):env.pop(k,None)
        env['LD_LIBRARY_PATH']=str(staged/'lib')
        event('original_sequence_start',command=command,library_path=env['LD_LIBRARY_PATH'])
        rc,elapsed,cleanup=run_bounded_child(command,env,out,start+1740,probe.kill_owned)
        event('original_sequence_returned',returncode=rc,worker_elapsed_seconds=elapsed)
        receipt=out/'original_sequence/terminal.json'
        if (rc or elapsed>180 or cleanup['group_still_exists'] is not False or cleanup['errors']
            or not receipt.is_file() or not (out/'original_sequence/runtime_origins.json').is_file()):
            raise ValueError('original sequence did not qualify within 180 seconds with clean group')
        v=json.loads(receipt.read_bytes())
        if (v.get('status')!='complete' or v.get('model_weights_loaded') is not False
            or v.get('training_updates')!=0 or v.get('environment_responses')!=0):
            raise ValueError('unexpected import receipt')
        event('cpu_startup_qualified')
        write(out/'terminal.json',dict(status='cpu_staged_original_sequence_qualified',manifest_sha256=mh,
            archive_sha256=manifest['archive_sha256'],worker_elapsed_seconds=elapsed,
            elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
            gpu_requested=False,model_weights_loaded=False,scientific_responses=0,
            model_service_qualified=False,automatic_gpu_retry=False,
            qualification_scope='one original-sequence CPU import process after measured archive/stage preparation'))
    except BaseException as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
            retry_allowed=False,model_service_qualified=False))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':main()
