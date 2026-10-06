"""Prepare/release one exclusive CPU Slurm chain after the frozen pilot gate.

Preparation produces reviewable scripts and consumes no responses or allocation.
Submission runs on CURC, records every attempt, and refuses any second launch.
Interrupted/ambiguous submissions require reconciliation, never blind retry.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from runner_delivery_confirmation import read,write,sha,utc
from delivery_prospective_batch import validate,allocation_plan,claim

SCRATCH=Path('/scratch/alpine/paco0228/ACE/results')


def clock_time(seconds):
    if type(seconds) is not int or not 0<seconds<=86400:raise ValueError('bounded CPU wall required')
    return f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'


def scripts_for(out,python,worker,resources):
    out,worker=Path(out),Path(worker)
    for path in (out,worker,Path(python)):
        if not path.is_absolute() or any(c.isspace() for c in str(path)) or '#' in str(path):
            raise ValueError('absolute whitespace-free Slurm paths required')
    if (resources['threads']!=1 or resources['rss_bytes']!=3*2**30 or
            resources['max_simultaneous_worlds']!=4 or resources['account']!='ucb736_asc1' or
            resources['partition']!='acpu' or resources['qos']!='cpu-normal'):
        raise ValueError('resource request changed')
    scripts={}
    for name,phase,size in (('qualification','qualification',None),('collect','collect',None),
                            ('fit5',None,5),('fit30',None,30),('evaluate','evaluate',None)):
        wall=resources['world_wall_seconds'][str(size)] if size else resources[
            {'collect':'collection','evaluate':'evaluation'}.get(phase,phase)+'_wall_seconds']
        array='\n#SBATCH --array=0-19%2' if size else ''
        log=out/'slurm_logs'/(name+'-%A_%a.log')
        head=f'''#!/bin/bash
#SBATCH --job-name=ace_delB_{name}
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time={clock_time(wall)}
#SBATCH --output={log}
#SBATCH --error={log}{array}
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
'''
        command=shlex.join([str(python),str(worker)])
        if size:
            tail=f'''printf -v ace_case '{size}:%02d' "$SLURM_ARRAY_TASK_ID"
{command} fit-world --out {shlex.quote(str(out))} --case "$ace_case"
'''
        else:tail=command+' supervise-phase --out '+shlex.quote(str(out))+' --phase '+phase+'\n'
        scripts[name]=head+tail
    return scripts


def prepare(out,destination):
    out=Path(out).resolve();destination=Path(destination).resolve();p=validate(out)
    if not out.is_relative_to(SCRATCH):raise ValueError('explicit ACE scratch output required')
    if any((out/name).exists() for name in ('collection_started.json','submission_started.json','evaluation_started.json')):
        raise ValueError('already started study; reconcile instead of preparing another launch')
    calculated=allocation_plan(read(out/'pilot_projection.json'))
    if calculated!=p['resources']:raise ValueError('requested CPU resource plan does not reproduce')
    destination.mkdir(parents=True,exist_ok=False);(out/'slurm_logs').mkdir(exist_ok=False)
    scripts=scripts_for(out,sys.executable,Path(__file__).with_name('delivery_prospective_batch.py').resolve(),calculated)
    hashes={}
    for name,body in scripts.items():
        path=destination/(name+'.sbatch');path.write_text(body);hashes[name]=sha(path)
    plan={'at':utc(),'registration_sha256':sha(out/'registration.json'),'source_revision':p['source_revision'],
        'out':str(out),'script_directory':str(destination),'script_hashes':hashes,'python':sys.executable,
        'resources':calculated,'dependencies':p['dependencies'],
        'dependency_chain':'qualification -> collect -> fit5[20] + fit30[20] -> evaluate',
        'no_new_responses_or_allocations_from_preparation':True}
    write(destination/'plan.json',plan)
    return plan


def submit(destination):
    destination=Path(destination).resolve();plan=read(destination/'plan.json');out=Path(plan['out']);p=validate(out)
    if not out.is_relative_to(SCRATCH) or sys.executable!=plan['python']:
        raise ValueError('target runtime/output changed')
    if (sha(out/'registration.json')!=plan['registration_sha256'] or p['resources']!=plan['resources'] or
            allocation_plan(read(out/'pilot_projection.json'))!=p['resources']):
        raise ValueError('frozen scientific/resource protocol changed')
    for name,h in plan['script_hashes'].items():
        if sha(destination/(name+'.sbatch'))!=h:raise ValueError('Slurm script changed')
    # Slurm scripts themselves must reproduce from validated resources/source.
    bodies=scripts_for(out,sys.executable,Path(__file__).with_name('delivery_prospective_batch.py').resolve(),p['resources'])
    if set(bodies)!=set(plan['script_hashes']) or any((destination/(n+'.sbatch')).read_text()!=b for n,b in bodies.items()):
        raise ValueError('unexpected job command or allocation')
    if any((out/name).exists() for name in ('collection_started.json','qualification_supervisor_started.json')):
        raise ValueError('started study must be reconciled, never duplicated')
    # Path readiness is checked immediately before the single submission claim.
    probe=out/'.submission_readiness_probe'
    with probe.open('x') as stream:stream.write(plan['registration_sha256'])
    if probe.read_text()!=plan['registration_sha256']:raise ValueError('scratch write/read failed')
    probe.unlink()
    claim(out/'submission_started.json',{'at':utc(),'plan_sha256':sha(destination/'plan.json'),
        'source_revision':p['source_revision'],'account':p['resources']['account']})
    jobs={};attempts=[]
    prerequisites={'qualification':[],'collect':['qualification'],'fit5':['collect'],
                   'fit30':['collect'],'evaluate':['fit5','fit30']}
    for name,deps in prerequisites.items():
        command=['sbatch','--parsable']
        if deps:command+=['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        command+=[str(destination/(name+'.sbatch'))]
        attempts.append({'phase':name,'started_at':utc(),'command':command})
        write(out/'submission_journal.json',{'attempts':attempts,'jobs':jobs})
        try:
            result=subprocess.run(command,text=True,capture_output=True,timeout=30)
            attempts[-1].update({'finished_at':utc(),'exit_code':result.returncode,
                                'stdout':result.stdout,'stderr':result.stderr})
            job=result.stdout.strip().split(';')[0]
            if result.returncode!=0 or not job.isdigit():raise RuntimeError('Slurm submission failed/ambiguous; reconcile journal')
            jobs[name]=job
        except BaseException as error:
            attempts[-1].update({'exception_at':utc(),'error':type(error).__name__+': '+str(error)})
            raise
        finally:write(out/'submission_journal.json',{'attempts':attempts,'jobs':jobs})
    write(out/'submission_complete.json',{'at':utc(),'jobs':jobs,'source_revision':p['source_revision'],
        'registration_sha256':plan['registration_sha256'],'resources':plan['resources'],'account':'ucb736_asc1'})
    return jobs


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','submit'))
    parser.add_argument('--out',type=Path);parser.add_argument('--scripts',type=Path,required=True)
    args=parser.parse_args()
    print(prepare(args.out,args.scripts) if args.command=='prepare' else submit(args.scripts))
