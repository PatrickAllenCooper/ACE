"""Execute a frozen Stage A matrix; never select cells using evaluation results."""
import argparse
import os
from pathlib import Path
import sys
import time
from runner_delivery_confirmation import read,write,sha,utc,supervise
from delivery_attribution import SEEDS,SETS


def cells(case,out):
    out=Path(out)
    result=[]
    for dataset in SETS:
        for arm in ('scm','flat'):
            for epochs in (100,30000):
                for init in (0,1,2):
                    result.append({'case':case,'row_set':dataset,'arm':arm,'epochs':epochs,'init':init,
                                   'path':str(out/case/f'{dataset}-{arm}-{epochs}-i{init}')})
    for arm in ('ridge','polynomial','tree'):
        result.append({'case':case,'row_set':'all_paid','arm':arm,'epochs':100,'init':0,
                       'path':str(out/case/f'all_paid-{arm}-100-i0')})
    result.append({'case':case,'row_set':'all_paid','arm':'flat','epochs':30000,'init':0,
                   'matched_cpu':True,'path':str(out/case/'all_paid-flat-matched-cpu-i0')})
    return result


def freeze(root,out,projection,development_scores,source,grid,wall_seconds=28800):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    projection=read(projection);scores=read(development_scores)
    if not projection['complete'] or not projection['within_cap'] or projection['estimated_full_cpu_core_hours']>80:
        raise ValueError('pilot projection has not released Stage A')
    simpler=[('flat','all_paid-flat-30000-i0'),('polynomial','all_paid-polynomial-100-i0'),
             ('tree','all_paid-tree-100-i0'),('ridge','all_paid-ridge-100-i0')]
    selected=min(simpler,key=lambda x:scores['results'][x[1]]['score']['nmse'])[0]
    if wall_seconds*6>80*3600:raise ValueError('requested full stage exceeds CPU ceiling')
    matrix=[cell for seed in SEEDS for cell in cells(str(seed),out)]
    reg={'frozen_at':utc(),'authorization':'Patrick explicit implementation of delivery-centered plan, 2026-10-06',
         'root':str(root),'protocol_sha256':sha(Path(root)/'protocol.json'),
         'source':str(source),'grid':str(grid),'grid_sha256':sha(grid),
         'adapter_sha256':sha(Path(__file__).with_name('delivery_attribution.py')),'batch_sha256':sha(__file__),
         'strongest_simpler_development':selected,'development_scores_sha256':sha(development_scores),
         'projection':projection,'wall_seconds':wall_seconds,'threads':6,'rss_bytes':8*2**30,
         'matrix':matrix,'evaluate_only_after_all_480_fits':True,'new_simulator_calls':0,
         'account':'ucb736_asc1','job_id':None}
    write(out/'registration.json',reg)
    return reg


def run(out):
    out=Path(out);reg=read(out/'registration.json')
    if sha(__file__)!=reg['batch_sha256'] or sha(Path(__file__).with_name('delivery_attribution.py'))!=reg['adapter_sha256']:
        raise ValueError('frozen worker changed')
    if sha(Path(reg['root'])/'protocol.json')!=reg['protocol_sha256'] or sha(reg['grid'])!=reg['grid_sha256']:
        raise ValueError('protocol/grid binding changed')
    # O_EXCL claim prevents concurrent or duplicate launches, including after interruption.
    with (out/'started.json').open('x') as f:
        import json
        json.dump({'at':utc(),'job_id':os.environ.get('SLURM_JOB_ID'),'registration_sha256':sha(out/'registration.json')},f)
    deadline=time.monotonic()+reg['wall_seconds'];attempts=[];env={**os.environ,'OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'6'}
    for cell in reg['matrix']:
        dest=Path(cell['path']);dest.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(Path(__file__).with_name('delivery_attribution.py')),'fit',
                 '--root',reg['root'],'--source',reg['source'],'--case',cell['case'],
                 '--row-set',cell['row_set'],'--arm',cell['arm'],'--epochs',str(cell['epochs']),
                 '--init',str(cell['init']),'--out',str(dest)]
        if cell.get('matched_cpu'):
            budget=read(dest.parent/'all_paid-scm-30000-i0'/'receipt.json')['fit_cpu_seconds']
            command+=['--cpu-seconds',str(budget)]
        r=supervise(command,deadline,reg['rss_bytes'],out/(cell['case']+'-'+dest.name+'.log'),interval=2,env=env)
        samples=r.pop('samples',[]);r['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
        attempts.append({'cell':cell,**r});write(out/'execution.json',{'attempts':attempts,'complete':False})
        if r['status']!='complete':raise RuntimeError('fit stopped: '+r['status'])
    # Full matrix seal before any new twelve-history scores. Evaluations are
    # sequential CPU1 children and share the same overall deadline/RSS guard.
    write(out/'fit_seal.json',{'registration_sha256':sha(out/'registration.json'),
         'receipts':{c['path']:sha(Path(c['path'])/'receipt.json') for c in reg['matrix']},'at':utc()})
    for seed in SEEDS:
        case=str(seed);case_cells=[c for c in reg['matrix'] if c['case']==case]
        command=[sys.executable,str(Path(__file__).with_name('delivery_attribution.py')),'evaluate',
                 '--root',reg['root'],'--source',reg['source'],'--case',case,'--grid',reg['grid'],
                 '--out',str(out/case/'scores.json'),'--fits',*[c['path'] for c in case_cells]]
        r=supervise(command,deadline,reg['rss_bytes'],out/(case+'-evaluation.log'),interval=2,env={**env,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        r.pop('samples',None)
        if r['status']!='complete':
            write(out/'evaluation_failure.json',{'case':case,**r})
            raise RuntimeError('evaluation stopped: '+r['status'])
    cpu=sum(read(Path(c['path'])/'receipt.json')['fit_cpu_seconds'] for c in reg['matrix'])
    write(out/'complete.json',{'at':utc(),'n_histories':12,'n_fits':len(reg['matrix']),'fit_cpu_core_hours':cpu/3600,
        'job_id':os.environ.get('SLURM_JOB_ID'),'new_queries':0,'registration_sha256':sha(out/'registration.json'),
        'score_hashes':{str(s):sha(out/str(s)/'scores.json') for s in SEEDS}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path);p.add_argument('--projection',type=Path);p.add_argument('--development-scores',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--grid',type=Path);p.add_argument('--wall-seconds',type=int,default=28800)
    a=p.parse_args()
    if a.command=='run':run(a.out)
    else:freeze(a.root,a.out,a.projection,a.development_scores,a.source,a.grid,a.wall_seconds)
