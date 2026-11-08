"""Development-only CPU timing gate for Stage B; no confirmation/test losses."""
import argparse
from pathlib import Path
import os
import sys
import time

from runner_delivery_confirmation import read, write, sha, utc, supervise
from delivery_prospective_models import dependencies

FILES=('delivery_prospective_pilot.py','delivery_prospective_models.py',
       'delivery_prospective_design.py','runner_delivery_confirmation.py','test_delivery_prospective_models.py')


def prepare(project, source, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    project,source=Path(project),Path(source)
    write(out/'registration.json',{'at':utc(),'stage':'B development-only resource pilot',
        'project':str(project),'source':str(source),'worker_hashes':{n:sha(Path(__file__).with_name(n)) for n in FILES},
        'source_hashes':{str(p.relative_to(source)):sha(p) for p in (source/'ace').glob('*.py')},
        'generator_hashes':{n:sha(project/n) for n in ('baselines.py','experiments/large_scale_scm.py')},
        'dependencies':dependencies(),'development_coefficient_seed':0,'action_seed':17,
        'rows_per_development_history':400,'scratch_epochs':2000,'online_tail_updates':10,
        'threads':1,'rss_bytes':3*2**30,'wall_seconds':900,'account':'ucb736_asc1',
        'output':str(out),'full_stage_ceiling_cpu_core_hours':150,
        'confirmation_outcomes_permitted':False,'test_losses_permitted':False,
        'authorization':'existing explicitly authorized Stage B allowance, development pilot only'})


def validate(out):
    p=read(Path(out)/'registration.json')
    for name,h in p['worker_hashes'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('pilot worker changed')
    for name,h in p['source_hashes'].items():
        if sha(Path(p['source'])/name)!=h:raise ValueError('original learner source changed')
    for name,h in p['generator_hashes'].items():
        if sha(Path(p['project'])/name)!=h:raise ValueError('generator source changed')
    if dependencies()!=p['dependencies'] or p['development_coefficient_seed']!=0:
        raise ValueError('pilot dependencies or development seed changed')
    if p['confirmation_outcomes_permitted'] or p['test_losses_permitted']:
        raise ValueError('pilot may not expose confirmation or evaluation outcomes')
    return p


def work(out):
    out=Path(out);p=validate(out)
    from runner_delivery_confirmation import no_network
    sys.addaudithook(no_network)
    with (out/'started.json').open('x') as f:
        import json
        json.dump({'at':utc(),'registration_sha256':sha(out/'registration.json'),'job_id':os.environ.get('SLURM_JOB_ID')},f)
    from delivery_prospective_design import world_spec,validate_world,shared_actions,structural_values
    from delivery_prospective_models import fit,runtime
    torch,_,_=runtime(p['source']);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    results={};cpu0=time.process_time();wall0=time.monotonic()
    for size in (5,30):
        spec=world_spec(size,0,p['project']);roots=validate_world(spec)
        # Random collection actions, no held-out responses. Every timing cell
        # sees exactly the same400development rows; no numerical model scores.
        actions=shared_actions(spec,'random',17,400)
        data={'order':spec['order'],'parents':spec['parents'],'roots':roots,
              'target':'X3' if size==5 else 'X30','rows':[]}
        for i,action in enumerate(actions):
            data['rows'].append({'query_index':i+1,'clamps':action,'node_values':structural_values(spec,action)})
        dest=out/str(size);dest.mkdir();write(dest/'development_input.json',data)
        results[str(size)]={}
        for arm in ('delivery','simpler','ablation','online'):
            kwargs={'epochs':2000} if arm in ('delivery','simpler') else {'online_tail':10} if arm=='online' else {}
            states,cost=fit(data,p['source'],arm,**kwargs)
            torch.save(states,dest/(arm+'_models.pt'))
            write(dest/(arm+'_cost.json'),{**cost,'input_sha256':sha(dest/'development_input.json'),
                'models_sha256':sha(dest/(arm+'_models.pt'))})
            results[str(size)][arm]=cost
            print(size,arm,cost['cpu_seconds'],flush=True)
    # Conservative projection includes all640 fits:320 init0 primary/ablation
    # fits plus320 init1/2 delivery/flat sensitivity fits. No scored selection.
    totals={}
    for size,v in results.items():
        per_history=(3*(v['delivery']['cpu_seconds']+v['simpler']['cpu_seconds'])*30000/2000
                     +v['ablation']['cpu_seconds']+v['online']['cpu_seconds']*400/10)
        totals[size]={'histories':40,'raw_full_fit_cpu_seconds':40*per_history}
    raw=sum(v['raw_full_fit_cpu_seconds'] for v in totals.values())
    # 3x hardware/optimizer/runtime margin;640x5s process startup plus5,000s
    # collection, evaluation, checks and statistics. Reserved pilot cap900s.
    estimated=(3*raw+640*5+5000+900)/3600
    projection={'at':utc(),'complete':True,'registration_sha256':sha(out/'registration.json'),
        'dependencies':p['dependencies'],'worker_hashes':p['worker_hashes'],'source_hashes':p['source_hashes'],
        'generator_hashes':p['generator_hashes'],'timings':results,'strata':totals,
        'matrix_fits':640,'estimated_full_cpu_core_hours':estimated,'within_cap':estimated<=150,
        'pilot_cpu_core_hours':(time.process_time()-cpu0)/3600,'pilot_wall_seconds':time.monotonic()-wall0,
        'ceiling_cpu_core_hours':150,'projection_formula':'3x projected fit CPU +640x5s startup +5000s overhead +900s reserved pilot',
        'test_losses_computed':False,'confirmation_responses_evaluated':0,'development_responses':800,
        'status':'resource advisory only; not Stage B scientific release'}
    write(out/'projection.json',projection)


def run(out):
    p=validate(out)
    if (Path(out)/'supervisor_started.json').exists():raise ValueError('pilot already launched; no duplicate')
    with (Path(out)/'supervisor_started.json').open('x') as f:
        import json
        json.dump({'at':utc(),'job_id':os.environ.get('SLURM_JOB_ID')},f)
    deadline=time.monotonic()+900
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','ACE_DELIVERY_RUNNER_SOURCE':p['source']}
    qualification=supervise([sys.executable,str(Path(__file__).with_name('test_delivery_prospective_models.py'))],
        deadline,3*2**30,Path(out)/'qualification.log',interval=2,env=env)
    samples=qualification.pop('samples',[]);qualification['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
    write(Path(out)/'qualification_execution.json',qualification)
    if qualification['status']!='complete':raise RuntimeError('target-runtime qualification failed')
    result=supervise([sys.executable,__file__,'work','--out',str(out)],deadline,3*2**30,
        Path(out)/'work.log',interval=2,env=env)
    samples=result.pop('samples',[]);result['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
    write(Path(out)/'execution.json',result)
    if result['status']!='complete':raise RuntimeError('development pilot stopped: '+result['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','work'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--project',type=Path);parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.project,args.source,args.out)
    else:globals()[args.command](args.out)
