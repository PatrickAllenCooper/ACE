"""Stage A: archived rows only; separate preparation, fitting and evaluation.

No simulator/model API is imported or called. Every destination is exclusive.
Fit workers cannot open the evaluation grid. Resource guards belong to the
supervisor (pilot or Slurm), not to an outcome-dependent stopping rule.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

from runner_delivery_confirmation import sha, read, write, utc, supervise, no_grid_access

DEFAULT_SOURCE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source')
DEFAULT_CAMPAIGN = DEFAULT_SOURCE.parent
DEFAULT_DEV = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/host-mirror/results/slot0/ace_results/5f89033d')
SETS = ('final_buffer', 'online_admitted', 'all_paid')
SEEDS = (27424209,1726880744,735595885,983656467,124753321,921441405,
         934168586,546725957,520888668,1091699608,585254818,884825602)


def reconstruct(rows, meta, dataset=None):
    """Union of rows available at real online updates, not probe clone updates.

    Frozen recipe: seed rows enter buffer; selected probe enters and triggers
    fitting; refreshes enter AFTER fitting. The selected row is duplicated by
    the frozen trainer. Keep unique rows for the factorial, record multiplicity.
    """
    counters = meta['query_counts']['ace']
    from runner_delivery_confirmation import validate_charged_counters
    validate_charged_counters(rows, counters)
    ids=[r['query_index'] for r in rows]
    if ids != sorted(set(ids)):
        raise ValueError('duplicate or reordered query ledger')
    buffer, used, uses, clamped = deque(maxlen=50), set(), Counter(), Counter()
    selected_steps = []
    for row in rows:
        if row['role'] == 'seed' or row['role'] == 'obs_refresh':
            buffer.append(row)
        if row.get('selected'):
            if row['role'] not in ('lookahead','teacher','breaker'):
                raise ValueError('unexpected selected row role')
            selected_steps.append(row['step'])
            buffer.append(row)
            entries = list(buffer) + [row]
            used.update(r['query_index'] for r in entries)
            uses.update(r['query_index'] for r in entries)
            for node in meta['node_mlps']:
                if node in row['interventions']:
                    clamped[node] += 1
    if selected_steps != list(range(1,meta['total_steps']+1)):
        raise ValueError('missing/reordered online update')
    if dataset is not None:
        with Path(dataset).open() as f:
            saved = list(csv.DictReader(f))
        if len(saved) != len(buffer):
            raise ValueError('final buffer length mismatch')
        for actual, expected in zip(saved,buffer):
            values = {**expected['params'],meta['target_name']:expected['outcome']}
            for key in [*meta['feature_names'],meta['target_name']]:
                if abs(float(actual[key])-float(values[key])) > 1e-8:
                    raise ValueError('final replay CSV mismatch: '+key)
            target = expected.get('intervention_target') or ''
            if actual['intervened'] != target:
                raise ValueError('final replay mask mismatch')
    return {
        'final_buffer': [r['query_index'] for r in buffer],
        'online_admitted': sorted(used),
        'all_paid': [r['query_index'] for r in rows],
        'online_use_counts': dict(sorted(uses.items())),
        'clamped_fast_adapt_updates': dict(clamped),
        'final_buffer_never_admitted': [r['query_index'] for r in buffer if r['query_index'] not in used],
    }


def specifications(meta):
    return {'flat': (meta['feature_names'],meta['target_name']),
            **{n:(p,n) for n,p in meta['causal_dag'].items() if p}}


def eligible(rows, meta, name):
    parents, target = specifications(meta)[name]
    nonroots = {n for n,p in meta['causal_dag'].items() if p}
    vectors, indices = [], []
    for row in rows:
        do = set(row['interventions'])
        if (name == 'flat' and do & nonroots) or (name != 'flat' and target in do):
            continue
        values = {**row['params'], **row['intermediates'],meta['target_name']:row['outcome']}
        vector = [float(values[p]) for p in parents]+[float(values[target])]
        import math
        if not all(math.isfinite(v) for v in vector):
            raise ValueError('nonfinite eligible row')
        vectors.append(vector); indices.append(row['query_index'])
    if not vectors:
        raise ValueError('empty eligible training set')
    return vectors, indices


def prepare(out, source=DEFAULT_SOURCE, campaign=DEFAULT_CAMPAIGN, dev=DEFAULT_DEV):
    out, source, campaign, dev = map(Path,(out,source,campaign,dev))
    out.mkdir(parents=True,exist_ok=False)
    bundles = {'development7001':dev, **{str(s):campaign/'cases'/str(s)/'online' for s in SEEDS}}
    cases = {}
    for case,bundle in bundles.items():
        meta=read(bundle/'meta.json')
        rows=[json.loads(s) for s in (bundle/'observations.ndjson').read_text().splitlines()]
        rows=[r for r in rows if r['method']=='ace']
        by_index={r['query_index']:r for r in rows}
        selection=reconstruct(rows,meta,bundle/'dataset.csv')
        dest=out/case;dest.mkdir()
        # Copy only training inputs into portable, sealed per-case JSON.
        write(dest/'input.json',{'meta':meta,'rows':rows,'selection':selection})
        cases[case]={'input_sha256':sha(dest/'input.json'),
                    'original_files':{str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()},
                    'bundle':str(bundle),
                    'counts':{s:{n:len(eligible([by_index[i] for i in selection[s]],meta,n)[0])
                                 for n in specifications(meta)} for s in SETS},
                    'never_admitted_final_rows':len(selection['final_buffer_never_admitted']),
                    'clamped_fast_adapt_updates':selection['clamped_fast_adapt_updates']}
    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn','pandas')}
    protocol={'stage':'A exploratory','frozen_at':utc(),'acquisition_calls':0,'model_api_calls':0,
              'source':str(source),'source_hashes':{str(p.relative_to(source)):sha(p) for p in (source/'ace').glob('*.py')},
              'adapter_sha256':sha(__file__),'guard_sha256':sha(Path(__file__).with_name('runner_delivery_confirmation.py')),
              'dependencies':versions,'cases':cases,'row_sets':list(SETS),
              'epochs':[100,30000],'inits':[0,1,2],'primary_init':0,
              'optimizer':{'name':'Adam','lr':0.002,'schedule':'constant','batch':'full'},
              'normalization':'training-set min/max only; zero-width becomes width 1',
              'architecture':'Linear(d,64),ReLU,Linear(64,64),ReLU,Linear(64,1)',
              'controls':{'ridge':{'alpha':1.0},'polynomial':{'degree':3,'alpha':1.0},
                          'tree':{'class':'ExtraTreesRegressor','n_estimators':256,'min_samples_leaf':2,'random_state':0,'n_jobs':1}},
              'matched_cpu':'flat all-paid init0 uses summed SCM all-paid 30000 init0 fit CPU seconds, including all 5 heads; excludes imports/evaluation',
              'control_selection':'development7001 continuous NMSE at init0; deterministic tie order flat, polynomial, tree, ridge',
              'resources':{'pilot_wall_seconds':3600,'threads':6,'rss_bytes':8*2**30,'full_cpu_core_hours':80},
              'evaluation':'full previously exposed grid, descriptive continuous MSE and exact-level error; no prospective inferential claim'}
    write(out/'protocol.json',protocol)
    print(json.dumps({'protocol_sha256':sha(out/'protocol.json'),'cases':{k:v['counts'] for k,v in cases.items()}},indent=2))


def validate(root, case, source=None):
    root=Path(root);p=read(root/'protocol.json');source=Path(source or p['source'])
    if sha(__file__) != p['adapter_sha256'] or sha(Path(__file__).with_name('runner_delivery_confirmation.py')) != p['guard_sha256']:
        raise ValueError('worker source changed after freeze')
    for name,v in p['dependencies'].items():
        if importlib.metadata.version(name) != v: raise ValueError('dependency mismatch: '+name)
    for name,h in p['source_hashes'].items():
        if sha(source/name)!=h: raise ValueError('runner source mismatch: '+name)
    if sha(root/case/'input.json') != p['cases'][case]['input_sha256']: raise ValueError('row selection changed')
    return p,read(root/case/'input.json'),source


def fit(root, case, row_set, arm, epochs, init, out, source=None, cpu_seconds=None, threads=6):
    p,data,source=validate(root,case,source)
    if row_set not in SETS or arm not in ('scm','flat','ridge','polynomial','tree') or init not in p['inits'] or epochs not in p['epochs']:
        raise ValueError('unregistered fit configuration')
    if cpu_seconds is not None and (arm!='flat' or init!=0 or row_set!='all_paid' or cpu_seconds<=0):
        raise ValueError('invalid matched-CPU arm')
    if not 1<=threads<=p['resources']['threads']:
        raise ValueError('thread request exceeds authorization')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sys.addaudithook(no_grid_access)
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    meta=data['meta'];by_index={r['query_index']:r for r in data['rows']}
    rows=[by_index[i] for i in data['selection'][row_set]]
    selected=[n for n in specifications(meta) if n!='flat'] if arm=='scm' else ['flat']
    states={};stats={};start=time.monotonic();fit_cpu=0.0
    for name in selected:
        vector,indices=eligible(rows,meta,name);z=np.asarray(vector,dtype=np.float32);x,y=z[:,:-1],z[:,-1]
        lo=x.min(axis=0);hi=x.max(axis=0);hi=np.where(hi>lo,hi,lo+1)
        cpu0=time.process_time();wall0=time.monotonic();updates=0
        if arm in ('scm','flat'):
            torch.random.default_generator.manual_seed(init+list(specifications(meta)).index(name))
            model=MLPSurrogate(x.shape[1]);model.set_input_range(lo.tolist(),hi.tolist())
            optimizer=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'])
            xt=torch.from_numpy(x);yt=torch.from_numpy(y)
            while (updates<epochs if cpu_seconds is None else time.process_time()-cpu0<cpu_seconds):
                optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(xt),yt)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();optimizer.step();updates+=1
                if updates%5000==0:print(name,updates,round(time.monotonic()-wall0,2),flush=True)
            states[name]=model.state_dict();params=sum(t.numel() for t in model.parameters())
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import ExtraTreesRegressor
            if arm=='tree':model=ExtraTreesRegressor(**{k:v for k,v in p['controls']['tree'].items() if k!='class'})
            elif arm=='polynomial':model=make_pipeline(PolynomialFeatures(3),Ridge(alpha=1.0))
            else:model=Ridge(alpha=1.0)
            scaled=(x-lo)/(hi-lo);model.fit(scaled,y)
            with (out/'regressor.pkl').open('wb') as f:pickle.dump({'model':model,'lo':lo,'hi':hi},f)
            params=(sum(t.tree_.node_count for t in model.estimators_) if arm=='tree' else
                    int(model[-1].coef_.size+1) if arm=='polynomial' else int(model.coef_.size+1))
        cpu=time.process_time()-cpu0;fit_cpu+=cpu
        stats[name]={'eligible_rows':len(indices),'query_indices_sha256':hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                     'parameters_or_tree_nodes':params,'optimizer_updates':updates,'fit_cpu_seconds':cpu,'fit_wall_seconds':time.monotonic()-wall0}
    if states:torch.save(states,out/'models.pt')
    model_file=out/('models.pt' if states else 'regressor.pkl')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    write(out/'receipt.json',{'complete':True,'case':case,'row_set':row_set,'arm':arm,'epochs':epochs,'init':init,
        'matched_cpu_seconds':cpu_seconds,'protocol_sha256':sha(Path(root)/'protocol.json'),
        'model_sha256':sha(model_file),'input_sha256':p['cases'][case]['input_sha256'],'heads':stats,
        'fit_cpu_seconds':fit_cpu,'worker_wall_seconds':time.monotonic()-start,'peak_rss_bytes':rss,'threads':threads,
        'finished_at':utc(),'new_queries':0,'evaluation':None})


def evaluate(root, case, fits, grid, out, source=None):
    p,data,source=validate(root,case,source);sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    from ace.grid_eval import GridTruth, score
    torch.set_num_threads(1)
    # Seal every expected fit BEFORE opening outcome data.
    for folder in fits:
        folder=Path(folder);r=read(folder/'receipt.json')
        file=folder/('models.pt' if (folder/'models.pt').exists() else 'regressor.pkl')
        if r['protocol_sha256']!=sha(Path(root)/'protocol.json') or r['input_sha256']!=p['cases'][case]['input_sha256'] or r['model_sha256']!=sha(file):
            raise ValueError('fit seal failed')
    meta=data['meta'];truth=GridTruth.load(grid,target=meta['target_name'],env_id=meta['env_id'])
    roots=meta['feature_names'];results={}
    by_index={r['query_index']:r for r in data['rows']}
    def predict(model,x):
        with torch.no_grad():
            return np.concatenate([model(torch.tensor(x[i:i+8192],dtype=torch.float32)).numpy() for i in range(0,len(x),8192)])
    for folder in map(Path,fits):
        r=read(folder/'receipt.json');node_diagnostics={}
        if (folder/'models.pt').exists():
            states=torch.load(folder/'models.pt',weights_only=True);models={n:MLPSurrogate.from_state_dict(s) for n,s in states.items()}
            if r['arm']=='scm':
                values={n:truth.nodes[n] for n in roots}
                for node,parents in meta['causal_dag'].items():
                    if not parents:continue
                    local=predict(models[node],np.column_stack([truth.nodes[n] for n in parents]))
                    free=predict(models[node],np.column_stack([values[n] for n in parents]));values[node]=free
                    train,_=eligible([by_index[i] for i in data['selection'][r['row_set']]],meta,node)
                    bounds=np.asarray(train)[:,:-1]
                    outside=np.any((np.column_stack([truth.nodes[n] for n in parents])<bounds.min(axis=0)) | (np.column_stack([truth.nodes[n] for n in parents])>bounds.max(axis=0)),axis=1)
                    node_diagnostics[node]={'observed_parent_mse':float(np.mean((local-truth.nodes[node])**2)),
                        'free_running_mse':float(np.mean((free-truth.nodes[node])**2)),
                        'propagated_prediction_shift_mse':float(np.mean((local-free)**2)),
                        'outside_training_parent_box_fraction':float(outside.mean())}
                pred=values[meta['target_name']]
            else:pred=predict(models['flat'],truth.inputs(roots))
        else:
            with (folder/'regressor.pkl').open('rb') as f:z=pickle.load(f)
            x=truth.inputs(roots);pred=z['model'].predict((x-z['lo'])/(z['hi']-z['lo']))
        levels=truth.levels();margins=None
        if levels is not None:
            boundaries=(levels[1:]+levels[:-1])/2
            margin=np.min(np.abs(truth.y[:,None]-boundaries),axis=1)
            margins={'fraction_prediction_error_below_true_margin':float((np.abs(pred-truth.y)<margin).mean()),
                     'margin_min':float(margin.min()),'absolute_error_quantiles':np.quantile(np.abs(pred-truth.y),[.5,.9,.99]).tolist()}
        results[folder.name]={'receipt_sha256':sha(folder/'receipt.json'),'score':score(pred,truth.y,levels),'node_diagnostics':node_diagnostics,'margins':margins}
    write(out,{'case':case,'grid_file_sha256':sha(grid),'results':results,'scope':'exposed grid; exploratory','new_queries':0})


def pilot(root, out, source=None):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');deadline=time.monotonic()+p['resources']['pilot_wall_seconds'];attempts=[]
    commands=[(s,'scm',100) for s in SETS]+[('all_paid','scm',30000),('all_paid','flat',30000),
               ('all_paid','ridge',100),('all_paid','polynomial',100),('all_paid','tree',100)]
    for s,arm,epochs in commands:
        dest=out/f'{s}-{arm}-{epochs}-i0'
        command=[sys.executable,__file__,'fit','--root',str(root),'--case','development7001','--row-set',s,'--arm',arm,'--epochs',str(epochs),'--init','0','--out',str(dest)]
        if source:command+=['--source',str(source)]
        run=supervise(command,deadline,8*2**30,out/(dest.name+'.log'),interval=1,
                      env={**os.environ,'OMP_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'6','MKL_NUM_THREADS':'6'})
        run.pop('samples',None);attempts.append({'cell':dest.name,**run});write(out/'execution.json',{'attempts':attempts,'resource_ceiling':p['resources']})
        if run['status']!='complete':return
    scm=read(out/'all_paid-scm-30000-i0'/'receipt.json');flat=read(out/'all_paid-flat-30000-i0'/'receipt.json')
    # Conservative projection: every SCM row set costs as much as all-paid;
    # 3 initializations, both budgets, flat equal epochs, one matched CPU flat,
    # 3 simple controls, 30% overhead. One CPU process at a time.
    s=scm['fit_cpu_seconds'];f=flat['fit_cpu_seconds']
    simple=sum(read(out/f'all_paid-{a}-100-i0'/'receipt.json')['fit_cpu_seconds'] for a in ('ridge','polynomial','tree'))
    estimate=1.3*12*(3*3*(s+f)*(1+100/30000)+s+simple)/3600
    write(out/'projection.json',{'complete':True,'estimated_full_cpu_core_hours':estimate,'ceiling':80,'within_cap':estimate<=80,
                               'scm_fit_cpu_seconds':s,'flat_fit_cpu_seconds':f,'guard':'full release requires validated fits and projection within cap'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','fit','evaluate','pilot'))
    parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--case');parser.add_argument('--row-set',choices=SETS,default='all_paid')
    parser.add_argument('--arm',default='scm');parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--init',type=int,default=0)
    parser.add_argument('--cpu-seconds',type=float);parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--fits',type=Path,nargs='+');parser.add_argument('--grid',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.out,a.source or DEFAULT_SOURCE)
    elif a.command=='fit':fit(a.root,a.case,a.row_set,a.arm,a.epochs,a.init,a.out,a.source,a.cpu_seconds,a.threads)
    elif a.command=='pilot':pilot(a.root,a.out,a.source)
    else:evaluate(a.root,a.case,a.fits,a.grid,a.out,a.source)


if __name__=='__main__':main()
