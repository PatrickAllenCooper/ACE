"""Frozen conditional physical validation, after the Stage A recipe gate.

No execution until --freeze consumes a complete attribution gate. The single
apparatus and all conditions are retained; no pooled independent-world test.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from runner_delivery_confirmation import read,write,sha,utc

ARCHIVE_SHA='584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9'


def validate_protocol(out):
    out=Path(out);p=read(out/'protocol.json')
    if sha(__file__)!=p['worker_sha256'] or sha(p['archive'])!=p['archive_sha256'] or p['archive_sha256']!=ARCHIVE_SHA:
        raise ValueError('source/archive changed')
    if sha(Path(__file__).with_name('runner_delivery_confirmation.py'))!=p['guard_sha256']:
        raise ValueError('resource guard changed')
    import importlib.metadata as m
    if any(m.version(n)!=v for n,v in p['dependencies'].items()):raise ValueError('dependency changed')
    for n,h in p['source_hashes'].items():
        if sha(Path(p['source'])/n)!=h:raise ValueError('runner source changed')
    with zipfile.ZipFile(p['archive']) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.csv') and Path(n).stem!='white_64')
    if p['conditions']!=names or len(names)!=11 or len({Path(n).stem for n in names})!=11:
        raise ValueError('external condition set changed')
    if p['selected_delivery'] not in ('scm','flat','ridge','polynomial','tree'):
        raise ValueError('unregistered delivery method')
    return p


def validate_fit_seal(out,p):
    """Verify every fitted artifact BEFORE evaluator opens prediction arrays."""
    out=Path(out);done=read(out/'fit_complete.json');seal=read(out/'fit_seal.json')
    if sha(out/'fit_seal.json')!=done['seal_sha256'] or seal['protocol_sha256']!=sha(out/'protocol.json'):
        raise ValueError('fit seal/protocol changed')
    if read(out/'started.json')['protocol_sha256']!=sha(out/'protocol.json'):
        raise ValueError('started protocol changed')
    summaries=seal['conditions']
    if set(summaries)!={Path(n).stem for n in p['conditions']}:
        raise ValueError('missing/extra sealed condition')
    required={'models.pt','linear_coefficients.json','predictions.npz'}
    if p['selected_delivery'] not in ('scm','flat'):required.add('selected_regressor.pkl')
    for condition,s in summaries.items():
        if set(s['artifact_hashes'])!=required:raise ValueError('incomplete model artifact seal')
        for file,h in s['artifact_hashes'].items():
            if sha(out/condition/file)!=h:raise ValueError('fitted artifact changed: '+condition+'/'+file)
        if s['train_rows']+s['test_rows']!=s['rows'] or s['train_variance']<=0:
            raise ValueError('invalid sealed split or normalizer')
    return done,summaries


def partition(angles):
    import numpy as np
    a=np.asarray(angles,float)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or not ((a>=-90)&(a<90)).all():raise ValueError('angle domain changed')
    blocks=np.floor((a+90)/30).astype(int)
    # Equal commands ALWAYS share a block. Never split repeated readings.
    return (blocks[:,0]+2*blocks[:,1])%5==0,blocks


def physics_features(angles):
    import numpy as np
    r=np.deg2rad(np.asarray(angles,float))
    return np.column_stack([np.ones(len(r)),np.cos(r[:,0]-r[:,1])**2])


def freeze(archive,gate,source,out,projection):
    if sha(archive)!=ARCHIVE_SHA:raise ValueError('archive changed')
    g=read(gate)
    if (g['stage']!='A exploratory attribution' or not g.get('complete_receipt_sha256')
            or not g.get('custody_audit',{}).get('full_acceptance')):raise ValueError('missing fully audited attribution gate')
    r=read(projection)
    if (not r['complete'] or not r['within_cap'] or r['stage_a_gate_sha256']!=sha(gate)
            or r['physical_worker_sha256']!=sha(__file__) or r['archive_sha256']!=sha(archive)
            or r['selected_delivery']!=g['selected_delivery'] or r['estimated_full_cpu_core_hours']>r['full_wall_ceiling_seconds']/3600
            or r['full_wall_ceiling_seconds']>17880):raise ValueError('physical pilot/resource gate failed')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.csv') and Path(n).stem!='white_64')
    if len(names)!=11:raise ValueError('external conditions changed')
    import importlib.metadata as m
    write(out/'protocol.json',{'at':utc(),'archive':str(archive),'archive_sha256':ARCHIVE_SHA,'stage_a_gate_sha256':sha(gate),
        'conditions':names,'development_excluded':'white_64','selected_delivery':g['selected_delivery'],
        'source':str(source),'source_hashes':{str(p.relative_to(source)):sha(p) for p in (Path(source)/'ace').glob('*.py')},
        'worker_sha256':sha(__file__),'guard_sha256':sha(Path(__file__).with_name('runner_delivery_confirmation.py')),
        'dependencies':{n:m.version(n) for n in ('torch','numpy','scikit-learn')},
        'endpoint':'test MSE divided by training variance; per-condition descriptive',
        'grouping':'joint 30-degree angle blocks; identical commands indivisible',
        'split':'(block1+2*block2)%5==0 test','epochs':30000,'init':0,'lr':.002,
        'online':'one 100-epoch fit per archived training row, last 50 rows; persistent Adam, causal masking vacuous',
        'neural_output_scaling':'training-only target mean and sd, both delivery and rolling buffer',
        'physics':'intercept + cos(relative angle)^2','fourier':'tensor of 1,sin2,cos2,sin4,cos4',
        'uncertainty':'paired action-block bootstrap conditional on observed conditions; not independent worlds',
        'bootstrap_seed':620061,'bootstrap_replicates':2000,'cpu_core_hour_ceiling':5,'threads':1,'new_physical_queries':0,
        'account':'ucb736_asc1','output':str(out),'rss_bytes':8*2**30,
        'execution_environment':'CPU; transport/output readiness must be validated before any CURC submission'})
    p=read(out/'protocol.json')
    if r['dependencies']!=p['dependencies'] or r['source_hashes']!=p['source_hashes']:raise ValueError('pilot runtime/source changed')
    p['resource_projection_sha256']=sha(projection);p['wall_seconds']=r['full_wall_ceiling_seconds']
    write(out/'protocol.json',p)


def fit(out):
    out=Path(out);p=validate_protocol(out)
    from runner_delivery_confirmation import no_network
    sys.addaudithook(no_network)
    with (out/'started.json').open('x') as f:json.dump({'at':utc(),'protocol_sha256':sha(out/'protocol.json')},f)
    sys.path.insert(0,p['source'])
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();cpu0=time.process_time();predictions={};summaries={}
    with zipfile.ZipFile(p['archive']) as z:
        for name in p['conditions']:
            rows=list(csv.DictReader(io.TextIOWrapper(z.open(name))))
            a=np.array([[float(r['pol_1']),float(r['pol_2'])] for r in rows]);y=np.array([float(r['vis_3']) for r in rows])
            if not np.isfinite(y).all():raise ValueError('nonfinite response')
            test,blocks=partition(a);train=~test;mean=float(y[train].mean());sd=float(y[train].std())
            if sd<=0 or test.sum()==0 or train.sum()<50:raise ValueError('degenerate condition')
            x=torch.tensor(a/90,dtype=torch.float32);t=torch.tensor((y-mean)/sd,dtype=torch.float32)
            state={};pred={};cost={};coefficients={}
            # Fit all requested methods without reading their test losses.
            selected=p['selected_delivery']
            methods=['rolling_buffer']+(['delivery_mlp'] if selected in ('scm','flat') else [])
            for method in methods:
                torch.random.default_generator.manual_seed(0);model=MLPSurrogate(2)
                opt=torch.optim.Adam(model.parameters(),lr=.002);fit0=time.process_time();updates=0
                sequence=[np.where(train)[0]] if method=='delivery_mlp' else [np.where(train)[0][max(0,k-49):k+1] for k in range(train.sum())]
                for indices in sequence:
                    for _ in range(30000 if method=='delivery_mlp' else 100):
                        opt.zero_grad();loss=torch.nn.functional.mse_loss(model(x[indices]),t[indices])
                        if not torch.isfinite(loss):raise ValueError('nonfinite neural training loss')
                        loss.backward();opt.step();updates+=1
                state[method]=model.state_dict();cost[method]={'cpu_seconds':time.process_time()-fit0,'updates':updates,'parameters':sum(q.numel() for q in model.parameters())}
                with torch.no_grad():pred[method]=model(x).numpy()*sd+mean
            r=np.deg2rad(a)
            def b(v):return np.column_stack([np.ones(len(v)),np.sin(2*v),np.cos(2*v),np.sin(4*v),np.cos(4*v)])
            features={'physics':physics_features(a),
                      'fourier':np.einsum('ni,nj->nij',b(r[:,0]),b(r[:,1])).reshape(len(y),-1)}
            for method,f in features.items():
                fit0=time.process_time()
                coef=np.linalg.lstsq(f[train],y[train],rcond=None)[0];pred[method]=f@coef
                coefficients[method]=coef.tolist()
                cost[method]={'cpu_seconds':time.process_time()-fit0,'parameters':len(coef),'updates':0}
            if selected not in ('scm','flat'):
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import make_pipeline
                from sklearn.ensemble import ExtraTreesRegressor
                model=(ExtraTreesRegressor(n_estimators=256,min_samples_leaf=2,random_state=0,n_jobs=1) if selected=='tree' else
                       make_pipeline(PolynomialFeatures(3),Ridge(alpha=1.0)) if selected=='polynomial' else Ridge(alpha=1.0))
                fit0=time.process_time();model.fit(a[train]/90,y[train]);pred['delivery']=model.predict(a/90)
                cost['delivery']={'cpu_seconds':time.process_time()-fit0,'updates':0,
                    'parameters_or_tree_nodes':sum(t.tree_.node_count for t in model.estimators_) if selected=='tree' else
                    int(model[-1].coef_.size+1) if selected=='polynomial' else int(model.coef_.size+1)}
            else:pred['delivery']=pred['delivery_mlp']
            condition=Path(name).stem;dest=out/condition;dest.mkdir(exist_ok=False)
            torch.save(state,dest/'models.pt')
            write(dest/'linear_coefficients.json',coefficients)
            if selected not in ('scm','flat'):
                import pickle
                with (dest/'selected_regressor.pkl').open('wb') as f:pickle.dump(model,f)
            np.savez(dest/'predictions.npz',**pred,y=y,test=test,blocks=blocks)
            # Persist fits/splits BEFORE scoring any of the eleven conditions.
            summaries[condition]={'rows':len(y),'train_rows':int(train.sum()),'test_rows':int(test.sum()),'train_variance':sd**2,
                'cost':cost,'artifact_hashes':{file.name:sha(file) for file in dest.iterdir() if file.is_file()}}
            print(condition,'fits sealed',int(train.sum()),int(test.sum()),flush=True)
            if time.monotonic()-start>5*3600:raise RuntimeError('Stage C CPU wall ceiling')
    write(out/'fit_seal.json',{'conditions':summaries,'protocol_sha256':sha(out/'protocol.json'),'at':utc()})
    write(out/'fit_complete.json',{'at':utc(),'cpu_core_hours':(time.process_time()-cpu0)/3600,
                                 'wall_seconds':time.monotonic()-start,'seal_sha256':sha(out/'fit_seal.json')})


def evaluate(out):
    # Evaluator is launched as a distinct process after all eleven fits seal.
    out=Path(out);p=validate_protocol(out);done,summaries=validate_fit_seal(out,p)
    from runner_delivery_confirmation import no_network
    sys.addaudithook(no_network)
    import numpy as np
    results={};rng=np.random.default_rng(p['bootstrap_seed']);start=time.monotonic();cpu0=time.process_time()
    for condition,s in summaries.items():
        with np.load(out/condition/'predictions.npz') as z:
            test=z['test'];y=z['y'][test];blocks=z['blocks'][test];ids=blocks[:,0]*6+blocks[:,1];unique=np.unique(ids)
            scores={method:float(np.mean((z[method][test]-y)**2)/s['train_variance']) for method in ('delivery','rolling_buffer','physics','fourier')}
            intervals={}
            for control in ('rolling_buffer','physics','fourier'):
                error=(z['delivery'][test]-y)**2;other=(z[control][test]-y)**2
                # Aggregate readings within command blocks before resampling.
                grouped=np.array([[error[ids==k].mean(),other[ids==k].mean()] for k in unique])
                ratios=[]
                for _ in range(p['bootstrap_replicates']):
                    sampled=grouped[rng.integers(0,len(unique),len(unique))].mean(axis=0)
                    ratios.append(sampled[0]/max(sampled[1],1e-12))
                intervals[control]={'block_weighted_ratio':float(grouped[:,0].mean()/max(grouped[:,1].mean(),1e-12)),
                                    'conditional_bootstrap_ci95':np.quantile(ratios,[.025,.975]).tolist(),'n_action_blocks':len(unique)}
            results[condition]={'nmse':scores,'conditional_uncertainty':intervals}
    write(out/'scores.json',{'conditions':results,'scope':'single apparatus, condition-specific archived prediction; no population-world significance','new_queries':0})
    write(out/'complete.json',{'at':utc(),'scores_sha256':sha(out/'scores.json'),'fit_seal_sha256':sha(out/'fit_seal.json'),
                              'cpu_core_hours':done['cpu_core_hours']+(time.process_time()-cpu0)/3600,
                              'wall_seconds':done['wall_seconds']+time.monotonic()-start})


def run(out):
    from runner_delivery_confirmation import supervise
    p=validate_protocol(out)
    if not 0<p['wall_seconds']<=17880:raise ValueError('physical wall ceiling changed')
    deadline=time.monotonic()+p['wall_seconds']
    for phase in ('fit','evaluate'):
        r=supervise([sys.executable,__file__,phase,'--out',str(out)],deadline,8*2**30,
                    Path(out)/(phase+'.log'),interval=2,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
        samples=r.pop('samples',[]);r['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
        write(Path(out)/(phase+'_execution.json'),r)
        if r['status']!='complete':raise RuntimeError('Stage C stopped: '+r['status'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=('freeze','run','fit','evaluate'));a.add_argument('--out',type=Path,required=True)
    a.add_argument('--archive',type=Path);a.add_argument('--gate',type=Path);a.add_argument('--source',type=Path)
    a.add_argument('--projection',type=Path);args=a.parse_args()
    if args.command=='freeze':freeze(args.archive,args.gate,args.source,args.out,args.projection)
    else:globals()[args.command](args.out)
