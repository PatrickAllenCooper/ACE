"""Development-only Stage C timing; no external condition outcomes or tuning."""
import argparse
import csv
import io
import os
from pathlib import Path
import sys
import time
import zipfile
from runner_delivery_confirmation import sha,read,write,utc,supervise
from chambers_delivery_validation import ARCHIVE_SHA,partition


def profile(archive,gate,source,out):
    cpu0=time.process_time();wall0=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    g=read(gate)
    if not g.get('custody_audit',{}).get('full_acceptance'):raise ValueError('Stage A gate not audited')
    if sha(archive)!=ARCHIVE_SHA:raise ValueError('archive changed')
    from runner_delivery_confirmation import no_network
    sys.addaudithook(no_network)
    write(out/'started.json',{'at':utc(),'gate_sha256':sha(gate),'worker_sha256':sha(__file__),
                              'physical_worker_sha256':sha(Path(__file__).with_name('chambers_delivery_validation.py'))})
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    import importlib.metadata as m
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if Path(n).stem=='white_64' and n.endswith('.csv')]
        if len(names)!=1:raise ValueError('development condition changed')
        rows=list(csv.DictReader(io.TextIOWrapper(z.open(names[0]))))
    a=np.array([[float(r['pol_1']),float(r['pol_2'])] for r in rows]);y=np.array([float(r['vis_3']) for r in rows])
    test,_=partition(a);train=~test
    x=torch.tensor(a[train]/90,dtype=torch.float32)
    t=torch.tensor((y[train]-y[train].mean())/y[train].std(),dtype=torch.float32)
    costs={};states={};selected=g['selected_delivery']
    methods=['rolling_buffer']+(['delivery_mlp'] if selected in ('scm','flat') else [])
    for method in methods:
        torch.random.default_generator.manual_seed(0);model=MLPSurrogate(2)
        opt=torch.optim.Adam(model.parameters(),lr=.002)
        # Match the delivery batch and the rolling learner's steady-state50
        # buffer. Fixed2,000updates estimate cost, never score/pick a recipe.
        xx,tt=(x[-50:],t[-50:]) if method=='rolling_buffer' else (x,t)
        begin=time.process_time()
        for _ in range(2000):
            opt.zero_grad();loss=torch.nn.functional.mse_loss(model(xx),tt)
            if not torch.isfinite(loss):raise ValueError('nonfinite pilot loss')
            loss.backward();opt.step()
        costs[method]={'updates':2000,'cpu_seconds':time.process_time()-begin}
        states[method]=model.state_dict()
    torch.save(states,out/'pilot_models.pt')
    simple_cpu=0.
    if selected not in ('scm','flat'):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import ExtraTreesRegressor
        import pickle
        model=(ExtraTreesRegressor(n_estimators=256,min_samples_leaf=2,random_state=0,n_jobs=1) if selected=='tree' else
               make_pipeline(PolynomialFeatures(3),Ridge(alpha=1.0)) if selected=='polynomial' else Ridge(alpha=1.0))
        begin=time.process_time();model.fit(a[train]/90,y[train]);simple_cpu=time.process_time()-begin
        with (out/'pilot_regressor.pkl').open('wb') as f:pickle.dump(model,f)
    # All archive cells contain1,000rows (validated metadata); use1,000 as
    # the conservative training upper bound,2x timing margin,900s overhead.
    projected=11*(costs['rolling_buffer']['cpu_seconds']/2000*100*1000+simple_cpu)
    if 'delivery_mlp' in costs:projected+=11*costs['delivery_mlp']['cpu_seconds']/2000*30000
    projected=2*projected+900
    write(out/'projection.json',{'at':utc(),'complete':True,'selected_delivery':selected,
        'stage_a_gate_sha256':sha(gate),'archive_sha256':ARCHIVE_SHA,
        'physical_worker_sha256':sha(Path(__file__).with_name('chambers_delivery_validation.py')),
        'pilot_worker_sha256':sha(__file__),'dependencies':{n:m.version(n) for n in ('torch','numpy','scikit-learn')},
        'source_hashes':{str(p.relative_to(source)):sha(p) for p in (Path(source)/'ace').glob('*.py')},
        'condition':'white_64 development only','costs':costs,'simple_fit_cpu_seconds':simple_cpu,
        'estimated_full_cpu_core_hours':projected/3600,'full_wall_ceiling_seconds':17880,
        'within_cap':projected<=17880,'pilot_cpu_core_hours':(time.process_time()-cpu0)/3600,
        'pilot_wall_seconds':time.monotonic()-wall0,'pilot_wall_ceiling_seconds':90,
        'margin':'2x measured update costs +900s; reserve120s from5h for pilot and supervision',
        'test_losses_computed':False,'external_conditions_read':0,'new_queries':0,
        'model_hashes':{f.name:sha(f) for f in out.glob('pilot_*') if f.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('profile','run'))
    for name in ('archive','gate','source','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.command=='profile':profile(a.archive,a.gate,a.source,a.out)
    else:
        result=supervise([sys.executable,__file__,'profile','--archive',str(a.archive),'--gate',str(a.gate),
                          '--source',str(a.source),'--out',str(a.out)],time.monotonic()+90,1024**3,
                         a.out.with_suffix('.log'),interval=2,
                         env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
        samples=result.pop('samples',[]);result['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
        write(a.out.with_suffix('.execution.json'),result)
        if result['status']!='complete':raise RuntimeError('physical pilot stopped: '+result['status'])
