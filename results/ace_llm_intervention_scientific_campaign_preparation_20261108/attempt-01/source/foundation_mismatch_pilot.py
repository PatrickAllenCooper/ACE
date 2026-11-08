#!/usr/bin/env python3
"""Fresh bounded development screen. Authentication precedes numerical imports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

SEEDS=tuple(range(92000,92006))
VARIANTS=('null','coefficient_M','missing_M','missing_Y')
METHODS=('grammar32','grammar24','pfn24','terminal24','mechanism24','prechange24')
ENDPOINTS=('M_local','Y_local','Y_composed')
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    raw=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
    with Path(str(path)+'.pending').open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(str(path)+'.pending',path);Path(str(path)+'.pending').unlink()


def load_source(name,path,pin):
    if name in sys.modules:raise ValueError('cached research module: '+name)
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('source bytes mismatch')
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def initialize(freeze):
    global np,base,sel
    required={'numpy','torch','transformers','tabpfn','scikit-learn','scipy','safetensors','tokenizers','huggingface-hub'}
    if set(freeze['dependencies'])!=required:raise ValueError('dependency closure')
    actual={n:importlib.metadata.version(n) for n in required}
    if actual!=freeze['dependencies']:raise ValueError('runtime versions mismatch')
    if sys.version!=freeze['python_version'] or str(Path(sys.executable).absolute())!=freeze['python']:
        raise ValueError('interpreter mismatch')
    if sha(freeze['checkpoint'])!=freeze['checkpoint_sha256'] or freeze['checkpoint_sha256']!='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736':
        raise ValueError('checkpoint mismatch')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    base=load_source('ace_mismatch_base',ROOT/'scripts/research/foundation_component_pilot.py',freeze['sources']['scripts/research/foundation_component_pilot.py'])
    sel=load_source('ace_mismatch_selection',ROOT/'scripts/research/foundation_mixture_selection.py',freeze['sources']['scripts/research/foundation_mixture_selection.py'])
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    return actual


def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))


def specification(seed,artificial=False):
    if artificial:
        if seed!=123456:raise ValueError('artificial identity')
        return {'families':['linear','quadratic'],'coefficients':[[.1,.8],[.2,1.1,.3]]}
    if seed not in SEEDS:raise ValueError('scientific seed outside protocol')
    generator=rng(seed,0);index=seed-SEEDS[0]
    families=[base.FAMILIES[index%3],base.FAMILIES[(index+1)%3]];coeff=[]
    for family in families:
        c=[float(generator.uniform(-.3,.3)),float(generator.uniform(.6,1.4))]
        if family=='quadratic':c.append(float(generator.uniform(-.7,.7)))
        coeff.append(c)
    return {'families':families,'coefficients':coeff}


def truth(spec,variant,node,x):
    c=np.array(spec['coefficients'][node],copy=True);family=spec['families'][node]
    if variant=='coefficient_M' and node==0:c[0]+=.2;c[1:]*=1.5
    if (variant=='missing_M' and node==0) or (variant=='missing_Y' and node==1):
        return c[0]+c[1]*np.sin(np.pi*np.asarray(x))
    return base.basis(x,family)@c


def history(spec,variant,seed,stream):
    gen=rng(seed,stream);menu=np.array([-1.,-.5,0.,.5,1.])
    x=np.concatenate((gen.uniform(-1,1,8),np.resize(menu,12),gen.uniform(-1,1,12)))
    m=truth(spec,variant,0,x)+gen.normal(0,.05,32);m[20:]=np.resize(menu,12)
    y=truth(spec,variant,1,m)+gen.normal(0,.05,32)
    return tuple(sel.Row(float(a),float(b),float(c),clamp) for a,b,c,clamp in zip(x,m,y,('none',)*8+('X',)*12+('M',)*12))


def arrays(rows):return np.array([(r.x,r.m,r.y) for r in rows]),np.array([r.clamp for r in rows])


class Head:
    """Only measured-parent fit arrays cross the learner boundary."""
    def __init__(self,method,x,y,checkpoint):
        self.fit_count=1;self.prediction_calls=0;self.prediction_rows=0
        self.model=base.make_model(method,checkpoint).fit(x.copy(),y.copy())
        self.fit_rows=len(y)
    def __call__(self,x):
        self.prediction_calls+=1;self.prediction_rows+=len(x)
        return np.asarray(self.model.predict(np.asarray(x).reshape(-1,1))).reshape(-1)
    def record(self):
        return {'fit_count':self.fit_count,'fit_rows':self.fit_rows,'prediction_calls':self.prediction_calls,'prediction_rows':self.prediction_rows,'family':getattr(self.model,'family',None),'coefficients':getattr(self.model,'coef',np.array([])).tolist()}


def fit_heads(rows,method,checkpoint):
    train,clamps=arrays(rows)
    return tuple(Head(method,x,y,checkpoint) for x,y in base.eligible(train,clamps))


def fit_or_failure(rows,method,checkpoint):
    start=time.monotonic();cpu=time.process_time()
    try:heads=fit_heads(rows,method,checkpoint);error=None
    except Exception:heads=None;error=traceback.format_exc()
    return heads,{'status':'complete' if heads else 'failed','error':error,'elapsed_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu}


def make_predictors(rows,pre,checkpoint):
    """Has no seed, variant, truth or private probes; outputs fixed predictors."""
    fit,calibration=sel.split_history(rows);heads={};events={};choices={};errors={}
    for name,method,data in [('grammar32','grammar',rows),('grammar24','grammar',fit),('pfn24','tabpfn_v2',fit)]:
        heads[name],events[name]=fit_or_failure(data,method,checkpoint)
        if heads[name] is None:errors[name]=events[name]['error']
    heads['prechange24']=pre
    if pre is None:errors['prechange24']='pre-change fit failed; see prechange_fit.json'
    for name,choose in [('terminal24',sel.terminal_choice),('mechanism24',sel.mechanism_choice)]:
        start=time.monotonic();cpu=time.process_time()
        try:
            if heads['grammar24'] is None or heads['pfn24'] is None:raise RuntimeError('required expert fit failed')
            choice=choose(calibration,*heads['grammar24'],*heads['pfn24'])
            choices[name]={'weights':choice.weights,'mse':choice.mse,'candidates':choice.candidates}
        except Exception:errors[name]=traceback.format_exc()
        events[name]={'elapsed_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu}
    return heads,choices,errors,events


def private_probes(spec,variant,seed):
    # Three distinct full responses; even duplicated values count once per declared query.
    x=rng(seed,100).uniform(-1,1,256);m=truth(spec,variant,0,x)
    composed=np.column_stack((x,m,truth(spec,variant,1,m)))
    x=rng(seed,101).uniform(-1,1,256);m=truth(spec,variant,0,x)
    local_m=np.column_stack((x,m,truth(spec,variant,1,m)))
    m=rng(seed,102).uniform(-2,2,256)
    local_y=np.column_stack((np.zeros(256),m,truth(spec,variant,1,m)))
    return composed,local_m,local_y


def predictions(name,heads,choices,probes):
    comp,lm,ly=probes;x=tuple(comp[:,0]);xm=tuple(lm[:,0]);ym=tuple(ly[:,1])
    if name in ('terminal24','mechanism24'):
        gm,gy=heads['grammar24'];pm,py=heads['pfn24'];weights=choices[name]['weights']
        if name=='terminal24':
            w=weights[0];m=sel.blend(gm(xm),pm(xm),w);y=sel.blend(gy(ym),py(ym),w)
            composed=sel.terminal_forecast(x,gm,gy,pm,py,w)
        else:
            m=sel.blend(gm(xm),pm(xm),weights[0]);y=sel.blend(gy(ym),py(ym),weights[1])
            composed=sel.mechanism_forecast(x,gm,gy,pm,py,weights)
    else:
        mhead,yhead=heads[name];m=mhead(xm);y=yhead(ym);composed=yhead(mhead(x))
    return np.column_stack((m,y,composed))


def metrics(pred,probes,rows):
    comp,lm,ly=probes;targets=np.column_stack((lm[:,1],ly[:,2],comp[:,2]))
    train,clamp=arrays(rows);data=base.eligible(train,clamp)
    variances=[float(np.var(data[i][1])) for i in (0,1,1)];out={}
    if pred.shape!=(256,3) or not np.isfinite(pred).all():raise ValueError('prediction shape/finiteness')
    for j,key in enumerate(ENDPOINTS):
        v=variances[j];error=sel.mse(pred[:,j],targets[:,j]);normalized=error/max(v,1e-12)
        if not np.isfinite([v,error,normalized]).all():raise ValueError('nonfinite metric')
        out[key]={'mse':error,'nmse':normalized,'training_variance':v,'floor_active':v<1e-12}
    return out


def run(output,checkpoint,mode):
    seeds=SEEDS if mode=='pilot' else (123456,);all_cells=[]
    for seed in seeds:
        parent=output/str(seed);parent.mkdir();spec=specification(seed,mode=='fixture')
        write(parent/'private_truth.json',spec)
        write(parent/'prehistory.reserved.json',{'responses':32,'at_unix':time.time(),'kind':'training'})
        pre_rows=history(spec,'null',seed,1);np.savez(parent/'prehistory.npz',train=arrays(pre_rows)[0],clamp=arrays(pre_rows)[1])
        write(parent/'prehistory.returned.json',{'responses':32,'at_unix':time.time(),'sha256':sha(parent/'prehistory.npz')})
        pre,_=sel.split_history(pre_rows);preheads,prefit=fit_or_failure(pre,'grammar',checkpoint)
        write(parent/'prechange_fit.json',prefit)
        for vi,variant in enumerate(VARIANTS):
            folder=parent/variant;folder.mkdir()
            write(folder/'training.reserved.json',{'responses':32,'at_unix':time.time(),'kind':'training'})
            rows=history(spec,variant,seed,10+vi)
            train,clamp=arrays(rows);np.savez(folder/'training.npz',train=train,clamp=clamp)
            write(folder/'training.returned.json',{'responses':32,'at_unix':time.time(),'sha256':sha(folder/'training.npz')})
            for method in METHODS:write(folder/(method+'.started.json'),{'seed':seed,'variant':variant,'method':method,'at_unix':time.time(),'status':'preparing_shared_experts'})
            start=time.monotonic();cpu=time.process_time()
            heads,choices,errors,events=make_predictors(rows,preheads,checkpoint)
            write(folder/'selection_seal.json',{'at_unix':time.time(),'training_sha256':sha(folder/'training.npz'),'fit_indices':sel.FIT_INDICES,'calibration_indices':sel.CALIBRATION_INDICES,'choices':choices,'errors':errors,'events':events,'evaluation_generated':False})
            write(folder/'evaluation.reserved.json',{'responses':768,'at_unix':time.time(),'kind':'private','selection_seal_sha256':sha(folder/'selection_seal.json')})
            probes=private_probes(spec,variant,seed)
            np.savez(folder/'private_probes.npz',composed=probes[0],local_m=probes[1],local_y=probes[2])
            write(folder/'evaluation.returned.json',{'at_unix':time.time(),'selection_seal_sha256':sha(folder/'selection_seal.json'),'private_responses':768,'sha256':sha(folder/'private_probes.npz')})
            for method in METHODS:
                row={'seed':seed,'variant':variant,'method':method,'status':'failed'}
                before=time.monotonic();pcpu=time.process_time()
                try:
                    if method in errors:raise RuntimeError(errors[method])
                    pred=predictions(method,heads,choices,probes);row['metrics']=metrics(pred,probes,rows)
                    np.save(folder/(method+'_predictions.npy'),pred);row['prediction_sha256']=sha(folder/(method+'_predictions.npy'))
                    row['status']='complete'
                except Exception:row['error']=traceback.format_exc()
                row['evaluation_elapsed_s']=time.monotonic()-before;row['evaluation_process_cpu_s']=time.process_time()-pcpu
                write(folder/(method+'.json'),row);all_cells.append(row)
            write(folder/'resources.json',{'elapsed_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu,'scope':'variant fitting,selection,private probe generation,evaluation,writing; overlaps events and cell timings','heads':{k:[h.record() for h in v] if v is not None else None for k,v in heads.items()},'prechange_prediction_counters_cumulative':True,'shared_experts_reused_by_mixtures':True})
            print(seed,variant,sum(r['status']=='complete' for r in all_cells[-6:]),flush=True)
    write(output/'complete.json',{'mode':mode,'cells':all_cells,'planned_cells':len(seeds)*24,'training_responses_total':len(seeds)*160,'private_responses_total':len(seeds)*3072,'scope':'fixed-history development, no acquisition or language arm','completed_cells':sum(r['status']=='complete' for r in all_cells)})


def main():
    p=argparse.ArgumentParser()
    for key in ('output','freeze'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--mode',choices=('pilot','fixture'),required=True);a=p.parse_args()
    if os.environ.get('ACE_MISMATCH_SUPERVISED')!='1' or not (a.output/'plan.json').exists():raise ValueError('supervisor required')
    raw=a.freeze.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.freeze_sha256:raise ValueError('freeze pin')
    f=json.loads(raw)
    if f['schema']!='ace-mismatch-freeze-v1' or f['mode']!=a.mode:raise ValueError('stage mismatch')
    for rel,pin in f['sources'].items():
        if sha(ROOT/rel)!=pin:raise ValueError('source drift: '+rel)
    actual=initialize(f)
    write(a.output/'preflight.json',{'actual_dependencies':actual,'checkpoint':f['checkpoint'],'actual_checkpoint_sha256':sha(f['checkpoint']),'python':sys.version,'at_unix':time.time(),'freeze_sha256':a.freeze_sha256})
    run(a.output,Path(f['checkpoint']),a.mode)

if __name__=='__main__':main()
