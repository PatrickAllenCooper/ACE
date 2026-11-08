#!/usr/bin/env python3
"""Fresh bounded development screen. Authentication precedes numerical imports."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

SEEDS=tuple(range(93000,93006))
VARIANTS=('null','coefficient_M','missing_M','missing_Y')
METHODS=('grammar32','rbf24','pfn24','prechange24','raw','local','interval','combined','combined_no_pfn')
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
    global np,base,sel,ret,flex
    required={'numpy','torch','transformers','tabpfn','scikit-learn','scipy','safetensors','tokenizers','huggingface-hub'}
    if set(freeze['dependencies'])!=required:raise ValueError('dependency closure')
    actual={n:importlib.metadata.version(n) for n in required}
    if actual!=freeze['dependencies']:raise ValueError('runtime versions mismatch')
    if sys.version!=freeze['python_version'] or str(Path(sys.executable).absolute())!=freeze['python']:
        raise ValueError('interpreter mismatch')
    if sha(freeze['checkpoint'])!=freeze['checkpoint_sha256'] or freeze['checkpoint_sha256']!='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736':
        raise ValueError('checkpoint mismatch')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    base=load_source('ace_retention_base',ROOT/'scripts/research/foundation_component_pilot.py',freeze['sources']['scripts/research/foundation_component_pilot.py'])
    sel=load_source('ace_retention_split',ROOT/'scripts/research/foundation_mixture_selection.py',freeze['sources']['scripts/research/foundation_mixture_selection.py'])
    ret=load_source('ace_retention_selection',ROOT/'scripts/research/foundation_retention_selection.py',freeze['sources']['scripts/research/foundation_retention_selection.py'])
    flex=load_source('ace_retention_flexible',ROOT/'scripts/research/foundation_flexible_control.py',freeze['sources']['scripts/research/foundation_flexible_control.py'])
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    return actual


def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))


def specification(seed,artificial=False):
    if artificial:
        if seed!=223456:raise ValueError('artificial identity')
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
    """Measured-parent learner interface; finite repeat/partition checks before use."""
    def __init__(self,method,x,y,checkpoint):
        self.fit_count=1;self.prediction_calls=0;self.prediction_rows=0
        self.method=method
        self.model=(flex.RBFControl().fit(x[:,0].copy(),y.copy()) if method=='rbf'
                    else base.make_model(method,checkpoint).fit(x.copy(),y.copy()))
        self.fit_rows=len(y)
        parents=x[:,0]
        probe=(float(min(parents)-1),float(min(parents)),float(np.median(parents)),float(max(parents)),float(max(parents)+1))
        first=np.asarray(self(probe));again=np.asarray(self(probe))
        partition=np.concatenate((self(probe[:2]),self(probe[2:])))
        if not np.array_equal(first,again):raise ValueError('head repeat determinism failed')
        delta=float(np.max(np.abs(first-partition)))
        if not np.allclose(first,partition,rtol=1e-6,atol=1e-6):raise ValueError('head batch partition check failed')
        self.pointwise_check={'parents':probe,'repeat_exact':True,'partition_max_abs_difference':delta,
            'rtol':1e-6,'atol':1e-6,'scope':'five fit-derived inputs only; finite numerical check, not universal pointwise proof'}
    def __call__(self,x):
        x=tuple(x);self.prediction_calls+=1;self.prediction_rows+=len(x)
        values=(self.model(x) if self.method=='rbf' else self.model.predict(np.asarray(x).reshape(-1,1)))
        return np.asarray(sel.finite(np.asarray(values).reshape(-1)))
    def record(self):
        return {'fit_count':self.fit_count,'fit_rows':self.fit_rows,'prediction_calls':self.prediction_calls,
            'prediction_rows':self.prediction_rows,'family':getattr(self.model,'family',None),
            'coefficients':getattr(self.model,'coef',np.array([])).tolist(),'pointwise_check':self.pointwise_check,
            'rbf_normalization':({'parent_mean':self.model.center_,'parent_std':self.model.scale_,
                'target_mean':self.model.target_center_,'gamma':1.,'alpha':.01} if self.method=='rbf' else None)}


def fit_or_failure(rows,method,checkpoint):
    start=time.monotonic();cpu=time.process_time();records=[];heads=[];error=None
    train,clamps=arrays(rows)
    for node,(x,y) in zip(('M','Y'),base.eligible(train,clamps)):
        record={'node':node,'fit_rows':len(y),'status':'started'};records.append(record)
        try:
            h=Head(method,x,y,checkpoint);heads.append(h);record.update(status='complete',head=h.record())
        except Exception:
            error=traceback.format_exc();record.update(status='failed',error=error)
            if node=='M':records.append({'node':'Y','status':'unattempted'})
            break
    return (tuple(heads) if error is None else None),{'status':'complete' if error is None else 'failed',
        'error':error,'nodes':records,'elapsed_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu}


def candidate_heads(heads,include_pfn=True):
    names={'retained':'prechange24','grammar':'grammar24','rbf':'rbf24'}
    if include_pfn:names['pfn']='pfn24'
    if any(heads[k] is None for k in names.values()):raise RuntimeError('required expert fit failed')
    return tuple({name:heads[key][i] for name,key in names.items()} for i in range(2))


def make_predictors(rows,pre,checkpoint):
    """Learner receives returned training rows only, never variant/truth/private probes."""
    fit,calibration=sel.split_history(rows);heads={};events={};choices={};errors={}
    for name,method,data in [('grammar32','grammar',rows),('grammar24','grammar',fit),
                             ('rbf24','rbf',fit),('pfn24','tabpfn_v2',fit)]:
        heads[name],events[name]=fit_or_failure(data,method,checkpoint)
        if heads[name] is None:errors[name]=events[name]['error']
    heads['prechange24']=pre
    if pre is None:errors['prechange24']='pre-change fit failed; see prechange_fit.json'
    for name in METHODS[4:]:
        start=time.monotonic();cpu=time.process_time()
        try:
            include=name!='combined_no_pfn'
            choice=ret.select(fit,calibration,candidate_heads(heads,include),
                              mode=name if include else 'combined',include_pfn=include)
            choices[name]=choice
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
    if name in METHODS[4:]:
        mhead,yhead=ret.selected_heads(choices[name],candidate_heads(heads,name!='combined_no_pfn'))
    else:mhead,yhead=heads[name]
    m=sel.predict(mhead,xm);y=sel.predict(yhead,ym);parents=sel.predict(mhead,x)
    composed=sel.predict(yhead,parents)
    return np.column_stack((m,y,composed)),np.asarray(parents).reshape(-1,1)


def diagnostics(pred,parents,probes,rows,name):
    comp,lm,ly=probes;fit,cal=sel.split_history(rows)
    intervals,_,_=ret.training_layout(fit,cal);local={}
    for i,(key,coords,targets) in enumerate((('M_local',lm[:,0],lm[:,1]),('Y_local',ly[:,1],ly[:,2]))):
        interval=intervals[i];inside=(coords>=interval.lower)&(coords<=interval.upper);parts={}
        for label,mask in (('inside',inside),('outside',~inside)):
            count=int(mask.sum())
            parts[label]={'count':count,'mse':sel.mse(pred[mask,i],targets[mask]) if count else None}
        local[key]=parts
    interval=intervals[1];inside=(parents[:,0]>=interval.lower)&(parents[:,0]<=interval.upper)
    return {'intervals':[asdict(i) for i in intervals],'local':local,
        'composed_Y_parent_inside':int(inside.sum()),'composed_Y_parent_outside':int((~inside).sum()),
        'composed_Y_parent_inside_rate':float(inside.mean()),
        'interval_gating_enabled':name in ('interval','combined','combined_no_pfn')}


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
    seeds=SEEDS if mode=='pilot' else (223456,);all_cells=[]
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
            write(folder/'selection_seal.json',{'at_unix':time.time(),'training_sha256':sha(folder/'training.npz'),'fit_indices':sel.FIT_INDICES,'calibration_indices':sel.CALIBRATION_INDICES,'choices':{k:asdict(v) for k,v in choices.items()},'errors':errors,'events':events,'evaluation_generated':False})
            write(folder/'evaluation.reserved.json',{'responses':768,'at_unix':time.time(),'kind':'private','selection_seal_sha256':sha(folder/'selection_seal.json')})
            probes=private_probes(spec,variant,seed)
            np.savez(folder/'private_probes.npz',composed=probes[0],local_m=probes[1],local_y=probes[2])
            write(folder/'evaluation.returned.json',{'at_unix':time.time(),'selection_seal_sha256':sha(folder/'selection_seal.json'),'private_responses':768,'sha256':sha(folder/'private_probes.npz')})
            for method in METHODS:
                row={'seed':seed,'variant':variant,'method':method,'status':'failed'}
                before=time.monotonic();pcpu=time.process_time()
                try:
                    if method in errors:raise RuntimeError(errors[method])
                    pred,parents=predictions(method,heads,choices,probes);row['metrics']=metrics(pred,probes,rows)
                    row['diagnostics']=diagnostics(pred,parents,probes,rows,method)
                    np.save(folder/(method+'_parents.npy'),parents);row['parent_sha256']=sha(folder/(method+'_parents.npy'))
                    np.save(folder/(method+'_predictions.npy'),pred);row['prediction_sha256']=sha(folder/(method+'_predictions.npy'))
                    row['status']='complete'
                except Exception:row['error']=traceback.format_exc()
                row['evaluation_elapsed_s']=time.monotonic()-before;row['evaluation_process_cpu_s']=time.process_time()-pcpu
                write(folder/(method+'.json'),row);all_cells.append(row)
            write(folder/'resources.json',{'elapsed_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu,'scope':'variant fitting,selection,private probe generation,evaluation,writing; overlaps events and cell timings','heads':{k:[h.record() for h in v] if v is not None else None for k,v in heads.items()},'prechange_prediction_counters_cumulative':True,'shared_experts_reused_by_selectors':True})
            print(seed,variant,sum(r['status']=='complete' for r in all_cells[-9:]),flush=True)
    write(output/'complete.json',{'mode':mode,'cells':all_cells,'planned_cells':len(seeds)*36,'training_responses_total':len(seeds)*160,'private_responses_total':len(seeds)*3072,'scope':'fixed-history development, no acquisition or language arm','completed_cells':sum(r['status']=='complete' for r in all_cells)})


def main():
    p=argparse.ArgumentParser()
    for key in ('output','freeze'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--mode',choices=('pilot','fixture'),required=True);a=p.parse_args()
    if os.environ.get('ACE_RETENTION_SUPERVISED')!='1' or not (a.output/'plan.json').exists():raise ValueError('supervisor required')
    raw=a.freeze.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.freeze_sha256:raise ValueError('freeze pin')
    f=json.loads(raw)
    if f['schema']!='ace-retention-freeze-v1' or f['mode']!=a.mode:raise ValueError('stage mismatch')
    for rel,pin in f['sources'].items():
        if sha(ROOT/rel)!=pin:raise ValueError('source drift: '+rel)
    actual=initialize(f)
    write(a.output/'preflight.json',{'actual_dependencies':actual,'checkpoint':f['checkpoint'],'actual_checkpoint_sha256':sha(f['checkpoint']),'python':sys.version,'at_unix':time.time(),'freeze_sha256':a.freeze_sha256})
    run(a.output,Path(f['checkpoint']),a.mode)

if __name__=='__main__':main()
