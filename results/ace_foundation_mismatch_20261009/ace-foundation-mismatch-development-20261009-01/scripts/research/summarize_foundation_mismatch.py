#!/usr/bin/env python3
"""Authenticate supervised outcome closure, reconstruct errors and preserve every pair."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

VARIANTS=('null','coefficient_M','missing_M','missing_Y')
METHODS=('grammar32','grammar24','pfn24','terminal24','mechanism24','prechange24')
ENDPOINTS=('M_local','Y_local','Y_composed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured(path,pin):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('pin mismatch '+str(path))
    return json.loads(raw)


def aggregate(values):
    if not values or any(v is None or not math.isfinite(v) or v<=0 for v in values):
        return {'arithmetic_mean':None,'geometric_mean':None,'defined':False}
    return {'arithmetic_mean':math.fsum(values)/len(values),'geometric_mean':math.exp(math.fsum(math.log(v) for v in values)/len(values)),'defined':True}


def summarize(cells,seeds):
    expected=list(itertools.product(seeds,VARIANTS,METHODS))
    keys=[(r['seed'],r['variant'],r['method']) for r in cells]
    if keys!=expected:raise ValueError('full ordered matrix required; no duplicate/replacement/omission')
    mapping=dict(zip(keys,cells));comparisons=[];harm=[]
    for variant,method,endpoint in itertools.product(VARIANTS,METHODS[1:],ENDPOINTS):
        pairs=[]
        for seed in seeds:
            g=mapping[seed,variant,'grammar32'];m=mapping[seed,variant,method];value=None
            if g['status']=='complete' and m['status']=='complete':
                a=m['metrics'][endpoint]['nmse'];b=g['metrics'][endpoint]['nmse']
                if math.isfinite(a) and math.isfinite(b) and a>0 and b>0:value=a/b
                if value is not None and not math.isfinite(value):value=None
            pairs.append({'seed':seed,'ratio':value,'candidate_status':m['status'],'reference_status':g['status']})
        comparisons.append({'variant':variant,'method':method,'endpoint':endpoint,'pairs':pairs,**aggregate([p['ratio'] for p in pairs])})
    for seed,variant,method,endpoint in itertools.product(seeds,VARIANTS,METHODS[:-1],ENDPOINTS[:2]):
        pre=mapping[seed,variant,'prechange24'];adapt=mapping[seed,variant,method];delta=norm=None
        if pre['status']=='complete' and adapt['status']=='complete':
            a=adapt['metrics'][endpoint];b=pre['metrics'][endpoint]
            if a['training_variance']!=b['training_variance']:raise ValueError('different normalization phases')
            delta=a['mse']-b['mse'];norm=delta/max(a['training_variance'],1e-12)
        unchanged=(variant=='null' or (endpoint=='Y_local' and variant in ('coefficient_M','missing_M')) or (endpoint=='M_local' and variant=='missing_Y'))
        harm.append({'seed':seed,'variant':variant,'method':method,'endpoint':endpoint,'mechanism_unchanged':unchanged,'mse_difference':delta,'normalized_difference':norm})
    return {'cells':cells,'comparisons':comparisons,'local_harm':harm,'interpretation':'development pipelines; positive local difference means harm; no acquisition-efficiency or equivalence inference'}


def validate_choices(seal):
    for method,n in (('terminal24',5),('mechanism24',25)):
        if method in seal['errors']:continue
        c=seal['choices'][method];grid=([ [w] for w in (0,.25,.5,.75,1)] if n==5 else [list(w) for w in itertools.product((0,.25,.5,.75,1),repeat=2)])
        if [p[0] for p in c['candidates']]!=grid or any(not math.isfinite(p[1]) or p[1]<0 for p in c['candidates']):raise ValueError('candidate grid/scores')
        best=min(c['candidates'],key=lambda p:p[1])
        if c['weights']!=best[0] or c['mse']!=best[1]:raise ValueError('selected weight/tie rule')


def verify(root,freeze,terminal):
    import numpy as np
    mode=freeze['mode'];seeds=tuple(range(92000,92006)) if mode=='pilot' else (123456,)
    if terminal['mode']!=mode or terminal['status'] not in ('complete','failed'):raise ValueError('supervisor mode/status')
    if terminal['status']=='failed':return failure_report(root,freeze,terminal)
    complete=json.loads((root/'complete.json').read_bytes())
    if complete['cells']!=terminal['cells'] or complete['planned_cells']!=len(seeds)*24:raise ValueError('supervisor/completion mismatch')
    cells=complete['cells'];summary=summarize(cells,seeds)
    reservations=returned=0
    for seed in seeds:
        parent=root/str(seed)
        r=json.loads((parent/'prehistory.reserved.json').read_bytes());q=json.loads((parent/'prehistory.returned.json').read_bytes())
        if r['responses']!=32 or q['responses']!=32 or q['sha256']!=sha(parent/'prehistory.npz') or q['at_unix']<r['at_unix']:raise ValueError('prehistory accounting')
        reservations+=32;returned+=32
        for variant in VARIANTS:
            d=parent/variant;seal=json.loads((d/'selection_seal.json').read_bytes())
            reservation=json.loads((d/'evaluation.reserved.json').read_bytes());done=json.loads((d/'evaluation.returned.json').read_bytes())
            tr=json.loads((d/'training.reserved.json').read_bytes());td=json.loads((d/'training.returned.json').read_bytes())
            if tr['responses']!=32 or td['responses']!=32 or td['sha256']!=sha(d/'training.npz') or td['at_unix']<tr['at_unix']:raise ValueError('training accounting')
            if seal['training_sha256']!=td['sha256'] or seal['at_unix']<td['at_unix'] or seal['evaluation_generated'] is not False:raise ValueError('selection order')
            if reservation['responses']!=768 or done['private_responses']!=768 or reservation['at_unix']<seal['at_unix'] or done['at_unix']<reservation['at_unix']:raise ValueError('evaluation accounting/order')
            if done['sha256']!=sha(d/'private_probes.npz') or reservation['selection_seal_sha256']!=sha(d/'selection_seal.json') or done['selection_seal_sha256']!=sha(d/'selection_seal.json'):raise ValueError('selection binding')
            reservations+=800;returned+=800
            with np.load(d/'training.npz',allow_pickle=False) as a:train=a['train'];clamp=a['clamp']
            if train.shape!=(32,3) or not np.isfinite(train).all() or tuple(clamp)!=('none',)*8+('X',)*12+('M',)*12:raise ValueError('training layout')
            with np.load(d/'private_probes.npz',allow_pickle=False) as a:comp=a['composed'];lm=a['local_m'];ly=a['local_y']
            if any(x.shape!=(256,3) or not np.isfinite(x).all() for x in (comp,lm,ly)):raise ValueError('private shape')
            targets=np.column_stack((lm[:,1],ly[:,2],comp[:,2]));variances=[float(np.var(train[clamp!='M',1])),float(np.var(train[:,2])),float(np.var(train[:,2]))]
            validate_choices(seal)
            for method in METHODS:
                row=json.loads((d/(method+'.json')).read_bytes())
                ref=next(r for r in cells if (r['seed'],r['variant'],r['method'])==(seed,variant,method))
                if row!=ref:raise ValueError('cell closure mismatch')
                if row['status']!='complete':continue
                if sha(d/(method+'_predictions.npy'))!=row['prediction_sha256']:raise ValueError('prediction pin')
                pred=np.load(d/(method+'_predictions.npy'),allow_pickle=False)
                if pred.shape!=(256,3) or not np.isfinite(pred).all():raise ValueError('prediction shape/finiteness')
                for i,key in enumerate(ENDPOINTS):
                    mse=float(np.mean((pred[:,i]-targets[:,i])**2));v=variances[i];m=row['metrics'][key]
                    if m['training_variance']!=v or m['floor_active']!=(v<1e-12):raise ValueError('normalization mismatch')
                    if not math.isfinite(mse) or not math.isclose(m['mse'],mse,rel_tol=1e-10,abs_tol=1e-12) or not math.isclose(m['nmse'],mse/max(v,1e-12),rel_tol=1e-10,abs_tol=1e-12):raise ValueError('metric mismatch')
    if complete['training_responses_total']!=len(seeds)*160 or complete['private_responses_total']!=len(seeds)*3072 or returned!=len(seeds)*3232:raise ValueError('response total')
    summary['attempt_status']='complete'
    summary['response_accounting']={'training':len(seeds)*160,'private':len(seeds)*3072,'reserved':reservations,'returned':returned}
    summary['selection']={str(seed)+':'+v:json.loads((root/str(seed)/v/'selection_seal.json').read_bytes())['choices'] for seed in seeds for v in VARIANTS}
    summary['supervised_resources']={k:terminal[k] for k in ('child_cpu_s','supervisor_process_cpu_s','elapsed_s','peak_child_rss_bytes','gpu_seconds')}
    return summary


def failure_report(root,freeze,terminal):
    """No success inference: authenticate available evidence, retain every planned cell."""
    import numpy as np
    mode=freeze['mode'];seeds=tuple(range(92000,92006)) if mode=='pilot' else (123456,)
    planned=list(itertools.product(seeds,VARIANTS,METHODS));raw=terminal['cells'];issues=[]
    mapping={};cells=[];selections={}
    for row in raw:
        key=tuple(row.get(k) for k in ('seed','variant','method'))
        if key not in planned or key in mapping:
            issues.append({'kind':'invalid_terminal_identity','record':row});continue
        mapping[key]=row
    accounting={k:{'planned':len(seeds)*(160 if k=='training' else 3072),'reserved':0,'validated_returned':0,'reserved_unknown_returns':0,'unreserved_planned':0,'invalid_blocks':[]} for k in ('training','private')}
    def block(directory,stem,category,count,array):
        dst=accounting[category];rp=directory/(stem+'.reserved.json');qp=directory/(stem+'.returned.json')
        if not rp.exists():
            dst['unreserved_planned']+=count
            if qp.exists():dst['invalid_blocks'].append({'path':str(qp),'reason':'return without reservation'})
            return False
        try:
            r=json.loads(rp.read_bytes())
            if r['responses']!=count or r['kind']!=category or not math.isfinite(r['at_unix']):raise ValueError('invalid reservation')
        except Exception as exc:
            dst['invalid_blocks'].append({'path':str(rp),'reason':str(exc)});return False
        dst['reserved']+=count
        try:
            q=json.loads(qp.read_bytes());key='private_responses' if category=='private' else 'responses'
            if q[key]!=count or not math.isfinite(q['at_unix']) or q['at_unix']<r['at_unix'] or q['sha256']!=sha(directory/array):raise ValueError('invalid return')
            if category=='private':
                seal=json.loads((directory/'selection_seal.json').read_bytes())
                if r['selection_seal_sha256']!=sha(directory/'selection_seal.json') or q['selection_seal_sha256']!=sha(directory/'selection_seal.json') or r['at_unix']<seal['at_unix'] or seal['evaluation_generated'] is not False:raise ValueError('unsealed evaluation')
            dst['validated_returned']+=count;return True
        except Exception as exc:
            dst['reserved_unknown_returns']+=count
            if qp.exists():dst['invalid_blocks'].append({'path':str(qp),'reason':str(exc)})
            return False
    for seed in seeds:
        parent=root/str(seed);block(parent,'prehistory','training',32,'prehistory.npz')
        for variant in VARIANTS:
            d=parent/variant;train_ok=block(d,'training','training',32,'training.npz');eval_ok=block(d,'evaluation','private',768,'private_probes.npz')
            seal=None
            try:
                if (d/'selection_seal.json').exists():
                    seal=json.loads((d/'selection_seal.json').read_bytes())
                    if seal['training_sha256']!=sha(d/'training.npz'):raise ValueError('training binding')
                    validate_choices(seal)
                    selections[str(seed)+':'+variant]=seal['choices']
            except Exception as exc:
                issues.append({'kind':'invalid_selection','seed':seed,'variant':variant,'reason':str(exc)});seal=None
            for method in METHODS:
                key=(seed,variant,method);rawrow=mapping.get(key)
                row=dict(rawrow) if rawrow else {'seed':seed,'variant':variant,'method':method,'status':'invalid_record','error':'missing authenticated terminal identity'}
                if row.get('status') not in ('complete','failed','unattempted','interrupted','invalid_record'):
                    row['status']='invalid_record';row['error']='invalid terminal disposition'
                if row['status']=='complete':
                    try:
                        if not train_ok or not eval_ok or seal is None:raise ValueError('unqualified response or selection closure')
                        td=json.loads((d/'training.returned.json').read_bytes())
                        if seal['at_unix']<td['at_unix']:raise ValueError('selection precedes training')
                        if json.loads((d/(method+'.json')).read_bytes())!=rawrow:raise ValueError('cell closure')
                        with np.load(d/'training.npz',allow_pickle=False) as a:train=a['train'];clamp=a['clamp']
                        if train.shape!=(32,3) or not np.isfinite(train).all() or tuple(clamp)!=('none',)*8+('X',)*12+('M',)*12:raise ValueError('training layout')
                        with np.load(d/'private_probes.npz',allow_pickle=False) as a:comp=a['composed'];lm=a['local_m'];ly=a['local_y']
                        if any(a.shape!=(256,3) or not np.isfinite(a).all() for a in (comp,lm,ly)):raise ValueError('probe layout')
                        if sha(d/(method+'_predictions.npy'))!=row['prediction_sha256']:raise ValueError('prediction pin')
                        pred=np.load(d/(method+'_predictions.npy'),allow_pickle=False)
                        if pred.shape!=(256,3) or not np.isfinite(pred).all():raise ValueError('prediction layout')
                        targets=np.column_stack((lm[:,1],ly[:,2],comp[:,2]));vs=[float(np.var(train[clamp!='M',1])),float(np.var(train[:,2])),float(np.var(train[:,2]))]
                        for j,e in enumerate(ENDPOINTS):
                            v=vs[j];m=row['metrics'][e];mse=float(np.mean((pred[:,j]-targets[:,j])**2))
                            if not math.isfinite(mse) or m['training_variance']!=v or m['floor_active']!=(v<1e-12) or not math.isclose(m['mse'],mse,rel_tol=1e-10,abs_tol=1e-12) or not math.isclose(m['nmse'],mse/max(v,1e-12),rel_tol=1e-10,abs_tol=1e-12):raise ValueError('metric mismatch')
                    except Exception as exc:
                        row['status']='invalid_record';row['verification_error']=str(exc)
                        issues.append({'kind':'invalid_completed_cell','seed':seed,'variant':variant,'method':method,'reason':str(exc)})
                cells.append(row)
    summary=summarize(cells,seeds)
    summary.update(attempt_status='failed',failure_reason=terminal.get('reason'),retry_authorized=False,raw_terminal_cells=raw,verification_issues=issues,response_accounting=accounting,selection=selections)
    summary['supervised_resources']={k:terminal[k] for k in ('child_cpu_s','supervisor_process_cpu_s','elapsed_s','peak_child_rss_bytes','gpu_seconds')}
    return summary


def main():
    p=argparse.ArgumentParser()
    for k in ('root','freeze','terminal','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('freeze-sha256','terminal-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();f=captured(a.freeze,a.freeze_sha256);t=captured(a.terminal,a.terminal_sha256)
    if t['freeze_sha256']!=a.freeze_sha256 or sha(__file__)!=f['sources']['scripts/research/summarize_foundation_mismatch.py']:raise ValueError('source/receipt mismatch')
    summary=verify(a.root,f,t);summary['lineage']={'freeze_sha256':a.freeze_sha256,'terminal_sha256':a.terminal_sha256,'reporter_sha256':sha(__file__)}
    with a.output.open('x') as out:json.dump(summary,out,indent=2,allow_nan=False);out.write('\n')

if __name__=='__main__':main()
