"""Outcome-independent Stage B world/action preparation, gated on attribution.

The five-node base has no coefficient seed. Its explicit parameterized extension
is labeled as such; sampling histories from the fixed base is not new worlds.
The thirty-node adapter reuses the audited generator with fixed graph/edge RNG.
Primary candidate estimand is the zero-noise structural map, not an expected
outcome obtained by silently composing conditional means.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path


def seed_for(label):
    return int.from_bytes(hashlib.sha256(('delivery-stage-B-v1:'+label).encode()).digest()[:4],'big')%2**31


def world_spec(size,seed,project):
    import numpy as np
    if size==5:
        rng=np.random.default_rng(seed)
        return {'size':5,'seed':seed,'family':'explicit parameterized extension of audited legacy five-node equations',
            'order':['X1','X4','X2','X3','X5'],
            'parents':{'X1':[],'X4':[],'X2':['X1'],'X3':['X1','X2'],'X5':['X4']},
            'coefficients':{'a':float(2*rng.uniform(.8,1.2)),'b':float(rng.uniform(.8,1.2)),
                'c':float(.5*rng.uniform(.8,1.2)),'d':float(rng.uniform(.8,1.2)),
                's':float(rng.uniform(.8,1.2)),'q':float(.2*rng.uniform(.8,1.2))},
            'noise_sd':.1,'source':str(Path(project)/'baselines.py')}
    if size!=30:raise ValueError('unregistered graph size')
    source=Path(project)/'experiments/large_scale_scm.py'
    spec=importlib.util.spec_from_file_location('delivery_audited_large_scm',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=np.random.get_state()
    try:
        np.random.seed(seed);world=module.LargeScaleSCM(n_nodes=30,coeff_seed=seed)
    finally:np.random.set_state(state)
    return {'size':30,'seed':seed,'family':'audited LargeScaleSCM, graph and coefficients pinned',
            'order':world.nodes,'parents':world.graph,'coefficients':world.coeffs,
            'nonlinear_nodes':[n for n in world.nodes if world.node_idx[n]%5==0],
            'noise_sd':world.noise_std,'source':str(source)}


def structural_values(spec,clamps,noise=None):
    """Deterministic structural evaluation; noise is supplied, never drawn here."""
    import math
    noise=noise or {};values={};coef=spec['coefficients']
    for n in spec['order']:
        if n in clamps:values[n]=float(clamps[n]);continue
        pa=spec['parents'][n]
        if not pa:raise ValueError('all root actions must be explicitly clamped')
        if spec['size']==5:
            if n=='X2':v=coef['a']*values['X1']+coef['b']
            elif n=='X3':v=coef['c']*values['X1']-coef['d']*values['X2']+coef['s']*math.sin(values['X2'])
            else:v=coef['q']*values['X4']**2
        else:
            v=sum(coef[n][p]*values[p] for p in pa)
            if n in spec['nonlinear_nodes']:v+=.2*math.sin(v)
        values[n]=v+noise.get(n,0.)
    return values


def validate_world(spec):
    """Graph/mechanism qualification without evaluating any planned world."""
    import math
    order=spec['order'];parents=spec['parents'];seen=set()
    if spec['size'] not in (5,30) or len(order)!=spec['size'] or len(set(order))!=len(order) or set(parents)!=set(order):
        raise ValueError('unregistered or incomplete world')
    for node in order:
        pa=parents[node]
        if len(set(pa))!=len(pa) or not set(pa)<=seen:raise ValueError('non-topological or duplicate parent')
        seen.add(node)
    roots=[node for node in order if not parents[node]]
    if len(roots)!=(2 if spec['size']==5 else 5):raise ValueError('root count changed')
    if spec['size']==5:
        if parents!={'X1':[],'X4':[],'X2':['X1'],'X3':['X1','X2'],'X5':['X4']}:
            raise ValueError('five-node family graph changed')
        coef=spec['coefficients']
        if set(coef)!={'a','b','c','d','s','q'} or any(not math.isfinite(v) or v<=0 for v in coef.values()):
            raise ValueError('invalid five-node mechanism')
    else:
        coef=spec['coefficients']
        if set(coef)!=set(order):raise ValueError('incomplete edge coefficients')
        for node in order:
            if set(coef[node])!=set(parents[node]) or any(not math.isfinite(v) or not .3<=v<=.7 for v in coef[node].values()):
                raise ValueError('invalid frozen edge coefficients')
        if spec['nonlinear_nodes']!=[node for i,node in enumerate(order,1) if i%5==0]:
            raise ValueError('nonlinear rule changed')
    return roots


def draft_manifest(project,out,count=400):
    """Prepare immutable proposed actions/coefficients; zero response collection.

    This is NOT release or a final prospective freeze. Recipe, online-update
    semantics, endpoint/normalizer and measured full-matrix CPU cost are gates.
    No world replacement when a descriptor/action check fails.
    """
    from runner_delivery_confirmation import sha,write,utc
    project,out=Path(project),Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest={'at':utc(),'stage':'B outcome-independent proposed manifest; not released',
        'adapter_sha256':sha(__file__),'source_hashes':{str(p.relative_to(project)):sha(p)
            for p in (project/'baselines.py',project/'experiments/large_scale_scm.py')},
        'structural_responses_evaluated':0,'proposed_rows_per_history':count,
        'candidate_estimand':'zero-noise structural map; not interventional expectation',
        'worlds':{},'unresolved':['Stage A recipe/control/ablation selection','online update/admission mapping',
            'primary nodes and training-only normalizer','fit/evaluation workers and resource projection',
            'final source/dependencies/manifest freeze before response collection']}
    seeds=[]
    for size in (5,30):
        for i in range(20):
            label=f'{size}:{i:02d}';seed=seed_for('world:'+label);seeds.append(seed)
            spec=world_spec(size,seed,project);roots=validate_world(spec)
            actions={strategy:shared_actions(spec,strategy,seed_for(strategy+':'+label),count)
                     for strategy in ('balanced','random','evaluation')}
            blocks={strategy:{action_block([a[root] for root in roots]) for a in rows}
                    for strategy,rows in actions.items()}
            if blocks['evaluation']&(blocks['random']|blocks['balanced']):raise ValueError('shared heldout action block')
            dest=out/label.replace(':','-');dest.mkdir()
            write(dest/'world.json',spec);write(dest/'actions.json',actions)
            manifest['worlds'][label]={'seed':seed,'graph_size':size,'world_sha256':sha(dest/'world.json'),
                'actions_sha256':sha(dest/'actions.json'),'roots':roots,
                'blocks_by_strategy':{k:len(v) for k,v in blocks.items()},
                'rows_by_strategy':{k:len(v) for k,v in actions.items()}}
    if len(set(seeds))!=40:raise ValueError('world seed collision; no silent replacement')
    write(out/'manifest.json',manifest)
    return manifest


def action_block(values):
    import math
    if any(not -3<=v<=3 for v in values):raise ValueError('root action outside frozen support')
    return tuple(min(4,int(math.floor((v+3)/1.2))) for v in values)


def reserved(block):
    # The same block can NEVER occur in collection and evaluation.
    h=hashlib.sha256(json.dumps(block,separators=(',',':')).encode()).digest()
    return int.from_bytes(h[:4],'big')%5==0


def shared_actions(spec,strategy,seed,count=400):
    import numpy as np
    rng=np.random.default_rng(seed);roots=[n for n in spec['order'] if not spec['parents'][n]]
    actions=[];rejected=0;counts=np.zeros((len(roots),11),dtype=int)
    while len(actions)<count:
        if strategy=='balanced':
            # Prefer least-used levels at every root; joint-block exclusions
            # may make the strictly minimum combination unavailable. Relax
            # one quota level only after bounded attempts, and audit balance.
            candidates=[]
            for trial in range(64):
                indices=[int(rng.choice(np.where(c<=c.min()+(trial>=32))[0])) for c in counts]
                v=[float(np.linspace(-3,3,11)[i]) for i in indices]
                if not reserved(action_block(v)):
                    candidates.append((sum(counts[j,i] for j,i in enumerate(indices)),indices,v))
            if not candidates:raise RuntimeError('balanced block feasibility failed')
            _,indices,values=min(candidates,key=lambda c:c[0])
        elif strategy=='random':values=rng.uniform(-3,3,len(roots)).tolist()
        elif strategy=='evaluation':values=rng.uniform(-3,3,len(roots)).tolist()
        else:raise ValueError('unknown strategy')
        holdout=reserved(action_block(values))
        if holdout!=(strategy=='evaluation'):
            rejected+=1
            if rejected>100000:raise RuntimeError('action rejection ceiling')
            continue
        actions.append(dict(zip(roots,values)))
        if strategy=='balanced':
            for j,i in enumerate(indices):counts[j,i]+=1
    return actions


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    draft_manifest(args.project,args.out)
