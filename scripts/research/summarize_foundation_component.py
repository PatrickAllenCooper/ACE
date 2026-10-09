#!/usr/bin/env python3
"""Fixed descriptive comparisons; never filter a failed/zero world out of aggregates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS=('polynomial','extra_trees','tabpfn_v2','grammar','language')
ENDPOINTS=('M_local','Y_local','Y_composed')
SEEDS=tuple(range(91000,91006))


def summarize(root):
    terminal=json.loads((root/'terminal.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    expected={(s,m) for s in SEEDS for m in METHODS}
    if plan['mode']!='pilot' or {(r['seed'],r['method']) for r in plan['cells']} != expected or len(plan['cells'])!=30:
        raise ValueError('not the fixed complete plan')
    rows={(r['seed'],r['method']):r for r in terminal['cells']}
    if set(rows)!=expected or len(terminal['cells'])!=30:
        raise ValueError('incomplete terminal disposition')
    def metric(row,endpoint):
        metrics=row.get('metrics')
        if not isinstance(metrics,dict):return None
        value=metrics.get(endpoint)
        return value.get('nmse') if isinstance(value,dict) else None
    comparisons=[]
    for method in METHODS:
        if method=='grammar':continue
        for endpoint in ENDPOINTS:
            worlds=[]
            for seed in SEEDS:
                a,b=rows[seed,method],rows[seed,'grammar'];ratio=None;why=None
                if a['status']!='complete' or b['status']!='complete':why='missing_or_failed_cell'
                else:
                    x=metric(a,endpoint);y=metric(b,endpoint)
                    if not all(type(v) in (int,float) for v in (x,y)):why='missing_or_nonnumeric_error'
                    elif not all(math.isfinite(v) and v>0 for v in (x,y)):why='zero_or_nonfinite_error'
                    else:
                        ratio=x/y
                        if not math.isfinite(ratio) or ratio<=0:ratio=None;why='nonfinite_ratio'
                worlds.append({'seed':seed,'ratio':ratio,'undefined_reason':why,
                               'language_fallback':a.get('fallback'),'proposal_valid':a.get('proposal_valid')})
            vals=[r['ratio'] for r in worlds]
            defined=all(v is not None for v in vals)
            comparisons.append({'method':method,'reference':'grammar','endpoint':endpoint,
                'direction':'candidate_NMSE / grammar_NMSE','worlds':worlds,'planned_worlds':6,
                'defined_worlds':sum(v is not None for v in vals),
                'arithmetic_mean_paired_ratios':math.fsum(v/6 for v in vals) if defined else None,
                'geometric_mean_paired_ratios':math.exp(math.fsum(math.log(v)/6 for v in vals)) if defined else None})
    invalid_numbers=[]
    def safe(value,path='cells'):
        if isinstance(value,float) and not math.isfinite(value):
            invalid_numbers.append({'path':path,'original_representation':repr(value)});return None
        if isinstance(value,dict):return {k:safe(v,path+'.'+str(k)) for k,v in value.items()}
        if isinstance(value,list):return [safe(v,path+'['+str(i)+']') for i,v in enumerate(value)]
        return value
    safe_cells=safe(list(rows.values()))
    return {'invalid_numeric_fields':invalid_numbers,'kind':'development, fixed histories, descriptive only','terminal_sha256':hashlib.sha256((root/'terminal.json').read_bytes()).hexdigest(),
            'cells':safe_cells,'comparisons':comparisons,'total_planned_cells':30,
            'language_valid_count':sum(r.get('proposal_valid') is True for r in rows.values()),
            'language_fallback_count':sum(r.get('fallback')=='grammar' for r in rows.values()),
            'inference':'No significance test, acquisition claim, or valid-only language aggregate'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=(json.dumps(summarize(a.root),indent=2,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
