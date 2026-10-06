"""Retain the complete scored matrix and descriptive Stage A diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from audit_delivery_attribution import audit
from runner_delivery_confirmation import read,write,sha,utc


def gm(values):
    return math.exp(sum(math.log(max(v,1e-12)) for v in values)/len(values))


def summarize(folder,gate,out):
    folder,out=Path(folder),Path(out);custody=audit(folder,require_complete=True)
    g=read(gate);done=read(folder/'complete.json');reg=read(folder/'registration.json')
    if sha(folder/'complete.json')!=g['complete_receipt_sha256']:raise ValueError('gate changed')
    out.mkdir(parents=True,exist_ok=False);(out/'scores').mkdir()
    for name in ('complete.json','registration.json','fit_seal.json'):
        shutil.copyfile(folder/name,out/name)
    grouped=defaultdict(list);rows=[];diagnosis={}
    for case in g['histories']:
        seed=str(case['seed']);file=folder/seed/'scores.json'
        shutil.copyfile(file,out/'scores'/f'{seed}.json');scores=read(file)['results']
        for c in (c for c in reg['matrix'] if c['case']==seed):
            receipt=read(Path(c['path'])/'receipt.json');label=Path(c['path']).name;s=scores[label]['score']
            entry={'seed':int(seed),'label':label,'score':s,
                'eligible_rows_by_head':{n:h['eligible_rows'] for n,h in receipt['heads'].items()},
                'optimizer_updates_by_head':{n:h['optimizer_updates'] for n,h in receipt['heads'].items()},
                'parameters_or_tree_nodes_by_head':{n:h['parameters_or_tree_nodes'] for n,h in receipt['heads'].items()},
                'fit_cpu_seconds':receipt['fit_cpu_seconds'],'worker_wall_seconds':receipt['worker_wall_seconds'],
                'peak_rss_bytes':receipt['peak_rss_bytes'],'receipt_sha256':sha(Path(c['path'])/'receipt.json')}
            rows.append(entry)
            grouped[label].append({'nmse':s['nmse'],'snapped_error':1-s['exact'],
                'nmse_ratio_online':s['nmse']/case['online']['nmse'],
                'snapped_ratio_online':(1-s['exact'])/(1-case['online']['exact']),
                'cpu_seconds':receipt['fit_cpu_seconds']})
        if seed=='124753321':
            diagnosis={'seed':int(seed),'retained_without_rescue':True,'online':case['online'],
                'fits':{label:scores[label] for label in ('all_paid-scm-30000-i0','online_admitted-scm-30000-i0',
                                                       'all_paid-flat-30000-i0')},
                'interpretation':'descriptive local residuals, propagated shifts and box/margin diagnostics; not certified bounds or causal attribution of failure'}
    summary={label:{'n_histories':len(values),'geomean_nmse':gm([v['nmse'] for v in values]),
        'geomean_snapped_error':gm([v['snapped_error'] for v in values]),
        'geomean_nmse_ratio_online':gm([v['nmse_ratio_online'] for v in values]),
        'geomean_snapped_ratio_online':gm([v['snapped_ratio_online'] for v in values]),
        'continuous_wins_online':sum(v['nmse_ratio_online']<1 for v in values),
        'fit_cpu_core_hours':sum(v['cpu_seconds'] for v in values)/3600} for label,values in grouped.items()}
    write(out/'summary.json',{'at':utc(),'stage':'A descriptive complete matrix','gate_sha256':sha(gate),
        'complete_sha256':sha(out/'complete.json'),'worker_sha256':sha(__file__),'custody':custody,
        'configurations':summary,'fits':rows,'worsening_history':diagnosis,
        'interpretation':'one emulator, exposed grid, exploratory; init0 primary, separate init1/2 sensitivity; no seed or initialization selection',
        'cpu_scope':'fit CPU excludes process startup/imports/evaluation; matched CPU includes all five SCM heads',
        'eligibility_scope':'SCM and flat share paid history but nonroot-clamp rows are eligible for some mechanisms and excluded from flat roots-only prediction'})
    write(out/'summary_complete.json',{'at':utc(),'summary_sha256':sha(out/'summary.json'),
        'score_hashes':done['score_hashes'],'n_fits':len(rows),'n_histories':12,'new_queries':0})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('folder','gate','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.folder,a.gate,a.out)
