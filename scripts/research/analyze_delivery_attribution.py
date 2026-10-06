"""Summarize fully sealed Stage A; explanatory gate, not a new confirmation."""
import argparse
import math
import json
from pathlib import Path
from delivery_attribution import SEEDS
from runner_delivery_confirmation import read,write,sha,utc
from audit_delivery_attribution import audit


def analyze(folder,online_scores,out):
    custody = audit(folder, require_complete=True)
    folder=Path(folder);done=read(folder/'complete.json');reg=read(folder/'registration.json')
    if done['n_histories']!=12 or done['n_fits']!=480 or done['registration_sha256']!=sha(folder/'registration.json'):
        raise ValueError('full study not complete')
    online=read(online_scores);original={str(p['seed']):p['online']['chain'] for p in online['pairs']}
    cells=[]
    for s in SEEDS:
        case=str(s);file=folder/case/'scores.json'
        if sha(file)!=done['score_hashes'][case]:raise ValueError('score changed')
        z=read(file)['results'];simple=reg['strongest_simpler_development']
        label='all_paid-flat-30000-i0' if simple=='flat' else f'all_paid-{simple}-100-i0'
        cells.append({'seed':s,'online':original[case],'delivery':z['all_paid-scm-30000-i0']['score'],
                      'simpler':z[label]['score'],'matched_cpu_flat':z['all_paid-flat-matched-cpu-i0']['score'],
                      'data_ablation':z['final_buffer-scm-30000-i0']['score'],
                      'optimization_ablation':z['all_paid-scm-100-i0']['score'],
                      'unused_observations_ablation':z['online_admitted-scm-30000-i0']['score']})
    def ratio(a,b):return math.exp(sum(math.log(max(c[a]['nmse'],1e-12)/max(c[b]['nmse'],1e-12)) for c in cells)/12)
    ratios={key:ratio('delivery',key) for key in ('online','simpler','matched_cpu_flat','data_ablation','optimization_ablation','unused_observations_ablation')}
    causal=ratios['simpler']<=.8 and ratios['matched_cpu_flat']<=.8
    ablations=('data_ablation','optimization_ablation','unused_observations_ablation')
    decisive=min(ablations,key=lambda k:ratios[k])
    selected='scm' if causal else reg['strongest_simpler_development']
    result={'at':utc(),'stage':'A exploratory attribution','complete_receipt_sha256':sha(folder/'complete.json'),
        'custody_audit':custody,
        'online_scores_sha256':sha(online_scores),'ratios_continuous_nmse_init0':ratios,'histories':cells,
        'causal_factorization_signal_exploratory':causal,'selected_delivery':selected,
        'strongest_simpler':reg['strongest_simpler_development'],'decisive_ablation':decisive,
        'stage_b_release':'protocol/resource/estimand validation still required; do not auto-launch from this result',
        'scope':('provisional causal-surrogate hypothesis for prospective testing' if causal else 'narrow delivery/accounting study; no architecture claim'),
        'failures_retained':True,'no_new_simulator_responses':True}
    out=Path(out)
    if out.exists():
        previous=read(out)
        comparable=lambda r:{k:v for k,v in r.items() if k not in ('at','custody_audit')}
        if not previous['custody_audit']['full_acceptance'] or comparable(previous)!=comparable(result):
            raise ValueError('existing immutable attribution gate differs; preserve it')
        return previous
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--online-scores',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();analyze(a.folder,a.online_scores,a.out)
