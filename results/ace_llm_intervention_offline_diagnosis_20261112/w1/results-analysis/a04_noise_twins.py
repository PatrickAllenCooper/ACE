"""Are measurement-noise draws shared across arms? Compare arms that took the same action at the same step.
Then use DPO vs SFT runs with identical action prefixes as 'accidental replicates' of one policy on one system."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
seq = {(w, p): r.sort_values('step') for (w, p), r in curves.groupby(['w', 'policy_id'])}
# 1) step-1 same action, same world, different arms: identical predictions?
same, diff = 0, 0
examples = []
for w in range(24):
    by = {}
    for p in POL:
        r = seq[(w, p)]
        if len(r) > 1:
            by.setdefault(r.action_id.values[1], []).append(p)
    for a, ps in by.items():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                x = seq[(w, ps[i])][PRED].values[1]; y = seq[(w, ps[j])][PRED].values[1]
                if np.allclose(x, y, rtol=0, atol=1e-12):
                    same += 1
                else:
                    diff += 1
                    if len(examples) < 3:
                        examples.append((w, a, ps[i], ps[j], float(np.abs(x - y).max())))
print('pairs of arms with the same first action in the same world: identical post-action predictions %d, different %d' % (same, diff))
print('examples (world, action, arm, arm, max abs prediction difference):', examples)

# 2) DPO vs SFT identical prefix length
rows = []
for w in range(24):
    a = seq[(w, 'dpo_llm')]; b = seq[(w, 'sft_llm')]
    sa = list(a.action_id.values[1:]); sb = list(b.action_id.values[1:])
    k = 0
    while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
        k += 1
    rows.append({'w': w, 'graph': meta.graph_stratum[w], 'regime': meta.regime[w], 'cost': meta.cost_stratum[w],
                 'sft_steps': len(sb), 'common_prefix': k,
                 'sft_non_xneg': ','.join('%d:%s' % (i + 1, ASHORT[x]) for i, x in enumerate(sb) if x != 'set_x_neg')})
tw = pd.DataFrame(rows).set_index('w')
print(tw.to_string())
print('total matched (system, step) twin points:', tw.common_prefix.sum(), '; systems with prefix >= 12:', (tw.common_prefix >= 12).sum(), '; full 23+:', (tw.common_prefix >= 23).sum())

# 3) replicate noise of log10 error at fixed number of identical set_x_neg actions
recs = []
for w in range(24):
    k = tw.common_prefix[w]
    a = seq[(w, 'dpo_llm')].normalized_mse.values; b = seq[(w, 'sft_llm')].normalized_mse.values
    for s in range(1, k + 1):
        recs.append({'w': w, 'step': s, 'dlog': np.log10(a[s]) - np.log10(b[s]), 'ea': a[s], 'eb': b[s]})
tr = pd.DataFrame(recs)
print()
print('Twin differences in log10 NMSE (DPO minus SFT, identical action prefix, independent measurement noise):')
for s in [1, 2, 4, 8, 12, 16, 20, 23]:
    d = tr[tr.step == s].dlog
    if len(d) > 1:
        print('  after %2d identical actions: n=%2d  mean %+.3f  SD %.3f  => single-run replicate SD of log10 NMSE ~ %.3f ; median |diff| %.3f (ratio x%.2f)'
              % (s, len(d), d.mean(), d.std(ddof=1), np.sqrt((d ** 2).mean() / 2), d.abs().median(), 10 ** d.abs().median()))
# pooled across steps per world then across worlds
pw = tr.groupby('w').dlog.agg(['mean', 'std', 'count'])
print('per-world mean twin difference over all common steps: mean %+.3f SD %.3f n=%d' % (pw['mean'].mean(), pw['mean'].std(ddof=1), len(pw)))
print('root-mean-square twin difference / sqrt(2) pooled over all %d points: %.3f log10 units' % (len(tr), np.sqrt((tr.dlog ** 2).mean() / 2)))
# 4) twin first-crossing: capped cost for the identical-prefix portion
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
rows = []
for w in range(24):
    k = tw.common_prefix[w]
    cd, cs = cw.loc[w, 'dpo_llm'], cw.loc[w, 'sft_llm']
    # crossing is 'covered' by the identical prefix if it happens within k actions for both, or neither crosses within k
    cost_k = seq[(w, 'dpo_llm')].total_cost_units.values[k] if k < len(seq[(w, 'dpo_llm')]) else 32
    a = seq[(w, 'dpo_llm')]; b = seq[(w, 'sft_llm')]
    ca = a.total_cost_units.values[:k + 1][a.normalized_mse.values[:k + 1] <= THRESH]
    cb = b.total_cost_units.values[:k + 1][b.normalized_mse.values[:k + 1] <= THRESH]
    rows.append({'w': w, 'prefix_actions': k, 'prefix_end_cost': cost_k,
                 'dpo_cross_in_prefix': ca[0] if len(ca) else np.nan, 'sft_cross_in_prefix': cb[0] if len(cb) else np.nan,
                 'dpo_capped': cd, 'sft_capped': cs})
x = pd.DataFrame(rows).set_index('w')
ini = curves[curves.step == 0].groupby('w').normalized_mse.first()
x['init_err'] = ini
print()
print(x[ini > THRESH].to_string())
