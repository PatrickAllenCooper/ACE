"""Permutation baseline for the fixed-sequence agreement: shuffle step order within each run (keeps each run's action mix)."""
import numpy as np, pandas as pd
from collections import Counter
from common import *
cells, curves, meta = load()
acq = curves[curves.step > 0]
seqs = {(w, p): list(r.sort_values('step').action_id.values) for (w, p), r in acq.groupby(['w', 'policy_id'])}
CAN = {'observe_all': 'obs', 'set_x_neg': 'Xext', 'set_x_pos': 'Xext', 'set_x_zero': 'x0', 'set_y_neg': 'Yext', 'set_y_pos': 'Yext', 'set_y_zero': 'y0', 'set_xy': 'xy'}
groups = meta.groupby(['graph_stratum', 'cost_stratum']).groups
def agree(S, p, loo, maxstep=24):
    h = t = 0
    for cell, ws in groups.items():
        for s in range(maxstep):
            acts = [(w, S[(w, p)][s]) for w in ws if len(S[(w, p)]) > s]
            if len(acts) < 2: continue
            if not loo:
                h += Counter(a for _, a in acts).most_common(1)[0][1]; t += len(acts)
            else:
                for w, a in acts:
                    c = Counter(b for v, b in acts if v != w); top = max(c.values())
                    h += (sorted(k for k, v in c.items() if v == top)[0] == a); t += 1
    return h / t
rng = np.random.default_rng(7)
for p in ['target_risk', 'local_information']:
    for name, S in [('raw', seqs), ('sign-free', {k: [CAN[a] for a in v] for k, v in seqs.items()})]:
        obs_m, obs_l = agree(S, p, False), agree(S, p, True)
        pm, pl = [], []
        for _ in range(400):
            Sp = {k: (list(rng.permutation(v)) if k[1] == p else v) for k, v in S.items()}
            pm.append(agree(Sp, p, False)); pl.append(agree(Sp, p, True))
        print('%-18s %-9s in-sample modal %.3f (within-run shuffle: mean %.3f, 95%% range %.3f-%.3f) ; LOO %.3f (shuffle mean %.3f, 95%% range %.3f-%.3f)'
              % (p, name, obs_m, np.mean(pm), np.quantile(pm, .025), np.quantile(pm, .975), obs_l, np.mean(pl), np.quantile(pl, .025), np.quantile(pl, .975)))
# first four steps only (where structure is strongest)
for p in ['target_risk', 'local_information']:
    S = {k: [CAN[a] for a in v] for k, v in seqs.items()}
    print('%-18s sign-free in-sample modal share, steps 1-4: %.3f ; steps 5-12: %.3f ; steps 13-24: %.3f'
          % (p, agree({k: v[:4] for k, v in S.items()}, p, False), agree({k: v[4:12] for k, v in S.items()}, p, False), agree({k: v[12:] for k, v in S.items()}, p, False)))
# share of X-extreme among first 3 actions
for p in ['target_risk', 'local_information']:
    f3 = [CAN[a] for w in range(24) for a in seqs[(w, p)][:3]]
    print('%-18s first three actions: %s' % (p, dict(Counter(f3))))
