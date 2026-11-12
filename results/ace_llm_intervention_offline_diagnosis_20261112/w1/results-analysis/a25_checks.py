import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
n = 0; ws = []
for w in range(24):
    r = curves[(curves.policy_id == 'learned_ranker') & (curves.w == w)].sort_values('step')
    a = list(r.action_id.values[1:])
    if a == ['set_x_neg'] * 7 + ['set_x_pos'] * 17: n += 1; ws.append(w)
print('ranker runs that are exactly x- x7 then x+ x17:', n, ws)
# all x+ runs
print('ranker runs all x+:', [w for w in range(24) if set(curves[(curves.policy_id == 'learned_ranker') & (curves.w == w) & (curves.step > 0)].action_id) == {'set_x_pos'}])
P = pd.read_csv('paired_endpoints.csv')
q = P[(P.subset == 'all24') & (P.ref == 'uniform_random') & P.policy.isin(['ranker', 'frozen', 'sft', 'dpo'])]
print(q.sort_values('signflip_p')[['endpoint', 'policy', 'mean', 'sd', 'wilcoxon_p', 'signflip_p']].head(8).to_string())
print('min wilcoxon:', q.wilcoxon_p.min(), 'min signflip:', q.signflip_p.min())
q = P[(P.subset == 'informative14') & (P.ref == 'uniform_random') & P.policy.isin(['ranker', 'frozen', 'sft', 'dpo'])]
print('informative-14 min wilcoxon:', q.wilcoxon_p.min(), 'min signflip:', q.signflip_p.min())
# TR obs share by step block; local_info
acq = curves[curves.step > 0]
for p in ['target_risk', 'local_information']:
    d = acq[acq.policy_id == p].copy(); d['b'] = pd.cut(d.step, [0, 2, 4, 8, 12, 16, 24])
    print(p, (d.assign(o=d.action_id == 'observe_all').groupby('b', observed=True).o.agg(['sum', 'count', 'mean'])).round(3).to_dict('index'))
# local_info Y-involving by graph
d = acq[acq.policy_id == 'local_information'].join(meta.graph_stratum, on='w')
print(d.assign(y=d.action_id.isin(['set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy'])).groupby('graph_stratum').y.agg(['sum', 'count']).to_dict('index'))
d = acq[acq.policy_id == 'target_risk'].join(meta.graph_stratum, on='w')
print(d.assign(y=d.action_id.isin(['set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy'])).groupby('graph_stratum').y.agg(['sum', 'count']).to_dict('index'))
# number of curve points per world
print('points per world min/max:', curves.groupby('w').size().min(), curves.groupby('w').size().max())
