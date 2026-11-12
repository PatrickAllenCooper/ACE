import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
S = wide_steps(curves)
ini_e = S['dpo_llm'][:, 0]; inf = ini_e > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
ini = curves[curves.step == 0].drop_duplicates('w').set_index('w')
xbar = ((ini.prediction_0 - ini.prediction_2) / (ini.prediction_3 - ini.prediction_2)).values
near = xbar < 0     # X = -1 is on the same side as the natural mean of X (less leverage)
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
lf = {p: np.log10(S[p][:, -1]) for p in POL}
for name, m in [('x- is the NEAR side (xbar_hat < 0)', near), ('x- is the FAR side (xbar_hat > 0)', ~near)]:
    for sub, mm in [('all', m), ('informative', m & inf)]:
        d = (cw['dpo_llm'] - cw['target_risk']).values[mm]
        print('%-36s %-11s n=%2d : DPO - target_risk cost %+.2f ; log-area %+.3f ; log final %+.3f ; DPO never attained %d ; DPO - random cost %+.2f ; DPO - frozen cost %+.2f'
              % (name, sub, mm.sum(), d.mean(), (aulc['dpo_llm'] - aulc['target_risk'])[mm].mean(), (lf['dpo_llm'] - lf['target_risk'])[mm].mean(),
                 (cw['dpo_llm'].values[mm] == 32).sum(), (cw['dpo_llm'] - cw['uniform_random']).values[mm].mean(), (cw['dpo_llm'] - cw['frozen_llm']).values[mm].mean()))
print('near-side systems:', list(np.where(near)[0]), ' informative among them:', list(np.where(near & inf)[0]))
