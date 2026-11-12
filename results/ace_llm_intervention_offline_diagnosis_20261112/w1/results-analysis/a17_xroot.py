"""Direct signature that X interventions never update the X root mechanism:
xbar_hat = (q_obs - q_do(X=0)) / (q_do(X=1) - q_do(X=0)) recovered from saved predictions at every step."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
T = np.load('truth_means.npy')
c = curves.copy()
c['xbar_hat'] = (c.prediction_0 - c.prediction_2) / (c.prediction_3 - c.prediction_2)
xt = (T[:, 0] - T[:, 2]) / (T[:, 3] - T[:, 2])
for p in POL:
    rng = c[c.policy_id == p].groupby('w').xbar_hat.agg(lambda s: s.max() - s.min())
    print('%-18s runs in which xbar_hat never changes (range < 1e-9): %2d of 24 ; median range %.3g' % (p, (rng < 1e-9).sum(), rng.median()))
ini = c[c.step == 0].drop_duplicates('w').set_index('w')
d = pd.DataFrame({'graph': meta.graph_stratum, 'regime': meta.regime, 'xbar_true': xt, 'xbar_hat_init': ini.xbar_hat, 'err': ini.xbar_hat - xt,
                  'slope_true_total_effect_of_X_on_Z': (T[:, 3] - T[:, 1]) / 2})
d['q0_error_floor_if_xbar_frozen'] = (d.slope_true_total_effect_of_X_on_Z * d.err / 2) ** 2 / 8
print(d.round(4).to_string())
print('\nIf the X root mean stays at its 8-observation estimate and every other mechanism were learned perfectly, the residual NMSE from the observational query alone')
print('(plus the do(Y) queries in forks and colliders, which also use the natural X mean) would be:')
fl = d.q0_error_floor_if_xbar_frozen.copy()
mult = meta.graph_stratum.map({'chain': 1, 'fork': 4, 'collider': 4})
d['nmse_floor_x_only'] = fl * mult
print(d[['graph', 'regime', 'q0_error_floor_if_xbar_frozen', 'nmse_floor_x_only']].round(5).to_string())
print('systems where that floor alone exceeds the 0.04 target:', list(d.index[d.nmse_floor_x_only > THRESH]), '; exceeds 0.02:', list(d.index[d.nmse_floor_x_only > 0.02]), '; exceeds 0.01:', list(d.index[d.nmse_floor_x_only > 0.01]))
fe = cells.pivot(index='w', columns='policy_id', values='final_error')
print(pd.DataFrame({'floor': d.nmse_floor_x_only, 'dpo_final': fe.dpo_llm, 'ranker_final': fe.learned_ranker, 'sft_final': fe.sft_llm, 'tr_final': fe.target_risk}).round(4).to_string())
print('rank correlation of the X-only floor with DPO final error: %.3f ; with ranker final error: %.3f ; with target_risk final error: %.3f'
      % (d.nmse_floor_x_only.corr(fe.dpo_llm, method='spearman'), d.nmse_floor_x_only.corr(fe.learned_ranker, method='spearman'), d.nmse_floor_x_only.corr(fe.target_risk, method='spearman')))
