"""How the fixed public scale (2) interacts with regime: spread of the true target means (reconstructed) vs initial error."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
T = np.load('truth_means.npy')
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]
d = meta[['graph_stratum', 'regime', 'cost_stratum']].copy()
d['init_nmse'] = ini
d['true_target_sd'] = T.std(axis=1, ddof=0)
d['true_target_rms'] = np.sqrt((T ** 2).mean(axis=1))
d['zero_predictor_nmse'] = (T ** 2).mean(axis=1) / 4           # error of the prior-mean prediction (all zeros), in the screen's units
d['constant_predictor_nmse'] = T.var(axis=1) / 4               # error of predicting the mean of the 8 targets
d['init_relative_to_zero_predictor'] = d.init_nmse / d.zero_predictor_nmse
d['init_rel_to_spread'] = d.init_nmse / d.constant_predictor_nmse
pd.set_option('display.width', 250)
print(d.round(4).to_string())
print()
print(d.groupby('regime')[['init_nmse', 'true_target_sd', 'zero_predictor_nmse', 'constant_predictor_nmse', 'init_relative_to_zero_predictor', 'init_rel_to_spread']].median().round(4).to_string())
print(d.groupby('graph_stratum')[['init_nmse', 'true_target_sd', 'zero_predictor_nmse', 'constant_predictor_nmse', 'init_relative_to_zero_predictor', 'init_rel_to_spread']].median().round(4).to_string())
print('\nsystems where even the prior-mean (all-zero) prediction is already within the 0.04 target:', list(d.index[d.zero_predictor_nmse <= THRESH]))
print('systems where a single constant (the mean of the 8 true targets) is within the 0.04 target:', list(d.index[d.constant_predictor_nmse <= THRESH]),
      '; within 0.02:', list(d.index[d.constant_predictor_nmse <= 0.02]))
at = ini <= THRESH
print('median true-target SD: at-target-at-init systems %.3f ; others %.3f' % (d.true_target_sd[at].median(), d.true_target_sd[~at].median()))
print('Spearman(init NMSE, true target SD) = %.3f ; Spearman(init NMSE, zero-predictor NMSE) = %.3f' % (d.init_nmse.corr(d.true_target_sd, method='spearman'), d.init_nmse.corr(d.zero_predictor_nmse, method='spearman')))
# a scale-free threshold: fraction of the system's own target spread
for f in [0.02, 0.05, 0.1]:
    th = f * d.constant_predictor_nmse.values
    at_init = (ini <= th).sum()
    tr_att = ((S['target_risk'] <= th[:, None]).any(axis=1)).sum()
    print('relative threshold = %.2f x Var(true targets)/4: at target at init %d ; target_risk ever attains %d ; target_risk final attains %d'
          % (f, at_init, tr_att, (S['target_risk'][:, -1] <= th).sum()))
