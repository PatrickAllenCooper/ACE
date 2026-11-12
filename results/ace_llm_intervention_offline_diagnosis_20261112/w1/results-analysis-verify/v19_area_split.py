"""Verifier step 19: log-area (grid 9..32) split by whether the system was at target after initialization."""
import numpy as np, pandas as pd, itertools
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
J = {p: i for i, p in enumerate(POL)}
LA = np.load("out_logarea.npy"); S = np.load("out_step_nmse.npy")
at = S[:, 0, 0] <= 0.04
def signflip(dv):
    dv = np.asarray(dv, float); dv = dv[dv != 0]
    s = np.array([0.0])
    for x in dv: s = np.concatenate([s + x, s - x])
    return float((np.abs(s) >= abs(dv.sum()) - 1e-12).mean())
print("policy minus TR, log-area | informative 14: mean, sd, -/+ , sign-flip p | at-target 10: mean, sd, -/+, sign-flip p")
for p in POL:
    if p == "target_risk": continue
    d = LA[:, J[p]] - LA[:, J["target_risk"]]
    a, b = d[~at], d[at]
    print(f"{p:18s} | {a.mean():+.3f} {a.std(ddof=1):.3f} {int((a<0).sum())}/{int((a>0).sum())} p={signflip(a):.4f} | {b.mean():+.3f} {b.std(ddof=1):.3f} {int((b<0).sum())}/{int((b>0).sum())} p={signflip(b):.4f}")
print("rank on informative 14:", " < ".join(f"{k}({v:.3f})" for k, v in pd.Series(LA[~at].mean(axis=0), index=POL).sort_values().items()))
print("rank on at-target 10:", " < ".join(f"{k}({v:.3f})" for k, v in pd.Series(LA[at].mean(axis=0), index=POL).sort_values().items()))
# initial log10 NMSE level for reference
print("mean log10 init NMSE: informative", np.log10(S[~at, 0, 0]).mean().round(3), "at-target", np.log10(S[at, 0, 0]).mean().round(3))
# improvement from init: mean over grid of log10(NMSE/init)
imp = LA - np.log10(S[:, :, 0])
print("mean log-area improvement over init (negative = better):")
print("   informative:", pd.Series(imp[~at].mean(axis=0), index=POL).round(3).to_dict())
print("   at-target:  ", pd.Series(imp[at].mean(axis=0), index=POL).round(3).to_dict())
