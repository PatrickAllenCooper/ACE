"""Verifier step 16: is the normal-theory (noncentral t) power calibrated for these zero-inflated, skewed paired differences?
Bootstrap systems from each observed policy-minus-TR difference vector and compare rejection rates with the formula, at n=24 and at larger n."""
import pandas as pd, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
C = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index()
def power_formula(delta, sd, n, alpha=0.05):
    df = n - 1; nc = delta / sd * np.sqrt(n); tc = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
rng = np.random.default_rng(3)
NB = 4000
print("policy minus TR | mean | sd | formula power n=24 | bootstrap t-test power n=24 | formula n=96 | bootstrap n=96")
for p in POL:
    if p == "target_risk": continue
    d = (C[p] - C.target_risk).values.astype(float)
    out = []
    for n in (24, 96):
        idx = rng.integers(0, 24, size=(NB, n))
        B = d[idx]
        m = B.mean(axis=1); s = B.std(axis=1, ddof=1)
        t = np.where(s > 0, m / (s / np.sqrt(n)), 0.0)
        rej = (np.abs(t) > stats.t.ppf(.975, n - 1)).mean()
        out.append((power_formula(abs(d.mean()), d.std(ddof=1), n), rej))
    print(f"{p:18s} {d.mean():+.3f} {d.std(ddof=1):.3f} | {out[0][0]:.3f} | {out[0][1]:.3f} | {out[1][0]:.3f} | {out[1][1]:.3f}")
# scaled-effect planning check: take the dpo-minus-TR difference distribution, shift nothing, ask: at n=305 what is the bootstrap power for a 1-unit mean difference?
# construct a 1-unit-mean population by mixing: with prob q take the observed (random - TR) vector, else take the sign-symmetrised (zero-mean) version.
d = (C.uniform_random - C.target_risk).values.astype(float)
q = 1.0 / d.mean()
for n in (150, 305, 408):
    rej = 0
    for _ in range(2000):
        b = d[rng.integers(0, 24, size=n)]
        flip = rng.random(n) > (0.5 + q / 2)      # P(keep sign) = (1+q)/2 -> expected mean = q * mean(d) = 1
        b = np.where(flip, -b, b)
        s = b.std(ddof=1)
        rej += abs(b.mean() / (s / np.sqrt(n))) > stats.t.ppf(.975, n - 1)
    sd_pop = np.sqrt((d ** 2).mean() - 1.0)
    print(f"mixture population with mean 1.0 built from random-minus-TR magnitudes (population SD {sd_pop:.2f}): n={n}: bootstrap power {rej/2000:.3f}; formula {power_formula(1, sd_pop, n):.3f}")
