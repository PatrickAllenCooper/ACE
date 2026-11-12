"""Verifier step 14: robustness of the power-planning inputs.
(1) exact threshold check for 'within 12.5 percent'; (2) sensitivity of the replicated-design sample size to SD rounding;
(3) data-based replicate share from the dpo/sft twins at cap 18; (4) bootstrap power of an oracle at the floor with the sign-flip test;
(5) SDs restricted to the 14 informative systems; (6) planning-SD sensitivity (which pairs define 'the' SD)."""
import pandas as pd, numpy as np, itertools, warnings
from scipy import stats
warnings.simplefilter("ignore")
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv"); curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
J = {p: i for i, p in enumerate(POL)}
S = np.load(OUT + "out_step_nmse.npy")
C = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index().values.astype(float)
init = S[:, 0, 0]; at = init <= 0.04
print("(1) init NMSE just above threshold:", [(f"w{i:02d}", repr(float(init[i])), f"{100*(init[i]/0.04-1):.4f}% above") for i in np.argsort(init) if 0.04 < init[i] < 0.05])

def power_paired(delta, sd, n, alpha=0.05):
    df = n - 1; nc = delta / sd * np.sqrt(n); tc = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
def n_needed(delta, sd, power=0.8):
    n = 3
    while power_paired(delta, sd, n) < power: n += 1
    return n
print("(2) replicated design, n for 1 unit at 80% power, 8 replicates:")
for sd in (6.2, 6.21, 6.2113):
    print(f"    SD {sd}: " + ", ".join(f"r={r}: {n_needed(1, sd*np.sqrt((1-r)+r/8))}" for r in (0.5, 0.6, 0.7, 0.8)))
# normal-approximation version
z = stats.norm.ppf(.975) + stats.norm.ppf(.8)
print("    normal approximation (no t correction), SD 6.2113: " + ", ".join(f"r={r}: {int(np.ceil(z**2 * 6.2113**2 * ((1-r)+r/8)))}" for r in (0.5, 0.8)))

# (3) replicate share from twins at cap 18
def capped_at(cap, th=0.04):
    M = np.full((24, 8), float(cap))
    for i in range(24):
        for j in range(8):
            idx = np.nonzero(S[i, j, :cap - 8 + 1] <= th)[0]
            if len(idx): M[i, j] = 8 + idx[0]
    return M
C18 = capped_at(18)
twin = C18[:, J["dpo_llm"]] - C18[:, J["sft_llm"]]
rep_var_diff = twin.var(ddof=1)          # variance of a difference between two independent runs of the SAME policy (pure replicate noise)
print(f"(3) cap 18: var of twin difference (pure replicate, same policy) = {rep_var_diff:.3f} (SD {np.sqrt(rep_var_diff):.3f})")
for p in POL:
    if p == "target_risk": continue
    dv = C18[:, J[p]] - C18[:, J["target_risk"]]
    print(f"    cap 18: {p:18s} minus TR: mean {dv.mean():+.3f} SD {dv.std(ddof=1):.3f} var {dv.var(ddof=1):.3f} | twin-replicate share of this variance = {rep_var_diff/dv.var(ddof=1):.2f}")
allv = [np.var(C18[:, a] - C18[:, b], ddof=1) for a, b in itertools.combinations(range(8), 2) if {POL[a], POL[b]} != {"dpo_llm", "sft_llm"}]
print(f"    cap 18: median variance over the other 27 pairs {np.median(allv):.3f} -> twin-replicate share {rep_var_diff/np.median(allv):.2f} (range {rep_var_diff/max(allv):.2f} to {rep_var_diff/min(allv):.2f})")

# (4) bootstrap power of oracle at the floor (resampling systems), sign-flip Monte Carlo test and t-test
floor = np.where(at, 8.0, 9.0)
dfl = C[:, J["target_risk"]] - floor
rng = np.random.default_rng(1)
def sf_p(dv, rng, B=2000):
    obs = abs(dv.sum()); sg = rng.choice([-1.0, 1.0], size=(B, len(dv)))
    return ((np.abs(sg @ dv) >= obs - 1e-12).sum() + 1) / (B + 1)
rej_t = rej_sf = 0; NB = 2000
for _ in range(NB):
    b = rng.choice(dfl, size=24, replace=True)
    if b.std(ddof=1) == 0:
        continue
    rej_t += stats.ttest_1samp(b, 0).pvalue < 0.05
    rej_sf += sf_p(b, rng, 1000) < 0.05
print(f"(4) bootstrap (systems resampled, n=24) power of an oracle at the floor vs TR: t-test {rej_t/NB:.3f}; sign-flip {rej_sf/NB:.3f}; (noncentral-t formula gave 0.676)")

# (5) informative systems only
print("(5) policy minus TR restricted to the 14 systems not at target after init:")
for p in POL:
    if p == "target_risk": continue
    dv = (C[:, J[p]] - C[:, J["target_risk"]])[~at]
    print(f"    {p:18s} mean {dv.mean():+.3f} SD {dv.std(ddof=1):.3f} d {dv.mean()/dv.std(ddof=1):.3f}")
sds14 = [np.std((C[:, J[p]] - C[:, J['target_risk']])[~at], ddof=1) for p in POL if p != "target_risk"]
print(f"    median SD on informative systems: {np.median(sds14):.3f}; n informative systems for a 1.714-unit difference (=1 unit average over a 24-system mix with 10 uninformative): {n_needed(24/14, np.median(sds14))}; for 1 unit: {n_needed(1, np.median(sds14))}; for 2 units: {n_needed(2, np.median(sds14))}")
# (6) planning SD sensitivity
print("(6) n for a 1-unit difference at 80% power under alternative planning SDs:")
for lab, sd in (("coverage - TR (most TR-like arm) 3.074", 3.074), ("min of 7", min(np.std(C[:, J[p]] - C[:, J['target_risk']], ddof=1) for p in POL if p != 'target_risk')),
                ("median of 7 vs-TR", 6.2113), ("max of 7 vs-TR 6.763", 6.763), ("median of all 28 pairs 6.977", 6.977), ("max of all 28 pairs 9.035", 9.035)):
    print(f"    {lab}: SD {sd:.3f} -> n = {n_needed(1, sd)} (80%), {n_needed(1, sd, .9)} (90%); MDD at n=24: {sd*(stats.t.ppf(.975,23)+stats.t.ppf(.8,23))/np.sqrt(24):.2f} (approx)")
