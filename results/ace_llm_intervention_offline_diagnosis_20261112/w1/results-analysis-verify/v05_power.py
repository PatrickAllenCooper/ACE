"""Verifier step 5: paired differences vs target_risk (capped cost), SDs, exact sign-flip p, and power planning.
Fresh code. Paired t-test power via the noncentral t distribution (scipy.stats.nct)."""
import pandas as pd, numpy as np, itertools
from scipy import stats
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
piv = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index()

def signflip_exact(d):
    """Exact two-sided sign-flip (paired randomization) p for the mean; enumerates 2^k over nonzero entries."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    k = len(nz)
    obs = abs(nz.sum())
    # enumerate all sign patterns via meet-in-the-middle-free brute force (k <= 24 -> at most 16.7M; here k<=14)
    if k == 0:
        return 1.0
    signs = np.array(list(itertools.product([1, -1], repeat=k)), dtype=np.int8) if k <= 20 else None
    if signs is None:
        # iterative convolution for larger k
        sums = np.array([0.0])
        for v in nz:
            sums = np.concatenate([sums + v, sums - v])
        tot = sums
    else:
        tot = signs @ nz
    return float((np.abs(tot) >= obs - 1e-12).mean())

def signflip_exact_big(d):
    d = np.asarray(d, float)
    nz = d[d != 0]
    sums = np.array([0.0])
    for v in nz:
        sums = np.concatenate([sums + v, sums - v])
    return float((np.abs(sums) >= abs(nz.sum()) - 1e-12).mean())

print("=== policy minus target_risk, capped total cost (n=24) ===")
rows = []
for p in POL:
    if p == "target_risk": continue
    d = (piv[p] - piv.target_risk).values
    t, pt = stats.ttest_1samp(d, 0)
    rows.append(dict(policy=p, mean=d.mean(), sd=d.std(ddof=1), neg=int((d < 0).sum()), zero=int((d == 0).sum()), pos=int((d > 0).sum()),
                     signflip_p=signflip_exact_big(d), t_p=pt, d_std=d.mean() / d.std(ddof=1)))
R = pd.DataFrame(rows).set_index("policy")
pd.set_option("display.width", 250)
print(R.round(4).to_string())
sds = R.sd.values
print("median of the 7 policy-minus-TR SDs:", np.median(sds))
# Holm for the 7 vs-TR tests
ps = R.signflip_p.sort_values()
m = len(ps)
holm = {}
run = 0
for i, (k, v) in enumerate(ps.items()):
    run = max(run, min(1, (m - i) * v)); holm[k] = run
print("Holm-adjusted (7 tests vs TR):", {k: round(v, 4) for k, v in holm.items()})

# all 28 pairs
allsd = []
for a, b in itertools.combinations(POL, 2):
    d = (piv[a] - piv[b]).values
    allsd.append(((a, b), d.std(ddof=1)))
print("median SD over all 28 pairs:", np.median([s for _, s in allsd]), " min:", min(s for _, s in allsd), " max:", max(s for _, s in allsd))

def power_paired(delta, sd, n, alpha=0.05):
    df = n - 1
    nc = delta / sd * np.sqrt(n)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)

def n_needed(delta, sd, power, alpha=0.05):
    n = 3
    while power_paired(delta, sd, n, alpha) < power:
        n += 1
    return n

def mdd(sd, n, power=0.8, alpha=0.05):
    lo, hi = 0.0, 100 * sd
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_paired(mid, sd, n, alpha) < power: lo = mid
        else: hi = mid
    return hi

SD = float(np.median(sds))
print(f"\n=== power planning at SD = {SD:.4f} ===")
print("MDD at n=24 (80% power, two-sided .05):", round(mdd(SD, 24), 3), "  (90%):", round(mdd(SD, 24, 0.9), 3))
print("MDD normal approx:", round(SD * (stats.norm.ppf(.975) + stats.norm.ppf(.8)) / np.sqrt(24), 3))
for delta in (1, 2, 3.0416667):
    print(f"n for delta={delta}: 80% -> {n_needed(delta, SD, .8)}, 90% -> {n_needed(delta, SD, .9)}")
# noninferiority, margin 1, true difference 0, one-sided alpha .025: test H0: mu >= 1 vs H1: mu < 1  -> same as delta=1 one-sided .025
def power_ni(margin, sd, n, alpha=0.025, true=0.0):
    df = n - 1
    nc = (margin - true) / sd * np.sqrt(n)
    tc = stats.t.ppf(1 - alpha, df)
    return stats.nct.sf(tc, df, nc)
def n_ni(margin, sd, power, alpha=0.025):
    n = 3
    while power_ni(margin, sd, n, alpha) < power: n += 1
    return n
print("noninferiority margin 1, true diff 0, one-sided .025: 80% ->", n_ni(1, SD, .8), " 90% ->", n_ni(1, SD, .9))
# oracle at the floor
init = curves[curves.total_cost_units == 8].groupby("world_id").normalized_mse.first().sort_index().values
floor = np.where(init <= 0.04, 8, 9)
dfl = piv.target_risk.values - floor
print("\nTR minus floor: mean", dfl.mean(), "sd", dfl.std(ddof=1), " n=24")
print("power of oracle-at-floor vs TR using its own paired SD:", round(power_paired(dfl.mean(), dfl.std(ddof=1), 24), 4))
print("power of a 3.0417 difference using SD", round(SD, 3), ":", round(power_paired(dfl.mean(), SD, 24), 4))
print("exact sign-flip p of TR minus floor:", signflip_exact_big(dfl), " t-test p:", stats.ttest_1samp(dfl, 0).pvalue)
# envelope
env = piv.min(axis=1).values
de = piv.target_risk.values - env
print("TR minus envelope: mean", de.mean(), "sd", de.std(ddof=1), " power:", round(power_paired(de.mean(), de.std(ddof=1), 24), 4))

# replicate share planning: Var(diff of means over R replicates) = SD^2 * ((1-r) + r/R)
print("\n=== replicated design: 8 replicates per arm per system, replicate share r of paired-difference variance ===")
for r in (0.5, 0.6, 0.7, 0.8):
    sd_eff = SD * np.sqrt((1 - r) + r / 8)
    print(f"r={r}: effective SD {sd_eff:.3f}; n for 1 unit 80% -> {n_needed(1, sd_eff, .8)}, 90% -> {n_needed(1, sd_eff, .9)}; for 2 units 80% -> {n_needed(2, sd_eff, .8)}")
R.to_csv("/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/out_vsTR_cost.csv")
