"""Verifier step 7: alternative endpoints. Log-area paired stats vs TR; sustained-attainment cost; geometric-mean final NMSE;
threshold sweep; standardized effects; coverage-vs-TR contrasts. Fresh code."""
import pandas as pd, numpy as np, itertools
from scipy import stats
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
W = sorted(cells.world_id.unique())
S = np.load(OUT + "out_step_nmse.npy")           # (24 worlds, 8 policies, grid 8..32)
L = np.log10(S)
TRI = POL.index("target_risk")

def signflip_mitm(d):
    """Exact two-sided sign-flip p-value by meet in the middle (all 2^n sign patterns, including zeros as-is)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0: return 1.0
    obs = abs(d.sum())
    h = n // 2
    def allsums(v):
        s = np.array([0.0])
        for x in v: s = np.concatenate([s + x, s - x])
        return s
    a, b = allsums(d[:h]), np.sort(allsums(d[h:]))
    tol = 1e-12 * max(1.0, obs)
    # count pairs with a+b >= obs - tol  or a+b <= -(obs - tol)
    hi = len(b) - np.searchsorted(b, (obs - tol) - a, side="left")
    lo = np.searchsorted(b, -(obs - tol) - a, side="right")
    cnt = hi.sum() + lo.sum()
    if obs - tol <= 0: cnt = len(a) * len(b)
    return float(cnt) / (len(a) * len(b))

def summarize(M, name, ref=TRI):
    rows = []
    for j, p in enumerate(POL):
        if j == ref: continue
        d = M[:, j] - M[:, ref]
        rows.append(dict(policy=p, mean=d.mean(), sd=d.std(ddof=1), neg=int((d < 0).sum()), zero=int((d == 0).sum()), pos=int((d > 0).sum()),
                         signflip_p=signflip_mitm(d), t_p=stats.ttest_1samp(d, 0).pvalue, d_std=d.mean() / d.std(ddof=1)))
    R = pd.DataFrame(rows).set_index("policy")
    print(f"--- {name}: policy minus target_risk ---")
    print(R.round(4).to_string())
    return R

pd.set_option("display.width", 250)
# 1. log-area (definition B: grid 9..32)
LA = L[:, :, 1:].mean(axis=2)
print("mean log-area (grid 9..32):", dict(zip(POL, LA.mean(axis=0).round(4))))
R_la = summarize(LA, "log-area, grid 9..32")
print("median of 7 vs-TR SDs (log-area):", np.median(R_la.sd.values))
LA_A = L.mean(axis=2)
R_laA = summarize(LA_A, "log-area, grid 8..32 (sensitivity)")

# 2. capped cost at threshold th
def capped(th):
    M = np.full((24, 8), 32.0)
    for i in range(24):
        for j in range(8):
            idx = np.nonzero(S[i, j] <= th)[0]
            if len(idx): M[i, j] = 8 + idx[0]
    return M
C04 = capped(0.04)
assert (C04 == cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index().values).all()
R_c04 = summarize(C04, "capped cost @0.04")
C02 = capped(0.02)
print("mean capped cost @0.02:", dict(zip(POL, C02.mean(axis=0).round(3))))
R_c02 = summarize(C02, "capped cost @0.02")
C01 = capped(0.01)
print("mean capped cost @0.01:", dict(zip(POL, C01.mean(axis=0).round(3))))
R_c01 = summarize(C01, "capped cost @0.01")

# 3. threshold sweep: at target after init / TR never attains / TR minus floor
print("--- threshold sweep ---")
for th, C in ((0.01, C01), (0.02, C02), (0.04, C04)):
    at = S[:, 0, 0] <= th
    floor = np.where(at, 8, 9)
    tr = C[:, TRI]
    never = int((np.array([ (S[i, TRI] <= th).any() for i in range(24)]) == False).sum())
    print(f"th={th}: at target after init={int(at.sum())}; TR never attains={never}; TR mean minus floor mean={(tr - floor).mean():.4f}; TR total={tr.sum():.0f}; floor total={floor.sum()}")
    env = C.min(axis=1)
    print(f"        envelope total={env.sum():.0f}; TR - envelope per system={(tr - env).mean():.4f}; ever-attain counts by policy={dict(zip(POL, [int(sum((S[i, j] <= th).any() for i in range(24))) for j in range(8)]))}")

# 4. sustained attainment cost: first cost from which the run stays <= 0.04 through its final state; 32 if final state above target
def sustained(th):
    M = np.full((24, 8), 32.0)
    for i, w in enumerate(W):
        for j, p in enumerate(POL):
            g = curves[(curves.world_id == w) & (curves.policy_id == p)]
            ok = (g.normalized_mse.values <= th)
            if not ok[-1]:
                continue
            k = len(ok) - 1
            while k > 0 and ok[k - 1]:
                k -= 1
            M[i, j] = g.total_cost_units.values[k]
    return M
SU = sustained(0.04)
m_su = pd.Series(SU.mean(axis=0), index=POL)
print("--- sustained attainment cost @0.04 ---")
print(m_su.round(3).to_dict())
print("ranking:", " < ".join(f"{k}({v:.2f})" for k, v in m_su.sort_values().items()))
R_su = summarize(SU, "sustained cost @0.04")
# alt sustained on the LOCF grid (same thing unless the definition differs)
# 5. geometric mean final NMSE
fin = cells.pivot(index="world_id", columns="policy_id", values="final_error")[POL].sort_index()
print("--- geometric-mean final NMSE ---")
print((10 ** np.log10(fin).mean()).round(5).to_dict())
print("arith mean final NMSE:", fin.mean().round(5).to_dict())
print("median final NMSE:", fin.median().round(5).to_dict())
# 6. fixed-cost endpoints: log10 NMSE at total cost 12, 16, 20, 24, 32
print("--- fixed-cost endpoints: mean log10 NMSE at given total cost; standardized effect random vs TR ---")
for c in (10, 12, 16, 20, 24, 28, 32):
    M = L[:, :, c - 8]
    d = M[:, 0] - M[:, TRI]
    print(f"cost {c}: mean log10 NMSE by policy {dict(zip(POL, M.mean(axis=0).round(3)))} | random-TR d_std={d.mean()/d.std(ddof=1):.3f}")
print("--- standardized effects random vs TR ---")
print("capped@0.04:", round(R_c04.loc['uniform_random', 'd_std'], 4), " log-area(9..32):", round(R_la.loc['uniform_random', 'd_std'], 4),
      " capped@0.02:", round(R_c02.loc['uniform_random', 'd_std'], 4), " capped@0.01:", round(R_c01.loc['uniform_random', 'd_std'], 4),
      " sustained:", round(R_su.loc['uniform_random', 'd_std'], 4))
print("--- coverage vs TR ---")
for nm, R in (("capped@0.04", R_c04), ("log-area", R_la), ("capped@0.02", R_c02), ("capped@0.01", R_c01), ("sustained", R_su)):
    r = R.loc["menu_coverage"]
    print(f"{nm}: mean {r['mean']:+.4f}, sd {r['sd']:.4f}, sign-flip p {r['signflip_p']:.4f}")
# log-area power planning
def power_paired(delta, sd, n, alpha=0.05):
    df = n - 1; nc = delta / sd * np.sqrt(n); tc = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
def n_needed(delta, sd, power):
    n = 3
    while power_paired(delta, sd, n) < power: n += 1
    return n
sdla = float(np.median(R_la.sd.values))
import warnings; warnings.simplefilter("ignore")
print("--- log-area planning, SD =", round(sdla, 4), "---")
for frac in (0.9, 0.8, 0.7, 0.5):
    delta = abs(np.log10(frac))
    print(f"{int((1-frac)*100)}% lower NMSE (delta {delta:.4f}): 80% -> {n_needed(delta, sdla, .8)}, 90% -> {n_needed(delta, sdla, .9)}")
for nm, R in (("la", R_la), ("c02", R_c02), ("su", R_su)):
    R.to_csv(OUT + f"out_vsTR_{nm}.csv")
np.save(OUT + "out_capped02.npy", C02); np.save(OUT + "out_sustained.npy", SU); np.save(OUT + "out_logarea.npy", LA)
