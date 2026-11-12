"""Verifier step 13: dpo minus TR by stratum; near-side/far-side split of dpo's log-area deficit with a permutation test;
pairwise tests among the five weak arms. Fresh code."""
import pandas as pd, numpy as np, itertools
from math import comb
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
J = {p: i for i, p in enumerate(POL)}
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
W = meta.index.tolist()
S = np.load(OUT + "out_step_nmse.npy"); L = np.log10(S)
LA = L[:, :, 1:].mean(axis=2)
C = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index().values.astype(float)
C02 = np.load(OUT + "out_capped02.npy"); SU = np.load(OUT + "out_sustained.npy")
d = pd.Series(C[:, J["dpo_llm"]] - C[:, J["target_risk"]], index=W)
print("dpo minus TR capped cost by stratum:")
print("  graph:", d.groupby(meta.graph_stratum).mean().round(3).to_dict())
print("  regime:", d.groupby(meta.regime).mean().round(3).to_dict())
print("  cost menu:", d.groupby(meta.cost_stratum).mean().round(3).to_dict())
# cross-check against strata.csv
st = pd.read_csv(D + "strata.csv")
pv = st.pivot(index="stratum", columns="policy_id", values="mean_capped_total_cost")
print("  strata.csv dpo - TR:", (pv.dpo_llm - pv.target_risk).round(3).to_dict())
print("  n strata rows:", len(st), " n distinct strata:", st.stratum.nunique())

# near side split
dl = pd.Series(LA[:, J["dpo_llm"]] - LA[:, J["target_risk"]], index=W)
near = meta.mux0 < 0     # X=-1 is nearer to the estimated natural X mean
print(f"\nX=-1 near side systems: {int(near.sum())}; dpo-TR log-area: near {dl[near].mean():+.4f}, other {dl[~near].mean():+.4f}; difference {dl[near].mean() - dl[~near].mean():+.4f}")
# exact permutation test: all C(24,7) = 346104 subsets
vals = dl.values
obs = dl[near].mean() - dl[~near].mean()
tot = vals.sum()
cnt = 0; n = 0
idx = np.arange(24)
# enumerate combos via numpy in chunks
from itertools import combinations
arr = np.fromiter((sum(vals[list(c)]) for c in combinations(range(24), 7)), dtype=float, count=comb(24, 7))
stat = arr / 7 - (tot - arr) / 17
p_two = (np.abs(stat) >= abs(obs) - 1e-12).mean()
p_one = (stat >= obs - 1e-12).mean()
print(f"exact permutation over C(24,7)={len(arr)} label assignments: two-sided p = {p_two:.5f}; one-sided p = {p_one:.5f}")
# same split for capped cost and for other always-X policies
for nm, M in (("capped cost@0.04", C), ("capped cost@0.02", C02), ("log-area", LA)):
    for p in ("dpo_llm", "sft_llm", "learned_ranker", "frozen_llm", "uniform_random", "menu_coverage", "local_information"):
        dd = pd.Series(M[:, J[p]] - M[:, J["target_risk"]], index=W)
        print(f"   {nm:18s} {p:18s} minus TR: near(7) {dd[near].mean():+.3f}   other(17) {dd[~near].mean():+.3f}")
# confound check: composition of the near group
print("near-side systems composition:", meta[near][["graph_stratum", "regime", "cost_stratum"]].value_counts().to_dict())
print("near-side: at target after init:", int((S[near.values, 0, 0] <= 0.04).sum()), "of 7; other:", int((S[~near.values, 0, 0] <= 0.04).sum()), "of 17")
# ranker picks x_pos or x_neg per system: does the ranker's choice follow the far side?
import pickle
seqs = pickle.load(open(OUT + "out_seqs.pkl", "rb"))
rk = {w: pd.Series(seqs[(w, "learned_ranker")]).value_counts().to_dict() for w in W}
far = np.where(meta.mux0 > 0, "set_x_neg", "set_x_pos")
agree = sum(max(rk[w], key=rk[w].get) == f for w, f in zip(W, far))
print("ranker majority action equals the far X extreme in", agree, "of 24 systems; ranker per-system action sets:", collections.Counter(tuple(sorted(v.items())) for v in rk.values()).most_common(6) if (collections := __import__("collections")) else None)

# weak-arm pairwise tests
def signflip(dv):
    dv = np.asarray(dv, float); dv = dv[dv != 0]
    if len(dv) == 0: return 1.0
    obs = abs(dv.sum()); h = len(dv) // 2
    def allsums(v):
        s = np.array([0.0])
        for x in v: s = np.concatenate([s + x, s - x])
        return s
    a, b = allsums(dv[:h]), np.sort(allsums(dv[h:]))
    tol = 1e-12 * max(1.0, obs)
    hi = len(b) - np.searchsorted(b, (obs - tol) - a, side="left"); lo = np.searchsorted(b, -(obs - tol) - a, side="right")
    return float(hi.sum() + lo.sum()) / (len(a) * len(b))
weak = ["uniform_random", "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm"]
fin = np.log10(cells.pivot(index="world_id", columns="policy_id", values="final_error")[POL].sort_index().values)
C01 = np.full((24, 8), 32.0)
for i in range(24):
    for j in range(8):
        k = np.nonzero(S[i, j] <= 0.01)[0]
        if len(k): C01[i, j] = 8 + k[0]
endpoints = {"capped@0.04": C, "capped@0.02": C02, "capped@0.01": C01, "sustained@0.04": SU, "log-area": LA, "log10 final NMSE": fin,
             "log10 NMSE at cost 16": L[:, :, 8], "log10 NMSE at cost 24": L[:, :, 16], "raw area": S[:, :, 1:].mean(axis=2)}
print("\npairwise sign-flip tests among random, ranker, frozen, sft, dpo (10 pairs per endpoint):")
allp = []
for nm, M in endpoints.items():
    ps = {}
    for a, b in itertools.combinations(weak, 2):
        ps[(a, b)] = signflip(M[:, J[a]] - M[:, J[b]])
    below = {k: round(v, 4) for k, v in ps.items() if v < 0.05}
    allp.append((nm, ps))
    print(f"  {nm:22s} min p {min(ps.values()):.4f}; n<0.05: {len(below)} {below}")
first7 = sum(sum(v < 0.05 for v in ps.values()) for nm, ps in allp[:7])
print("total below 0.05 over the first 7 endpoints (70 tests):", first7, "| over all 9 endpoints (90 tests):", sum(sum(v < 0.05 for v in ps.values()) for nm, ps in allp))
# TR vs each (which are separated) across endpoints
print("\nTR vs each policy: sign-flip p by endpoint")
tab = pd.DataFrame({nm: {p: signflip(M[:, J[p]] - M[:, J['target_risk']]) for p in POL if p != 'target_risk'} for nm, M in endpoints.items()})
pd.set_option("display.width", 250)
print(tab.round(4).to_string())
