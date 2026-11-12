"""Verifier step 18: is the near-side/far-side split specific to the always-X=-1 policy, or do those 7 systems separate every weak arm from TR?
Exact permutation over C(24,7) label assignments for several contrasts (log-area, grid 9..32)."""
import numpy as np, pandas as pd
from itertools import combinations
from math import comb
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
J = {p: i for i, p in enumerate(POL)}
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
LA = np.load(OUT + "out_logarea.npy")
near = (meta.mux0 < 0).values
combos = np.array(list(combinations(range(24), 7)))
def perm(v):
    obs = v[near].mean() - v[~near].mean()
    s = v[combos].sum(axis=1)
    stat = s / 7 - (v.sum() - s) / 17
    return obs, float((np.abs(stat) >= abs(obs) - 1e-12).mean())
print("contrast (log-area) | near(7) mean | other(17) mean | near-minus-other | exact two-sided permutation p")
for a, b in [("dpo_llm", "target_risk"), ("sft_llm", "target_risk"), ("frozen_llm", "target_risk"), ("uniform_random", "target_risk"), ("learned_ranker", "target_risk"),
             ("menu_coverage", "target_risk"), ("local_information", "target_risk"),
             ("dpo_llm", "frozen_llm"), ("dpo_llm", "uniform_random"), ("dpo_llm", "menu_coverage"), ("dpo_llm", "local_information"), ("sft_llm", "frozen_llm"),
             ("dpo_llm", "learned_ranker")]:
    v = LA[:, J[a]] - LA[:, J[b]]
    obs, p = perm(v)
    print(f"{a:18s} - {b:18s} | {v[near].mean():+.3f} | {v[~near].mean():+.3f} | {obs:+.3f} | {p:.4f}")
# TR's own level on near vs other systems
tr = LA[:, J["target_risk"]]
print("TR mean log-area: near", round(tr[near].mean(), 3), "other", round(tr[~near].mean(), 3))
for p in POL:
    print(f"   {p:18s} mean log-area near {LA[near, J[p]].mean():+.3f} other {LA[~near, J[p]].mean():+.3f}")
