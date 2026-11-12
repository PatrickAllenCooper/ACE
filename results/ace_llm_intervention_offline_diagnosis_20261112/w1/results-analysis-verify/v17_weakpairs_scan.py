"""Verifier step 17: scan fixed-cost endpoints (log10 NMSE at each total cost 9..32) for the 10 pairs among the five weak arms."""
import numpy as np, itertools
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
J = {p: i for i, p in enumerate(POL)}
L = np.log10(np.load(OUT + "out_step_nmse.npy"))
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
tot = below = 0
for c in range(9, 33):
    hits = []
    for a, b in itertools.combinations(weak, 2):
        p = signflip(L[:, J[a], c - 8] - L[:, J[b], c - 8]); tot += 1
        if p < 0.05: below += 1; hits.append((a, b, round(p, 4)))
    if hits: print("cost", c, hits)
print(f"fixed-cost scan: {below} of {tot} tests below 0.05 (expected by chance at 5%: {0.05*tot:.1f})")
