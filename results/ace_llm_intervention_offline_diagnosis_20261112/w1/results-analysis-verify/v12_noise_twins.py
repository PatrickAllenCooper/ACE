"""Verifier step 12: replicate noise. (a) do arms share measurement noise? (b) dpo/sft identical-prefix twins. (c) selection in the envelope.
(g) non-monotone error. (h) sft stops. Fresh code."""
import pandas as pd, numpy as np, itertools, pickle
from scipy import stats
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
PC = [f"prediction_{i}" for i in range(8)]
W = sorted(cells.world_id.unique())
sh = lambda w: f"w{int(str(w)[-2:]):02d}"
runs = {(w, p): g.reset_index(drop=True) for (w, p), g in curves.groupby(["world_id", "policy_id"])}
S = np.load(OUT + "out_step_nmse.npy")
init_at = S[:, 0, 0] <= 0.04

# (a) same-first-action pairs
npairs = nident = 0
nident_nmse = 0
for w in W:
    for a, b in itertools.combinations(POL, 2):
        ra, rb = runs[(w, a)], runs[(w, b)]
        if ra.action_id[1] == rb.action_id[1]:
            npairs += 1
            same = np.allclose(ra.loc[1, PC].values.astype(float), rb.loc[1, PC].values.astype(float), rtol=0, atol=1e-12)
            nident += same
            nident_nmse += abs(ra.normalized_mse[1] - rb.normalized_mse[1]) < 1e-15
print(f"(a) same-first-action policy pairs: {npairs}; with identical post-action predictions: {nident}; identical NMSE: {nident_nmse}")
# sanity: the init predictions ARE identical across arms
print("    init predictions identical across arms within world:", all(np.allclose(runs[(w, a)].loc[0, PC].values.astype(float), runs[(w, POL[0])].loc[0, PC].values.astype(float), atol=0) for w in W for a in POL))

# (b) dpo / sft twin prefix
pref = []
for w in W:
    a = runs[(w, "dpo_llm")].action_id.tolist()[1:]
    b = runs[(w, "sft_llm")].action_id.tolist()[1:]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k] == "set_x_neg":
        k += 1
    pref.append(k)
print("(b) length of common set_x_neg prefix dpo/sft by system:", dict(zip([sh(w) for w in W], pref)))
print("    min prefix:", min(pref), " max:", max(pref), " n systems with prefix>=10:", sum(k >= 10 for k in pref))
J = {p: POL.index(p) for p in POL}
def capped_at(cap, th=0.04):
    M = np.full((24, 8), float(cap))
    for i in range(24):
        for j in range(8):
            idx = np.nonzero(S[i, j, :cap - 8 + 1] <= th)[0]   # states with total cost <= cap
            if len(idx): M[i, j] = 8 + idx[0]
    return M
for cap in (18,):
    C = capped_at(cap)
    d = C[:, J["dpo_llm"]] - C[:, J["sft_llm"]]
    print(f"    cap {cap}: dpo vs sft capped costs differ on {int((d[~init_at] != 0).sum())} of {int((~init_at).sum())} informative systems; diffs (informative): {d[~init_at].astype(int).tolist()}")
    print(f"    cap {cap}: mean |diff|/2 over 24 systems (best-of-two gain over the mean of two): {np.abs(d).mean()/2:.4f}; SD of diff over 24: {d.std(ddof=1):.3f}; over 14 informative: {d[~init_at].std(ddof=1):.3f}")
    print(f"    cap {cap}: best-of-two gain over ONE fixed run (mean(dpo) - mean(min)): {(C[:, J['dpo_llm']] - np.minimum(C[:, J['dpo_llm']], C[:, J['sft_llm']])).mean():.4f} ; (sft - min): {(C[:, J['sft_llm']] - np.minimum(C[:, J['dpo_llm']], C[:, J['sft_llm']])).mean():.4f}")
    env = C.min(axis=1)
    print(f"    cap {cap}: TR mean {C[:, J['target_risk']].mean():.4f}; envelope mean {env.mean():.4f}; envelope gain over TR per system {(C[:, J['target_risk']] - env).mean():.4f}")
    print(f"    cap {cap}: per-policy mean capped cost: {dict(zip(POL, C.mean(axis=0).round(3)))}")
L = np.log10(S)
dl = L[:, J["dpo_llm"], 1:11] - L[:, J["sft_llm"], 1:11]   # total cost 9..18
print("    SD across 24 systems of log10 NMSE(dpo) - log10 NMSE(sft) at total cost 9..18:", np.round(dl.std(axis=0, ddof=1), 3).tolist())
print("    mean of that difference at cost 9..18:", np.round(dl.mean(axis=0), 3).tolist())
print("    implied single-run replicate SD of log10 NMSE (diff SD / sqrt2):", np.round(dl.std(axis=0, ddof=1) / np.sqrt(2), 3).tolist())
# replicate noise in the capped cost itself (cap 18): variance of single-run capped cost from twins = var(diff)/2
C18 = capped_at(18)
d18 = C18[:, J["dpo_llm"]] - C18[:, J["sft_llm"]]
print(f"    twin-based replicate SD of capped cost at cap 18 (sd(diff)/sqrt2): {d18.std(ddof=1)/np.sqrt(2):.3f}; between-system SD of twin mean: {((C18[:, J['dpo_llm']] + C18[:, J['sft_llm']])/2).std(ddof=1):.3f}")
# full prefix-length twins at the full cap: how long are dpo/sft identical in action? (sft deviates late) -> compare cap 32 too but those are not pure twins
# (g) non-monotone error
rise = tot = 0
left = cross = 0
left_at = runs_at = 0
for (w, p), g in runs.items():
    e = g.normalized_mse.values
    rise += int((np.diff(e) > 0).sum()); tot += len(e) - 1
    ok = e <= 0.04
    if ok.any():
        cross += 1
        k = int(np.argmax(ok))
        if (~ok[k:]).any():
            left += 1
            if init_at[W.index(w)]: left_at += 1
    if init_at[W.index(w)]: runs_at += 1
print(f"(g) error rose after {rise} of {tot} acquisitions ({100*rise/tot:.2f}%); crossing runs {cross}; later above target at least once: {left}; on at-target systems: {left_at} of {runs_at}")
# also: ended above target after crossing
ended_above = sum(1 for (w, p), g in runs.items() if (g.normalized_mse.values <= 0.04).any() and g.normalized_mse.values[-1] > 0.04)
print(f"    crossing runs whose FINAL state is above target: {ended_above}")
byp = {p: (sum(1 for w in W if (runs[(w, p)].normalized_mse.values <= .04).any() and (runs[(w, p)].normalized_mse.values[int(np.argmax(runs[(w, p)].normalized_mse.values <= .04)):] > .04).any())) for p in POL}
print("    crossing-then-left by policy:", byp)
risep = {p: round(np.mean(np.concatenate([np.diff(runs[(w, p)].normalized_mse.values) > 0 for w in W])), 3) for p in POL}
print("    share of acquisitions after which error rose, by policy:", risep)

# (h) sft stops
c = cells[cells.policy_id == "sft_llm"].set_index("world_id").sort_index()
st = c[c.termination_reason == "policy_stop"]
nact = pd.Series({w: len(runs[(w, "sft_llm")]) - 1 for w in W})
print(f"(h) sft policy stops: {len(st)} systems; non-stop system: {[sh(w) for w in c.index if c.loc[w, 'termination_reason'] != 'policy_stop']}")
print("    actions at stop: min", nact[st.index].min(), "max", nact[st.index].max(), "| total cost at stop: min", st.actual_total_cost.min(), "max", st.actual_total_cost.max(), "| unspent total:", int((32 - c.actual_total_cost).sum()))
print("    distribution of stop total cost:", st.actual_total_cost.value_counts().sort_index().to_dict(), " actions:", nact[st.index].value_counts().sort_index().to_dict())
ab = st[st.final_error > 0.04]
print("    stops above target:", [(sh(w), round(e, 4)) for w, e in ab.final_error.items()])
print("    corr(stop total cost, log10 final error) over the 23 stops: pearson", round(np.corrcoef(st.actual_total_cost, np.log10(st.final_error))[0, 1], 4),
      " spearman", round(stats.spearmanr(st.actual_total_cost, np.log10(st.final_error))[0], 4),
      " | corr(n actions, log10 err):", round(np.corrcoef(nact[st.index], np.log10(st.final_error))[0, 1], 4),
      " | over all 24:", round(np.corrcoef(c.actual_total_cost, np.log10(c.final_error))[0, 1], 4))
print("    sft action mix over positions (last actions before stop):", pd.Series([runs[(w, 'sft_llm')].action_id.iloc[-1] for w in st.index]).value_counts().to_dict())
print("    per-system sft: ", [(sh(w), int(c.loc[w, 'actual_total_cost']), int(nact[w]), c.loc[w, 'cost_stratum'][:3]) for w in W])
