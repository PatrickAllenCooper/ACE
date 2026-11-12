"""(4) state dependence of preference labels and (5) label noise. Offline only.

Inputs are scratch pickles written by my own scripts in this session (trusted).
"""
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

D = pickle.load(open(sys.argv[1], "rb"))
P = pd.read_pickle(sys.argv[2])
ACTIONS = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos", "set_y_neg", "set_y_zero", "set_y_pos", "set_xy"]
IDX = {a: i for i, a in enumerate(ACTIONS)}
strict = P[~P.tie].copy()
sel = P[P.status == "selected"].copy()

print("=" * 100)
print("(4) STATE DEPENDENCE OF PREFERENCE LABELS")
print("=" * 100)
rows = []
for a, b in combinations(ACTIONS, 2):
    sub = P[((P.left == a) & (P.right == b)) | ((P.left == b) & (P.right == a))]
    s = sub[~sub.tie]
    wa, wb = int((s.chosen == a).sum()), int((s.chosen == b).sum())
    n = len(sub)
    maj = max(wa, wb)
    # two-sided exact binomial vs 0.5 among strict
    pv = stats.binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else float("nan")
    rows.append({"pair": f"{a} vs {b}", "states": n, "tie": n - wa - wb, "first_wins": wa, "second_wins": wb,
                 "first_win_frac_of_states": round(wa / n, 4), "second_win_frac_of_states": round(wb / n, 4),
                 "first_share_of_strict": round(wa / (wa + wb), 3) if wa + wb else None,
                 "majority_share_of_strict": round(maj / (wa + wb), 3) if wa + wb else None,
                 "binom_p_vs_half": round(pv, 4)})
T = pd.DataFrame(rows)
print(T.to_string())
tot_strict = int(T.first_wins.sum() + T.second_wins.sum())
tot_major = int(np.maximum(T.first_wins, T.second_wins).sum())
print(f"\nPer-pair majority direction (28 free parameters, in-sample upper bound for ANY state-independent pairwise rule): "
      f"{tot_major}/{tot_strict} = {tot_major / tot_strict:.4f}")
print("pairs whose strict labels deviate from 50/50 at p<0.05 (uncorrected):", int((T.binom_p_vs_half < 0.05).sum()), "of 28;",
      "Bonferroni p<0.05/28:", int((T.binom_p_vs_half < 0.05 / 28).sum()))
print("largest majority share among the 28 pairs:", T.majority_share_of_strict.max(), " smallest:", T.majority_share_of_strict.min())


def fit_bt(pairs, n_iter=5000, ridge=1e-3):
    """Bradley-Terry by MM with a tiny symmetric prior (ridge pseudo-wins) for identifiability."""
    W = np.zeros((8, 8))
    for c, r in pairs:
        W[IDX[c], IDX[r]] += 1
    W = W + ridge * (1 - np.eye(8))
    p = np.ones(8)
    for _ in range(n_iter):
        wins = W.sum(axis=1)
        denom = np.zeros(8)
        for i in range(8):
            for j in range(8):
                if i != j:
                    denom[i] += (W[i, j] + W[j, i]) / (p[i] + p[j])
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-12:
            p = new
            break
        p = new
    return np.log(p)


def acc(strength, pairs):
    k = sum(1 for c, r in pairs if strength[IDX[c]] > strength[IDX[r]])
    t = sum(1 for c, r in pairs if strength[IDX[c]] == strength[IDX[r]])
    return k, t, len(pairs)


all_pairs = list(zip(strict.chosen, strict.rejected))
bt = fit_bt(all_pairs)
print("\nBradley-Terry log-strengths fitted on all 687 strict pairs (geometric mean = 0):")
order = np.argsort(-bt)
for i in order:
    print(f"   {ACTIONS[i]:12s} {bt[i]:+.4f}")
k, t, n = acc(bt, all_pairs)
print(f"constant BT ranking predicts strict labels IN-SAMPLE: {k}/{n} = {k / n:.4f}")
# simple win-rate ranking
wr = np.array([(strict.chosen == a).sum() / ((strict.chosen == a).sum() + (strict.rejected == a).sum()) for a in ACTIONS])
k, t, n = acc(wr, all_pairs)
print(f"constant win-rate ranking predicts strict labels IN-SAMPLE: {k}/{n} = {k / n:.4f}")
print("win-rate order:", [(ACTIONS[i], round(float(wr[i]), 4)) for i in np.argsort(-wr)])

# leave-one-world-out
k_tot = n_tot = 0
k_wr = 0
for w in sorted(strict.world_index.unique()):
    tr, te = strict[strict.world_index != w], strict[strict.world_index == w]
    b = fit_bt(list(zip(tr.chosen, tr.rejected)))
    kk, _, nn = acc(b, list(zip(te.chosen, te.rejected)))
    k_tot += kk
    n_tot += nn
    wrr = np.array([(tr.chosen == a).sum() / max(1, ((tr.chosen == a).sum() + (tr.rejected == a).sum())) for a in ACTIONS])
    k_wr += acc(wrr, list(zip(te.chosen, te.rejected)))[0]
print(f"constant BT ranking, LEAVE-ONE-WORLD-OUT: {k_tot}/{n_tot} = {k_tot / n_tot:.4f}")
print(f"constant win-rate ranking, LEAVE-ONE-WORLD-OUT: {k_wr}/{n_tot} = {k_wr / n_tot:.4f}")

# permutation null: how well does a BT ranking fit pure coin-flip directions on the same pair structure (in-sample)?
rng = np.random.default_rng(20261112)
null_acc = []
arr = np.array(all_pairs)
for _ in range(300):
    flip = rng.random(len(arr)) < 0.5
    perm = [(r, c) if f else (c, r) for (c, r), f in zip(all_pairs, flip)]
    b = fit_bt(perm, n_iter=400)
    kk, _, nn = acc(b, perm)
    null_acc.append(kk / nn)
print(f"in-sample BT accuracy when directions are coin flips (300 draws): mean {np.mean(null_acc):.4f}, "
      f"95th pct {np.percentile(null_acc, 95):.4f}, max {np.max(null_acc):.4f}")

# selected 119
sp = list(zip(sel.chosen, sel.rejected))
k, t, n = acc(bt, sp)
print(f"\nBT (fit on all 687) applied to the 119 SELECTED DPO pairs: {k}/{n} = {k / n:.4f}")
bts = fit_bt(sp)
k, t, n = acc(bts, sp)
print(f"BT fit on the 119 selected pairs themselves, in-sample: {k}/{n} = {k / n:.4f}")
print("   strengths (selected-only fit):", [(ACTIONS[i], round(float(bts[i]), 3)) for i in np.argsort(-bts)])
k_tot = n_tot = 0
for w in sorted(sel.world_index.unique()):
    tr, te = sel[sel.world_index != w], sel[sel.world_index == w]
    b = fit_bt(list(zip(tr.chosen, tr.rejected)))
    kk, _, nn = acc(b, list(zip(te.chosen, te.rejected)))
    k_tot += kk
    n_tot += nn
print(f"BT fit on selected pairs, LEAVE-ONE-WORLD-OUT over the 119: {k_tot}/{n_tot} = {k_tot / n_tot:.4f}")
null_acc = []
for _ in range(300):
    flip = rng.random(len(sp)) < 0.5
    perm = [(r, c) if f else (c, r) for (c, r), f in zip(sp, flip)]
    b = fit_bt(perm, n_iter=400)
    kk, _, nn = acc(b, perm)
    null_acc.append(kk / nn)
print(f"   coin-flip null for 119-pair in-sample BT accuracy: mean {np.mean(null_acc):.4f}, 95th pct {np.percentile(null_acc, 95):.4f}")

# A policy that always says "set_x_neg preferred when present, else coin flip"
has = sel[(sel.chosen == "set_x_neg") | (sel.rejected == "set_x_neg")]
print(f"\nselected pairs involving set_x_neg: {len(has)}; set_x_neg preferred in {int((has.chosen == 'set_x_neg').sum())}, rejected in {int((has.rejected == 'set_x_neg').sum())}")
hasA = strict[(strict.chosen == "set_x_neg") | (strict.rejected == "set_x_neg")]
print(f"all strict pairs involving set_x_neg: {len(hasA)}; preferred {int((hasA.chosen == 'set_x_neg').sum())}, rejected {int((hasA.rejected == 'set_x_neg').sum())}; "
      f"binomial p vs 0.5 = {stats.binomtest(int((hasA.chosen == 'set_x_neg').sum()), len(hasA), 0.5).pvalue:.4f}")
hasP = strict[(strict.chosen == "set_x_pos") | (strict.rejected == "set_x_pos")]
print(f"all strict pairs involving set_x_pos: {len(hasP)}; preferred {int((hasP.chosen == 'set_x_pos').sum())}, rejected {int((hasP.rejected == 'set_x_pos').sum())}; "
      f"binomial p vs 0.5 = {stats.binomtest(int((hasP.chosen == 'set_x_pos').sum()), len(hasP), 0.5).pvalue:.6f}")

print("\nStratified constant rankings (does a ranking that depends only on graph x cost menu fit better?)")
for keys in (["graph"], ["cost_menu"], ["graph", "cost_menu"], ["snapshot"], ["regime"]):
    k_in = n_in = 0
    for _, g in strict.groupby(keys):
        prs = list(zip(g.chosen, g.rejected))
        b = fit_bt(prs)
        kk, _, nn = acc(b, prs)
        k_in += kk
        n_in += nn
    # LOWO
    k_out = n_out = 0
    for w in sorted(strict.world_index.unique()):
        te = strict[strict.world_index == w]
        tr = strict[strict.world_index != w]
        for kv, gte in te.groupby(keys):
            mask = np.ones(len(tr), bool)
            kvt = kv if isinstance(kv, tuple) else (kv,)
            for kname, v in zip(keys, kvt):
                mask &= (tr[kname] == v).values
            gtr = tr[mask]
            if len(gtr) == 0:
                gtr = tr
            b = fit_bt(list(zip(gtr.chosen, gtr.rejected)))
            kk, _, nn = acc(b, list(zip(gte.chosen, gte.rejected)))
            k_out += kk
            n_out += nn
    print(f"  cells={'+'.join(keys):18s} in-sample {k_in}/{n_in} = {k_in / n_in:.4f}   leave-one-world-out {k_out}/{n_out} = {k_out / n_out:.4f}")

print("\nMeasured-best action per record (argmax of 4-replicate mean gain per cost), 128 records:")
best, teach_match, ranks = [], 0, []
recs = [r for r in D["records"] if r["feedback"]]
for r in recs:
    mg = r["feedback"]["mean_gains_per_cost"]
    b = max(mg, key=lambda a: mg[a])
    best.append(b)
    ta = r["supervised_action_id"]
    teach_match += int(b == ta)
    ranks.append(1 + sum(mg[a] > mg[ta] for a in mg))
bc = Counter(best)
print({a: bc.get(a, 0) for a in ACTIONS})
print("teacher (supervised) action == measured-best action:", teach_match, "/ 128  (chance 1/8 = 16)")
print("rank of supervised action by measured gain (1=best..8): mean", round(float(np.mean(ranks)), 3), " distribution", dict(sorted(Counter(ranks).items())))
by_t = defaultdict(list)
for r, rk in zip(recs, ranks):
    by_t[r["teacher_method"]].append(rk)
for t, v in by_t.items():
    print(f"   teacher {t:18s} mean rank {np.mean(v):.3f}  top1 {sum(x == 1 for x in v)}/{len(v)}  top3 {sum(x <= 3 for x in v)}/{len(v)}")
print("expected mean rank under no relationship: 4.5")
mean_neg = [sum(r["feedback"]["mean_gains_per_cost"][a] < 0 for a in ACTIONS) for r in recs]
print("number of candidates (of 8) with NEGATIVE measured mean gain per record: mean", round(float(np.mean(mean_neg)), 3),
      " total negative candidate-means:", int(np.sum(mean_neg)), "/", 8 * len(recs))

print()
print("=" * 100)
print("(5) LABEL NOISE")
print("=" * 100)
# t statistics with 3 df
P["t"] = P.mean_gain_difference / P.paired_standard_error
P["p"] = 2 * stats.t.sf(P.t.abs(), df=3)
alpha_rule = 2 * stats.t.sf(2.0, df=3)
print(f"Tie rule |mean| <= 2*SE with R=4 replicates is a paired t-test with df=3 at two-sided alpha = P(|t3|>2) = {alpha_rule:.6f}")
print(f"Expected strict labels if EVERY comparison were pure noise (Gaussian null): {alpha_rule * len(P):.1f} of {len(P)}; observed strict = {int((~P.tie).sum())}")
print(f"Excess over the global null: {int((~P.tie).sum()) - alpha_rule * len(P):.1f} pairs = {(int((~P.tie).sum()) - alpha_rule * len(P)) / int((~P.tie).sum()):.4f} of strict labels")
for lam in (0.5, 0.6, 0.7, 0.8, 0.9):
    pi0 = (P.p > lam).mean() / (1 - lam)
    print(f"  Storey pi0 estimate at lambda={lam}: {pi0:.4f}  -> est. false-discovery share among strict labels = {min(1, pi0 * alpha_rule * len(P) / (~P.tie).sum()):.4f}")
# p-value histogram
hist, edges = np.histogram(P.p, bins=10, range=(0, 1))
print("p-value histogram (10 bins 0..1):", hist.tolist(), " (uniform would be 358.4 per bin)")
# conventional thresholds
for a in (0.05, 0.01, 0.001):
    tc = stats.t.isf(a / 2, df=3)
    print(f"  two-sided p<{a} (|t3|>{tc:.3f}): {int((P.p < a).sum())}/{len(P)} = {(P.p < a).mean():.4f} (null expectation {a * len(P):.1f}); "
          f"selected DPO pairs passing: {int((P[P.status == 'selected'].p < a).sum())}/119")
# BH
ps = np.sort(P.p.values)
m = len(ps)
for q in (0.05, 0.10, 0.20):
    thresh = q * np.arange(1, m + 1) / m
    passed = np.where(ps <= thresh)[0]
    nrej = passed.max() + 1 if len(passed) else 0
    cut = ps[nrej - 1] if nrej else 0
    nsel = int((P[P.status == "selected"].p <= cut).sum()) if nrej else 0
    print(f"  Benjamini-Hochberg q={q}: {nrej}/{m} comparisons survive (p cutoff {cut:.2e}); of the 119 selected DPO pairs: {nsel}")
print("  Bonferroni 0.05/3584:", int((P.p < 0.05 / m).sum()))

# variance components per record: two-way layout candidates x replicates (no interaction replication)
print("\nVariance components of per-replicate gains G[c,r] (8 candidates x 4 replicates) per record:")
rel, Fp, sig_c, sig_e, icc1 = [], [], [], [], []
split_corr = []
for r in recs:
    G = np.array([r["feedback"]["per_replicate_gains_per_cost"][a] for a in ACTIONS])  # 8 x 4
    gm = G.mean()
    ss_c = 4 * ((G.mean(axis=1) - gm) ** 2).sum()
    ss_w = ((G - G.mean(axis=1, keepdims=True)) ** 2).sum()
    ms_c, ms_w = ss_c / 7, ss_w / 24
    F = ms_c / ms_w
    Fp.append(stats.f.sf(F, 7, 24))
    s2c = max(0.0, (ms_c - ms_w) / 4)
    sig_c.append(s2c)
    sig_e.append(ms_w)
    rel.append(s2c / (s2c + ms_w / 4) if (s2c + ms_w / 4) > 0 else 0.0)
    icc1.append(s2c / (s2c + ms_w) if (s2c + ms_w) > 0 else 0.0)
    cs = []
    for h1 in ((0, 1), (0, 2), (0, 3)):
        h2 = tuple(x for x in range(4) if x not in h1)
        a_, b_ = G[:, h1].mean(axis=1), G[:, h2].mean(axis=1)
        cs.append(np.corrcoef(a_, b_)[0, 1])
    split_corr.append(np.mean(cs))
rel, Fp, split_corr = np.array(rel), np.array(Fp), np.array(split_corr)
print(f"  one-way ANOVA across the 8 candidates (F 7,24): records with p<0.05: {int((Fp < 0.05).sum())}/128, p<0.01: {int((Fp < 0.01).sum())}/128, p<0.001: {int((Fp < 0.001).sum())}/128 (null expectation 6.4 / 1.28 / 0.13)")
print(f"  reliability of the 4-replicate candidate mean (signal var / total var of the mean): median {np.median(rel):.4f}, mean {np.mean(rel):.4f}, "
      f"records with reliability == 0 (MS_between <= MS_within): {int((rel == 0).sum())}/128, reliability > 0.5: {int((rel > 0.5).sum())}/128, > 0.8: {int((rel > 0.8).sum())}/128")
print(f"  single-replicate ICC (signal share of one measured gain): median {np.median(icc1):.4f}, mean {np.mean(icc1):.4f}")
print(f"  pooled: sum(signal var)/(sum(signal var)+sum(MS_within)/4) = {np.sum(sig_c) / (np.sum(sig_c) + np.sum(sig_e) / 4):.4f}  (scale-weighted; dominated by high-variance records)")
print(f"  split-half correlation of candidate means (2 vs 2 replicates, mean of the 3 splits) across records: mean {np.nanmean(split_corr):.4f}, median {np.nanmedian(split_corr):.4f}; "
      f"Spearman-Brown reliability of the 4-replicate mean from the mean split-half r: {2 * np.nanmean(split_corr) / (1 + np.nanmean(split_corr)):.4f}")
print(f"  records with negative split-half correlation: {int((split_corr < 0).sum())}/128")

# ratio of replicate SD to margin
print("\nReplicate-to-replicate variability vs margin size:")
sd_diff = P.replicate_differences.apply(lambda d: np.std(d, ddof=1))
ratio = sd_diff / P.mean_gain_difference.abs()
print("  per-comparison SD of replicate differences / |mean difference| quantiles (all 3584):",
      {k: round(float(v), 3) for k, v in ratio.quantile([.05, .25, .5, .75, .95]).items()})
print("  same, strict only:", {k: round(float(v), 3) for k, v in ratio[~P.tie].quantile([.05, .25, .5, .75, .95]).items()})
print("  same, selected 119:", {k: round(float(v), 3) for k, v in ratio[P.status == 'selected'].quantile([.05, .25, .5, .75, .95]).items()})
# sign consistency of replicate differences
signs = P.replicate_differences.apply(lambda d: int(sum(np.sign(x) == np.sign(np.mean(d)) for x in d)))
print("  number of the 4 replicates agreeing in sign with the mean: all pairs", dict(sorted(Counter(signs).items())),
      " strict", dict(sorted(Counter(signs[~P.tie]).items())), " selected", dict(sorted(Counter(signs[P.status == 'selected']).items())))
print("  (under pure noise, all 4 agree with prob 2*(1/2)^4 = 0.125; strict labels REQUIRE near-unanimity by construction)")

# leave-one-replicate-out label stability for strict pairs: does the sign of mean of other 3 replicates predict held-out replicate sign
agree = tot = 0
for d in P[~P.tie].replicate_differences:
    d = np.array(d)
    for i in range(4):
        rest = np.delete(d, i).mean()
        agree += int(np.sign(rest) == np.sign(d[i]))
        tot += 1
print(f"  strict pairs: held-out replicate sign matches the other-3 mean sign in {agree}/{tot} = {agree / tot:.4f} (selection-biased upward)")
agree = tot = 0
for d in P.replicate_differences:
    d = np.array(d)
    for i in range(4):
        rest = np.delete(d, i).mean()
        agree += int(np.sign(rest) == np.sign(d[i]))
        tot += 1
print(f"  ALL pairs (no selection): held-out replicate sign matches other-3 mean sign in {agree}/{tot} = {agree / tot:.4f} (0.5 = pure noise)")

# Split-half label reproducibility with the SAME rule family: label from replicates (a,b) direction vs (c,d) direction for all pairs
agree = tot = 0
for d in P.replicate_differences:
    d = np.array(d)
    for h1 in ((0, 1), (0, 2), (0, 3)):
        h2 = [x for x in range(4) if x not in h1]
        agree += int(np.sign(d[list(h1)].mean()) == np.sign(d[h2].mean()))
        tot += 1
print(f"  ALL pairs: split-half (2 vs 2 replicates) direction agreement {agree}/{tot} = {agree / tot:.4f} (0.5 = pure noise)")

# magnitude context: base loss vs gains
bl = np.array([np.mean(r["feedback"]["base_losses"][0]) for r in recs])
print("\nContext: base normalized MSE at the 128 snapshot states: quantiles",
      {k: float(f"{v:.3e}") for k, v in pd.Series(bl).quantile([0, .25, .5, .75, 1]).items()},
      " states already <= 0.04 tolerance:", int((bl <= 0.04).sum()), "/128")
mg_all = np.array([[r["feedback"]["mean_gains_per_cost"][a] for a in ACTIONS] for r in recs])
print("mean gain per cost across all 1024 candidate-means: mean", float(f"{mg_all.mean():.3e}"), " median", float(f"{np.median(mg_all):.3e}"),
      " share negative", round(float((mg_all < 0).mean()), 4))
print("per-action grand mean of measured mean gains (128 records):", {a: float(f"{mg_all[:, i].mean():.3e}") for i, a in enumerate(ACTIONS)})
print("per-action median:", {a: float(f"{np.median(mg_all[:, i]):.3e}") for i, a in enumerate(ACTIONS)})
P.to_pickle(sys.argv[3])
