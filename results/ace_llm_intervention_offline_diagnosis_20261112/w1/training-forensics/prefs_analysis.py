"""Preference-comparison forensics from saved records + saved exposure proposal. Offline only.

collection.pkl is a scratch file written by load_collection.py in this session (trusted input).
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

D = pickle.load(open(sys.argv[1], "rb"))
EXPOSURE = json.load(open("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/"
                          "terminal01/remote/cpu/preparation/session/worker/artifacts/exposure.json"))
ACTIONS = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos", "set_y_neg", "set_y_zero", "set_y_pos", "set_xy"]

P = pd.DataFrame(D["prefs"])
print("TOTAL preference comparisons:", len(P), " records with comparisons:", P.record_id.nunique())
print("exposure.json counts:", EXPOSURE["counts"])

# attach selected status from the exposure proposal
status = {}
for rec in EXPOSURE["records"]:
    for row in rec["pairs"]:
        status[(rec["record_id"], row["preference_index"])] = (row["status"], row["pair_index"],
                                                              row["chosen_command"], row["rejected_command"])
P["status"] = [status[(r, i)][0] for r, i in zip(P.record_id, P.preference_index)]
P["pair_index"] = [status[(r, i)][1] for r, i in zip(P.record_id, P.preference_index)]
# consistency: exposure chosen command equals record chosen
for (r, i, ch, rj) in zip(P.record_id, P.preference_index, P.chosen, P.rejected):
    st = status[(r, i)]
    if ch is None:
        assert st[0] == "tied"
    else:
        assert json.loads(st[2])["action_id"] == ch and json.loads(st[3])["action_id"] == rj
print("status counts:", P.status.value_counts().to_dict())

print("\n== ties vs strict")
print("tie:", int(P.tie.sum()), f"({P.tie.mean():.4f})", " strict:", int((~P.tie).sum()), f"({(~P.tie).mean():.4f})")
print("tie rule in data: tie_multiplier values", sorted(set(r["feedback"]["tie_multiplier"] for r in D["records"] if r["feedback"])),
      "minimum_gain_gap values", sorted(set(r["feedback"]["minimum_gain_gap"] for r in D["records"] if r["feedback"])),
      "replicate_count values", sorted(set(r["feedback"]["replicate_count"] for r in D["records"] if r["feedback"])))
# verify tie rule exactly reproduces labels
recomputed_tie = (P.mean_gain_difference.abs() <= P.tie_threshold)
print("tie flag == (|mean| <= threshold) for all rows:", bool((recomputed_tie == P.tie).all()))
print("threshold == 2*SE for all rows:", bool(np.allclose(P.tie_threshold, 2 * P.paired_standard_error, rtol=0, atol=1e-15)))
# verify SE from replicate differences
se_chk = P.replicate_differences.apply(lambda d: np.std(d, ddof=1) / np.sqrt(len(d)))
print("SE recomputed from replicate differences max abs err:", float((se_chk - P.paired_standard_error).abs().max()))
print("replicates per comparison:", P.replicate_differences.apply(len).value_counts().to_dict())
strict = P[~P.tie].copy()
strict["chosen_is_left"] = strict.chosen == strict.left
print("strict direction: chosen==left", int(strict.chosen_is_left.sum()), " chosen==right", int((~strict.chosen_is_left).sum()))

print("\n== strict pairs per record (128 nonterminal records)")
per_rec = P.groupby("record_id").apply(lambda g: int((~g.tie).sum()))
print(per_rec.value_counts().sort_index().to_dict())
print("records with zero strict pairs:", int((per_rec == 0).sum()), " mean strict/record:", round(per_rec.mean(), 3),
      " median:", per_rec.median(), " max:", per_rec.max())
print("\n== strict fraction by stratum")
for col in ("snapshot", "graph", "cost_menu", "regime", "noise_variance", "teacher_method", "behavior_method"):
    g = P.groupby(col).apply(lambda x: pd.Series({"pairs": len(x), "strict": int((~x.tie).sum()),
                                                  "strict_frac": round((~x.tie).mean(), 4)}))
    print(g.to_string())
    print()
print("== strict fraction by graph x cost_menu")
print(P.groupby(["graph", "cost_menu"]).apply(lambda x: pd.Series({"pairs": len(x), "strict": int((~x.tie).sum()),
                                                                 "strict_frac": round((~x.tie).mean(), 4)})).to_string())

print("\n== all 687 strict pairs: ordered (preferred -> rejected) counts matrix (rows preferred, cols rejected)")
M = pd.crosstab(strict.chosen, strict.rejected).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
M["WINS(total preferred)"] = M.sum(axis=1)
print(M.to_string())
losses = pd.crosstab(strict.chosen, strict.rejected).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0).sum(axis=0)
summ = pd.DataFrame({"preferred_n": M["WINS(total preferred)"], "rejected_n": losses})
summ["appearances"] = summ.preferred_n + summ.rejected_n
summ["win_rate"] = (summ.preferred_n / summ.appearances).round(4)
summ["share_of_preferred_side"] = (summ.preferred_n / len(strict)).round(4)
summ["share_of_rejected_side"] = (summ.rejected_n / len(strict)).round(4)
print("\n== per-action totals over all 687 strict pairs")
print(summ.to_string())

sel = P[P.status == "selected"].copy()
print("\n== the 119 strict pairs SELECTED for DPO updates")
print("n selected:", len(sel), " records:", sel.record_id.nunique())
M2 = pd.crosstab(sel.chosen, sel.rejected).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
M2["WINS"] = M2.sum(axis=1)
print(M2.to_string())
l2 = pd.crosstab(sel.chosen, sel.rejected).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0).sum(axis=0)
s2 = pd.DataFrame({"preferred_n": M2["WINS"], "rejected_n": l2})
s2["appearances"] = s2.preferred_n + s2.rejected_n
s2["win_rate"] = (s2.preferred_n / s2.appearances.replace(0, np.nan)).round(4)
s2["share_preferred_side"] = (s2.preferred_n / len(sel)).round(4)
s2["share_rejected_side"] = (s2.rejected_n / len(sel)).round(4)
s2["net(pref-rej)"] = s2.preferred_n - s2.rejected_n
print(s2.to_string())
print("uniform expectation per side for 8 actions: 1/8 = 0.125 ->", 119 / 8, "pairs")
print("\nselected pairs by stratum:")
for col in ("snapshot", "graph", "cost_menu", "regime", "teacher_method"):
    print(col, sel[col].value_counts().sort_index().to_dict())
print("\nselected pairs: set_x_neg preferred by graph/cost:")
print(pd.crosstab([sel.graph, sel.cost_menu], sel.chosen == "set_x_neg"))
print("selected pairs: set_x_neg rejected by graph/cost:")
print(pd.crosstab([sel.graph, sel.cost_menu], sel.rejected == "set_x_neg"))

print("\n== cost structure of strict pairs (does 'cheaper action wins' explain labels?)")
strict["chosen_cost"] = [l if c == le else r for c, le, l, r in zip(strict.chosen, strict.left, strict.left_cost, strict.right_cost)]
strict["rejected_cost"] = [r if c == le else l for c, le, l, r in zip(strict.chosen, strict.left, strict.left_cost, strict.right_cost)]
print(pd.crosstab([strict.cost_menu, strict.chosen_cost], strict.rejected_cost))
het = strict[(strict.cost_menu == "heterogeneous") & (strict.chosen_cost != strict.rejected_cost)]
print("heterogeneous-menu strict pairs with unequal cost:", len(het), " cheaper preferred:", int((het.chosen_cost < het.rejected_cost).sum()))

print("\n== margin distribution")
P["abs_mean"] = P.mean_gain_difference.abs()
P["z"] = P.abs_mean / P.paired_standard_error.replace(0, np.nan)
q = [0, .05, .25, .5, .75, .95, 1]
print("all pairs |mean gain difference| quantiles:", {k: float(f"{v:.3e}") for k, v in P.abs_mean.quantile(q).items()})
print("strict   |mean gain difference| quantiles:", {k: float(f"{v:.3e}") for k, v in P[~P.tie].abs_mean.quantile(q).items()})
print("tied     |mean gain difference| quantiles:", {k: float(f"{v:.3e}") for k, v in P[P.tie].abs_mean.quantile(q).items()})
print("selected |mean gain difference| quantiles:", {k: float(f"{v:.3e}") for k, v in P[P.status == 'selected'].abs_mean.quantile(q).items()})
print("paired SE quantiles (all):", {k: float(f"{v:.3e}") for k, v in P.paired_standard_error.quantile(q).items()})
print("z=|mean|/SE quantiles all:", {k: round(float(v), 3) for k, v in P.z.quantile(q).items()})
print("z quantiles strict:", {k: round(float(v), 3) for k, v in P[~P.tie].z.quantile(q).items()})
print("z quantiles selected:", {k: round(float(v), 3) for k, v in P[P.status == 'selected'].z.quantile(q).items()})
for thr in (2, 2.5, 3, 3.182, 4, 5, 5.841, 7.453, 10, 12.92):
    print(f"  pairs with z > {thr}: all {int((P.z > thr).sum())}/{len(P)}  selected {int((P[P.status=='selected'].z > thr).sum())}/119")

P.to_pickle(sys.argv[2])
