"""Verifier step 1: recompute cells.csv fields from curves.csv; reproduce policies.csv, pairs.csv, actions.csv.
Fresh code, written without reference to the earlier agent's scripts."""
import pandas as pd, numpy as np, json, itertools
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
TH = 0.04
CAP = 32
bad = 0
rows = []
for (w, p), g in curves.groupby(["world_id", "policy_id"], sort=False):
    g = g.reset_index(drop=True)
    # order check: file order should be nondecreasing in total cost and accepted receipts increasing by 1
    assert (np.diff(g.total_cost_units.values) > 0).all(), (w, p)
    assert (np.diff(g.accepted_receipts.values) == 1).all(), (w, p)
    assert g.total_cost_units.iloc[0] == 8 and g.acquisition_cost_units.iloc[0] == 0
    assert pd.isna(g.action_id.iloc[0]) and g.action_id.iloc[1:].notna().all()
    assert ((g.total_cost_units - g.acquisition_cost_units) == 8).all()
    cross = g[g.normalized_mse <= TH]
    first = cross.total_cost_units.iloc[0] if len(cross) else np.nan
    capped = first if len(cross) else CAP
    c = cells[(cells.world_id == w) & (cells.policy_id == p)].iloc[0]
    ok = True
    ok &= (c.capped_total_cost == capped)
    ok &= (c.capped_acquisition_cost == capped - 8)
    ok &= abs(c.final_error - g.normalized_mse.iloc[-1]) < 1e-15
    ok &= (c.ever_attained == (len(cross) > 0))
    ok &= (c.final_attained == (g.normalized_mse.iloc[-1] <= TH))
    ok &= (c.actual_total_cost == g.total_cost_units.iloc[-1])
    ok &= ((np.isnan(c.first_crossing_total) and np.isnan(first)) or (c.first_crossing_total == first))
    if not ok:
        bad += 1
        print("INCONSISTENT", w, p)
    rows.append(dict(world_id=w, policy_id=p, n_points=len(g), n_actions=len(g) - 1))
print("runs checked:", len(rows), "inconsistent:", bad)
print("total curve points:", len(curves))

# NMSE recomputation requires truth, which is not in the CSV; skip here.

# policies.csv
pol = pd.read_csv(D + "policies.csv").set_index("policy_id")
agg = cells.groupby("policy_id").agg(
    mean_capped_total_cost=("capped_total_cost", "mean"),
    mean_capped_acquisition_cost=("capped_acquisition_cost", "mean"),
    ever_attained=("ever_attained", "sum"),
    final_attained=("final_attained", "sum"),
    mean_final_error=("final_error", "mean"),
    actual_total_cost=("actual_total_cost", "sum"),
    stopped=("termination_reason", lambda s: (s == "policy_stop").sum()),
).loc[POL]
print(agg.to_string())
for col in agg.columns:
    d = np.abs(agg[col].astype(float) - pol.loc[POL, col].astype(float)).max()
    print("policies.csv max abs diff", col, d)

# pairs.csv
pairs = pd.read_csv(D + "pairs.csv")
piv = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost").sort_index()
nbad = 0
for _, r in pairs.iterrows():
    diff = (piv[r.left] - piv[r.right]).values
    saved = np.array(json.loads(r.differences))
    wl = json.loads(r.world_ids)
    if wl != piv.index.tolist() or not (diff == saved).all() or abs(diff.mean() - r.mean_difference) > 1e-12:
        nbad += 1; print("pairs mismatch", r.left, r.right)
    # check sign convention of wins / losses
    a = (int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum()))
    b = (int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum()))
    if (r.wins, r.ties, r.losses) not in (a, b):
        nbad += 1; print("pairs wtl mismatch", r.left, r.right, (r.wins, r.ties, r.losses), a)
print("pairs rows:", len(pairs), "mismatching:", nbad)
r0 = pairs.iloc[0]
d0 = (piv[r0.left] - piv[r0.right]).values
print("convention example:", r0.left, r0.right, "wins", r0.wins, "ties", r0.ties, "losses", r0.losses, "| left<right:", (d0 < 0).sum(), "left>right:", (d0 > 0).sum())

# actions.csv
act = pd.read_csv(D + "actions.csv")
mine = curves.dropna(subset=["action_id"]).groupby(["policy_id", "action_id"]).size().rename("n").reset_index()
m = act.merge(mine, on=["policy_id", "action_id"], how="outer")
print("actions.csv rows", len(act), "rows mismatching:", int((m.accepted_acquisitions != m.n).sum()))
print("acquired actions per policy:", mine.groupby("policy_id").n.sum().loc[POL].to_dict())
print("total acquisitions:", int(mine.n.sum()))
print("accepted responses total (incl. init 8 per run):", int(mine.n.sum()) + 8 * 192)
print("logical cost total:", int(cells.actual_total_cost.sum()))
