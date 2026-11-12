"""Verifier step 4: headroom of the hindsight best-of-8 envelope and the trivial floor against target_risk. Fresh code."""
import pandas as pd, numpy as np
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
piv = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL].sort_index()
piv.index = [f"w{int(str(i)[-2:]):02d}" for i in piv.index]
init = curves[curves.total_cost_units == 8].groupby("world_id").normalized_mse.first().sort_index().values
at = init <= 0.04
tr = piv.target_risk
env = piv.min(axis=1)
floor = pd.Series(np.where(at, 8, 9), index=piv.index)   # 8 if already at target, else at least one 1-unit acquisition
tab = piv.copy(); tab["envelope"] = env; tab["floor"] = floor; tab["TR-env"] = tr - env; tab["TR-floor"] = tr - floor
tab["best_arms"] = [",".join(p for p in POL if piv.loc[i, p] == env[i]) if not at[k] else "(all)" for k, i in enumerate(piv.index)]
pd.set_option("display.width", 300)
print(tab.to_string())
print()
print("TR total:", int(tr.sum()), " mean:", tr.mean())
print("envelope total:", int(env.sum()), " mean:", env.mean(), " TR - envelope:", int((tr - env).sum()), " per system:", (tr - env).mean())
print("floor total:", int(floor.sum()), " TR - floor (ceiling):", int((tr - floor).sum()), " per system:", (tr - floor).mean())
sv = (tr - env).sort_values(ascending=False)
print("largest envelope savings:", sv.head(8).to_dict())
print("top-5 systems' savings sum:", int(sv.head(5).sum()), "of", int(sv.sum()))
print("systems where some arm strictly beats TR:", int((env < tr).sum()), (env < tr)[env < tr].index.tolist())
print("systems where TR is (jointly) best:", int((env == tr).sum()), "; of those not at target after init:", int(((env == tr) & ~at).sum()))
print("TR acquisition cost on the 14 informative systems (sorted):", sorted((tr[~at] - 8).tolist()))
print("TR acquisition cost on the 14 informative systems (by system):", (tr[~at] - 8).to_dict())
print("TR never attains (capped at 32):", tr[tr == 32].index.tolist())
print("per-policy totals:", piv.sum().to_dict())
print("per-policy total on 14 informative systems (acq cost):", (piv[~at] - 8).sum().to_dict())
print("per-policy number of systems where policy == envelope among 14 informative:", {p: int((piv.loc[~at, p] == env[~at]).sum()) for p in POL})
print("per-policy number of systems strictly beating TR:", {p: int((piv[p] < tr).sum()) for p in POL})
# which arms deliver the envelope savings
print("envelope savings vs TR, with arms achieving the minimum:")
for i in sv[sv > 0].index:
    print("  ", i, int(sv[i]), "TR", int(tr[i]), "env", int(env[i]), [p for p in POL if piv.loc[i, p] == env[i]])
