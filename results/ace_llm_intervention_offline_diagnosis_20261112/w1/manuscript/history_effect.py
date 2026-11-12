"""Descriptive (NOT a manuscript claim): paired effect of collection history on raw held-out MSE, init 0.
Held-out actions are shared across the two histories within a system (paper.tex 3415), so raw MSE is comparable.
Usage: python3 history_effect.py all_cells.csv
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1])
df = df[df.init == 0]
print("training_target_variance by history (geometric mean) per family:")
tv = df.drop_duplicates(["system", "history"]).groupby(["graph_size", "history"]).training_target_variance.apply(lambda s: math.exp(np.log(s).mean()))
print(tv)
for gs in (5, 30):
    for arm in ("delivery", "online", "simpler", "ablation"):
        d = df[(df.graph_size == gs) & (df.arm == arm)].pivot_table(index="system", columns="history", values="mse")
        z = np.log(d["balanced_varied_value"]) - np.log(d["matched_random"])
        n = len(z); m = z.mean(); se = z.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
        print(f"{gs}-node {arm:9s}: balanced/random raw-MSE geometric ratio={math.exp(m):.3f} "
              f"[{math.exp(m - t*se):.3f},{math.exp(m + t*se):.3f}] balanced better in {(z<0).sum()}/{n} systems")
