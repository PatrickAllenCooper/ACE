"""Read-only recomputation of manuscript prospective claims from the saved all-cells CSV.
Usage: python3 recompute_prospective.py /path/to/delivery_prospective_all_cells.csv
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats

p = sys.argv[1]
df = pd.read_csv(p)
print("rows", len(df), "arms", sorted(df.arm.unique()), "inits", sorted(df.init.unique()),
      "histories", sorted(df.history.unique()), "graph sizes", sorted(df.graph_size.unique()))
print(df.groupby(["graph_size", "arm", "init"]).size().unstack(fill_value=0))
FLOOR = 1e-12
df["lognmse"] = np.log(np.maximum(df.nmse, FLOOR))
print("floor active count:", int((df.nmse < FLOOR).sum()), "| floor_active col True:", int((df.floor_active.astype(str) == "True").sum()))

def cell(gs, arm, init):
    d = df[(df.graph_size == gs) & (df.arm == arm) & (df.init == init)]
    return d.set_index(["system", "history"]).lognmse

def contrast(gs, a, c, ia=0, ic=0):
    z = (cell(gs, a, ia) - cell(gs, c, ic)).unstack("history")  # system x history
    zs = z.mean(axis=1)
    n = len(zs)
    m, sd = zs.mean(), zs.std(ddof=1)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    praw = 2 * stats.t.sf(abs(m / se), n - 1)
    wins = int((zs < 0).sum())
    return dict(ratio=math.exp(m), lo=math.exp(m - tcrit * se), hi=math.exp(m + tcrit * se), p_raw=praw,
                n=n, wins=wins, zs=zs, z=z)

print("\n== Registered primary contrasts (init 0), delivery/control ==")
res = {}
for gs in (5, 30):
    for c in ("online", "simpler"):
        r = contrast(gs, "delivery", c)
        res[(gs, c)] = r
ps = sorted(((k, v["p_raw"]) for k, v in res.items()), key=lambda kv: kv[1])
holm = {}
running = 0
for i, (k, pv) in enumerate(ps):
    adj = min(1.0, (len(ps) - i) * pv)
    running = max(running, adj)
    holm[k] = running
for k, r in res.items():
    loo = [math.exp(r["zs"].drop(s).mean()) for s in r["zs"].index]
    print(f"{k}: ratio={r['ratio']:.6g} CI=[{r['lo']:.6g},{r['hi']:.6g}] p_raw={r['p_raw']:.6g} holm={holm[k]:.6g} "
          f"systems delivery better={r['wins']}/{r['n']} LOO range=[{min(loo):.3f},{max(loo):.3f}] "
          f"gate={'met' if (r['ratio']<=0.8 and r['hi']<1 and holm[k]<0.05) else 'not met'}")

print("\n== History-specific (init 0) geometric ratios and delivery wins out of 20 ==")
for gs in (5, 30):
    for c in ("online", "simpler"):
        z = res[(gs, c)]["z"]
        for h in z.columns:
            print(f"  {gs}-node {h:22s} vs {c:8s}: ratio={math.exp(z[h].mean()):.4f} wins={(z[h]<0).sum()}/20")

print("\n== Rolling vs flat directly (init 0), online/simpler ==")
for gs in (5, 30):
    r = contrast(gs, "online", "simpler")
    print(f"  {gs}-node: ratio={r['ratio']:.4f} rolling better in {r['wins']}/20 CI=[{r['lo']:.3f},{r['hi']:.3f}]")
    z = r["z"]
    for h in z.columns:
        print(f"     {h:22s}: ratio={math.exp(z[h].mean()):.4f} rolling wins={(z[h]<0).sum()}/20")

print("\n== Absolute scale at init 0: median sqrt(NMSE) over 40 cells per arm/family (percent of train SD) ==")
for gs in (5, 30):
    for arm in ("delivery", "online", "simpler", "short_fit" if "short_fit" in set(df.arm) else None):
        if arm is None: continue
        d = df[(df.graph_size == gs) & (df.arm == arm) & (df.init == 0)]
        if len(d) == 0: continue
        print(f"  {gs}-node {arm:10s}: n={len(d)} median RMSE/SD={100*np.sqrt(d.nmse).median():.3f}%  median NMSE={d.nmse.median():.4g} geo NMSE={math.exp(np.log(d.nmse).mean()):.4g} max NMSE={d.nmse.max():.4g}")

print("\n== Per-init delivery/control ratios and seed-averaged ==")
for gs in (5, 30):
    for c in ("online", "simpler"):
        out = []
        for i in (0, 1, 2):
            ic = i if c == "simpler" else 0
            out.append(contrast(gs, "delivery", c, i, ic)["ratio"])
        # seed-averaged
        dl = pd.concat([cell(gs, "delivery", i) for i in (0, 1, 2)], axis=1).mean(axis=1)
        if c == "simpler":
            cl = pd.concat([cell(gs, "simpler", i) for i in (0, 1, 2)], axis=1).mean(axis=1)
        else:
            cl = cell(gs, "online", 0)
        zs = (dl - cl).unstack("history").mean(axis=1)
        print(f"  {gs}-node vs {c:8s}: init0={out[0]:.3f} init1={out[1]:.3f} init2={out[2]:.3f} seed-avg={math.exp(zs.mean()):.3f}")

print("\n== Within-history worst/best delivery NMSE across 3 inits ==")
for gs in (5, 30):
    d = df[(df.graph_size == gs) & (df.arm == "delivery")].pivot_table(index=["system", "history"], columns="init", values="nmse")
    rr = d.max(axis=1) / d.min(axis=1)
    print(f"  {gs}-node: median={rr.median():.3f} p90={rr.quantile(0.9):.2f} max={rr.max():.1f} (n={len(rr)})")

print("\n== Cell 5:08 matched_random ==")
print(df[(df.system == "5:08") & (df.history == "matched_random")][["arm", "init", "mse", "nmse"]].to_string(index=False))

print("\n== Arms present and which arm is best per system/history at init 0 ==")
for gs in (5, 30):
    d = df[(df.graph_size == gs) & (df.init == 0) & (df.arm.isin(["delivery", "online", "simpler"]))].pivot_table(index=["system", "history"], columns="arm", values="nmse")
    best = d.idxmin(axis=1)
    print(f"  {gs}-node best-arm counts over 40 system/history cells:", best.value_counts().to_dict())
    for h in ("balanced_varied_value", "matched_random"):
        print(f"     {h}:", best.xs(h, level="history").value_counts().to_dict())

print("\n== Collection-history effect on each arm's own error (init 0): geometric NMSE by history ==")
for gs in (5, 30):
    for arm in ("delivery", "online", "simpler"):
        d = df[(df.graph_size == gs) & (df.init == 0) & (df.arm == arm)]
        g = d.groupby("history").lognmse.mean().apply(math.exp)
        ratio = g["balanced_varied_value"] / g["matched_random"]
        print(f"  {gs}-node {arm:9s}: balanced={g['balanced_varied_value']:.4g} random={g['matched_random']:.4g} balanced/random={ratio:.3f}")
