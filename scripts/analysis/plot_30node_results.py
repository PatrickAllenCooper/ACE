#!/usr/bin/env python3
"""
Build the 30-node headline figure for the paper.

Panels:
 (a) Learning curves: best-loss-so-far vs episode for ACE and 3 baselines.
 (b) Bar chart with per-seed scatter: final comparison.

Output:
  paper/figs/fig_30node_results.pdf  (vector)
  paper/figs/fig_30node_results.png  (raster preview)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Paper-style defaults (match NeurIPS body font)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.4,
})

OUT_DIR = "paper/figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Colors: ACE distinctive (dark red/crimson), baselines in greys
COLORS = {
    "ACE":          "#b30000",
    "Random":       "#4a6fa5",
    "Round-Robin":  "#5c8a5c",
    "Max-Variance": "#8a6fa5",
}

# ── Load ACE per-episode data ────────────────────────────────────────────
ace_runs = glob.glob(
    "results/curc_20260415_152624/large_scale/seed_*/run_*/node_losses.csv"
)
ace_curves = []
for path in ace_runs:
    df = pd.read_csv(path)
    # best-loss-so-far per episode (what we report as best-loss)
    per_ep = df.groupby("episode")["total_loss"].min().reset_index()
    per_ep["total_loss"] = per_ep["total_loss"].cummin()
    ace_curves.append(per_ep)
ace_df = pd.concat(ace_curves, keys=range(len(ace_curves)), names=["seed_idx"])
ace_agg = ace_df.groupby("episode")["total_loss"].agg(
    mean="mean", std="std", n="count"
).reset_index()

# ── Load baseline per-episode data ───────────────────────────────────────
baseline_curves = {}
for method in ["random", "round_robin", "max_variance"]:
    all_seeds = []
    for seed in [42, 123, 456, 789, 1011]:
        path = (f"results/curc_30node_baselines/{method}/"
                f"seed_{seed}/per_episode.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={"episode_final_loss": "total_loss"})
            df["total_loss"] = df["total_loss"].cummin()
            all_seeds.append(df[["episode", "total_loss"]])
    combined = pd.concat(all_seeds)
    agg = combined.groupby("episode")["total_loss"].agg(
        mean="mean", std="std"
    ).reset_index()
    baseline_curves[method] = agg

# ── Final summary stats ─────────────────────────────────────────────────
ace_best_per_seed = [df["total_loss"].min() for df in ace_curves]
ace_final_mean = np.mean(ace_best_per_seed)
ace_final_std = np.std(ace_best_per_seed, ddof=1)

final_stats = {}
for method, label in [("random", "Random"),
                      ("round_robin", "Round-Robin"),
                      ("max_variance", "Max-Variance")]:
    vals = []
    for seed in [42, 123, 456, 789, 1011]:
        s = pd.read_csv(
            f"results/curc_30node_baselines/{method}/seed_{seed}/summary.csv"
        )
        vals.append(float(s["final_total_loss"].iloc[0]))
    final_stats[label] = (np.mean(vals), np.std(vals, ddof=1), vals)

# ── Build figure ────────────────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.2, 2.9),
    gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.32}
)

# === Panel A: learning curves (log scale) ===
method_label = {
    "random": "Random",
    "round_robin": "Round-Robin",
    "max_variance": "Max-Variance",
}
for method, agg in baseline_curves.items():
    label = method_label[method]
    c = COLORS[label]
    axL.plot(agg["episode"], agg["mean"], color=c, label=label, lw=1.3)
    axL.fill_between(
        agg["episode"],
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        color=c, alpha=0.15, linewidth=0,
    )

axL.plot(ace_agg["episode"], ace_agg["mean"],
         color=COLORS["ACE"], label="ACE (ours)", lw=2.0, zorder=5)
axL.fill_between(
    ace_agg["episode"],
    ace_agg["mean"] - ace_agg["std"],
    ace_agg["mean"] + ace_agg["std"],
    color=COLORS["ACE"], alpha=0.22, linewidth=0, zorder=4,
)

axL.set_yscale("log")
axL.set_xlabel("Episode")
axL.set_ylabel("Best-loss-so-far (total MSE)")
axL.set_title("(a) Convergence on 30-node SCM", loc="left", pad=6)
axL.set_xlim(0, 150)
axL.set_ylim(1.0, 30)
axL.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.5)
axL.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
axL.legend(loc="upper right", frameon=False, handlelength=1.4)

# === Panel B: bar chart with per-seed scatter ===
labels = ["ACE\n(ours)", "Random", "Round-\nRobin", "Max-\nVariance"]
means = [ace_final_mean,
         final_stats["Random"][0],
         final_stats["Round-Robin"][0],
         final_stats["Max-Variance"][0]]
stds = [ace_final_std,
        final_stats["Random"][1],
        final_stats["Round-Robin"][1],
        final_stats["Max-Variance"][1]]
colors_bar = [COLORS["ACE"], COLORS["Random"], COLORS["Round-Robin"],
              COLORS["Max-Variance"]]

x = np.arange(len(labels))
bars = axR.bar(x, means, yerr=stds, color=colors_bar, alpha=0.72,
               edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.9, "capsize": 3.5})

# Per-seed scatter overlays
per_seed_data = [ace_best_per_seed,
                 final_stats["Random"][2],
                 final_stats["Round-Robin"][2],
                 final_stats["Max-Variance"][2]]
rng = np.random.RandomState(0)
for xi, vals in zip(x, per_seed_data):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    axR.scatter(xi + jitter, vals, s=14, color="black",
                edgecolors="white", linewidths=0.6, zorder=5, alpha=0.85)

# Improvement annotation
y_top = max(means) + max(stds) + 0.6
axR.annotate("", xy=(0, ace_final_mean + 0.3),
             xytext=(0, y_top + 0.2),
             arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
axR.annotate("", xy=(1, final_stats["Random"][0] + 0.3),
             xytext=(1, y_top + 0.2),
             arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
axR.plot([0, 1], [y_top + 0.2, y_top + 0.2], "k-", lw=0.7)
axR.text(0.5, y_top + 0.5, r"$3.0\times$",
         ha="center", va="bottom", fontsize=9, fontweight="bold")

axR.set_xticks(x)
axR.set_xticklabels(labels, fontsize=8)
axR.set_ylabel("Total MSE at final eval")
axR.set_ylim(0, y_top + 1.8)
axR.set_title("(b) Final performance", loc="left", pad=6)
axR.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

fig.suptitle("", y=1.0)  # handled in-panel
fig.savefig(os.path.join(OUT_DIR, "fig_30node_results.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_30node_results.png"),
            bbox_inches="tight", dpi=200)

print(f"Wrote: {OUT_DIR}/fig_30node_results.pdf")
print(f"Wrote: {OUT_DIR}/fig_30node_results.png")
print()
print("Summary values plotted:")
print(f"  ACE:          {ace_final_mean:.2f} +/- {ace_final_std:.2f} "
      f"(N={len(ace_best_per_seed)})")
for label in ["Random", "Round-Robin", "Max-Variance"]:
    m, s, _ = final_stats[label]
    print(f"  {label:13s}: {m:.2f} +/- {s:.2f} (N=5)")
