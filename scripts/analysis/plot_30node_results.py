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

# Default output goes to the active NeurIPS paper directory; falls back to
# the legacy paper/figs path if the new dir does not exist yet.
OUT_DIR = ("paper/neurips_ace_2026/figs"
           if os.path.isdir("paper/neurips_ace_2026/figs")
           else "paper/figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Colors taken from the shared palette in paper/figs/ace_palette.tex so the
# matplotlib figure visually harmonises with the TikZ figures (hero, 5-node
# SCM, 30-node SCM). ACE uses the ruby-dark accent; the three static
# baselines use sky / teal / violet; Bayesian OED uses amber (lookahead);
# PPO uses slate (neutral baseline color, since PPO is the value-based RL
# foil to ACE's preference-based approach).
COLORS = {
    "ACE":          "#be123c",  # aceRubyDk
    "Random":       "#1e40af",  # aceSkyDk
    "Round-Robin":  "#0f766e",  # aceTealDk
    "Max-Variance": "#6b21a8",  # aceVioDk
    "Bayesian OED": "#c2410c",  # aceAmberDk
    "PPO":          "#475569",  # aceSlateDk
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
# The first three live under results/curc_30node_baselines/{method}/seed_X/.
# Bayesian OED was added in the rebuttal and lives at a triply-nested path
# under results/curc_30node_rebuttal/bayesian_oed/bayesian_oed/bayesian_oed/.
baseline_paths = {
    "random":       "results/curc_30node_baselines/random",
    "round_robin":  "results/curc_30node_baselines/round_robin",
    "max_variance": "results/curc_30node_baselines/max_variance",
    "bayesian_oed": ("results/curc_30node_rebuttal/bayesian_oed/"
                     "bayesian_oed/bayesian_oed"),
    "ppo":          "results/curc_30node_rebuttal/ppo/ppo",
}
baseline_curves = {}
for method, root in baseline_paths.items():
    all_seeds = []
    for seed in [42, 123, 456, 789, 1011]:
        path = f"{root}/seed_{seed}/per_episode.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={"episode_final_loss": "total_loss"})
            df["total_loss"] = df["total_loss"].cummin()
            all_seeds.append(df[["episode", "total_loss"]])
    if not all_seeds:
        print(f"WARNING: no per_episode.csv found for {method}")
        continue
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
final_stats_paths = {
    "Random":       "results/curc_30node_baselines/random",
    "Round-Robin":  "results/curc_30node_baselines/round_robin",
    "Max-Variance": "results/curc_30node_baselines/max_variance",
    "Bayesian OED": ("results/curc_30node_rebuttal/bayesian_oed/"
                     "bayesian_oed/bayesian_oed"),
    "PPO":          "results/curc_30node_rebuttal/ppo/ppo",
}
for label, root in final_stats_paths.items():
    vals = []
    for seed in [42, 123, 456, 789, 1011]:
        sp = f"{root}/seed_{seed}/summary.csv"
        if os.path.exists(sp):
            s = pd.read_csv(sp)
            vals.append(float(s["final_total_loss"].iloc[0]))
    if vals:
        final_stats[label] = (np.mean(vals), np.std(vals, ddof=1), vals)
    else:
        print(f"WARNING: no summary.csv found for {label}")

# ── Build figure ────────────────────────────────────────────────────────
# Panel B widened to fit 6 bars (was 5 bars).
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.4, 3.1),
    gridspec_kw={"width_ratios": [1.35, 1.4], "wspace": 0.34}
)

# === Panel A: learning curves (linear scale, distinct line styles) ===
# Linear scale exposes the gap dramatically; log compresses it.
method_label = {
    "random":       "Random",
    "round_robin":  "Round-Robin",
    "max_variance": "Max-Variance",
    "bayesian_oed": "Bayesian OED",
    "ppo":          "PPO",
}
linestyles = {
    "random":       "-",
    "round_robin":  "--",
    "max_variance": ":",
    "bayesian_oed": "-.",
    "ppo":          (0, (3, 1, 1, 1)),  # densely dashdotted
}
# Plot baselines first (thin) then ACE on top (thick) for clear z-order.
for method, agg in baseline_curves.items():
    label = method_label[method]
    c = COLORS[label]
    axL.plot(agg["episode"], agg["mean"],
             color=c, label=label, lw=1.2,
             linestyle=linestyles[method])
    axL.fill_between(
        agg["episode"],
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        color=c, alpha=0.10, linewidth=0,
    )

axL.plot(ace_agg["episode"], ace_agg["mean"],
         color=COLORS["ACE"], label="ACE (ours)", lw=2.4, zorder=5)
axL.fill_between(
    ace_agg["episode"],
    ace_agg["mean"] - ace_agg["std"],
    ace_agg["mean"] + ace_agg["std"],
    color=COLORS["ACE"], alpha=0.22, linewidth=0, zorder=4,
)

# Cluster annotation: arrow + label pointing at the baseline plateau.
axL.annotate(
    "Baselines collapse\nto common plateau",
    xy=(110, 5.86),
    xytext=(60, 4.2),
    fontsize=8.5, color="#475569",  # aceSlateDk
    ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.7,  # aceMuted
                    connectionstyle="arc3,rad=0.18"),
)
# ACE annotation
axL.annotate(
    "ACE descends\nto 1.95",
    xy=(70, 2.0),
    xytext=(105, 2.6),
    fontsize=8.5, color=COLORS["ACE"],
    ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color=COLORS["ACE"], lw=0.7,
                    connectionstyle="arc3,rad=-0.18"),
)

axL.set_xlabel("Episode")
axL.set_ylabel("Best-loss-so-far (total MSE)")
axL.set_title("(a) Convergence on 30-node SCM", loc="left", pad=6)
axL.set_xlim(0, 150)
axL.set_ylim(0, 8)
axL.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
axL.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    edgecolor="#94a3b8",  # slate-medium
    handlelength=1.8,
    fontsize=8,
    borderpad=0.4,
    labelspacing=0.4,
)

# === Panel B: bar chart with per-seed scatter, cleaner bracket ===
# Order: ACE first, then Bayesian OED (principled baseline) and PPO
# (value-based RL foil), then the three static heuristics.
bar_methods = ["ACE", "Bayesian OED", "PPO",
               "Random", "Round-Robin", "Max-Variance"]
labels = ["ACE\n(ours)", "Bayesian\nOED", "PPO",
          "Random", "Round-\nRobin", "Max-\nVariance"]
means = [ace_final_mean] + [final_stats[m][0] for m in bar_methods[1:]]
stds  = [ace_final_std]  + [final_stats[m][1] for m in bar_methods[1:]]
colors_bar = [COLORS[m] for m in bar_methods]

x = np.arange(len(labels))
axR.bar(x, means, yerr=stds, color=colors_bar, alpha=0.78,
        edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 0.9, "capsize": 3.5})

# Per-seed scatter overlays
per_seed_data = [ace_best_per_seed] + [final_stats[m][2] for m in bar_methods[1:]]
rng = np.random.RandomState(0)
for xi, vals in zip(x, per_seed_data):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    axR.scatter(xi + jitter, vals, s=16, color="black",
                edgecolors="white", linewidths=0.6, zorder=5, alpha=0.85)

# Improvement bracket: spans ACE (x=0) -> Bayesian OED (x=1, the strongest
# principled baseline). Horizontal dashed reference line at the common
# baseline plateau makes it visually clear that all 4 baselines collapse.
y_ace = ace_final_mean
y_boed = final_stats["Bayesian OED"][0]
y_base = np.mean([final_stats[m][0] for m in
                  ["Bayesian OED", "Random", "Round-Robin", "Max-Variance"]])
# Horizontal reference line at the common baseline plateau
axR.axhline(y=y_base, color="#94a3b8", linestyle="--", lw=0.6, alpha=0.7,
            zorder=1)
# Bracket between ACE (x=0) and Bayesian OED (x=1)
y_top = y_base + 0.95
axR.plot([0, 0], [y_ace + 0.35, y_top], color="black", lw=0.7)
axR.plot([1, 1], [y_boed + 0.35, y_top], color="black", lw=0.7)
axR.plot([0, 1], [y_top, y_top], color="black", lw=0.7)
ratio = y_boed / y_ace
axR.text(0.5, y_top + 0.18, rf"$\mathbf{{{ratio:.1f}\times}}$",
         ha="center", va="bottom", fontsize=11, fontweight="bold")

axR.set_xticks(x)
# Compact 2-line labels for all 6 bars. With one more bar, fontsize drops
# slightly to fit cleanly.
axR.set_xticklabels(labels, fontsize=7)
# Highlight ACE in ruby + bold; baselines in slate.
slate = "#475569"
tick_label_colors = [COLORS["ACE"]] + [slate] * (len(bar_methods) - 1)
for tick_label, c in zip(axR.get_xticklabels(), tick_label_colors):
    tick_label.set_color(c)
    tick_label.set_fontweight("bold" if c == COLORS["ACE"] else "normal")
axR.set_ylabel("Total MSE at final eval")
axR.set_ylim(0, y_top + 1.6)
axR.set_title("(b) Final performance", loc="left", pad=6)
axR.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

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
for label in ["Bayesian OED", "PPO", "Random", "Round-Robin", "Max-Variance"]:
    if label not in final_stats:
        print(f"  {label:13s}: MISSING")
        continue
    m, s, vals = final_stats[label]
    print(f"  {label:13s}: {m:.2f} +/- {s:.2f} (N={len(vals)})")
