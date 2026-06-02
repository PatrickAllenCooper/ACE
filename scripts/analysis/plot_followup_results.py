#!/usr/bin/env python3
"""
Figure for the LM-prior-vs-DPO follow-up (anon30 + nodes50).

Two panels:
  (a) anon30 best-MSE learning curves: ACE vs zero-shot LM (mean +/- std over
      seeds, per-episode cumulative-min total_loss). Shows whether DPO opens a
      gap over the pretrained LM prior when node names are anonymised.
  (b) Best-MSE bar chart for the clean cells with per-seed scatter:
      anon30 ACE, anon30 ZSL, nodes50 ZSL. (nodes50 ACE is omitted by default
      because the May 2026 runs were contaminated/timeout-limited; pass
      --include-nodes50-ace to show it with a hatched 'partial' bar.)

Reads:
  results/curc_30node_followup/aggregate.csv         (bars; from aggregate_followup_results.py)
  results/curc_30node_followup/<cond>/<method>/seed_*/job_*/run_*/node_losses.csv  (curves)

Writes:
  paper/neurips_ace_2026/figs/fig_followup.pdf (+ .png)
  (falls back to results/curc_30node_followup/fig_followup.* if the paper dir
   is absent)

Usage:
  python scripts/analysis/aggregate_followup_results.py   # refresh aggregate.csv first
  python scripts/analysis/plot_followup_results.py
"""
import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = "results/curc_30node_followup"

# Palette aligned with paper/neurips_ace_2026/figs/ace_palette.tex.
COLORS = {
    ("anon30", "ace"):          "#9F1239",  # aceRubyDk  (ACE)
    ("anon30", "zero_shot_lm"): "#15803D",  # aceLeafDk  (zero-shot LM)
    ("nodes50", "zero_shot_lm"):"#0369A1",  # aceSkyDk   (ZSL @ scale)
    ("nodes50", "ace"):         "#475569",  # aceSlateDk (contaminated)
}
LABEL = {
    ("anon30", "ace"):          "ACE (anon-30)",
    ("anon30", "zero_shot_lm"): "Zero-shot LM (anon-30)",
    ("nodes50", "zero_shot_lm"):"Zero-shot LM (50-node)",
    ("nodes50", "ace"):         "ACE (50-node, partial)",
}
SEEDS = [42, 123, 456, 789, 1011]


def per_episode_curve(nl_path):
    """Cumulative-min of per-episode-final total_loss (the 'best so far')."""
    df = pd.read_csv(nl_path)
    if "episode" not in df.columns or "total_loss" not in df.columns:
        return None
    pe = df.groupby("episode")["total_loss"].last().sort_index()
    return pe.cummin()


def latest_node_losses(cond, method, seed):
    seed_dir = os.path.join(ROOT, cond, method, f"seed_{seed}")
    cands = glob(os.path.join(seed_dir, "**", "node_losses.csv"), recursive=True)
    if not cands:
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def aligned_mean_std(curves):
    """Align a list of pandas Series (indexed by episode) on common episodes."""
    if not curves:
        return None, None, None
    max_ep = min(int(c.index.max()) for c in curves)
    eps = np.arange(0, max_ep + 1)
    mat = []
    for c in curves:
        c2 = c.reindex(range(0, max_ep + 1)).ffill().bfill()
        mat.append(c2.loc[0:max_ep].values)
    mat = np.vstack(mat)
    return eps, mat.mean(axis=0), mat.std(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-nodes50-ace", action="store_true",
                    help="Show the contaminated/partial nodes50 ACE bar (hatched).")
    args = ap.parse_args()

    agg_path = os.path.join(ROOT, "aggregate.csv")
    if not os.path.isfile(agg_path):
        raise SystemExit(f"Missing {agg_path}; run aggregate_followup_results.py first.")
    agg = pd.read_csv(agg_path)
    agg = agg[agg["status"] == "ok"].copy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Panel A: anon30 ACE vs ZSL learning curves --------------------------
    for method in ("ace", "zero_shot_lm"):
        curves = []
        for s in SEEDS:
            nl = latest_node_losses("anon30", method, s)
            if nl:
                c = per_episode_curve(nl)
                if c is not None and len(c) > 1:
                    curves.append(c)
        eps, mean, std = aligned_mean_std(curves)
        if eps is None:
            continue
        col = COLORS[("anon30", method)]
        axA.plot(eps, mean, color=col, lw=2.2, label=LABEL[("anon30", method)])
        axA.fill_between(eps, mean - std, mean + std, color=col, alpha=0.18)
    axA.set_xlabel("Episode")
    axA.set_ylabel("Best total mechanism MSE")
    axA.set_title("(a) Anonymised 30-node: ACE vs zero-shot LM")
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.grid(alpha=0.25)

    # ---- Panel B: best-MSE bars with per-seed scatter ------------------------
    cells = [("anon30", "ace"), ("anon30", "zero_shot_lm"),
             ("nodes50", "zero_shot_lm")]
    if args.include_nodes50_ace:
        cells.append(("nodes50", "ace"))

    xs, means, stds, labels, colors, scatter = [], [], [], [], [], []
    for i, (cond, method) in enumerate(cells):
        cell = agg[(agg["condition"] == cond) & (agg["method"] == method)]
        bests = pd.to_numeric(cell["best_mse"], errors="coerce").dropna().values
        if len(bests) == 0:
            continue
        xs.append(i)
        means.append(bests.mean())
        stds.append(bests.std() if len(bests) > 1 else 0.0)
        labels.append(LABEL[(cond, method)])
        colors.append(COLORS[(cond, method)])
        scatter.append(bests)

    bars = axB.bar(xs, means, yerr=stds, capsize=4, color=colors,
                   alpha=0.85, edgecolor="black", linewidth=0.6,
                   error_kw={"elinewidth": 1.2})
    # Hatch the contaminated nodes50 ACE bar if shown.
    if args.include_nodes50_ace and ("nodes50", "ace") in cells:
        bars[-1].set_hatch("//")
    for xi, vals in zip(xs, scatter):
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.25
        axB.scatter(np.full(len(vals), xi) + jitter, vals, s=22,
                    color="black", alpha=0.6, zorder=3)
    axB.set_xticks(xs)
    axB.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    axB.set_ylabel("Best total mechanism MSE")
    axB.set_title("(b) Best MSE per cell (points = seeds)")
    axB.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    out_dir = "paper/neurips_ace_2026/figs"
    if not os.path.isdir(out_dir):
        out_dir = ROOT
    base = os.path.join(out_dir, "fig_followup")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"Wrote {base}.pdf and {base}.png")

    # Console summary for quick reading.
    print("\nBest-MSE per cell (N, mean +/- std):")
    for (cond, method) in cells:
        cell = agg[(agg["condition"] == cond) & (agg["method"] == method)]
        bests = pd.to_numeric(cell["best_mse"], errors="coerce").dropna().values
        if len(bests):
            print(f"  {cond:<8} {method:<13} N={len(bests)}  "
                  f"{bests.mean():.3f} +/- {bests.std() if len(bests)>1 else 0:.3f}")


if __name__ == "__main__":
    main()
