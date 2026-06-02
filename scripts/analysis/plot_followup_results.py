#!/usr/bin/env python3
"""
Ablation figure for the LM-prior-vs-DPO follow-up (anon30 + nodes50).

This is the APPENDIX ablation figure. The main-text scaling figure (per-node
best MSE vs N) is produced by plot_scaling.py. Here we isolate DPO's marginal
contribution over the pretrained LM prior under anonymised node names.

Two panels:
  (a) anon30 best-MSE learning curves: ACE vs ACE w/o DPO (LM + lookahead, i.e.
      --no_dpo) -- mean +/- std over seeds, per-episode cumulative-min
      total_loss. A dashed line marks the best-MSE plateau (mean best_episode),
      which documents that the comparison is fair even though anon30 ACE was
      wall-time truncated.
  (b) PER-NODE best MSE (total / n_nodes) with per-seed scatter for the clean
      cells, grouped by scale so the larger 50-node graph is not penalised by
      the mechanical growth of the summed total. nodes50 ACE is omitted by
      default (May 2026 runs were timeout-limited); pass --include-nodes50-ace.

IMPORTANT framing: "ACE w/o DPO" is NOT naive zero-shot generation -- the LM
still proposes K candidates and lookahead selects the best; only the DPO weight
updates are disabled.

Reads:
  results/curc_30node_followup/aggregate.csv         (bars; from aggregate_followup_results.py)
  results/curc_30node_followup/<cond>/<method>/seed_*/**/node_losses.csv  (curves)

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

# Node counts per condition (for per-node normalisation).
N_NODES = {"anon30": 30, "nodes50": 50}

# Palette aligned with paper/neurips_ace_2026/figs/ace_palette.tex.
COLORS = {
    ("anon30", "ace"):          "#9F1239",  # aceRubyDk  (ACE)
    ("anon30", "zero_shot_lm"): "#15803D",  # aceLeafDk  (ACE w/o DPO)
    ("nodes50", "zero_shot_lm"):"#0369A1",  # aceSkyDk   (ACE w/o DPO @ scale)
    ("nodes50", "ace"):         "#475569",  # aceSlateDk (partial)
}
# "ACE w/o DPO" == --no_dpo ablation (LM proposer + lookahead, no weight
# updates). Deliberately NOT "zero-shot LM" to avoid implying no-training wins.
LABEL = {
    ("anon30", "ace"):          "ACE (anon-30)",
    ("anon30", "zero_shot_lm"): "ACE w/o DPO (anon-30)",
    ("nodes50", "zero_shot_lm"):"ACE w/o DPO (50-node)",
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
    # Convergence annotation: mean episode at which best-MSE was first reached
    # for anon30 ACE. Documents that best-MSE had plateaued before the wall-time
    # cut-off, so the (truncated) anon30 ACE vs ablation comparison is fair.
    ace_cells = agg[(agg["condition"] == "anon30") & (agg["method"] == "ace")]
    if "best_episode" in ace_cells.columns:
        be = pd.to_numeric(ace_cells["best_episode"], errors="coerce").dropna()
        if len(be):
            be_mean = float(be.mean())
            axA.axvline(be_mean, color="#475569", ls="--", lw=1.3)
            ymax = axA.get_ylim()[1]
            axA.text(be_mean, ymax * 0.96,
                     f"best-MSE plateau\n(~ep {be_mean:.0f})",
                     fontsize=8, color="#475569", ha="left", va="top")
    axA.set_xlabel("Episode")
    axA.set_ylabel("Best total mechanism MSE")
    axA.set_title("(a) Anonymised 30-node: ACE vs ACE w/o DPO")
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.grid(alpha=0.25)

    # ---- Panel B: best-MSE bars with per-seed scatter ------------------------
    cells = [("anon30", "ace"), ("anon30", "zero_shot_lm"),
             ("nodes50", "zero_shot_lm")]
    if args.include_nodes50_ace:
        cells.append(("nodes50", "ace"))

    def per_node_values(cell, cond):
        """Per-node best MSE for a cell: prefer the aggregator column, else
        divide total best_mse by the condition node count."""
        if "best_mse_per_node" in cell.columns:
            v = pd.to_numeric(cell["best_mse_per_node"], errors="coerce").dropna().values
            if len(v):
                return v
        tot = pd.to_numeric(cell["best_mse"], errors="coerce").dropna().values
        return tot / N_NODES.get(cond, 1)

    xs, means, stds, labels, colors, scatter = [], [], [], [], [], []
    for i, (cond, method) in enumerate(cells):
        cell = agg[(agg["condition"] == cond) & (agg["method"] == method)]
        vals = per_node_values(cell, cond)
        if len(vals) == 0:
            continue
        xs.append(i)
        means.append(vals.mean())
        stds.append(vals.std() if len(vals) > 1 else 0.0)
        labels.append(LABEL[(cond, method)])
        colors.append(COLORS[(cond, method)])
        scatter.append(vals)

    bars = axB.bar(xs, means, yerr=stds, capsize=4, color=colors,
                   alpha=0.85, edgecolor="black", linewidth=0.6,
                   error_kw={"elinewidth": 1.2})
    # Hatch the partial nodes50 ACE bar if shown.
    if args.include_nodes50_ace and ("nodes50", "ace") in cells:
        bars[-1].set_hatch("//")
    for xi, vals in zip(xs, scatter):
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.25
        axB.scatter(np.full(len(vals), xi) + jitter, vals, s=22,
                    color="black", alpha=0.6, zorder=3)
    # Separator + annotation between the 30-node and 50-node groups, making it
    # explicit that nodes50 is a larger/harder graph, not a worse method.
    n30 = sum(1 for (c, _m) in cells[:len(xs)] if c == "anon30")
    if 0 < n30 < len(xs):
        axB.axvline(n30 - 0.5, color="#94a3b8", ls=":", lw=1.0)
        axB.text(n30 - 0.5, axB.get_ylim()[1] * 0.98, "  50-node (larger graph)",
                 fontsize=8, color="#64748b", ha="left", va="top")
    axB.set_xticks(xs)
    axB.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    axB.set_ylabel("Best PER-NODE mechanism MSE")
    axB.set_title("(b) Per-node best MSE (points = seeds)")
    axB.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    out_dir = "paper/neurips_ace_2026/figs"
    if not os.path.isdir(out_dir):
        out_dir = ROOT
    base = os.path.join(out_dir, "fig_followup")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"Wrote {base}.pdf and {base}.png")

    # Console summary for quick reading (per-node, matching Panel b).
    print("\nBest PER-NODE MSE per cell (N seeds, mean +/- std):")
    for (cond, method) in cells:
        cell = agg[(agg["condition"] == cond) & (agg["method"] == method)]
        vals = per_node_values(cell, cond)
        if len(vals):
            print(f"  {cond:<8} {method:<13} N={len(vals)}  "
                  f"{vals.mean():.4f} +/- {vals.std() if len(vals)>1 else 0:.4f}")


if __name__ == "__main__":
    main()
