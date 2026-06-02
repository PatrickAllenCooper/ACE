#!/usr/bin/env python3
"""
Main-text scaling figure: PER-NODE best mechanism MSE vs graph size N.

Message: "ACE scales to larger N without architectural change." Per-node
normalisation (total MSE / N) makes scales comparable, so a larger graph is not
penalised by the mechanical growth of the summed total -- this is the fix for
"50 looks worse" in the old fig_followup.

Lines (per-node best MSE, mean +/- std over seeds):
  - ACE                       (full ACE pipeline)
  - ACE w/o DPO (LM+lookahead) (--no_dpo ablation; NOT naive zero-shot)
  - Random                    (passive reference)

Consistent hierarchical family (LargeScaleSCM) at N in {15,30,50} from the
scaling sweep. The N=5 diagnostic (bespoke SCM, different family) is drawn as a
faint, separately-labelled ACE anchor if --show-n5 is passed.

Reads (robust to both result formats):
  results/scaling/nodes<N>/<method>/seed_<s>/**/node_losses.csv   (ace, zero_shot_lm)
  results/scaling/nodes<N>/<method>/seed_<s>/summary.csv          (random/baselines)

Writes:
  paper/neurips_ace_2026/figs/fig_scaling.pdf (+ .png)
  (falls back to <root>/fig_scaling.* if the paper dir is absent)

Usage:
  python scripts/analysis/plot_scaling.py
  python scripts/analysis/plot_scaling.py --root results/scaling --scales 15 30 50 --show-n5
"""
import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


METHODS = ["ace", "zero_shot_lm", "random"]
LABEL = {
    "ace":          "ACE",
    "zero_shot_lm": "ACE w/o DPO (LM+lookahead)",
    "random":       "Random",
}
COLOR = {
    "ace":          "#9F1239",  # aceRubyDk
    "zero_shot_lm": "#15803D",  # aceLeafDk
    "random":       "#64748B",  # aceSlate
}
MARKER = {"ace": "o", "zero_shot_lm": "s", "random": "^"}

# Paper's bespoke 5-node diagnostic ACE anchor (different SCM family): best
# total MSE ~0.61 median over 5 nodes. Shown only with --show-n5.
N5_ACE_PER_NODE = 0.61 / 5.0


def best_per_node_from_node_losses(path):
    df = pd.read_csv(path)
    if "episode" not in df.columns or "total_loss" not in df.columns:
        return None
    n_nodes = sum(1 for c in df.columns if c.startswith("loss_"))
    per_ep = df.groupby("episode")["total_loss"].last()
    if per_ep.empty or n_nodes == 0:
        return None
    return float(per_ep.min()) / n_nodes


def best_per_node_from_summary(path):
    df = pd.read_csv(path)
    if "min_total_loss" not in df.columns or "n_nodes" not in df.columns:
        return None
    row = df.iloc[0]
    n = float(row["n_nodes"]) or 1.0
    return float(row["min_total_loss"]) / n


def cell_per_node(root, scale, method):
    """Return a list of per-node best MSE values across seeds for one cell."""
    base = os.path.join(root, f"nodes{scale}", method)
    if not os.path.isdir(base):
        return []
    vals = []
    for seed_dir in sorted(glob(os.path.join(base, "seed_*"))):
        # ace / zero_shot_lm -> node_losses.csv (deepest/most recent)
        nls = glob(os.path.join(seed_dir, "**", "node_losses.csv"), recursive=True)
        v = None
        if nls:
            nls.sort(key=os.path.getmtime)
            v = best_per_node_from_node_losses(nls[-1])
        if v is None:
            # baseline runner -> summary.csv
            summ = glob(os.path.join(seed_dir, "**", "summary.csv"), recursive=True)
            if summ:
                summ.sort(key=os.path.getmtime)
                v = best_per_node_from_summary(summ[-1])
        if v is not None and np.isfinite(v):
            vals.append(v)
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/scaling")
    ap.add_argument("--scales", type=int, nargs="+", default=[15, 30, 50])
    ap.add_argument("--show-n5", action="store_true",
                    help="Overlay the bespoke 5-node diagnostic ACE anchor.")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    print("Per-node best MSE (mean +/- std over seeds):")
    any_data = False
    for method in METHODS:
        xs, ys, es = [], [], []
        for scale in args.scales:
            vals = cell_per_node(args.root, scale, method)
            if not vals:
                continue
            xs.append(scale)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            print(f"  N={scale:<3} {method:<13} n={len(vals)}  "
                  f"{np.mean(vals):.4f} +/- {(np.std(vals) if len(vals)>1 else 0):.4f}")
        if not xs:
            continue
        any_data = True
        ax.errorbar(xs, ys, yerr=es, marker=MARKER[method], color=COLOR[method],
                    lw=2.0, ms=6, capsize=3, label=LABEL[method])

    if args.show_n5:
        ax.scatter([5], [N5_ACE_PER_NODE], marker="o", s=55,
                   facecolors="none", edgecolors=COLOR["ace"], lw=1.6, zorder=4)
        ax.annotate("ACE (5-node diagnostic,\nseparate SCM family)",
                    xy=(5, N5_ACE_PER_NODE), xytext=(5.5, N5_ACE_PER_NODE * 1.6),
                    fontsize=7.5, color=COLOR["ace"])

    ax.set_xscale("log")
    ax.set_xticks(args.scales + ([5] if args.show_n5 else []))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Graph size N (nodes, log scale)")
    ax.set_ylabel("Best PER-NODE mechanism MSE")
    ax.set_title("ACE scales to larger N without architectural change")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if not any_data:
        print("\nWARNING: no scaling data found under "
              f"{args.root}. Run jobs/curc_submit_scaling.sh first.")

    out_dir = "paper/neurips_ace_2026/figs"
    if not os.path.isdir(out_dir):
        out_dir = args.root
        os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "fig_scaling")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"\nWrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
