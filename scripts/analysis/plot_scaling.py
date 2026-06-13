#!/usr/bin/env python3
"""
Main-text scaling figure: PER-NODE best mechanism MSE vs graph size N.

Message: LM-driven intervention (ACE and ACE w/o DPO) stays well below passive
Random sampling as N grows; passive heuristics degrade.  ACE and ACE w/o DPO
form a tight band -- the pretrained LM prior plus lookahead carries the load at
this scale (DPO is a refinement, not the headline).

Per-node normalisation (total MSE / N) makes scales comparable.

Reads:
  results/scaling/nodes<N>/<method>/seed_<s>/**/node_losses.csv   (ace, zero_shot_lm)
  results/scaling/nodes<N>/<method>/seed_<s>/**/summary.csv      (random)

Writes:
  paper/neurips_ace_2026/figs/fig_scaling.pdf (+ .png)

Usage:
  python scripts/analysis/plot_scaling.py
  python scripts/analysis/plot_scaling.py --root results/scaling --scales 15 30 50
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
    "ace":          "#9F1239",
    "zero_shot_lm": "#15803D",
    "random":       "#64748B",
}
MARKER = {"ace": "o", "zero_shot_lm": "s", "random": "^"}
LM_METHODS = ("ace", "zero_shot_lm")


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
    if "min_total_loss" not in df.columns:
        return None
    row = df.iloc[0]
    n = float(row.get("n_nodes", 1)) or 1.0
    return float(row["min_total_loss"]) / n


def discover_seeds(base):
    if not os.path.isdir(base):
        return []
    seeds = []
    for d in glob(os.path.join(base, "seed_*")):
        try:
            seeds.append(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(seeds))


def cell_per_node(root, scale, method, seeds=None):
    base = os.path.join(root, f"nodes{scale}", method)
    if not os.path.isdir(base):
        return []
    seed_list = seeds or discover_seeds(base)
    vals = []
    for seed in seed_list:
        seed_dir = os.path.join(base, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        nls = glob(os.path.join(seed_dir, "**", "node_losses.csv"), recursive=True)
        v = None
        if nls:
            nls.sort(key=os.path.getmtime)
            v = best_per_node_from_node_losses(nls[-1])
        if v is None:
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
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Restrict to these seeds (default: all seed_* dirs)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # Collect LM-method curves first so we can shade the band.
    lm_y_by_scale = {}
    series = {}

    print("Per-node best MSE (mean +/- std over seeds):")
    for method in METHODS:
        xs, ys, es, ns = [], [], [], []
        for scale in args.scales:
            vals = cell_per_node(args.root, scale, method, args.seeds)
            if not vals:
                continue
            xs.append(scale)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            ns.append(len(vals))
            print(f"  N={scale:<3} {method:<13} n={len(vals)}  "
                  f"{np.mean(vals):.4f} +/- {(np.std(vals) if len(vals)>1 else 0):.4f}")
        series[method] = (xs, ys, es, ns)
        if method in LM_METHODS:
            for x, y in zip(xs, ys):
                lm_y_by_scale.setdefault(x, []).append(y)

    # Shaded LM-driven band (ACE + ACE w/o DPO).
    band_scales = sorted(lm_y_by_scale.keys())
    if band_scales:
        lo = [min(lm_y_by_scale[s]) for s in band_scales]
        hi = [max(lm_y_by_scale[s]) for s in band_scales]
        ax.fill_between(band_scales, lo, hi, color="#86EFAC", alpha=0.35,
                        label="LM + lookahead (ACE band)", zorder=1)

    for method in METHODS:
        xs, ys, es, _ = series.get(method, ([], [], [], []))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, marker=MARKER[method], color=COLOR[method],
                    lw=2.0, ms=6, capsize=3, label=LABEL[method], zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(args.scales)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Graph size $N$ (nodes, log scale)")
    ax.set_ylabel("Best PER-NODE mechanism MSE")
    ax.set_title("LM-driven intervention scales with $N$;\npassive sampling degrades")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(
        0.98, 0.02,
        "Fixed $\\sim$40-ep plateau budget per cell;\n"
        "consistent LargeScaleSCM family",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        color="#475569",
    )
    fig.tight_layout()

    if not any(series.get(m, ([], [], [], []))[0] for m in METHODS):
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
