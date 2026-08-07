#!/usr/bin/env python3
"""
Model-scale sweep figure: does mechanism recovery improve monotonically with
LM prior capability, and does DPO's marginal contribution shrink as the prior
strengthens? Enabled by the Aug 2026 Alpine H200/RTX Pro 6000 expansion
(jobs/curc_submit_model_scale_sweep.sh).

Two panels:
  (left)  Phase A -- zero-shot (LM prior + lookahead, no DPO) best per-node
          MSE vs. model size (log-scale params), one line per graph size N.
  (right) Phase B -- DPO vs. no-DPO best per-node MSE at N=30, for the three
          model sizes DPO is tractable at ({0.5B,1.5B,3B}).

Reads: results/curc_model_scale_sweep/aggregate.csv (from
       aggregate_model_scale_sweep.py)
Writes: paper/iclr_ace_2027/figs/fig_model_scale.{pdf,png}

Usage:
  python scripts/analysis/aggregate_model_scale_sweep.py   # first, to build the CSV
  python scripts/analysis/plot_model_scale_sweep.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_TAG_PARAMS_B = {
    "0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0,
}
TAG_ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/curc_model_scale_sweep/aggregate.csv")
    ap.add_argument("--out_dir", default="paper/iclr_ace_2027/figs")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"ERROR: {args.csv} not found. Run "
                          "aggregate_model_scale_sweep.py first.")
    df = pd.read_csv(args.csv)
    df = df[df["status"] == "ok"].copy()
    df["model_params_b"] = df["model_tag"].map(MODEL_TAG_PARAMS_B)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- Panel A: capability scaling law ----------------------------------
    phase_a = df[df["phase"] == "A"]
    colors = {5: "#2563EB", 30: "#9F1239"}
    markers = {5: "o", 30: "s"}
    for scale in sorted(phase_a["scale"].unique()):
        cell = phase_a[phase_a["scale"] == scale]
        xs, ys, es = [], [], []
        for tag in TAG_ORDER:
            vals = cell[cell["model_tag"] == tag]["best_mse_per_node"]
            if vals.empty:
                continue
            xs.append(MODEL_TAG_PARAMS_B[tag])
            ys.append(float(vals.mean()))
            es.append(float(vals.std()) if len(vals) > 1 else 0.0)
        if not xs:
            continue
        ax_a.errorbar(xs, ys, yerr=es, marker=markers.get(scale, "d"),
                      color=colors.get(scale, "#64748B"), lw=2.0, ms=6,
                      capsize=3, label=f"N={scale}")
    ax_a.set_xscale("log")
    ax_a.set_xticks(list(MODEL_TAG_PARAMS_B.values()))
    ax_a.set_xticklabels(list(MODEL_TAG_PARAMS_B.keys()))
    ax_a.set_xlabel("Qwen2.5 policy size (params, log scale)")
    ax_a.set_ylabel("Best per-node mechanism MSE")
    ax_a.set_title("Zero-shot (LM prior + lookahead only):\ncapability scaling law")
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False, fontsize=8.5)

    # ---- Panel B: DPO's marginal contribution vs. scale --------------------
    phase_b = df[df["phase"] == "B"]
    b_tags = [t for t in ["0.5B", "1.5B", "3B"] if not phase_b[phase_b["model_tag"] == t].empty]
    x = np.arange(len(b_tags))
    width = 0.32
    dpo_means, dpo_stds, nodpo_means, nodpo_stds = [], [], [], []
    for tag in b_tags:
        dpo_vals = phase_b[phase_b["model_tag"] == tag]["best_mse_per_node"]
        nodpo_vals = phase_a[(phase_a["model_tag"] == tag) & (phase_a["scale"] == 30)]["best_mse_per_node"]
        dpo_means.append(float(dpo_vals.mean()))
        dpo_stds.append(float(dpo_vals.std()) if len(dpo_vals) > 1 else 0.0)
        nodpo_means.append(float(nodpo_vals.mean()) if not nodpo_vals.empty else np.nan)
        nodpo_stds.append(float(nodpo_vals.std()) if len(nodpo_vals) > 1 else 0.0)
    ax_b.bar(x - width / 2, nodpo_means, width, yerr=nodpo_stds, capsize=3,
              color="#15803D", label="No DPO (LM+lookahead)")
    ax_b.bar(x + width / 2, dpo_means, width, yerr=dpo_stds, capsize=3,
              color="#9F1239", label="ACE (+DPO)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(b_tags)
    ax_b.set_xlabel("Qwen2.5 policy size")
    ax_b.set_ylabel("Best per-node mechanism MSE")
    ax_b.set_title("N=30: DPO's marginal contribution\nvs. prior strength")
    ax_b.grid(alpha=0.25, axis="y")
    ax_b.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "fig_model_scale")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
