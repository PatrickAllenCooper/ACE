#!/usr/bin/env python3
"""
Aggregate the model-scale sweep under results/curc_model_scale_sweep/
(jobs/curc_submit_model_scale_sweep.sh), enabled by the Aug 2026 Alpine
expansion (H200/RTX Pro 6000).

Two phases, sharing the same seed-directory layout as the N-scaling sweep
(scripts/analysis/scaling_common.py), just rooted per model tag instead of
directly per scale:

  Phase A (LM-prior capability sweep): results/.../phaseA/<tag>/nodes<N>/
           zero_shot_lm/seed_<seed>/   for tag in {0.5B,1.5B,3B,7B,14B,32B},
           N in {5,30}
  Phase B (DPO marginal gain vs scale): results/.../phaseB/<tag>/nodes30/
           ace/seed_<seed>/            for tag in {0.5B,1.5B,3B}
           (Phase A's zero_shot_lm@N=30 for the same tags is the no-DPO
           comparison point -- reused here, not re-run.)

Usage:
  python scripts/analysis/aggregate_model_scale_sweep.py
  python scripts/analysis/aggregate_model_scale_sweep.py --root results/curc_model_scale_sweep
"""
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scaling_common import discover_seeds, metrics_cost, summarize_seed_dir

# Model tag -> approximate parameter count (billions), for the log-scale x-axis.
MODEL_TAG_PARAMS_B = {
    "0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0,
}
PHASE_A_TAGS = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
PHASE_A_SCALES = [5, 30]
PHASE_B_TAGS = ["0.5B", "1.5B", "3B"]


def collect_rows(root):
    rows = []
    for tag in PHASE_A_TAGS:
        tag_root = os.path.join(root, "phaseA", tag)
        for scale in PHASE_A_SCALES:
            seeds = discover_seeds(tag_root, scale, "zero_shot_lm")
            for seed in seeds:
                seed_dir = os.path.join(tag_root, f"nodes{scale}", "zero_shot_lm", f"seed_{seed}")
                summ = summarize_seed_dir(seed_dir)
                cost = metrics_cost(seed_dir)
                rows.append(_row("A", tag, scale, "zero_shot_lm", seed, summ, cost))
    for tag in PHASE_B_TAGS:
        tag_root = os.path.join(root, "phaseB", tag)
        seeds = discover_seeds(tag_root, 30, "ace")
        for seed in seeds:
            seed_dir = os.path.join(tag_root, "nodes30", "ace", f"seed_{seed}")
            summ = summarize_seed_dir(seed_dir)
            cost = metrics_cost(seed_dir)
            rows.append(_row("B", tag, 30, "ace", seed, summ, cost))
    return rows


def _row(phase, tag, scale, method, seed, summ, cost):
    row = {
        "phase": phase, "model_tag": tag,
        "model_params_b": MODEL_TAG_PARAMS_B.get(tag, ""),
        "scale": scale, "method": method, "seed": seed,
        "status": "ok" if summ else "missing",
    }
    if summ:
        row.update({
            "n_episodes": summ["n_ep"],
            "n_nodes": summ["n_nodes"],
            "best_episode": summ["best_episode"],
            "best_mse_per_node": f"{summ['per_node_best']:.6f}",
            "final_mse_per_node": f"{summ['per_node_final']:.6f}",
            **{k: (f"{v:.4f}" if v == v else "") for k, v in cost.items()},
        })
    else:
        row.update({
            "n_episodes": 0, "n_nodes": 0, "best_episode": "",
            "best_mse_per_node": "", "final_mse_per_node": "",
            "prompt_tokens_mean": "", "peak_vram_gb": "",
        })
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/curc_model_scale_sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(args.root)
    out = args.out or os.path.join(args.root, "aggregate.csv")
    fieldnames = [
        "phase", "model_tag", "model_params_b", "scale", "method", "seed",
        "n_episodes", "n_nodes", "best_episode", "best_mse_per_node",
        "final_mse_per_node", "prompt_tokens_mean", "peak_vram_gb",
        "status",
    ]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {out} ({len(rows)} rows)\n")

    print("Phase A (zero-shot LM-prior capability): mean +/- std per-node best MSE")
    print(f"{'tag':>6} {'N':>4} {'n':>3} {'mean':>10} {'std':>9}")
    print("-" * 40)
    for tag in PHASE_A_TAGS:
        for scale in PHASE_A_SCALES:
            cell = [r for r in rows if r["phase"] == "A" and r["model_tag"] == tag
                    and r["scale"] == scale and r["status"] == "ok"]
            if not cell:
                print(f"{tag:>6} {scale:>4} {0:>3}  (no data)")
                continue
            vals = [float(r["best_mse_per_node"]) for r in cell]
            mean = st.mean(vals)
            std = st.stdev(vals) if len(vals) > 1 else 0.0
            flag = " under-seeded" if len(vals) < 3 else ""
            print(f"{tag:>6} {scale:>4} {len(vals):>3} {mean:>10.4f} {std:>9.4f}{flag}")

    print("\nPhase B (DPO vs. no-DPO at N=30): mean best-MSE/node, DPO arm vs. "
          "Phase A's zero_shot_lm@30 for the same tag")
    print(f"{'tag':>6} {'dpo_mean':>10} {'no_dpo_mean':>12} {'delta':>9} {'n_dpo':>6} {'n_nodpo':>8}")
    print("-" * 60)
    for tag in PHASE_B_TAGS:
        dpo_cell = [r for r in rows if r["phase"] == "B" and r["model_tag"] == tag
                    and r["status"] == "ok"]
        nodpo_cell = [r for r in rows if r["phase"] == "A" and r["model_tag"] == tag
                      and r["scale"] == 30 and r["status"] == "ok"]
        if not dpo_cell or not nodpo_cell:
            print(f"{tag:>6}  (incomplete: dpo n={len(dpo_cell)}, no_dpo n={len(nodpo_cell)})")
            continue
        dpo_mean = st.mean(float(r["best_mse_per_node"]) for r in dpo_cell)
        nodpo_mean = st.mean(float(r["best_mse_per_node"]) for r in nodpo_cell)
        print(f"{tag:>6} {dpo_mean:>10.4f} {nodpo_mean:>12.4f} "
              f"{dpo_mean - nodpo_mean:>9.4f} {len(dpo_cell):>6} {len(nodpo_cell):>8}")


if __name__ == "__main__":
    main()
