#!/usr/bin/env python3
"""
Aggregate the scaling sweep under results/scaling/.

Usage:
  python scripts/analysis/aggregate_scaling_results.py
  python scripts/analysis/aggregate_scaling_results.py --root results/scaling
"""
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scaling_common import (
    PLATEAU_EP_BUDGET,
    SCALING_METHODS,
    SCALING_SCALES,
    discover_seeds,
    latest_file,
    metrics_cost,
    summarize_seed_dir,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/scaling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root
    out = args.out or os.path.join(root, "aggregate.csv")
    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    all_seeds = set()
    for scale in SCALING_SCALES:
        for method in SCALING_METHODS:
            all_seeds.update(discover_seeds(root, scale, method))
    seeds = sorted(all_seeds)

    rows = []
    for scale in SCALING_SCALES:
        for method in SCALING_METHODS:
            for seed in seeds:
                seed_dir = os.path.join(root, f"nodes{scale}", method, f"seed_{seed}")
                status = "missing"
                summ = None
                src = ""
                cost = {}
                if os.path.isdir(seed_dir):
                    nl = latest_file(seed_dir, "node_losses.csv")
                    sm = latest_file(seed_dir, "summary.csv")
                    summ = summarize_seed_dir(seed_dir)
                    src = nl or sm or ""
                    cost = metrics_cost(seed_dir)
                    if summ:
                        status = "ok"

                row = {"scale": scale, "method": method, "seed": seed, "status": status,
                       "source_csv": src or ""}
                if summ:
                    row.update({
                        "n_episodes": summ["n_ep"],
                        "n_nodes": summ["n_nodes"],
                        "best_episode": summ["best_episode"],
                        "best_mse": f"{summ['best']:.6f}",
                        "final_mse": f"{summ['final']:.6f}",
                        "best_mse_per_node": f"{summ['per_node_best']:.6f}",
                        "final_mse_per_node": f"{summ['per_node_final']:.6f}",
                        **{k: (f"{v:.4f}" if v == v else "")
                           for k, v in cost.items()},
                    })
                else:
                    row.update({
                        "n_episodes": 0, "n_nodes": 0, "best_episode": "",
                        "best_mse": "", "final_mse": "",
                        "best_mse_per_node": "", "final_mse_per_node": "",
                        "prompt_tokens_mean": "", "peak_vram_gb": "",
                    })
                rows.append(row)

    fieldnames = [
        "scale", "method", "seed", "n_episodes", "n_nodes", "best_episode",
        "best_mse", "final_mse", "best_mse_per_node", "final_mse_per_node",
        "prompt_tokens_mean", "peak_vram_gb", "status", "source_csv",
    ]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {out} ({len(rows)} rows)\n")
    print(f"{'scale':>5} {'method':<14} {'seed':>5} {'n_ep':>5} {'bestEp':>6} "
          f"{'best/node':>10}  status")
    print("-" * 60)
    for row in rows:
        if row["status"] != "ok":
            continue
        print(f"{row['scale']:>5} {row['method']:<14} {row['seed']:>5} "
              f"{row.get('n_episodes', 0):>5} {str(row.get('best_episode', '')):>6} "
              f"{row.get('best_mse_per_node', ''):>10}  {row['status']}")

    print("\nPer-cell aggregate (mean +/- std, seeds with status='ok'):")
    print(f"{'scale':>5} {'method':<14} {'N':>3} {'pernode_mean':>12} "
          f"{'pernode_std':>11} {'n_ep_mean':>10} {'bestEp_mean':>11}  flags")
    print("-" * 78)
    for scale in SCALING_SCALES:
        for method in SCALING_METHODS:
            cell = [r for r in rows
                    if r["scale"] == scale and r["method"] == method
                    and r["status"] == "ok"]
            n = len(cell)
            flags = []
            if n < 3:
                flags.append(f"under-seeded(n={n})")
            if n == 0:
                print(f"{scale:>5} {method:<14} {0:>3}  (no data)")
                continue
            per_nodes = [float(r["best_mse_per_node"]) for r in cell]
            n_eps = [int(r["n_episodes"]) for r in cell]
            best_eps = [int(r["best_episode"]) for r in cell
                        if str(r.get("best_episode", "")).strip() != ""]
            pm = st.mean(per_nodes)
            ps = st.stdev(per_nodes) if n > 1 else 0.0
            em = st.mean(n_eps)
            bem = st.mean(best_eps) if best_eps else float("nan")
            if method == "ace" and em < PLATEAU_EP_BUDGET - 5:
                flags.append("below-plateau-budget")
            flag_str = ",".join(flags) if flags else "ok"
            print(f"{scale:>5} {method:<14} {n:>3} {pm:>12.4f} {ps:>11.4f} "
                  f"{em:>10.1f} {bem:>11.1f}  {flag_str}")


if __name__ == "__main__":
    main()
