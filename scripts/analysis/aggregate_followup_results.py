#!/usr/bin/env python3
"""
Aggregate the curc_30node_followup tree directly from `node_losses.csv`.

The May 2026 followup batch never wrote `summary.csv` because most jobs hit
the SLURM wall-time boundary during post-training analysis (training itself
ran to completion or near-completion). This script reads `node_losses.csv`
for every (condition, method, seed) under `results/curc_30node_followup/`
and computes the same Best MSE / Final MSE / N_episodes triples that
plot_30node_results.py uses for the rebuttal table.

Convention (matches scripts/analysis/plot_30node_results.py:138-142):
- per_episode_raw = node_losses.groupby("episode")["total_loss"].last()
  i.e. the total mechanism MSE at the *final* step of each episode.
- Best MSE       = min(per_episode_raw)
- Final MSE      = last(per_episode_raw)   (raw, no cummin)
- N_episodes     = number of unique episodes in the file

If multiple jobs exist under one (cond,method,seed) (e.g. resubmit + earlier
partial), the most recently-modified `node_losses.csv` is used.

Outputs:
  results/curc_30node_followup/aggregate.csv

Usage:
  python scripts/analysis/aggregate_followup_results.py
  python scripts/analysis/aggregate_followup_results.py --root <path>
"""
import argparse
import csv
import os
import sys
from glob import glob

import pandas as pd


CONDITIONS = ["anon30", "nodes50"]
METHODS = ["ace", "zero_shot_lm"]
SEEDS = [42, 123, 456, 789, 1011]


def latest_node_losses(seed_dir: str):
    """Return the most-recent node_losses.csv path under seed_dir, or None."""
    candidates = glob(os.path.join(seed_dir, "**", "node_losses.csv"),
                      recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def summarize(nl_path: str):
    """Compute summary stats from a node_losses.csv.

    Returns a dict with best/final total MSE, the *per-node* equivalents
    (scale-invariant for cross-N comparison), the node count, and the episode
    at which best-MSE was first reached (convergence evidence for the prose
    caveat).
    """
    df = pd.read_csv(nl_path)
    if "episode" not in df.columns or "total_loss" not in df.columns:
        return None
    # n_nodes inferred from the per-node loss columns (robust to canonical
    # X1..XN names and anonymised n_xxxx tokens alike).
    n_nodes = sum(1 for c in df.columns if c.startswith("loss_"))
    per_ep = (df.groupby("episode")["total_loss"].last()
                .reset_index()
                .sort_values("episode"))
    if per_ep.empty:
        return None
    best = float(per_ep["total_loss"].min())
    final = float(per_ep["total_loss"].iloc[-1])
    n_ep = int(per_ep["episode"].nunique())
    # Episode at which the running best-MSE was first achieved (a fair, honest
    # convergence indicator even when a run was wall-time truncated).
    best_ep = int(per_ep.loc[per_ep["total_loss"].idxmin(), "episode"])
    per_node = (best / n_nodes) if n_nodes else float("nan")
    per_node_final = (final / n_nodes) if n_nodes else float("nan")
    return {
        "best": best, "final": final, "n_ep": n_ep,
        "n_nodes": n_nodes, "best_episode": best_ep,
        "per_node_best": per_node, "per_node_final": per_node_final,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/curc_30node_followup",
                        help="Root of the followup results tree")
    parser.add_argument("--out", default=None,
                        help="Output CSV (default: <root>/aggregate.csv)")
    args = parser.parse_args()

    root = args.root
    out = args.out or os.path.join(root, "aggregate.csv")

    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for cond in CONDITIONS:
        for method in METHODS:
            for seed in SEEDS:
                seed_dir = os.path.join(root, cond, method, f"seed_{seed}")
                nl = latest_node_losses(seed_dir) if os.path.isdir(seed_dir) else None
                if nl is None:
                    rows.append({
                        "condition": cond, "method": method, "seed": seed,
                        "node_losses_csv": "",
                        "n_episodes": 0, "n_nodes": 0, "best_episode": "",
                        "best_mse": "", "final_mse": "",
                        "best_mse_per_node": "", "final_mse_per_node": "",
                        "status": "missing",
                    })
                    continue
                summ = summarize(nl)
                if summ is None:
                    rows.append({
                        "condition": cond, "method": method, "seed": seed,
                        "node_losses_csv": nl,
                        "n_episodes": 0, "n_nodes": 0, "best_episode": "",
                        "best_mse": "", "final_mse": "",
                        "best_mse_per_node": "", "final_mse_per_node": "",
                        "status": "empty_or_malformed",
                    })
                    continue
                rows.append({
                    "condition": cond, "method": method, "seed": seed,
                    "node_losses_csv": nl,
                    "n_episodes": summ["n_ep"],
                    "n_nodes": summ["n_nodes"],
                    "best_episode": summ["best_episode"],
                    "best_mse": f"{summ['best']:.6f}",
                    "final_mse": f"{summ['final']:.6f}",
                    "best_mse_per_node": f"{summ['per_node_best']:.6f}",
                    "final_mse_per_node": f"{summ['per_node_final']:.6f}",
                    "status": "ok",
                })

    fieldnames = ["condition", "method", "seed", "n_episodes", "n_nodes",
                  "best_episode", "best_mse", "final_mse",
                  "best_mse_per_node", "final_mse_per_node",
                  "status", "node_losses_csv"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {out} ({len(rows)} rows)\n")

    print(f"{'condition':<8} {'method':<14} {'seed':>5} {'n_ep':>5} {'nN':>4} "
          f"{'bestEp':>6} {'best_mse':>10} {'final_mse':>10} {'best/node':>10}  status")
    print("-" * 92)
    for row in rows:
        print(f"{row['condition']:<8} {row['method']:<14} "
              f"{row['seed']:>5} {row['n_episodes']:>5} {row.get('n_nodes', 0):>4} "
              f"{str(row.get('best_episode', '')):>6} "
              f"{row['best_mse']:>10} {row['final_mse']:>10} "
              f"{row.get('best_mse_per_node', ''):>10}  {row['status']}")

    print("\nPer-cell aggregate (mean ± std over seeds with status='ok'):")
    print(f"{'condition':<8} {'method':<14} {'N':>3} "
          f"{'best_mean':>10} {'best_std':>10} "
          f"{'pernode_mean':>13} "
          f"{'n_ep_mean':>10} {'bestEp_mean':>11}")
    print("-" * 92)
    import statistics as st
    for cond in CONDITIONS:
        for method in METHODS:
            cell = [r for r in rows
                    if r["condition"] == cond and r["method"] == method
                    and r["status"] == "ok"]
            n = len(cell)
            if n == 0:
                print(f"{cond:<8} {method:<14} {0:>3}  (no usable data)")
                continue
            bests = [float(r["best_mse"]) for r in cell]
            per_nodes = [float(r["best_mse_per_node"]) for r in cell
                         if r.get("best_mse_per_node") not in ("", None)]
            n_eps = [int(r["n_episodes"]) for r in cell]
            best_eps = [int(r["best_episode"]) for r in cell
                        if str(r.get("best_episode", "")).strip() != ""]
            bm, bs = st.mean(bests), (st.stdev(bests) if n > 1 else 0.0)
            pnm = st.mean(per_nodes) if per_nodes else float("nan")
            em = st.mean(n_eps)
            bem = st.mean(best_eps) if best_eps else float("nan")
            print(f"{cond:<8} {method:<14} {n:>3} "
                  f"{bm:>10.4f} {bs:>10.4f} "
                  f"{pnm:>13.4f} "
                  f"{em:>10.1f} {bem:>11.1f}")


if __name__ == "__main__":
    main()
