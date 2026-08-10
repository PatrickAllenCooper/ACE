#!/usr/bin/env python3
"""
Aggregate the budget-fairness decision-gate suite (5-node or 30-node) into a
single comparison table: ACE under both lookahead accountings (env-based and
student-based) versus every baseline, all at the same total-environment-query
budget.

This is the actual replacement for the NeurIPS 2026 Table 1 comparison that
4 of 5 reviewers flagged as episode-matched rather than query-matched (see
docs/development/guidance/current_status.txt, "BLOCKING" section). It does
not decide anything on its own -- it prints the numbers the decision gate
described in that document is conditioned on.

Usage:
    python scripts/analysis/aggregate_budget_fairness.py \
        --root /scratch/alpine/paco0228/ACE/results/curc_30node_budget_fairness

    python scripts/analysis/aggregate_budget_fairness.py \
        --root /scratch/alpine/paco0228/ACE/results/curc_5node_budget_fairness \
        --out results/curc_5node_budget_fairness_aggregate.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scaling_common import summarize_seed_dir  # noqa: E402


def find_query_total(seed_dir: str) -> float | None:
    matches = sorted(glob.glob(os.path.join(seed_dir, "**", "query_budget.json"), recursive=True))
    if not matches:
        return None
    # Most-recently-modified in case of checkpoint-resume duplicates.
    matches.sort(key=os.path.getmtime)
    with open(matches[-1]) as fh:
        summary = json.load(fh)
    return summary.get("total", {}).get("samples")


def collect_method(root: str, method_dir: str, label: str) -> list[dict]:
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed_*"))):
        seed = os.path.basename(seed_dir).split("_", 1)[-1]
        summary = summarize_seed_dir(seed_dir)
        if summary is None:
            print(f"  WARNING: no node_losses.csv/summary.csv found under {seed_dir}", file=sys.stderr)
            continue
        query_total = find_query_total(seed_dir)
        rows.append({
            "method": label,
            "seed": seed,
            "per_node_best": summary["per_node_best"],
            "per_node_final": summary["per_node_final"],
            "n_nodes": summary["n_nodes"],
            "n_episodes": summary["n_ep"],
            "query_total": query_total,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                     help="curc_5node_budget_fairness or curc_30node_budget_fairness results root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for mode in ("ace_env", "ace_student"):
        mode_dir = os.path.join(root, mode)
        if os.path.isdir(mode_dir):
            all_rows += collect_method(root, mode_dir, mode)

    baselines_root = os.path.join(root, "baselines")
    if os.path.isdir(baselines_root):
        for method_dir in sorted(glob.glob(os.path.join(baselines_root, "*"))):
            if not os.path.isdir(method_dir) or os.path.basename(method_dir) == "logs":
                continue
            all_rows += collect_method(root, method_dir, os.path.basename(method_dir))

    if not all_rows:
        print(f"No results found under {root}", file=sys.stderr)
        sys.exit(1)

    out = args.out or os.path.join(root, "budget_fairness_aggregate.csv")
    fieldnames = ["method", "seed", "per_node_best", "per_node_final",
                  "n_nodes", "n_episodes", "query_total"]
    import csv
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {out}\n")

    methods = sorted(set(r["method"] for r in all_rows))
    print("=== Decision-gate comparison: per-node best MSE, query-budget-matched ===")
    print(f"{'Method':<16} {'n_seeds':>8} {'mean_best':>12} {'std_best':>10} "
          f"{'mean_final':>12} {'mean_queries':>14}")
    print("-" * 78)
    summary_by_method = {}
    for method in methods:
        vals = [r["per_node_best"] for r in all_rows if r["method"] == method]
        finals = [r["per_node_final"] for r in all_rows if r["method"] == method]
        queries = [r["query_total"] for r in all_rows if r["method"] == method and r["query_total"] is not None]
        mean_best = st.mean(vals)
        std_best = st.stdev(vals) if len(vals) > 1 else 0.0
        mean_final = st.mean(finals)
        mean_q = st.mean(queries) if queries else float("nan")
        summary_by_method[method] = mean_best
        print(f"{method:<16} {len(vals):>8} {mean_best:>12.4f} {std_best:>10.4f} "
              f"{mean_final:>12.4f} {mean_q:>14.0f}")

    print("\n=== Decision gate ===")
    baseline_methods = [m for m in methods if not m.startswith("ace_")]
    if baseline_methods:
        best_baseline = min(baseline_methods, key=lambda m: summary_by_method[m])
        best_baseline_val = summary_by_method[best_baseline]
        for ace_mode in ("ace_env", "ace_student"):
            if ace_mode not in summary_by_method:
                continue
            ace_val = summary_by_method[ace_mode]
            if ace_val < best_baseline_val:
                pct = (best_baseline_val - ace_val) / best_baseline_val * 100
                print(f"  {ace_mode}: WINS vs best baseline ({best_baseline}, "
                      f"{best_baseline_val:.4f}) by {pct:.1f}% at matched query budget")
            else:
                pct = (ace_val - best_baseline_val) / best_baseline_val * 100
                print(f"  {ace_mode}: LOSES to best baseline ({best_baseline}, "
                      f"{best_baseline_val:.4f}) by {pct:.1f}% at matched query budget")
    else:
        print("  No baseline results found yet -- cannot resolve the decision gate.")


if __name__ == "__main__":
    main()
