#!/usr/bin/env python3
"""
Aggregate query_budget.json files (written by ace_experiments.py, baselines.py,
and run_30node_baseline_seed.py) under a results directory and print the
per-method, per-tag environment-query totals.

This is the tool for the ICLR resubmission's budget-fairness decision gate:
it answers "how many ground-truth environment samples did each method
actually consume", broken down by tag (executed / lookahead / candidate_probe
/ observational), across all seeds found.

Usage:
    # Phase 1 decision: what QUERY_BUDGET should Phase 2 baselines use?
    python scripts/runners/aggregate_query_budget.py \
        results/curc_5node_budget_fairness --group-by ace_env ace_student

    # Phase 2 check: did the matched-budget baselines actually land near the
    # target budget (query_budget stops between episodes, not mid-episode)?
    python scripts/runners/aggregate_query_budget.py \
        results/curc_5node_budget_fairness/baselines --group-by "*"
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def find_query_budget_files(root):
    return sorted(glob.glob(os.path.join(root, "**", "query_budget.json"), recursive=True))


def method_key_from_path(root, path, group_by):
    """
    Best-effort extraction of a method/condition label from a query_budget.json
    path, using the first path component (relative to root) that matches one
    of the requested group labels, or "*" to accept any top-level component.
    """
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if "*" in group_by:
        return parts[0] if parts else "unknown"
    for part in parts:
        if part in group_by:
            return part
    return parts[0] if parts else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Aggregate query_budget.json files for budget-fairness reporting")
    parser.add_argument("root", type=str, help="Results directory to scan recursively")
    parser.add_argument("--group-by", nargs="+", default=["*"],
                         help="Path component(s) identifying method/condition "
                              "(e.g. ace_env ace_student random round_robin "
                              "max_variance bayesian_oed dpo sft_best ranking). "
                              "Use '*' (default) to group by first path component.")
    args = parser.parse_args()

    files = find_query_budget_files(args.root)
    if not files:
        print(f"No query_budget.json files found under {args.root}")
        sys.exit(1)

    per_method_totals = defaultdict(list)
    per_method_tag_totals = defaultdict(lambda: defaultdict(list))

    for f in files:
        with open(f) as fh:
            summary = json.load(fh)
        method = method_key_from_path(args.root, f, args.group_by)
        total_samples = summary.get("total", {}).get("samples", 0)
        per_method_totals[method].append(total_samples)
        for tag, bucket in summary.items():
            if tag == "total":
                continue
            per_method_tag_totals[method][tag].append(bucket.get("samples", 0))

    print(f"Found {len(files)} query_budget.json files under {args.root}\n")

    print("=== Total environment samples per method (mean +/- std, N runs) ===")
    for method in sorted(per_method_totals):
        vals = per_method_totals[method]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {method:20s} mean={mean:>10.1f}  std={std:>8.1f}  n={len(vals)}")

    print("\n=== Per-tag breakdown (mean samples per run) ===")
    for method in sorted(per_method_tag_totals):
        print(f"  {method}:")
        for tag in sorted(per_method_tag_totals[method]):
            vals = per_method_tag_totals[method][tag]
            print(f"    {tag:20s} mean={np.mean(vals):>10.1f}  n={len(vals)}")

    print("\n=== Suggested --query_budget values (integer mean total, rounded) ===")
    for method in sorted(per_method_totals):
        print(f"  {method:20s} --query_budget {int(round(np.mean(per_method_totals[method])))}")


if __name__ == "__main__":
    main()
