#!/usr/bin/env python3
"""
Aggregate 30-node baseline results and print the paper table numbers.

Run after all CURC jobs complete:
    python scripts/runners/aggregate_30node_baselines.py

Prints ready-to-paste LaTeX rows for the 30-node table.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

RESULTS_ROOT = "results/curc_30node_baselines"
METHODS = ["random", "round_robin", "max_variance"]
SEEDS = [42, 123, 456, 789, 1011]


def load_summaries():
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = os.path.join(RESULTS_ROOT, method, f"seed_{seed}", "summary.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                rows.append(df.iloc[0].to_dict())
            else:
                print(f"  MISSING: {path}")
    return pd.DataFrame(rows)


def main():
    print(f"Loading from: {RESULTS_ROOT}")
    df = load_summaries()

    if df.empty:
        print("No results found.")
        sys.exit(1)

    print("\n=== Per-seed final total loss ===")
    pivot = df.pivot_table(
        index="seed", columns="method", values="final_total_loss")
    print(pivot.to_string())

    print("\n=== Summary (mean +/- std, N seeds) ===")
    summary = df.groupby("method")["final_total_loss"].agg(
        mean="mean", std="std", n="count")
    print(summary.to_string())

    print("\n=== LaTeX table rows (paste into paper) ===")
    method_labels = {
        "random":       "Random",
        "round_robin":  "Round-Robin",
        "max_variance": "Max-Variance",
    }
    for method in METHODS:
        grp = df[df["method"] == method]
        if grp.empty:
            print(f"% {method_labels[method]}: no data")
            continue
        m = grp["final_total_loss"].mean()
        s = grp["final_total_loss"].std(ddof=1)
        n = len(grp)
        ratio = 11.286 / m  # improvement over Random (ACE CURC result)
        print(f"{method_labels[method]} & {m:.2f}$\\pm${s:.2f} & "
              f"{n} & {ratio:.1f}$\\times$ \\\\")


if __name__ == "__main__":
    main()
