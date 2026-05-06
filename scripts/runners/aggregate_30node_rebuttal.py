#!/usr/bin/env python3
"""
Aggregate 30-node rebuttal experiment results.

Run after CURC jobs complete and `scp` is done:
    python scripts/runners/aggregate_30node_rebuttal.py

Prints summary stats and ready-to-paste LaTeX rows.
"""

import os
import glob
import numpy as np
import pandas as pd

ROOT = "results/curc_30node_rebuttal"
SEEDS = [42, 123, 456, 789, 1011]


def load_baseline_summaries(method):
    """For ppo, bayesian_oed: results saved by run_30node_baseline_seed.py."""
    rows = []
    for seed in SEEDS:
        path = f"{ROOT}/{method}/{method}/seed_{seed}/summary.csv"
        if not os.path.exists(path):
            # Try alternate path
            path = f"{ROOT}/{method}/seed_{seed}/summary.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            rows.append(df.iloc[0].to_dict())
        else:
            print(f"  MISSING: {method} seed {seed}")
    return pd.DataFrame(rows)


def load_zero_shot_lm():
    """For zero_shot_lm: results from ace_experiments.py via job-tagged dirs."""
    rows = []
    for seed in SEEDS:
        pattern = f"{ROOT}/zero_shot_lm/seed_{seed}/job_*/run_*/node_losses.csv"
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"  MISSING: zero_shot_lm seed {seed}")
            continue
        df = pd.read_csv(candidates[-1])
        per_ep = df.groupby("episode")["total_loss"].last()
        best = df["total_loss"].min()
        final = per_ep.iloc[-1]
        rows.append({"seed": seed, "best": best, "final": final,
                     "episodes": int(df["episode"].max() + 1)})
    return pd.DataFrame(rows)


def main():
    print(f"Loading from: {ROOT}\n")

    print("=== PPO on 30-node ===")
    ppo = load_baseline_summaries("ppo")
    if not ppo.empty:
        print(ppo[["seed", "final_total_loss", "min_total_loss"]].to_string(index=False))
        m, s = ppo["final_total_loss"].mean(), ppo["final_total_loss"].std(ddof=1)
        print(f"  PPO final mean+/-std: {m:.3f}+/-{s:.3f}")
        m, s = ppo["min_total_loss"].mean(), ppo["min_total_loss"].std(ddof=1)
        print(f"  PPO best  mean+/-std: {m:.3f}+/-{s:.3f}")

    print("\n=== Bayesian OED on 30-node ===")
    boed = load_baseline_summaries("bayesian_oed")
    if not boed.empty:
        print(boed[["seed", "final_total_loss", "min_total_loss"]].to_string(index=False))
        m, s = boed["final_total_loss"].mean(), boed["final_total_loss"].std(ddof=1)
        print(f"  BOED final mean+/-std: {m:.3f}+/-{s:.3f}")
        m, s = boed["min_total_loss"].mean(), boed["min_total_loss"].std(ddof=1)
        print(f"  BOED best  mean+/-std: {m:.3f}+/-{s:.3f}")

    print("\n=== Zero-shot LM (ACE --no_dpo) on 30-node ===")
    zsl = load_zero_shot_lm()
    if not zsl.empty:
        print(zsl.to_string(index=False))
        m, s = zsl["final"].mean(), zsl["final"].std(ddof=1)
        print(f"  ZSL final mean+/-std: {m:.3f}+/-{s:.3f}")
        m, s = zsl["best"].mean(), zsl["best"].std(ddof=1)
        print(f"  ZSL best  mean+/-std: {m:.3f}+/-{s:.3f}")

    print("\n=== LaTeX rows for paper / rebuttal addendum ===")
    for label, df, key_final, key_best in [
        ("PPO",          ppo,  "final_total_loss", "min_total_loss"),
        ("Bayesian OED", boed, "final_total_loss", "min_total_loss"),
        ("Zero-shot LM", zsl,  "final",            "best"),
    ]:
        if df.empty:
            print(f"% {label}: no data")
            continue
        bf = df[key_best].mean()
        bs = df[key_best].std(ddof=1)
        ff = df[key_final].mean()
        fs = df[key_final].std(ddof=1)
        print(f"{label} & {bf:.2f}$\\pm${bs:.2f} & {ff:.2f}$\\pm${fs:.2f} & {len(df)} \\\\")


if __name__ == "__main__":
    main()
