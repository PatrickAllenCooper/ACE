#!/usr/bin/env python3
"""
30-node SCM CPU baselines: Random, Round-Robin, Max-Variance.

Uses the same LargeScaleSCM and per-node running-mean learner as the
existing Random baseline (CURC run_all_reviewer_experiments.sh Phase C),
so results are directly comparable to the reported Random 11.29 +/- 1.02.

Each policy runs N_SEEDS x N_EPISODES and saves:
  results/30node_baselines/summary.csv      -- one row per (seed, method)
  results/30node_baselines/per_episode.csv  -- per-episode losses for plots
"""

import sys
import os
import random
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import pandas as pd

from experiments.large_scale_scm import LargeScaleSCM

# ── Configuration ────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1011]
N_EPISODES = 300
N_SAMPLES_PER_EP = 50      # samples per intervention
N_EVAL_SAMPLES = 1000      # evaluation samples at end
OUT_DIR = "results/30node_baselines"
os.makedirs(OUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


class NodeLearner:
    """
    Per-node online running-mean predictor.
    Tracks the empirical mean of recent observations and running variance.
    """
    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.obs: dict[str, list] = {}
        self.var_estimates: dict[str, float] = {}

    def reset(self, nodes: list[str]):
        self.obs = {n: [] for n in nodes}
        self.var_estimates = {n: 1.0 for n in nodes}  # high-uncertainty prior

    def update(self, node_data: dict, intervened: str | None = None):
        """Update running stats with new batch."""
        for name, vals in node_data.items():
            arr = vals.numpy() if isinstance(vals, torch.Tensor) else np.array(vals)
            self.obs[name].extend(arr.tolist())
            # Keep only last 150 observations for running estimates
            if len(self.obs[name]) > 150:
                self.obs[name] = self.obs[name][-150:]
            if len(self.obs[name]) > 1:
                self.var_estimates[name] = float(np.var(self.obs[name]))

    def predict(self, node: str) -> float:
        if self.obs[node]:
            return float(np.mean(self.obs[node][-50:]))
        return 0.0

    def get_variance(self, node: str) -> float:
        return self.var_estimates.get(node, 1.0)

    def compute_total_mse(self, scm: 'LargeScaleSCM') -> float:
        eval_data = scm.generate(N_EVAL_SAMPLES)
        total = 0.0
        for name in scm.nodes:
            pred = self.predict(name)
            true_vals = eval_data[name].numpy()
            total += float(np.mean((true_vals - pred) ** 2))
        return total


def run_policy(policy_name: str, seed: int) -> tuple[dict, list]:
    """
    Run one (policy, seed) combination.
    Returns (summary_row, per_episode_rows).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scm = LargeScaleSCM(30)
    nodes = scm.nodes
    n_nodes = len(nodes)
    learner = NodeLearner(n_nodes)
    learner.reset(nodes)

    per_episode = []

    # Collect an initial observational batch so variance estimates are populated
    obs_data = scm.generate(200)
    learner.update(obs_data)

    t0 = time.time()
    for ep in range(1, N_EPISODES + 1):
        # ── Select intervention node ──────────────────────────────────────
        if policy_name == "random":
            node = random.choice(nodes)

        elif policy_name == "round_robin":
            node = nodes[(ep - 1) % n_nodes]

        elif policy_name == "max_variance":
            # Choose node with highest current variance estimate
            node = max(nodes, key=lambda n: learner.get_variance(n))

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # ── Choose a random intervention value ────────────────────────────
        value = random.uniform(-5.0, 5.0)

        # ── Execute and update ────────────────────────────────────────────
        data = scm.generate(N_SAMPLES_PER_EP, interventions={node: value})
        learner.update(data, intervened=node)

        # Record per-episode MSE every 10 episodes for learning curve
        if ep % 10 == 0:
            ep_mse = learner.compute_total_mse(scm)
            per_episode.append({
                'seed': seed, 'method': policy_name,
                'episode': ep, 'total_mse': ep_mse
            })

    final_mse = learner.compute_total_mse(scm)
    elapsed = time.time() - t0

    print(f"  [{policy_name:12s} seed={seed:4d}]  final_mse={final_mse:.4f}  "
          f"elapsed={elapsed:.1f}s")

    summary = {
        'seed': seed, 'method': policy_name, 'n_nodes': 30,
        'episodes': N_EPISODES, 'total_mse': final_mse,
        'elapsed_s': elapsed
    }
    return summary, per_episode


def main():
    print("=" * 65)
    print("30-node SCM CPU Baselines: Random, Round-Robin, Max-Variance")
    print(f"Seeds: {SEEDS}   Episodes: {N_EPISODES}")
    print("=" * 65)

    all_summaries = []
    all_per_ep = []

    for policy in ["random", "round_robin", "max_variance"]:
        print(f"\n--- {policy.upper()} ---")
        method_losses = []
        for seed in SEEDS:
            row, per_ep = run_policy(policy, seed)
            all_summaries.append(row)
            all_per_ep.extend(per_ep)
            method_losses.append(row['total_mse'])
        m = np.mean(method_losses)
        s = np.std(method_losses, ddof=1)
        print(f"  {policy}: mean={m:.3f} +/- {s:.3f}")

    # Save results
    pd.DataFrame(all_summaries).to_csv(
        os.path.join(OUT_DIR, "summary.csv"), index=False)
    pd.DataFrame(all_per_ep).to_csv(
        os.path.join(OUT_DIR, "per_episode.csv"), index=False)

    # Print final summary table
    df = pd.DataFrame(all_summaries)
    print("\n" + "=" * 65)
    print("FINAL SUMMARY (mean +/- std over 5 seeds)")
    print("=" * 65)
    for method, grp in df.groupby('method'):
        m = grp['total_mse'].mean()
        s = grp['total_mse'].std(ddof=1)
        print(f"  {method:15s}: {m:.3f} +/- {s:.3f}")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
