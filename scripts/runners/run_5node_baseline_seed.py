#!/usr/bin/env python3
"""
5-node SCM baseline for one (method, seed) pair, at a total-environment-
query budget matched to ACE's own lookahead cost.

Why this exists rather than calling ``baselines.py`` directly: ``baselines.py
--all_with_ppo`` runs every method inside a single process and writes
``{method}_results.csv`` / ``{method}_query_budget.json`` into one shared,
timestamped run directory. That is convenient for a quick local comparison,
but it does not match the one-method-per-job, ``{method}/seed_{seed}/``
directory layout that ``scripts/analysis/aggregate_budget_fairness.py``
(and its 30-node counterpart, ``run_30node_baseline_seed.py``) expect for
the budget-fairness decision-gate aggregation. This script is the 5-node
analog of ``run_30node_baseline_seed.py``: one method, one seed, one job,
written as ``{output}/{method}/seed_{seed}/{node_losses.csv,summary.csv,
query_budget.json}`` so the same aggregator handles both scales.

It is a thin wrapper around the already-tested ``baselines.py`` primitives
(``GroundTruthSCM``, ``InstrumentedOracle``, ``run_baseline``, and the four
policy classes) -- no baseline logic is reimplemented here.

Usage:
    python scripts/runners/run_5node_baseline_seed.py \
        --method random --seed 42 --query_budget 483460 \
        --output results/curc_5node_budget_fairness/baselines
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch

from baselines import (
    GroundTruthSCM,
    InstrumentedOracle,
    MaxVariancePolicy,
    PPOPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    run_baseline,
    EnsembleStudentSCM, EnsembleLearner, PropagatedVariancePolicy,
    StudentSCM, SCMLearner,
)


def main():
    parser = argparse.ArgumentParser(
        description="5-node GroundTruthSCM baseline for one method/seed pair")
    parser.add_argument("--method", required=True,
                         choices=["random", "round_robin", "max_variance", "ppo",
                                  "pev", "random_ens", "round_robin_ens"])
    parser.add_argument("--tag", type=str, default=None, help="output subdirectory name (default: the method name)")
    parser.add_argument("--student_arch", choices=list(StudentSCM.ARCHS), default="ace",
                        help="student MLP per mechanism: 'ace' = (64,64) as in ace_experiments.py (default); 'small' = (16,) as in every pre-audit baseline")
    parser.add_argument("--train_epochs", type=int, default=100, help="learner epochs per step (ACE uses 100; pre-audit baselines used 50)")
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--pev_values", type=int, default=11)
    parser.add_argument("--pev_scoring", choices=["ivr", "var"], default="ivr", help="PEV: expected variance reduction over the evaluation domain (ivr) or naive visited variance (var)")
    parser.add_argument("--pev_sim", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--obs_train_interval", type=int, default=3)
    parser.add_argument("--obs_train_samples", type=int, default=200)
    parser.add_argument("--query_budget", type=int, default=None,
                         help="If set, run episodes until cumulative "
                              "environment sample count reaches this value "
                              "instead of a fixed --episodes count, for "
                              "total-query-matched comparison against ACE.")
    parser.add_argument("--output", type=str,
                         default="results/curc_5node_budget_fairness/baselines")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = os.path.join(args.output, args.tag or args.method, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "run.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"5-node baseline: method={args.method} seed={args.seed} "
                 f"episodes={args.episodes} query_budget={args.query_budget}")
    t0 = time.time()

    base_oracle = GroundTruthSCM()
    nodes = base_oracle.nodes
    oracle = InstrumentedOracle(base_oracle)

    if args.method == "random":
        policy = RandomPolicy(nodes)
    elif args.method == "round_robin":
        policy = RoundRobinPolicy(nodes)
    elif args.method == "max_variance":
        policy = MaxVariancePolicy(nodes)
    elif args.method == "ppo":
        policy = PPOPolicy(nodes)
    elif args.method == "pev":
        policy = PropagatedVariancePolicy(nodes, n_values=args.pev_values, n_sim=args.pev_sim, scoring=args.pev_scoring)
    elif args.method == "random_ens":
        policy = RandomPolicy(nodes)
    elif args.method == "round_robin_ens":
        policy = RoundRobinPolicy(nodes)

    dims = StudentSCM.ARCHS[args.student_arch]
    student_factory = lambda o: StudentSCM(o, hidden_dims=dims)
    learner_factory = lambda st, o: SCMLearner(st, oracle=o)
    if args.method in {"pev", "random_ens", "round_robin_ens"}:
        K = args.ensemble_size
        student_factory = lambda o: EnsembleStudentSCM(o, n_members=K, hidden_dims=dims)
        learner_factory = lambda st, o: EnsembleLearner(st, oracle=o)

    df = run_baseline(
        policy, oracle,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        obs_train_interval=args.obs_train_interval,
        obs_train_samples=args.obs_train_samples,
        query_budget=args.query_budget,
        student_factory=student_factory,
        learner_factory=learner_factory,
        n_train_epochs=args.train_epochs,
    )

    final_loss = df.tail(1)["total_loss"].item()
    elapsed = time.time() - t0
    logging.info(f"  Done. final_loss={final_loss:.4f}  elapsed={elapsed:.0f}s")

    # node_losses.csv: the per-step schema scaling_common.summarize_node_losses
    # expects (episode, total_loss, loss_<node> columns) -- run_baseline's df
    # already has exactly these columns, so no reshaping is needed.
    df.to_csv(os.path.join(run_dir, "node_losses.csv"), index=False)

    per_ep = (df.groupby("episode")["total_loss"]
                .last()
                .reset_index()
                .rename(columns={"total_loss": "episode_final_loss"}))
    per_ep.to_csv(os.path.join(run_dir, "per_episode.csv"), index=False)

    summary = {
        "method": args.method,
        "seed": args.seed,
        "n_nodes": len(nodes),
        "episodes": int(df["episode"].nunique()),
        "final_total_loss": final_loss,
        "min_total_loss": df["total_loss"].min(),
        "ace_final_total_loss": float(df.tail(1)["ace_total_loss"].item()),
        "ace_min_total_loss": float(df["ace_total_loss"].min()),
        "elapsed_s": elapsed,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    with open(os.path.join(run_dir, "query_budget.json"), "w") as f:
        json.dump(oracle.query_summary(), f, indent=2)
    logging.info(f"  [Query Budget] {args.method} breakdown: {oracle.query_summary()}")
    logging.info(f"  Results saved to {run_dir}")


if __name__ == "__main__":
    main()
