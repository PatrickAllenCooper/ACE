#!/usr/bin/env python3
"""
30-node SCM baselines using the same MLP-based student learner as ACE.

This runner uses StudentSCM / SCMLearner / ScientificCritic from baselines.py
(duck-typed against LargeScaleSCM, which has the same interface as GroundTruthSCM:
  .nodes, .graph, .generate(n_samples, interventions)).

Results are saved in the same format as baselines.py so they can be
directly compared against ACE (best-loss 1.95 +/- 0.77 on the same system).

Usage (CURC worker script calls this):
    python scripts/runners/run_30node_baseline_seed.py \
        --method round_robin \
        --seed 42 \
        --episodes 150 \
        --output results/curc_30node_baselines
"""

import sys
import os
import copy
import random
import argparse
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import pandas as pd

from experiments.large_scale_scm import LargeScaleSCM
from baselines import (
    StudentSCM, SCMLearner, ScientificCritic,
    RandomPolicy, RoundRobinPolicy, MaxVariancePolicy, PPOPolicy,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.runners.run_reviewer_experiments import BayesianOEDBaseline


# ---------------------------------------------------------------------------
# Patched RoundRobinPolicy that uses the actual node list (not the hardcoded
# 5-node ["X1","X4","X2","X5","X3"] order in baselines.py).
# ---------------------------------------------------------------------------
class RoundRobinPolicy30(RoundRobinPolicy):
    def __init__(self, nodes, value_min=-5.0, value_max=5.0):
        self.nodes = list(nodes)       # override the hardcoded 5-node list
        self.value_min = value_min
        self.value_max = value_max
        self.step = 0
        self.name = "Round-Robin"


# ---------------------------------------------------------------------------
# Wrapper for BayesianOED with reduced search to make 30-node tractable.
# Full grid (30 nodes x 10 values x 20 MC) per step is infeasible; we sample
# a random subset of (node, value) pairs each step.
# ---------------------------------------------------------------------------
class BayesianOEDFast:
    def __init__(self, scm, n_candidates=10, n_mc_samples=3, value_min=-5.0, value_max=5.0):
        self.scm = scm
        self.nodes = scm.nodes
        self.n_candidates = n_candidates
        self.n_mc_samples = n_mc_samples
        self.value_min = value_min
        self.value_max = value_max
        self.name = "Bayesian-OED"

    def select_intervention(self, student, oracle=None, node_losses=None):
        # Convert SCMLearner.evaluate() output (dict of node->loss) to losses dict
        # for EIG estimation.
        if hasattr(student, "_critic") and student._critic is not None:
            _, current_losses = student._critic.evaluate(student.student if hasattr(student, "student") else student)
        else:
            critic = ScientificCritic(self.scm)
            _, current_losses = critic.evaluate(student.student if hasattr(student, "student") else student)

        best_node, best_value, best_eig = None, None, -float("inf")
        for _ in range(self.n_candidates):
            node = random.choice(self.nodes)
            value = random.uniform(self.value_min, self.value_max)
            eig = self._estimate_eig(student, node, value, current_losses)
            if eig > best_eig:
                best_eig = eig
                best_node = node
                best_value = value
        return best_node, best_value

    def _estimate_eig(self, learner_obj, node, value, current_losses):
        learner = learner_obj if hasattr(learner_obj, "train_step") else learner_obj
        gains = []
        for _ in range(self.n_mc_samples):
            cloned_student = copy.deepcopy(learner.student)
            cloned_learner = SCMLearner(cloned_student, oracle=self.scm)
            data = self.scm.generate(50, interventions={node: value})
            cloned_learner.train_step(data, intervened=node, n_epochs=20)
            new_losses = cloned_learner.evaluate()
            gains.append(sum(current_losses.values()) - sum(new_losses.values()))
        return float(np.mean(gains))


# ---------------------------------------------------------------------------
# Episode loop (mirrors run_baseline from baselines.py but configurable)
# ---------------------------------------------------------------------------
def run_episode_loop(
    policy,
    oracle,
    n_episodes: int,
    steps_per_episode: int = 25,
    obs_train_interval: int = 3,
    obs_train_samples: int = 200,
    n_train_epochs: int = 50,
) -> pd.DataFrame:
    """Run policy against LargeScaleSCM and return per-step DataFrame."""
    critic = ScientificCritic(oracle)
    all_records = []

    for episode in range(n_episodes):
        student = StudentSCM(oracle)
        learner = SCMLearner(student, oracle=oracle)

        if hasattr(policy, 'reset'):
            policy.reset()

        prev_loss, prev_node_losses = critic.evaluate(student)

        for step in range(steps_per_episode):
            target, value = policy.select_intervention(student, oracle=oracle)

            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data, intervened=target, n_epochs=n_train_epochs)

            if obs_train_interval > 0 and step > 0 and step % obs_train_interval == 0:
                learner.observational_train(oracle, n_samples=obs_train_samples,
                                            n_epochs=n_train_epochs)

            total_loss, node_losses = critic.evaluate(student)

            record = {
                "episode": episode,
                "step": step,
                "target": target,
                "value": value,
                "total_loss": total_loss,
                **{f"loss_{n}": v for n, v in node_losses.items()},
            }
            all_records.append(record)

            prev_loss = total_loss
            prev_node_losses = node_losses

        if episode % 10 == 0:
            logging.info(f"  Episode {episode}/{n_episodes}, "
                         f"loss={total_loss:.4f}")

    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="30-node SCM MLP-learner baseline for one seed")
    parser.add_argument("--method", required=True,
                        choices=["random", "round_robin", "max_variance",
                                 "ppo", "bayesian_oed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--obs_train_interval", type=int, default=3)
    parser.add_argument("--obs_train_samples", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="results/curc_30node_baselines")
    args = parser.parse_args()

    # Seed everything before building the SCM so the graph is deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = os.path.join(args.output, args.method, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "run.log")),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"30-node baseline: method={args.method} seed={args.seed} "
                 f"episodes={args.episodes}")
    t0 = time.time()

    # Build SCM (graph wiring is seed-controlled by the np.random.seed above)
    scm = LargeScaleSCM(30)
    nodes = scm.nodes
    logging.info(f"  SCM: {len(nodes)} nodes, "
                 f"{sum(len(v) for v in scm.graph.values())} edges, "
                 f"{sum(1 for n in nodes if len(scm.get_parents(n))>=2)} colliders")

    if args.method == "random":
        policy = RandomPolicy(nodes)
    elif args.method == "round_robin":
        policy = RoundRobinPolicy30(nodes)
    elif args.method == "max_variance":
        policy = MaxVariancePolicy(nodes)
    elif args.method == "ppo":
        policy = PPOPolicy(nodes)
    elif args.method == "bayesian_oed":
        policy = BayesianOEDFast(scm, n_candidates=10, n_mc_samples=3)

    df = run_episode_loop(
        policy, scm,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        obs_train_interval=args.obs_train_interval,
        obs_train_samples=args.obs_train_samples,
    )

    # Final loss = last step of last episode
    final_loss = df.tail(1)["total_loss"].item()
    elapsed = time.time() - t0
    logging.info(f"  Done. final_loss={final_loss:.4f}  elapsed={elapsed:.0f}s")

    # Save full trajectory
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    # Save compact summary
    per_ep = (df.groupby("episode")["total_loss"]
                .last()
                .reset_index()
                .rename(columns={"total_loss": "episode_final_loss"}))
    per_ep.to_csv(os.path.join(run_dir, "per_episode.csv"), index=False)

    summary = {
        "method": args.method,
        "seed": args.seed,
        "n_nodes": 30,
        "episodes": args.episodes,
        "final_total_loss": final_loss,
        "min_total_loss": df["total_loss"].min(),
        "elapsed_s": elapsed,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(run_dir, "summary.csv"), index=False)

    logging.info(f"  Results saved to {run_dir}")


if __name__ == "__main__":
    main()
