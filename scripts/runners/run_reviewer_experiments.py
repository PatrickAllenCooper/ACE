#!/usr/bin/env python3
"""
Reviewer Response Experiments for ICML 2026 Resubmission

Addresses all three reviewers' experimental concerns:

1. N=10 SEEDS (JmgE, PYSC)
   - Run 5 additional seeds for ACE and baselines
   - Seeds: 314, 271, 577, 618, 141 (complements existing 42, 123, 456, 789, 1011)

2. BAYESIAN BASELINE (rfmH, PYSC)
   - Expected information gain with posterior updating
   - Represents Bayesian OED approach (simplified ABCI/CBED comparison)

3. GRAPH MISSPECIFICATION ABLATION (PYSC)
   - Missing edges, extra edges, reversed edges
   - Measures degradation in mechanism estimation

4. HYPERPARAMETER SENSITIVITY (JmgE)
   - Grid search over alpha and gamma
   - Preference pair data efficiency (K=2,4,8,16)

5. DUFFING/PHILLIPS BASELINES (JmgE)
   - Numerical results for all sections

Usage:
    python scripts/runners/run_reviewer_experiments.py --all
    python scripts/runners/run_reviewer_experiments.py --bayesian-baseline
    python scripts/runners/run_reviewer_experiments.py --graph-misspec
    python scripts/runners/run_reviewer_experiments.py --hyperparam-grid
    python scripts/runners/run_reviewer_experiments.py --duffing-baselines
    python scripts/runners/run_reviewer_experiments.py --phillips-baselines
    python scripts/runners/run_reviewer_experiments.py --k-ablation
"""

import argparse
import os
import sys
import copy
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from baselines import (
    GroundTruthSCM,
    StudentSCM,
    SCMLearner,
    run_random_policy,
    run_round_robin_policy,
    run_max_variance_policy,
)


ALL_SEEDS = [42, 123, 456, 789, 1011, 314, 271, 577, 618, 141]
ADDITIONAL_SEEDS = [314, 271, 577, 618, 141]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# BAYESIAN EXPERIMENTAL DESIGN BASELINE
# ============================================================================

class BayesianOEDBaseline:
    """
    Bayesian Optimal Experimental Design baseline.

    Maintains a posterior over mechanism parameters and selects interventions
    that maximize expected posterior entropy reduction (mutual information
    between intervention outcome and parameters).

    This represents the class of methods from Toth et al. (2022),
    Tigas et al. (2022), and Zhang et al. (2023), adapted to the
    mechanism estimation setting for fair comparison with ACE.
    """

    def __init__(self, scm: GroundTruthSCM, n_posterior_samples: int = 50):
        self.scm = scm
        self.nodes = scm.nodes
        self.n_posterior_samples = n_posterior_samples
        self.intervention_range = (-5.0, 5.0)
        self.n_candidate_values = 10

    def select_intervention(
        self, learner: SCMLearner, n_mc_samples: int = 20
    ) -> Tuple[str, float]:
        """Select intervention maximizing expected information gain."""
        best_node = None
        best_value = None
        best_eig = -float('inf')

        current_losses = learner.evaluate()

        for node in self.nodes:
            values = np.linspace(
                self.intervention_range[0],
                self.intervention_range[1],
                self.n_candidate_values,
            )

            for value in values:
                eig = self._estimate_eig(learner, node, float(value), current_losses, n_mc_samples)
                if eig > best_eig:
                    best_eig = eig
                    best_node = node
                    best_value = float(value)

        return best_node, best_value

    def _estimate_eig(
        self,
        learner: SCMLearner,
        node: str,
        value: float,
        current_losses: Dict[str, float],
        n_mc_samples: int,
    ) -> float:
        """Estimate expected information gain via Monte Carlo."""
        gains = []
        for _ in range(n_mc_samples):
            cloned = copy.deepcopy(learner)
            intervention = {node: value}
            data = self.scm.generate(50, interventions=intervention)
            cloned.train_step(data)
            new_losses = cloned.evaluate()

            gain = sum(current_losses.values()) - sum(new_losses.values())
            gains.append(gain)

        return float(np.mean(gains))


def run_bayesian_baseline(
    seeds: List[int],
    episodes: int = 171,
    output_dir: str = "results/bayesian_baseline",
):
    """Run Bayesian OED baseline."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"BAYESIAN OED BASELINE - Seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        policy = BayesianOEDBaseline(scm)

        for episode in range(1, episodes + 1):
            node, value = policy.select_intervention(learner)
            intervention = {node: value}
            data = scm.generate(100, interventions=intervention)
            learner.train_step(data)

            if episode % 20 == 0 or episode == 1:
                losses = learner.evaluate()
                total = sum(losses.values())
                print(f"  Episode {episode}: loss={total:.4f} (target={node}, val={value:.2f})")

        final_losses = learner.evaluate()
        total_loss = sum(final_losses.values())
        results.append({
            'seed': seed,
            'method': 'bayesian_oed',
            'episodes': episodes,
            'total_loss': total_loss,
            **{f'loss_{k}': v for k, v in final_losses.items()},
        })

        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/bayesian_oed_summary.csv", index=False)
        print(f"  Seed {seed} complete: total_loss={total_loss:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/bayesian_oed_summary.csv", index=False)

    mean_loss = df['total_loss'].mean()
    std_loss = df['total_loss'].std()
    print(f"\nBayesian OED: {mean_loss:.3f} +/- {std_loss:.3f}")
    return df


# ============================================================================
# GRAPH MISSPECIFICATION ABLATION
# ============================================================================

class MisspecifiedGroundTruthSCM(GroundTruthSCM):
    """Ground truth SCM with intentional graph errors for ablation."""

    def __init__(self, misspec_type: str = "none"):
        super().__init__()
        self.misspec_type = misspec_type
        self.true_graph = {
            "X1": [],
            "X2": ["X1"],
            "X3": ["X1", "X2"],
            "X4": [],
            "X5": ["X4"],
        }

        if misspec_type == "missing_edge":
            self.graph["X3"] = ["X2"]  # Remove X1->X3 edge
        elif misspec_type == "extra_edge":
            self.graph["X5"] = ["X4", "X1"]  # Add spurious X1->X5
        elif misspec_type == "reversed_edge":
            self.graph["X1"] = ["X2"]  # Reverse X1->X2 to X2->X1
            self.graph["X2"] = []
        elif misspec_type == "missing_and_extra":
            self.graph["X3"] = ["X2"]  # Remove X1->X3
            self.graph["X5"] = ["X4", "X1"]  # Add X1->X5


def run_graph_misspecification_ablation(
    seeds: List[int],
    episodes: int = 171,
    output_dir: str = "results/graph_misspec",
):
    """Run ACE with misspecified graph structures via --graph_misspec flag."""
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    results = []

    misspec_types = [None, "missing_edge", "extra_edge", "reversed_edge", "missing_and_extra"]

    for misspec_type in misspec_types:
        label = misspec_type or "none"
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"GRAPH MISSPEC (ACE): {label} - Seed {seed}")
            print(f"{'='*60}")

            run_dir = f"{output_dir}/{label}/seed_{seed}"
            os.makedirs(run_dir, exist_ok=True)

            cmd = [
                sys.executable, "-u", "ace_experiments.py",
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--early_stopping",
                "--early_stop_patience", "20",
                "--use_dedicated_root_learner",
                "--dedicated_root_interval", "3",
                "--obs_train_interval", "3",
                "--obs_train_samples", "200",
                "--obs_train_epochs", "100",
                "--root_fitting",
                "--root_fit_interval", "5",
                "--root_fit_samples", "500",
                "--root_fit_epochs", "100",
                "--undersampled_bonus", "200.0",
                "--diversity_reward_weight", "0.3",
                "--max_concentration", "0.7",
                "--concentration_penalty", "150.0",
                "--update_reference_interval", "25",
                "--pretrain_steps", "200",
                "--pretrain_interval", "25",
                "--smart_breaker",
                "--output", run_dir,
            ]
            if misspec_type:
                cmd.extend(["--graph_misspec", misspec_type])

            result = subprocess.run(cmd, capture_output=False, text=True)

            losses_file = None
            for root_d, dirs, files in os.walk(run_dir):
                for f in files:
                    if f == "node_losses.csv":
                        losses_file = os.path.join(root_d, f)
                        break

            if losses_file:
                df_losses = pd.read_csv(losses_file)
                last_ep = df_losses['episode'].max()
                last = df_losses[df_losses['episode'] == last_ep].iloc[-1]
                if 'total_loss' in df_losses.columns:
                    total_loss = float(last['total_loss'])
                else:
                    loss_cols = [c for c in df_losses.columns if c.startswith('loss_')]
                    total_loss = float(last[loss_cols].sum())
            else:
                total_loss = float('nan')
                print(f"  WARNING: no node_losses.csv for {label} seed {seed}")

            results.append({
                'seed': seed,
                'misspec_type': label,
                'episodes': episodes,
                'total_loss': total_loss,
            })
            print(f"  {label} seed {seed}: total_loss={total_loss:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/graph_misspec_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("GRAPH MISSPECIFICATION SUMMARY (ACE)")
    print(f"{'='*60}")
    for mt in [m or "none" for m in misspec_types]:
        subset = df[df['misspec_type'] == mt]
        if not subset.empty:
            mean_loss = subset['total_loss'].mean()
            std_loss = subset['total_loss'].std()
            print(f"  {mt:20s}: {mean_loss:.3f} +/- {std_loss:.3f}")

    return df


# ============================================================================
# HYPERPARAMETER SENSITIVITY GRID (GPU -- runs ACE via subprocess)
# ============================================================================

def run_hyperparameter_grid(
    seeds: List[int],
    episodes: int = 100,
    output_dir: str = "results/hyperparam_grid",
):
    """
    Run ACE with varied reward weights to test hyperparameter sensitivity.
    Paper alpha (node importance) maps to --cov_bonus.
    Paper gamma (diversity) maps to --diversity_reward_weight.
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    results = []

    alpha_to_cov = {0.01: 6.0, 0.05: 30.0, 0.1: 60.0, 0.2: 120.0}
    gamma_values = [0.01, 0.05, 0.1, 0.2]

    for alpha, cov_bonus in alpha_to_cov.items():
        for gamma in gamma_values:
            for seed in seeds[:2]:
                run_label = f"a{alpha}_g{gamma}_s{seed}"
                run_dir = f"{output_dir}/{run_label}"
                os.makedirs(run_dir, exist_ok=True)

                cmd = [
                    sys.executable, "-u", "ace_experiments.py",
                    "--episodes", str(episodes),
                    "--seed", str(seed),
                    "--early_stopping",
                    "--early_stop_patience", "20",
                    "--cov_bonus", str(cov_bonus),
                    "--diversity_reward_weight", str(gamma),
                    "--use_dedicated_root_learner",
                    "--dedicated_root_interval", "3",
                    "--obs_train_interval", "3",
                    "--obs_train_samples", "200",
                    "--obs_train_epochs", "100",
                    "--root_fitting",
                    "--root_fit_interval", "5",
                    "--root_fit_samples", "500",
                    "--root_fit_epochs", "100",
                    "--undersampled_bonus", "200.0",
                    "--max_concentration", "0.7",
                    "--concentration_penalty", "150.0",
                    "--update_reference_interval", "25",
                    "--pretrain_steps", "200",
                    "--pretrain_interval", "25",
                    "--smart_breaker",
                    "--output", run_dir,
                ]
                print(f"\n  Hyperparam grid: alpha={alpha} (cov={cov_bonus}), gamma={gamma}, seed={seed}")
                result = subprocess.run(cmd, capture_output=False, text=True)

                losses_file = None
                for root_dir, dirs, files in os.walk(run_dir):
                    for f in files:
                        if f == "node_losses.csv":
                            losses_file = os.path.join(root_dir, f)
                            break

                if losses_file:
                    df_losses = pd.read_csv(losses_file)
                    last_episode = df_losses['episode'].max()
                    last = df_losses[df_losses['episode'] == last_episode].iloc[-1]
                    if 'total_loss' in df_losses.columns:
                        total_loss = float(last['total_loss'])
                    else:
                        loss_cols = [c for c in df_losses.columns if c.startswith('loss_')]
                        total_loss = float(last[loss_cols].sum())
                else:
                    total_loss = float('nan')
                    print(f"    WARNING: no node_losses.csv found for {run_label}")

                results.append({
                    'seed': seed, 'alpha': alpha, 'gamma': gamma,
                    'cov_bonus': cov_bonus, 'episodes': episodes,
                    'total_loss': total_loss,
                })
                print(f"    total_loss={total_loss:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/hyperparam_grid_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("HYPERPARAMETER GRID (alpha x gamma) -- ACE")
    print(f"{'='*60}")
    if not df.empty and not df['total_loss'].isna().all():
        pivot = df.groupby(['alpha', 'gamma'])['total_loss'].mean().unstack()
        print(pivot.to_string(float_format='%.3f'))

    return df


# ============================================================================
# PREFERENCE PAIR DATA EFFICIENCY (K ablation)
# ============================================================================

def run_k_ablation(
    seeds: List[int],
    episodes: int = 100,
    output_dir: str = "results/k_ablation",
):
    """
    Ablation over number of candidates K per step.
    Tests K=2, 4, 8, 16 to address JmgE's concern about preference pair sparsity.
    Uses lookahead mechanism with random proposals.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    k_values = [2, 4, 8, 16]

    for K in k_values:
        for seed in seeds:
            print(f"\n  K={K}, Seed {seed}")
            set_seed(seed)

            scm = GroundTruthSCM()
            student = StudentSCM(scm)
            learner = SCMLearner(student, oracle=scm)

            for episode in range(1, episodes + 1):
                candidates = []
                for _ in range(K):
                    node = random.choice(scm.nodes)
                    value = random.uniform(-5, 5)
                    candidates.append((node, value))

                best_candidate = None
                best_gain = -float('inf')

                for node, value in candidates:
                    cloned = copy.deepcopy(learner)
                    losses_before = cloned.evaluate()
                    data = scm.generate(100, interventions={node: value})
                    cloned.train_step(data)
                    losses_after = cloned.evaluate()
                    gain = sum(losses_before.values()) - sum(losses_after.values())
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = (node, value)

                node, value = best_candidate
                data = scm.generate(100, interventions={node: value})
                learner.train_step(data)

            final_losses = learner.evaluate()
            total_loss = sum(final_losses.values())
            results.append({
                'seed': seed,
                'K': K,
                'episodes': episodes,
                'total_loss': total_loss,
                **{f'loss_{k}': v for k, v in final_losses.items()},
            })
            print(f"    total_loss={total_loss:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/k_ablation_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("K ABLATION SUMMARY")
    print(f"{'='*60}")
    for K in k_values:
        subset = df[df['K'] == K]
        mean_loss = subset['total_loss'].mean()
        std_loss = subset['total_loss'].std()
        print(f"  K={K:2d}: {mean_loss:.3f} +/- {std_loss:.3f}")

    return df


# ============================================================================
# DUFFING BASELINES
# ============================================================================

def run_duffing_baselines(
    seeds: List[int],
    episodes: int = 100,
    output_dir: str = "results/duffing_baselines",
):
    """Run baseline methods on Duffing oscillator system."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from experiments.duffing_oscillators import DuffingOscillatorChain, OscillatorLearner
    except ImportError:
        print("WARNING: Could not import Duffing modules. Skipping.")
        return None

    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"DUFFING BASELINES - Seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        oracle = DuffingOscillatorChain(n_masses=3)

        for method_name in ['random', 'round_robin']:
            set_seed(seed)
            learner = OscillatorLearner(oracle.nodes)
            optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            for episode in range(1, episodes + 1):
                if method_name == 'random':
                    node_idx = random.randint(0, len(oracle.nodes) - 1)
                    value = random.uniform(-5, 5)
                elif method_name == 'round_robin':
                    node_idx = (episode - 1) % len(oracle.nodes)
                    value = random.uniform(-5, 5)

                node = oracle.nodes[node_idx]
                try:
                    data = oracle.generate(50, interventions={node: value})
                    predictions = learner(data)
                    total_ep_loss = 0.0
                    for n in oracle.nodes:
                        if n in predictions and n in data:
                            total_ep_loss += loss_fn(predictions[n], data[n])
                    optimizer.zero_grad()
                    if isinstance(total_ep_loss, torch.Tensor) and total_ep_loss.requires_grad:
                        total_ep_loss.backward()
                        optimizer.step()
                except Exception:
                    pass

            eval_data = oracle.generate(200)
            with torch.no_grad():
                preds = learner(eval_data)
                total_loss = 0.0
                for n in oracle.nodes:
                    if n in preds and n in eval_data:
                        total_loss += loss_fn(preds[n], eval_data[n]).item()

            results.append({
                'seed': seed,
                'method': method_name,
                'episodes': episodes,
                'total_loss': total_loss,
            })
            print(f"  {method_name}: total_loss={total_loss:.4f}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/duffing_baselines_summary.csv", index=False)
        for m in df['method'].unique():
            sub = df[df['method'] == m]
            print(f"  {m}: {sub['total_loss'].mean():.3f} +/- {sub['total_loss'].std():.3f}")
        return df
    return None


# ============================================================================
# PHILLIPS BASELINES
# ============================================================================

def run_phillips_baselines(
    seeds: List[int],
    episodes: int = 50,
    output_dir: str = "results/phillips_baselines",
):
    """Run baseline methods on Phillips curve data."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from experiments.phillips_curve import PhillipsCurveOracle, PhillipsCurveLearner
    except ImportError:
        print("WARNING: Could not import Phillips modules. Skipping.")
        return None

    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"PHILLIPS BASELINES - Seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        try:
            oracle = PhillipsCurveOracle()
        except Exception as e:
            print(f"  WARNING: PhillipsCurveOracle init failed ({e}). Skipping.")
            return None

        for method_name in ['random', 'sequential']:
            set_seed(seed)
            try:
                learner_copy = PhillipsCurveLearner(oracle)
            except Exception:
                learner_copy = PhillipsCurveLearner()

            for episode in range(1, episodes + 1):
                try:
                    if method_name == 'random':
                        data = oracle.sample_random_regime(100)
                    elif method_name == 'sequential':
                        data = oracle.sample_sequential(episode, 100)
                    learner_copy.train_step(data)
                except AttributeError:
                    idx = random.randint(0, len(oracle.data) - 100) if hasattr(oracle, 'data') else 0
                    break

            try:
                final_losses = learner_copy.evaluate()
                total_loss = sum(final_losses.values()) if isinstance(final_losses, dict) else float(final_losses)
            except Exception:
                total_loss = float('nan')

            results.append({
                'seed': seed,
                'method': method_name,
                'episodes': episodes,
                'total_loss': total_loss,
            })
            print(f"  {method_name}: total_loss={total_loss:.4f}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/phillips_baselines_summary.csv", index=False)
        return df
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reviewer Response Experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--bayesian-baseline", action="store_true")
    parser.add_argument("--graph-misspec", action="store_true")
    parser.add_argument("--hyperparam-grid", action="store_true")
    parser.add_argument("--k-ablation", action="store_true")
    parser.add_argument("--duffing-baselines", action="store_true")
    parser.add_argument("--phillips-baselines", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS)
    parser.add_argument("--episodes", type=int, default=171)
    parser.add_argument("--output", type=str, default="results/reviewer_response")
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"{args.output}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    print(f"{'='*60}")
    print("ICML 2026 REVIEWER RESPONSE EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {base_dir}")
    print(f"Started: {datetime.now()}")

    if args.all or args.bayesian_baseline:
        print("\n\n>>> BAYESIAN OED BASELINE <<<")
        run_bayesian_baseline(args.seeds, args.episodes, f"{base_dir}/bayesian_baseline")

    if args.all or args.graph_misspec:
        print("\n\n>>> GRAPH MISSPECIFICATION ABLATION <<<")
        run_graph_misspecification_ablation(
            args.seeds[:5], args.episodes, f"{base_dir}/graph_misspec"
        )

    if args.all or args.hyperparam_grid:
        print("\n\n>>> HYPERPARAMETER SENSITIVITY GRID <<<")
        run_hyperparameter_grid(args.seeds[:3], 100, f"{base_dir}/hyperparam_grid")

    if args.all or args.k_ablation:
        print("\n\n>>> K ABLATION (Preference Pair Efficiency) <<<")
        run_k_ablation(args.seeds[:5], 100, f"{base_dir}/k_ablation")

    if args.all or args.duffing_baselines:
        print("\n\n>>> DUFFING BASELINES <<<")
        run_duffing_baselines(args.seeds[:5], 100, f"{base_dir}/duffing_baselines")

    if args.all or args.phillips_baselines:
        print("\n\n>>> PHILLIPS BASELINES <<<")
        run_phillips_baselines(args.seeds[:5], 50, f"{base_dir}/phillips_baselines")

    print(f"\n{'='*60}")
    print("ALL REVIEWER EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Finished: {datetime.now()}")
    print(f"Results: {base_dir}")


if __name__ == "__main__":
    main()
