#!/usr/bin/env python3
"""
Baseline Comparison Script for ACE (Active Causal Experimentalist)

This script implements three baseline intervention strategies for comparison
against the learned ACE policy:

1. Random Policy (Lower Bound)
   - Uniformly samples target node and intervention value
   - Represents unguided, passive exploration

2. Round-Robin (Systematic Heuristic)
   - Cycles through nodes in fixed topological order
   - Ensures uniform coverage, tests if adaptive sampling is necessary

3. Max-Variance (Uncertainty Sampling)
   - Greedy active learning using MC Dropout
   - Selects interventions maximizing predictive variance
   - Represents standard "greedy" optimal experimental design

Usage:
    python baselines.py --baseline random --episodes 100
    python baselines.py --baseline round_robin --episodes 100
    python baselines.py --baseline max_variance --episodes 100
    python baselines.py --all --episodes 100  # Run all baselines
"""

import argparse
import os
import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
# 1. SCM DEFINITIONS (Copied from ace_experiments.py for standalone use)
# ----------------------------------------------------------------

class GroundTruthSCM:
    """Ground truth SCM with known structural equations."""
    
    def __init__(self):
        self.nodes = ["X1", "X2", "X3", "X4", "X5"]
        self.graph = {
            "X1": [],
            "X2": ["X1"],
            "X3": ["X1", "X2"],
            "X4": [],
            "X5": ["X4"],
        }
        self.noise_std = 0.1
        
    def get_parents(self, node: str) -> List[str]:
        return self.graph.get(node, [])
    
    def generate(self, n_samples: int, interventions: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Generate samples, optionally with interventions."""
        interventions = interventions or {}
        
        # Root nodes
        if "X1" in interventions:
            X1 = torch.full((n_samples,), interventions["X1"])
        else:
            X1 = torch.randn(n_samples) * 1.0 + 0.0
            
        if "X4" in interventions:
            X4 = torch.full((n_samples,), interventions["X4"])
        else:
            X4 = torch.randn(n_samples) * 1.0 + 2.0
            
        # X2 = 2*X1 + 1
        if "X2" in interventions:
            X2 = torch.full((n_samples,), interventions["X2"])
        else:
            X2 = 2 * X1 + 1 + torch.randn(n_samples) * self.noise_std
            
        # X3 = 0.5*X1 - X2 + sin(X2) (collider)
        if "X3" in interventions:
            X3 = torch.full((n_samples,), interventions["X3"])
        else:
            X3 = 0.5 * X1 - X2 + torch.sin(X2) + torch.randn(n_samples) * self.noise_std
            
        # X5 = 0.2*X4^2
        if "X5" in interventions:
            X5 = torch.full((n_samples,), interventions["X5"])
        else:
            X5 = 0.2 * X4**2 + torch.randn(n_samples) * self.noise_std
            
        return {"X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}


class StudentSCM(nn.Module):
    """Learnable SCM that approximates the ground truth."""
    
    def __init__(self, oracle: GroundTruthSCM, hidden_dim: int = 16):
        super().__init__()
        self.nodes = oracle.nodes
        self.graph = oracle.graph
        self.mechanisms = nn.ModuleDict()
        
        for node in self.nodes:
            parents = self.get_parents(node)
            if not parents:
                # Root node: learnable mean
                self.mechanisms[node] = nn.ParameterDict({
                    'mu': nn.Parameter(torch.zeros(1))
                })
            else:
                # Non-root: MLP
                n_parents = len(parents)
                self.mechanisms[node] = nn.Sequential(
                    nn.Linear(n_parents, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                
    def get_parents(self, node: str) -> List[str]:
        return self.graph.get(node, [])
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with teacher forcing."""
        predictions = {}
        for node in self.nodes:
            parents = self.get_parents(node)
            if not parents:
                predictions[node] = self.mechanisms[node]['mu'].expand(data[node].shape[0])
            else:
                parent_tensor = torch.stack([data[p] for p in parents], dim=1)
                predictions[node] = self.mechanisms[node](parent_tensor).squeeze(-1)
        return predictions


# ----------------------------------------------------------------
# 2. BASELINE POLICIES
# ----------------------------------------------------------------

class RandomPolicy:
    """
    Random Policy (Lower Bound)
    
    At each step, samples a target node uniformly from V and an
    intervention value uniformly from the valid range [-5, 5].
    """
    
    def __init__(self, nodes: List[str], value_min: float = -5.0, value_max: float = 5.0):
        self.nodes = nodes
        self.value_min = value_min
        self.value_max = value_max
        self.name = "Random"
        
    def select_intervention(self, student: StudentSCM, **kwargs) -> Tuple[str, float]:
        target = random.choice(self.nodes)
        value = random.uniform(self.value_min, self.value_max)
        return target, value


class RoundRobinPolicy:
    """
    Round-Robin (Systematic Heuristic)
    
    Deterministically cycles through intervention targets in a fixed
    topological order. Tests if adaptive, non-uniform sampling is necessary.
    """
    
    def __init__(self, nodes: List[str], value_min: float = -5.0, value_max: float = 5.0):
        # Topological order for our SCM: X1, X4, X2, X5, X3
        self.nodes = ["X1", "X4", "X2", "X5", "X3"]
        self.value_min = value_min
        self.value_max = value_max
        self.step = 0
        self.name = "Round-Robin"
        
    def select_intervention(self, student: StudentSCM, **kwargs) -> Tuple[str, float]:
        target = self.nodes[self.step % len(self.nodes)]
        value = random.uniform(self.value_min, self.value_max)
        self.step += 1
        return target, value
    
    def reset(self):
        self.step = 0


class MaxVariancePolicy:
    """
    Max-Variance (Uncertainty Sampling)
    
    Greedy active learning strategy using epistemic uncertainty as proxy
    for information gain. Uses MC Dropout to approximate posterior.
    Selects intervention maximizing aggregate predictive variance.
    """
    
    def __init__(self, nodes: List[str], value_min: float = -5.0, value_max: float = 5.0,
                 n_candidates: int = 64, n_mc_samples: int = 10):
        self.nodes = nodes
        self.value_min = value_min
        self.value_max = value_max
        self.n_candidates = n_candidates
        self.n_mc_samples = n_mc_samples
        self.name = "Max-Variance"
        
    def _enable_dropout(self, model: nn.Module):
        """Enable dropout during inference for MC Dropout."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def _compute_variance(self, student: StudentSCM, oracle: GroundTruthSCM,
                          target: str, value: float) -> float:
        """Compute predictive variance using MC Dropout."""
        # Generate interventional data
        data = oracle.generate(n_samples=50, interventions={target: value})
        
        # MC Dropout: run multiple forward passes
        predictions = {node: [] for node in student.nodes}
        
        student.train()  # Enable dropout
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                preds = student(data)
                for node in student.nodes:
                    predictions[node].append(preds[node])
        student.eval()
        
        # Compute variance across MC samples
        total_variance = 0.0
        for node in student.nodes:
            stacked = torch.stack(predictions[node], dim=0)  # [n_mc, n_samples]
            variance = stacked.var(dim=0).mean().item()
            total_variance += variance
            
        return total_variance
    
    def select_intervention(self, student: StudentSCM, oracle: GroundTruthSCM = None,
                            **kwargs) -> Tuple[str, float]:
        if oracle is None:
            # Fallback to random if no oracle provided
            return random.choice(self.nodes), random.uniform(self.value_min, self.value_max)
        
        # Generate candidate interventions
        candidates = []
        for _ in range(self.n_candidates):
            target = random.choice(self.nodes)
            value = random.uniform(self.value_min, self.value_max)
            candidates.append((target, value))
            
        # Evaluate variance for each candidate
        best_candidate = None
        best_variance = -float('inf')
        
        for target, value in candidates:
            variance = self._compute_variance(student, oracle, target, value)
            if variance > best_variance:
                best_variance = variance
                best_candidate = (target, value)
                
        return best_candidate


# ----------------------------------------------------------------
# 3. TRAINING LOOP
# ----------------------------------------------------------------

class SCMLearner:
    """Learner that trains the student SCM on experimental data."""
    
    def __init__(self, student: StudentSCM, lr: float = 2e-3, buffer_size: int = 50):
        self.student = student
        self.optimizer = optim.Adam(student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = []
        self.buffer_size = buffer_size
        
    def train_step(self, data: Dict[str, torch.Tensor], intervened: Optional[str] = None,
                   n_epochs: int = 50):
        """Train on new data, respecting intervention masks."""
        self.student.train()
        
        # Add to buffer
        self.buffer.append({"data": data, "intervened": intervened})
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Collate buffer data
        combined_data = {node: torch.cat([b["data"][node] for b in self.buffer])
                         for node in self.student.nodes}
        combined_mask = {}
        for node in self.student.nodes:
            masks = []
            for b in self.buffer:
                n = b["data"][node].shape[0]
                if b.get("intervened") == node:
                    masks.append(torch.zeros(n, dtype=torch.bool))
                else:
                    masks.append(torch.ones(n, dtype=torch.bool))
            combined_mask[node] = torch.cat(masks)
            
        # Training loop
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            total_loss = 0
            
            for node in self.student.nodes:
                parents = self.student.get_parents(node)
                y_true = combined_data[node]
                mask = combined_mask[node]
                
                if mask.sum().item() == 0:
                    continue
                    
                if not parents:
                    y_pred = self.student.mechanisms[node]['mu'].expand_as(y_true)
                else:
                    p_tensor = torch.stack([combined_data[p] for p in parents], dim=1)
                    y_pred = self.student.mechanisms[node](p_tensor).squeeze()
                    
                loss = self.loss_fn(y_pred[mask], y_true[mask])
                total_loss += loss
                
            total_loss.backward()
            self.optimizer.step()
            
        return total_loss.item()
    
    def observational_train(self, oracle: GroundTruthSCM, n_samples: int = 100,
                            n_epochs: int = 50):
        """Periodic observational training to preserve all mechanisms."""
        obs_data = oracle.generate(n_samples=n_samples, interventions=None)
        return self.train_step(obs_data, intervened=None, n_epochs=n_epochs)


class ScientificCritic:
    """Evaluates mechanism quality."""
    
    def __init__(self, oracle: GroundTruthSCM):
        self.oracle = oracle
        self.val_data = oracle.generate(n_samples=500)
        
    def evaluate(self, student: StudentSCM) -> Tuple[float, Dict[str, float]]:
        """Compute MSE for each mechanism."""
        student.eval()
        with torch.no_grad():
            preds = student(self.val_data)
            
        node_losses = {}
        total_loss = 0
        for node in student.nodes:
            mse = ((preds[node] - self.val_data[node])**2).mean().item()
            node_losses[node] = mse
            total_loss += mse
            
        return total_loss, node_losses


def run_baseline(policy, oracle: GroundTruthSCM, n_episodes: int = 100,
                 steps_per_episode: int = 25, obs_train_interval: int = 5,
                 obs_train_samples: int = 100) -> pd.DataFrame:
    """Run a baseline policy and collect metrics."""
    
    critic = ScientificCritic(oracle)
    all_records = []
    
    for episode in range(n_episodes):
        # Fresh student each episode
        student = StudentSCM(oracle)
        learner = SCMLearner(student)
        
        # Reset policy state if needed
        if hasattr(policy, 'reset'):
            policy.reset()
            
        for step in range(steps_per_episode):
            # Select intervention
            target, value = policy.select_intervention(student, oracle=oracle)
            
            # Execute intervention
            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data, intervened=target)
            
            # Periodic observational training
            if obs_train_interval > 0 and step > 0 and step % obs_train_interval == 0:
                learner.observational_train(oracle, n_samples=obs_train_samples)
            
            # Evaluate
            total_loss, node_losses = critic.evaluate(student)
            
            record = {
                "episode": episode,
                "step": step,
                "target": target,
                "value": value,
                "total_loss": total_loss,
                **{f"loss_{node}": loss for node, loss in node_losses.items()}
            }
            all_records.append(record)
            
        if episode % 10 == 0:
            logging.info(f"  Episode {episode}/{n_episodes}, Final Loss: {total_loss:.4f}")
            
    return pd.DataFrame(all_records)


# ----------------------------------------------------------------
# 4. VISUALIZATION
# ----------------------------------------------------------------

def plot_comparison(results: Dict[str, pd.DataFrame], output_dir: str):
    """Generate comparison plots for all baselines."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Baseline Comparison: Learning Curves", fontsize=14)
    
    colors = {"Random": "red", "Round-Robin": "blue", "Max-Variance": "green", "ACE": "purple"}
    
    # 1. Total loss over steps (averaged across episodes)
    ax = axes[0, 0]
    for name, df in results.items():
        mean_loss = df.groupby("step")["total_loss"].mean()
        std_loss = df.groupby("step")["total_loss"].std()
        ax.plot(mean_loss.index, mean_loss.values, label=name, color=colors.get(name, "gray"))
        ax.fill_between(mean_loss.index, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.2, color=colors.get(name, "gray"))
    ax.set_xlabel("Step")
    ax.set_ylabel("Total MSE Loss")
    ax.set_title("Convergence Rate")
    ax.legend()
    ax.set_yscale("log")
    
    # 2-6. Per-node losses
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    for idx, node in enumerate(nodes):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        for name, df in results.items():
            mean_loss = df.groupby("step")[f"loss_{node}"].mean()
            ax.plot(mean_loss.index, mean_loss.values, label=name, color=colors.get(name, "gray"))
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.set_title(f"{node} Mechanism Loss")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"), dpi=150)
    plt.close()
    
    # Intervention distribution plot
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
        
    for idx, (name, df) in enumerate(results.items()):
        ax = axes[idx]
        counts = df["target"].value_counts()
        ax.bar(counts.index, counts.values, color=colors.get(name, "gray"))
        ax.set_xlabel("Target Node")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: Intervention Distribution")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intervention_distribution.png"), dpi=150)
    plt.close()
    
    logging.info(f"Saved plots to {output_dir}")


def print_summary(results: Dict[str, pd.DataFrame]):
    """Print summary statistics for each baseline."""
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    
    for name, df in results.items():
        print(f"\n--- {name} ---")
        
        # Final step losses (averaged across episodes)
        final_step = df["step"].max()
        final_df = df[df["step"] == final_step]
        
        print(f"  Final Total Loss: {final_df['total_loss'].mean():.4f} ± {final_df['total_loss'].std():.4f}")
        
        for node in ["X1", "X2", "X3", "X4", "X5"]:
            col = f"loss_{node}"
            mean_loss = final_df[col].mean()
            status = "✓" if mean_loss < 0.5 else "✗"
            print(f"  {node}: {mean_loss:.4f} {status}")
            
        # Intervention distribution
        counts = df["target"].value_counts(normalize=True)
        print(f"  Intervention Distribution:")
        for node in counts.index:
            print(f"    {node}: {counts[node]:.1%}")
            
    print("\n" + "=" * 70)


# ----------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ACE Baseline Comparison")
    parser.add_argument("--baseline", type=str, choices=["random", "round_robin", "max_variance"],
                        help="Which baseline to run")
    parser.add_argument("--all", action="store_true", help="Run all baselines")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=25, help="Steps per episode")
    parser.add_argument("--obs_train_interval", type=int, default=5,
                        help="Observational training interval (0=disabled)")
    parser.add_argument("--obs_train_samples", type=int, default=100,
                        help="Observational samples per injection")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"baselines_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "baselines.log")),
            logging.StreamHandler()
        ]
    )
    
    # Ground truth
    oracle = GroundTruthSCM()
    nodes = oracle.nodes
    
    # Select baselines to run
    results = {}
    
    baselines_to_run = []
    if args.all:
        baselines_to_run = ["random", "round_robin", "max_variance"]
    elif args.baseline:
        baselines_to_run = [args.baseline]
    else:
        parser.print_help()
        return
        
    # Run each baseline
    for baseline_name in baselines_to_run:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running {baseline_name.upper()} baseline...")
        logging.info(f"{'='*50}")
        
        if baseline_name == "random":
            policy = RandomPolicy(nodes)
        elif baseline_name == "round_robin":
            policy = RoundRobinPolicy(nodes)
        elif baseline_name == "max_variance":
            policy = MaxVariancePolicy(nodes)
            
        df = run_baseline(
            policy, oracle,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            obs_train_interval=args.obs_train_interval,
            obs_train_samples=args.obs_train_samples
        )
        
        results[policy.name] = df
        df.to_csv(os.path.join(run_dir, f"{baseline_name}_results.csv"), index=False)
        
    # Generate comparison plots and summary
    if len(results) > 1:
        plot_comparison(results, run_dir)
    print_summary(results)
    
    logging.info(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
