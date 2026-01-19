#!/usr/bin/env python3
"""
Complex 15-Node SCM Experiment - Hard Benchmark for Active Learning

This SCM is designed to be HARD for random/naive strategies:
- 15 nodes (vs 5 in simple benchmark)
- 4 colliders (vs 1) including nested collider
- Hierarchical dependencies (3 levels deep)
- Mix of linear, polynomial, trigonometric, interaction terms
- Higher noise to require more samples

Structure:
    Layer 1 (Roots):     R1, R2, R3
    Layer 2 (Linear):    L1 = f(R1, R2)  [Collider 1]
                         L2 = f(R1)
                         L3 = f(R3)
    Layer 3 (Nonlinear): N1 = f(L1, L2)  [Collider 2, nested]
                         N2 = f(L1)
                         N3 = f(L3)
    Layer 4 (Complex):   C1 = f(N1, N2)  [Collider 3]
                         C2 = f(N2, N3)  [Collider 4]
    Layer 5 (Final):     F1 = f(C1)
                         F2 = f(C2)
                         F3 = f(C1, C2)  [Final collider]

Why This is Hard:
1. Multiple colliders require discovering each separately
2. Nested collider (N1 depends on L1 which is itself a collider)
3. Random sampling spreads samples too thin across 15 nodes
4. Need strategic focus on high-loss colliders
5. Deep hierarchy requires careful intervention ordering
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter


class ComplexGroundTruthSCM:
    """15-node SCM with 4 colliders and hierarchical structure."""
    
    def __init__(self):
        self.nodes = ["R1", "R2", "R3",  # Layer 1: Roots
                     "L1", "L2", "L3",   # Layer 2: Linear
                     "N1", "N2", "N3",   # Layer 3: Nonlinear
                     "C1", "C2",         # Layer 4: Complex
                     "F1", "F2", "F3"]   # Layer 5: Final
        
        self.graph = {
            # Roots
            "R1": [],
            "R2": [],
            "R3": [],
            # Layer 2
            "L1": ["R1", "R2"],  # Collider 1
            "L2": ["R1"],
            "L3": ["R3"],
            # Layer 3
            "N1": ["L1", "L2"],  # Collider 2 (nested - parents are affected by R1)
            "N2": ["L1"],
            "N3": ["L3"],
            # Layer 4
            "C1": ["N1", "N2"],  # Collider 3
            "C2": ["N2", "N3"],  # Collider 4
            # Layer 5
            "F1": ["C1"],
            "F2": ["C2"],
            "F3": ["C1", "C2"],  # Final collider
        }
        
        self.noise_std = 0.2  # Higher noise than simple SCM
        
    def get_parents(self, node: str) -> List[str]:
        return self.graph.get(node, [])
    
    def generate(self, n_samples: int, interventions: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Generate samples from the SCM."""
        interventions = interventions or {}
        
        # Layer 1: Roots
        R1 = torch.full((n_samples,), interventions["R1"]) if "R1" in interventions else \
             torch.randn(n_samples) * 1.0
        R2 = torch.full((n_samples,), interventions["R2"]) if "R2" in interventions else \
             torch.randn(n_samples) * 1.5 + 1.0
        R3 = torch.full((n_samples,), interventions["R3"]) if "R3" in interventions else \
             torch.randn(n_samples) * 0.8 - 0.5
        
        # Layer 2: Linear combinations
        L1 = torch.full((n_samples,), interventions["L1"]) if "L1" in interventions else \
             1.5 * R1 - 0.8 * R2 + 2.0 + torch.randn(n_samples) * self.noise_std
        
        L2 = torch.full((n_samples,), interventions["L2"]) if "L2" in interventions else \
             -2.0 * R1 + 1.5 + torch.randn(n_samples) * self.noise_std
        
        L3 = torch.full((n_samples,), interventions["L3"]) if "L3" in interventions else \
             3.0 * R3 - 1.0 + torch.randn(n_samples) * self.noise_std
        
        # Layer 3: Nonlinear combinations
        N1 = torch.full((n_samples,), interventions["N1"]) if "N1" in interventions else \
             0.5 * L1**2 - L2 + torch.sin(L1) + torch.randn(n_samples) * self.noise_std
        
        N2 = torch.full((n_samples,), interventions["N2"]) if "N2" in interventions else \
             torch.cos(L1) + 0.3 * L1 + torch.randn(n_samples) * self.noise_std
        
        N3 = torch.full((n_samples,), interventions["N3"]) if "N3" in interventions else \
             0.2 * L3**2 + torch.randn(n_samples) * self.noise_std
        
        # Layer 4: Complex interactions
        C1 = torch.full((n_samples,), interventions["C1"]) if "C1" in interventions else \
             N1 * N2 + 0.5 * N1 - 0.3 * N2 + torch.randn(n_samples) * self.noise_std
        
        C2 = torch.full((n_samples,), interventions["C2"]) if "C2" in interventions else \
             torch.tanh(N2 + N3) + 0.5 * N3 + torch.randn(n_samples) * self.noise_std
        
        # Layer 5: Final outputs
        F1 = torch.full((n_samples,), interventions["F1"]) if "F1" in interventions else \
             0.8 * C1 + torch.randn(n_samples) * self.noise_std
        
        F2 = torch.full((n_samples,), interventions["F2"]) if "F2" in interventions else \
             1.2 * C2 + torch.randn(n_samples) * self.noise_std
        
        F3 = torch.full((n_samples,), interventions["F3"]) if "F3" in interventions else \
             0.4 * C1 - 0.6 * C2 + torch.sin(C1 - C2) + torch.randn(n_samples) * self.noise_std
        
        return {
            "R1": R1, "R2": R2, "R3": R3,
            "L1": L1, "L2": L2, "L3": L3,
            "N1": N1, "N2": N2, "N3": N3,
            "C1": C1, "C2": C2,
            "F1": F1, "F2": F2, "F3": F3
        }


class ComplexStudentSCM(nn.Module):
    """Learner for complex SCM."""
    
    def __init__(self, oracle: ComplexGroundTruthSCM, hidden_dim: int = 32):
        super().__init__()
        self.nodes = oracle.nodes
        self.graph = oracle.graph
        self.mechanisms = nn.ModuleDict()
        
        for node in self.nodes:
            parents = self.get_parents(node)
            if not parents:
                # Root: learnable mean
                self.mechanisms[node] = nn.ParameterDict({
                    'mu': nn.Parameter(torch.zeros(1))
                })
            else:
                # Non-root: MLP
                n_parents = len(parents)
                self.mechanisms[node] = nn.Sequential(
                    nn.Linear(n_parents, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
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


class ComplexSCMLearner:
    """Learner with replay buffer for complex SCM."""
    
    def __init__(self, student: ComplexStudentSCM, lr: float = 1e-3, buffer_size: int = 50):
        self.student = student
        self.optimizer = optim.Adam(student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = []
        self.buffer_size = buffer_size
        
    def train_step(self, data: Dict[str, torch.Tensor], intervened: Optional[str] = None,
                   n_epochs: int = 100):
        """Train on new data, respecting intervention masks."""
        self.student.train()
        
        # Add to buffer
        self.buffer.append({"data": data, "intervened": intervened})
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Collate buffer
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
    
    def observational_train(self, oracle, n_samples: int = 100, n_epochs: int = 50):
        """Periodic observational training."""
        obs_data = oracle.generate(n_samples=n_samples, interventions=None)
        return self.train_step(obs_data, intervened=None, n_epochs=n_epochs)


class ComplexCritic:
    """Evaluates mechanism quality."""
    
    def __init__(self, oracle: ComplexGroundTruthSCM):
        self.oracle = oracle
        self.val_data = oracle.generate(n_samples=500)
    
    def evaluate(self, student: ComplexStudentSCM) -> tuple:
        """Compute total and per-node losses."""
        student.eval()
        with torch.no_grad():
            preds = student(self.val_data)
        
        node_losses = {}
        total_loss = 0
        collider_loss = 0
        
        # Identify colliders
        colliders = [n for n in student.nodes if len(student.get_parents(n)) > 1]
        
        for node in student.nodes:
            mse = ((preds[node] - self.val_data[node])**2).mean().item()
            node_losses[node] = mse
            total_loss += mse
            
            if node in colliders:
                collider_loss += mse
        
        return total_loss, node_losses, collider_loss


def run_complex_scm_experiment(policy_type: str = "random", n_episodes: int = 200,
                               steps_per_episode: int = 30, output_dir: str = "results"):
    """Run experiment on complex SCM."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"complex_scm_{policy_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "experiment.log")),
            logging.StreamHandler()
        ]
    )
    
    # Setup
    oracle = ComplexGroundTruthSCM()
    critic = ComplexCritic(oracle)
    
    # Identify colliders for special tracking
    colliders = [n for n in oracle.nodes if len(oracle.get_parents(n)) > 1]
    logging.info(f"Complex SCM with {len(oracle.nodes)} nodes")
    logging.info(f"Colliders (multi-parent): {colliders}")
    
    all_records = []
    
    for episode in range(n_episodes):
        student = ComplexStudentSCM(oracle)
        learner = ComplexSCMLearner(student, lr=2e-3)
        
        intervention_counts = Counter()
        
        for step in range(steps_per_episode):
            # Policy selection
            if policy_type == "random":
                target = np.random.choice(oracle.nodes)
                value = np.random.uniform(-5, 5)
            
            elif policy_type == "smart_random":
                # 50% random, 50% collider-focused
                if np.random.rand() < 0.5:
                    target = np.random.choice(oracle.nodes)
                else:
                    # Get current losses
                    _, node_losses, _ = critic.evaluate(student)
                    
                    # Find highest-loss collider
                    collider_losses = {c: node_losses[c] for c in colliders}
                    target_collider = max(collider_losses, key=collider_losses.get)
                    
                    # Intervene on one of its parents
                    parents = oracle.get_parents(target_collider)
                    target = np.random.choice(parents)
                
                value = np.random.uniform(-5, 5)
            
            elif policy_type == "greedy_collider":
                # Always focus on highest-loss collider's parents
                _, node_losses, _ = critic.evaluate(student)
                
                collider_losses = [(c, node_losses[c]) for c in colliders]
                collider_losses.sort(key=lambda x: x[1], reverse=True)
                
                # Pick highest-loss collider
                target_collider = collider_losses[0][0]
                parents = oracle.get_parents(target_collider)
                
                # Round-robin through parents
                parent_counts = {p: intervention_counts[p] for p in parents}
                target = min(parent_counts, key=parent_counts.get)
                value = np.random.uniform(-5, 5)
            
            # Execute intervention
            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data, intervened=target, n_epochs=100)
            intervention_counts[target] += 1
            
            # Periodic observational training
            if step > 0 and step % 5 == 0:
                learner.observational_train(oracle, n_samples=100, n_epochs=50)
            
            # Evaluate
            total_loss, node_losses, collider_loss = critic.evaluate(student)
            
            record = {
                "episode": episode,
                "step": step,
                "target": target,
                "value": value,
                "total_loss": total_loss,
                "collider_loss": collider_loss,
            }
            # Add per-node losses
            for node, loss in node_losses.items():
                record[f"loss_{node}"] = loss
            
            all_records.append(record)
        
        if episode % 20 == 0:
            logging.info(f"Episode {episode}/{n_episodes}: Total={total_loss:.3f}, Collider={collider_loss:.3f}")
    
    # Save results
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    
    # Final analysis
    final_df = df[df["step"] == df["step"].max()]
    
    logging.info("\n" + "=" * 70)
    logging.info(f"FINAL RESULTS ({policy_type})")
    logging.info("=" * 70)
    logging.info(f"Total Loss: {final_df['total_loss'].mean():.4f} ± {final_df['total_loss'].std():.4f}")
    logging.info(f"Collider Loss: {final_df['collider_loss'].mean():.4f} ± {final_df['collider_loss'].std():.4f}")
    
    logging.info("\nPer-Collider Losses:")
    for collider in colliders:
        col = f"loss_{collider}"
        mean_loss = final_df[col].mean()
        status = "✓" if mean_loss < 1.0 else "✗"
        logging.info(f"  {collider}: {mean_loss:.4f} {status}")
    
    # Intervention distribution
    total_interventions = sum(intervention_counts.values())
    logging.info("\nIntervention Distribution:")
    for node in sorted(intervention_counts, key=intervention_counts.get, reverse=True)[:10]:
        pct = 100.0 * intervention_counts[node] / total_interventions
        logging.info(f"  {node}: {pct:.1f}%")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss convergence
    ax = axes[0, 0]
    for ep in range(0, n_episodes, 40):
        ep_data = df[df["episode"] == ep]
        ax.plot(ep_data["step"], ep_data["total_loss"], alpha=0.3)
    
    mean_loss = df.groupby("step")["total_loss"].mean()
    ax.plot(mean_loss.index, mean_loss.values, 'k-', linewidth=2, label='Mean')
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"Convergence: {policy_type}")
    ax.set_yscale("log")
    ax.legend()
    
    # Collider loss convergence
    ax = axes[0, 1]
    mean_collider = df.groupby("step")["collider_loss"].mean()
    ax.plot(mean_collider.index, mean_collider.values, 'r-', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Collider Loss (sum of 5 colliders)")
    ax.set_title("Collider Learning Progress")
    ax.set_yscale("log")
    ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='Target (1.0 each)')
    ax.legend()
    
    # Per-collider final losses
    ax = axes[1, 0]
    collider_final = {c: final_df[f"loss_{c}"].mean() for c in colliders}
    bars = ax.bar(collider_final.keys(), collider_final.values())
    ax.axhline(y=1.0, color='red', linestyle='--', label='Target')
    ax.set_ylabel("Final Loss")
    ax.set_title("Per-Collider Performance")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Color bars by success
    for bar, (collider, loss) in zip(bars, collider_final.items()):
        bar.set_color('green' if loss < 1.0 else 'red')
    
    # Intervention distribution
    ax = axes[1, 1]
    top_10 = dict(intervention_counts.most_common(10))
    ax.barh(list(top_10.keys()), list(top_10.values()))
    ax.set_xlabel("Intervention Count")
    ax.set_title("Top 10 Intervention Targets")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "complex_scm_analysis.png"), dpi=150)
    plt.close()
    
    logging.info(f"\nResults saved to {run_dir}")
    
    return run_dir, df


def visualize_scm_structure(oracle: ComplexGroundTruthSCM, output_path: str):
    """Visualize the complex SCM structure."""
    G = nx.DiGraph()
    
    for node in oracle.nodes:
        G.add_node(node)
        for parent in oracle.get_parents(node):
            G.add_edge(parent, node)
    
    # Identify node types
    colliders = [n for n in oracle.nodes if len(oracle.get_parents(n)) > 1]
    roots = [n for n in oracle.nodes if len(oracle.get_parents(n)) == 0]
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    plt.figure(figsize=(14, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=20)
    
    # Draw nodes by type
    nx.draw_networkx_nodes(G, pos, nodelist=roots, node_color='lightgreen', 
                          node_size=800, label='Roots')
    nx.draw_networkx_nodes(G, pos, nodelist=colliders, node_color='red',
                          node_size=1000, label='Colliders', alpha=0.7)
    
    other_nodes = [n for n in oracle.nodes if n not in roots and n not in colliders]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='lightblue',
                          node_size=700)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Complex 15-Node SCM Structure\n5 Colliders (red), 3 Roots (green)", 
             fontsize=14, pad=20)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved SCM structure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex SCM Experiment")
    parser.add_argument("--policy", type=str, default="random",
                       choices=["random", "smart_random", "greedy_collider"],
                       help="Intervention policy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    
    # Visualize structure
    oracle = ComplexGroundTruthSCM()
    os.makedirs(args.output, exist_ok=True)
    visualize_scm_structure(oracle, os.path.join(args.output, "complex_scm_structure.png"))
    
    # Run experiment
    run_dir, df = run_complex_scm_experiment(
        policy_type=args.policy,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output
    )
    
    print(f"\n✓ Experiment complete: {run_dir}")
