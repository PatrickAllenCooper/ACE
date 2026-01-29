#!/usr/bin/env python3
"""
Coupled Duffing Oscillators Experiment

A non-linear mass-spring-damper chain governed by ODEs:
    ẍ_i + δẋ_i + αx_i + βx_i³ = F_ext(t) + k(x_{i-1} - x_i) + k(x_{i+1} - x_i)

The learner must discover the chain topology by breaking spurious correlations
through strategic "clamping" interventions (do(x_mid = 0)).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import logging
import argparse
from datetime import datetime
import pandas as pd


class DuffingOscillatorChain:
    """
    Ground truth: Chain of coupled Duffing oscillators.
    
    True causal structure: X1 → X2 → X3 → X4 (chain)
    But under observation, synchronization creates spurious X1 ↔ X4 correlations.
    """
    
    def __init__(self, n_masses: int = 4, delta: float = 0.3, alpha: float = 1.0,
                 beta: float = 0.2, coupling: float = 0.5, dt: float = 0.01):
        self.n_masses = n_masses
        self.delta = delta  # Damping
        self.alpha = alpha  # Linear stiffness
        self.beta = beta    # Nonlinear stiffness
        self.coupling = coupling  # Inter-mass coupling
        self.dt = dt
        
        # True graph: chain topology
        self.nodes = [f"X{i+1}" for i in range(n_masses)]
        self.true_graph = {self.nodes[i]: [self.nodes[i-1]] if i > 0 else [] 
                          for i in range(n_masses)}
        
    def _dynamics(self, t: float, state: np.ndarray, 
                  interventions: Dict[str, float] = None) -> np.ndarray:
        """ODE right-hand side for the coupled system."""
        interventions = interventions or {}
        n = self.n_masses
        x = state[:n]   # Positions
        v = state[n:]   # Velocities
        
        dxdt = v.copy()
        dvdt = np.zeros(n)
        
        for i in range(n):
            node = self.nodes[i]
            
            # If intervened, clamp position and velocity
            if node in interventions:
                dxdt[i] = 0
                dvdt[i] = 0
                continue
            
            # Duffing dynamics
            dvdt[i] = -self.delta * v[i] - self.alpha * x[i] - self.beta * x[i]**3
            
            # Coupling to neighbors
            if i > 0:
                if self.nodes[i-1] in interventions:
                    x_left = interventions[self.nodes[i-1]]
                else:
                    x_left = x[i-1]
                dvdt[i] += self.coupling * (x_left - x[i])
            if i < n - 1:
                if self.nodes[i+1] in interventions:
                    x_right = interventions[self.nodes[i+1]]
                else:
                    x_right = x[i+1]
                dvdt[i] += self.coupling * (x_right - x[i])
                
        return np.concatenate([dxdt, dvdt])
    
    def generate(self, n_samples: int, interventions: Dict[str, float] = None,
                 t_span: Tuple[float, float] = (0, 10)) -> Dict[str, torch.Tensor]:
        """Generate trajectory data via RK45 integration."""
        interventions = interventions or {}
        
        # Random initial conditions
        x0 = np.random.randn(self.n_masses) * 0.5
        v0 = np.random.randn(self.n_masses) * 0.2
        
        # Apply interventions to initial conditions
        for node, val in interventions.items():
            idx = self.nodes.index(node)
            x0[idx] = val
            v0[idx] = 0  # Clamped = zero velocity
        
        state0 = np.concatenate([x0, v0])
        
        # Integrate
        t_eval = np.linspace(t_span[0], t_span[1], n_samples)
        sol = solve_ivp(
            lambda t, y: self._dynamics(t, y, interventions),
            t_span, state0, t_eval=t_eval, method='RK45'
        )
        
        # Extract positions
        data = {}
        for i, node in enumerate(self.nodes):
            if node in interventions:
                data[node] = torch.full((n_samples,), interventions[node])
            else:
                data[node] = torch.tensor(sol.y[i], dtype=torch.float32)
                
        return data
    
    def get_parents(self, node: str) -> List[str]:
        return self.true_graph.get(node, [])


class OscillatorLearner(nn.Module):
    """Learner that tries to discover the causal structure."""
    
    def __init__(self, nodes: List[str], hidden_dim: int = 32):
        super().__init__()
        self.nodes = nodes
        self.n_nodes = len(nodes)
        
        # Learn adjacency weights (which nodes influence which)
        self.adjacency = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        
        # Per-edge MLPs for mechanism learning
        self.mechanisms = nn.ModuleDict()
        for i, target in enumerate(nodes):
            self.mechanisms[target] = nn.Sequential(
                nn.Linear(self.n_nodes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict each node from learned parents."""
        x = torch.stack([data[n] for n in self.nodes], dim=1)  # [batch, n_nodes]
        
        # Soft adjacency mask
        adj = torch.sigmoid(self.adjacency)  # [n_nodes, n_nodes]
        
        predictions = {}
        for i, target in enumerate(self.nodes):
            # Weighted input from potential parents
            weighted_input = x * adj[:, i].unsqueeze(0)  # [batch, n_nodes]
            pred = self.mechanisms[target](weighted_input).squeeze(-1)
            predictions[target] = pred
            
        return predictions
    
    def get_learned_graph(self, threshold: float = 0.5) -> Dict[str, List[str]]:
        """Extract learned causal structure."""
        adj = torch.sigmoid(self.adjacency).detach().numpy()
        graph = {}
        for i, target in enumerate(self.nodes):
            parents = [self.nodes[j] for j in range(self.n_nodes) 
                      if adj[j, i] > threshold and j != i]
            graph[target] = parents
        return graph


def run_duffing_experiment(n_episodes: int = 100, steps_per_episode: int = 20,
                           output_dir: str = "results"):
    """Run the Duffing oscillator causal discovery experiment."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"duffing_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Setup
    oracle = DuffingOscillatorChain(n_masses=4)
    learner = OscillatorLearner(oracle.nodes)
    optimizer = optim.Adam(learner.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    records = []
    
    logging.info(f"True graph: {oracle.true_graph}")
    
    for episode in range(n_episodes):
        # Reset learner
        learner = OscillatorLearner(oracle.nodes)
        optimizer = optim.Adam(learner.parameters(), lr=1e-3)
        
        intervention_counts = {n: 0 for n in oracle.nodes}
        
        for step in range(steps_per_episode):
            # Strategy: intervene on middle nodes to break synchronization
            # This is the "clamping" strategy the paper describes
            if step < 5:
                # Initial: observe without intervention
                interventions = None
                target = None
            else:
                # Clamp middle nodes to break spurious correlations
                middle_idx = len(oracle.nodes) // 2
                target = oracle.nodes[middle_idx]
                interventions = {target: 0.0}  # Clamp to zero
                intervention_counts[target] += 1
            
            # Generate data
            data = oracle.generate(n_samples=100, interventions=interventions)
            
            # Train learner
            learner.train()
            for _ in range(10):
                optimizer.zero_grad()
                preds = learner(data)
                loss = sum(loss_fn(preds[n], data[n]) for n in oracle.nodes)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            learner.eval()
            with torch.no_grad():
                preds = learner(data)
                eval_loss = sum(loss_fn(preds[n], data[n]).item() for n in oracle.nodes)
            
            records.append({
                "episode": episode,
                "step": step,
                "target": target,
                "loss": eval_loss
            })
        
        # Check learned graph
        if episode % 20 == 0:
            learned = learner.get_learned_graph()
            logging.info(f"Episode {episode}: Loss={eval_loss:.4f}, Learned={learned}")
    
    # Save results
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, "duffing_results.csv"), index=False)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    for ep in range(0, n_episodes, 20):
        ep_data = df[df["episode"] == ep]
        ax.plot(ep_data["step"], ep_data["loss"], label=f"Ep {ep}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Duffing Oscillators: Causal Discovery via Clamping")
    ax.legend()
    plt.savefig(os.path.join(run_dir, "duffing_learning.png"), dpi=150)
    plt.close()
    
    logging.info(f"Results saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    
    run_duffing_experiment(args.episodes, args.steps, args.output)
