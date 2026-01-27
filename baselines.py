#!/usr/bin/env python3
"""
Baseline Comparison Script for ACE (Active Causal Experimentalist)

This script implements four baseline intervention strategies for comparison
against the learned ACE (DPO) policy:

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

4. PPO (Proximal Policy Optimization)
   - Actor-Critic RL baseline with same reward signal as DPO
   - Uses GAE for advantage estimation, clipped surrogate objective
   - Fair comparison: same bonuses, replay buffer, observational training
   - Tests paper's claim that DPO outperforms value-based RL

Usage:
    python baselines.py --baseline random --episodes 100
    python baselines.py --baseline round_robin --episodes 100
    python baselines.py --baseline max_variance --episodes 100
    python baselines.py --baseline ppo --episodes 100
    python baselines.py --all --episodes 100           # Run all except PPO
    python baselines.py --all_with_ppo --episodes 100  # Run all including PPO
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
# 2.5 PPO POLICY (FAIR COMPARISON WITH DPO)
# ----------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO baseline.
    
    Architecture matches the complexity of the DPO policy:
    - Shared feature extractor processes state (node losses, intervention history)
    - Actor head outputs action distribution (node selection + value)
    - Critic head estimates state value for advantage computation
    
    This is designed for fair comparison with the LLM-based DPO policy.
    """
    
    def __init__(self, nodes: List[str], hidden_dim: int = 128, 
                 value_min: float = -5.0, value_max: float = 5.0,
                 n_value_bins: int = 21):
        super().__init__()
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.value_min = value_min
        self.value_max = value_max
        self.n_value_bins = n_value_bins
        
        # State: [node_losses (5), intervention_counts (5), recent_targets_onehot (5)]
        state_dim = self.n_nodes * 3
        
        # Shared feature extractor (similar complexity to LLM embedding)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor: outputs logits for node selection and value bin selection
        self.actor_node = nn.Linear(hidden_dim, self.n_nodes)
        self.actor_value = nn.Linear(hidden_dim, n_value_bins)
        
        # Critic: estimates V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _encode_state(self, node_losses: Dict[str, float], 
                      intervention_counts: Dict[str, int],
                      recent_targets: List[str]) -> torch.Tensor:
        """Encode state into tensor for network input."""
        # Node losses (normalized)
        losses = [node_losses.get(n, 1.0) for n in self.nodes]
        max_loss = max(losses) + 1e-6
        losses_norm = [l / max_loss for l in losses]
        
        # Intervention counts (normalized)
        counts = [intervention_counts.get(n, 0) for n in self.nodes]
        total_counts = sum(counts) + 1e-6
        counts_norm = [c / total_counts for c in counts]
        
        # Recent targets (one-hot sum of last 10)
        recent_onehot = [0.0] * self.n_nodes
        for t in recent_targets[-10:]:
            if t in self.nodes:
                idx = self.nodes.index(t)
                recent_onehot[idx] += 0.1
                
        state = losses_norm + counts_norm + recent_onehot
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning node logits, value logits, and state value."""
        features = self.shared(state)
        node_logits = self.actor_node(features)
        value_logits = self.actor_value(features)
        state_value = self.critic(features)
        return node_logits, value_logits, state_value
    
    def get_action(self, node_losses: Dict[str, float],
                   intervention_counts: Dict[str, int],
                   recent_targets: List[str]) -> Tuple[str, float, torch.Tensor, torch.Tensor]:
        """Sample action from policy and return log probabilities."""
        state = self._encode_state(node_losses, intervention_counts, recent_targets)
        node_logits, value_logits, state_value = self.forward(state)
        
        # Sample node
        node_dist = torch.distributions.Categorical(logits=node_logits)
        node_idx = node_dist.sample()
        node_log_prob = node_dist.log_prob(node_idx)
        target = self.nodes[node_idx.item()]
        
        # Sample value bin
        value_dist = torch.distributions.Categorical(logits=value_logits)
        value_idx = value_dist.sample()
        value_log_prob = value_dist.log_prob(value_idx)
        
        # Convert bin to continuous value
        bin_width = (self.value_max - self.value_min) / (self.n_value_bins - 1)
        value = self.value_min + value_idx.item() * bin_width
        # Add small noise for diversity
        value += random.uniform(-bin_width/4, bin_width/4)
        value = np.clip(value, self.value_min, self.value_max)
        
        total_log_prob = node_log_prob + value_log_prob
        
        return target, value, total_log_prob, state_value.squeeze()
    
    def evaluate_action(self, state: torch.Tensor, node_idx: torch.Tensor, 
                        value_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy for given state-action pair."""
        node_logits, value_logits, state_value = self.forward(state)
        
        node_dist = torch.distributions.Categorical(logits=node_logits)
        value_dist = torch.distributions.Categorical(logits=value_logits)
        
        node_log_prob = node_dist.log_prob(node_idx)
        value_log_prob = value_dist.log_prob(value_idx)
        
        entropy = node_dist.entropy() + value_dist.entropy()
        
        return node_log_prob + value_log_prob, entropy, state_value.squeeze()


class PPOPolicy:
    """
    Proximal Policy Optimization baseline for fair comparison with DPO.
    
    Key fairness considerations:
    - Same reward signal: Information Gain (ΔL) + coverage bonuses
    - Same replay buffer for learner training
    - Same observational training interval
    - Similar network capacity to LLM policy
    
    PPO-specific components:
    - Actor-Critic architecture
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function loss
    """
    
    def __init__(self, nodes: List[str], value_min: float = -5.0, value_max: float = 5.0,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 n_value_bins: int = 21, ppo_epochs: int = 4, mini_batch_size: int = 8):
        self.nodes = nodes
        self.value_min = value_min
        self.value_max = value_max
        self.name = "PPO"
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.n_value_bins = n_value_bins
        
        # Actor-Critic network
        self.ac = PPOActorCritic(nodes, value_min=value_min, value_max=value_max,
                                  n_value_bins=n_value_bins)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Trajectory buffer for PPO updates
        self.states = []
        self.node_indices = []
        self.value_indices = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Tracking for fair comparison
        self.intervention_counts = {n: 0 for n in nodes}
        self.recent_targets = []
        
        # Logging
        self.loss_history = []
        self.value_loss_history = []
        self.policy_loss_history = []
        
    def select_intervention(self, student: StudentSCM, oracle: GroundTruthSCM = None,
                            node_losses: Dict[str, float] = None, **kwargs) -> Tuple[str, float]:
        """Select intervention using current policy."""
        if node_losses is None:
            node_losses = {n: 1.0 for n in self.nodes}
            
        target, value, log_prob, state_value = self.ac.get_action(
            node_losses, self.intervention_counts, self.recent_targets
        )
        
        # Store for later update
        state = self.ac._encode_state(node_losses, self.intervention_counts, self.recent_targets)
        node_idx = self.nodes.index(target)
        bin_width = (self.value_max - self.value_min) / (self.n_value_bins - 1)
        value_idx = int((value - self.value_min) / bin_width)
        value_idx = np.clip(value_idx, 0, self.n_value_bins - 1)
        
        self.states.append(state)
        self.node_indices.append(node_idx)
        self.value_indices.append(value_idx)
        self.log_probs.append(log_prob.detach())
        self.values.append(state_value.detach())
        
        # Update tracking
        self.intervention_counts[target] = self.intervention_counts.get(target, 0) + 1
        self.recent_targets.append(target)
        if len(self.recent_targets) > 50:
            self.recent_targets.pop(0)
            
        return target, value
    
    def store_reward(self, reward: float, done: bool = False):
        """Store reward for the last action."""
        self.rewards.append(reward)
        self.dones.append(done)
        
    def compute_gae(self, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t].item()
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """PPO policy update using collected trajectories."""
        if len(self.rewards) < self.mini_batch_size:
            return
            
        # Compute advantages
        advantages, returns = self.compute_gae()
        
        # Prepare batch data
        states = torch.cat(self.states, dim=0)
        node_indices = torch.tensor(self.node_indices, dtype=torch.long)
        value_indices = torch.tensor(self.value_indices, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(len(self.rewards))
            
            for start in range(0, len(self.rewards), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(self.rewards))
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_node_idx = node_indices[batch_indices]
                batch_value_idx = value_indices[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                new_log_probs, entropy, new_values = self.ac.evaluate_action(
                    batch_states, batch_node_idx, batch_value_idx
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss - FIX: Ensure shapes match
                # new_values and batch_returns should both be 1D tensors of same length
                if new_values.dim() == 0:
                    new_values = new_values.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
                
        # Log
        if n_updates > 0:
            self.loss_history.append(total_policy_loss / n_updates)
            self.value_loss_history.append(total_value_loss / n_updates)
            self.policy_loss_history.append(total_entropy / n_updates)
            
        # Clear buffer
        self.clear_buffer()
        
    def clear_buffer(self):
        """Clear trajectory buffer."""
        self.states = []
        self.node_indices = []
        self.value_indices = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def reset_episode(self):
        """Reset per-episode state (not the learned policy)."""
        self.intervention_counts = {n: 0 for n in self.nodes}
        self.recent_targets = []


def calculate_reward_with_bonuses(loss_before: float, loss_after: float,
                                   target: str, node_losses: Dict[str, float],
                                   intervention_counts: Dict[str, int],
                                   nodes: List[str], graph: Dict[str, List[str]]) -> float:
    """
    Calculate reward with same bonuses as DPO for fair comparison.
    
    This mirrors the score calculation in ace_experiments.py to ensure
    PPO sees the same reward signal that shapes DPO preferences.
    """
    # Base reward: information gain (scaled same as DPO)
    delta = loss_before - loss_after
    reward = delta * 10.0
    reward = float(np.clip(reward, -2.0, 400.0))
    
    # Coverage bonus: encourage exploring under-sampled nodes
    total_interventions = sum(intervention_counts.values()) + 1
    target_count = intervention_counts.get(target, 0)
    coverage_bonus = 60.0 * (1.0 - target_count / total_interventions)
    
    # Collapse penalty: discourage fixating on one node
    if total_interventions > 10:
        top_count = max(intervention_counts.values())
        top_frac = top_count / total_interventions
        if top_frac > 0.30:
            collapse_penalty = 150.0 * (top_frac - 0.30)
        else:
            collapse_penalty = 0.0
    else:
        collapse_penalty = 0.0
        
    # Leaf penalty: discourage intervening on nodes with no children
    children = [n for n in nodes if target in graph.get(n, [])]
    if not children:
        leaf_penalty = 40.0
    else:
        leaf_penalty = 0.0
        
    # High-loss bonus: prefer interventions on high-loss mechanisms
    target_loss = node_losses.get(target, 0.0)
    avg_loss = np.mean(list(node_losses.values())) + 1e-6
    urgency_bonus = 20.0 * (target_loss / avg_loss) if avg_loss > 0 else 0.0
    
    total_reward = reward + coverage_bonus - collapse_penalty - leaf_penalty + urgency_bonus
    return total_reward


# ----------------------------------------------------------------
# 3. TRAINING LOOP
# ----------------------------------------------------------------

class SCMLearner:
    """Learner that trains the student SCM on experimental data."""
    
    def __init__(self, student: StudentSCM, lr: float = 2e-3, buffer_size: int = 50, oracle: GroundTruthSCM = None):
        self.student = student
        self.oracle = oracle
        self.optimizer = optim.Adam(student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = []
        self.buffer_size = buffer_size
        self._critic = None
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current mechanism losses."""
        if self._critic is None and self.oracle is not None:
            self._critic = ScientificCritic(self.oracle)
        
        if self._critic is None or self._critic.val_data is None:
            # No oracle/validation data - return zeros
            return {node: 0.0 for node in self.student.nodes}
        
        _, node_losses = self._critic.evaluate(self.student)
        return node_losses
    
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


# ----------------------------------------------------------------
# SIMPLE BASELINE RUNNERS FOR CRITICAL EXPERIMENTS
# ----------------------------------------------------------------

def run_random_policy(scm: GroundTruthSCM, learner, episodes: int):
    """Run random intervention policy for fixed number of episodes."""
    for ep in range(episodes):
        node = random.choice(scm.nodes)
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data, intervened=node)
    return learner.evaluate()


def run_round_robin_policy(scm: GroundTruthSCM, learner, episodes: int):
    """Run round-robin intervention policy."""
    for ep in range(episodes):
        node = scm.nodes[ep % len(scm.nodes)]
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data, intervened=node)
    return learner.evaluate()


def run_max_variance_policy(scm: GroundTruthSCM, learner, episodes: int):
    """Run max-variance (uncertainty sampling) policy."""
    for ep in range(episodes):
        # Select node with highest prediction variance
        max_var = -1
        best_node = scm.nodes[0]
        
        learner.student.eval()  # Evaluation mode for variance estimation
        with torch.no_grad():
            for node in scm.nodes:
                # Simple variance estimation from prediction uncertainty
                # Generate test data
                test_data = scm.generate(10, interventions=None)
                preds = learner.student(test_data)
                
                # Variance of predictions for this node
                variance = preds[node].var().item()
                
                if variance > max_var:
                    max_var = variance
                    best_node = node
        
        learner.student.train()  # Back to training mode
        
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={best_node: value})
        learner.train_step(data, intervened=best_node)
    
    return learner.evaluate()


def run_baseline(policy, oracle: GroundTruthSCM, n_episodes: int = 100,
                 steps_per_episode: int = 25, obs_train_interval: int = 5,
                 obs_train_samples: int = 100) -> pd.DataFrame:
    """Run a baseline policy and collect metrics."""
    
    critic = ScientificCritic(oracle)
    all_records = []
    is_ppo = isinstance(policy, PPOPolicy)
    
    for episode in range(n_episodes):
        # Fresh student each episode
        student = StudentSCM(oracle)
        learner = SCMLearner(student, oracle=oracle)
        
        # Reset policy state if needed
        if hasattr(policy, 'reset'):
            policy.reset()
        if hasattr(policy, 'reset_episode'):
            policy.reset_episode()
            
        # Get initial loss for reward computation
        prev_loss, prev_node_losses = critic.evaluate(student)
        
        for step in range(steps_per_episode):
            # Select intervention (PPO needs node_losses for state encoding)
            if is_ppo:
                target, value = policy.select_intervention(
                    student, oracle=oracle, node_losses=prev_node_losses
                )
            else:
                target, value = policy.select_intervention(student, oracle=oracle)
            
            # Execute intervention
            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data, intervened=target)
            
            # Periodic observational training
            if obs_train_interval > 0 and step > 0 and step % obs_train_interval == 0:
                learner.observational_train(oracle, n_samples=obs_train_samples)
            
            # Evaluate
            total_loss, node_losses = critic.evaluate(student)
            
            # For PPO: compute and store reward
            if is_ppo:
                reward = calculate_reward_with_bonuses(
                    prev_loss, total_loss, target, prev_node_losses,
                    policy.intervention_counts, oracle.nodes, oracle.graph
                )
                done = (step == steps_per_episode - 1)
                policy.store_reward(reward, done)
            
            record = {
                "episode": episode,
                "step": step,
                "target": target,
                "value": value,
                "total_loss": total_loss,
                **{f"loss_{node}": loss for node, loss in node_losses.items()}
            }
            all_records.append(record)
            
            # Update for next step
            prev_loss = total_loss
            prev_node_losses = node_losses
            
        # PPO: update policy at end of episode
        if is_ppo:
            policy.update()
            
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
    
    colors = {
        "Random": "red", 
        "Round-Robin": "blue", 
        "Max-Variance": "green", 
        "PPO": "orange",
        "ACE": "purple"
    }
    
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
            status = "[OK]" if mean_loss < 0.5 else "[FAIL]"
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
    parser.add_argument("--baseline", type=str, 
                        choices=["random", "round_robin", "max_variance", "ppo"],
                        help="Which baseline to run")
    parser.add_argument("--all", action="store_true", help="Run all baselines (excluding PPO)")
    parser.add_argument("--all_with_ppo", action="store_true", help="Run all baselines including PPO")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=25, help="Steps per episode")
    parser.add_argument("--obs_train_interval", type=int, default=3,
                        help="Observational training interval (0=disabled)")
    parser.add_argument("--obs_train_samples", type=int, default=200,
                        help="Observational samples per injection")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    # PPO-specific arguments
    parser.add_argument("--ppo_lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--ppo_clip", type=float, default=0.2, help="PPO clip epsilon")
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
    if args.all_with_ppo:
        baselines_to_run = ["random", "round_robin", "max_variance", "ppo"]
    elif args.all:
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
        elif baseline_name == "ppo":
            policy = PPOPolicy(
                nodes, 
                lr=args.ppo_lr,
                clip_epsilon=args.ppo_clip,
                ppo_epochs=args.ppo_epochs
            )
            logging.info(f"  PPO Config: lr={args.ppo_lr}, clip={args.ppo_clip}, epochs={args.ppo_epochs}")
            
        df = run_baseline(
            policy, oracle,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            obs_train_interval=args.obs_train_interval,
            obs_train_samples=args.obs_train_samples
        )
        
        results[policy.name] = df
        df.to_csv(os.path.join(run_dir, f"{baseline_name}_results.csv"), index=False)
        
        # Save PPO-specific training curves
        if baseline_name == "ppo" and hasattr(policy, 'loss_history') and policy.loss_history:
            ppo_df = pd.DataFrame({
                "policy_loss": policy.loss_history,
                "value_loss": policy.value_loss_history,
            })
            ppo_df.to_csv(os.path.join(run_dir, "ppo_training.csv"), index=False)
            logging.info(f"  Saved PPO training curves ({len(policy.loss_history)} updates)")
        
    # Generate comparison plots and summary
    if len(results) > 1:
        plot_comparison(results, run_dir)
    print_summary(results)
    
    logging.info(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
