#!/usr/bin/env python3
"""
Critical Experiments for ICML Reviewer Response

This script runs all experiments needed to address reviewer concerns:

1. EQUAL-BUDGET BASELINES (171 episodes)
   - Extend baselines to match ACE's average episode count
   - Enables fair final-accuracy comparison

2. EQUAL-BUDGET ACE (100 episodes) 
   - Cap ACE at 100 episodes to match baseline budget
   - Isolates algorithmic contribution from extra data

3. LOOKAHEAD ABLATION (Random Proposer)
   - Same lookahead mechanism (K=4, simulate, select best)
   - Random candidate generation instead of learned policy
   - Shows value of DPO training vs evaluate-and-select

4. LEARNING CURVES
   - Save per-episode loss for all methods
   - Enables AUC/sample-efficiency comparisons

5. 15-NODE COMPLEX SCM
   - Run all methods on harder benchmark
   - Validates scaling claims

Usage:
    python run_critical_experiments.py --all              # Run everything
    python run_critical_experiments.py --extended-baselines
    python run_critical_experiments.py --capped-ace
    python run_critical_experiments.py --lookahead-ablation
    python run_critical_experiments.py --complex-scm
    python run_critical_experiments.py --seeds 42 123 456 789 1011
"""

import argparse
import os
import sys
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import copy

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from existing codebase
from baselines import (
    GroundTruthSCM,
    StudentSCM,
    SCMLearner, 
    run_random_policy,
    run_round_robin_policy,
    run_max_variance_policy
)
from experiments.complex_scm import ComplexGroundTruthSCM


# ============================================================================
# LEARNING CURVE TRACKING
# ============================================================================

class LearningCurveTracker:
    """Track per-episode losses for learning curve analysis."""
    
    def __init__(self):
        self.episode_losses = []
        self.per_node_losses = []
    
    def record(self, episode: int, total_loss: float, node_losses: Dict[str, float]):
        self.episode_losses.append({'episode': episode, 'total_loss': total_loss})
        for node, loss in node_losses.items():
            self.per_node_losses.append({
                'episode': episode, 
                'node': node, 
                'loss': loss
            })
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_losses)
    
    def save(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Saved learning curve to {path}")


# ============================================================================
# EXTENDED BASELINES (171 episodes)
# ============================================================================

def run_extended_baselines(
    seeds: List[int],
    episodes: int = 171,
    output_dir: str = "results/extended_baselines"
):
    """Run all baselines for extended episode count."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"EXTENDED BASELINES - Seed {seed}, Episodes {episodes}")
        print(f"{'='*60}")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        scm = GroundTruthSCM()
        
        # Run each baseline
        for baseline_name, run_fn in [
            ('random', run_random_policy),
            ('round_robin', run_round_robin_policy),
            ('max_variance', run_max_variance_policy),
        ]:
            print(f"\nRunning {baseline_name}...")
            # Import StudentSCM from baselines
            from baselines import StudentSCM
            student = StudentSCM(scm)
            learner = SCMLearner(student, oracle=scm)
            tracker = LearningCurveTracker()
            
            # Modified to track learning curves
            final_losses = run_baseline_with_tracking(
                run_fn, scm, learner, episodes, tracker
            )
            
            total_loss = sum(final_losses.values())
            results.append({
                'seed': seed,
                'method': baseline_name,
                'episodes': episodes,
                'total_loss': total_loss,
                **{f'loss_{k}': v for k, v in final_losses.items()}
            })
            
            tracker.save(f"{output_dir}/{baseline_name}_seed{seed}_curve.csv")
        
        # CRITICAL: Save all results after each seed
        df_all = pd.DataFrame(results)
        df_all.to_csv(f"{output_dir}/extended_baselines_summary.csv", index=False)
        print(f"\n  ✓ Seed {seed} complete - {len(results)} total runs saved")
    
    # Save final summary
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/extended_baselines_summary.csv", index=False)
    print(f"\nSaved extended baseline results to {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXTENDED BASELINES SUMMARY (171 episodes)")
    print("="*60)
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        mean_loss = subset['total_loss'].mean()
        std_loss = subset['total_loss'].std()
        print(f"{method:15s}: {mean_loss:.3f} +/- {std_loss:.3f}")
    
    return df


def run_baseline_with_tracking(run_fn, scm, learner, episodes, tracker):
    """Wrapper to run baseline with learning curve tracking."""
    # Store original evaluate method
    original_evaluate = learner.evaluate
    episode_counter = [0]
    
    def tracked_evaluate():
        losses = original_evaluate()
        episode_counter[0] += 1
        tracker.record(episode_counter[0], sum(losses.values()), losses)
        return losses
    
    learner.evaluate = tracked_evaluate
    
    # Run the baseline
    final_losses = run_fn(scm, learner, episodes)
    
    return final_losses


# ============================================================================
# LOOKAHEAD ABLATION (Random Proposer)
# ============================================================================

def run_lookahead_ablation(
    seeds: List[int],
    episodes: int = 171,
    K: int = 4,
    output_dir: str = "results/lookahead_ablation"
):
    """
    Run lookahead mechanism with RANDOM candidate proposals.
    
    This isolates the value of the evaluate-and-select mechanism
    from the learned DPO policy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"LOOKAHEAD ABLATION (Random Proposer) - Seed {seed}")
        print(f"{'='*60}")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        scm = GroundTruthSCM()
        from baselines import StudentSCM
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        tracker = LearningCurveTracker()
        
        # Run lookahead with random proposals
        for episode in range(1, episodes + 1):
            # Generate K random candidate interventions
            candidates = []
            for _ in range(K):
                node = random.choice(scm.nodes)
                value = random.uniform(-5, 5)
                candidates.append((node, value))
            
            # Simulate each on cloned learner, measure info gain
            best_candidate = None
            best_gain = -float('inf')
            
            for node, value in candidates:
                # Clone learner
                cloned_learner = copy.deepcopy(learner)
                
                # Get loss before
                losses_before = cloned_learner.evaluate()
                total_before = sum(losses_before.values())
                
                # Apply intervention and train
                intervention = {node: value}
                data = scm.generate(100, interventions=intervention)
                cloned_learner.train_step(data)
                
                # Get loss after
                losses_after = cloned_learner.evaluate()
                total_after = sum(losses_after.values())
                
                # Compute info gain
                gain = total_before - total_after
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = (node, value)
            
            # Execute best candidate on actual learner
            node, value = best_candidate
            intervention = {node: value}
            data = scm.generate(100, interventions=intervention)
            learner.train_step(data)
            
            # Track learning curve
            if episode % 10 == 0 or episode == 1:
                losses = learner.evaluate()
                tracker.record(episode, sum(losses.values()), losses)
                print(f"Episode {episode}: Total Loss = {sum(losses.values()):.4f}")
        
        # Final evaluation
        final_losses = learner.evaluate()
        total_loss = sum(final_losses.values())
        
        results.append({
            'seed': seed,
            'method': 'random_lookahead',
            'episodes': episodes,
            'K': K,
            'total_loss': total_loss,
            **{f'loss_{k}': v for k, v in final_losses.items()}
        })
        
        tracker.save(f"{output_dir}/random_lookahead_seed{seed}_curve.csv")
        
        # CRITICAL: Save all results after each seed
        df_all = pd.DataFrame(results)
        df_all.to_csv(f"{output_dir}/lookahead_ablation_summary.csv", index=False)
        print(f"\n  ✓ Seed {seed} complete - saved to summary CSV")
    
    # Save final summary
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/lookahead_ablation_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("LOOKAHEAD ABLATION SUMMARY (Random Proposer, K=4)")
    print("="*60)
    mean_loss = df['total_loss'].mean()
    std_loss = df['total_loss'].std()
    print(f"Random Lookahead: {mean_loss:.3f} +/- {std_loss:.3f}")
    
    return df


# ============================================================================
# 15-NODE COMPLEX SCM
# ============================================================================

def run_complex_scm_experiments(
    seeds: List[int],
    episodes: int = 200,
    output_dir: str = "results/complex_scm_full"
):
    """Run all methods on 15-node complex SCM."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"COMPLEX 15-NODE SCM - Seed {seed}")
        print(f"{'='*60}")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        scm = ComplexGroundTruthSCM()
        
        # PPO baseline (PRIORITY - run first)
        print("\nRunning PPO...")
        learner = create_complex_learner(scm)
        final_losses = run_complex_ppo(scm, learner, episodes)
        results.append({
            'seed': seed, 'method': 'ppo', 'episodes': episodes,
            'total_loss': sum(final_losses.values()),
            'collider_loss': get_collider_loss(final_losses, scm)
        })
        
        # CRITICAL: Save immediately after PPO
        df_ppo = pd.DataFrame(results)
        df_ppo.to_csv(f"{output_dir}/complex_scm_summary.csv", index=False)
        df_ppo.to_csv(f"{output_dir}/complex_scm_seed{seed}_ppo.csv", index=False)
        print(f"  ✓ PPO complete - SAVED")
        
        # Random baseline
        print("\nRunning Random...")
        learner = create_complex_learner(scm)
        final_losses = run_complex_random(scm, learner, episodes)
        results.append({
            'seed': seed, 'method': 'random', 'episodes': episodes,
            'total_loss': sum(final_losses.values()),
            'collider_loss': get_collider_loss(final_losses, scm)
        })
        
        # Round-robin baseline
        print("\nRunning Round-Robin...")
        learner = create_complex_learner(scm)
        final_losses = run_complex_round_robin(scm, learner, episodes)
        results.append({
            'seed': seed, 'method': 'round_robin', 'episodes': episodes,
            'total_loss': sum(final_losses.values()),
            'collider_loss': get_collider_loss(final_losses, scm)
        })
        
        # Greedy collider-focused
        print("\nRunning Greedy Collider...")
        learner = create_complex_learner(scm)
        final_losses = run_complex_greedy_collider(scm, learner, episodes)
        results.append({
            'seed': seed, 'method': 'greedy_collider', 'episodes': episodes,
            'total_loss': sum(final_losses.values()),
            'collider_loss': get_collider_loss(final_losses, scm)
        })
        
        # Random lookahead (ablation)
        print("\nRunning Random Lookahead...")
        learner = create_complex_learner(scm)
        final_losses = run_complex_random_lookahead(scm, learner, episodes, K=4)
        results.append({
            'seed': seed, 'method': 'random_lookahead', 'episodes': episodes,
            'total_loss': sum(final_losses.values()),
            'collider_loss': get_collider_loss(final_losses, scm)
        })
        
        # CRITICAL: Save all results after EVERY seed completion
        df_all = pd.DataFrame(results)
        df_all.to_csv(f"{output_dir}/complex_scm_summary.csv", index=False)
        
        # Also save seed-specific results
        seed_results = [r for r in results if r['seed'] == seed]
        df_seed = pd.DataFrame(seed_results)
        df_seed.to_csv(f"{output_dir}/complex_scm_seed{seed}_results.csv", index=False)
        
        print(f"\n  ✓ Seed {seed} complete - saved {len(seed_results)} method results")
        print(f"  Total progress: {len(results)}/{len(seeds)*5} runs (5 methods per seed)")
    
    # Save final summary
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/complex_scm_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("COMPLEX 15-NODE SCM SUMMARY")
    print("="*60)
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        mean_total = subset['total_loss'].mean()
        std_total = subset['total_loss'].std()
        mean_collider = subset['collider_loss'].mean()
        print(f"{method:20s}: Total={mean_total:.3f}+/-{std_total:.3f}, Collider={mean_collider:.3f}")
    
    return df


def create_complex_learner(scm):
    """Create learner for complex SCM."""
    from experiments.complex_scm import ComplexSCMLearner, ComplexStudentSCM
    student = ComplexStudentSCM(scm)
    return ComplexSCMLearner(student, oracle=scm)


def get_collider_loss(losses: Dict[str, float], scm) -> float:
    """Get average loss on collider nodes."""
    colliders = ['L1', 'N1', 'C1', 'C2', 'F3']  # Nodes with multiple parents
    collider_losses = [losses.get(c, 0) for c in colliders if c in losses]
    return np.mean(collider_losses) if collider_losses else 0.0


def run_complex_random(scm, learner, episodes):
    """Random policy for complex SCM."""
    for ep in range(episodes):
        node = random.choice(scm.nodes)
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data)
    return learner.evaluate()


def run_complex_round_robin(scm, learner, episodes):
    """Round-robin policy for complex SCM."""
    for ep in range(episodes):
        node = scm.nodes[ep % len(scm.nodes)]
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data)
    return learner.evaluate()


def run_complex_greedy_collider(scm, learner, episodes):
    """Greedy strategy focusing on collider parents."""
    collider_parents = ['R1', 'R2', 'L1', 'L2', 'N1', 'N2', 'N3', 'C1', 'C2']
    for ep in range(episodes):
        # 80% focus on collider parents
        if random.random() < 0.8:
            node = random.choice(collider_parents)
        else:
            node = random.choice(scm.nodes)
        value = random.uniform(-5, 5)
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data)
    return learner.evaluate()


def run_complex_random_lookahead(scm, learner, episodes, K=4):
    """Random lookahead for complex SCM."""
    for ep in range(episodes):
        # Generate K random candidates
        candidates = [(random.choice(scm.nodes), random.uniform(-5, 5)) for _ in range(K)]
        
        # Evaluate each
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
        
        # Execute best
        node, value = best_candidate
        data = scm.generate(100, interventions={node: value})
        learner.train_step(data)
    
    return learner.evaluate()


def run_complex_ppo(scm, learner, episodes):
    """
    PPO policy for complex SCM.
    Uses value-based RL with same reward as ACE (information gain + bonuses).
    Simplified implementation without full actor-critic for speed.
    """
    from baselines import PPOPolicy
    
    # Create PPO policy for complex SCM
    # PPOPolicy constructor: (nodes, value_min, value_max, lr, n_value_bins)
    policy = PPOPolicy(
        nodes=scm.nodes,
        value_min=-5.0,
        value_max=5.0
    )
    
    for ep in range(episodes):
        # Get current node losses for state
        node_losses = learner.evaluate()
        
        # Select intervention via PPO (expects student and oracle)
        # PPO's select_intervention needs StudentSCM
        target, value = policy.select_intervention(learner.student, oracle=scm, node_losses=node_losses)
        
        # Execute intervention
        losses_before = learner.evaluate()
        data = scm.generate(100, interventions={target: value})
        learner.train_step(data)
        losses_after = learner.evaluate()
        
        # Compute reward (simple information gain for complex SCM)
        reward = sum(losses_before.values()) - sum(losses_after.values())
        
        # Store for PPO update
        done = (ep == episodes - 1)
        policy.store_reward(reward, done)
        
        # Update policy every episode
        policy.update()
    
    return learner.evaluate()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run critical experiments for ICML response")
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--extended-baselines', action='store_true', 
                        help='Run baselines for 171 episodes')
    parser.add_argument('--lookahead-ablation', action='store_true',
                        help='Run lookahead with random proposer')
    parser.add_argument('--complex-scm', action='store_true',
                        help='Run 15-node complex SCM experiments')
    parser.add_argument('--seeds', nargs='+', type=int, 
                        default=[42, 123, 456, 789, 1011],
                        help='Random seeds to use')
    parser.add_argument('--output-dir', type=str, default='results/critical_experiments',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"{args.output_dir}_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CRITICAL EXPERIMENTS FOR ICML REVIEWER RESPONSE")
    print(f"{'='*70}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {base_output}")
    print(f"{'='*70}\n")
    
    results = {}
    
    if args.all or args.extended_baselines:
        print("\n[1/3] EXTENDED BASELINES (171 episodes)")
        results['extended_baselines'] = run_extended_baselines(
            args.seeds, 
            episodes=171,
            output_dir=f"{base_output}/extended_baselines"
        )
    
    if args.all or args.lookahead_ablation:
        print("\n[2/3] LOOKAHEAD ABLATION (Random Proposer)")
        results['lookahead_ablation'] = run_lookahead_ablation(
            args.seeds,
            episodes=171,
            K=4,
            output_dir=f"{base_output}/lookahead_ablation"
        )
    
    if args.all or args.complex_scm:
        print("\n[3/3] COMPLEX 15-NODE SCM")
        results['complex_scm'] = run_complex_scm_experiments(
            args.seeds,
            episodes=200,
            output_dir=f"{base_output}/complex_scm"
        )
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {base_output}")
    
    # Save combined summary
    with open(f"{base_output}/experiment_config.json", 'w') as f:
        json.dump({
            'seeds': args.seeds,
            'timestamp': timestamp,
            'experiments_run': list(results.keys())
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()
