#!/usr/bin/env python3
"""
Baseline comparisons for Duffing oscillators.
Runs Random, Round-Robin, Max-Variance for fair comparison to ACE.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import random
import torch
import numpy as np
import pandas as pd
from experiments.duffing_oscillators import DuffingOracle, DuffingLearner


def run_random_policy_duffing(oracle, learner, episodes=100, steps=20):
    """Random intervention selection for Duffing."""
    results = []
    for episode in range(episodes):
        for step in range(steps):
            # Random selection
            target = random.choice(['X1', 'X2', 'X3'])
            value = random.uniform(-2, 2)
            
            # Apply and train
            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data)
            
            # Evaluate
            loss = learner.evaluate(oracle)
            results.append({'episode': episode, 'step': step, 'target': target, 'loss': loss})
    
    return pd.DataFrame(results)


def run_round_robin_duffing(oracle, learner, episodes=100, steps=20):
    """Round-robin intervention for Duffing."""
    nodes = ['X1', 'X2', 'X3']
    results = []
    
    for episode in range(episodes):
        for step in range(steps):
            # Round-robin selection
            target = nodes[step % len(nodes)]
            value = random.uniform(-2, 2)
            
            # Apply and train
            data = oracle.generate(n_samples=50, interventions={target: value})
            learner.train_step(data)
            
            # Evaluate
            loss = learner.evaluate(oracle)
            results.append({'episode': episode, 'step': step, 'target': target, 'loss': loss})
    
    return pd.DataFrame(results)


def run_max_variance_duffing(oracle, learner, episodes=100, steps=20):
    """Max-variance (uncertainty sampling) for Duffing."""
    nodes = ['X1', 'X2', 'X3']
    results = []
    
    for episode in range(episodes):
        for step in range(steps):
            # Select node with highest prediction variance
            max_var = -1
            best_node = nodes[0]
            
            learner.model.eval()
            with torch.no_grad():
                test_data = oracle.generate(n_samples=10)
                for node in nodes:
                    preds = learner.model(test_data)
                    var = preds.var().item() if hasattr(preds, 'var') else 0.1
                    if var > max_var:
                        max_var = var
                        best_node = node
            
            value = random.uniform(-2, 2)
            
            # Apply and train
            data = oracle.generate(n_samples=50, interventions={best_node: value})
            learner.train_step(data)
            
            # Evaluate
            loss = learner.evaluate(oracle)
            results.append({'episode': episode, 'step': step, 'target': best_node, 'loss': loss})
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Duffing baselines")
    parser.add_argument('--all', action='store_true', help='Run all baselines')
    parser.add_argument('--policy', type=str, choices=['random', 'round_robin', 'max_variance'],
                        help='Baseline policy to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 1011])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='results/duffing_baselines')
    
    args = parser.parse_args()
    
    # Determine policies to run
    if args.all:
        policies = ['random', 'round_robin', 'max_variance']
    elif args.policy:
        policies = [args.policy]
    else:
        print("Error: Specify --all or --policy")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    
    for policy_name in policies:
        policy_fn = {
            'random': run_random_policy_duffing,
            'round_robin': run_round_robin_duffing,
            'max_variance': run_max_variance_duffing
        }[policy_name]
        
        for seed in args.seeds:
            print(f"\nRunning {policy_name} seed {seed}...")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            oracle = DuffingOracle()
            learner = DuffingLearner(oracle)
            
            df = policy_fn(oracle, learner, episodes=args.episodes)
            
            # Save individual run
            output_file = f"{args.output_dir}/{policy_name}_seed{seed}_results.csv"
            df.to_csv(output_file, index=False)
            
            # Aggregate
            final_loss = df['loss'].iloc[-1]
            all_results.append({
                'method': policy_name,
                'seed': seed,
                'final_loss': final_loss
            })
            
            print(f"  Final loss: {final_loss:.4f}")
    
    # Save summary
    summary = pd.DataFrame(all_results)
    summary.to_csv(f"{args.output_dir}/duffing_baselines_summary.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}/duffing_baselines_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
