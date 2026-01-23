#!/usr/bin/env python3
"""
Comparison experiment: ACE vs Recent Methods (CORE, GACBO).

Compares ACE against:
1. CORE-inspired (Deep RL, 2024)
2. GACBO-inspired (Causal Bayesian Optimization, 2024)
3. Greedy Information Maximization

This addresses reviewer concern: "Why not compare to recent methods?"
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import GroundTruthSCM, run_baseline
from baselines_recent_methods import COREPolicy, GAGBOPolicy, GreedyInfoMaxPolicy


def main():
    parser = argparse.ArgumentParser(description='Compare ACE to recent methods')
    parser.add_argument('--episodes', type=int, default=50, 
                       help='Episodes per method')
    parser.add_argument('--steps', type=int, default=25,
                       help='Steps per episode')
    parser.add_argument('--output', type=str, default='results/recent_methods_comparison',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        import torch, random, numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ACE vs Recent Methods Comparison")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Episodes: {args.episodes}")
    print("")
    
    # Create oracle
    oracle = GroundTruthSCM()
    
    # Methods to compare
    methods = {
        'CORE-Inspired': COREPolicy(oracle.nodes),
        'GACBO-Inspired': GAGBOPolicy(oracle.nodes),
        'Greedy-InfoMax': GreedyInfoMaxPolicy(oracle.nodes)
    }
    
    results = {}
    
    # Run each method
    for name, policy in methods.items():
        print(f"\nRunning {name}...")
        
        df = run_baseline(
            policy=policy,
            oracle=oracle,
            n_episodes=args.episodes,
            steps_per_episode=args.steps
        )
        
        results[name] = df
        
        # Save individual results
        output_file = output_dir / f"{name.lower().replace(' ', '_')}_results.csv"
        df.to_csv(output_file, index=False)
        
        # Print summary
        final_loss = df['total_loss'].iloc[-1]
        print(f"  Final loss: {final_loss:.4f}")
    
    # Create comparison summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, df in results.items():
        final_loss = df['total_loss'].iloc[-1]
        episodes_run = df['episode'].max() + 1
        print(f"\n{name}:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Episodes: {episodes_run}")
    
    # Save summary
    summary_file = output_dir / "COMPARISON_SUMMARY.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Recent Methods Comparison Summary\n")
        f.write("=" * 70 + "\n\n")
        
        for name, df in results.items():
            final_loss = df['total_loss'].iloc[-1]
            f.write(f"{name}:\n")
            f.write(f"  Final Loss: {final_loss:.4f}\n")
            f.write(f"  Episodes: {args.episodes}\n\n")
        
        f.write("\nNote: Compare to ACE results from main experiments.\n")
        f.write("Expected: ACE should be competitive with these methods,\n")
        f.write("potentially showing advantages on collider identification.\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    
    print("\nNext: Compare to ACE results and add to paper Discussion")


if __name__ == '__main__':
    main()
