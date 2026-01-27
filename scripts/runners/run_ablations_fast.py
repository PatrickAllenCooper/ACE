#!/usr/bin/env python3
"""
Fast Ablation Studies for ICML Reviewer Response

Streamlined ablations with early stopping for quick turnaround.
Each ablation runs to convergence (typically 40-80 episodes) instead of fixed 200.

Ablations:
1. no_dpo: Use custom transformer instead of DPO-trained LLM
2. no_convergence: Disable per-node convergence (run fixed 100 episodes)
3. no_root_learner: Disable dedicated root learner
4. no_diversity: Disable diversity reward

Usage:
    python scripts/runners/run_ablations_fast.py --all --seeds 42 123 456
    python scripts/runners/run_ablations_fast.py --ablation no_dpo --seeds 42
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from typing import List
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_ablation(ablation: str, seed: int, output_dir: str, max_episodes: int = 100):
    """Run a single ablation experiment.
    
    Args:
        ablation: One of 'no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity'
        seed: Random seed
        output_dir: Base output directory
        max_episodes: Maximum episodes (default 100 for speed)
    """
    # Map ablation to flags
    ablation_flags = {
        'no_dpo': ['--custom'],
        'no_convergence': ['--no_per_node_convergence'],
        'no_root_learner': ['--no_dedicated_root_learner'],
        'no_diversity': ['--no_diversity_reward'],
    }
    
    if ablation not in ablation_flags:
        raise ValueError(f"Unknown ablation: {ablation}. Valid: {list(ablation_flags.keys())}")
    
    # Build output path
    run_output = os.path.join(output_dir, ablation, f"seed_{seed}")
    os.makedirs(run_output, exist_ok=True)
    
    # Base command with standard settings
    cmd = [
        sys.executable,
        "ace_experiments.py",
        "--episodes", str(max_episodes),
        "--steps", "25",
        "--seed", str(seed),
        "--output", run_output,
        # Early stopping for efficiency
        "--early_stopping",
        "--early_stop_patience", "15",  # Reduced for faster completion
        "--early_stop_min_episodes", "30",  # Minimum 30 episodes
        "--use_per_node_convergence",  # Will be overridden by no_convergence
        # Observational training
        "--obs_train_interval", "3",
        "--obs_train_samples", "200",
        "--obs_train_epochs", "100",
        # Root learner
        "--root_fitting",
        "--use_dedicated_root_learner",
        "--dedicated_root_interval", "3",
        # Diversity
        "--undersampled_bonus", "200.0",
        "--diversity_reward_weight", "0.3",
        "--max_concentration", "0.7",
        # Pretraining
        "--pretrain_steps", "200",
        "--pretrain_interval", "25",
        "--smart_breaker",
    ]
    
    # Add ablation-specific flags
    cmd.extend(ablation_flags[ablation])
    
    print(f"\n{'='*70}")
    print(f"Running: {ablation} (seed={seed})")
    print(f"Max episodes: {max_episodes}")
    print(f"Output: {run_output}")
    print(f"Flags: {' '.join(ablation_flags[ablation])}")
    print(f"{'='*70}\n")
    
    # Run experiment
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logging.error(f"Ablation {ablation} seed {seed} failed with code {result.returncode}")
        return False
    
    # Verify output files exist
    expected_files = ['metrics.csv', 'node_losses.csv']
    run_dirs = [d for d in os.listdir(run_output) if d.startswith('run_')]
    
    if run_dirs:
        actual_run_dir = os.path.join(run_output, run_dirs[0])
        for fname in expected_files:
            fpath = os.path.join(actual_run_dir, fname)
            if not os.path.exists(fpath):
                logging.warning(f"Expected file not found: {fpath}")
    
    print(f"✓ Completed: {ablation} seed {seed}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run fast ablation studies")
    parser.add_argument('--all', action='store_true', help='Run all ablations')
    parser.add_argument('--ablation', type=str, choices=['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity'],
                        help='Run specific ablation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds to use')
    parser.add_argument('--max-episodes', type=int, default=100,
                        help='Maximum episodes per ablation (default: 100)')
    parser.add_argument('--output-dir', type=str, default='results/ablations_fast',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"{args.output_dir}_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FAST ABLATION STUDIES")
    print(f"{'='*70}")
    print(f"Seeds: {args.seeds}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Output: {base_output}")
    print(f"{'='*70}\n")
    
    # Determine which ablations to run
    if args.all:
        ablations = ['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity']
    elif args.ablation:
        ablations = [args.ablation]
    else:
        print("Error: Specify --all or --ablation <type>")
        return 1
    
    # Run ablations
    results = {}
    for ablation in ablations:
        results[ablation] = {}
        for seed in args.seeds:
            success = run_ablation(ablation, seed, base_output, args.max_episodes)
            results[ablation][seed] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    
    total_runs = len(ablations) * len(args.seeds)
    successful = sum(sum(1 for s in seeds.values() if s) for seeds in results.values())
    
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_runs - successful}")
    
    for ablation, seeds in results.items():
        successes = sum(1 for s in seeds.values() if s)
        print(f"  {ablation}: {successes}/{len(seeds)} seeds completed")
    
    print(f"\nResults saved to: {base_output}")
    print(f"{'='*70}\n")
    
    return 0 if successful == total_runs else 1


if __name__ == "__main__":
    sys.exit(main())
