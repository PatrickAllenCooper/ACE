#!/usr/bin/env python3
"""
Analyze and summarize ACE experiment results.
Computes statistics across multiple seeds and generates comparison tables.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path


def load_run_results(results_dir, metric='reward'):
    """Load results from all runs in a directory."""
    pattern = os.path.join(results_dir, "**/metrics.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"Warning: No metrics.csv files found in {results_dir}")
        return []
    
    results = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            # Get final episode
            final_episode = df['episode'].max()
            final_data = df[df['episode'] == final_episode]
            
            # Compute mean of metric over final episode
            mean_value = final_data[metric].mean()
            results.append({
                'file': filepath,
                'final_value': mean_value,
                'num_episodes': int(final_episode) + 1
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return results


def compute_statistics(results):
    """Compute mean and std from results list."""
    if not results:
        return None, None, 0
    
    values = [r['final_value'] for r in results]
    return np.mean(values), np.std(values), len(values)


def compare_methods(base_dir):
    """Compare ACE against baselines."""
    print("=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)
    print()
    
    methods = {
        'ACE': 'ace',
        'Random': 'baselines/random',
        'Greedy': 'baselines/greedy_collider',
        'Round-Robin': 'baselines/round_robin',
        'Max-Variance': 'baselines/max_variance',
        'PPO': 'baselines/ppo'
    }
    
    results_table = []
    
    for name, path in methods.items():
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
            continue
        
        results = load_run_results(full_path, metric='reward')
        mean, std, n = compute_statistics(results)
        
        if mean is not None:
            results_table.append({
                'Method': name,
                'Mean Reward': mean,
                'Std': std,
                'N': n
            })
            print(f"{name:15} {mean:8.4f} ± {std:.4f}  (N={n})")
    
    print()
    return pd.DataFrame(results_table)


def analyze_ablations(ablations_dir):
    """Analyze ablation study results."""
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print()
    
    ablations = {
        'Full ACE': 'full_ace',
        'No Diversity': 'no_diversity',
        'No Root Learner': 'no_root_learner',
        'No Convergence': 'no_convergence'
    }
    
    results_table = []
    
    for name, path in ablations.items():
        full_path = os.path.join(ablations_dir, path)
        if not os.path.exists(full_path):
            continue
        
        results = load_run_results(full_path, metric='reward')
        mean, std, n = compute_statistics(results)
        
        if mean is not None:
            results_table.append({
                'Configuration': name,
                'Mean Reward': mean,
                'Std': std,
                'N': n
            })
            print(f"{name:20} {mean:8.4f} ± {std:.4f}  (N={n})")
    
    print()
    return pd.DataFrame(results_table)


def analyze_convergence(results_dir):
    """Analyze convergence behavior."""
    pattern = os.path.join(results_dir, "**/metrics.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("No results found for convergence analysis")
        return
    
    print("=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    print()
    
    for filepath in files[:3]:  # Show first 3 runs
        df = pd.read_csv(filepath)
        
        run_name = Path(filepath).parent.name
        final_episode = df['episode'].max()
        
        # Compute average reward by episode
        episode_rewards = df.groupby('episode')['reward'].mean()
        
        # Check convergence
        recent_10 = episode_rewards.iloc[-10:] if len(episode_rewards) >= 10 else episode_rewards
        convergence_std = recent_10.std()
        
        print(f"Run: {run_name}")
        print(f"  Episodes: {int(final_episode) + 1}")
        print(f"  Final reward: {episode_rewards.iloc[-1]:.4f}")
        print(f"  Recent std (last 10): {convergence_std:.4f}")
        print()


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print()
        print("Examples:")
        print("  python analyze_results.py results/ace_multi_seed")
        print("  python analyze_results.py results/ablations")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    print()
    print("ACE Results Analysis")
    print(f"Directory: {results_dir}")
    print()
    
    # Detect analysis type based on directory structure
    if 'ablation' in results_dir.lower():
        df = analyze_ablations(results_dir)
    elif any(os.path.exists(os.path.join(results_dir, m)) for m in ['baselines', 'ace']):
        df = compare_methods(results_dir)
    else:
        # Generic analysis
        results = load_run_results(results_dir)
        mean, std, n = compute_statistics(results)
        
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()
        print(f"Runs found: {n}")
        print(f"Mean final reward: {mean:.4f} ± {std:.4f}")
        print()
        
        analyze_convergence(results_dir)
    
    # Save summary if DataFrame created
    if 'df' in locals() and df is not None:
        output_path = os.path.join(results_dir, "analysis_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
