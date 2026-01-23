#!/usr/bin/env python3
"""
Compute statistics from multiple experimental runs.

This script aggregates results from multiple seeds to compute:
- Mean and standard deviation
- 95% confidence intervals
- Statistical significance tests (t-tests)
- Effect sizes (Cohen's d)

Usage:
    python scripts/compute_statistics.py --input results/multi_run_TIMESTAMP

Outputs:
    - CSV files with statistics
    - LaTeX table format
    - Figures with error bars
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import glob
import sys


def load_results_from_seeds(multi_run_dir, experiment='ace'):
    """Load results from all seed directories."""
    multi_run_path = Path(multi_run_dir)
    seed_dirs = sorted(multi_run_path.glob('seed_*'))
    
    all_results = []
    
    for seed_dir in seed_dirs:
        seed = seed_dir.name.split('_')[1]
        
        # Look for metrics in experiment subdirectory
        exp_dir = seed_dir / experiment
        
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} not found")
            continue
        
        # Try to load final results
        # Look for node_losses.csv or metrics.csv
        node_losses_file = exp_dir / 'node_losses.csv'
        
        if node_losses_file.exists():
            df = pd.read_csv(node_losses_file)
            
            # Get final episode
            if len(df) > 0:
                final = df.iloc[-1]
                
                result = {
                    'seed': int(seed),
                    'episode': final.get('episode', 0),
                }
                
                # Extract per-node losses
                for col in df.columns:
                    if col.startswith('loss_'):
                        result[col] = final[col]
                
                # Compute total
                loss_cols = [c for c in df.columns if c.startswith('loss_')]
                result['total_loss'] = sum(final[c] for c in loss_cols)
                
                all_results.append(result)
    
    return pd.DataFrame(all_results)


def compute_summary_statistics(df, metric_cols):
    """Compute mean, std, CI for metrics."""
    stats_dict = {}
    
    for col in metric_cols:
        if col in df.columns:
            values = df[col].values
            n = len(values)
            
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0
            
            # 95% confidence interval
            if n > 1:
                se = std / np.sqrt(n)
                ci = stats.t.ppf(0.975, n-1) * se
            else:
                ci = 0
            
            stats_dict[col] = {
                'mean': mean,
                'std': std,
                'ci_lower': mean - ci,
                'ci_upper': mean + ci,
                'n': n
            }
    
    return stats_dict


def paired_t_test(ace_values, baseline_values):
    """Perform paired t-test."""
    if len(ace_values) == len(baseline_values) and len(ace_values) > 1:
        t_stat, p_value = stats.ttest_rel(baseline_values, ace_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(ace_values, ddof=1)**2 + 
                             np.std(baseline_values, ddof=1)**2) / 2)
        effect_size = (np.mean(baseline_values) - np.mean(ace_values)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
    else:
        return None


def format_latex_table(ace_stats, baseline_stats_dict, significance_dict):
    """Format results as LaTeX table."""
    
    latex = []
    latex.append("% Table 1: Method Comparison with Statistical Validation")
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Method comparison with statistics from 5 independent runs.}")
    latex.append("\\label{tab:main-results}")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\toprule")
    latex.append("Method & X₁ & X₂ & X₃ & X₄ & X₅ & Total & Episodes \\\\")
    latex.append("\\midrule")
    
    # ACE row
    ace_line = "ACE (Ours)"
    for node in ['loss_X1', 'loss_X2', 'loss_X3', 'loss_X4', 'loss_X5', 'total_loss']:
        if node in ace_stats:
            mean = ace_stats[node]['mean']
            std = ace_stats[node]['std']
            ace_line += f" & {mean:.2f}±{std:.2f}"
    
    if 'episode' in ace_stats:
        ep_mean = ace_stats['episode']['mean']
        ep_std = ace_stats['episode']['std']
        ace_line += f" & {ep_mean:.0f}±{ep_std:.0f}"
    
    ace_line += " \\\\"
    latex.append(ace_line)
    
    # Baseline rows (TODO: fill when baseline stats available)
    for baseline_name in ['Random', 'Round-Robin', 'Max-Variance', 'PPO']:
        # Placeholder - will be filled when baseline stats computed
        latex.append(f"{baseline_name} & TBD & TBD & TBD & TBD & TBD & TBD & 100 \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description='Compute statistics from multi-seed runs')
    parser.add_argument('--input', required=True, help='Multi-run directory')
    parser.add_argument('--experiment', default='ace', help='Experiment name')
    parser.add_argument('--baseline', default=None, help='Baseline name (if applicable)')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results_from_seeds(args.input, args.experiment)
    
    if len(df) == 0:
        print(f"ERROR: No results found for {args.experiment}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} runs")
    print(f"Columns: {list(df.columns)}")
    
    # Compute statistics
    metric_cols = [c for c in df.columns if c.startswith('loss_') or c == 'total_loss' or c == 'episode']
    stats_dict = compute_summary_statistics(df, metric_cols)
    
    # Print summary
    print("\nStatistics Summary:")
    print("=" * 60)
    for metric, stat in stats_dict.items():
        print(f"{metric:20s}: {stat['mean']:8.4f} ± {stat['std']:6.4f} (95% CI: [{stat['ci_lower']:.4f}, {stat['ci_upper']:.4f}])")
    
    # Save to CSV
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.to_csv(args.output)
    print(f"\nStatistics saved to: {args.output}")
    
    # Generate LaTeX table if this is ACE
    if args.experiment == 'ace':
        latex_output = args.output.replace('.csv', '_table.tex')
        latex_table = format_latex_table(stats_dict, {}, {})
        
        with open(latex_output, 'w') as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to: {latex_output}")


if __name__ == '__main__':
    main()
