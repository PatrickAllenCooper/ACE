#!/usr/bin/env python3
"""
Analyze ablation study results.

Compares ACE with and without each component to quantify contribution.

Usage:
    python scripts/analyze_ablations.py results/ablations_TIMESTAMP
    python scripts/analyze_ablations.py results/ablations_TIMESTAMP --latex
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_ablation_results(ablation_dir, ace_dir):
    """Load results from ablation and baseline ACE."""
    ablation_path = Path(ablation_dir)
    ace_path = Path(ace_dir)
    
    # Load ACE baseline (full method)
    ace_results = []
    for seed_dir in sorted(ace_path.glob("seed_*")):
        run_dirs = list(seed_dir.glob("run_*"))
        if not run_dirs:
            continue
        
        node_losses = run_dirs[0] / "node_losses.csv"
        if node_losses.exists():
            df = pd.read_csv(node_losses)
            if len(df) > 0:
                final = df.iloc[-1]
                ace_results.append({
                    'seed': seed_dir.name.split('_')[1],
                    'total_loss': final['total_loss'],
                    'episodes': int(final['episode'])
                })
    
    # Load ablation results
    ablation_results = {}
    for ablation_name in ['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity']:
        ablation_subdir = ablation_path / ablation_name
        if not ablation_subdir.exists():
            continue
        
        results = []
        for seed_dir in sorted(ablation_subdir.glob("seed_*")):
            node_losses = seed_dir / "node_losses.csv"
            if node_losses.exists():
                df = pd.read_csv(node_losses)
                if len(df) > 0:
                    final = df.iloc[-1]
                    results.append({
                        'seed': seed_dir.name.split('_')[1],
                        'total_loss': final['total_loss'],
                        'episodes': int(final['episode'])
                    })
        
        if results:
            ablation_results[ablation_name] = pd.DataFrame(results)
    
    return pd.DataFrame(ace_results), ablation_results


def compute_stats(df, metric='total_loss'):
    """Compute mean, std, CI for metric."""
    values = df[metric].values
    n = len(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    
    if n > 1:
        se = std / np.sqrt(n)
        ci = stats.t.ppf(0.975, n-1) * se
    else:
        ci = 0
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': mean - ci,
        'ci_upper': mean + ci,
        'n': n
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation studies')
    parser.add_argument('ablation_dir', help='Directory with ablation results')
    parser.add_argument('--ace', default='results/ace_multi_seed_20260125_115453',
                        help='ACE baseline results directory')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX table')
    parser.add_argument('--output', help='Output file for table')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    ace_df, ablation_dfs = load_ablation_results(args.ablation_dir, args.ace)
    
    if len(ace_df) == 0:
        print("ERROR: No ACE baseline results found")
        return
    
    # Compute statistics
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    ace_stats = compute_stats(ace_df)
    print(f"\nACE (Full Method):")
    print(f"  Mean: {ace_stats['mean']:.4f}")
    print(f"  Std:  {ace_stats['std']:.4f}")
    print(f"  95% CI: [{ace_stats['ci_lower']:.4f}, {ace_stats['ci_upper']:.4f}]")
    print(f"  N: {ace_stats['n']}")
    
    # Component contributions
    print("\n" + "="*60)
    print("COMPONENT CONTRIBUTIONS")
    print("="*60)
    
    results_table = []
    
    component_names = {
        'no_dpo': 'DPO Training',
        'no_convergence': 'Per-Node Convergence',
        'no_root_learner': 'Dedicated Root Learner',
        'no_diversity': 'Diversity Reward'
    }
    
    for ablation_name, component_name in component_names.items():
        if ablation_name not in ablation_dfs:
            print(f"\nWARNING: {ablation_name} results not found")
            continue
        
        ablation_df = ablation_dfs[ablation_name]
        ablation_stats = compute_stats(ablation_df)
        
        # Compute degradation
        degradation = ablation_stats['mean'] - ace_stats['mean']
        degradation_pct = (degradation / ace_stats['mean']) * 100
        
        # Statistical test
        if len(ace_df) > 1 and len(ablation_df) > 1:
            t_stat, p_value = stats.ttest_ind(ace_df['total_loss'], 
                                               ablation_df['total_loss'])
        else:
            p_value = np.nan
        
        print(f"\n{component_name} (without):")
        print(f"  Mean: {ablation_stats['mean']:.4f}")
        print(f"  Std:  {ablation_stats['std']:.4f}")
        print(f"  Degradation: +{degradation:.4f} ({degradation_pct:+.1f}%)")
        if not np.isnan(p_value):
            print(f"  p-value: {p_value:.4f}")
        
        results_table.append({
            'Component': component_name,
            'ACE Mean': ace_stats['mean'],
            'Without Mean': ablation_stats['mean'],
            'Degradation': degradation,
            'Degradation %': degradation_pct,
            'p-value': p_value
        })
    
    # Create summary table
    results_df = pd.DataFrame(results_table)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # LaTeX output
    if args.latex:
        print("\n" + "="*60)
        print("LATEX TABLE")
        print("="*60)
        
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Ablation study results. Each row shows performance degradation when component is removed.}\n"
        latex += "\\label{tab:ablations}\n"
        latex += "\\begin{tabular}{lccc}\n"
        latex += "\\toprule\n"
        latex += "Component Removed & ACE (Full) & Without & Degradation \\\\\n"
        latex += "\\midrule\n"
        
        for _, row in results_df.iterrows():
            sig = "$^{***}$" if row['p-value'] < 0.001 else ("$^{**}$" if row['p-value'] < 0.01 else ("$^{*}$" if row['p-value'] < 0.05 else ""))
            latex += f"{row['Component']} & {row['ACE Mean']:.2f} & {row['Without Mean']:.2f} & +{row['Degradation']:.2f} ({row['Degradation %']:+.0f}\\%){sig} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        print(latex)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(latex)
            print(f"\nLaTeX table saved to: {args.output}")
    
    # Save summary
    summary_file = Path(args.ablation_dir) / "ablation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"ACE (Full): {ace_stats['mean']:.4f} ± {ace_stats['std']:.4f}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n")
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
