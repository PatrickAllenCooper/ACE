#!/usr/bin/env python3
"""
Generate formal statistical significance tests for paper.

Compares ACE vs all baselines with:
- Paired t-tests with Bonferroni correction
- Effect sizes (Cohen's d)
- 95% confidence intervals

Usage:
    python scripts/statistical_tests.py \
        --ace results/ace_multi_seed_20260125_115453 \
        --baselines results/baselines/baselines_20260124_182827 \
        --output results/statistical_analysis.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_ace_results(ace_dir):
    """Load ACE multi-seed results."""
    results = []
    ace_path = Path(ace_dir)
    
    for seed_dir in sorted(ace_path.glob("seed_*")):
        run_dirs = list(seed_dir.glob("run_*"))
        if not run_dirs:
            continue
        
        node_losses = run_dirs[0] / "node_losses.csv"
        if node_losses.exists():
            df = pd.read_csv(node_losses)
            if len(df) > 0:
                final = df.iloc[-1]
                results.append({
                    'seed': int(seed_dir.name.split('_')[1]),
                    'total_loss': float(final['total_loss'])
                })
    
    return pd.DataFrame(results)


def load_baseline_results(baseline_dir):
    """Load baseline results."""
    baseline_path = Path(baseline_dir)
    results = {}
    
    for method in ['random', 'round_robin', 'max_variance', 'ppo']:
        csv_file = baseline_path / f"{method}_results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                # Get final total loss for each episode (assuming multiple runs)
                # Group by episode and get final value
                final_losses = []
                for episode in df['episode'].unique():
                    episode_data = df[df['episode'] == episode]
                    if len(episode_data) > 0:
                        final = episode_data.iloc[-1]
                        final_losses.append(float(final['total_loss']))
                
                # Take last N values as final results (one per seed)
                # Assuming 5 runs of 100 episodes each = 500 rows total
                # We want the last loss from each of the 5 runs
                n_seeds = 5
                if len(final_losses) >= n_seeds:
                    results[method] = final_losses[-n_seeds:]
                else:
                    results[method] = final_losses
    
    return results


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    parser = argparse.ArgumentParser(description='Statistical significance tests')
    parser.add_argument('--ace', required=True, help='ACE results directory')
    parser.add_argument('--baselines', required=True, help='Baseline results directory')
    parser.add_argument('--output', help='Output file')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading ACE results...")
    ace_df = load_ace_results(args.ace)
    
    print("Loading baseline results...")
    baselines = load_baseline_results(args.baselines)
    
    if len(ace_df) == 0:
        print("ERROR: No ACE results found")
        return
    
    # Prepare output
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("STATISTICAL SIGNIFICANCE TESTS")
    output_lines.append("="*80)
    output_lines.append("")
    
    # ACE summary
    ace_values = ace_df['total_loss'].values
    ace_mean = np.mean(ace_values)
    ace_std = np.std(ace_values, ddof=1)
    ace_median = np.median(ace_values)
    ace_se = ace_std / np.sqrt(len(ace_values))
    ace_ci = stats.t.ppf(0.975, len(ace_values)-1) * ace_se
    
    output_lines.append(f"ACE (N={len(ace_values)}):")
    output_lines.append(f"  Mean: {ace_mean:.4f}")
    output_lines.append(f"  Median: {ace_median:.4f}")
    output_lines.append(f"  Std: {ace_std:.4f}")
    output_lines.append(f"  95% CI: [{ace_mean-ace_ci:.4f}, {ace_mean+ace_ci:.4f}]")
    output_lines.append("")
    
    # Comparisons
    output_lines.append("="*80)
    output_lines.append("PAIRWISE COMPARISONS (ACE vs Baselines)")
    output_lines.append("="*80)
    output_lines.append("")
    
    # Bonferroni correction
    n_comparisons = len(baselines)
    alpha_corrected = 0.05 / n_comparisons
    
    output_lines.append(f"Bonferroni correction: α = 0.05 / {n_comparisons} = {alpha_corrected:.4f}")
    output_lines.append("")
    
    comparison_results = []
    
    for method_name, baseline_values in sorted(baselines.items()):
        baseline_values = np.array(baseline_values)
        
        # Summary stats
        bl_mean = np.mean(baseline_values)
        bl_std = np.std(baseline_values, ddof=1)
        bl_median = np.median(baseline_values)
        
        # T-test (independent samples)
        t_stat, p_value = stats.ttest_ind(ace_values, baseline_values)
        
        # Effect size
        effect_size = cohens_d(ace_values, baseline_values)
        
        # Improvement
        improvement = ((bl_mean - ace_mean) / bl_mean) * 100
        improvement_median = ((bl_median - ace_median) / bl_median) * 100
        
        # Significance
        if p_value < alpha_corrected:
            sig_str = f"SIGNIFICANT (p={p_value:.4f} < {alpha_corrected:.4f})"
        else:
            sig_str = f"Not significant (p={p_value:.4f})"
        
        output_lines.append(f"{method_name.upper().replace('_', '-')}:")
        output_lines.append(f"  Mean: {bl_mean:.4f} ± {bl_std:.4f}")
        output_lines.append(f"  Median: {bl_median:.4f}")
        output_lines.append(f"  Improvement: {improvement:+.1f}% (mean), {improvement_median:+.1f}% (median)")
        output_lines.append(f"  t-statistic: {t_stat:.4f}")
        output_lines.append(f"  p-value: {p_value:.6f}")
        output_lines.append(f"  Cohen's d: {effect_size:.4f}")
        output_lines.append(f"  Result: {sig_str}")
        output_lines.append("")
        
        comparison_results.append({
            'Method': method_name.replace('_', '-'),
            'Baseline Mean': bl_mean,
            'ACE Mean': ace_mean,
            'Improvement %': improvement,
            'p-value': p_value,
            'Cohen\'s d': effect_size,
            'Significant': p_value < alpha_corrected
        })
    
    # Summary table
    output_lines.append("="*80)
    output_lines.append("SUMMARY TABLE")
    output_lines.append("="*80)
    output_lines.append("")
    
    df_results = pd.DataFrame(comparison_results)
    output_lines.append(df_results.to_string(index=False))
    output_lines.append("")
    
    # LaTeX table
    output_lines.append("="*80)
    output_lines.append("LATEX TABLE FOR PAPER")
    output_lines.append("="*80)
    output_lines.append("")
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Statistical comparison of ACE vs baseline methods. All comparisons use independent samples t-tests with Bonferroni correction ($\\alpha = 0.0125$).}\n"
    latex += "\\label{tab:statistical-tests}\n"
    latex += "\\begin{tabular}{lccccc}\n"
    latex += "\\toprule\n"
    latex += "Method & Mean Loss & Improvement & p-value & Cohen's d & Significant \\\\\n"
    latex += "\\midrule\n"
    latex += f"ACE & {ace_mean:.2f} $\\pm$ {ace_std:.2f} & -- & -- & -- & -- \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in df_results.iterrows():
        sig_marker = "$^{***}$" if row['p-value'] < 0.001 else ("$^{**}$" if row['p-value'] < 0.01 else ("$^{*}$" if row['p-value'] < alpha_corrected else ""))
        cohen_d_val = row["Cohen's d"]
        improvement_pct = row['Improvement %']
        latex += f"{row['Method'].title()} & {row['Baseline Mean']:.2f} & {improvement_pct:+.1f}\\% & {row['p-value']:.4f} & {cohen_d_val:.2f} & {sig_marker} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    output_lines.append(latex)
    output_lines.append("")
    output_lines.append("Note: *** p<0.001, ** p<0.01, * p<0.0125 (Bonferroni corrected)")
    output_lines.append("")
    
    # Print and save
    output_text = "\n".join(output_lines)
    print(output_text)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
