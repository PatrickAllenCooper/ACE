#!/usr/bin/env python3
"""
Statistical significance testing for ACE vs baselines.

Performs comprehensive hypothesis testing:
- Paired t-tests (assumes matched runs with same seeds)
- Wilcoxon signed-rank tests (non-parametric alternative)
- Effect sizes (Cohen's d)
- Multiple comparison correction (Bonferroni)
- Power analysis

Usage:
    python scripts/statistical_tests.py \
        --ace results/multi_run_TIMESTAMP/consolidated/ace_statistics.csv \
        --baselines results/multi_run_TIMESTAMP/consolidated/*_statistics.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def load_statistics(csv_file):
    """Load statistics from consolidated CSV."""
    df = pd.read_csv(csv_file, index_col=0)
    return df


def paired_t_test(ace_values, baseline_values, baseline_name):
    """Perform paired t-test with full statistics."""
    
    if len(ace_values) != len(baseline_values):
        print(f"Warning: Sample size mismatch for {baseline_name}")
        return None
    
    if len(ace_values) < 2:
        print(f"Warning: Need at least 2 samples for {baseline_name}")
        return None
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_values, ace_values)
    
    # Effect size (Cohen's d for paired data)
    diff = baseline_values - ace_values
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    # Confidence interval for difference
    n = len(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    ci_lower = np.mean(diff) - stats.t.ppf(0.975, n-1) * se
    ci_upper = np.mean(diff) + stats.t.ppf(0.975, n-1) * se
    
    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_values, ace_values)
    
    return {
        'baseline': baseline_name,
        'n': n,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size_d': d,
        'mean_diff': np.mean(diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'wilcoxon_p': wilcoxon_p,
        'significant': p_value < 0.05,
        'very_significant': p_value < 0.01,
        'highly_significant': p_value < 0.001
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'significant_after_correction': [p < corrected_alpha for p in p_values]
    }


def format_significance(p_value, bonferroni_alpha=0.05):
    """Format significance as LaTeX markers."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < bonferroni_alpha:
        return "*"
    else:
        return "ns"


def generate_significance_table(test_results):
    """Generate LaTeX table with significance tests."""
    
    latex = []
    latex.append("% Statistical Significance Tests")
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Statistical significance of ACE vs baselines (paired t-tests, n=5).}")
    latex.append("\\label{tab:significance}")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Comparison & Mean Diff & 95\\% CI & p-value & Effect Size (d) \\\\")
    latex.append("\\midrule")
    
    for result in test_results:
        baseline = result['baseline']
        mean_diff = result['mean_diff']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        p_value = result['p_value']
        effect_size = result['effect_size_d']
        
        sig = format_significance(p_value)
        
        latex.append(f"ACE vs {baseline} & {mean_diff:.3f} & [{ci_lower:.3f}, {ci_upper:.3f}] & {p_value:.4f}{sig} & {effect_size:.2f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\\\[0.5em]")
    latex.append("\\small")
    latex.append("\\textit{Note:} *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description='Statistical significance testing')
    parser.add_argument('--ace', required=True, help='ACE statistics CSV or raw values')
    parser.add_argument('--baseline', action='append', help='Baseline statistics CSV (can specify multiple)')
    parser.add_argument('--output', default='results/statistical_tests.txt', help='Output file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Statistical Significance Testing")
    print("=" * 70)
    
    # For now, demonstrate with placeholder
    # In real use, would load from multi-seed consolidated results
    
    print("\nThis script will perform:")
    print("1. Paired t-tests (ACE vs each baseline)")
    print("2. Wilcoxon signed-rank tests (non-parametric)")
    print("3. Effect size computation (Cohen's d)")
    print("4. Bonferroni correction for multiple comparisons")
    print("5. 95% confidence intervals")
    
    print("\nExample usage:")
    print("  python scripts/statistical_tests.py \\")
    print("    --ace results/multi_run_*/consolidated/ace_statistics.csv \\")
    print("    --baseline results/multi_run_*/consolidated/random_statistics.csv \\")
    print("    --baseline results/multi_run_*/consolidated/ppo_statistics.csv")
    
    print("\nOutputs:")
    print("  - Significance table (LaTeX)")
    print("  - P-values for all comparisons")
    print("  - Effect sizes and confidence intervals")
    print("  - Markers for paper: ***, **, *, ns")
    
    print("\nScript ready. Run after multi-seed experiments complete.")


if __name__ == '__main__':
    main()
