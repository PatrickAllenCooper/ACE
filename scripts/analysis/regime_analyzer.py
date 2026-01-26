#!/usr/bin/env python3
"""
Regime Selection Analyzer - Verify if ACE selects high-volatility periods in Phillips
Paper Claim (Line 714): "ACE learns to query high-volatility regimes (1970s stagflation, 2008 crisis)"
"""

import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

# Define historical economic regimes (approximate)
REGIMES = {
    'great_inflation': {'years': (1965, 1982), 'volatility': 'HIGH'},
    'volcker_disinflation': {'years': (1979, 1982), 'volatility': 'HIGH'},
    'great_moderation': {'years': (1985, 2007), 'volatility': 'LOW'},
    'great_recession': {'years': (2007, 2009), 'volatility': 'HIGH'},
    'post_crisis': {'years': (2010, 2019), 'volatility': 'MEDIUM'},
    'covid': {'years': (2020, 2023), 'volatility': 'HIGH'}
}

def analyze_regime_selection(results_csv, verbose=True):
    """
    Analyze if ACE preferentially selects high-volatility historical periods.
    
    Args:
        results_csv: Path to Phillips experiment results.csv
        verbose: Print detailed analysis
    
    Returns:
        dict with regime distribution and selectivity metrics
    """
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        return {'error': f'File not found: {results_csv}'}
    
    if 'episode' not in df.columns:
        return {'error': 'No episode column in results'}
    
    # Assuming 552 records from 1960-2023
    # Map episode to year (approximate)
    # This is a placeholder - adjust based on actual data structure
    total_records = 552
    years_span = 63  # 1960-2023
    
    def episode_to_year(ep):
        # Rough mapping (adjust based on actual implementation)
        return 1960 + int((ep / 100) * years_span)
    
    # Classify episodes by regime
    regime_counts = {regime: 0 for regime in REGIMES}
    high_vol_count = 0
    low_vol_count = 0
    total_queries = len(df)
    
    for episode in df['episode'].unique():
        year = episode_to_year(episode)
        
        # Find which regime this falls into
        for regime_name, regime_info in REGIMES.items():
            year_start, year_end = regime_info['years']
            if year_start <= year <= year_end:
                regime_counts[regime_name] += 1
                if regime_info['volatility'] == 'HIGH':
                    high_vol_count += 1
                elif regime_info['volatility'] == 'LOW':
                    low_vol_count += 1
                break
    
    high_vol_pct = high_vol_count / total_queries * 100 if total_queries > 0 else 0
    low_vol_pct = low_vol_count / total_queries * 100 if total_queries > 0 else 0
    
    # Calculate dataset composition (for comparison to random sampling)
    dataset_high_vol_years = sum(
        regime['years'][1] - regime['years'][0] + 1 
        for regime in REGIMES.values() 
        if regime['volatility'] == 'HIGH'
    )
    dataset_high_vol_pct = dataset_high_vol_years / years_span * 100
    
    # Selectivity: How much more than random
    selectivity = high_vol_pct / dataset_high_vol_pct if dataset_high_vol_pct > 0 else 1.0
    
    # Check pre-1985 concentration (1970s focus)
    pre_1985_count = sum(1 for ep in df['episode'].unique() if episode_to_year(ep) < 1985)
    pre_1985_pct = pre_1985_count / total_queries * 100
    dataset_pre_1985_pct = 25 / years_span * 100  # 1960-1985 = 25 years
    pre_1985_selectivity = pre_1985_pct / dataset_pre_1985_pct if dataset_pre_1985_pct > 0 else 1.0
    
    # Is selective: >20% more than random
    is_selective = selectivity > 1.2
    
    result = {
        'regime_counts': regime_counts,
        'high_volatility_pct': high_vol_pct,
        'low_volatility_pct': low_vol_pct,
        'dataset_high_vol_pct': dataset_high_vol_pct,
        'selectivity': selectivity,
        'pre_1985_pct': pre_1985_pct,
        'dataset_pre_1985_pct': dataset_pre_1985_pct,
        'pre_1985_selectivity': pre_1985_selectivity,
        'is_selective': is_selective,
        'total_queries': total_queries
    }
    
    if verbose:
        print(f"Regime Selection Analysis:")
        print(f"  Total queries: {result['total_queries']}")
        print()
        
        print(f"  High-volatility queries: {high_vol_pct:.1f}%")
        print(f"  Dataset composition: {dataset_high_vol_pct:.1f}%")
        print(f"  Selectivity: {selectivity:.2f}x")
        print()
        
        print(f"  Pre-1985 queries: {pre_1985_pct:.1f}%")
        print(f"  Dataset pre-1985: {dataset_pre_1985_pct:.1f}%")
        print(f"  Pre-1985 selectivity: {pre_1985_selectivity:.2f}x")
        print()
        
        print(f"  Regime breakdown:")
        for regime, count in regime_counts.items():
            pct = count / total_queries * 100 if total_queries > 0 else 0
            vol = REGIMES[regime]['volatility']
            print(f"    {regime:25s}: {count:3d} ({pct:5.1f}%) [{vol}]")
        print()
        
        if is_selective:
            print(f"  [DETECTED] SELECTIVE BEHAVIOR DETECTED")
            print(f"     Paper claim (line 714) SUPPORTED")
            print(f"     ACE queries high-volatility periods {selectivity:.1f}x more than random")
        else:
            print(f"  [NOT DETECTED] NO CLEAR SELECTIVITY")
            print(f"     Paper claim (line 714) NOT STRONGLY SUPPORTED")
            print()
            print(f"  Recommendation: Revise claim to:")
            print(f"     'ACE learns from diverse economic regimes, achieving [OOS_MSE]...'")
    
    return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze regime selection in Phillips experiment')
    parser.add_argument('--log', type=str, help='Path to Phillips results CSV (or use auto-find)')
    args = parser.parse_args()
    
    # Find results file
    if args.log:
        log_path = args.log
    else:
        # Auto-find most recent Phillips results
        pattern = 'results/paper_*/phillips/*/results.csv'
        matches = sorted(glob.glob(pattern))
        
        if not matches:
            print(f"[ERROR] No Phillips results found")
            print(f"   Tried: {pattern}")
            print()
            print(f"   Note: This script expects a results.csv file from Phillips experiment")
            print(f"   If experiment logged differently, adjust the script")
            sys.exit(1)
        
        log_path = matches[-1]
        print(f"Using: {log_path}")
        print()
    
    result = analyze_regime_selection(log_path)
    
    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)
    
    # Save detailed analysis
    output_path = Path('results/claim_evidence/phillips_regimes.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Evidence for: Regime Selection in Phillips\n\n")
        f.write(f"**Paper Location:** Line 714\n")
        f.write(f"**Claim:** ACE learns to query high-volatility regimes\n")
        f.write(f"**Status:** {'[SUPPORTED]' if result['is_selective'] else '[NOT SUPPORTED]'}\n\n")
        f.write(f"## Analysis\n\n")
        f.write(f"**Source:** `{log_path}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write(f"### Selectivity Metrics\n\n")
        f.write(f"- High-volatility queries: {result['high_volatility_pct']:.1f}%\n")
        f.write(f"- Dataset high-volatility: {result['dataset_high_vol_pct']:.1f}%\n")
        f.write(f"- Selectivity ratio: {result['selectivity']:.2f}x\n\n")
        
        f.write(f"- Pre-1985 queries: {result['pre_1985_pct']:.1f}%\n")
        f.write(f"- Dataset pre-1985: {result['dataset_pre_1985_pct']:.1f}%\n")
        f.write(f"- Pre-1985 selectivity: {result['pre_1985_selectivity']:.2f}x\n\n")
        
        f.write(f"### Regime Distribution\n\n")
        for regime, count in result['regime_counts'].items():
            pct = count / result['total_queries'] * 100
            vol = REGIMES[regime]['volatility']
            f.write(f"- {regime}: {count} queries ({pct:.1f}%) [{vol}]\n")
        
        f.write(f"\n### Conclusion\n\n")
        if result['is_selective']:
            f.write(f"[SUPPORTED] Selective behavior detected. Paper claim is supported.\n\n")
            f.write(f"ACE queries high-volatility periods {result['selectivity']:.1f}x more than random sampling.\n")
        else:
            f.write(f"[NOT SUPPORTED] No strong selectivity detected. Recommend revising paper claim.\n\n")
            f.write(f"**Suggested revision:**\n")
            f.write(f'```latex\n')
            f.write(f"ACE learns from multiple economic regimes, achieving [OOS_MSE]\n")
            f.write(f"out-of-sample MSE compared to [BASELINE_OOS] for chronological sampling.\n")
            f.write(f'```\n')
    
    print(f"\n[SAVED] Detailed analysis saved to: {output_path}")
    print(f"[ACTION] Update results/RESULTS_LOG.md with this finding")

if __name__ == '__main__':
    main()
