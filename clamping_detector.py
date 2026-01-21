#!/usr/bin/env python3
"""
Clamping Detector - Verify if ACE learned clamping strategy in Duffing
Paper Claim (Line 661): "ACE discovers a 'clamping' strategy: by intervening 
to hold the middle oscillator fixed (do(X2 = 0))"
"""

import sys
import numpy as np
import glob
from pathlib import Path

def detect_clamping(experiment_log, target_node='X2', threshold=0.5, verbose=True):
    """
    Detect if interventions cluster near zero (clamping behavior).
    
    Args:
        experiment_log: Path to experiment log file
        target_node: Which node to check (default X2 for middle oscillator)
        threshold: Consider "near zero" if abs(value) < threshold
        verbose: Print detailed analysis
    
    Returns:
        dict with mean, std, clamping_percentage, is_clamping
    """
    # Extract intervention values
    try:
        with open(experiment_log) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {'error': f'File not found: {experiment_log}'}
    
    values = []
    for line in lines:
        if f'DO {target_node} = ' in line:
            try:
                # Parse: "DO X2 = -3.5" or similar
                val_str = line.split(f'{target_node} = ')[1].split()[0].strip('"\'')
                val = float(val_str)
                values.append(val)
            except (IndexError, ValueError):
                continue
    
    if not values:
        return {'error': f'No {target_node} interventions found in log'}
    
    # Statistics
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    near_zero = sum(1 for v in values if abs(v) < threshold) / len(values) * 100
    
    # Clamping criteria:
    # 1. Mean close to 0 (|mean| < 0.5)
    # 2. Low variance (std < 1.0) 
    # 3. Most values near 0 (>50%)
    is_clamping = abs(mean) < 0.5 and std < 1.0 and near_zero > 50
    
    result = {
        'mean': mean,
        'median': median,
        'std': std,
        'min': min(values),
        'max': max(values),
        'near_zero_pct': near_zero,
        'is_clamping': is_clamping,
        'total_interventions': len(values),
        'values': values
    }
    
    if verbose:
        print(f"Clamping Analysis for {target_node}:")
        print(f"  Total interventions: {result['total_interventions']}")
        print(f"  Mean: {result['mean']:.3f}")
        print(f"  Median: {result['median']:.3f}")
        print(f"  Std Dev: {result['std']:.3f}")
        print(f"  Range: [{result['min']:.3f}, {result['max']:.3f}]")
        print(f"  Near zero (|x| < {threshold}): {result['near_zero_pct']:.1f}%")
        print()
        
        if is_clamping:
            print(f"  ✅ CLAMPING DETECTED")
            print(f"     Paper claim (line 661) SUPPORTED")
        else:
            print(f"  ❌ NO CLEAR CLAMPING")
            print(f"     Paper claim (line 661) NOT SUPPORTED")
            print()
            print(f"  Recommendation: Revise claim to:")
            print(f"     'ACE successfully identifies the chain coupling structure'")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Detect clamping strategy in Duffing experiment')
    parser.add_argument('--log', type=str, help='Path to experiment log (or use glob pattern)')
    parser.add_argument('--node', type=str, default='X2', help='Node to check (default: X2)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Near-zero threshold')
    args = parser.parse_args()
    
    # Find log file
    if args.log:
        log_path = args.log
    else:
        # Auto-find most recent Duffing log
        pattern = 'results/paper_*/duffing/*/experiment.log'
        matches = sorted(glob.glob(pattern))
        if not matches:
            pattern2 = 'logs copy/duffing_*.err'
            matches = sorted(glob.glob(pattern2))
        
        if not matches:
            print(f"❌ No Duffing logs found")
            print(f"   Tried: {pattern}")
            sys.exit(1)
        
        log_path = matches[-1]
        print(f"Using: {log_path}")
        print()
    
    result = detect_clamping(log_path, target_node=args.node, threshold=args.threshold)
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
        sys.exit(1)
    
    # Save detailed analysis
    output_path = Path('results/claim_evidence/duffing_clamping.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Evidence for: Clamping Strategy in Duffing\n\n")
        f.write(f"**Paper Location:** Line 661\n")
        f.write(f"**Claim:** ACE discovers a 'clamping' strategy\n")
        f.write(f"**Status:** {'✅ Supported' if result['is_clamping'] else '❌ Not supported'}\n\n")
        f.write(f"## Analysis\n\n")
        f.write(f"**Source:** `{log_path}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"### Statistics\n\n")
        f.write(f"- Total {args.node} interventions: {result['total_interventions']}\n")
        f.write(f"- Mean: {result['mean']:.3f}\n")
        f.write(f"- Median: {result['median']:.3f}\n")
        f.write(f"- Std Dev: {result['std']:.3f}\n")
        f.write(f"- Range: [{result['min']:.3f}, {result['max']:.3f}]\n")
        f.write(f"- Near zero (<{args.threshold}): {result['near_zero_pct']:.1f}%\n\n")
        
        if result['is_clamping']:
            f.write(f"### Conclusion\n\n")
            f.write(f"✅ Clamping behavior detected. Paper claim is supported.\n")
        else:
            f.write(f"### Conclusion\n\n")
            f.write(f"❌ No clear clamping pattern. Recommend revising paper claim.\n\n")
            f.write(f"**Suggested revision:**\n")
            f.write(f'```latex\n')
            f.write(f"ACE successfully identifies the chain coupling structure, achieving\n")
            f.write(f"[STRUCTURE_F1] F1 score for structure identification.\n")
            f.write(f'```\n')
    
    print(f"\n📝 Detailed analysis saved to: {output_path}")
    print(f"✅ Update results/RESULTS_LOG.md with this finding")

if __name__ == '__main__':
    main()
