#!/usr/bin/env python3
"""
Track ablation results as they complete and update paper estimates.
"""

import pandas as pd
import glob
import numpy as np
from pathlib import Path

def get_final_loss(run_dir):
    """Extract final total loss from a completed run."""
    node_loss_file = Path(run_dir) / "node_losses.csv"
    if not node_loss_file.exists():
        return None
    
    df = pd.read_csv(node_loss_file)
    last = df.iloc[-1]
    
    # Sum all loss_X columns
    total = sum(last[col] for col in df.columns if col.startswith('loss_'))
    return total

def track_ablations():
    """Track all ablation results."""
    
    results = {}
    
    # Check each ablation type
    for ablation in ['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity']:
        results[ablation] = {}
        
        # Check each seed
        for seed in [42, 123, 456]:
            pattern = f"results/ablations_complete/{ablation}/seed_{seed}/run_*/node_losses.csv"
            files = glob.glob(pattern)
            
            if files:
                run_dir = Path(files[0]).parent
                loss = get_final_loss(run_dir)
                
                if loss:
                    results[ablation][seed] = loss
    
    # Print summary
    print("="*70)
    print("ABLATION RESULTS TRACKER")
    print("="*70)
    
    ace_full = 0.92
    
    for ablation in ['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity']:
        seeds_data = results.get(ablation, {})
        
        if seeds_data:
            losses = list(seeds_data.values())
            mean = np.mean(losses)
            std = np.std(losses, ddof=1) if len(losses) > 1 else 0
            degradation = ((mean - ace_full) / ace_full * 100)
            
            print(f"\n{ablation}:")
            print(f"  Seeds completed: {len(losses)}/3")
            for seed, loss in seeds_data.items():
                print(f"    Seed {seed}: {loss:.4f}")
            
            if len(losses) >= 1:
                print(f"  Mean: {mean:.4f} ± {std:.4f}")
                print(f"  Degradation: +{degradation:.0f}%")
        else:
            print(f"\n{ablation}: No results yet")
    
    print("\n" + "="*70)
    print(f"Total ablations complete: {sum(len(v) for v in results.values())}/12")
    print("="*70)
    
    return results

if __name__ == "__main__":
    track_ablations()
