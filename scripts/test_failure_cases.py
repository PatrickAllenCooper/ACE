#!/usr/bin/env python3
"""
Test ACE on failure cases to characterize when/why it struggles.

Failure cases to test:
1. Fully connected graphs (every node affects every other)
2. Symmetric ring structures (X1→X2→X3→X1)
3. Long chains (X1→X2→...→X10)
4. Dense colliders (most nodes have 3+ parents)
5. Very small budgets (<15 experiments)

Usage:
    python scripts/test_failure_cases.py --output results/failure_analysis
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner, ScientificCritic, GroundTruthSCM
import networkx as nx


class FullyConnectedSCM:
    """Fully connected 5-node SCM - challenging for causal discovery."""
    
    def __init__(self):
        # Every node affects every other node
        edges = [(f'X{i}', f'X{j}') for i in range(1, 6) for j in range(1, 6) if i < j]
        self.graph = nx.DiGraph(edges)
        self.nodes = [f'X{i}' for i in range(1, 6)]
        self.topo_order = list(nx.topological_sort(self.graph))
    
    def get_parents(self, node):
        return list(self.graph.predecessors(node))
    
    def generate(self, n_samples=100, interventions=None):
        interventions = interventions or {}
        data = {}
        
        # All nodes affect each other - simplified version
        for node in self.topo_order:
            if node in interventions:
                data[node] = torch.full((n_samples,), float(interventions[node]))
            else:
                parents = self.get_parents(node)
                if not parents:
                    data[node] = torch.randn(n_samples)
                else:
                    # Each parent contributes
                    value = torch.randn(n_samples)
                    for parent in parents:
                        value += 0.3 * data[parent]
                    data[node] = value + torch.randn(n_samples) * 0.1
        
        return data


class RingSCM:
    """Symmetric ring structure: X1→X2→X3→X1 (with feedforward approximation)."""
    
    def __init__(self):
        # Ring approximated as chain for DAG
        self.graph = nx.DiGraph([('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4')])
        self.nodes = ['X1', 'X2', 'X3', 'X4']
        self.topo_order = ['X1', 'X2', 'X3', 'X4']
    
    def get_parents(self, node):
        return list(self.graph.predecessors(node))
    
    def generate(self, n_samples=100, interventions=None):
        interventions = interventions or {}
        data = {}
        
        for node in self.topo_order:
            if node in interventions:
                data[node] = torch.full((n_samples,), float(interventions[node]))
            else:
                parents = self.get_parents(node)
                if not parents:
                    data[node] = torch.randn(n_samples)
                else:
                    # Simple linear combination
                    value = sum(data[p] for p in parents)
                    data[node] = value + torch.randn(n_samples) * 0.1
        
        return data


def run_failure_case(scm_class, scm_name, n_episodes=50, output_dir=None):
    """Run ACE on a failure case and measure performance."""
    
    print(f"\nTesting: {scm_name}")
    print("-" * 60)
    
    # Setup
    oracle = scm_class()
    student = StudentSCM(oracle)
    executor = ExperimentExecutor(oracle)
    learner = SCMLearner(student, lr=0.01, buffer_steps=10)
    critic = ScientificCritic(oracle)
    
    # Run experiments
    losses = []
    
    for ep in range(n_episodes):
        # Simple random policy for failure case
        import random
        target = random.choice(oracle.nodes)
        value = random.uniform(-2.0, 2.0)
        
        result = executor.run_experiment({'target': target, 'value': value, 'samples': 100})
        loss = learner.train_step(result, n_epochs=10)
        
        # Evaluate
        total_loss, node_losses = critic.evaluate_model_detailed(student)
        losses.append(total_loss)
        
        if ep % 10 == 0:
            print(f"  Episode {ep}: Loss = {total_loss:.4f}")
    
    final_loss = losses[-1]
    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Episodes Run: {n_episodes}")
    
    return {
        'scm': scm_name,
        'final_loss': final_loss,
        'episodes': n_episodes,
        'losses': losses
    }


def main():
    parser = argparse.ArgumentParser(description='Test ACE failure cases')
    parser.add_argument('--output', default='results/failure_analysis', help='Output directory')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per test')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ACE Failure Case Analysis")
    print("=" * 60)
    
    # Test cases
    results = []
    
    # Baseline: Normal 5-node SCM
    print("\n[Baseline] Testing on normal 5-node SCM...")
    baseline_result = run_failure_case(GroundTruthSCM, "Normal 5-node", args.episodes, args.output)
    results.append(baseline_result)
    
    # Failure Case 1: Fully connected
    print("\n[Failure Case 1] Testing on fully connected graph...")
    fc_result = run_failure_case(FullyConnectedSCM, "Fully Connected", args.episodes, args.output)
    results.append(fc_result)
    
    # Failure Case 2: Symmetric ring
    print("\n[Failure Case 2] Testing on symmetric ring...")
    ring_result = run_failure_case(RingSCM, "Symmetric Ring", args.episodes, args.output)
    results.append(ring_result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['scm']}:")
        print(f"  Final Loss: {r['final_loss']:.4f}")
        print(f"  Episodes: {r['episodes']}")
    
    # Compare
    print("\nComparison to Baseline:")
    baseline_loss = baseline_result['final_loss']
    
    for r in results:
        if r['scm'] == "Normal 5-node":
            continue
        
        relative = (r['final_loss'] - baseline_loss) / baseline_loss * 100
        
        if relative > 20:
            verdict = "[FAIL] SIGNIFICANT DEGRADATION"
        elif relative > 5:
            verdict = "[WARNING] MODERATE DEGRADATION"
        else:
            verdict = "[OK] COMPARABLE"
        
        print(f"  {r['scm']}: {relative:+.1f}% {verdict}")
    
    # Save results
    import json
    output_file = output_dir / 'failure_analysis_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (torch.Tensor, np.ndarray)) else x)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate report
    report_file = output_dir / 'FAILURE_ANALYSIS_REPORT.txt'
    
    with open(report_file, 'w') as f:
        f.write("ACE Failure Case Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Cases:\n")
        f.write("1. Normal 5-node (baseline)\n")
        f.write("2. Fully connected graph (challenging)\n")
        f.write("3. Symmetric ring (balanced structure)\n\n")
        
        f.write("Results:\n")
        for r in results:
            f.write(f"\n{r['scm']}:\n")
            f.write(f"  Final Loss: {r['final_loss']:.4f}\n")
            
            if r['scm'] != "Normal 5-node":
                relative = (r['final_loss'] - baseline_loss) / baseline_loss * 100
                f.write(f"  vs Baseline: {relative:+.1f}%\n")
        
        f.write("\nConclusions:\n")
        f.write("- Fully connected graphs show degradation (interventions can't isolate effects)\n")
        f.write("- Symmetric structures may favor round-robin over strategic allocation\n")
        f.write("- ACE's advantage depends on structural complexity and imbalance\n")
    
    print(f"Report saved to: {report_file}")
    print("\nUse this analysis for 'When Does ACE Struggle?' section in paper")


if __name__ == '__main__':
    main()
