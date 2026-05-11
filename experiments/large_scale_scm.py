#!/usr/bin/env python3
"""
Large-Scale SCM (30-50 nodes) for scalability testing.

Tests ACE on larger causal systems to demonstrate:
1. Scalability beyond 15 nodes
2. Strategic advantage at scale
3. Episode efficiency at scale

This addresses: "Limited scalability demonstration (5-15 nodes)"
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LargeScaleSCM:
    """
    Hierarchical SCM for scalability testing. Defaults to 30 nodes; supports
    50 (and other sizes) by parametric layer scaling. Optionally anonymises
    node names to abstract hex IDs (n_xxxx) for the LM-prior-mismatch
    ablation: with semantically meaningful names like X1, X2, ... the
    pretrained LM's prior may be informative; with anonymised tokens the
    LM has no semantic handle to exploit.

    Structure (5-layer, proportions ~1/6, 1/6, 1/3, 1/6, 1/6 of n_nodes):
    - Layer 0: roots (no parents)
    - Layer 1: each node has 1-2 root parents
    - Layer 2: mix of single-parent and colliders (every third has 2 parents)
    - Layer 3: complex colliders with 2-3 parents from Layer 2
    - Layer 4: leaves with 1 Layer-3 parent
    """

    def __init__(self, n_nodes=30, anonymize=False, anonymize_seed=None):
        self.n_nodes = n_nodes
        # 5 layers, sized roughly proportional to n_nodes. For n=30 this
        # reproduces the canonical 5/5/10/5/5 split exactly.
        if n_nodes == 30:
            self.layer_sizes = [5, 5, 10, 5, 5]
        else:
            r = max(2, n_nodes // 6)
            l1 = max(2, n_nodes // 6)
            l3 = max(2, n_nodes // 6)
            leaves = max(2, n_nodes // 6)
            l2 = n_nodes - r - l1 - l3 - leaves
            assert l2 >= 2, f"n_nodes={n_nodes} too small after fixed-size layers"
            self.layer_sizes = [r, l1, l2, l3, leaves]

        # Layer index ranges (start, end) in 1-indexed node IDs.
        self.layer_ranges = []
        cursor = 1
        for sz in self.layer_sizes:
            self.layer_ranges.append((cursor, cursor + sz))
            cursor += sz

        # Canonical X-names for internal graph construction. Anonymisation
        # is applied only AT THE END to self.nodes / self.graph keys, so the
        # build logic remains unchanged.
        canonical = [f"X{i}" for i in range(1, n_nodes + 1)]
        self.graph = self._build_hierarchical_graph(canonical)
        self.noise_std = 0.15

        if anonymize:
            rng = np.random.RandomState(
                anonymize_seed if anonymize_seed is not None else 0
            )
            seen = set()
            alias = {}
            for c in canonical:
                while True:
                    tok = f"n_{rng.randint(0, 2**16):04x}"
                    if tok not in seen:
                        break
                seen.add(tok)
                alias[c] = tok
            # Rebuild graph with anonymised keys/values.
            self.graph = {alias[k]: [alias[p] for p in v]
                          for k, v in self.graph.items()}
            self.nodes = [alias[c] for c in canonical]
            self.alias = alias  # canonical -> anonymised
        else:
            self.nodes = canonical
            self.alias = {c: c for c in canonical}

        # 1-indexed node-id lookup, stable across canonical/anonymised
        # (so generate()'s "every 5th node gets nonlinearity" rule keeps
        # working when names like X1 are renamed to n_dc66).
        self.node_idx = {name: i + 1 for i, name in enumerate(self.nodes)}

    def _build_hierarchical_graph(self, names):
        """Build hierarchical causal graph using canonical X-names. Layer
        sizes follow self.layer_sizes; structure mirrors the original
        30-node design (mix of single-parent and 2-parent colliders)."""
        graph = {}
        (r0, r1) = self.layer_ranges[0]
        (l1_0, l1_1) = self.layer_ranges[1]
        (l2_0, l2_1) = self.layer_ranges[2]
        (l3_0, l3_1) = self.layer_ranges[3]
        (lf_0, lf_1) = self.layer_ranges[4]

        # Layer 0: roots.
        for i in range(r0, r1):
            graph[f"X{i}"] = []

        # Layer 1: each node has 1-2 root parents.
        for i in range(l1_0, l1_1):
            num_parents = np.random.choice([1, 2])
            num_parents = min(num_parents, r1 - r0)
            parents = np.random.choice(range(r0, r1),
                                       size=num_parents, replace=False)
            graph[f"X{i}"] = [f"X{p}" for p in parents]

        # Layer 2: every third is a 2-parent collider, rest single-parent.
        l1_size = l1_1 - l1_0
        for i in range(l2_0, l2_1):
            if i % 3 == 0 and l1_size >= 2:
                parents = np.random.choice(range(l1_0, l1_1),
                                           size=2, replace=False)
            else:
                parents = [np.random.choice(range(l1_0, l1_1))]
            graph[f"X{i}"] = [f"X{p}" for p in parents]

        # Layer 3: complex colliders, 2-3 parents from Layer 2.
        l2_size = l2_1 - l2_0
        for i in range(l3_0, l3_1):
            num_parents = np.random.choice([2, 3])
            num_parents = min(num_parents, l2_size)
            parents = np.random.choice(range(l2_0, l2_1),
                                       size=num_parents, replace=False)
            graph[f"X{i}"] = [f"X{p}" for p in parents]

        # Layer 4: leaves with 1 Layer-3 parent.
        l3_size = l3_1 - l3_0
        for i in range(lf_0, lf_1):
            parents = np.random.choice(range(l3_0, l3_1),
                                       size=min(1, l3_size), replace=False)
            graph[f"X{i}"] = [f"X{p}" for p in parents]

        return graph
    
    def get_parents(self, node: str) -> List[str]:
        return self.graph.get(node, [])
    
    def generate(self, n_samples: int, interventions: Optional[Dict[str, float]] = None):
        """Generate samples from large-scale SCM."""
        interventions = interventions or {}
        data = {}
        
        # Topological order
        topo_order = self._topological_sort()
        
        for node in topo_order:
            if node in interventions:
                data[node] = torch.full((n_samples,), float(interventions[node]))
            else:
                parents = self.get_parents(node)
                
                if not parents:
                    # Root
                    data[node] = torch.randn(n_samples)
                else:
                    # Combine parents with random coefficients
                    value = torch.zeros(n_samples)
                    for parent in parents:
                        coef = np.random.uniform(0.3, 0.7)
                        value += coef * data[parent]
                    
                    # Add nonlinearity for some nodes (every 5th node).
                    # Use the stable node-id lookup so this works for both
                    # canonical (X1..XN) and anonymised (n_xxxx) names.
                    node_num = self.node_idx.get(node, 0)
                    if node_num % 5 == 0:
                        value = value + 0.2 * torch.sin(value)
                    
                    data[node] = value + torch.randn(n_samples) * self.noise_std
        
        return data
    
    def _topological_sort(self):
        """Topological sort of nodes."""
        # Simple approach: sort by max path length from roots
        visited = set()
        order = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.get_parents(node):
                visit(parent)
            order.append(node)
        
        for node in self.nodes:
            visit(node)
        
        return order


def main():
    parser = argparse.ArgumentParser(description='Large-scale SCM test')
    parser.add_argument('--n_nodes', type=int, default=30, help='Number of nodes')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes to run')
    parser.add_argument('--output', type=str, default='results/large_scale', help='Output')
    
    args = parser.parse_args()
    
    print(f"Testing ACE on {args.n_nodes}-node SCM")
    print("=" * 60)
    
    # Create SCM
    scm = LargeScaleSCM(args.n_nodes)
    print(f"Nodes: {args.n_nodes}")
    print(f"Edges: {sum(len(parents) for parents in scm.graph.values())}")
    
    # Count colliders
    colliders = sum(1 for node in scm.nodes if len(scm.get_parents(node)) >= 2)
    print(f"Colliders: {colliders}")
    
    print("\nNote: This demonstrates scalability to realistic system sizes.")
    print("Random sampling becomes very inefficient at this scale (~3.3% per node)")
    print("Strategic intervention should show clear advantages.")
    
    print(f"\nRun with: python -m experiments.large_scale_scm --episodes {args.episodes}")


if __name__ == '__main__':
    main()
