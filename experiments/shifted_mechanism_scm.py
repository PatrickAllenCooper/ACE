#!/usr/bin/env python3
"""Prespecified function-family shift on the known 30-node hierarchical DAG."""
from __future__ import annotations

import torch

from experiments.large_scale_scm import LargeScaleSCM


class ShiftedMechanismSCM(LargeScaleSCM):
    """Stationary mechanisms with bounded nonlinear and interaction terms.

    Graph, root law, coefficients, and observation interface match LargeScaleSCM.
    Function forms differ from the homogeneous confirmation family. No policy
    receives the form labels; they exist for provenance and error diagnosis.
    """

    def __init__(self, n_nodes=30, coeff_seed=None):
        super().__init__(n_nodes=n_nodes, coeff_seed=coeff_seed)
        self.forms = {node: ('root' if not self.graph[node] else
                             ('saturating' if self.node_idx[node] % 3 == 0 else
                              'interaction' if self.node_idx[node] % 3 == 1 else
                              'ripple')) for node in self.nodes}

    def mechanisms(self, data, node, n_samples=1):
        n = next(iter(data.values())).shape[0] if data else n_samples
        noise = torch.randn(n) * self.noise_std
        parents = self.get_parents(node)
        if not parents:
            return torch.randn(n)
        additive = sum((self.coeffs[node][p] * data[p] for p in parents), torch.zeros(n))
        form = self.forms[node]
        if form == 'saturating':
            value = 1.2 * torch.tanh(additive / 1.2)
        elif form == 'interaction':
            value = additive
            if len(parents) >= 2:
                value = value + 0.35 * torch.tanh(data[parents[0]]) * torch.tanh(data[parents[1]])
        else:
            value = additive + 0.35 * torch.sin(2.1 * additive)
        return value + noise
