"""Shifted nonlinear mechanisms on a sparse, ordered random DAG.

This changes the graph generator while preserving the node count, root count,
coefficient law, noise law, and function family of ``ShiftedMechanismSCM``.
"""
from __future__ import annotations

import numpy as np

from experiments.shifted_mechanism_scm import ShiftedMechanismSCM


class RandomOrderShiftedSCM(ShiftedMechanismSCM):
    """Five roots, then one or two parents drawn from any earlier node.

    Unlike the five-layer generator, edges may skip arbitrarily far back in
    the topological order. The runner seeds NumPy before construction, so a
    given seed fixes the graph; coefficient draws are separately reseeded by
    the inherited ``freeze_coefficients`` method.
    """

    def _build_hierarchical_graph(self, names):
        if len(names) != 30:
            raise ValueError('The random-order graph protocol is defined for 30 nodes')
        graph = {name: [] for name in names[:5]}
        for index, node in enumerate(names[5:], start=5):
            n_parents = int(np.random.choice((1, 2)))
            parents = np.random.choice(names[:index], size=n_parents, replace=False)
            graph[node] = [str(parent) for parent in parents]
        return graph
