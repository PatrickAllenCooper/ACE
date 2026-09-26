#!/usr/bin/env python3
"""Connected sparse SCM simulator for the next joint-intervention gate.

Motif j has parents (the previous motif child, a fresh root), except motif 0,
which has two roots. Remaining nodes form a linear descendant chain. An action
may set one or both parents of one motif; if it sets a previous child, that
child's natural mechanism is masked from the acquired training batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class System:
    nodes: int
    motifs: int
    edges: tuple[tuple[int, int], ...]
    parents: tuple[tuple[int, int], ...]
    children: tuple[int, ...]
    coefficients: np.ndarray
    root_sd: float
    child_sd: float = .15


def make_system(seed: int, nodes: int, motifs: int, root_sd: float) -> System:
    if motifs < 1 or nodes < 2 * motifs + 1 or root_sd < 0:
        raise ValueError('Invalid connected motif system')
    rng = np.random.default_rng(seed + 81211)
    # Node 0 and node 1 are roots; motif-0 child is node 2.
    # Every later motif contributes one fresh root and one child.
    parents = [(0, 1)]
    children = [2]
    edges = [(0, 2), (1, 2)]
    for j in range(1, motifs):
        root = 2*j + 1
        child = 2*j + 2
        parents.append((children[-1], root))
        children.append(child)
        edges.extend(((parents[-1][0], child), (root, child)))
    for node in range(2*motifs + 1, nodes):
        edges.append((node-1, node))
    coeff = np.column_stack((rng.normal(.45, .05, motifs),
                             rng.normal(-.3, .05, motifs),
                             rng.choice((-1., 1.), motifs) * rng.uniform(.7, 1.1, motifs)))
    return System(nodes, motifs, tuple(edges), tuple(parents), tuple(children),
                  coeff, root_sd)


def action_menu(system: System, pair: bool):
    menu = []
    for j, (p1, p2) in enumerate(system.parents):
        if pair:
            menu.extend((j, (p1, p2), (a, b)) for a in (-2., 2.) for b in (-2., 2.))
        else:
            menu.extend((j, (p,), (v,)) for p in (p1, p2) for v in (-2., 2.))
    return menu


def sample(system: System, rng: np.random.Generator, n: int, action=None,
           coefficients: np.ndarray | None = None,
           padding_rng: np.random.Generator | None = None):
    """Return observed nodes, motif features, and natural-child observation mask."""
    if n < 1:
        raise ValueError('n must be positive')
    values = np.empty((n, system.nodes))
    if padding_rng is None:
        padding_rng = rng
    if coefficients is None:
        coefficients = system.coefficients
    if coefficients.shape != (system.motifs, 3):
        raise ValueError('Invalid mechanism coefficients')
    natural = np.ones((n, system.motifs), dtype=bool)
    interventions = {}
    if action is not None:
        _, targets, levels = action
        interventions = dict(zip(targets, levels))
        if len(interventions) != len(targets) or any(t < 0 or t >= system.nodes for t in targets):
            raise ValueError('Invalid intervention targets')
    child_to_motif = {node: j for j, node in enumerate(system.children)}
    root_nodes = {0, 1} | {2*j+1 for j in range(1, system.motifs)}
    for node in range(system.nodes):
        if node in interventions:
            values[:, node] = interventions[node]
            if node in child_to_motif:
                natural[:, child_to_motif[node]] = False
        elif node in root_nodes:
            values[:, node] = rng.normal(0, system.root_sd, n)
        elif node in child_to_motif:
            j = child_to_motif[node]
            p1, p2 = system.parents[j]
            x1, x2 = values[:, p1], values[:, p2]
            a, b, c = coefficients[j]
            values[:, node] = a*x1 + b*x2 + c*x1*x2 + rng.normal(0, system.child_sd, n)
        else:
            values[:, node] = .5*values[:, node-1] + padding_rng.normal(0, system.child_sd, n)
    features = np.stack([np.column_stack((values[:, a], values[:, b],
                                           values[:, a]*values[:, b]))
                         for a, b in system.parents], axis=1)
    return values, features, natural
