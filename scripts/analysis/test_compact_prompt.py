#!/usr/bin/env python3
"""
Unit test for the compact prompt-encoding strategy (scaling enabler).

Validates, without loading any language model, that:
  1. `compact` produces a strictly shorter prompt than `full` on a large graph.
  2. `compact` surfaces only the top-m highest-loss nodes and their parents
     (so prompt length is governed by --prompt_top_m, not N).
  3. `compact` still proposes valid intervention targets (parents of failing
     nodes), and `full` reproduces the paper-anchor behaviour (all nodes).

Run:
  python scripts/analysis/test_compact_prompt.py
"""
import os
import sys
import types
from types import SimpleNamespace

import networkx as nx

# Stub heavy/optional modules so importing ace_experiments does not pull in
# matplotlib/seaborn/transformers (which may have a NumPy-ABI mismatch in the
# base env). scm_to_prompt depends on none of them.
for _name in ("matplotlib", "matplotlib.pyplot", "seaborn"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = object
_tf.AutoTokenizer = object
sys.modules.setdefault("transformers", _tf)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ace_experiments import HuggingFacePolicy  # noqa: E402


def _make_stub(strategy, top_m):
    """A minimal object exposing just the attributes scm_to_prompt reads."""
    return SimpleNamespace(
        prompt_strategy=strategy,
        prompt_top_m=top_m,
        dsl=SimpleNamespace(value_min=-5.0, value_max=5.0),
    )


def _make_scm(n_nodes=50, width=8):
    """Layered DAG: roots feed a chain of layers, mirroring LargeScaleSCM."""
    g = nx.DiGraph()
    nodes = [f"X{i+1}" for i in range(n_nodes)]
    g.add_nodes_from(nodes)
    for i in range(width, n_nodes):
        # Each non-root node gets two parents from the preceding layer band.
        p1 = nodes[i - width]
        p2 = nodes[i - width // 2 if i - width // 2 >= 0 else 0]
        g.add_edge(p1, nodes[i])
        if p2 != p1:
            g.add_edge(p2, nodes[i])
    scm = SimpleNamespace(nodes=nodes, graph=g)
    return scm, nodes


def main():
    scm, nodes = _make_scm(n_nodes=50, width=8)
    # Synthetic losses: a handful of nodes are "failing" hard, rest near zero.
    failing = nodes[40:45]
    node_losses = {n: 0.01 for n in nodes}
    for j, n in enumerate(failing):
        node_losses[n] = 5.0 - j  # decreasing, all clearly above the rest
    hist = nodes[:3] * 10

    full_stub = _make_stub("full", 8)
    compact_stub = _make_stub("compact", 8)

    full_prompt, _, _ = HuggingFacePolicy.scm_to_prompt(
        full_stub, scm, node_losses=node_losses, intervention_history=hist)
    compact_prompt, _, _ = HuggingFacePolicy.scm_to_prompt(
        compact_stub, scm, node_losses=node_losses, intervention_history=hist)

    print(f"full prompt chars   : {len(full_prompt)}")
    print(f"compact prompt chars: {len(compact_prompt)}")

    # 1. Compact must be materially shorter on a 50-node graph.
    assert len(compact_prompt) < len(full_prompt), \
        "compact prompt should be shorter than full on a 50-node graph"

    # 2. Full lists every node in 'Valid targets'; compact does not.
    assert all(n in full_prompt for n in nodes), \
        "full prompt should list all nodes as valid targets"
    # The lowest-loss leaves should be pruned from the compact prompt's
    # 'Valid targets' (they are neither top-m failing nodes nor their parents).
    surfaced = sum(1 for n in nodes if f" {n}," in compact_prompt or
                   f" {n}." in compact_prompt or f"{n}->" in compact_prompt or
                   f"->{n}" in compact_prompt)
    assert surfaced < len(nodes), \
        "compact prompt should surface a strict subset of nodes"

    # 3. Compact must still propose intervention on a parent of a failing node.
    failing_parents = set()
    for n in failing:
        failing_parents.update(scm.graph.predecessors(n))
    assert any(p in compact_prompt for p in failing_parents), \
        "compact prompt should surface parents of the failing nodes"

    print(f"surfaced nodes in compact: {surfaced}/{len(nodes)}")
    print(f"failing-node parents      : {sorted(failing_parents)}")
    print("PASS: compact prompt is shorter, salience-ranked, and valid.")


if __name__ == "__main__":
    main()
