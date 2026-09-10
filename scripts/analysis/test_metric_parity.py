#!/usr/bin/env python3
"""Parity tests for the Sept 2026 metric audit.

1. The frozen-coefficient ``LargeScaleSCM(n, coeff_seed=s)`` must generate
   bit-identical samples to the ``_LargeGroundTruthSCM`` adapter that
   ``ace_experiments.py --large_scale`` builds at seed ``s`` (same graph, same
   coefficients, same noise), so a baseline and an ACE run at one seed are on
   one system.
2. ``baselines.evaluate_mechanisms_broadrange`` must equal
   ``ace_experiments.ScientificCritic.evaluate_mechanisms_detailed`` on the
   same student, at both scales, so the ``ace_*`` columns the baseline runners
   now log are ACE's metric and not an approximation of it.

Run from the repo root:  python scripts/analysis/test_metric_parity.py
Exit status is non-zero on any failure.
"""
from __future__ import annotations

import random
import re
import sys
import textwrap

import numpy as np
import torch

sys.path.insert(0, ".")
import ace_experiments as A  # noqa: E402
import baselines as B  # noqa: E402
from experiments.large_scale_scm import LargeScaleSCM  # noqa: E402

FAILS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILS
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    FAILS += 0 if ok else 1


def build_ace_adapter(n_nodes: int, seed: int):
    """Reconstruct ace_experiments.main()'s --large_scale system exactly."""
    src = open("ace_experiments.py").read()
    i = src.find("        class _LargeGroundTruthSCM(CausalModel):")
    j = src.find("        M_star = _LargeGroundTruthSCM(", i)
    cls_src = textwrap.dedent(src[i:j])
    ns = {"CausalModel": A.CausalModel, "torch": torch, "np": np}
    exec(cls_src, ns)  # noqa: S102 - trusted repo source
    # same construction order as main(): seed, build graph, then re-seed and draw coeffs
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    lscm = LargeScaleSCM(n_nodes, coeff_seed=seed)
    edges = [(p, node) for node, parents in lscm.graph.items() for p in parents]
    np.random.seed(seed)
    coeffs = {node: {p: float(np.random.uniform(0.3, 0.7)) for p in parents}
              for node, parents in lscm.graph.items()}
    node_idx = dict(lscm.node_idx)
    return ns["_LargeGroundTruthSCM"](edges, coeffs, node_idx), lscm, coeffs


def test_system_parity(n_nodes: int, seed: int) -> None:
    print(f"\n[1] system parity, N={n_nodes}, seed={seed}")
    adapter, lscm, coeffs = build_ace_adapter(n_nodes, seed)
    check("coefficients identical", lscm.coeffs == coeffs)
    # mechanism-level: same parent data + same torch seed -> bit-identical, every node
    ctx = {n: torch.randn(2000) for n in lscm.nodes}
    bad = []
    for n in lscm.nodes:
        if n not in adapter.nodes:
            continue  # the adapter drops isolated nodes
        p_data = {p: ctx[p] for p in lscm.get_parents(n)}
        torch.manual_seed(9); ya = adapter.mechanisms(p_data, n, n_samples=2000)
        torch.manual_seed(9); yb = lscm.mechanisms(p_data, n, n_samples=2000)
        if not torch.equal(ya, yb):
            bad.append(n)
    check(f"mechanisms() bit-identical on all {len(adapter.nodes)} shared nodes", not bad, f"mismatch {bad[:4]}" if bad else "")
    # sample-level: generate() walks a different topological order (networkx vs
    # LargeScaleSCM's own), so compare distributions, not bits
    node = next(n for n in lscm.nodes if len(lscm.get_parents(n)) >= 2)
    iv = {p: 1.5 for p in lscm.get_parents(node)}
    torch.manual_seed(0); a = adapter.generate(200_000, interventions=iv)
    torch.manual_seed(1); b = lscm.generate(200_000, interventions=iv)
    shared = [n for n in lscm.nodes if n in a]
    worst = max(abs(float(a[n].mean() - b[n].mean())) for n in shared)
    worst_sd = max(abs(float(a[n].std() - b[n].std())) for n in shared)
    check(f"generate() same distribution on {len(shared)} shared nodes under do({iv})", worst < 0.02 and worst_sd < 0.02, f"max |dmean|={worst:.4f} max |dsd|={worst_sd:.4f}")
    # and stationary: two calls, same torch seed, same mean
    torch.manual_seed(1); m1 = float(lscm.generate(4000, interventions=iv)[node].mean())
    torch.manual_seed(1); m2 = float(lscm.generate(4000, interventions=iv)[node].mean())
    check("frozen coefficients are stationary across calls", m1 == m2, f"E[{node}]={m1:.4f}")


def test_evaluator_parity_5node() -> None:
    print("\n[2a] evaluator parity, 5-node")
    torch.manual_seed(3)
    oB = B.GroundTruthSCM(); oA = A.GroundTruthSCM()
    student = B.StudentSCM(oB)
    # train a few steps so the student is not at init
    learner = B.SCMLearner(student, oracle=oB)
    for _ in range(5):
        learner.train_step(oB.generate(100, interventions={"X1": 2.0}), intervened="X1")
    critB = B.ScientificCritic(oB)
    critA = A.ScientificCritic(oA); critA.val_data = critB.val_data  # same root targets
    torch.manual_seed(11); tA, nA = critA.evaluate_mechanisms_detailed(student)
    torch.manual_seed(11); tB, nB = B.evaluate_mechanisms_broadrange(oB, student, critB.val_data)
    check("weighted totals equal", abs(tA - tB) < 1e-9, f"{tA:.6f} vs {tB:.6f}")
    check("per-node equal", all(abs(nA[k] - nB[k]) < 1e-9 for k in nA) and set(nA) == set(nB))
    tO, _ = critB.evaluate(student)
    check("observational total differs from broad-range total (sanity: they are different metrics)", abs(tO - tA) > 1e-6, f"obs={tO:.4f} broad={tA:.4f}")


def test_evaluator_parity_30node(seed: int = 42) -> None:
    print(f"\n[2b] evaluator parity, 30-node, seed={seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    lscm = LargeScaleSCM(30, coeff_seed=seed)
    oracle = B.InstrumentedOracle(lscm)
    student = B.StudentSCM(lscm)
    learner = B.SCMLearner(student, oracle=oracle)
    for _ in range(3):
        learner.train_step(oracle.generate(50, interventions={"X1": 1.0}), intervened="X1")
    critB = B.ScientificCritic(oracle)
    critA = A.ScientificCritic(lscm); critA.val_data = critB.val_data
    torch.manual_seed(5); tA, nA = critA.evaluate_mechanisms_detailed(student)
    torch.manual_seed(5); tB, nB = critB.evaluate_broadrange(student)
    check("weighted totals equal (through InstrumentedOracle)", abs(tA - tB) < 1e-9, f"{tA:.6f} vs {tB:.6f}")
    check("per-node equal", all(abs(nA[k] - nB[k]) < 1e-9 for k in nA) and set(nA) == set(nB))
    roots = [n for n in lscm.nodes if not lscm.get_parents(n)]
    check("root weighting reproduces total from per-node", abs(sum((0.2 if k in roots else 1.0) * v for k, v in nB.items()) - tB) < 1e-9)


if __name__ == "__main__":
    test_system_parity(30, 42)
    test_system_parity(30, 2024)
    test_system_parity(15, 123)
    test_evaluator_parity_5node()
    test_evaluator_parity_30node(42)
    print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILURE(S)'}")
    sys.exit(1 if FAILS else 0)
