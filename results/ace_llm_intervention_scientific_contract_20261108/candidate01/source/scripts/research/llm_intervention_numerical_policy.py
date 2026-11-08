"""Public-state numerical controls for artificial episode qualification.

Uniform random and least-used menu coverage are distinct policies. The third
control estimates one-step information about eligible local regression heads
per response cost. It is a reference teacher, not a claim to be the strongest
numerical policy for final prediction or operating decisions.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from llm_intervention_contract import Action, Node, canonical, finite, integer, policy_prompt, validate_design
from llm_intervention_scm import Head, LinearGaussianSCM


def restore(public_state):
    policy_prompt(public_state)
    graph = tuple(Node(n["name"], tuple(n["parents"]), n["unit"],
                       tuple(n["intervention_bounds"]) if n["intervention_bounds"] is not None else None)
                  for n in public_state["graph"])
    menu = tuple(Action(a["action_id"], a["kind"], tuple(tuple(x) for x in a["targets"]),
                        tuple(a["measure"]), a["cost_units"]) for a in public_state["menu"])
    validate_design(graph, menu)
    summaries = {s["node"]: s for s in public_state["scm"]}
    if len(summaries) != len(public_state["scm"]) or set(summaries) != {n.name for n in graph}:
        raise ValueError("one fitted summary per mechanism required")
    learner = LinearGaussianSCM(graph, {n.name: summaries[n.name]["noise_variance"] for n in graph})
    for node in graph:
        summary = summaries[node.name]
        d = 1 + len(node.parents)
        mean = np.array(summary["parameter_mean"], dtype=float)
        cov = np.array(summary["parameter_covariance"], dtype=float)
        if (summary["terms"] != ["intercept", *node.parents] or mean.shape != (d,)
                or cov.shape != (d, d) or not np.isfinite(mean).all() or not np.isfinite(cov).all()
                or not np.allclose(cov, cov.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(cov).min() < 0 or not integer(summary["eligible_rows"])):
            raise ValueError("invalid public posterior")
        learner.heads[node.name] = Head(mean, cov, summary["noise_variance"], summary["eligible_rows"])
    return graph, menu, learner


def choose(public_state, *, method, seed, draws=16):
    if method not in ("uniform_random", "menu_coverage", "local_information"):
        raise ValueError("unknown numerical policy")
    if not integer(seed) or not integer(draws, 1) or draws > 256:
        raise ValueError("invalid fixed planning RNG or draw count")
    graph, menu, learner = restore(public_state)
    if not integer(public_state["remaining_units"]) or type(public_state["terminal"]) is not bool:
        raise ValueError("invalid budget/terminal state")
    feasible = sorted((a for a in menu if a.cost_units <= public_state["remaining_units"]),
                      key=lambda a: a.action_id)
    if public_state["terminal"] or not feasible:
        return {"stop": True}, {"planning_draws": 0}
    rng = np.random.default_rng(seed)
    if method == "uniform_random":
        action = feasible[int(rng.integers(len(feasible)))]
        return {"action_id": action.action_id}, {"planning_draws": 0}
    if method == "menu_coverage":
        counts = {a.action_id: sum(r["action_id"] == a.action_id for r in public_state["history"])
                  for a in feasible}
        action = min(feasible, key=lambda a: (counts[a.action_id], a.action_id))
        return {"action_id": action.action_id}, {"planning_draws": 0, "counts": counts}
    scores = {}
    for action in feasible:
        samples = []
        for _ in range(draws):
            predicted = learner.simulate(action, rng)
            available = set(action.measure) | {name for name, value in action.targets}
            targets = {name for name, value in action.targets}
            information = 0.
            for node in graph:
                if node.name in targets or node.name not in action.measure or not set(node.parents) <= available:
                    continue
                head = learner.heads[node.name]
                phi = np.array([1.] + [predicted[p] for p in node.parents])
                variance = float(phi @ head.covariance @ phi)
                information += .5 * np.log1p(max(variance, 0.) / head.noise_variance)
            samples.append(information / action.cost_units)
        scores[action.action_id] = float(np.mean(samples))
    if not all(map(finite, scores.values())):
        raise ValueError("nonfinite information score")
    best = min(feasible, key=lambda a: (-scores[a.action_id], a.action_id))
    return {"action_id": best.action_id}, {"planning_draws": len(feasible)*draws, "scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("uniform_random", "menu_coverage", "local_information"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draws", type=int, default=16)
    args = parser.parse_args()
    state = json.load(sys.stdin)
    # A fresh, decision-indexed stream avoids restarting the same random choice
    # at every subprocess call. Final seed namespaces belong to a frozen runner.
    if not integer(state["requests"]):
        raise ValueError("invalid request count")
    decision_seed = np.random.SeedSequence([args.seed, state["requests"]]).generate_state(2)
    seed = int(decision_seed[0]) + (int(decision_seed[1]) << 32)
    decision, metadata = choose(state, method=args.method, seed=seed, draws=args.draws)
    print(canonical(metadata), file=sys.stderr)
    print(canonical(decision))


if __name__ == "__main__":
    main()
