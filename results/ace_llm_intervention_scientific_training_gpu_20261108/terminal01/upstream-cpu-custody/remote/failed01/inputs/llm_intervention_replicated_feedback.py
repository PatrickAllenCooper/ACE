"""Pure, task-bound aggregation of declared training or artificial feedback.

This module executes no actions and authenticates neither paid responses nor
independent branch/scoring draws, world separation, or an honest task. A runner
must establish those facts. Matching a task digest establishes byte identity
under the public task's canonical JSON convention, not scientific provenance.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics

from llm_intervention_contract import Action, canonical, digest


VERSION = "ace-replicated-training-feedback-v1"


def _number(value, name, *, positive=False, nonnegative=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if (not valid or (positive and value <= 0) or (nonnegative and value < 0)):
        raise ValueError(name + " must be a finite real number with the declared sign")
    return value


def _ids(value, name, minimum):
    if (type(value) not in (list, tuple) or len(value) < minimum
            or any(type(x) is not str or not x for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(name + " must contain distinct nonempty string IDs")
    return list(value)


def _matrix(value, rows, columns, name):
    if type(value) not in (list, tuple) or len(value) != rows:
        raise ValueError(name + " must have exactly one row per replicate")
    result = []
    for row in value:
        if type(row) not in (list, tuple) or len(row) != columns:
            raise ValueError(name + " must have exactly one column per task query")
        result.append([_number(x, name) for x in row])
    return result


def _task_binding(task, task_sha256, target, scale, query_action_ids, query_weights):
    # Accept PredictionTask.payload(), without constructing a graph or learner.
    fields = {"target", "queries", "weights", "scale", "normalized_mse_tolerance"}
    if type(task) is not dict or set(task) != fields:
        raise ValueError("exact public prediction-task payload required")
    if type(task["target"]) is not str or not task["target"]:
        raise ValueError("task target must be a nonempty string")
    _number(task["scale"], "task scale", positive=True)
    _number(task["normalized_mse_tolerance"], "task tolerance", positive=True)
    if type(task["queries"]) is not list or not 1 <= len(task["queries"]) <= 64:
        raise ValueError("one to 64 public query actions required")
    task_ids = []
    for query in task["queries"]:
        if (type(query) is not dict
                or set(query) != {"action_id", "kind", "targets", "measure", "cost_units"}
                or type(query["targets"]) is not list or type(query["measure"]) is not list
                or any(type(x) is not str or not x for x in query["measure"])):
            raise ValueError("invalid public query action payload")
        for pair in query["targets"]:
            if (type(pair) is not list or len(pair) != 2
                    or type(pair[0]) is not str or not pair[0]):
                raise ValueError("invalid query intervention target")
            _number(pair[1], "query intervention value")
        Action(query["action_id"], query["kind"], tuple(tuple(x) for x in query["targets"]),
               tuple(query["measure"]), query["cost_units"])
        task_ids.append(query["action_id"])
    _ids(task_ids, "task query IDs", 1)
    if type(task["weights"]) is not list or len(task["weights"]) != len(task_ids):
        raise ValueError("one fixed weight per task query required")
    for weight in task["weights"]:
        _number(weight, "task query weight", positive=True)
    # Same sum tolerance as PredictionTask.validate; never renormalize weights.
    if any(w > 1 for w in task["weights"]) or abs(math.fsum(task["weights"])-1.) > 1e-12:
        raise ValueError("task query weights must sum to one")
    ids = _ids(query_action_ids, "query action IDs", 1)
    if type(query_weights) not in (list, tuple) or len(query_weights) != len(ids):
        raise ValueError("one supplied weight per query required")
    for weight in query_weights:
        _number(weight, "query weight", positive=True)
    _number(scale, "scale", positive=True)
    if (type(target) is not str or target != task["target"] or scale != task["scale"]
            or ids != task_ids or list(query_weights) != task["weights"]):
        raise ValueError("target, scale, ordered query IDs and exact weights must match the task")
    if (type(task_sha256) is not str or len(task_sha256) != 64
            or any(c not in "0123456789abcdef" for c in task_sha256)
            or digest(task) != task_sha256):
        raise ValueError("task digest differs from the declared pin")
    return ids, list(query_weights), json.loads(canonical(task))


def aggregate_replicated_feedback(*, scope, task, task_sha256, target, scale,
        query_action_ids, query_weights, candidate_ids, candidate_costs, replicate_ids,
        base_predictions, branch_predictions, scoring_truths, tie_multiplier, minimum_gain_gap):
    """Aggregate fixed-query gains, using independent replicate blocks as units.

    ``task`` is the full public PredictionTask.payload(). IDs and weights must
    match its query order exactly; no sorting, normalization or broadcasting is
    performed. ``candidate_costs`` and ``branch_predictions`` are dictionaries
    with exactly the candidate IDs. The latter holds replicate-by-query matrices;
    base predictions and shared scoring truths each have that same matrix shape.
    Replicate IDs declare the common row alignment across all candidates.

    For candidate c, replicate r, query q, g[c,r,q] =
    (((base[r,q]-truth[r,q])/scale)**2 -
     ((branch[c,r,q]-truth[r,q])/scale)**2) / cost[c].
    First compute G[c,r] = sum_q weight[q]*g[c,r,q], then mean_r G[c,r].
    For every candidate pair, SE = sample_std(G[left]-G[right], ddof=1)/sqrt(R).
    A tie is abs(mean_difference) <= minimum_gain_gap + tie_multiplier*SE.
    This fixed rule is a heuristic, not an inference certificate or stop rule.

    Inputs are copied, never mutated. Only finite JSON-style numeric arrays are
    accepted, with R >= 2. Costs are declared positive numbers; authenticity of
    their charges and independence of joint branch/scoring blocks are external
    obligations. Even successful aggregation returns provenance_qualified=False.
    """
    if type(scope) is not str or scope not in ("policy_train", "artificial"):
        raise ValueError("feedback aggregation is restricted to policy_train or artificial scope")
    queries, weights, task_copy = _task_binding(
        task, task_sha256, target, scale, query_action_ids, query_weights)
    candidates = _ids(candidate_ids, "candidate IDs", 2)
    replicates = _ids(replicate_ids, "replicate IDs", 2)
    if (type(candidate_costs) is not dict or set(candidate_costs) != set(candidates)
            or type(branch_predictions) is not dict or set(branch_predictions) != set(candidates)):
        raise ValueError("costs and branch predictions must cover exactly all candidate IDs")
    costs = {c: _number(candidate_costs[c], "candidate cost", positive=True) for c in candidates}
    _number(tie_multiplier, "tie multiplier", nonnegative=True)
    _number(minimum_gain_gap, "minimum gain gap", nonnegative=True)
    nr, nq = len(replicates), len(queries)
    base = _matrix(base_predictions, nr, nq, "base predictions")
    truths = _matrix(scoring_truths, nr, nq, "scoring truths")
    branches = {c: _matrix(branch_predictions[c], nr, nq, "branch " + c) for c in candidates}

    def checked(value):
        return _number(value, "derived feedback")

    def losses(predictions):
        return [[checked(((p-t)/scale)**2) for p, t in zip(row, truth)]
                for row, truth in zip(predictions, truths)]

    try:
        base_losses = losses(base)
        branch_losses, reductions, gains, replicate_gains, means = {}, {}, {}, {}, {}
        for c in candidates:
            branch_losses[c] = losses(branches[c])
            reductions[c] = [[checked(a-b) for a, b in zip(left, right)]
                             for left, right in zip(base_losses, branch_losses[c])]
            gains[c] = [[checked(x/costs[c]) for x in row] for row in reductions[c]]
            replicate_gains[c] = [checked(math.fsum(w*g for w, g in zip(weights, row)))
                                  for row in gains[c]]
            means[c] = checked(math.fsum(replicate_gains[c])/nr)
        pairs = []
        for left, right in itertools.combinations(candidates, 2):
            differences = [checked(a-b) for a, b in zip(replicate_gains[left], replicate_gains[right])]
            mean = checked(math.fsum(differences)/nr)
            se = checked(statistics.stdev(differences)/math.sqrt(nr))
            threshold = checked(minimum_gain_gap + tie_multiplier*se)
            tie = abs(mean) <= threshold
            pairs.append({"left": left, "right": right, "replicate_differences": differences,
                "mean_gain_difference": mean, "paired_standard_error": se,
                "tie_threshold": threshold, "tie": tie,
                "chosen": None if tie else (left if mean > 0 else right),
                "rejected": None if tie else (right if mean > 0 else left)})
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("nonfinite derived feedback") from exc

    return {"version": VERSION, "status": "complete", "scope": scope,
        "task": task_copy, "task_sha256": task_sha256, "target": target, "scale": scale,
        "query_action_ids": queries, "query_weights": weights,
        "candidate_ids": candidates, "candidate_costs": costs, "replicate_ids": replicates,
        "replicate_count": nr, "base_predictions": base, "scoring_truths": truths,
        "branch_predictions": branches, "base_losses": base_losses, "branch_losses": branch_losses,
        "normalized_loss_reductions": reductions, "gains_per_cost": gains,
        "per_replicate_gains_per_cost": replicate_gains, "mean_gains_per_cost": means, "pairs": pairs,
        "tie_multiplier": tie_multiplier, "minimum_gain_gap": minimum_gain_gap,
        "standard_error_ddof": 1, "standard_error_unit": "joint_branch_scoring_replicate_block",
        "aggregation_order": "fixed_query_weights_within_replicate_then_equal_replicate_mean",
        "tie_rule": "abs(mean_gain_difference) <= minimum_gain_gap + tie_multiplier * paired_standard_error",
        "tie_rule_scope": "fixed heuristic, not an inference certificate",
        "stop_inference_performed": False, "provenance_qualified": False,
        "provenance_note": "Task identity and declared array alignment only. Paid execution, response truth, "
            "independent branch/scoring replicate blocks and world separation are not authenticated."}
