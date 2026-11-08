"""Shared, outcome-free prediction task for every intervention policy.

The envelope adds a public query distribution to the existing charged state.
It does not alter the frozen GPU qualification or executor. A trusted runner
must still supply honest public text and keep private evaluators out of the
policy process. Schema validation is not a sandbox against malicious code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

from llm_intervention_contract import Action, Node, canonical, digest, finite, integer, policy_prompt, validate_design
from llm_intervention_episode import ProcessPolicy, run_episode, save
from llm_intervention_numerical_policy import restore


VERSION = "ace-public-prediction-task-v1"
MAX_NODES, MAX_ACTIONS, MAX_QUERIES, MAX_HISTORY = 8, 32, 64, 80


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("unexpected public object fields")


def action_from_dict(value):
    keys(value, ("action_id", "kind", "targets", "measure", "cost_units"))
    if type(value["targets"]) is not list or type(value["measure"]) is not list:
        raise ValueError("public action arrays required")
    return Action(value["action_id"], value["kind"], tuple(tuple(x) for x in value["targets"]),
                  tuple(value["measure"]), value["cost_units"])


@dataclass(frozen=True)
class PredictionTask:
    target: str
    queries: tuple[Action, ...]
    weights: tuple[float, ...]
    scale: float
    normalized_mse_tolerance: float

    def validate(self, graph):
        if (type(self.target) is not str or self.target not in {n.name for n in graph}
                or type(self.queries) is not tuple or not 1 <= len(self.queries) <= MAX_QUERIES
                or type(self.weights) is not tuple or len(self.weights) != len(self.queries)
                or not all(finite(w) and w > 0 for w in self.weights)
                or abs(sum(self.weights)-1.) > 1e-12
                or not finite(self.scale) or self.scale <= 0
                or not finite(self.normalized_mse_tolerance) or self.normalized_mse_tolerance <= 0):
            raise ValueError("invalid declared prediction task")
        validate_design(graph, self.queries)
        return self

    def payload(self):
        return json.loads(canonical(asdict(self)))

    def risk_arguments(self):
        return {name: getattr(self, name) for name in ("target", "queries", "weights", "scale")}


def validate_state(state):
    """Check the full structured public allowlist before crossing a process boundary."""
    policy_prompt(state)
    if (not isinstance(state["objective"], str) or not state["objective"]
            or not isinstance(state["domain_notes"], str)
            or type(state["terminal"]) is not bool):
        raise ValueError("invalid public text or terminal flag")
    if any(type(state[name]) is not list for name in ("graph", "menu", "scm", "history")):
        raise ValueError("public arrays required")
    if (not 1 <= len(state["graph"]) <= MAX_NODES
            or not 1 <= len(state["menu"]) <= MAX_ACTIONS
            or len(state["history"]) > MAX_HISTORY):
        raise ValueError("public state exceeds fixed representation bounds")
    for n in state["graph"]:
        keys(n, ("name", "parents", "unit", "intervention_bounds"))
        if (type(n["parents"]) is not list or not all(type(p) is str for p in n["parents"])
                or (n["intervention_bounds"] is not None and type(n["intervention_bounds"]) is not list)):
            raise ValueError("invalid public graph arrays")
    menu = tuple(action_from_dict(a) for a in state["menu"])
    for s in state["scm"]:
        keys(s, ("node", "terms", "parameter_mean", "parameter_covariance", "noise_variance", "eligible_rows"))
        if (type(s["terms"]) is not list or type(s["parameter_mean"]) is not list
                or type(s["parameter_covariance"]) is not list
                or not all(finite(v) for v in s["parameter_mean"])
                or any(type(row) is not list or not all(finite(v) for v in row)
                       for row in s["parameter_covariance"])):
            raise ValueError("invalid public mechanism arrays")
    graph, _, learner = restore(state)
    if any(not integer(state[k]) for k in ("spent_units", "remaining_units", "requests", "returned_responses")):
        raise ValueError("invalid public counters")
    # Policies are called only between completed charged responses. Failed or
    # ambiguous attempts terminate at the driver and cannot become new input.
    if state["requests"] != state["returned_responses"] or state["requests"] != len(state["history"]):
        raise ValueError("incomplete or inconsistent public response history")
    by_id = {a.action_id: a for a in menu}
    spent = 0
    eligible = {n.name: 0 for n in graph}
    for request, r in enumerate(state["history"]):
        keys(r, ("request_id", "action_id", "kind", "targets", "values", "cost_units"))
        if r["action_id"] not in by_id or type(r["request_id"]) is not int or r["request_id"] != request:
            raise ValueError("inconsistent public receipt identity")
        a = by_id[r["action_id"]]
        if (r["kind"] != a.kind or type(r["targets"]) is not dict
                or not all(finite(v) for v in r["targets"].values())
                or r["targets"] != dict(a.targets) or not integer(r["cost_units"], 1)
                or r["cost_units"] != a.cost_units
                or type(r["values"]) is not dict or set(r["values"]) != set(a.measure)
                or not all(finite(v) for v in r["values"].values())
                or any(r["values"][k] != v for k, v in a.targets if k in r["values"])):
            raise ValueError("public receipt does not match its charged action")
        spent += a.cost_units
        available = set(a.measure) | set(dict(a.targets))
        for n in graph:
            if n.name not in dict(a.targets) and n.name in a.measure and set(n.parents) <= available:
                eligible[n.name] += 1
    if spent != state["spent_units"] or any(learner.heads[n].eligible_rows != count for n, count in eligible.items()):
        raise ValueError("public history accounting or eligible-row count differs")
    return graph, menu, learner


def make_envelope(state, task: PredictionTask):
    graph, _, _ = validate_state(state)
    task.validate(graph)
    return json.loads(canonical({"version": VERSION, "state": state, "task": task.payload()}))


def unpack_envelope(envelope):
    keys(envelope, ("version", "state", "task"))
    if envelope["version"] != VERSION:
        raise ValueError("unsupported public task version")
    graph, menu, learner = validate_state(envelope["state"])
    t = envelope["task"]
    keys(t, ("target", "queries", "weights", "scale", "normalized_mse_tolerance"))
    if type(t["queries"]) is not list or type(t["weights"]) is not list:
        raise ValueError("public query arrays required")
    task = PredictionTask(t["target"], tuple(action_from_dict(q) for q in t["queries"]),
                          tuple(t["weights"]), t["scale"], t["normalized_mse_tolerance"]).validate(graph)
    return envelope["state"], task, graph, menu, learner


def task_prompt(envelope):
    unpack_envelope(envelope)
    return ("Choose the next experimental intervention using this public SCM, paid history and prediction task. "
            "The task queries and weights describe where accurate predictions are useful. They contain no observed answers. "
            "Predict the noise-disabled structural response. Error is squared error divided by scale squared. "
            "The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. "
            "Every environment response costs the listed units. Return exactly one JSON object containing an affordable "
            "action_id from state.menu, or {\"stop\":true}. Do not invent measurements or actions.\n"
            + canonical(envelope) + "\nDecision:\n")


class TaskProcessPolicy(ProcessPolicy):
    """Use the existing charged episode with the same task envelope for every arm."""
    def __init__(self, command, timeout_seconds, task):
        super().__init__(command, timeout_seconds)
        if not isinstance(task, PredictionTask):
            raise ValueError("immutable prediction task required")
        self.task = task

    def choose(self, public_state, directory, decision_index):
        return super().choose(make_envelope(public_state, self.task), directory, decision_index)


def run_task_episode(*, task, command, timeout_seconds, output, **episode):
    """Bind the task even when initialization consumes the whole budget."""
    graph, menu = episode["graph"], episode["menu"]
    validate_design(graph, menu)
    task.validate(graph)
    if not 1 <= len(graph) <= MAX_NODES or not 1 <= len(menu) <= MAX_ACTIONS:
        raise ValueError("episode exceeds fixed graph/menu bounds")
    initial, budget = episode["initial_actions"], episode["budget"]
    actions = {a.action_id: a for a in menu}
    if type(initial) is not tuple or not set(initial) <= set(actions) or not integer(budget, 1):
        raise ValueError("invalid initial schedule or budget")
    initial_cost = sum(actions[name].cost_units for name in initial)
    if initial_cost > budget:
        raise ValueError("initial schedule exceeds budget")
    maximum_responses = len(initial)+(budget-initial_cost)//min(a.cost_units for a in menu)
    if maximum_responses > MAX_HISTORY:
        raise ValueError("episode can exceed the fixed history bound")
    policy = TaskProcessPolicy(command, timeout_seconds, task)
    output.mkdir(exist_ok=False)
    save(output/"public_task.json", {"version": VERSION, "task": task.payload(),
        "task_sha256": digest(task.payload()), "policy_command": command,
        "policy_timeout_seconds": timeout_seconds, "maximum_possible_responses": maximum_responses})
    return run_episode(policy=policy, output=output/"episode", **episode)


def numerical_features(envelope):
    """Complete bounded numeric state plus one candidate descriptor per row.

    Retains full coefficient covariances, adjacency, intervention/measurement
    masks, paid history and query law. Textual names, units and domain prose are
    omitted. Node positions use the supplied topological order. This flattened
    control is not permutation invariant across alternate graph orderings.
    No target-risk scores, reward labels or environment truths are features.
    """
    state, task, graph, menu, learner = unpack_envelope(envelope)
    index = {n.name: i for i, n in enumerate(graph)}
    action_index = {a.action_id: i for i, a in enumerate(menu)}
    def encode_action(action):
        v = np.zeros(3*MAX_NODES+3)
        v[0] = 1.
        v[1] = action.kind == "intervene"
        v[2] = action.cost_units
        for name, value in action.targets:
            v[3+index[name]] = 1.
            v[3+MAX_NODES+index[name]] = value
        for name in action.measure:
            v[3+2*MAX_NODES+index[name]] = 1.
        return v
    nodes = np.zeros((MAX_NODES, 6+MAX_NODES+(MAX_NODES+1)+(MAX_NODES+1)**2))
    for i, node in enumerate(graph):
        head = learner.heads[node.name]
        nodes[i, :3] = [1., head.noise_variance, head.eligible_rows]
        if node.intervention_bounds is not None:
            nodes[i, 3:6] = [1., *node.intervention_bounds]
        for p in node.parents:
            nodes[i, 6+index[p]] = 1.
        coordinates = [0] + [1+index[p] for p in node.parents]
        mean = np.zeros(MAX_NODES+1)
        cov = np.zeros((MAX_NODES+1, MAX_NODES+1))
        mean[coordinates] = head.mean
        cov[np.ix_(coordinates, coordinates)] = head.covariance
        nodes[i, 6+MAX_NODES:] = np.r_[mean, cov.ravel()]
    menu_values = np.zeros((MAX_ACTIONS, 3*MAX_NODES+3))
    for i, a in enumerate(menu):
        menu_values[i] = encode_action(a)
    history = np.zeros((MAX_HISTORY, 4*MAX_NODES+MAX_ACTIONS+3))
    for i, r in enumerate(state["history"]):
        history[i, :3] = [1., r["request_id"], r["cost_units"]]
        history[i, 3+action_index[r["action_id"]]] = 1.
        offset = 3+MAX_ACTIONS
        for name, value in r["targets"].items():
            history[i, offset+index[name]] = 1.
            history[i, offset+MAX_NODES+index[name]] = value
        for name, value in r["values"].items():
            history[i, offset+2*MAX_NODES+index[name]] = 1.
            history[i, offset+3*MAX_NODES+index[name]] = value
    queries = np.zeros((MAX_QUERIES, 3*MAX_NODES+4))
    for i, (q, weight) in enumerate(zip(task.queries, task.weights)):
        queries[i] = np.r_[encode_action(q), weight]
    target = np.zeros(MAX_NODES)
    target[index[task.target]] = 1.
    shared = np.r_[state["spent_units"], state["remaining_units"], state["requests"],
                   state["returned_responses"], int(state["terminal"]), task.scale,
                   task.normalized_mse_tolerance, target, nodes.ravel(), menu_values.ravel(),
                   history.ravel(), queries.ravel()]
    feasible = sorted((a for a in menu if not state["terminal"] and a.cost_units <= state["remaining_units"]),
                      key=lambda a: a.action_id)
    ids = tuple(a.action_id for a in feasible) + (None,)
    descriptors = [np.r_[0., encode_action(a)] for a in feasible] + [np.r_[1., np.zeros(3*MAX_NODES+3)]]
    values = np.stack([np.r_[shared, descriptor] for descriptor in descriptors])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite numeric policy feature")
    # Fixed, invertible signed-log transform. No statistics from other worlds.
    values = np.sign(values)*np.log1p(np.abs(values))
    return ids, values.astype(np.float64), {"task_sha256": digest(task.payload()),
        "envelope_sha256": digest(envelope), "feature_width": values.shape[1],
        "encoding": "complete-padded-numeric-state-signed-log-v1"}
