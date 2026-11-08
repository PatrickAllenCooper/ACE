"""Pure, prospective first-screen schedule. No draws, responses or execution.

The JSON proposal is inspectable and hash-bindable, but is not a scientific
freeze, namespace admission, source authentication or authorization to collect.
Only public_design() exports policy-facing design; the full plan is private.
"""
from __future__ import annotations

from dataclasses import asdict
import json

from llm_intervention_contract import Action, Node, canonical, digest
from llm_intervention_task import (
    MAX_ACTIONS, MAX_HISTORY, MAX_NODES, MAX_QUERIES, PredictionTask, action_from_dict,
)
from llm_intervention_training_dataset import (
    MAX_FEEDBACK_RESPONSES, MAX_REPLICATES, MAX_SNAPSHOTS, METHODS, SOURCE_FILES, Snapshot,
)
from llm_intervention_world import VERSION as WORLD_VERSION, MAX_REQUESTS


VERSION = "ace-first-screen-proposal-20261107-v1"
TRAINING_START, DEVELOPMENT_START = 202611070000, 202611070100
CONFIRMATION_START, CONFIRMATION_STOP = 202611071000, 202611072000
REQUIRED_SOURCE_PATHS = tuple(sorted({"scripts/research/"+name for name in SOURCE_FILES}
    | {"scripts/research/llm_intervention_screen_protocol.py"}))
REGIMES = (("matched_low", 0., .25), ("matched_high", 0., 1.),
           ("shifted_low", 1.5, .25), ("shifted_high", 1.5, 1.))
GRAPHS = ("chain", "fork", "collider")
COSTS = ("uniform", "heterogeneous")
ARMS = METHODS + ("numerical_ranker", "llm_frozen", "llm_sft", "llm_dpo")
MILESTONES = (0, 6, 12, 18, 24)
REPLICATES = ("r0", "r1", "r2", "r3")
PENDING = (
    "Independent scientific review/adoption of prospective graph allocation, laws, menu/query law, scale, ties and planning draws.",
    "Admit these namespace ranges against prior registries; separately authorize realization and pin actual descriptors. No replacement worlds.",
    "Authenticate admitted source bytes, CPU/model runtimes, models/tokenizers/adapters and complete parent service qualification.",
    "Specify matched numerical/language optimization exposure, architecture, SFT/DPO schedules and separate fixed post-SFT references; no allocation yet.",
    "Integrate common initial response keys, separately charged arm journals, public-only transport and full failure accounting with no fallback.",
    "Finalize private evaluator implementation, capped-failure reporting, contrasts/multiplicity, development selection rule and later powered confirmation design.",
    "Measure preparation/inference demand and separately authorize resource limits, deadlines and scientific collection/training/evaluation.",
)


def _copy(value):
    return json.loads(canonical(value))


def _pins(pins):
    if (type(pins) is not dict or set(pins) != set(REQUIRED_SOURCE_PATHS)
            or any(type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for v in pins.values())):
        raise ValueError("exact dependency-closed lowercase SHA256 map required")
    return dict(sorted(pins.items()))


def _graph(kind):
    parents = {"chain": ((), ("X",), ("Y",)), "fork": ((), ("X",), ("X",)),
               "collider": ((), (), ("X", "Y"))}[kind]
    return tuple(Node(name, p, "dimensionless", (-2., 2.) if name != "Z" else None)
                 for name, p in zip(("X", "Y", "Z"), parents))


def _menu(cost):
    targets = ((), (("X", -1.),), (("X", 0.),), (("X", 1.),),
               (("Y", -1.),), (("Y", 0.),), (("Y", 1.),), (("X", -1.), ("Y", 1.)))
    names = ("observe_all", "set_x_neg", "set_x_zero", "set_x_pos",
             "set_y_neg", "set_y_zero", "set_y_pos", "set_xy")
    return tuple(Action(name, "intervene" if t else "observe", t, ("X", "Y", "Z"),
                        2 if cost == "heterogeneous" and i >= 4 else 1)
                 for i, (name, t) in enumerate(zip(names, targets)))


def _public(graph_kind, cost, noise):
    graph, menu = _graph(graph_kind), _menu(cost)
    queries = tuple(Action(f"q{i}", a.kind, a.targets, ("Z",), 1) for i, a in enumerate(menu))
    task = PredictionTask("Z", queries, (.125,)*8, 2., .04).validate(graph)
    return _copy({"graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "task": task.payload(), "task_sha256": digest(task.payload()),
        "learner": {"noise_variances": {n.name: noise for n in graph}, "prior_variance": 1.,
            "parameter_mean": "zero for every intercept and edge",
            "parameter_covariance": "identity within each independent local head"},
        "objective": "Predict the noise-disabled structural mean of Z under the supplied public query distribution.",
        "domain_notes": "Synthetic dimensionless variables. The supplied graph and hard interventions are known. "
            "The learner uses public known noise variances and zero-mean independent Gaussian parameter priors. "
            "The task tolerance is a goal, not evidence of attainment.",
        "initial_actions": ["observe_all"]*8, "initial_response_count": 8,
        "initial_cost_units": 8, "additional_cost_units": 24, "total_budget": 32})


def _binding(scope, wid, stream, namespace, replicate):
    return {"scope": scope, "world_id": wid, "stream": stream,
            "request_namespace": namespace, "replicate_id": replicate}


def _namespace(scope, wid, purpose, arm=None):
    return digest({"protocol": VERSION, "scope": scope, "world_id": wid, "purpose": purpose, "arm": arm})


def _row(scope, index, regime_index, graph, cost, pair_index):
    rid, coefficient_mean, noise = REGIMES[regime_index]
    wid = (TRAINING_START if scope == "policy_train" else DEVELOPMENT_START) + index
    public = _public(graph, cost, noise)
    namespace = _namespace(scope, wid, "trajectory")
    initial = _binding(scope, wid, "initial", namespace, "trajectory")
    nodes = _graph(graph)
    law = {"family": "independent Gaussian linear parameters; stationary DAG",
        "intercept_means": [0.]*3, "coefficient_means": [[coefficient_mean]*len(n.parents) for n in nodes],
        "intercept_sd": 1., "coefficient_sd": 1., "noise_variances": [noise]*3,
        "noise_is_known_public": True, "realized_parameters": None, "descriptor_sha256": None,
        "parameter_key_addresses": [[scope, wid, "mechanism", n.name, "intercept"] for n in nodes]
            + [[scope, wid, "mechanism", n.name, "parent", p] for n in nodes for p in n.parents],
        "prior_relation": "exact constructor prior" if coefficient_mean == 0. else "edge means shifted; intercept law unchanged"}
    row = {"scope": scope, "world_id": wid, "identity": f"{scope}:{wid}", "regime": rid,
        "graph_stratum": graph, "cost_stratum": cost, "stratum_pair": f"{scope}:{rid}:pair-{pair_index}",
        "cost_pair_semantics": "matched graph/law strata, distinct world draws; each world is paired across policy arms",
        "public": public, "truth_law": law, "generated": False,
        "initial_binding": initial, "initial_request_ids": list(range(8)),
        "private_evaluation": {"binding": _binding(scope, wid, "private_eval",
                _namespace(scope, wid, "private_evaluation"), "private_evaluation"),
            "law": "deterministic noise-disabled structural means on the public task queries",
            "task_sha256": public["task_sha256"], "used_for_policy_decisions": False,
            "execution_status": "pending_parent_integration", "executed": False}}
    if scope == "policy_train":
        ci = COSTS.index(cost)
        behavior = (pair_index + ci + regime_index) % 4
        teacher_base = (pair_index + 2*ci + 2*regime_index) % 4
        candidates = [a["action_id"] for a in public["menu"]]
        snapshots = []
        for si, threshold in enumerate(MILESTONES):
            block = digest({"namespace": namespace, "snapshot_index": si})
            snapshots.append({"acquisition_cost_units": threshold, "teacher_method": METHODS[(teacher_base+si)%4],
                "candidates": candidates if si < 4 else [],
                "counterfactual_bindings": [_binding(scope, wid, "measurement", block, r) for r in REPLICATES] if si < 4 else [],
                "score_bindings": [_binding(scope, wid, "train_score", block, r) for r in REPLICATES] if si < 4 else []})
        row["training"] = {"behavior_method": METHODS[behavior], "request_namespace": namespace,
            "measurement_binding": _binding(scope, wid, "measurement", namespace, "trajectory"),
            "snapshots": snapshots, "replicate_ids": list(REPLICATES), "structural_scores": True,
            "score_scope": "charged privileged training structural labels; never learner updates or private evaluation",
            "planning_draws": 32, "tie_multiplier": 2., "minimum_gain_gap": 0.,
            "tie_rule_scope": "fixed replicate-SE heuristic, not an inference certificate",
            "expected_branch_responses": 128, "expected_scoring_responses": 128,
            "expected_branch_cost_units": 16*sum(a["cost_units"] for a in public["menu"]),
            "expected_scoring_cost_units": 128}
        row["development_episodes"] = []
    else:
        row["training"] = None
        row["development_episodes"] = [{"arm": arm, "initial_binding": _copy(initial),
            "measurement_binding": _binding(scope, wid, "measurement", _namespace(scope, wid, "trajectory", arm), "trajectory"),
            "policy_seed_prefix": [scope, wid, "policy", _namespace(scope, wid, "trajectory", arm), "behavior"],
            "additional_cost_units": 24, "total_budget": 32, "weights_frozen_during_episode": True,
            "training_feedback_allowed": False} for arm in ARMS]
    return row


def _proposal(pins):
    training, development = [], []
    # Four paired graph/cost blocks per regime. Extra graph rotates prospectively;
    # 32 cannot be divisible by three, so the exact graph imbalance is declared.
    extra_graphs = ("chain", "fork", "collider", "chain")
    for ri in range(4):
        for pi, graph in enumerate(GRAPHS + (extra_graphs[ri],)):
            for cost in COSTS:
                training.append(_row("policy_train", len(training), ri, graph, cost, pi))
        for pi, graph in enumerate(GRAPHS):
            for cost in COSTS:
                development.append(_row("development", len(development), ri, graph, cost, pi))
    if not (3 <= MAX_NODES and 8 <= MAX_ACTIONS and 8 <= MAX_QUERIES and 32 <= MAX_HISTORY
            and 32 <= MAX_REQUESTS and 5 <= MAX_SNAPSHOTS and 4 <= MAX_REPLICATES
            and 256 <= MAX_FEEDBACK_RESPONSES and len(METHODS) == 4):
        raise ValueError("proposal exceeds admitted task/world/assembly bounds")
    return _copy({"version": VERSION, "status": "prospective_unqualified_proposal",
        "scientific_protocol_frozen": False, "scientific_execution_authorized": False,
        "provenance_qualified": False, "worlds_generated": 0, "responses_generated": 0,
        "outcomes_consulted": False, "source_pins": pins,
        "source_admission": "Exact declared dependency map only; caller must authenticate actual bytes/runtime before execution.",
        "regimes": [{"id": r, "edge_mean": m, "known_noise_variance": n} for r, m, n in REGIMES],
        "training_worlds": training, "development_worlds": development,
        "confirmation_reservation": {"scope": "confirmation", "world_id_start_inclusive": CONFIRMATION_START,
            "world_id_stop_exclusive": CONFIRMATION_STOP, "identities": [], "generated": False,
            "status": "proposed_reserved_namespace_not_registry_admission", "sample_size": None,
            "families_and_laws": None, "execution_authorized": False},
        "balance": {"training_worlds": 32, "development_worlds": 24, "development_arms": list(ARMS),
            "training_per_regime": 8, "development_per_regime": 6,
            "training_per_cost": 16, "development_per_cost": 12,
            "training_per_graph": {"chain": 12, "fork": 10, "collider": 10},
            "development_per_graph": {g: 8 for g in GRAPHS}, "training_per_behavior": 8,
            "training_per_teacher_at_each_snapshot": 8, "training_per_behavior_teacher_at_each_snapshot": 2,
            "training_supervised_records": 160, "training_nonterminal_pair_records_including_ties": 3584},
        "budget_basis": {"initial_responses": 8, "initial_cost_units": 8, "additional_cost_units": 24,
            "total_trajectory_budget": 32, "maximum_trajectory_responses": 32,
            "maximum_action_cost": 2, "milestones": list(MILESTONES),
            "crossing_upper_costs": [0, 7, 13, 19, 24],
            "remaining_units_lower_bounds": [24, 17, 11, 5, 0],
            "minimum_menu_cost": 1, "all_candidates_feasible_at_nonterminal_crossings": True,
            "initial_charging": "Every arm reserves eight initial responses in its own journal; identical keys pair their values, not free responses.",
            "training_maximum_trajectory_cost": 1024, "training_branch_cost": 5120, "training_scoring_cost": 4096,
            "training_total_response_cost_ceiling": 10240, "development_total_trajectory_cost_ceiling": 6144,
            "allocation_status": "unallocated; response cost units are not CPU/GPU seconds"},
        "seed_law": {"world_key_version": WORLD_VERSION,
            "hash_rule": "SHA256(canonical([world_key_version,scope,world_id,stream,key_components])); big-endian integer",
            "parameter_draw_rule": "A fresh local random.Random(keyed_seed).gauss(0,1) per parameter, only after separate authorization.",
            "adapter_key_components": ["request_namespace", "replicate_id", "journal_request_id", "action_id",
                                       "digest({kind:action.kind,targets:action.targets})", "node_name"],
            "initial_keys": "Use each row.initial_binding and request IDs 0..7; no arm in the initial key.",
            "training_policy_keys": "(scope,world_id,policy,training.request_namespace,behavior,decision_index) or (...,teacher,snapshot_index)",
            "training_snapshot_namespace": "digest({namespace:training.request_namespace,snapshot_index:index}); exact assembly convention",
            "development_policy_keys": "Append decision_index to each episode.policy_seed_prefix.",
            "counterfactual_access": "policy_train only, measurement stream with distinct snapshot namespace and replicate key",
            "score_access": "policy_train only, train_score stream; no development/confirmation feedback",
            "private_evaluation": "No noise draw for structural means; private_eval keys reserved for a separately specified noisy evaluator.",
            "world_pairing": "All development arms share the same pinned world and initial keys. Cost-stratum pairs have distinct world identities.",
            "global_rng_mutation": False},
        "optimization_schedule": {"status": "pending_parent_integration", "allocated": False,
            "settings": None, "data_matching": "same shared records/strict pairs; retain ties and all failures",
            "reference_rule": "Each preference-trained family needs its own pinned fixed post-supervised reference."},
        "evaluation_reporting": {"target": "Z", "normalized_mse_tolerance": .04,
            "unit": "world, paired across arms; choices/replicates are not independent worlds",
            "report": ["private first-crossing cost", "attainment", "final error", "full curves", "actual spending", "failures"],
            "nonattainment_and_failure": "retain and assign common within-world cap for capped-cost reporting",
            "autonomous_stopping_claim": False, "implementation_status": "pending_parent_integration"},
        "remaining_required_decisions": list(PENDING)})


def build_proposal(*, source_pins):
    """Return a fresh JSON proposal. Does no I/O, RNG, model or response work."""
    return _proposal(_pins(source_pins))


def validate_proposal(proposal, *, expected_source_pins):
    """Reject extra/missing/modified members, including truths and scope leakage.

    This version is one explicit prospective design, not a tunable experiment
    schema. Changes require a new reviewed proposal version, not silent edits.
    Canonical comparison distinguishes JSON numeric/bool types and rejects NaNs.
    Independent expected pins are required; copying them out of an untrusted plan
    is not authentication. The function performs no filesystem verification.
    """
    expected = _proposal(_pins(expected_source_pins))
    try:
        valid = type(proposal) is dict and canonical(proposal) == canonical(expected)
    except (ValueError, TypeError, OverflowError, RecursionError):
        valid = False
    if not valid:
        raise ValueError("proposal differs from exact prospective schedule or admitted source map")
    return digest(proposal)


def _lookup(proposal, scope, world_id, expected_source_pins):
    validate_proposal(proposal, expected_source_pins=expected_source_pins)
    if type(scope) is not str or scope not in ("policy_train", "development") or type(world_id) is not int:
        raise ValueError("only declared training/development identities have recipes")
    rows = proposal["training_worlds" if scope == "policy_train" else "development_worlds"]
    for row in rows:
        if row["world_id"] == world_id:
            return row
    raise ValueError("identity is outside this proposal")


def public_design(proposal, *, scope, world_id, expected_source_pins):
    """Detached graph/menu/task/prior/text only; no scope, seed, law or truth."""
    return _copy(_lookup(proposal, scope, world_id, expected_source_pins)["public"])


def assembly_recipe(proposal, *, world_id, expected_source_pins):
    """Typed construction inputs, not execution; four runtime bindings are absent.

    No learner, world descriptor or responses are constructed. Parent must bind
    world, independent expected_world_sha256, fresh learner and exclusive output
    after separate scientific authorization and runtime/source authentication.
    """
    row = _lookup(proposal, "policy_train", world_id, expected_source_pins)
    public, training = row["public"], row["training"]
    graph = tuple(Node(n["name"], tuple(n["parents"]), n["unit"],
                       None if n["intervention_bounds"] is None else tuple(n["intervention_bounds"])) for n in public["graph"])
    menu = tuple(action_from_dict(a) for a in public["menu"])
    t = public["task"]
    task = PredictionTask(t["target"], tuple(action_from_dict(q) for q in t["queries"]),
                          tuple(t["weights"]), t["scale"], t["normalized_mse_tolerance"]).validate(graph)
    return {"execution_authorized": False,
        "runtime_bindings_pending": ("world", "expected_world_sha256", "learner", "output"),
        "learner_parameters": {"graph": graph, "noise_variances": dict(public["learner"]["noise_variances"]), "prior_variance": 1.},
        "assembly_parameters": {"scope": "policy_train", "graph": graph, "menu": menu, "task": task,
            "expected_task_sha256": public["task_sha256"], "initial_actions": tuple(public["initial_actions"]),
            "total_budget": 32, "behavior_method": training["behavior_method"],
            "snapshots": tuple(Snapshot(s["acquisition_cost_units"], s["teacher_method"], tuple(s["candidates"])) for s in training["snapshots"]),
            "replicate_ids": tuple(training["replicate_ids"]), "planning_draws": training["planning_draws"],
            "tie_multiplier": training["tie_multiplier"], "minimum_gain_gap": training["minimum_gain_gap"],
            "structural_scores": training["structural_scores"], "objective": public["objective"],
            "domain_notes": public["domain_notes"], "request_namespace": training["request_namespace"]}}
