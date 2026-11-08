"""Pure construction of the two supplied artificial paired-campaign plans.

Call numeric_policy_bindings() first to assemble caller-owned commands. Then
build_artificial_campaign() freezes those exact commands, identities and declared
provenance through the accepted preparation APIs. Nothing is written or executed.
The returned bundle is private driver material, not a policy-facing envelope.

Pins are semantic SHA256 digests, not artifact byte hashes or evidence of source,
checkpoint-role, namespace or runtime admission. The outer driver owns those
checks, the stable launch shim, execution and private reporting.
"""
from __future__ import annotations

import json

from llm_intervention_campaign import prepare_campaign
from llm_intervention_contract import Action, Node, canonical, digest
from llm_intervention_paired_episode import prepare_plan
from llm_intervention_screen_policy import PLANNING_DRAWS, PROPOSAL_VERSION, bind_policy
from llm_intervention_task import PredictionTask
from llm_intervention_world import LinearMechanism, WorldDescriptor


VERSION = "ace-artificial-paired-campaign-20261108-v1"
WORLD_IDS = (9101, 9102)
NUMERIC_POLICIES = ("uniform_random", "menu_coverage", "local_information", "target_risk")
LLM_POLICIES = ("frozen_llm", "sft_llm", "dpo_llm")
POLICY_IDS = NUMERIC_POLICIES + ("learned_ranker",) + LLM_POLICIES
INITIAL_ACTIONS = ("observe_all", "observe_all")
ACQUISITION_CAP_UNITS = 4
DOMAIN_NOTES = (
    "Synthetic dimensionless variables with a known graph and hard interventions. "
    "All nodes are measured after each action. The learner uses public known "
    "noise variances and zero-mean independent Gaussian parameter priors. "
    "The task tolerance is a goal, not evidence of attainment."
)


def _namespace(world_id, arm=None):
    return digest({"protocol": PROPOSAL_VERSION, "scope": "artificial",
                   "world_id": world_id, "purpose": "trajectory", "arm": arm})


def numeric_policy_bindings():
    """Return eight detached {world_id, policy_id, binding, binding_sha256} rows.

    Each numerical command must end with ["--binding-json", canonical(binding),
    "--binding-sha256", binding_sha256]. Its executable/shim prefix is supplied
    entirely by the caller. No policy is called and no seed is drawn here.
    """
    rows = []
    for world_id in WORLD_IDS:
        for policy_id in NUMERIC_POLICIES:
            namespace = _namespace(world_id, policy_id)
            binding = bind_policy(
                seed_prefix=["artificial", world_id, "policy", namespace, "behavior"],
                scope="artificial", world_id=world_id, request_namespace=namespace,
                arm=policy_id, role="behavior", initial_actions=INITIAL_ACTIONS,
                initial_cost_units=2, total_budget=6, planning_draws=PLANNING_DRAWS)
            rows.append({"world_id": world_id, "policy_id": policy_id,
                         "binding": binding, "binding_sha256": digest(binding)})
    return rows


def _specs(specs, bindings):
    if (type(specs) not in (tuple, list) or len(specs) != len(POLICY_IDS)
            or any(type(s) is not dict for s in specs)
            or [s.get("policy_id") for s in specs] != list(POLICY_IDS)):
        raise ValueError("exact ordered eight-policy roster required for each world")
    for spec in specs:
        pid = spec["policy_id"]
        kind = "persistent" if pid in LLM_POLICIES else "task_process"
        if spec.get("kind") != kind:
            raise ValueError("five CPU task_process policies and three persistent LLM policies required")
        command = spec.get("command")
        if (type(command) is not list or not 1 <= len(command) <= 64
                or any(type(s) is not str or not s or len(s) > 4096 or "\0" in s for s in command)):
            raise ValueError("bounded caller-owned argv without NUL required")
        if pid in NUMERIC_POLICIES:
            row = bindings[pid]
            suffix = ["--binding-json", canonical(row["binding"]),
                      "--binding-sha256", row["binding_sha256"]]
            if (len(command) <= len(suffix) or command[-4:] != suffix
                    or any(s.startswith("--binding-") for s in command[:-4])):
                raise ValueError("numerical command must bind this world's exact screen-policy binding once")
    # prepare_plan validates the remaining exact schema, timeout and pin fields.
    return tuple(specs)


def _world(world_id):
    graph = (Node("X", (), "dimensionless", (-2., 2.)),
             Node("Y", ("X",), "dimensionless"))
    if world_id == 9101:
        mechanisms = (LinearMechanism("X", 0., (), .25),
                      LinearMechanism("Y", 1., (("X", 2.),), .25))
        means = [-1., 3.]
    else:
        graph += (Node("Z", ("Y",), "dimensionless"),)
        mechanisms = (LinearMechanism("X", 0., (), .25),
                      LinearMechanism("Y", -.5, (("X", 1.),), .25),
                      LinearMechanism("Z", .25, (("Y", -2.),), .25))
        means = [3.25, -.75]
    return WorldDescriptor("artificial", world_id, graph, mechanisms), means


def build_artificial_campaign(*, policy_specs_by_world, services,
                              outer_provenance_receipt_sha256):
    """Prepare the fixed 9101/9102 design without responses, fitting or I/O.

    policy_specs_by_world is an exact integer-keyed {9101: specs, 9102: specs}
    mapping. Each list/tuple is the ordered POLICY_IDS roster, using prepare_plan's
    exact policy-spec schema. No command, identity or provenance pin is invented.
    Task-process commands may differ by world; persistent specs must be identical.

    services uses prepare_campaign's exact settings schema for the three LLM IDs:
    {identity, startup_seconds, cleanup_seconds, max_requests}. Each service must
    declare exactly eight requests, covering four decisions in each world. The
    caller supplies the shared outer provenance receipt pin before preparation.

    Returns campaign, campaign_sha256, ordered source_plan_pins, numerical binding
    rows and private_query_means (analytical declarations, never evaluated scores).
    The campaign carries only a descriptive case stratum: graph and costs vary
    together. Every arm separately pays two initial responses and at most four
    acquisition units: 16 cells, at most 96 responses/units and 24 LLM decisions.
    The accepted APIs detach the plans; retain their pins before any execution.
    """
    if (type(policy_specs_by_world) is not dict
            or any(type(wid) is not int for wid in policy_specs_by_world)
            or set(policy_specs_by_world) != set(WORLD_IDS)):
        raise ValueError("exact artificial world IDs 9101 and 9102 required")
    if (type(services) is not dict or set(services) != set(LLM_POLICIES)
            or any(type(s) is not dict or type(s.get("max_requests")) is not int
                   or s["max_requests"] != 8 for s in services.values())):
        raise ValueError("exact three-service roster with eight requests per service required")

    bindings = numeric_policy_bindings()
    sources, private_means = [], []
    for world_id in WORLD_IDS:
        world, means = _world(world_id)
        target = world.graph[-1].name
        measurements = tuple(n.name for n in world.graph)
        menu = (Action("observe_all", "observe", (), measurements, 1),
                Action("x_minus", "intervene", (("X", -1.),), measurements, 1),
                Action("x_plus", "intervene", (("X", 1.),), measurements,
                       1 if world_id == 9101 else 2))
        queries = tuple(Action("q_" + a.action_id, "intervene", a.targets, (target,), 1)
                        for a in menu[1:])
        task = PredictionTask(target, queries, (.5, .5), 2., .04)
        specs = _specs(policy_specs_by_world[world_id],
                       {r["policy_id"]: r for r in bindings if r["world_id"] == world_id})
        plan = prepare_plan(
            world=world, world_sha256=world.sha256, task=task, menu=menu,
            noise_variances={n.name: .25 for n in world.graph}, prior_variance=1.,
            initial_actions=INITIAL_ACTIONS, acquisition_cap_units=ACQUISITION_CAP_UNITS,
            initial_binding={"request_namespace": _namespace(world_id), "replicate_id": "trajectory"},
            measurement_bindings={pid: {"request_namespace": _namespace(world_id, pid),
                                        "replicate_id": "trajectory"} for pid in POLICY_IDS},
            policy_specs=specs,
            objective=f"Predict the noise-disabled structural mean of {target} under the supplied public query distribution.",
            domain_notes=DOMAIN_NOTES,
            outer_provenance_receipt_sha256=outer_provenance_receipt_sha256)
        sources.append({"plan": plan, "plan_sha256": digest(plan),
                        "strata": {"case": "two_node_uniform" if world_id == 9101 else "three_node_heterogeneous"}})
        private_means.append({"world_id": world_id, "world_sha256": world.sha256,
                              "task_sha256": digest(task.payload()), "target": target,
                              "query_ids": [q.action_id for q in queries], "means": means})
    campaign = prepare_campaign(
        sources=sources, strata_fields=["case"],
        policy_phases={pid: "gpu" if pid in LLM_POLICIES else "cpu" for pid in POLICY_IDS},
        services=services)
    return json.loads(canonical({
        "version": VERSION, "campaign": campaign, "campaign_sha256": digest(campaign),
        "source_plan_pins": [{"world_id": s["plan"]["world"]["world_id"], "sha256": s["plan_sha256"]}
                             for s in sources],
        "numeric_policy_bindings": bindings, "private_query_means": private_means}))
