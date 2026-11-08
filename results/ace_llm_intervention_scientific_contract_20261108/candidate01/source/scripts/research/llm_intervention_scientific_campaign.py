"""Pure assembler for the existing 24-world development proposal, not a freeze.

All worlds and launch/service bindings are supplied by the caller. No world or
response is drawn, no policy/model is loaded, and nothing is written or executed.
The returned campaign contains private descriptors: it is not a policy envelope.
Pins authenticate supplied declarations against independent caller expectations;
they do not authenticate filesystem bytes, checkpoint roles, namespace admission,
runtime, scheduler, scientific provenance or permission to execute.
"""
from __future__ import annotations

import json

from llm_intervention_campaign import prepare_campaign
from llm_intervention_contract import canonical, digest
from llm_intervention_paired_episode import prepare_plan
from llm_intervention_persistent_policy import _identity
from llm_intervention_screen_policy import PLANNING_DRAWS, bind_policy
from llm_intervention_screen_protocol import validate_proposal
from llm_intervention_task import PredictionTask, action_from_dict
from llm_intervention_world import WorldDescriptor


VERSION = "ace-supplied-development-campaign-20261108-v1"
NUMERIC_POLICIES = ("uniform_random", "menu_coverage", "local_information", "target_risk")
LLM_POLICIES = ("frozen_llm", "sft_llm", "dpo_llm")
POLICY_IDS = NUMERIC_POLICIES + ("learned_ranker",) + LLM_POLICIES
# Execution IDs differ from the original proposal's arm labels. Never rederive
# a namespace using an execution alias: preserve the proposal's seed law.
PROPOSAL_ARMS = NUMERIC_POLICIES + ("numerical_ranker", "llm_frozen", "llm_sft", "llm_dpo")
STRATA_FIELDS = ("regime", "graph_stratum", "cost_stratum")
SERVICE_REQUESTS = 24 * 24


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _mapping(value, keys, label):
    _need(type(value) is dict and set(value) == set(keys), "exact " + label + " required")


def _proposal(proposal, proposal_sha256, expected_source_pins):
    _need(_pin(proposal_sha256), "independent proposal semantic pin required")
    observed = validate_proposal(proposal, expected_source_pins=expected_source_pins)
    _need(observed == proposal_sha256, "proposal pin differs")
    # Detach once before consuming any caller-owned declarations.
    return json.loads(canonical(proposal))


def _bindings(proposal):
    rows = []
    for world in proposal["development_worlds"]:
        public = world["public"]
        for episode in world["development_episodes"][:4]:
            binding = bind_policy(
                seed_prefix=episode["policy_seed_prefix"], scope=world["scope"],
                world_id=world["world_id"],
                request_namespace=episode["measurement_binding"]["request_namespace"],
                arm=episode["arm"], role="behavior", initial_actions=public["initial_actions"],
                initial_cost_units=public["initial_cost_units"], total_budget=public["total_budget"],
                planning_draws=PLANNING_DRAWS)
            rows.append({"world_id": world["world_id"], "policy_id": episode["arm"],
                         "binding": binding, "binding_sha256": digest(binding)})
    return rows


def numeric_policy_bindings(*, proposal, proposal_sha256, expected_source_pins):
    """96 detached bindings; append their exact four-argument suffix to caller argv.

    Suffix: ["--binding-json", canonical(binding), "--binding-sha256", pin].
    This helper authenticates the unchanged prospective proposal, without draws.
    """
    return _bindings(_proposal(proposal, proposal_sha256, expected_source_pins))


def _specs(specs, bindings):
    _need(type(specs) in (tuple, list) and len(specs) == 8
          and all(type(s) is dict for s in specs)
          and [s.get("policy_id") for s in specs] == list(POLICY_IDS),
          "exact ordered eight-policy roster required for each world")
    for spec in specs:
        pid = spec["policy_id"]
        _need(spec.get("kind") == ("persistent" if pid in LLM_POLICIES else "task_process"),
              "five task-process and three persistent policies required")
        command = spec.get("command")
        _need(type(command) is list and 1 <= len(command) <= 64
              and all(type(s) is str and 0 < len(s) <= 4096 and "\0" not in s for s in command),
              "bounded caller-owned argv without NUL required")
        if pid in NUMERIC_POLICIES:
            row = bindings[pid]
            suffix = ["--binding-json", canonical(row["binding"]),
                      "--binding-sha256", row["binding_sha256"]]
            _need(len(command) > 4 and command[-4:] == suffix
                  and not any(s.startswith("--binding-") for s in command[:-4]),
                  "numerical command must bind this exact world and arm once")
    # Existing prepare_plan enforces the remaining exact spec schema and pins.
    return tuple(specs)


def build_scientific_campaign(*, proposal, proposal_sha256, expected_source_pins,
                              worlds_by_id, expected_world_pins, policy_specs_by_world,
                              services, expected_service_identity_pins,
                              outer_provenance_receipt_sha256):
    """Assemble the existing 24 development worlds x 8 policies, with no execution.

    worlds_by_id: exact integer-ID -> WorldDescriptor map for the proposal's
    development roster. expected_world_pins is a separate exact map of semantic
    descriptor pins. Graphs and known noises must match public design; supplied
    realized coefficients are preserved, never drawn or checked against outcomes.

    policy_specs_by_world uses prepare_plan's ordered six-field policy specs and
    the execution IDs in POLICY_IDS. services uses prepare_campaign's exact
    {identity,startup_seconds,cleanup_seconds,max_requests} schema. Each of the
    three services declares exactly 576 possible acquisition requests. The
    separate expected_service_identity_pins map pins the complete existing
    identity schema, including model, adapter/checkpoint, source and recipe.
    Actual checkpoint-role/byte authentication remains the caller's obligation.

    Each arm pays eight initial cost units with shared keys, then at most 24
    acquisition units with arm-specific keys. CPU/GPU phases are declarations,
    not allocations. Return the existing campaign + semantic campaign/source-plan
    pins, explicit proposal/source/world/service bindings and qualification flags.
    Existing preparation constructs fresh zero-prior SCMs only for validation;
    it does not fit or update them. No output path or execution hook is accepted.
    """
    p = _proposal(proposal, proposal_sha256, expected_source_pins)
    rows = p["development_worlds"]
    ids = [r["world_id"] for r in rows]
    for value, label in ((worlds_by_id, "supplied worlds"), (expected_world_pins, "world pins"),
                         (policy_specs_by_world, "policy specifications")):
        _mapping(value, ids, label)
        _need(all(type(wid) is int for wid in value), "integer development identities required")
    _mapping(services, LLM_POLICIES, "three services")
    _mapping(expected_service_identity_pins, LLM_POLICIES, "three independent service identity pins")
    for pid in LLM_POLICIES:
        settings = services[pid]
        _mapping(settings, ("identity", "startup_seconds", "cleanup_seconds", "max_requests"), "service fields")
        _need(type(settings["max_requests"]) is int and settings["max_requests"] == SERVICE_REQUESTS,
              "exact 576-request bound per persistent service required")
        _need(_pin(expected_service_identity_pins[pid])
              and digest(_identity(settings["identity"])) == expected_service_identity_pins[pid],
              "independent service identity/checkpoint declaration differs")
    # Authenticate the entire supplied roster before assembling any plan.
    for row in rows:
        wid = row["world_id"]
        world = worlds_by_id[wid]
        _need(type(world) is WorldDescriptor and type(world.world_id) is int
              and world.scope == "development" and world.world_id == wid,
              "supplied world scope or identity differs")
        _need(_pin(expected_world_pins[wid]) and world.sha256 == expected_world_pins[wid],
              "independent supplied world pin differs")
        _need(canonical(world.public_design()["graph"]) == canonical(row["public"]["graph"]),
              "supplied graph differs from prospective public design")
        _need(canonical({m.node: m.noise_variance for m in world.mechanisms})
              == canonical(row["public"]["learner"]["noise_variances"]),
              "supplied known noise differs from public learner design")

    bindings = _bindings(p)
    indexed = {(r["world_id"], r["policy_id"]): r for r in bindings}
    sources = []
    for row in rows:
        wid, public = row["world_id"], row["public"]
        world = worlds_by_id[wid]
        t = public["task"]
        task = PredictionTask(t["target"], tuple(action_from_dict(q) for q in t["queries"]),
                              tuple(t["weights"]), t["scale"], t["normalized_mse_tolerance"])
        episodes = row["development_episodes"]
        _need([e["arm"] for e in episodes] == list(PROPOSAL_ARMS), "proposal arm mapping differs")
        def binding(value):
            return {k: value[k] for k in ("request_namespace", "replicate_id")}
        plan = prepare_plan(
            world=world, world_sha256=expected_world_pins[wid], task=task,
            menu=tuple(action_from_dict(a) for a in public["menu"]),
            noise_variances=public["learner"]["noise_variances"],
            prior_variance=public["learner"]["prior_variance"],
            initial_actions=tuple(public["initial_actions"]), acquisition_cap_units=public["additional_cost_units"],
            initial_binding=binding(row["initial_binding"]),
            measurement_bindings={pid: binding(e["measurement_binding"]) for pid, e in zip(POLICY_IDS, episodes)},
            policy_specs=_specs(policy_specs_by_world[wid], {pid: indexed[wid, pid] for pid in NUMERIC_POLICIES}),
            objective=public["objective"], domain_notes=public["domain_notes"],
            outer_provenance_receipt_sha256=outer_provenance_receipt_sha256)
        sources.append({"plan": plan, "plan_sha256": digest(plan),
                        "strata": {field: row[field] for field in STRATA_FIELDS}})
    campaign = prepare_campaign(
        sources=sources, strata_fields=list(STRATA_FIELDS), services=services,
        policy_phases={pid: "gpu" if pid in LLM_POLICIES else "cpu" for pid in POLICY_IDS})
    return json.loads(canonical({
        "version": VERSION, "scope": "development", "proposal_sha256": proposal_sha256,
        "source_pins": p["source_pins"], "campaign": campaign, "campaign_sha256": digest(campaign),
        "source_plan_pins": [{"world_id": s["plan"]["world"]["world_id"], "sha256": s["plan_sha256"]}
                             for s in sources],
        "world_pins": [{"world_id": wid, "sha256": expected_world_pins[wid]} for wid in ids],
        "service_identity_pins": expected_service_identity_pins,
        "proposal_arm_mapping": dict(zip(POLICY_IDS, PROPOSAL_ARMS)),
        "numeric_policy_bindings": bindings,
        "worlds_generated": 0, "responses_generated": 0, "execution_authorized": False,
        "scientific_protocol_frozen": False, "scientific_provenance_qualified": False,
        "source_qualified": False, "runtime_qualified": False, "scheduler_qualified": False}))
