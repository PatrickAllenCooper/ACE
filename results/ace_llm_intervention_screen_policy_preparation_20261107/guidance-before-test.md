# Public numerical policies with prospective decision keys

This new component connects the four numerical controls to `TaskProcessPolicy`
using the prospective screen's exact policy RNG rule. It leaves the historical
numerical CLI and all original sources and fixtures unchanged. It does not admit
scientific worlds or change the separate CPU runtime work.

The implementation is `scripts/research/llm_intervention_screen_policy.py`. It
provides the pure `bind_policy`, `validate_binding` and `decision_rng` helpers,
plus `choose_public` and an executable CLI. The controls are `uniform_random`,
`menu_coverage`, `local_information` and `target_risk`. Planning draws are fixed
at 32 as declared in `ace-first-screen-proposal-20261107-v1`. The random and
coverage controls do not perform 32 fantasies. Their original algorithms remain
unchanged. The information and risk controls receive 32 draws per feasible action.
Planning fantasies use the public working SCM and are not environment responses.

## Binding and exact seed

The trusted caller supplies a fixed `seed_prefix`, `scope`, `world_id`,
`request_namespace`, `arm`, `role`, `initial_actions`, `initial_cost_units`,
`total_budget` and `planning_draws` to `bind_policy`. It returns a detached JSON
binding. The declared request namespace must equal:

```python
digest({"protocol": PROPOSAL_VERSION, "scope": scope, "world_id": world_id,
        "purpose": "trajectory", "arm": arm})
```

The prefix must be exactly `[scope, world_id, "policy", request_namespace,
"behavior"]`. The arm is already represented inside the namespace digest. It is
not appended again. For decision index `i`, the seed is exactly:

```python
keyed_seed(scope, world_id, "policy", request_namespace, "behavior", i)
```

This is the big-endian integer of SHA256 over canonical JSON:

```text
["ace-linear-gaussian-world-v1", scope, world_id, "policy",
 [request_namespace, "behavior", i]]
```

There is no truncation, additional seed mixing or global RNG mutation. The old
CLI's `SeedSequence([seed, state["requests"]])` remains untouched and must not be
used as a substitute for this interface in the prospective comparison.

`development` bindings accept only the proposal's 24 declared IDs, with eight
`observe_all` initial actions costing eight units and a total budget of 32.
The `artificial` scope allows smaller explicit schedules for fixtures. Neither
form generates a world. Range and schema checks are not registry admission,
execution authorization or proof of source/runtime authentication.

## Decision count, initial history and costs

The helper validates the full existing public task envelope before selection.
Requests, returned responses and history length must agree. Receipt IDs must be
contiguous. Every receipt must match a legal action, its clamps, measurements and
cost, and the public eligible-row counts must agree with the history.

It then verifies the fixed initial action prefix and its declared cost. Spent plus
remaining cost must equal the declared total budget. Only then does it derive:

```python
decision_index = state["requests"] - len(binding["initial_actions"])
```

The first call after eight shared initial responses has index zero. After one
two-unit intervention the next index is one, although nine responses have been
requested and ten cost units spent. Initial values remain public paid history.
They are not incorporated into the policy key or duplicated in the binding.

This equality relies on the existing normal episode loop: one accepted response
per successful acquisition call, no retry after a failed or ambiguous call, and
no later call after stopping. Terminal inputs, incomplete histories and inputs
with no affordable intervention are rejected. The executor owns budget stopping.
The helper can additionally check an explicit `expected_decision_index` against
the derived index when used in process. A stateless CLI cannot detect an unlogged
retry of an identical input. The outer supervisor must reconcile actual call
indices, process receipts and executor journals.

## Executable public transport

For a validated binding `b` with an independently recorded `binding_pin`, the
existing interface can be constructed without changing `TaskProcessPolicy`:

```python
command = (qualified_python, "-B", absolute_new_script_path,
           "--binding-json", canonical(b), "--binding-sha256", binding_pin)
policy = TaskProcessPolicy(command, declared_timeout_seconds, public_task)
```

The CLI receives only `{version, state, task}` JSON on stdin. It writes one
decision JSON object on stdout. Stderr contains a single receipt with the exact
canonical key, full seed in decimal and 64-digit hex, binding and input hashes,
arm, decision index, request count, cost counters, fixed draw setting, decision
hash and original planning metadata. Decimal seeds are strings so downstream
JavaScript cannot silently round the 256-bit integer. `ProcessPolicy` already
retains command, stdin digest, stdout, stderr and exit status.

The binding has no private mechanism parameters, realized outcomes or private
scores. Its scope/world identifier is control metadata used only to address the
policy RNG. Do not add this metadata or the numerical receipt to LLM prompt text.
The complete private proposal and paired plan must stay in the trusted driver.
The interface is not an OS sandbox against a malicious process with the same
filesystem access. Public prose must also be supplied honestly by the driver.

Duplicate JSON keys, nonfinite constants, altered bindings, unexpected fields,
oversized input and schema failures produce nonzero exit with no decision on
stdout. There is no fallback to the old CLI or a different arm. Budget failure is
not converted into an apparently successful policy stop.

## Training compatibility

This version rejects `policy_train`, `confirmation`, teacher roles and snapshot
roles. It only implements artificial/development behavior calls. The existing
training assembler already uses the same world rule for behavior indices, but
its namespace is the shared training trajectory namespace. Teacher calls use
`teacher, snapshot_index`, which is distinct from the number of acquired
responses and from the acquisition cost milestone. This CLI does not infer a
teacher index from either one. An outer training integration must retain or
explicitly bind the existing behavior and teacher conventions separately.

## Verification and remaining integration

Only the new `test_llm_intervention_screen_policy.py` suite is scheduled, once,
under the existing local test interpreter from an unrelated working directory.
It uses hand-written one-node public history and posterior fixtures. No environment
callback, scientific world realization, pretrained model, target runtime import,
remote action or submission is authorized or performed by these tests. Numerical
working-model planning and its temporary conjugate updates are expected.

The fixtures cover the independent canonical SHA256 oracle, initial-history
offset, heterogeneous costs, all four real CLI subprocesses through
`TaskProcessPolicy`, algorithm agreement with the existing pure controls,
namespace separation, deterministic replay, unchanged inputs/global RNG,
zero-initial and budget-terminal cases, invalid bindings/public inputs and CLI
failure retention. A pure metadata comparison checks the 96 declared numerical
development bindings against the current proposal without realizing those IDs.
Its placeholder source pins do not qualify the original scientific source.

The single invocation and exact source snapshots will be retained under the new
exclusive preparation capsule in `results` and its external counterpart. No older
suite or fixture is rerun. Independent-agent tools are unavailable in this
session. Static self-review and automated fixtures are not an independent review.

Before scientific integration the outer driver must authenticate the new source
closure, runtime and command, bind the public graph/menu/task/prior/prose to the
adopted proposal, and pin this binding in each paired policy specification. It
must match scope/world/arm and measurement namespace across the binding and the
paired plan, preserve the distinct shared initial namespace, verify actual call
counts against these receipts, and retain all failures. Existing source pins do
not cover the new file. Whole-episode accounting/sealing, independent review,
multiworld supervision, scientific admission, source/resource freeze and any
training/teacher integration remain outside this component. No LLM efficiency
result or target-runtime qualification follows from this preparation.
