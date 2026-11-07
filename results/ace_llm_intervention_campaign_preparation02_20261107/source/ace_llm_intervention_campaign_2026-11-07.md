# Supplied-plan multiworld campaign

`llm_intervention_campaign.py` integrates the existing paired runner and complete
roster reporter. It accepts only supplied, independently pinned artificial or
development paired plans. It never generates a world, selects a policy, trains,
allocates resources, retries or substitutes a failed world. The accepted paired,
transport, sealing and reporting components are unchanged.

## Callable workflow

Run with `scripts/research` on Python's import path. The outer controller supplies
`plans`, their previously retained canonical `digest(plan)` pins, ordered strata,
phase assignments and authenticated persistent identities. These are private
controller inputs; only existing public task envelopes reach policy workers.

```python
import llm_intervention_campaign as cc
from llm_intervention_contract import digest

campaign = cc.prepare_campaign(
    sources=[{"plan": p, "plan_sha256": pin, "strata": labels}
             for p, pin, labels in supplied_sources],
    strata_fields=["regime", "cost"],
    policy_phases=declared_policy_phases,  # every policy -> "cpu" or "gpu"
    services=declared_services)
# Freeze/retain these independently before execution; don't derive new expected
# pins from whatever happens to be present when validating later.
campaign_pin = digest(campaign)
source_pins = [{"world_id": s["plan"]["world"]["world_id"],
                "sha256": s["plan_sha256"]} for s in campaign["sources"]]
binding = dict(campaign=campaign, campaign_sha256=campaign_pin,
               source_plan_pins=source_pins, output=exclusive_output)
cc.initialize_campaign(**binding)

# On CPU resources, invoke once per declared CPU policy, persisting each returned
# receipt byte pin outside the mutable attempt directory.
pin = cc.run_policy(**binding, policy_id=cpu_policy_id, phase="cpu")
# Repeat for OTHER CPU policies, never the same policy. Only after all CPU
# inventories are terminal may a separately admitted GPU phase start:
pin = cc.run_policy(**binding, policy_id=gpu_policy_id, phase="gpu")
# Each call runs the complete supplied world roster. Retain every returned pin.

# Once ALL policies have terminal inventories and every service is terminated:
summary = cc.finalize_campaign(**binding, attempt_pins=all_policy_receipt_pins)
```

`all_policy_receipt_pins` is an exact mapping from policy ID to its returned
`sha256`. A policy deliberately never admitted still needs a terminal inventory:
call `run_policy(..., unstarted_reason="explicit reason")` in its declared phase.
This claims that policy attempt permanently and retains every world as unstarted.
The GPU label does not allocate or prove GPU use; the tests use a model-free CPU
worker in that phase to exercise orchestration. No scheduler or shell CLI is
provided; these functions are the executable integration interface.

`sources` contains 1–64 exact `{plan, plan_sha256, strata}` objects. World IDs are
unique and scope is uniform. Every plan must have the same ordered policy roster.
Every policy's kind, timeout, identity and provenance must agree across worlds.
Persistent commands must agree too; task-process commands may contain per-world
seeds. Strata fields and all labels are fixed before any arm executes. Neither
labels nor world selection are inferred from outcomes.

`services` contains exactly the persistent IDs. Each value has `identity`
(the existing persistent transport's five-field attestation), `startup_seconds`,
`cleanup_seconds` and `max_requests`. Identity must match every source-plan pin;
startup/cleanup limits are finite positive values at most 3,600 seconds. Request
capacity must cover the sum of each world's maximum affordable acquisitions and
be at most 100,000. The worker's own transport limit must also accommodate that
roster; no worker command is rewritten. A too-small worker-side limit fails
normally. Task-process policies are CPU-only; persistent policies may be assigned
to either phase. CPU inventories and closure are checked before GPU execution.

## Ownership, evidence and failure

The campaign root and each indexed policy directory are exclusively created.
`attempt.jsonl` durably records the initial full roster, service start, arm intent,
arm end and terminal inventory with timestamps. Arm raw artifacts, service logs
and partial failures stay in place. Each attempt publishes an fsynced candidate
then an exclusive hard-linked `attempt.json`, including all evidence byte pins.
A second call for that policy cannot restart it, even when the first failed.

One persistent service is initialized for each persistent policy attempt, reused
across its worlds, and closed after the last attempted arm. The existing transport
may terminate itself on failure; the controller does not close it twice. Failed
transport leaves subsequent worlds explicitly unstarted. Ordinary arm failures
with a usable service allow later predeclared worlds to proceed. Interruptions
propagate after cleanup and inventory publication where storage is available.
Journal/controller errors also enter cleanup before further evidence writes.

Finalization first checks exact campaign/source-plan pins, all policy attempt
pins, full world inventories and byte custody. It then checks every service's
termination evidence before invoking **any** private reporter. Confirmed failed
termination is distinct from clean success: nonzero exits with a reaped leader,
no surviving group and no cleanup errors permit the campaign gate, but the
existing per-world finalizer makes those persistent arms unresolved. Missing or
uncertain cleanup blocks all private reporting. A never-launched service must
have no attempted arms; failed startup must retain its launch/failure evidence.

Per-world finalization authenticates charged histories and shared initialization.
The campaign feeds its saved report bytes and pins into `summarize_screen` across
the entire world × policy cross-product, restoring the predeclared strata.
Failed/unstarted/unresolved cells remain at each world's acquisition cap; full
initial cost and actual partial charges stay separate. The shared-initial receipt
hash is established by authenticated execution, not invented in advance. Its
schedule and bindings were fixed in the source plans. Finalization itself is an
exclusive once-only attempt and retains partial reports on failure.

An exact intended `finalization.json` remains authoritative if an interrupt arrives
after publication. The interrupt propagates and is retained separately in
`private/post-publication-diagnostic.json`; no terminal failure is then emitted.
Only confirmed absence of the published receipt permits `private/failure.json`.
Unreadable or unexpected published bytes instead produce
`private/publication-uncertain.json` for outer reconciliation. None of these paths
authorizes retry. Focused publication regressions use fabricated reporting inputs
and filesystem faults only, without replaying any world or policy.

## Scope and validation

The new test file uses two hand-written one-node, zero-noise artificial worlds
with IDs 51 and 52, real tiny public subprocesses and model-free persistent
workers. It checks analytical cost crossings, full strata/pairs, per-world seeded
commands, one service lifecycle, phase barriers, invalid pins/specs/inventories,
startup/transport/arm/journal failures and interruption cleanup. Only these new
fixtures run; older suites and scientific plans are not replayed. The exclusive
preparation capsule records the actual command, source hashes, outputs, response
counts, CPU/wall/RSS and failures, with matching external custody.

This is trusted-process integration, not a sandbox or scientific admission.
Canonical plan pins differ from newline-terminated artifact byte pins. Assertions
about source/model/runtime provenance still require independent authentication;
this controller cannot prove them by hashing supplied claims. Parent supervision
owns total deadlines, hard-kill/disk-failure reconciliation, task-process descendant
cleanup, allocation telemetry, GPU startup/use limits, namespace admission,
checkpoint lineage and independent review. A hard termination without a terminal
inventory blocks finalization; there is no automatic salvage, retry or restart.
A separate exclusive root is not a global duplicate-submission lock. Preparation
must happen on CPU before a real GPU allocation, and future GPU use still requires
measured sizing and independent utilization/progress signals. No live operation,
real model, scientific ID or scientific response is part of this implementation's
validation.
