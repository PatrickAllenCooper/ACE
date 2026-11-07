# Pure episode sealing adapter (2026-11-07)

`llm_intervention_episode_seal.seal_episode` converts supplied original episode
bytes into the exact prospective `ace-offline-cost-curve-v1` input bundle. It
performs no I/O, posterior replay, world generation, RNG calls or private scoring.
It does not call `run_episode`, `ResponseAdapter`, or the cost reporter.

```python
result = seal_episode(
    inputs=original_bytes, expected_sha256=independent_input_byte_pins,
    shared_initial_receipt_sha256=common_receipt_pins,
    scope=scope, world_id=world_id, policy_id=policy_id,
    acquisition_cap_units=acquisition_cap_units)
# result has exactly bundle, bundle_sha256, provenance
```

Only `artificial` and `development` scopes are admitted. Source validity,
independent pins, runtime/adapter honesty, process closure and the outer provenance
receipt are caller obligations, not scientific qualification conferred by sealing.

## Inputs and encoding

`inputs` and `expected_sha256` have identical keys: `world.json`, `task.json`,
`learner.json`, `episode-plan.json`, `executor.jsonl`, `episode-disposition.json`,
`outer-outcome.json`, plus `return-NNN.json` for every captured callback return.
No optional adapter logs or other extras are admitted. The total limit is 8 MiB,
80 charged requests, 243 journal lines, and 128 KiB per journal line. Graphs have
at most 8 nodes, menus 32 actions and tasks 64 equal-weight target-measuring
queries. Total initial plus acquisition budget is at most 1,000,000 units.

Every raw SHA-256 pin is checked before any JSON decoding. JSON rejects duplicate
fields, nonfinite constants and overflowing float literals. The supplied world
descriptor is validated with `WorldDescriptor.from_private_payload`; no world is
drawn or evaluated. Task and learner fields are schema-validated without fitting.
The original episode plan must bind the same graph/menu and total budget equal
to fixed initial cost plus the caller's acquisition cap.

**Every emitted bundle member is `(canonical(obj) + '\n').encode('utf-8')`.**
This includes accepted receipts and nonaccepted JSON-return `.bin` members.
World/task/learner bytes are explicitly normalized too; original byte hashes and
lengths remain in provenance. The receipt object contains exactly the six fields
`request_id, action_id, kind, targets, values, cost_units`, extracted from the
executor's `response` event. It excludes journal envelope fields. The parent's
candidate shared-initial pins must use this same canonical-newline encoding.

## Journal, return and terminal rules

The adapter checks every journal hash link and monotonic timestamp, the original
begin binding, contiguous typed proposal/request IDs, fixed initial proposals,
process decision indices, parse/budget rejection, stop, reservation cost and
accepted receipt semantics, including measured clamps. A final durable `stop`,
`rejected`, `failure` or `driver_failure` is required. A dangling proposal or
reservation, truncated line, `closed`-only journal or any continuation after
failure is rejected as ambiguous. This adapter covers `run_episode`'s
`execute_completion` path, not the separate generated-token executor API.

Each callback wrapper has exactly `request_id, action, returned_to_wrapper,
response`. `returned_to_wrapper` is true; `action` is the reserved serialized
Action. `response` is the existing `_raw_return` capture: `python_type`, either
`raw_response_repr` or `repr_error`, `raw_json_available`, and `raw_json` exactly
when available. Capture is required before returning to the executor. Accepted
responses must agree with captured finite values after executor float conversion.

* Executor response plus consistent capture: accepted canonical receipt.
* Capture with valid response JSON but no accepted response: `returned_unaccepted`.
* Capture with invalid response JSON: `returned_invalid`, preserving canonical
  serialization of that captured JSON; no fabricated measurements.
* Durable failed reservation without a capture and with unknown driver-return
  status: `unknown`, retaining the full reservation cost.
* Known driver return without a wrapper, or any non-JSON Python return: `SealError`.

An adapter's prepared `outcome` is never evidence of a returned callback. No retry,
fallback or completion of missing initialization is performed. The full tuple of
shared-initial receipt hashes is caller-supplied, including for partial arms; any
accepted initial prefix must match it. An empty initial tuple is supported.
Choosing consistent shared pins across arms remains the paired finalizer's job.

The outer object has exactly the agreed `ace-paired-episode-v1` fields: version,
scope, world_id, policy_id, status, reason, episode_returned_cleanly,
adapters_closed_cleanly, episode_disposition_sha256,
outer_provenance_receipt_sha256. Completion requires both clean flags and the
journal-derived `policy_stop`/`budget_exhausted` reason. Failed outer reasons use
`episode_failure:` or `adapter_close_failure:` with the corresponding false flag.
A valid complete original terminal followed by adapter-close failure seals as
**failed**, preserving its original accepted history and accounting.

Original success/failure fields, initial/decision counts, terminal public history
and charge counters must agree with the journal. Public SCM summaries are checked
for schema, noise configuration and eligible-row counts, not numerically refitted.
A last accepted receipt can precede an atomic learner-update failure. An original
explicit `state_unavailable` failure is retained if its journal is sealed; the
journal remains the accounting authority. A missing/ambiguous journal is rejected.

## Output, failures and preparation limits

The output bundle has world/task/learner, a new pinned plan, shared-initial pins,
history, accepted receipt/nonaccepted return evidence, and a terminal sealing all
other member hashes. It contains all charged attempts. Provenance retains every
original input hash and length, output hashes, source/sealed status and chain tip.
No score, science acceptance or namespace qualification is claimed.

`SealError` means the caller retains original bytes and the unresolved common-cap
roster cell, with raw actual-cost evidence still available. It must not drop the
arm or treat rejection as zero cost. Even a sealed failure remains a failed arm
when the later cost reporter computes its capped endpoint.

The six new methods use handcrafted two-node payloads, journal events, return
captures and placeholder summary values. They do not execute old tests, the
executor, a real world, the cost reporter, or a model. They check exact new bundle
shape/encoding and failure/charge/terminal rejection paths. Parent integration
with the actual episode runner and downstream reporters remains separately owned.
