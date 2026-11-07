# Offline private cost/error curves

November 7, 2026. This CPU primitive evaluates already sealed paid records.
It generates no worlds or responses, invokes no policy/model, and performs no
training. It implements retrospective structural error reporting toward the
first-screen endpoint; scientific execution and provenance remain unqualified.

`scripts/research/llm_intervention_cost_curves.py` exposes
`report_cost_curve(inputs, expected_sha256=pins) -> JSON-only dict`.
`inputs` and `pins` are exact dictionaries mapping artifact filenames to raw
bytes and independent SHA-256 byte digests. There is no CLI, filesystem access,
callback, random draw or mutation of inputs. The caller loads the files and
persists the returned private report with its own exclusive custody and timing.
Maximum bundle size is 8 MiB, with 80 reserved requests and 32 menu actions;
existing world/task bounds are eight nodes and 64 queries.

## Prospective sealing contract

This is a deliberately new narrow input format, not a claim that the existing
`run_episode` API authenticates a study. The prospective runner must reconcile
its original reservations, returns, learner configuration and terminal before
sealing this projection. Keep original journals/response evidence. Expected pins
must come from independently trusted protocol/closeout custody, not from an
untrusted replacement bundle. An attacker replacing both records and pins is
outside this API's guarantee. All fields listed below are exact allowlists.

Required files:

- `world.json`: existing `WorldDescriptor.private_payload()` JSON. Its **raw-byte
  hash** is independently pinned; the world API's semantic digest is computed
  only after byte authentication for private evaluator binding.
- `task.json`: existing `PredictionTask.payload()` fields `target`, `queries`,
  `weights`, `scale`, `normalized_mse_tolerance`. Query actions must measure the
  target. Each weight must equal `1 / query_count`. Scale is fixed, positive and
  public. Use tolerance **0.04**, or an explicitly predeclared supplied tolerance.
- `learner.json`: `graph`, `noise_variances`, `prior_variance`. Graph matches the
  world; the fresh `LinearGaussianSCM` starts with zero coefficient means and the
  supplied positive isotropic prior variance/positive known noise. These are
  separately pinned modeling assumptions, never inferred from private truth.
- `shared-initial.json`: `receipt_sha256`, the ordered raw-byte digests of the
  shared initial receipts, frozen before policy acquisition. Failed prefixes may
  lack some initial receipts; observed accepted initial receipts must match.
- `plan.json`: `version` = `ace-offline-cost-curve-v1`, `scope`, `world_id`,
  `policy_id`, `world_sha256`, `task_sha256`, `learner_sha256`,
  `shared_initial_sha256`, `menu`, `initial_actions`, `acquisition_cap_units`,
  `threshold`. The four pins bind the corresponding files. Threshold must equal
  the task tolerance. Initial actions are an ordered list of menu IDs. The
  positive acquisition cap excludes initialization; initial cost plus cap is at
  most 1,000,000. Menu actions use the existing exact `Action` JSON schema.
- `history.json`: `version`, `plan_sha256`, `attempts`. Each attempt has exactly
  `request_id`, `phase`, `action_id`, `cost_units`, `outcome`, `evidence_file`.
  IDs start at zero and are consecutive. Initial requests follow the fixed
  schedule; acquisitions follow only after all initial responses were accepted.
  Costs must match the menu and fit the cap. Every reservation is retained.
- `terminal.json`: `version`, `sealed: true`, `input_sha256`, `status`, `reason`,
  `attempt_count`, `accepted_count`, `known_return_count`, `initial_cost_units`,
  `acquisition_cost_units`. Its independently pinned byte digest is outside the
  terminal; `input_sha256` must exactly bind every other input file. Counts and
  phase charges must reconcile. `complete` requires complete initialization,
  all attempts accepted, and reason `policy_stop` or `budget_exhausted`; the latter
  requires no affordable action. `failed` requires another nonempty reason and
  permits partial initialization, an unresolved response, or failure between
  accepted responses. Unsealed input is rejected, never interpreted as a stop.

All version fields above use `ace-offline-cost-curve-v1`. Extra evidence files,
missing receipts, duplicate JSON fields, nonfinite accepted values, reordered
IDs, altered pins, wrong action/clamp/cost and contradictory terminal records
are rejected. The plan's assignments, common cap and threshold must precede
outcomes; hashes check bindings, not chronology. The caller must enforce the
same world/task/learner/shared-initial/cap across paired policies and keep every
frozen roster entry, including failures. This single-history API cannot certify
cross-policy roster completeness or scientific split cleanliness.

Attempt outcomes and their additional files:

- `accepted`: `receipt-NNN.json`, where NNN is the request ID. It is the exact
  contract receipt with `request_id`, `action_id`, `kind`, `targets`, `values`,
  `cost_units`. Only these accepted receipts update the fresh replay learner.
- `returned_invalid`: `return-NNN.bin`, exact returned bytes that fail strict
  finite JSON measurement/clamp validation, including malformed/nonfinite data.
- `returned_unaccepted`: `return-NNN.bin`, exact valid JSON measurement bytes
  known to have returned but not accepted, for example before persistence failed.
- `unknown`: `evidence_file: null`; no returned-response evidence is asserted.

Every additional file has an independent pin and appears in the terminal
inventory. The prospective adapter must establish what those returned bytes
represent; a typed Python `repr` is not silently promoted to original JSON wire
bytes. An adapter's prepared outcome is not a known return. All three unaccepted
outcomes retain the reservation, terminate the attempt sequence and require a
failed terminal. The reporter performs no retries or refunds.

## Private endpoint and integration

The evaluator uses `private_structural_means` without noise, then replays only
saved accepted receipts into a fresh learner. For each fixed query it composes
the posterior-mean mechanisms and averages `((prediction − truth) / scale)^2`
equally across queries. The complete curve contains the shared-initial point at
acquisition cost zero and a point after every accepted paid acquisition. Each
point includes predictions, error, receipt count and actual acquisition cost.

First crossing uses **error <= threshold**. `attained`, `final_attained`,
`first_crossing_cost_units`, `final_error` and `capped_cost_units` are separate.
Attainment exactly at the cap differs from nonattainment despite equal capped
cost. A later error increase does not erase the first crossing. Any operational
failure or numerical evaluation failure gets the common cap, even if its
retained prefix crossed. Nonattainment also gets the cap. Actual phase spending,
accepted/known/invalid/unknown counts and costs are always separate and retained.
Early policy stop is a recorded policy decision, never private certification.

Failed partial initialization has no comparable shared-initial point: its curve
is empty and final error is null. Numerical evaluation failures retain the valid
curve prefix, full verified charges and error details, with null final error.
`curve_complete` means all accepted receipts after complete initialization were
evaluated, not that an operationally failed episode succeeded. Attainment on a
failed run describes only its observed accepted prefix. Scores are retrospective
replay results, not authentication of an archived live learner state.

`CurveInputError` denotes malformed/contradictory evidence, distinct from a
properly sealed operational failure. Preserve that rejection and all input bytes;
the outer frozen roster must retain its capped unresolved row and reconcile
actual charges from trusted originals. Never turn validation errors into zero
spending or complete-case exclusion. Shared initial charges are exposed
separately; the study ledger must deduplicate physical initialization reused by
policies rather than sum per-history reports as a whole-study bill.

Reports are private evaluator output and must never return to a policy, select
actions, tune thresholds or govern stopping. Every report explicitly leaves
source, split and execution-provenance qualification pending with the caller.
No source authentication, study admission, world independence, model utility or
new allocation follows from this primitive. Parent integration owns the complete
sealed-roster adapter, original-evidence custody, policy/evaluator separation,
protocol module, resource accounting and scientific endpoint analysis.

New qualification is restricted to seven tiny hand-solvable artificial test
methods, with supplied static receipts and direct artificial descriptors. No
generator/response callback or existing test suite is run. Exact new sources,
dependencies, invocation/environment, outputs, rejected fixtures and subprocess
CPU/wall time are preserved under
`results/ace_llm_intervention_cost_curves_preparation_20261107` and the exclusive
external `ACE-LLM-intervention-cost-curves-preparation-20261107-01` directory.
Test-process CPU excludes unmetered implementation/inspection/custody work and
does not reset any cumulative preparation allowance or change scheduler costs.
