# Private final-release acceptance adapter, 2026-10-24

This note supersedes the pending-adapter and pending-accounting statements in the
original final-release preparation notes. It describes implementation only. The
parent owns actual terminal evidence, acceptance, report assembly, manuscript,
final plan invocation and any package build. Independent code review is still
required. No model, reporter, statistical procedure, package build, remote action,
commit or publication was performed by this implementation subtask.

## Caller trust and interface

`prepare_delivery_final_release_plan.py` requires a private request with schema
`delivery-final-release-inputs-v1`, exactly the keys `schema` and `inputs`, and an
exact input membership listed below. Every input is `{path, sha256}` with an
absolute regular-file path and a full SHA256 digest. All inputs are captured and
checked, including review text and structured review witnesses. Frozen source
bytes are inspected and hashed; they are never executed by this adapter.

The CLI requires `--request`, `--request-sha256`, `--report-acceptance-sha256`,
`--accounting-sha256`, and `--destination`. The Python interface is
`prepare(request_file, request_sha, destination, *, acceptance_sha, accounting_sha)`.
The three supplied digests are caller trust roots: obtain them from independent
review/parent custody, not from a checksum supplied by the packet being checked.
A self-consistent packet cannot pass with a missing or mismatching independent
acceptance/accounting pin. The destination must be new, exclusive and outside all
input directories. No actual invocation was made as part of implementation.

Exact input keys and meanings:

- Existing preparation inputs: `plan`, `manifest`, `bundle`, `snapshot_inventory`,
  `replay_acceptance`, `pdf`, `checker`, `planner`.
- Report evidence: `report_freeze`, `report_execution`, `report_receipt`,
  `report_runtime` (the report_runtime_verified.json bytes), `report_custody`
  (custody_verification.json), `report_acceptance`.
- Additional report inputs: `execution_verification`, `report_review`,
  `report_review_witness`, `runtime_inventory`, `remote_inventory`,
  `report_input_pins` (the launch input_pins.json).
- Accounting inputs: `accounting` (publication_accounting.json),
  `accounting_source` (the original 57-allocation incremental_accounting.json),
  `failed_report_inventory` (report01 remote_inventory.json), `accounting_review`,
  `accounting_review_witness`.

The full report custody root is the parent of the pinned report_freeze input.
It must contain exactly the remote inventory's 50 regular files, 9,147,453 bytes;
no extra files or symlinks. Do not place new review or acceptance files inside
that `raw` directory. The separately pinned launch input map must contain all 17
frozen inputs. Original historical inputs are bound through the frozen map,
accepted witness and the existing predecessor/publication lineage checks;
checkpoint bytes are never loaded or replayed here.

## Report acceptance schema

The adapter consumes the parent's `ace-delivery-report-acceptance-v1` as written:

- `accepted: true`, integer `job_id: 34867191`,
  `actual_execution_checks_pass: true`,
  `independent_scientific_output_review_pending: false`,
  `review_required_findings: 0`, `publication_approved: false`.
- Every execution_verification field is preserved exactly, except that acceptance
  may have a later timezone-aware `at` and closes the historical pending-review
  boolean. `execution_verification_sha256` binds the unchanged original record.
- `freeze_sha256`, `execution_sha256`, `report_receipt_sha256`,
  `runtime_verified_sha256`, `remote_inventory_sha256`, `custody_sha256`,
  `primary_index_sha256`, `descriptive_index_sha256` bind the captured evidence.
- `source_pins`, `metadata_pins`, `original_input_pins` equal the exact frozen
  report maps; `counts` and `reviewed_output_counts` equal the six named counts
  640/320/80/8/2/4. `new_fits`, `new_optimizer_updates`, `new_responses` remain zero.
- `review_sha256` binds the independent review text; `review_witness_sha256`
  binds `ace-report02-independent-review-witness-v1`. Its job, zero required
  findings, counts, 13 outputs, launch/historical/output maps and source revision
  are cross-checked. Optional `custody_file`, if supplied, must match the request's
  custody path. Extra parent annotation fields remain private.

The adapter validates scheduler parent/batch/extern `COMPLETED 0:0`, report02
identity, 1 CPU/3 GiB/3600 seconds and the named account/partition/QOS, the stored
live allocation capture and digest, cleanup and deadline reserve, accepted
replay12/runtime lineage, 28,577 inventory entries, all 618 loaded statistics
module-file entries, NumPy/SciPy versions and backing files, and the exact two
index memberships and 13 output hashes. Report01 job 34864950 is permanently
ineligible as successful report evidence. Scheduler success alone is insufficient.

Known parent acceptance at implementation time: SHA256
`86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10`.
Its independent review and witness pins are respectively
`1f1f4789615fff42ff1e0536fef0282fbd4f1b36190c2377ab9c3bc7f2f33fac` and
`6e00103b6a5015a7f52a4884fe5cf51fef16af00100f6e016cf72ccf110cfb64`.
These are documentation of supplied custody, not an implementation-generated
acceptance. The CLI still requires the caller to supply its independent pin.

## Reviewed public accounting

The public artifact retains the parent's exact
`ace-delivery-publication-accounting-v1` bytes. The input location is generic;
there is no requirement that its directory name end in 03. The supplied final03
artifact's digest was
`51865d9dbbada810cca2fa601deb08af95a5ce999b00464372e1a44783e27985`.

The schema has exactly `schema`, `scope`, `cutoff`, `attempts`, `reservation`,
`original_fit_measurements`, `report_measurements`, `unknowns`,
`response_accounting`, `measurement_scopes_overlap`, `source_receipt_sha256s`,
`publication_approved`, and `measurement_caveats`. Types, names, scopes, units,
nulls, and the six reviewed caveats match final03. Booleans are not numbers;
negative/nonfinite measurements fail. All 59 attempt IDs must be unique; the
46 original requested wall times remain null. The private join preserves every
field and failure state from the original 57 rows, and separately authenticates
both report scheduler rows. The source-receipt set must match the original
accounting and both report remote inventories.

Cutoff is report02 at `2026-10-24T22:09:27Z`, 59 allocations and 132,995 allocated
CPU-seconds: 128,690 + 1,792 failed report01 + 2,513 accepted report02.
Reservation is 94.36833333333334 of 150 CPU-core-hours, explicitly requested.
Original fit CPU/elapsed/RSS, report child/wrapper/scheduler/RSS, reservations and
confirmation response accounting retain their distinct scopes. Whole-sprint,
confirmation and complete preparation CPU, and interrupted returned responses
remain unknown. Replay06's recorded zero CPU retains its explicit uncertainty.

The private accounting review witness schema is
`ace-final-accounting-delta-review-witness-v1`. It must have no failed checks or
projection mismatches, affirm the required checks, bind the exact accounting and
three source objects by digest and size, and agree on both report measurement
objects, scheduler row digests, failure states, cutoff and caveats. Review text
and witness are independently request-pinned; their digests enter the public
contract, while their full contents remain private.

The exact accounting bytes become `publication/accounting.json` with an identity
manifest entry and an ordinary artifact binding. `accounting_disclosure` contains
`status: reviewed-artifact-bound`, `path`, `sha256`, `review_sha256`, and
`review_witness_sha256`. Missing, substituted or pending disclosure fails. The
public checker cross-checks report02 quantities against the projected execution,
receipt and resource limits. It returns `accounting_disclosure_complete: true`
only for this separately bound artifact; it still returns
`private_execution_revalidated: false` and `public_release_approved: false`.

All 3,893 predecessor file entries and first 5,068 bindings, required notices,
source/runtime dispositions, both bootstrap dependency pins, and all five false
qualification/approval flags remain unchanged. A new writing manifest or report
artifact grants no training, anonymous execution, publication or submission
qualification.

## Regression and custody

Only `test_delivery_final_release_acceptance.py` is run. It creates fabricated
packets and exercises the new private gate, independent trust roots, scheduler,
cleanup, runtime and output corruption, custody, accounting failure/unknown/type
boundaries, and public accounting binding. It imports the earlier `Packet`
fixture helper without invoking any earlier test methods. Passing these tests is
implementation evidence, never acceptance of actual reports or a real package.

Pre-edit correction01 closed evidence and current sources are preserved under
`results/delivery_final_release_preparation_20261024/acceptance_preparation01/preedit`,
with a pinned inventory. An exclusive external copy is under
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-release-acceptance-preparation-20261024-01`.
Each new test run has exclusive source copies, complete stdout/stderr, timestamps,
command, exit code and hashes. No earlier run is overwritten. The final receipt
records final source pins and the passing run. Separate independent code review
and parent integration remain downstream.
