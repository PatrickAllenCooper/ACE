# Bounded delivery compute disposition review — 2026-10-08

Disposition: **changes required for the new compute disposition**. This review does not change any previously accepted A/C summary or original confirmation, statistical, scientific, source-transition, or release disposition.

## Scope and reviewed hashes

Reviewed only the new accounting implementation and its fabricated-metadata tests in `/Users/pat/code/ACE`:

- `scripts/research/audit_delivery_compute_accounting.py`: SHA-256 `89024b315412bbff99d55df05a3ee1eb6f155d5ad4754de54d4e940a6386d1c6`.
- `scripts/research/test_delivery_compute_accounting.py`: SHA-256 `780d4ee9671cac559774c9b74ec81628ae825e0d5964502fe3629435af9a8c6d`.

Both hashes were unchanged after testing. Static review and CPU-only fabricated metadata checks were sufficient. No agents were spawned, no fits or installations were performed, and no remote, git, manuscript, model, prediction, or scientific outcome work was done. The existing test fixture reads the pinned confirmation worker bytes for its digest/provenance check; the worker was neither executed nor separately reviewed. Actual metadata-source closure/custody and accounting documentation remain with the main reviewer. The findings below demonstrate validator defects on fabricated inputs, not corruption or excess cost in actual captured metadata.

## Required fixes

### 1. [P1] Bind raw Slurm identities to the authorized ACE identities before counting allocations

Implementation lines 117–133 check raw-ID string uniqueness and exclude a dot, but apply the ACE allowlist only to `job_id`. There is no verified relationship between `job_id_raw` and `job_id`. Starting with the shipped fixture and changing its second raw ID from `33513116` to `99999999`, while leaving interpreted ID `33507421_15` and its five allocated seconds intact, still returns an accepted disposition with 15 chain allocated CPU seconds. An unrelated physical allocation can therefore be represented as an authorized ACE array task and counted.

Required fix: validate the raw-ID syntax and its captured relationship to the authorized parent/task identity using the trusted scheduler mapping or an explicitly authenticated mapping supplied by the main metadata closure. Reject unmapped, foreign, or contradictory raw/interpreted pairs before addition. Numeric raw Slurm IDs can legitimately differ from array display IDs, as the existing fixture demonstrates; requiring simple string equality would be incorrect. Add a regression that rejects the foreign raw ID while accepting the verified differing raw/array pair. The present `unrelated` test changes only interpreted `job_id`, so it misses this case.

### 2. [P1] Deduplicate canonical array tasks, including aggregate coverage

Implementation lines 118–119 deduplicate strings, while lines 127–132 interpret numeric task indices without requiring canonical spelling. Adding a five-second row with a distinct raw ID `33513117` and interpreted ID `33507421_015` beside the existing `33507421_15`, and adjusting the declared sum to 20, is accepted. Both interpreted IDs resolve to parent `33507421`, task 15; the reported allocation rises from 15 to 20 seconds despite repeated task identity. Checking the declared sum cannot discover that duplication.

The same logic accepts a pending `33507420_[0-19%2]` row together with pending `33507420_5`. Their current zero cost prevents monetary inflation in this example, but the representation repeats task membership and does not establish the claimed distinct allocation scope.

Required fix: require canonical supported job-ID forms or normalize them before uniqueness checks, and compare array identities as `(parent, integer task index)`. Track task coverage for pending ranges and parent summaries, rejecting conflicting aggregate/individual representations or explicitly treating summaries as non-countable representations. Continue excluding child steps. Add regressions for padded task aliases, intersecting pending ranges, and parent/range/individual overlap, with a valid distinct-task control. Do not resolve overlap by silently assigning additional cost or treating unknown allocation cost as zero.

### 3. [P2] Validate the reserved-core-hours field and represent unknowns explicitly

Implementation line 160 copies `rounded_total_reserved_CPU_core_hours` into the numeric disposition without measurement validation. The bounded checks accepted both `-100` and the string `"CPU measurement unknown"` as `B_current.reserved_CPU_core_hours`; both results also serialize successfully with `allow_nan=False`. Thus an invalid or differently scoped source value can appear as a reserved-core-hours measurement even though other seconds fields use `number()`.

Required fix: validate this field as finite, nonnegative numeric core-hours when recorded, and preserve `null` with an explicit reason/scope when genuinely unavailable. Reject booleans, strings, negative values, and nonfinite numbers. Have the main metadata closure establish that it represents the intended reservation basis, not actual process CPU or accrued allocated CPU seconds, before retaining that label. Keep it separate from the allocated-core-hours subtotal and completed-fit CPU. Add focused invalid-value and explicit-unknown regression cases. A `null` mutation already passes; this review does not require converting that unknown into an estimate.

## Scope separation and verification evidence

At the reviewed hashes, the returned structure correctly keeps F monotonic refit wall seconds separate from supervisor stage wall seconds and the final nested charged/reserved wall value (lines 148–155). It does not add refit and supervisor time, or sum nested aggregates. F process CPU and the total sprint CPU remain `null`. B completed-fit CPU, completed-fit wall, accrued Slurm allocation, and reservations occupy separate fields (lines 156–162); the only arithmetic subtotal across those output rows combines allocated CPU seconds with the accepted allocated pilot seconds. Accepted A/C summaries are inherited rather than replayed. No new scope-mixing arithmetic was found in those output paths; the allocation identity defects above prevent accepting their distinct-allocation claim yet.

Using `/Users/pat/code/ACE-Runner/.venv311/bin/python -B`:

- The existing focused unittest suite ran four tests, all passing.
- Six independent fixture mutations were run without persistent test/source edits: foreign raw ID; padded array-task alias; overlapping pending range/task; negative reservation; text reservation; and `null` reservation. All were accepted. The alias mutation increased accrued allocation to 20 seconds from the baseline 15; the pending overlap retained 15 because both added records had zero accrued allocation.
- These tests used temporary fabricated JSON and recomputed input digests, matching the existing tests' method of testing semantic validation after custody checks. They did not open actual B metadata or scientific outcomes.

The required fixes concern the new disposition and its focused tests. Preserve its incomplete-total status, explicit unknowns, nonadditive scope descriptions, and prior accepted dispositions when implementing them.

## Final bounded static recheck — 2026-10-08

Updated disposition: **no remaining critical findings or required implementation fixes within this bounded review**. This append supersedes the initial changes-required disposition for the new compute accounting implementation at the hashes below; the original findings and evidence remain preserved as review history. Prior accepted A/C, confirmation, statistical, scientific, source-transition, and release dispositions remain unchanged.

Rechecked SHA-256:

- `scripts/research/audit_delivery_compute_accounting.py`: `2bfbfd4c2531122df2f2305a37220c0c94e12837f39e35378aa5371e98c486f4`.
- `scripts/research/test_delivery_compute_accounting.py`: `3e1d1c22a16e59392ae8fa61bfa8de78348769bb16b4159cca9297d21242587d`.

All three findings are addressed in the reviewed implementation:

1. Raw/interpreted identity pairs must match the separately digest-pinned mapping (lines 103, 113–123, 143–152). The mapping must carry the original B revision and registration, expected ACE account, and a syntactically valid captured-receipt SHA-256. Raw IDs require canonical positive decimal syntax; scalar jobs require raw/display equality. An unmapped foreign raw ID fails before its allocation contributes to the sum, while an authorized differing numeric raw/array display pair remains supported.
2. Array indices and range bounds now require canonical decimal spelling. Coverage is compared using `(parent, integer task index)` and range expansion (lines 153–162), rejecting intersecting task/range representations. Bare array parents fail the supported-identity check. These checks apply even when an invalid alias or intersecting representation appears in the supplied authorized mapping.
3. The mapping's original registered reservation basis is finite and nonnegative, bounded at 150 core-hours, and explicitly labeled as the original full-registration requested total. A numeric scheduler reservation must be finite, nonnegative, and equal that basis. A missing scheduler reservation remains `null` and requires a nonblank explicit reason, which is retained in the result (lines 121–130, 191–192). Reservations remain separate from measured process CPU and accrued allocation.

Static inspection of the seven test methods confirms new regression coverage for the foreign raw pair, purportedly authorized padded aliases, task/range and range/range overlap, bare parent summaries, valid distinct array tasks with differing raw/display IDs, invalid reservations, and explicit unknown reservation preservation. The existing scope-separation controls remain. The user reports all seven tests passing; this recheck did not execute them or claim an independent final test result. The main reviewer owns the measured final suite and actual metadata audit.

The implementation continues to report F refit monotonic wall time, supervisor elapsed wall, nested charged/reserved wall, B per-fit CPU, B per-fit wall, Slurm allocation, and reservation as separate scopes. No additional overlap sum or conversion to process CPU was introduced. F process CPU and total sprint CPU remain unknown.

Provenance boundary: the auditor verifies the supplied mapping digest and identity/basis fields, records its captured-receipt digest, and validates that digest's format. It does not independently open that historical receipt or registration to derive the mapping or reservation basis. The main reviewer is explicitly freezing that closure from the separately published `source_accounting_20261008T000339Z` receipt (user-provided digest prefix `7d6136bf...`) and original registration `e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5`, rather than candidate totals. This static disposition is conditional on that independent provenance closure; this reviewer has not verified the actual mapping, actual costs, or original sources.

Only this report was appended during the recheck. No other files were written and no agents, fits, installs, remote operations, git operations, manuscript work, or scientific outcome inspection were performed.

## Final disposition after reported input freeze and measured runs — 2026-10-08

**Accepted within the bounded static review: all three findings are closed; no remaining critical findings or required fixes.** The previously stated provenance condition is reported complete by the main reviewer. No further work remains for this bounded reviewer. This disposition preserves the earlier accepted scientific and release dispositions and does not constitute an independent historical-source or actual-cost audit.

Independently rechecked final file hashes match the reviewed implementation and tests exactly:

- Accounting code: `2bfbfd4c2531122df2f2305a37220c0c94e12837f39e35378aa5371e98c486f4`.
- Focused tests: `3e1d1c22a16e59392ae8fa61bfa8de78348769bb16b4159cca9297d21242587d`.

The main reviewer reports a frozen 42-pair identity mapping derived from the separately committed `e9d5f7c8` historical `source_accounting_20261008T000339Z` receipt (reported SHA prefix `7d6136...`), with original registration `e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5` checked for the requested reservation total of approximately 86.868333 core-hours. This is reported independent source closure, rather than derivation from current candidate totals. The full frozen inputs and run evidence belong to `delivery-compute-disposition-preparation-20261008T0109Z`; this reviewer did not open or alter those artifacts.

Main-reported final verification, with both reviewed files unchanged through the runs:

- Actual metadata audit succeeded: child process CPU 0.039790 seconds, elapsed wall 0.043223 seconds, reported RSS 32456704.
- All seven focused tests succeeded using temporary files under `/tmp`: child process CPU 0.166467 seconds, elapsed wall 0.171942 seconds, reported RSS 23494656.

These are verification-run measurements reported by the main reviewer, not independently rerun measurements or additional historical research costs. Process CPU and elapsed wall remain distinct; no subtotal is inferred from them. The main reviewer reports no model/score access, original source changes, or new fits. This concluding action independently checked the two file hashes and appended only this report.
