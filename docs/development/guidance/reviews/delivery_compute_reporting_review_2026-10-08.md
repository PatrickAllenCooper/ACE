# Historical compute disclosure review — October 8, 2026

Reviewed at 2026-10-08 01:59:44 UTC. **No findings; no actionable fixes within the reviewed scope.**

## Actual limited scope

One distinct, bounded static prose review of `paper/aistats_ace_2027/paper.tex`, lines 3157–3164, beginning “The confirmation worker measures refit durations” and ending “their identified runs only.” Compared only with the specified preparation JSON and compute-disposition guidance. Adjacent pre-existing accounting prose was read for context, not re-reviewed.

No scientific, theory, source-contract, or accounting implementation review was repeated. No CURC access, network activity, model/outcome access, original-worker execution, experiments, tests, or manuscript compilation occurred. The only repository write is this report. Main retains responsibility for fresh original B custody and manuscript compilation. Existing Stage B science remains unopened until full original acceptance; this review supplies no acceptance or release authorization.

## Evidence digests

SHA-256 of the bytes read:

- `paper/aistats_ace_2027/paper.tex`: `8f0aeb56064c4f391898caa210dfe055138944c9630371921cf74b094b94478f`
- Reviewed disclosure only (UTF-8, preserving internal line breaks, excluding trailing newline): `a901888d15e2832d66c5bf3e398a59d043208306f0e8784a7d6e617ad6915b1c`
- `results/delivery_release_preparation_20261007/compute_accounting_preparation_20261008.json`: `8be132b6b0fd8cdd300bedefddc5949edea0d97f9879262d25e832a09a6b5baf`
- `docs/development/guidance/delivery_compute_disposition_2026-10-08.md`: `0ca24c561fff59a3fa95dee971fed8c5f357e2a8bab87d0d9673016705e6f46e`

## Findings assessment

- **Missing CPU is not inferred.** The disclosure explicitly calls complete historical confirmation process CPU unmeasured/unknown and declines a whole-project CPU total. This matches JSON `verified_accounting.F_confirmation.process_cpu_seconds = null`, `total_sprint_cpu_core_hours = null`, and `unknown_costs`; guidance lines 34–36 and 64–68 preserve the same limitation.
- **Overlapping costs are not added.** Refit durations are described as monotonic elapsed time overlapping acquisition stages; nested charged/reserved wall budgets are described as cumulative records. This matches JSON `verified_accounting.F_confirmation.scope` and `no_sum_across_incompatible_or_overlapping_scopes = true`, and guidance lines 24–35. The disclosure does not sum elapsed time, reservations, fit process CPU, or scheduler allocation. Identified replay/packaging/review runs remain bounded subsets, consistent with JSON run scopes and `unknown_costs`, and guidance lines 66–68 and 89–94.
- **No newly unbound numerical claim.** The reviewed eight lines contain no numerical literals or numerical macros and introduce no derived total. Their qualitative measurement claims are supported by the two supplied records. Neighboring pre-existing macros and qualification figures are outside this review.

These conclusions establish consistency with the supplied accounting snapshot and guidance. Underlying worker code, historical receipts, input completeness, and live/final B accounting were not independently authenticated; the JSON itself limits what its metadata pins establish.
