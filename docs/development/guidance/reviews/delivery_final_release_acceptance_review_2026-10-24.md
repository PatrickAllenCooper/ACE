# Independent final-release acceptance adapter review — October 24, 2026

**Disposition: 0 required fixes and 0 optional findings for the pinned new adapter delta.** The implementation is suitable for the parent's actual integration attempt, subject to the existing independently pinned request and downstream checks. This review does not accept a final package, its PDF, anonymous execution or publication.

Scope is the new report02 acceptance and final-accounting adapter in `prepare_delivery_final_release_plan.py`, `verify_delivery_publication_release.py`, the new `test_delivery_final_release_acceptance.py`, its interface note and `acceptance_preparation01`. Baseline commit `606323161111237c9fd991e512163e96f9205f06` already closes R1/R2/R3/O1; that review and suite were not repeated. No production source, report, original artifact, accounting JSON, manuscript or parent driver was changed here.

## Findings and trust boundary

The new production entry captures and checks the independent request digest before decoding its input map. It requires exactly 25 named absolute inputs and separately supplied acceptance/accounting trust pins before reading dependent evidence. The acceptance bytes must be `86abaf25…`, as independently supplied by the parent, rather than a checksum self-declared by the packet. The supplied accounting trust root is `51865d9d…`. The implementation permits independently reviewed future pins as caller inputs; it does not infer approval from self-consistency, a job ID or scheduler success. Invalid/missing prerequisites reject before final-plan output creation.

The accepted report is joined to the unchanged execution-verification record, all seven named evidence digests, the closed independent review and witness, source/metadata/original-input maps, counters and both output indexes. The historical pending-review boolean is explicitly allowed to become false in acceptance; its original verification bytes remain separately pinned. The actual acceptance has no `custody_file`, so that field's optional handling is correct. The two supplied witness schemas and dictionary layouts match the implementation.

The report gate requires the exact 50-file / 9,147,453-byte raw custody inventory and 17 frozen inputs, rejecting extra files, symlinks or altered bytes. All ten frozen source files and three metadata files are checked as data; no frozen reporter/worker source is executed. The report's 618 loaded statistics entries must have canonical environment-relative paths and matching file-kind/digest entries in the accepted 28,577-object runtime inventory. The NumPy/SciPy versions, origins and root backing hashes join the runtime receipt and report receipt. Both report-index memberships and all 13 output hashes must match the independently accepted records. Original historical input pins remain distinct from projected public metadata and are checked against the preserved contract before plan output. The six actual digest edges matched in the bounded metadata comparison.

Resource checks require report02's successful parent/batch/extern states, 1 CPU, 3 GiB, account/partition/QOS, 3,600-second frozen limit, independently hashed live allocation capture and deadline policy. The existing projection/production checks also retain child completion, cleanup and zero work/exit scope. Job34864950 remains excluded as successful evidence. The public checker still says that private execution was not revalidated by it; this adapter consumes independent acceptance rather than rerunning the reports.

## Accounting and public disclosure

The new accounting join accepts the reviewed `ace-delivery-publication-accounting-v1` schema and separately pinned `ace-final-accounting-delta-review-witness-v1`. Required review checks must be true, with no failed checks or projection mismatches. The witness binds the accounting and three source objects by digest and size and joins both report measurements, row hashes, states, cutoff and caveats.

The 46 original plus 11 prior preparation/replay projections must equal their preserved source rows. Report01 and report02 separately join their exact saved scheduler rows and batch-RSS measurements. The full membership and cutoff remain **59 allocations / 132,995 allocated CPU-seconds**, including the failed report. Report02's child/wrapper/self-RSS/sample-tree-RSS quantities join its acceptance and stored execution record. Reservation remains the distinct **94.36833333333334 of 150 CPU-core-hours**; overlapping fit, child, wrapper, scheduler and allocation scopes are not added. Missing measurements and all four unknown totals remain null, and the replay06 zero-CPU uncertainty stays explicit.

`publication/accounting.json` is included byte-for-byte with an artifact binding and review/witness digests. The public checker validates its exact scope, fields, finite nonnegative numeric types (rejecting booleans), units, nulls, failure dispositions and source-receipt set. It also joins report02 measurements and requested limits to the public report projections. `accounting_disclosure_complete: true` means this reviewed artifact is bound; it does not assert a measured whole-sprint CPU total or human approval. The new schema/projections expose neutral labels, measurements and hashes rather than private paths or raw scheduler/account information.

The predecessor-preservation and notice checks were not changed by this delta: all 3,893 inherited file entries, the first 5,068 bindings and original relationships remain exact. The existing independently authenticated checker bootstrap is unchanged. All five training/anonymous-execution/anonymity/public-release/submission flags remain false. The integrated writing/PDF and new final manifest still need the parent's actual invocation and final review.

## Actual-schema compatibility and targeted concern

I inspected `actual_mapping_notes.json`. It records 23 existing inputs and leaves the parent-selected final PDF/snapshot unresolved, as expected. Its specific concern was whether exact historical-row equality would reject decimal scheduler CPU values despite the previous review's 1e-9 comparison tolerance. A minimal standard-library probe extracted only the reviewed `slurm_seconds` function and compared all 46 supplied original durations with the pinned final03 projection: **zero unequal values**. There is no observed mismatch in these fixed inputs and no basis to weaken the comparison. The full new accounting function also accepted the supplied metadata in the separate probe below.

The parent's proposed sibling output `/Users/pat/ACE_Study_Results/delivery-final-release-plan-20261024-01` avoids the legacy plan input's parent directory. That is consistent with the unchanged exclusive-destination guard; this review does not authorize weakening it or overwrite an existing attempt.

Only new, small, read-only compatibility probes were used:

1. The duration comparison above, with zero mismatches.
2. Six original-input metadata digest edges, all equal.
3. `validate_report_resources` on pinned stored report02 metadata: pass.
4. `validate_private_accounting` (including the new public schema validator) on pinned stored accounting/review/receipt metadata: pass.

No `prepare()`, full `require_report_acceptance()`, public `check()`, package verification/build, exporter, reporter, model, scientific-output decoding, remote operation or test-suite invocation occurred. The resource/accounting probes imported trusted local metadata helpers only; they executed no captured reporting sources or numerical libraries. Thus they check new actual-schema compatibility without repeating the completed custody, numerical or whole-accounting reviews. Review CPU is unmetered preparation overhead.

The existing ten-method fabricated run was read, not rerun. Its command/source/output hashes match the implementation receipt and current sources. The bounded tests cover independent pins, failed/pending acceptance, scheduler/cleanup/resource mutations, runtime/output corruption, custody membership, accounting failure/unknown/type boundaries and public accounting bindings. Their positive fixtures remain implementation evidence, not actual execution qualification.

## Reviewed pins and saved witness

- Planner: `7137b2288eaa7975e6241a37ba9f93d520a932f51928d20dd03841f8158f48e2`.
- Public checker: `35384816ff131fef7404205f3205b49e9f300d6d397847f610f0dbc0168b887c`.
- New tests: `c4cf42fcd8ce322d90df4c0874c40a10de95752bb1b9fa2ed9d2e1636cc06791`.
- Interface note: `08f36fdb392464166b482c076384836e3ac2bdb8b87f863d1c610c7df4156180`.
- Actual report acceptance: `86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10`.
- Independent report witness: `6e00103b6a5015a7f52a4884fe5cf51fef16af00100f6e016cf72ccf110cfb64`.
- Public accounting03: `51865d9dbbada810cca2fa601deb08af95a5ce999b00464372e1a44783e27985`.
- Independent accounting witness: `00b76e1d76d83b18c229d9cf45c2968228601c8e54670f77cc729ccf47474cda`.

The [review witness](/Users/pat/code/ACE/results/delivery_final_release_preparation_20261024/acceptance_review/witness.json) records full evidence/source hashes, scope and probe dispositions; its SHA256 is `d34124963bdc7491bdf9262559c217926e063a3388ca6dfe11819faf58d5b7da`. Only this review and that witness were authored. No commit was made. There is no correction requiring a delta recheck at these source pins.
