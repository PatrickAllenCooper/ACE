# Final release implementation boundary — October 24, 2026

The new [private planner](../../../scripts/research/prepare_delivery_final_release_plan.py) and [public offline checker](../../../scripts/research/verify_delivery_publication_release.py) implement the bounded successor mechanics from the [integration plan](delivery_final_release_integration_plan_2026-10-24.md). No actual report outcomes, package build, model/replay, installation, publication or commit occurred. Existing candidates, workers, reporters, manuscript, guide and generic verifier were not edited. Separate independent review is still required.

**Production preparation deliberately rejects every report-acceptance record until the parent supplies its accepted terminal schema.** `require_report_acceptance` first rejects missing/unaccepted records, then rejects even `accepted: true` with an unsupported schema. There is no override, synthetic-success flag or supplied executable validator. The parent must implement the concrete schema check here after terminal acceptance; merely setting the documented schema constant cannot enable preparation. This is an explicit unresolved integration gate, not actual report qualification.

**Report01 terminal update (parent-supplied, not independently reinspected here):** job34864950 failed its deadline before the inventory/import stage completed; neither primary nor descriptive output exists. All 26 terminal objects/8,165,035 bytes and 17 matching pins are retained by the parent. This job is permanently ineligible as report-success evidence. The parent now reports report02/job34867191 submitted once and running, with all 17 inputs and allocation verified. This operational update was not independently reinspected here; it establishes no completion or acceptance and does not open the private gate.

## Inputs and entry points

The planner accepts `--request PATH --request-sha256 SHA256 --destination NEW_PRIVATE_DIRECTORY`. The independently pinned request has exactly `schema: "delivery-final-release-inputs-v1"` and `inputs`. Every input is an explicit absolute `path` plus full `sha256`; required keys are:

- `plan`, `manifest`: immutable candidate03 recovery plan and manifest, with the fixed independent pins in the checker. The manifest's parent is the original package root.
- `bundle`: accepted original 19-file assembly's bundle JSON; the other 18 files reside alongside it.
- `snapshot_inventory`: parent's final 42-file committed publication capture. Its parent must contain `source/<original-repository-path>` files. Inventory, bundle, captured helper/report sources and all 16 companions must agree.
- `replay_acceptance`: the fixed independently accepted replay12 receipt.
- `report_freeze`, `report_execution`, `report_receipt`, `report_runtime`, `report_custody`, `report_acceptance`: parent's separately pinned, accepted report-job evidence. Do not substitute report-job proofs for replay12 proofs.
- `pdf`: parent's final PDF bytes; this binds bytes, not a compiler execution claim. The parent retains source-to-PDF compilation/page-review evidence.
- `checker`, `planner`: independently pinned exact executing new source files.

The missing acceptance adapter must independently bind the **actually successful successor job** across the parent's acceptance, report execution, scheduler and complete custody records, then validate cleanup, every supplied report-evidence digest, accepted replay12/inventory lineage, original input/source/runtime bindings and both complete original output directories. No successful job ID is hard-coded. The public projection requires a canonical numeric job identity and rejects known-failed34864950; passing this syntax/exclusion check is not acceptance. Only the explicit projected runner fields are checked downstream; this does not validate the complete runner, scheduler, acceptance or custody schemas. The future private adapter must enforce the exact ten-source/three-metadata memberships below against its independently accepted bytes, validate every loaded statistics file against the accepted runtime inventory, and close the public failure/accounting disclosure gate. The report-custody schema and its link from the parent's acceptance must be implemented together, not inferred from a filename or status string.

After that gate is implemented and reviewed, preparation writes only a new private directory: `plan.json`, `public/` staged additions and `private/` pinned inputs/projection originals/request/derivation. It reuses the reviewed snapshot writer and existing generic projection semantics. It never calls the package builder. The eventual public checker accepts `--root PACKAGE --expected-manifest-sha256 INDEPENDENT_PIN` and uses the unchanged generic verifier first.

## Contract and scope

The root package remains `delivery-anonymous-derived-v1`. Exact candidate03 entries (3,893) and ordered bindings (5,068), including original/projected digests, notices and predecessor contracts, must survive unchanged. New bindings have a fixed generated inventory. The new authoritative record is `publication/source_contract.json`, schema `delivery-publication-source-dispositions-v1`; the exact old manifest is retained at `publication/replay_input_manifest.json`.

The contract binds all public writing/report/style/source/PDF artifacts, all 42 snapshot dispositions, the 24 original source relationships, both report indexes and all 13 outputs. The original exporter is provenance source; public original-worker/training execution is not claimed. The existing metadata helper is explicitly a checker dependency. Public report generation still requires a matching independently accepted replay receipt.

Seven public evidence projections cover replay freeze/execution/receipt and report freeze/execution/receipt/runtime. `SPECS` is the fixed selector/type allowlist. Nested child status/exit/cleanup, dependency versions/backing hashes and digest maps are selected explicitly; private commands, origins, account/authorization and other paths are omitted. The two scientific analysis trees retain their full values and must match the frozen acceptance tree's exact keys/strings and existing numerical tolerance (`rel_tol=1e-9`, `abs_tol=1e-12`). This does not calculate new statistics. Private derivation records enumerate omitted leaf selectors and preserve original bytes.

Each projection has its own original and derived hashes; projected evidence never impersonates an original receipt. Eight hash-only `private-custody` references cover request/planner/base plan/bundle/snapshot/replay acceptance/report acceptance/report custody. The public checker explicitly reports `private_execution_revalidated: false`. An externally trusted release pin and private acceptance remain necessary. All anonymity, anonymous execution, training and submission/public-approval flags remain false. Required source notices and resolved ownership are inherited unchanged.

## New fabricated checks and retained outcomes

Only [test_delivery_final_release.py](../../../scripts/research/test_delivery_final_release.py) ran. Positive cases construct in-memory artificial manifests with the full 3893/5068 cardinalities; they do not create a real package or bypass the production acceptance gate. Existing component suites/reviews and numerical work were not repeated. Production staging has not been exercised end-to-end while the parent schema is unavailable.

Command history (existing `/opt/homebrew/bin/python3.11`, always `-B`):

- Initial full `test_delivery_final_release.py -v` from `/tmp`: 9 tests, 1.346s, exit1, one failure in the source-scope mutation case. Constructor and checker shared the same mutable source-role map. The constructor now copies that map; projection rule maps are also copied. The failure remains part of this record.
- Focused `FinalReleaseTests.test_private_source_scope_and_authoritative_contract FinalReleaseTests.test_complete_downstream_contract_with_private_evidence FinalReleaseTests.test_nested_projection_rules_and_types_reject_leaks -v` from `/tmp`: 3 tests, 0.416s, exit0.
- Subsequent full nine-test run from the repository: 9 tests, 1.347s, exit0.
- Both new entry points' `--help` commands from `/tmp`: exit0. No input/package was opened.
- After correcting an overly strict exact-hash comparison to retain the existing scientific tolerance, focused `FinalReleaseTests.test_recorded_analysis_uses_frozen_tolerance_and_schema FinalReleaseTests.test_complete_downstream_contract_with_private_evidence -v` from `/tmp`: 2 tests, 0.219s, exit0. The final regression also rejects added scientific-tree fields despite mutually updated public hashes.

Pre-report01-update full-suite command, cwd `/tmp`:

```bash
/opt/homebrew/bin/python3.11 -B /Users/pat/code/ACE/scripts/research/test_delivery_final_release.py -v
```

```text
Ran 10 tests in 1.533s
OK
exit_code: 0
```

Coverage includes missing/unaccepted/unsupported acceptance before output, independent request pin failure, inherited entry/binding/notice changes, nested path/type injection, old-versus-derived selector confusion, report input/runtime/count/cleanup mismatches, omitted report/companion files, final source role/approval overclaims and frozen scientific tolerance/schema. No whole-sprint CPU claim is made from these timings.

Pre-report01-update SHA256 (historical ten-test revision; superseded, retained):

- `prepare_delivery_final_release_plan.py`: `8f441ea47622ff1a2e15586e951a66ec5b649a722bcd42fbacb9e5fa962c27e5`
- `verify_delivery_publication_release.py`: `afa6eceb15bceb5431a12e12fa3c2801da02e45aaea600a97c3a965f0bc7bec6`
- `test_delivery_final_release.py`: `b3246238e0b611682fc8e1cb455ccfdcd348320dc75bce9559a3b3e6ed28d554`

Remaining actual inputs: parent's accepted report/custody schema and all evidence pins; accepted 19-file bundle; final committed 42-file SAME-manuscript snapshot including refreshed reviewer guide and exact reporter/helper sources; final PDF and separate compile/page-review evidence; independent source review. Final anonymous-content/redistribution/integrated scientific checks and human publication approval remain outside these tools. No actual release is qualified by the fabricated tests.

## Report01 failure correction and compact preparation receipt

The failed report01 update supersedes the earlier integration plan's running-job cutoff. The fixture now uses explicitly synthetic job IDs, never a guessed report02 ID. A new regression rejects an otherwise fabricated-complete34864950 packet and confirms that different syntactically valid synthetic jobs still cannot open the production acceptance gate. Only that new check and four affected existing integration checks ran after this correction; parent driver/diagnostic files were not changed.

The retained exact command, stdout/stderr and source/note hashes are indexed by [preparation receipt](../../../results/delivery_final_release_preparation_20261024/receipt.json). `command01.json` records the five selected methods, cwd `/tmp`, timestamps and exit0; captured output reports `Ran 5 tests in 0.580s` and `OK`. `historical_test_outcomes.json` preserves the initial failure, corrections, prior test outcomes and pre-update hashes as explicitly labeled transcriptions from prior tool results. It does not pretend that original raw historical streams or unknown intermediate source hashes were captured.

Historical pre-review SHA256 (superseded by correction01, retained):

- `prepare_delivery_final_release_plan.py`: `ecacb0ebf776d26ef5f94aaf97e24aa1721eee7fd046c19ba171a519cbb1d6bf`
- `verify_delivery_publication_release.py`: `3f783963563cec7e64e5c931c50ef9593d95562071b3e390daf1a846298fdc2f`
- `test_delivery_final_release.py`: `3d4c420e0435c52f6c036ba9735e1feb8846eaa131c97084184d1b774e0b2d69`

Status remains preparation only: real success schema/evidence unavailable, production gate closed, separate independent review required, no actual build or commit.


## R1/R2/R3 correction01 — blocked preparation, awaiting Russell review

The [new review](reviews/delivery_final_release_preparation_review_2026-10-24.md) is unchanged. Current originals, predecessor receipts/streams and the review were captured before edits under [correction01 originals](../../../results/delivery_final_release_preparation_20261024/correction01/originals.json). The [correction receipt](../../../results/delivery_final_release_preparation_20261024/correction01/receipt.json) supplies final source/document pins, exact commands, test streams and revision snapshots. Predecessor evidence is never overwritten. This section supersedes the earlier bootstrap, incomplete source-map fixture and resource-omission behavior; historical test results above remain historical.

**R1 — complete public executable bootstrap.** The reader must first authenticate this checker using trusted code and an independently trusted checker digest. With a trusted Python 3.11+ installation/standard library, invoke it using `-I -S -B` to exclude path/environment imports, site startup and bytecode caches. Before executing either local dependency, the checker captures and checks both against fixed digests inside its authenticated source:

- `delivery_prospective_publication.py`: `598e9071d5fdb5bebff81679b59578e1b5f16487f1c0584dc417ee6606998db9`.
- `verify_delivery_release.py`: `9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52`.

These two dependencies import only the standard library. The loader executes the captured authenticated bytes, bypassing project import search, cached modules and bytecode. Then the generic verifier checks the independently supplied final manifest pin; the package copies must also match the bootstrap pins. This applies even to `--help`. A helper self-hash or adjacent checksum is not independent authentication. Changing dependencies requires source review and a new trusted checker digest. The future reader guide must state this exact order and all three executable source files. This is the public-checker bootstrap; private planner/capture/build tooling still requires trusted local source custody and is not advertised as a reader entry point.

**R2 — complete known report membership and dispositions.** Both report freeze and receipt must have exactly these ten `source_pins` members, with no omissions or additions:

- `run_delivery_prospective_reporting.py`
- `qualify_delivery_replay_runtime.py`
- `supervise_delivery_qualified_replay.py`
- `delivery_replay_supervision.py`
- `runner_delivery_confirmation.py`
- `audit_delivery_prospective_results.py`
- `verify_delivery_release.py`
- `replay_delivery_prospective_release.py`
- `export_delivery_prospective_claims.py`
- `prepare_delivery_prospective_supplement.py`

The exact three `metadata_pins` members are `runtime_inventory.json`, `replay_receipt.json` and `replay_acceptance.json`. The observed historical freeze supplies membership only: no report01 or report02 source digest is promoted to successful execution. The future acceptance adapter must use these same memberships and independently authenticate their actual accepted values; changing membership requires coordinated review of the adapter and public schema.

`report_source_dispositions` binds the exporter and descriptive reporter to their hashed public artifacts. Each of the other eight **executed report versions** has an explicit digest plus `private-custody` disposition bound to the report custody digest. A similarly named historical file elsewhere in the package is not silently substituted for that executed version. Private custody remains externally accepted, not publicly revalidated.

The complete recorded `loaded_statistics_files` map survives as module → `{path, sha256}`. Identifiers are limited to NumPy/SciPy dotted module names; paths must be canonical environment-relative `lib/pythonX.Y/site-packages/<numpy-or-scipy>/...` names, without traversal, absolute origins or extra fields. Both roots are required and their hashes must match the projected dependency hashes. Aliased modules sharing a file must agree on its digest. The future acceptance adapter, with private inventory access, must validate completeness and every module/path/hash relationship against the accepted runtime inventory. This public consistency check cannot establish loaded-code completeness from unavailable original custody.

**R3 — recorded costs and limits retain type, units and scope.** Resource selectors now produce `{value, status, unit, scope, basis}` quantities. `basis` distinguishes requested limits/reservations from observed usage. Count/byte/KiB fields require nonnegative integers; durations/CPU reservations require finite nonnegative integers or floats. Booleans, numeric strings, negative values, NaN and infinity are rejected. Missing or null facts become `status: unknown, value: null`, never zero. Unit/scope/basis changes are rejected on public validation.

Report projections retain frozen CPU count, RSS-byte limit and wall-second limit; wrapper elapsed seconds; runtime-verification CPU/wall seconds; and receipt `process_cpu_seconds`, `elapsed_seconds`, `peak_self_rss_kib`. CPU time is in CPU-seconds, wall time in seconds and Linux self peak RSS in KiB. Child timings start after context validation; runtime-verification timings are a prefix of the child receipt, so they must not be added together. Self peak RSS is not a process-tree peak or scheduler allocation.

The accepted replay's actual metadata shape was read locally without model access. Its selected facts use their existing names: freeze CPU/RSS/wall/child-deadline/GPU limits, core-hour reservations and known prior allocated seconds; execution `wrapper_elapsed_seconds`; receipt `replay_cpu_seconds` and `replay_wall_seconds`. Reservation hours and known allocated elapsed seconds are not measured whole-sprint CPU time. No synthetic report fixture values are claims about a real job. Full scheduler records, sampled RSS, absolute origins, authorization and failed-attempt details remain in private original evidence for the eventual disclosure adapter.

The manifest-bound contract now explicitly retains a **pending** `accounting_disclosure` requirement: a separately reviewed public failure/accounting artifact, an explicit reporting cutoff, every failure through that cutoff, requested-versus-observed distinctions and missing-measurement disclosure. Whole-sprint process CPU remains unknown, the actual disclosure cutoff remains null, and failed34864950 is permanently ineligible as success. The public result reports `accounting_disclosure_complete: false`; this requirement is not itself the finished disclosure artifact. Implementing/binding that artifact is future reviewed work, along with actual report/custody acceptance. Neither gate is waived by these tests.

Focused correction regressions cover authenticated load order and substitution rejection (including expected failing subprocess streams), exact pin memberships, loaded statistics identifiers/digests, public/private source dispositions, resource preservation/unknowns, and invalid resource types/units/scopes. Only named new regressions and affected fabricated tests are run; commands, outputs, exit status and the precise tested source revisions are retained in correction01. Production acceptance remains unconditionally blocked, including when an `accepted: true` record or a different numeric job ID is supplied. No real package or report is created. Russell's distinct recheck is still pending.

Retained correction test results: `run01` passed the five new and five affected fabricated methods (10 tests, 1.666s, exit0). After tightening isolated bootstrap startup, statistics path namespaces and replay reservation selectors, `run02` passed the same focused set on the final source revision (10 tests, 1.694s, exit0). No test failure occurred; expected substitution-rejection subprocesses exited1 and their complete tracebacks remain in each run's stdout. No old component suite was run.
