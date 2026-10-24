# Final publication release preparation review — October 24, 2026

**Disposition: retain as blocked preparation; address the three required findings below before actual use.** This is a distinct review of the new planner, public checker, fabricated tests, schema note, integration plan and preparation receipt. It does not accept a report, a final package or publication. Replay12 remains qualified against candidate03. Report01/job34864950 is FAILED and permanently ineligible; actual reports and final-publication data acceptance remain future work.

The intentionally closed `require_report_acceptance` is correct for this stage and is **not a finding**. It unconditionally raises after checking `accepted is True`; changing `REPORT_ACCEPTANCE_SCHEMA` alone cannot enable it. The request reaches this gate before opening the package, report indexes or writing, and before creating a destination. The public projection separately rejects job34864950. A different numeric job ID establishes no acceptance. The eventual concrete acceptance/custody adapter must receive a separate review.

## Required findings

### R1 — P2: authenticate the newly imported helper before executing the public checker

**Locations:** `scripts/research/verify_delivery_publication_release.py:3`, `:12–15`, `:379–387`; schema note `:19–23`; integration plan `:61`.

The new checker imports `delivery_prospective_publication` at module load and immediately delegates `require`, `digest`, `decode` and `pin` to it. Its documented bootstrap instructs readers to authenticate this checker and the generic verifier. It does not require authentication of the new helper before that import. `check()` invokes the generic verifier only after the import, and the later comparison with `reports.SOURCE_SHA256` is a comparison to a value supplied by the already executing helper.

Concrete trigger: a reader authenticates the two files named in the checker docstring, then invokes the checker on a package whose colocated helper was substituted. The substituted helper executes before the independently pinned manifest is checked, even if the generic verifier subsequently rejects the package. The later equality check cannot make that execution safe or independently authenticate its own validator. An ordinary `--help` invocation also imports it. This is a new dependency introduced by this checker, not a request to rereview the old helper implementation.

**Required correction:** establish and document a complete bootstrap trust chain. Either authenticate the helper bytes with an independently trusted digest using trusted code before importing them, remove this executable dependency, or require an independently authenticated generic-verifier pass against the final external manifest pin before loading the new checker/helper from that verified package. Make the eventual reader guide and checker documentation explicit about this order and all executable dependencies. A hash comparison performed by the helper after import is not a substitute. Add only a bounded loader/order regression; no real package or report is needed.

This finding concerns future use of the public checker. It does not demonstrate a bypass of the current closed private planner gate.

### R2 — P2: enforce complete report source and runtime lineage instead of accepting arbitrary digest-map membership

**Locations:** `scripts/research/verify_delivery_publication_release.py:111–133`, `:150–161`, `:217–236`, `:263–266`; `scripts/research/test_delivery_final_release.py:117–130`; schema note `:21`, `:31`; integration plan `:36–40`.

The `sources` and `metadata` projection rules allow any nonempty dictionary with syntactically valid basenames. Their membership is not fixed. Downstream validation only requires the exporter/reporter source names and three metadata names. The positive fixture itself supplies only those two source pins. It omits the runner, runtime qualifier, process guard, auditor and other imported implementation pins needed by the report job, while returning a successful publication-lineage result. Adding unrecognized source or metadata names also succeeds.

The new projection additionally drops `loaded_statistics_files` altogether. Only the NumPy/SciPy top-level dependency versions and hashes survive. The runner records individual loaded statistics files and their environment-relative paths/hashes; those safe lineage facts can be retained without publishing absolute origins. A hash of the private original receipt is useful custody evidence, but it does not make the omitted per-module relationships available to this public consistency checker.

**Observed targeted counterexample:** using the existing artificial `Packet` constructor, retain its two source pins, add `unreviewed_import.py` and `unreviewed_evidence.json`, and recompute only the fabricated freeze/receipt/execution links. `construct()` and `validate()` both succeed. Source keys are exactly `export_delivery_prospective_claims.py`, `prepare_delivery_prospective_supplement.py`, `unreviewed_import.py`; metadata keys are exactly the three required names plus `unreviewed_evidence.json`. The returned `private_execution_revalidated` remains false. This establishes incomplete/unknown-map acceptance in the new pure validator, not successful private acceptance or an independently pinned real-package forgery.

**Required correction:** define the complete allowed source/metadata memberships for the eventual accepted report schema, require their presence and reject unknown members, preserve the loaded statistics module digest map with validated module/environment-relative identifiers, and bind each included source to its public artifact or an explicit private-custody disposition as appropriate. Couple the final memberships to the future parent acceptance adapter rather than guessing a successful job or report02 pin. Add a new negative case removing a required imported source and one adding an unknown member; the current two-source positive fixture must cease representing complete report lineage. Until this exists, narrow the schema note's claim that the known runner schemas are already checked downstream.

### R3 — P2: preserve recorded resource quantities and units in the public projections

**Locations:** `scripts/research/verify_delivery_publication_release.py:97–134`; `scripts/research/prepare_delivery_final_release_plan.py:143–147`; integration plan `:36`, `:66`, `:69`.

None of the seven projection specifications retains resource accounting. In particular, the known report receipt fields `process_cpu_seconds`, `elapsed_seconds` and `peak_self_rss_kib` are discarded. The report-runtime record's elapsed/CPU fields and the report freeze's resource limits are also outside the selector inventory. The omission record correctly preserves this fact privately, but the public contract has no separate accounting artifact or disposition that supplies the missing measurements.

**Observed targeted counterexample:** add synthetic `process_cpu_seconds: 12.5`, `elapsed_seconds: 30.5` and `peak_self_rss_kib: 123456` to a fabricated report receipt. `project('report_receipt', ...)` succeeds, and all three selectors appear in `omitted_selectors`; none appears in the public projection. Thus even available per-job measurements are lost, contrary to the integration plan's explicit preservation requirement. This is separate from the legitimately unknown whole-sprint CPU total.

**Required correction:** retain the available bounded resource facts with explicit units and scope, either through typed projection selectors or an explicitly bound accounting artifact. Validate finite nonnegative measurements with booleans rejected, retain requested versus observed distinctions, and represent genuinely unavailable totals as unknown. Preserve failures and the reporting cutoff through the eventual public disclosure artifact without promoting report01 to success. The correction requires only synthetic metadata checks; no measurement run or numerical work is needed.

## Checks that are sound within the stated boundary

- **Exact predecessor preservation:** `inheritance()` compares all 3,893 entries by full dictionary equality and the first 5,068 bindings in order. It then requires the exact generated binding suffix and addition set. `inherited_plan()` also checks original digests, roles, transforms, reconciled custody copies and the resulting inherited bytes. Projected A/C/F originals are not silently rebased to identity copies. Candidate03's preserved manifest has a fixed independent digest; replay qualification is not transferred to execution of the successor.
- **Notices and original relationships:** exact inherited entries preserve the notices and historical source contracts together with the generic byte verifier. All 24 original source relationships must equal the retained source contract; the added original exporter is matched to its historical source hash. The planner requires resolved Runner authority. The five style/provenance objects are also checked. No ownership blocker is reopened and no verifier screening exemption is expanded. Actual inherited payloads were not reread in this review.
- **Public/private separation:** all 42 snapshot members have explicit destinations or private-only dispositions. The seven evidence projections have separate original and derived hashes, exact outer selector/rule schemas, constrained scalar types and an explicit derived scope. Nested commands, account/authorization fields and absolute origins are omitted; original bytes and omitted leaf selectors are retained privately. Eight private references permit only digest and `private-custody` availability fields. R2 identifies the remaining digest-map/runtime completeness gap.
- **Report and manuscript bindings:** the two distinct indexes, all 13 report outputs, six original report inputs, historical-versus-derived acceptance/score selectors, exporter/reporter hashes, analysis-tree structure/tolerance and coverage counters are checked. All 16 embedded companions must match the separately bound companion bytes and hashes. Final writing, style files, claim-generator source, guide and PDF have explicit artifact bindings. This is metadata and byte consistency; it does not recompute statistics, prove CSV science, compile the PDF or establish page correctness. The PDF header test is not treated as compilation evidence.
- **Qualification scope:** the returned result says `private_execution_revalidated: false`; private acceptance is an external prerequisite. Training reproduction, anonymous execution, anonymity review, submission and publication approval remain false. The original exporter is provenance-only; descriptive regeneration still needs an independently accepted replay matching the chosen manifest. No successful successor report job is inferred or hard-coded.
- **Independent pins:** the request digest is checked before decoding its input map; candidate03, its recovery plan and replay12 acceptance are fixed to the declared pins. Each subsequent supplied input has a full independently supplied digest. The public entry point requires a final manifest pin, and the unchanged generic verifier reports that pin origin itself is not verified. R1 concerns authenticating executable dependencies before these checks can run.

## Optional follow-up

**O1 — mark the integration plan's historical cutoff explicitly.** Its line 5 still calls job34864950 running, and line 28 says the new interfaces do not exist. The schema note and receipt explicitly supersede these statements and the implementation rejects the failed job, so this is not an acceptance bypass. Add a short supersession note or update those sentences before the plan is reused as operational guidance. This review makes no edits to the plan, parent driver or manuscript.

## Evidence and review limits

All eight file/digest/byte entries in `results/delivery_final_release_preparation_20261024/receipt.json` matched the current files, including the retained command output and historical-outcome transcription. The retained five-method command reports exit 0 and `Ran 5 tests in 0.580s`; it was inspected, not rerun. The historical transcription is not promoted to original raw command evidence. The integration plan is outside that receipt's file list and was separately hashed.

Reviewed SHA256 identities:

- Planner: `ecacb0ebf776d26ef5f94aaf97e24aa1721eee7fd046c19ba171a519cbb1d6bf`.
- Public checker: `3f783963563cec7e64e5c931c50ef9593d95562071b3e390daf1a846298fdc2f`.
- New tests: `3d4c420e0435c52f6c036ba9735e1feb8846eaa131c97084184d1b774e0b2d69`.
- Schema note: `9bb2626fc49136af351d42b81bd09fbe60c8d4246480493bf6e5c709c928c751`.
- Integration plan: `671c24d396afdea6e3219b63862104651be2e8ea8a1208e9ae1af9170467854a`.
- Preparation receipt: `f6aa223f6ab89f25914060a7c1f92274ab8e3b3da44cc6b50b10cf4871953bb0`.

One new, bounded Python 3.11 `-B` invocation from `/tmp` exercised the synthetic cases described in R2/R3. It imported the new test fixture, called no unittest methods, wrote no files and did not override the production gate. It also confirmed omission of `/loaded_statistics_files/numpy.fixture/path` and `/loaded_statistics_files/numpy.fixture/sha256` when those fabricated fields were supplied. No old component tests/reviews were repeated. Existing helper code was read only where needed to trace the new interfaces and their declared report-record fields.

The sole authored artifact of this review is this Markdown file. No package build, report generation, model/runtime execution, scientific or numerical test, remote action, installation, job submission, worker change, parent-driver/manuscript mutation, publication or commit was performed. Parent reporter recovery is outside this review. Successful report/custody acceptance, integrated committed writing/bundle/PDF inputs, actual release checks, compilation/page evidence and human publication approval remain future gates.
