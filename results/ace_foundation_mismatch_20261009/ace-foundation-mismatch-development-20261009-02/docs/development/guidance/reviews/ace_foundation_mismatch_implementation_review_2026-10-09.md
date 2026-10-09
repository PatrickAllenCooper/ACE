# Foundation mismatch implementation review

Date: October 9, 2026. **Disposition: one required reporting correction before freeze.** No additional scientific design change is requested.

## Scope and exact reviewed bytes

Static review of the new runner and reporter against the current mismatch protocol, including its newly explicit RNG streams, artificial fixture and response-journal semantics. Read the directly used selector and inherited component helpers to resolve delegated eligibility, model configuration and mixture behavior. Read the five artificial test methods without executing them. Consulted the supervisor's terminal-ledger construction only to establish how interrupted attempts reach the reporter; this is not a complete supervisor/resource review.

SHA-256:

- [foundation_mismatch_pilot.py](/Users/pat/code/ACE/scripts/research/foundation_mismatch_pilot.py): `7d50891299fda616f9ac9befa3852c76027af7d65fc78c2dd1da6322ba4a586c`.
- [summarize_foundation_mismatch.py](/Users/pat/code/ACE/scripts/research/summarize_foundation_mismatch.py): `facbb9c9e05cd3f6658f797114fd13f92cfb6b58ac5b2116ace35522e8028ee7`.
- [Protocol](/Users/pat/code/ACE/docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md): `4b3f59584f0f8b1ab15202c35b37b7fe2949374b85f9b5ae95e65e345ad8b025`.
- [foundation_mixture_selection.py](/Users/pat/code/ACE/scripts/research/foundation_mixture_selection.py): `294dd5c00f5fdd4b532aea837e5d5e09f3ee0e8b226700ca1049045a227e75df`.
- [Inherited component helper](/Users/pat/code/ACE/scripts/research/foundation_component_pilot.py): `0d0c9c040d594523c48fd199662fda1bb581c09375af7157e2e2d6bb1c5997f6`.
- [test_foundation_mismatch.py](/Users/pat/code/ACE/scripts/research/test_foundation_mismatch.py): `444e24d0f351e5235b08234463ac49fc10d4184e52a26ebfa09feeb452a84d64`.

No research programs, scientific generation, model imports or tests were executed. Only this review was written. The reported five passing methods are the author's execution evidence, not an independently reproduced test result.

## Required finding: interrupted attempts cannot receive the prespecified report

Location: [reporter verify(), line 59](/Users/pat/code/ACE/scripts/research/summarize_foundation_mismatch.py:59), with unconditional completion/artifact reads at lines 60–80 and full-return totals at lines 101–103.

`verify()` rejects every terminal whose overall status is not `complete`. It then requires `complete.json`, every prehistory/training/selection/evaluation record and every variant directory. Thus a deadline, signal, generation/persistence exception or final completion-publication failure prevents any report, even when the authenticated supervisor terminal preserves all 144 planned dispositions and some completed cells. Catching model-fit or cell-evaluation exceptions inside an otherwise finished worker is handled; this does not cover an interrupted attempt.

This conflicts with the protocol's full-failure ledger and its new rule that a response block reserved but lacking a returned record has **unknown returned count**, not zero. The journals now preserve the evidence needed for that distinction, but the reporter accepts only attempts in which all blocks returned. Its constant full totals cannot describe a partial attempt.

Required correction:

1. Add an authenticated failure-report path using the supervisor terminal's full ordered cell ledger. Preserve overall failed status, reason, resource receipt and every planned disposition. A failure report is not a successful qualification or permission to retry.
2. Require `complete.json` and full ledger/artifact closure for successful completion. For failed attempts, validate available completed-cell artifacts and journal records; represent unavailable pairs as undefined while retaining all six planned worlds in every comparison. Any affected full-six aggregate remains undefined. Corrupt records must be disclosed and must not be treated as verified completed results.
3. Report planned training/private totals separately from observed reservations, validated returned blocks and reserved blocks with unknown returns. Read journals conditionally per block; missing return evidence must not become zero or the planned full total. Preserve training/private categories separately. Publish available selection records without requiring unattempted variant directories to exist.

The existing artificial failure/zero test calls `summarize()` directly. The saved-interface test supplies a successful terminal and all artifacts. Neither exercises this failed-terminal/missing-artifact path. This review did not run either test or request a scientific run.

## Requested implementation checks with no additional required finding

- **RNG and variants:** `SeedSequence([seed, stream])` matches the updated protocol: coefficients 0; shared prehistory 1; post histories 10–13 in variant order; private coordinates 100/101/102. Coefficient/family draw order and history draw order agree. Natural M is unclipped; intervention rows overwrite M before Y is generated. The coefficient change modifies only M's intercept/non-intercept coefficients; sine replacement affects only the specified head. Separate post-history randomness is intentional and now explicit. Private parent coordinates repeat across variants within a seed.
- **Eligibility and selection:** inherited eligibility excludes M-clamped M labels and supplies measured M to Y. The selector uses the exact disjoint 24/8 split, yielding 15/24 fit labels and 20/32 for Grammar32. Calibration excludes its three M-clamped rows and uses only X and noisy Y from the five natural-M rows. No measured M enters composed calibration forecasts. Predictor construction has no variant/truth/private-probe input; the durable selection seal precedes private generation. These are the stated in-process boundaries, not adversarial isolation.
- **Mixtures and retention:** PFN weights, five/25 candidate grids, stable first-minimum tie selection, terminal complete-forecast blending and mixed-head composition match the equations. Terminal local diagnostics remain separate blended heads. Experts are fitted once and reused without post-selection refits; the pre-change grammar is fitted once per seed on its 24-row split and reused across variants. Shared-expert counters include calibration/private calls; variant/event/cell timings overlap and are not additive costs.
- **Private targets and normalization:** root composition and local probes are noise-disabled. Local-Y probes use M in [−2,2] with legal root value X=0; natural root-action M may lie outside that intervention domain. Metrics use the correct distinct probe columns and full post-history eligible variances for every method, including retention. Reporter harm is adapted-minus-retained MSE, normalized with that same floored variance; unchanged-head flags are correct for all four variants. Main ratios are candidate/Grammar32, with zero/failed pairs preventing full-stratum aggregates.
- **Completed-attempt accounting:** scientific loops produce 6×4×6=144 cells, 6×(32+4×32)=960 training responses and 6×4×768=18,432 private responses. Reserved journals precede generation; returned journals follow persistence. The reporter checks successful journal counts/order, selection bindings and saved-array error reconstruction. It preserves ordinary failed cells in completed attempts. The required finding concerns interruption and incomplete returns, not these completed-attempt totals.
- **Artificial boundary:** fixture mode uses only seed 123456 and fixed coefficients `[0.1,0.8]` / `[0.2,1.1,0.3]`, with four variants and six methods. Its 24 cells, 160 training responses and 3,072 private responses are distinct from the scientific plan. The read test source uses artificial data/stub learners and does not call a fresh scientific seed or pretrained model.

This disposition is limited to the reviewed bytes and requested scientific/reporting implementation boundaries. It does not constitute runtime fixture qualification, a frozen-source launch approval or a repeated review of the previously closed design findings.

## Narrow partial-report recheck — October 9, 2026

Current reviewed SHA-256:

- Runner: `c561ad8dac60cd01a160717a3578447b507b72e58981ef5ba63ffda20c9a7172`.
- Reporter: `511c1376159807ac8ecf9774dcda45cbc1a5dfa6134942dcc1fd36016181d9a5`.
- Artificial test source: `88ba9e97190a7835e64bb35d1bf98ad934177bc6063e476bba043a66c5077c72`.
- Protocol remains `4b3f59584f0f8b1ab15202c35b37b7fe2949374b85f9b5ae95e65e345ad8b025`.

**Disposition: the original rejection of failed attempts is fixed, but the finding is not fully closed; two residual corrections are required in the new failure-report path.** This recheck does not reopen the scientific design or previously accepted runner behavior.

### Implemented portions accepted

`verify()` now routes an authenticated failed terminal into `failure_report()` without requiring a completion record or unattempted directories. The report retains the planned matrix, original terminal cells, failed attempt status/reason and resource fields. Missing or invalid completed-cell evidence becomes `invalid_record`, preventing those cells from entering paired ratios or harm calculations. Available choices use the shared grid/tie validator. Training and private accounting separately distinguish planned, reserved, validated-returned, reserved-with-unknown-return, unreserved and invalid blocks. The runner's evaluation-return record now binds the persisted private archive hash, which both reporting paths check.

The new partial fixture explicitly covers a 24-cell artificial ledger with six completed cells, 768 validated private returns, 768 unknown reserved private returns and 1,536 unreserved private responses, followed by prediction corruption. Its source was inspected, not executed; the author's nine passing methods do not cover the two residual cases below.

### Residual 1: invalid prehistory does not invalidate the retention reference

Location: [reporter line 150](/Users/pat/code/ACE/scripts/research/summarize_foundation_mismatch.py:150), and completed-cell qualification at [line 169](/Users/pat/code/ACE/scripts/research/summarize_foundation_mismatch.py:169).

The prehistory `block()` result is discarded. Completed `prechange24` cells require only the post-change training/evaluation blocks and selection seal. If the shared prehistory archive is missing or its hash no longer matches its return journal, accounting records that defect but the retained predictor can still remain `complete` and supply apparently verified local harm comparisons. That leaves the training provenance of the counterfactual reference unqualified.

Required correction: retain a per-seed `prehistory_ok` result and require it when qualifying each completed `prechange24` cell. With missing/invalid prehistory closure, mark that reference `invalid_record` across the seed's variants, preserve its original terminal evidence, and leave dependent harm and reference-cell ratios undefined. Independently verified adapted-cell metrics can remain available. This requires no refit or generation.

### Residual 2: a non-object terminal cell aborts the failure report

Location: [reporter lines 120–121](/Users/pat/code/ACE/scripts/research/summarize_foundation_mismatch.py:120).

The loop calls `row.get()` before checking that the record is a dictionary. A terminal containing a corrupt cell represented by valid JSON `null`, a list or a scalar raises here, outside the per-cell verification handler. This is a reachable evidence-corruption path: the supervisor preserves a successfully JSON-decoded cell value without first enforcing object shape, then publishes the failed terminal. The reporter consequently fails instead of producing the promised complete failure ledger.

Required correction: validate record shape before accessing identity fields, preserve non-object values in raw evidence/verification diagnostics, and fill the corresponding absent planned identities with `invalid_record`. Produce the full ordered matrix with undefined affected pairs. Keep terminal/freeze authentication intact; malformed cell contents are not grounds for treating an attempt as successful.

No tests, models or scientific runs were executed. Only this scoped disposition was appended; the earlier review remains preserved.

## Final residual recheck — October 9, 2026

**Disposition: both residuals closed; 0 remaining required issues within this scoped implementation review.** The original partial-report finding is now closed. Earlier findings are preserved above as review history.

Current reviewed SHA-256:

- Reporter: `6580c9ce6c8cda52e668b4a4b5b8ea14959a5a3bf44cf47794d9cf48eade805f`.
- Runner: `c561ad8dac60cd01a160717a3578447b507b72e58981ef5ba63ffda20c9a7172`.
- Artificial test source: `25f2ddde8521d1437ebe93d75ad66a5a0f05725d6c8ae959621b2a29f8a48736`.
- Protocol: `4b3f59584f0f8b1ab15202c35b37b7fe2949374b85f9b5ae95e65e345ad8b025`.

- **Prehistory residual closed:** the per-seed block result is retained as `pre_ok` (reporter line 152) and required for every completed `prechange24` cell (line 172). Missing or invalid closure changes the reference disposition to `invalid_record`; the existing complete-status requirements then leave dependent ratios and local harm undefined while preserving independently validated adapted cells.
- **Malformed-record residual closed:** line 121 checks dictionary shape before accessing fields and requires exact integer/string identity types before constructing or looking up the key. Rejected records remain in diagnostics and raw terminal evidence; absent planned identities receive `invalid_record` in the complete ordered matrix.

The extended artificial fixture source contains the missing-prehistory-return and raw-null-cell assertions, including undefined null-variant harm, all 24 planned slots and preserved raw null evidence. It was read, not run. This recheck covered only the two residual corrections; no broader review, model execution or scientific generation occurred. Only this final disposition was appended.
