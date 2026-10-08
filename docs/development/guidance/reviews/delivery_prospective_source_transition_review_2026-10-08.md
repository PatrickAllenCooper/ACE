# Prospective source transition review — 2026-10-08

Reviewed at 2026-10-08T00:13:37Z. Two required findings remain in the source-contract successor and its planner integration. This is a bounded implementation review, not scientific acceptance or runtime qualification.

## Scope and reviewed bytes

Only these three files are reviewed:

- `scripts/research/delivery_prospective_source_contract.py` — SHA256 `b104bd73d3ea1f011144233db7b41823625cd0eb58f41fe9677489bbf269f1b3`
- `scripts/research/extend_delivery_prospective_release_plan.py` — SHA256 `add76ce67815c1e4db46a84dd4e0fc5213756c72d01838b203f68f74e3ceb62d`
- `scripts/research/test_delivery_prospective_source_contract.py` — final reviewed SHA256 `6166e6f37becba1255861b16571e16691a8235ef944c6f26557fb3db488d6837`

Dependency definitions and the existing fabricated fixture were read solely to establish behavior of these files. No completed scientific or A/C/F review was repeated. The supplied custody state remains context: original B revision `45ebeb89`, 562/640 fits, no real B acceptance or opened outcomes; immutable candidate16 predecessor contract SHA256 `9f4766216adbc1bfc217994c4c1122ccc874fa927880114ba311395c600ef3c2`. This review does not independently recount fits or certify that custody state.

No callable subagent tool was exposed, so the authorized distinct agents could not be launched. Review used static inspection and bounded local synthetic checks with `/Users/pat/code/ACE-Runner/.venv311/bin/python -B`, with bytecode writes disabled. No installs, inference, real models/outcomes, remote access, jobs, or new fits/responses were used. The only persistent file written by this reviewer is this report.

## Required findings

### R1 — P1: source qualification checks still follow accepted score decoding

Locations: planner lines 319–322, 360–364 and 500–501; successor lines 66–85 and 144–146.

Calling `preflight()` before `metadata('B/scores.json', ...)` does not finish source preflight. It authenticates the predecessor document and its disposition flags, but comparison of predecessor B identities to the frozen closure, included source identities, and inherited notice identities is deferred to `transition()`. An explicitly supplied adapter is checked only for being non-`None` before score decoding; its file/custody validation follows decoding.

Reproduced using the fabricated fixture and its independently supplied synthetic predecessor pin, without changing predecessor bytes: replace the inherited MIT notice with different synthetic bytes and update that file entry's digest. The planner decoded `scores.json`, created the private directory and derived outputs, then raised `ValueError: inherited source notice differs`. Supplying a nonexistent adapter similarly decoded scores before raising `FileNotFoundError` (without creating outputs). Thus source failure can occur after the prohibited boundary, even though the original full-acceptance gate remains first.

Required fix: perform a pure source validation step before planner line 322, using the already checked frozen closure/core contract and the actual supplied adapter. Check predecessor B source identities, inherited notices and any existing helper identity/transform conflicts there. Validate adapter existence and every claimed copy before decoding. Keep transition's final identity checks as consistency checks, while deferring only comparisons that require newly generated replay metadata. Original full acceptance must remain the first operation.

Required coverage: synthetic notice conflict, missing/invalid adapter, and predecessor B source mismatch must fail with a score-decoding sentinel untouched and both private/output paths absent. The existing missing-`None` test does not cover an invalid supplied adapter or late cross-contract checks.

### R2 — P2: inherited interface drift can produce a successfully returned, inconsistent plan

Locations: successor lines 43–57 and 140–143; planner lines 500–510; test lines 67–104.

The predecessor's five interface digests and runtime-record digests are copied into the successor and rebound without checking that the inherited plan still supplies matching artifacts. Authenticating the pinned predecessor authenticates those recorded values, not the current inherited file entries or their custody copies.

Reproduced without changing predecessor bytes: change the first inherited interface to different fabricated bytes and update its plan entry's digest. `planner.extend()` returned success and wrote the output plan. The inherited artifact digest was `fca2263230c1f841a971c90a937ad56584650df72c915d44a15481f691647205`, while both predecessor and successor recorded `f9f74d70fb575b670bb36499d56d029d33e1e11c66abc48d4d3e5dbafeba986e`; the output retained two incompatible manifest bindings. The downstream builder/verifier would reject these edges, so this does not establish a package-verification bypass. It does establish that successful planning can emit an unbuildable successor after decoding outcomes.

Required fix: before score decoding, reconcile the inherited interfaces and their runtime records against the pinned predecessor. Compare the actual planned package digest, honoring approved projections where applicable, rather than assuming every runtime record's package digest equals its original digest. Reject missing, conflicting or changed artifacts. Preserve the old bindings by retargeting them to the historical predecessor, and add successor bindings only after checking their targets. This is mechanical inheritance validation, not renewed A/C/F scientific review.

Required coverage: mutate an inherited interface digest/copy and separately a runtime-record digest/copy; require rejection before score decoding/output creation. Strengthen the positive test to compare every inherited binding with its expected retargeted form and check all successor interface/runtime binding targets. Its current assertions check existence of representative historical edges, but do not cover drift.

## Verified behavior and check results

Static inspection confirms original acceptance is the planner's first operation. The fabricated positive test passed through planning, package building, relocation and verification: predecessor bytes were preserved exactly, original source digests were retained, exactly one original design helper was marked included, a derived numerical core and sixth adapter were recorded, and numerical/anonymity/training/public qualification stayed false. These checks establish implementation behavior on synthetic custody, not actual B numerical replay or public readiness.

All three scoped files parsed successfully. The four-test run completed in 25.638 seconds: three tests passed; the missing-adapter test failed because its expected message said “adapter” while the planner said “interface.” During review, another writer corrected that test; its initial SHA256 was `fbd45d4e33ee00d37706a9e29003f4d258fcbcd55e135c807e54a95df89a09a6`. The corrected focused test passed in 0.750 seconds. The two implementation files remained at the hashes above. The regex mismatch is resolved and is not an outstanding required fix. The full suite was not rerun after that single-test correction.

No implementation, ledger, manuscript, Git state or jobs were edited. Required changes are limited to predecode source/interface validation and focused synthetic regression coverage in the reviewed files; custody/accounting documentation remains outside this review.

## Final bounded static recheck — 2026-10-08

Disposition: **R1 resolved; R2 resolved by static inspection. No remaining required fix identified within the two findings and their regression coverage.** This disposition supersedes the outstanding-finding status above for the following reviewed bytes:

- `scripts/research/delivery_prospective_source_contract.py` — SHA256 `013bc2c2c97c1be51104e9dd04df140c6ee7c08c827c8247c186a2262c59373f`
- `scripts/research/extend_delivery_prospective_release_plan.py` — SHA256 `35cf76c4c749ca400f31faaccef6eb291361e0e54d95f8df0c2abb7b2a894a55`
- `scripts/research/test_delivery_prospective_source_contract.py` — SHA256 `696de0eeafdeddbb41884c72286b8b52bd17abaaa247ac7ba095e3346a587b57`

For R1, successor `validate_before_scores()` (lines 60–103) performs read-only checks of all 20 B predecessor source edges against the already verified frozen closure, original auditor/utility core identities, inherited notice bytes, and any inherited design-helper conflict. Planner lines 319–327 complete predecessor/source validation and actual adapter digest/custody checks before the first score decode at line 328. AST inspection confirms the original full-acceptance gate remains the first executable statement of `extend()`. Transition retains its final consistency checks.

For R2, the validator requires exactly the five inherited interface paths and each interface's exact runtime set. It reconciles all claimed copies and compares the SHA256 of planned package bytes to the pinned predecessor values. `snapshot_entry()` restricts runtime projections to `A/protocol.json`, `C/protocol.json`, `F/protocol.json` and `F/runtime.json`, and reproduces the builder's JSON serialization; projected digests are therefore compared without substituting original-input digests. Interface and notice entries require identity transforms.

Coverage is adequate for these fixes on static inspection. Seven negative subcases cover notice, interface, runtime, B-source, missing adapter, symlink adapter and helper conflicts. Their score sentinel raises `AssertionError`, which the expected exception tuple does not swallow, and every case asserts absent private/output paths. The positive test checks every inherited binding's retargeted form, every successor interface/runtime digest and corresponding manifest binding, and an explicit projected-runtime/original-digest difference. Existing predecessor-byte, source-identity and false qualification assertions remain present.

Verification during this recheck was static inspection and successful AST parsing of all three files, with bytecode writes disabled. Neither the 25-second positive test nor the full suite was repeated; the main task's separately measured suite is not claimed as passed here. Predecessor preservation, the identity helper/derived core/sixth adapter distinction, and false numerical/anonymity/training/public qualification remain intact by inspection. No real B outcomes, models, data or remote resources were accessed. Only this report was appended.

## Final disposition with measured evidence — 2026-10-08

**Review concluded: R1 and R2 are resolved. No remaining required fixes within the scoped source transition, integration and regression coverage.** The final hash check independently confirms that the three files are unchanged from the bounded static recheck:

- `scripts/research/delivery_prospective_source_contract.py` — SHA256 `013bc2c2c97c1be51104e9dd04df140c6ee7c08c827c8247c186a2262c59373f`
- `scripts/research/extend_delivery_prospective_release_plan.py` — SHA256 `35cf76c4c749ca400f31faaccef6eb291361e0e54d95f8df0c2abb7b2a894a55`
- `scripts/research/test_delivery_prospective_source_contract.py` — SHA256 `696de0eeafdeddbb41884c72286b8b52bd17abaaa247ac7ba095e3346a587b57`

The main task reports that the final five-test suite, including seven negative subcases, passed from `/tmp`, with child CPU time `30.728969 s`, elapsed time `30.803681 s` and peak RSS `72,122,368 bytes`; source hashes remained unchanged through the run. It also reports successful real candidate16 source prevalidation against the private 24-edge inventory and original derived core contract, checking all 20 B source edges, five actual interfaces, four planned runtime projections and two required notices. Evidence is identified as `delivery-prospective-source-transition-preparation-20261008T0014Z/final-tests.json` and `delivery-prospective-source-transition-preparation-20261008T0014Z/final-real-preflight.json`. These run results are supplied by the main task; this reviewer did not independently open those private evidence files or repeat either run.

The supplied passing results supplement the static disposition above and supersede its pending-suite statement. They establish regression-test and source-prevalidation success only. No actual successor package, inference or real B outcomes were produced/opened by these checks. Exact-runtime numerical qualification, anonymous execution, training reproduction and public approval remain unqualified/false. The immutable predecessor pin remains `9f4766216adbc1bfc217994c4c1122ccc874fa927880114ba311395c600ef3c2`. This final step appended only this report; no implementation, ledger, manuscript, Git state or jobs were changed.
