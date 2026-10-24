# Notice-status correction — independent code delta review

**Verdict: 0 required fixes.** The reviewed source correction addresses the code-level requirement in P1 of `delivery_final_package_review_2026-10-24.md`. Actual candidate02 construction, its independent pins, semantic/privacy review and final author README are separate follow-ups. This review does not declare P1 closed for an unbuilt successor or grant release/submission approval.

## Exact scope and pins

Reviewed only the notice-status delta in the planner and publication checker against committed `681bda3076f09930346cc8b648fada15279f8869`, plus the new fixture source and its saved outcome. The two preserved `notice_correction01/before/` sources exactly match that commit. Earlier acceptance, reporting, scientific, accounting and bootstrap reviews remain inherited evidence, not repeated work.

- Planner SHA256: `d8f52818f8041c902ac1a19d0e636f8ae6c5ebe56ca5bbf93d0cef4a02cb49c3`.
- Checker SHA256: `f5c753caf846606018365b2011b209b9d3f369df1f89011a203d18cbc08b4be0`.
- New test source SHA256: `969c2151d5abc245c537303b473dec369a8dffb3164943bca53a992af65bfee0`.
- Saved execution SHA256: `d1f89c8b0c0cd5e1d84e8bbec4cef1a4d8506e8125e41f464da4fddf42b42839`.
- Saved stderr SHA256: `1c061a413e7a49e4c84c5fb04c2ac537016352f727c64302b9165156e1cfa7ab`.

## Findings and disposition

1. **Authoritative status is bound.** The planner inserts `notice_status` into the authoritative successor contract. The checker requires this field and compares its canonical JSON bytes with the complete expected object. Missing/extra fields, changed status strings or digests, and integer `0` substituted for boolean `false` cannot satisfy that comparison. Non-finite JSON serialization is rejected. The outer approval flags remain independently required to be exactly false.

2. **Historical and current states are distinguished.** The helper identifies `notices/provenance.json`, its hash and `/remaining`, and explicitly limits supersession to the historical pending list. It separately describes accepted original B, privately accepted candidate03 checkpoint replay, both accepted reports and the reviewed identified-allocation cutoff. It expressly denies numerical replay of the new final manifest, whole-project CPU measurement, anonymous original-training qualification and human release/submission approval. It neither reopens owner authority nor grants a new license.

3. **The evidence joins are consistent with the existing trust boundary.** Notice and original-B hashes come from the authenticated candidate03 manifest. Replay/report acceptance hashes come from the private evidence set populated after the existing production acceptance gates. The replay remains tied to candidate03 and its fixed accepted replay pin. Accounting references the same public artifact and private digest checked by the existing accounting disclosure and artifact binding. Private acceptance digests are references to retained private proof, not public re-execution evidence. The original-B edge authenticates the unchanged inherited `B/acceptance.json` artifact; it does not relabel that artifact as the complete private original acceptance bytes.

4. **All four new edges and original preservation are correct.** The new JSON pointers bind historical provenance, MIT notice, license scope and original-B acceptance. The existing artifact edges already bind the public accounting object. The exact appended-binding check requires these edges; complete predecessor entries and the original binding prefix remain unchanged. The planner copies the predecessor and rejects publication collisions with different bytes. No original notice is rewritten or removed.

5. **Privacy and approval scope are explicit.** `strictly_anonymous`, `human_anonymity_handling_approved`, `anonymous_original_training_qualified` and `human_release_submission_approved` are false. The required human action concerns venue handling of attribution-bearing materials while retaining notices. The status introduces no private location/account fields; private evidence remains digest-only. The retained MIT and scope files name Patrick Cooper, so the explanation that mandatory attribution names the rights holder is supported. No strict-anonymity guarantee is asserted.

6. **Executable bootstrap is unchanged.** The diff changes no dependency pins, captured-source authentication, import path, execution order, CLI trust inputs or generic verifier behavior. The corrected checker itself needs its new independent digest in candidate02, as required by the existing workflow.

Read-only spot inspection confirmed the four referenced inherited artifacts match the independently pinned candidate03 manifest (`b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9`): provenance `f366135e1e65246630355b3784e1f85b0f89a0b8d5f08a2a88a348a18e5c9463`, MIT `206c4b6a50d0291622cae04f1ec98c6bf64609b53836f727b0d470e00528491c`, scope `69d2ce3c2e0bc7e3297d01fb40ef6e54a1a9428f6e38b98fabc5e755b6cf0af8`, and B acceptance `5808b925bb69e8b2fb6e9d15fd80880c5a32adeecf78791630d54decf7b8ee9e`. This was not package execution or a repeated full inventory check.

## Saved fixture outcome and limits

The existing `notice_correction01/execution.json` records exit 0. Its source pins match the current three files; stderr reports **3 methods passing**, including **9 negative subcases** (five overstated/type-invalid flags, three missing fields and one changed historical digest), and the four pointer-edge checks. Recorded child CPU is 0.040634 seconds, elapsed 0.04341837501851842 seconds and peak child RSS 26,099,712 bytes. These are the parent's saved fixture measurements, not this review's cost or a fresh run.

No tests, package builders/checkers, reports, models, pilots, remote work or commits were executed. Only this review file was authored. Parent may proceed with a new exclusive candidate02 while preserving candidate01. The separate actual-package/privacy reviewer must check the resulting bound status and README; human handling and final submission approval remain pending.
