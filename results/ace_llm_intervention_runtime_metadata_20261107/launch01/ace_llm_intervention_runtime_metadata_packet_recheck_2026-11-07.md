# Runtime metadata packet delta recheck

November 7, 2026. **Remaining required corrections: 0 within the original review and this narrow delta.** R1 is withdrawn as a reviewer false positive. The genuine R2 reporting defect is resolved. The optional node-local custody clarification is also resolved.

The initial review remains byte-for-byte unchanged at `ace_llm_intervention_runtime_metadata_packet_review_2026-11-07.md`, SHA-256 `16f7a32e1efb1d303c0c7df2de8732179567ef09ddf0a9bc47ef4bf83a7a3ed1`. This document supersedes its R1 finding and its count of outstanding corrections. It does not replace its bounded implementation assessment or extend it to actual runtime qualification.

Only this recheck document was authored. Operations were local file reads, hashes, comparisons, JSON parsing and static AST inspection. No writer probe, test suite, launcher, diagnostic, copied package or model was executed. No remote work, allocation, source edit, ledger edit or commit occurred. The unchanged diagnostic implementation and earlier reviews were not reviewed again.

## R1 erratum

The original launcher already wrote a real newline. My initial review misinterpreted the escaped representation across the two Python source levels. Its claim that the writer appended bytes `5c 6e` was wrong.

The preserved BEFORE launcher has the originally reviewed SHA-256 `7d537f86ad3ee5c6400d51dc24a87f6f4dc2f342c1384d838535d160f7e5ef5b`. Parsing its outer `common` string and then the embedded writer source, without executing either, gives a `write` argument of **length 1, code point 10**. This independently resolves the escaping question.

The parent's saved actual-old-writer probe agrees. `old-writer-fixture/journal.json` is 20 bytes, decodes to `{"check": true}`, and ends with bytes `7d 0a`. Its full bytes match `R1-clarification.json`'s recorded hex. The new `chr(10)` expression is therefore a clarity change, not a bug fix.

The three prior raw journals were read and parsed only to settle this disputed finding. All match the recorded hashes, parse as JSON, and end with a real newline:

| Retained journal | Bytes | SHA-256 |
| --- | ---: | --- |
| `readiness.json` | 9,601 | `f6af11d9947865224839b1ce4d0d1d9f3177feaa161e82d5ecf0048af6b1c3b5` |
| `submission.intent.json` | 10,898 | `dfe1b1d5c9e01cfa9969eca261016a5053135e538bc3aa2606d289653cf9fcfa` |
| `submission.result.json` | 220 | `67157d506deb9052199bc15de81268cbb952fa4a586682a5ba23591e185828a4` |

These files are under `/Users/pat/code/ACE/results/ace_llm_intervention_staged_runtime_20261107/terminal01/remote`. They were not rewritten. The `legacy_scope` prose in the retained `correction.json` incorrectly implied a shared-template defect. Its own three `json_error: null` results do not support that implication. The subsequent `R1-clarification.json` explicitly corrects it. Neither that obsolete prose nor the withdrawn initial finding should be used to infer invalid historical journals or a historical custody failure.

## R2 disposition and documentation delta

The collector's sole implementation delta replaces the unconditional `runtime_archive_reused_unchanged: True` claim with `planned_runtime_archive_reuse: True` and an explicit `observed_staging_scope` stating that the collector does not infer completed staging. It directs subsequent assessment to the retained stage receipt and independent runtime review.

This resolves the original counterexample. A failure before staging can now be collected without being labeled as verified reuse. The existing complete-custody status remains pending independent review, and model-service qualification, scientific-result and runtime-adoption flags remain false. A failure after successful staging can still retain that separate stage evidence without the collector conflating it with import success.

The guide now says that original source/archive and retained job evidence remain in custody, while the node-local copy is ephemeral and represented by its verified stage pins. This closes the optional clarity note. It does not claim that an uncompleted future stage has already been verified.

The launcher delta consists only of the equivalent `chr(10)` expression and changing its required committed review path to this recheck. The collector delta is the reporting-field change above. The guide delta is the two-line custody clarification. The generated diagnostic, job script and packet are unchanged. Current diagnostic and test source hashes still match the previously reviewed bytes. The older preparation receipt continues to bind the older guide, while this recheck binds the clarified guide below.

## Evidence and current bindings

All 11 files in `launch01/parent-correction01` match their external counterparts. The preserved BEFORE review matches the original review, each AFTER source matches its current implementation, and the current ten top-level launch files match the external launch copies. This checks the new delta's custody without repeating the 145-file preparation review.

The correction evidence roots are:

- `/Users/pat/code/ACE/results/ace_llm_intervention_runtime_metadata_20261107/launch01/parent-correction01`
- `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-20261107-01/parent-correction01`

All pins below are SHA-256 of bytes read during this recheck.

| Object | Current SHA-256 |
| --- | --- |
| Launcher | `c93e0791abf7c19791a8985336d4ce12562c6aa556cc3131c7fb1598f6421a4d` |
| Collector | `9f757547b71880fcb47d76fcebedf0cd04370172375bfc131ea6301ca3099b3d` |
| Clarified guide | `caf26212848af6f12a537784fcf92b05469b4c5a6fa964697391d7c1e9c02f01` |
| Generated packet | `c32df16f0f0960f25d8871b40d2ecd970ef0d3b93ffd352dcb4075f3cb5f7c63` |
| Generated diagnostic JSON | `a6f1fda50fc238b8c4b19e2113fc5bbc6fafdcf315cf21cf45677a3a41b3e555` |
| Generated job script | `69d64504ca8fcf678a27684567f82ec77f777e06365a39d7f5375979973b2c6c` |
| Diagnostic source, current and launch copy | `9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49` |
| Test source | `05b22530a8a2c28e08fc9a81df07fcb5b53eb4f95e54a1ea10f183ca85f49b5f` |
| Archive helper | `7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e` |
| Probe | `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951` |
| Packet preparer | `5d77f420307db017f720217032cba28324252da91082bbb696f7040905f3c0e1` |
| Monitor | `b0b3854768a9ba849521b66ef0233a7c63e9947969fc7bc43201fc454e85e3c1` |
| Correction receipt, legacy prose superseded | `2325d1d1872f3f2b62692ae04e0587d878576253146bce9706d4da646bb25c6d` |
| R1 clarification receipt | `f87c19377b346a9f41c6d49636015fd58e91eedf0b7214ff38e718caf9831a0d` |
| Saved new-writer probe output | `95c85d0d99ff946143f8042dbaa0725873e88819ce15b9853701d55598971377` |
| Saved original-writer probe output | `22602c004be57442e00c85896eb40f36e13cd8f7278c4625b3c9cf29d71cd06f` |

The parent's new-writer probe receipt records approximately 0.001868 process CPU seconds and 0.00221425 wall seconds. Its original-writer clarification probe records approximately 0.001764 process CPU seconds and 0.001825042 wall seconds. These are saved local parent measurements, not work rerun here, not old fixture-suite reruns and not whole-cycle costs.

The request and admission are unchanged: one CPU, 3 GiB, ten minutes, zero GPUs, with 6,215 closed CPU seconds plus the 3,600-second preparation allowance and 600-second proposed reservation totaling 10,415 CPU seconds. No new allocation is established by this review. The launcher's fresh controller, journal, scheduler, transport and committed-source checks remain execution prerequisites. Successful staging, actual imports, cleanup and complete terminal custody remain future evidence requirements. This recheck closes the static review findings without qualifying the runtime, model service or scientific study.
