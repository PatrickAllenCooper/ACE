# Scientific mount-alias delta review — 2026-11-08

**Current disposition: 0 required findings; R1/R2 closed by delta02 below.** The initial two findings and subsequent scope correction are preserved. Static review of the new known-root correction and its integration only. This is not failed-attempt acceptance, a recovery amendment, or permission to rerun job35402725.

## Required findings

**R1 — CPU campaign only: preserve physical disjointness when admitting the mount alias.** `prepare_llm_intervention_scientific_campaign_cpu.py:783` calls the old campaign base `B.separate()`, whose implementation at `prepare_llm_intervention_campaign_cpu.py:183–186` compares only lexical paths. The new helper validates an output individually; it does not compare its resolved identity with the protected roots. For example, a missing output `/scratch/alpine/<suffix>/inputs/new` and an existing packet tree `/gpfs/alpine1/scratch/<suffix>/inputs` pass lexical separation despite the output being inside the input tree. Exclusive `mkdir` does not prevent creating that new child. The old output-spelling rejection blocked this particular logical-output case; the correction now admits it.

Apply resolved equality/ancestor comparisons to the complete existing separation operands before any output claim, preserving the original path strings. Cover all CPU-campaign separation sites: authentication at lines229–230, parent staging at line727, and preclaim at line783 (line numbers refer to reviewed source `75a7179a…`). Keep the closed base helper immutable if desired: the new CPU-campaign wrapper can supply resolved operands or use a small local comparison. Do not weaken leaf/member symlink guards, hashes or exclusive publication. The existing finalizer's `separate()` at lines123–127 demonstrates the intended distinction between contractual spelling and physical overlap.

**R2 — Collection still rejects the admitted output alias.** The unchanged `collect_llm_intervention_scientific_training_cpu.py:339–340` requires `output == output.resolve()`. The patched supervisor passes the unchanged logical `out/'task'` to `collection.run_collection()` at lines318–320. On the admitted compute alias, collect mode therefore passes the new outer guard but fails before its task directory is created. Correct this boundary with the same narrowly admitted identity/known-root policy, or deliberately reconcile the physical task path and all consumers while retaining contractual identities. This is required before claiming collection-stage alias compatibility; it is not a claim that the realize-only recovery necessarily enters collection. The seven helper tests do not exercise this downstream call.

## Exact reviewed bytes

At the initial review, files equalled their retained `test01` snapshots; successor pins are recorded in the delta02 disposition below:

| Source | SHA256 |
|---|---|
| `supervise_llm_intervention_scientific_cpu.py` | `e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8` |
| `prepare_llm_intervention_scientific_campaign_cpu.py` | `75a7179a8e27be4425b148c9bae706b81aaceb706c1d55a84ccf018e798fef82` |
| `qualify_llm_intervention_scientific_campaign_gpu.py` | `58d226310940c9956979ace8c46e178c31f109c5e8374c1053a6286eef102be3` |
| `test_llm_intervention_scientific_mount_alias.py` | `69116c229d8aa5f8f2d6cce4433de6ace53b48322bd147c3e07aeecbfccf9d1b` |

The retained predecessors are the adopted `794fd63a…`, `8938d0d3…`, and `1a4ce8c4…` sources. Their diffs contain the two inline helpers, replacement of the output-spelling guards, new start-record observations, and the sole dependent `CPU_ENTRY_SHA` change in the GPU outer. No other implementation changes were present.

The helper pairs are AST-identical. Production callers fix `/scratch/alpine` and `/gpfs/alpine1/scratch`; arbitrary alternate roots are not packet inputs. Identity is accepted. Otherwise the code requires the exact resolved suffix, canonical physical root, equal root device/inode, and no descendant symlink. It preserves logical output paths and packet bytes and records the observed binding before authentication/staging/work. These are appropriate local checks for the known alias, subject to R1/R2; they do not establish a fresh compute-node observation in this review.

## Evidence, consumers and remaining scope

Evidence root: `/var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T/ace-scientific-mount-alias-preparation-krw3_3dg`, named by `/tmp/ace-scientific-mount-alias-preparation-current.txt`. `test01/receipt.json` SHA256 `4c6bd17b8850706fd067a9c558d684444e6b1b1e3ee18c88b682de9658b197d2`; stderr SHA256 `d013071211f59c218f90034596b68bd4e84e946098952c74a13d77d4436652d4`. Retained stderr records seven new filesystem-only methods passing, receipt returncode0, child CPU0.126292s, elapsed0.12998875s, peak child RSS24182784 bytes. Tests extract only the new functions using AST. I read this evidence and test source; I did not execute them. These checks cover the helper and static call/pin presence, not complete stage execution or resolved cross-root separation.

The added `output_path_bindings`/`output_path_binding` fields are in outer/worker `start.json` records. No exact-key consumer of those records was found in the surveyed scientific chain. GPU/finalizer exact campaign-root checks concern the separate campaign `start.json`, which this delta does not change. Logical path strings, scientific output contracts, deadlines and cleanup remain unchanged.

The GPU outer's line36 `CPU_ENTRY_SHA` equals the reviewed CPU-campaign bytes; its diff is that literal only. No production source hardcodes the new scientific GPU outer's hash. Finalizer `CPU_SHA`/`GPU_SHA` identify old base helpers and remain unchanged. Replacement metadata must propagate the supervisor pin through realize/collect; CPU-campaign pin through CPUcampaign/GPUcampaign/finalizer; and changed GPU-outer pin through GPUcampaign/finalizer. The realization authoring builder's independent entry pin and subsequent review/closure/freeze/packet bindings also need prospective refresh. This review does not bless an amended candidate or alter adopted/failed artifacts.

The bounded survey of candidate01's 67 pinned sources found no further scratch ancestor-alias rejection in token preparation, GPU training, GPU campaign or finalization. Their relevant resolved comparisons enforce disjointness, while symlink/`O_NOFOLLOW` checks protect leaves or members. The strict runtime provenance guard receives canonical staged-runtime paths; the NVIDIA canonical-device guard concerns `/dev/nvidia*`. Neither should be loosened for this change. The October23 recovery note documents the historical alias on `c3cpu-e2-u1`; the reported denied fresh SSH observation is not affirmative path evidence, and no bypass or fresh-node claim is made here.

No optional suggestions. No project/model/scientific execution, tests, remote action, source edits or commits were performed. Only this review document was written.

## Scope correction before successor recheck

Parent correctly identified an error in the initial R1 attribution: the two wrappers use different `B` modules. The CPU supervisor's `_base()` loads `finalize_llm_intervention_campaign_cpu.py`, independently pinned to `88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f`; that file's `separate()` at lines119–124 already resolves every operand. I verified both the working helper and adopted candidate01 copy have this exact hash and read the actual binding and implementation. **The CPU supervisor has no R1 separation defect and requires no separation change.** My earlier survey and initial review incorrectly attributed the campaign base's lexical-only implementation to that supervisor. R1 above is corrected to CPU campaign only; R2 remains the collection boundary finding.

This is a correction to the review, not a successor-code disposition. The announced CPU-campaign local resolved separation and collection known-root correction await final source/test pins and focused retained evidence. No tests or project code were executed for this verification.

## Final focused delta02 recheck — R1/R2 resolved

**0 required findings on the exact successor bytes below.** Compared CPU-campaign/GPU changes against retained test01 and the collection change against adopted candidate01. All listed current sources match delta02 snapshots and their retained receipt pins.

| Production source | Final reviewed SHA256 |
|---|---|
| CPU supervisor (unchanged from test01) | `e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8` |
| CPU campaign | `ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893` |
| Collection wrapper | `2bc872349f9fbfbafe1564b751cde899705f487f3bf822cce5518e0865bd9a4d` |
| GPU campaign, dependent pin only | `2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1` |

R1 is resolved by CPU-campaign `separate()` at lines107–109: it resolves every operand before invoking the unchanged lexical base comparison. All four existing call sites use this adapter (lines234,235,732,788); only the adapter calls `B.separate`. Thus equality and either-direction ancestry through the mount alias reject, while logical contractual strings remain untouched. The CPU supervisor requires no corresponding change: its separately pinned old-finalizer base already resolves operands, as corrected above.

R2 is resolved by collection's known-root helper at lines45–69 and its preclaim call at line366. It retains exclusive `mkdir`, passes the logical path unchanged through authentication and collection, and records the binding only in the wrapper `intent.json` at lines372–376. The helper AST matches the two outer helpers after normalizing only `B.need` to `_need`. The new observation does not alter the nested `collection/intent.json` consumed by token preparation or its existing plan/hash lineage. No new exact-key record-consumer conflict was found. GPU `CPU_ENTRY_SHA` equals `ba439d81…`; its only additional delta is this dependent literal.

Retained evidence is `delta02/` under the same preparation root. New test source SHA256 `3b6dbb438372765818323c08b42cd48b09f69f812b2ccf00a426a416a57a4908`; receipt SHA256 `02a9eabc890ec2a51e0c61d9c541c98200fc16f07d8e253baefe6c2e54c171f7`; stderr SHA256 `5e3c1713199f98fb396d3d140cdca938afe13d84fa3d581bd8d3b13ddc34b6d3`. The receipt and stderr record three new methods passing, returncode0, child CPU0.062994s, elapsed0.066971084s and peak child RSS23904256 bytes. Their scope is resolved separation at both outer boundaries; collection reaching a substituted artificial authentication stop while retaining logical paths, failure evidence, exclusivity and symlink rejection; and helper/dependent-pin agreement. The collection fixture imports only the stdlib-only wrapper and stops before captured scientific imports or collection/draw. The original seven methods were not repeated. This reviewer read the source and retained evidence without executing tests or project functions.

Closure comment: comparing current files against all 67 candidate01 `source_pins` yields exactly the four production changes in the table, with no new production file required. Roster membership stays 67. Collection's new hash additionally propagates through the collect-stage source maps; the previously identified supervisor/campaign/GPU propagation still applies. The old base-finalizer `88102baf…` and base-campaign `850ca3d4…` remain unchanged. This is a source-closure delta observation, not a newly built/adopted freeze or final Git binding. Prospective review/closure/packet metadata must bind these successor bytes without rewriting candidate01 or failed-attempt custody.

No optional suggestions, stage execution, model/scientific calls, remote actions, source edits or commits. Final disposition is limited to this infrastructure correction and its stated static integration scope; it does not assert the current compute-node mapping or authorize recovery execution.
