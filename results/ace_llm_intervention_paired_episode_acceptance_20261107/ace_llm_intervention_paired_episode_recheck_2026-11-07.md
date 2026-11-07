**Independent paired-episode delta and saved-evidence recheck — November 7, 2026**

**Paired-module delta: 0 remaining required findings; its original P2 is closed. Combined component acceptance remains on hold.** The separate [sealer review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_episode_seal_review_2026-11-07.md) identifies an open P2: saved terminal posterior means/covariances are not reconciled against accepted-receipt replay. That dependency correction, its independent delta review and the needed affected integration cases must pass before combined acceptance. The seven saved integration passes below do not discharge that requirement.

The [preliminary paired review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_paired_episode_review_2026-11-07.md) remains byte-identical, as do its copies and predecessor source snapshots in the new capsule. This separate recheck preserves the original finding against the earlier source.

The implementation delta is limited to [lines 194–204](/Users/pat/code/ACE/scripts/research/llm_intervention_paired_episode.py:194): the measurement adapter starts as `None`, is constructed only when the admitted acquisition bound is positive, and an impossible callback without its admitted adapter explicitly rejects. The nonempty initial adapter remains independent. Thus empty initialization with cap1 and minimum action cost2 reaches the ordinary episode budget-stop path without constructing a zero-bound adapter. The guide describes the prior-only endpoint. The only test change from the preserved reviewed version adds the seventh method; the earlier six and their corrected dual private-score guard remain unchanged.

The new [zero-request integration method](/Users/pat/code/ACE/scripts/research/test_llm_intervention_paired_episode.py:187) patches both adapter construction and policy choice to raise if called, then uses the actual sealing/reporting interfaces. Its authenticated saved artifacts show:

- An episode journal containing only `begin`, `proposal`, `stop`; no reservation, return capture or decision-input file.
- A clean complete source/outer disposition with reason `budget_exhausted`, zero initialized responses and policy decisions, empty history, prior mean0/covariance1 and zero actual spend.
- A complete private cost report with one shared-initial point, prediction0, normalized error1, zero known/unknown/accepted returns and zero initial/acquisition charges.
- A reported, nonfailed roster cell with nonattainment and capped acquisition cost1. The cap is an endpoint penalty, not actual spend.

These observations close the specific admitted-zero-request failure; no new execution was needed for this review.

Independently verified the new local capsule `results/ace_llm_intervention_paired_episode_preparation_20261107` and external `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-paired-episode-preparation-20261107-01`: **621 files / 1,148,353 bytes each**, all **620 manifest members** matching their sizes/hashes, identical manifests and no missing/extra files. All **15 source pins** matched both frozen copies and current files when checked. The preliminary review's preserved hash and its current bytes also match.

The receipt/log records **seven newly frozen methods passed in one invocation**, cwd `/tmp`, Python3.9.6, exit0, from `2026-11-07T17:49:40.726709+00:00` to `17:49:41.476070+00:00`. Recorded cost is **0.656927 child CPU seconds / 0.749015375 elapsed seconds / 34,504,704 bytes peak child RSS**. The receipt scopes CPU to the command child and its waited children; RSS is not an aggregate process-tree peak, and these are not whole-development costs. This review did not rerun that invocation.

Other new saved integration evidence was inspected for consistency with the original source review, without repeating the sealer implementation review:

- The public-process fixture preserves all five arms and all ten policy pairs. Capped costs are `[2,3,3,3,3]`; the informative arm ends at approximately0.04, stop/costly arms at0.25, the broken arm is failed, and the missing arm remains unresolved.
- The prepared-but-unreturned acquisition retains one unknown return and total actual cost2; its accepted initial response remains known. The invalid-return fixture records a distinct returned-invalid acquisition with total cost2. Partial initialization retains one unknown initial charge, no final error and cap3.
- Conflicting authenticated initial histories retain both cells unresolved at cap3. Changed-pin and failed-cleanup controls also retain explicit unresolved cells. All eight saved finalizations bind their summary/report hashes correctly and retain false scientific acceptance/provenance flags, with zero new finalizer responses or policy updates.
- The model-free persistent service's actual saved closure matches the copy supplied to finalization: PID/process group9793, return code0, no cleanup errors, group absent and no TERM/KILL sent. Arm counters go from count0/request−1 to count3/request2; closure records count3/request2 and the expected identity binding is retained. The negative cleanup fixture remains unresolved. These are one artificial service/world's receipts, not evidence of a real model, GPU work or reuse across multiple worlds.

The separate sealer finding is acknowledged from its review, not independently re-reviewed here. The capsule uses sealer SHA `b614744cecdd055720b8ec2fb35f245a75db569acacada38e0b9e1f2403f8478`, the version with that open finding. Preserve these results as evidence of their exact source freeze; do not relabel them as validation of a later sealer. Outer original source/model/runtime and closure authentication, multiworld supervision/reuse, resource admission and scientific adoption remain unestablished. No scientific, model or target-runtime qualification is granted.

This recheck used static diffs and standard-library reads/hashes of saved artifacts only. No project module, test, reporter, process policy, model or world generator was executed; no remote work, ledger changes, commits or snapshot edits occurred. This report is the only new file written.

SHA-256 bindings:

| Artifact | SHA-256 |
| --- | --- |
| Corrected paired module | `f4061bf16743212e4ae95b459749205595116f4f60db9cadd446c532509d07c3` |
| Seven-method test source | `29ad9f2f9240c18a479c1abeea66cff5c5ee449b023286e455121710e1480f55` |
| Corrected guide | `9457bd4af7fc993986dfcca474f1343dc1b56d5685a2c6dc309fa8867fe8dfba` |
| Integration manifest | `6878bb3ea550a234658cef89b8c29fe5c96aafa2fc91ad002fb7a642d1100576` |
| Integration execution receipt | `faec6885f6c0775fcc08c294ffd5c5fa0dd2d56a3474aa3c223feb7409d0b39a` |
| Integration closeout | `54a3557a45dfc73df99e43054969e59acffda34804171f902577ccbcfa7be73f` |
| Source manifest | `d1ec3e49ad6b140301a61e52e41fe04685b72e50591db5e87744ee548981ae75` |
| Preserved preliminary paired review | `99eafba619c2147f1f45d13fd0e25609871638cc74b837f82cbd508b57c3f4f7` |
| Separate sealer review read here | `57135c276ebc6d2d5a554e4a6cc44f4db99d2c3bc3483de82ce7cda7836fe46d` |

**Affected saved-finalization execution delta — November 7, 2026, 18:00 UTC**

**Result: passed; 0 new paired-integration findings. Combined acceptance remains conditional on Zeno's independent review of the corrected sealer.** This is the newly authorized affected integration check, not a rerun of the seven-method suite. The preceding review text is preserved; its pre-append SHA was `fb0827db2c9a4e90e928c357c7dd5304c763945283c96fd0bcd2c7533646efd4`.

Mencius supplied a ready handoff for corrected sealer `91b7a40829157cc21d586fd22ae17eee6556d1b73da1dfbf2f4b104564f39811` after four focused correction methods passed. The new [harness](/Users/pat/code/ACE/results/ace_llm_intervention_paired_episode_recheck_20261107/recheck_finalizations.py) first authenticated the original independent manifest `6878bb3ea550a234658cef89b8c29fe5c96aafa2fc91ad002fb7a642d1100576` and every member in both original custody locations. It then froze the complete 13-module import closure before launching one child from `/tmp`. Only the sealer differs from the original closure; all other project modules match the original frozen bytes. The correction readiness receipt is preserved with SHA `3f72b0c00a23cc96c0fb46ae7610ba24c56bc92adafd8ce61da6fb9cf0710a6b`.

The child reinvoked only the eight saved finalizers using original external arm paths, original plan/inventory bytes, original intentionally invalid pins and each finalization's saved persistent closure bytes. It retained every case, including negative cases. Environment adapters, world generation/noise draws, policy construction/choice and episode execution were patched to reject; audit guards also rejected subprocess/network/model-import activity and writes to the original capsules. All loaded project modules came from the frozen closure. There were **zero attempted blocked operations**, no environment responses, policy queries, models or scientific draws.

All **eight summary files / 17 roster rows are byte-identical** to their original files. All rejection files are byte-identical. All cost reports, sealed bundle members and other corresponding finalizer artifacts are unchanged. The only differences are **12 sealing-provenance files**, which now record posterior replay, matching whole-prefix posterior reconciliation, the explicit atol1e-10/rtol1e-8 comparison rule and the strengthened consistency scope. Original and new provenance, file hashes, rejection differences, per-case outcomes and logs are retained. The zero-request endpoint and every failed/unresolved cap cell remain unchanged.

The single child ran from `2026-11-07T18:00:33.236600+00:00` to `18:00:33.526894+00:00`, exit0, with empty stderr: **0.231226 CPU seconds / 0.290342083 elapsed seconds / 35,307,520 bytes peak child RSS**. The driver separately records **0.330756 self CPU seconds / 0.713769500 elapsed seconds** through closeout preparation. Driver CPU excludes child CPU, startup/imports and final manifest/mirroring; its elapsed interval includes waiting for the child. These are scoped measurements, not whole-development cost or a reset preparation allowance.

The new repository capsule `results/ace_llm_intervention_paired_episode_recheck_20261107` and exclusive external `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-paired-episode-recheck-20261107-01` each contain **273 files / 795,005 bytes**. All 272 manifest members, exact inventories and manifest bytes were independently checked after mirroring. The original 621-file/1,148,353-byte capsules were verified again after execution with no changes. The preliminary report remains unchanged. No parent implementation, source snapshot, ledger, CURC job or allocation was changed, and no commit occurred.

This affected recheck supplies the missing paired-finalization compatibility evidence for the corrected sealer pin. It does not replace Zeno's independent arithmetic/state-reconciliation review. At this append, that corrected-sealer disposition has not been observed; combined acceptance is therefore still conditional. Scientific, actual-model, target-runtime and multiworld-supervision qualification remain outside this result.

| New artifact | SHA-256 |
| --- | --- |
| Recheck harness | `dfd73216bbe8f661cb9308f8352a6368aa98ba5c89540f63c4c18e17ec4a1ea4` |
| Recheck manifest | `835ee4a725080575676030e61e04364de93c7752347b6c0479423345dc230572` |
| Frozen source manifest | `b068ff40ce8d7c7a00cfea470f9b5adff7b8677ede0fdbe0118a5951d0bc2a73` |
| Measured child receipt | `ab4a2772773bf08412a8d7da308b6bb1da4d1e20183f39620f04770928c1286e` |
| Eight-case worker result | `74f6d0990beb04abe8cc0da4e5b9972d562dc6673540d983a9be5efb37d76683` |
| Recheck closeout | `5729b60ea80b314506ad89a568a5469180c9f178344a071da1d9376e5a95b3c7` |
