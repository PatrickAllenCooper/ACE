# Screen training GPU01 concrete packet/coordinator review — 2026-11-08

**Required findings: 1. This coordinator is not yet accepted for launch.** The concrete worker inputs, committed component-review bindings and job resource/bootstrap contract match; the outstanding defect is the launcher's historical accounting/custody baseline authentication. This is a bounded static packet/coordinator review, not a repeat of closed component reviews or tests. No coordinator, worker, tokenizer, model, test or remote command was executed.

## Required correction

**[P1] Compare the exact accepted 37 historical rows and 76 journal pins before submission.** In `launch01/launch_once.py`, lines 43–47 accept any 37 terminal rows with aggregate CPU/GPU totals 10568/1483 and any 76 journal files. The freshly computed journal hashes are never compared with an independently pinned accepted map. For example, changing one historical journal's bytes leaves the count unchanged and passes this check. Replacing/changing accounting rows while preserving totals also passes. The second `reconcile()` at line 86 repeats the same aggregate checks and installs its observations as the new remote intent's baseline. Consequently, the terminal collector's exact extension checks could preserve a baseline already changed before this launch.

Bind the existing accepted CPU terminal accounting extension (or the equivalent already accepted exact row/map data), and compare every normalized historical accounting row and every journal path/hash at both readiness and immediately before the remote intent/`sbatch`. Require unique job IDs and exact roster equality, retaining the existing aggregate-cost checks. Carry that independently pinned baseline into the submission metadata. No new admission schema is needed. The existing source is `results/ace_llm_intervention_screen_prompt_cpu_20261108/terminal01/accounting-extension.json`, SHA-256 `7d0539c0531cb33a2fedbb3e0e86ed60b5b81e8611497d3c13df28c315a7e51b`; its `related_accounting` and `journal_pins` contain the accepted 37 rows and 76 pins. A focused correction/pin recheck is sufficient; no closed source or component tests need repeating for this review.

This finding binds launcher SHA-256 `ba309007b3127bd1e56a0ff83b93765a2b8051643abd64aeb9c47326386ad47a`.

## Concrete packet verification

Repository root: `results/ace_llm_intervention_screen_training_gpu_20261108/launch01`.
External mirror: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-training-gpu-20261108-01`.
Remote root: `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_training_gpu_20261108_01`.

Both local mirrors have identical relative file rosters and bytes. All 33 inner inputs and 28 source names match their packet hashes. Every captured source, the archive helper, and the three captured reviews match Git objects at source commit `01daec872ba5c2f4ac9466edf85a264524be143b`. The prepared artifact retains all 18 source bindings and its canonical plan pin. Capacity packet runtime/model fields are unchanged. The complete 39-object transfer is the outer packet's 38 `new_input_files` plus `packet.json`; locally retained builder, launcher, monitor and collector files do not silently expand that transfer.

The shell job uses explicit account **`ucb736_asc2`**, partition `aa100`, QOS `gpu-testing`, four requested CPUs, 12 GiB, one `a100_3g.20gb` slice, 600 seconds, no requeue, and root-specific stdout/stderr paths. The inner contract permits at most eight allocated CPUs. Shell token inspection confirms the exact captured diagnostic bootstrap, qualifier source hash, inner packet hash and qualification output path. No shell or bootstrap payload was executed during this inspection.

Reservation arithmetic is correct: `10568 + 3600 + 4800 = 18968` CPU-seconds and `1483 + 600 = 2083` GPU-slice-seconds, within 57600/7200 ceilings. The reservation uses the maximum eight CPUs for all 600 seconds. Sizing is a bounded artificial full-context smoke; the new singleton DPO shape's actual cost and peak demand remain unmeasured. Accepted preparation supplies four fabricated SFT and two fabricated DPO updates using saved 4638/4744-token prompts, with no GPU tokenization or scientific responses. Component reviews are reused, not re-adjudicated here.

The launcher checks committed transfer bytes, its captured launcher and the concrete review before remote work; takes both shared controller locks; checks related live jobs, account/QOS, partition memory ratio and slice availability; records scratch ownership and write/read/fsync/delete readiness plus free space; verifies transferred hashes and repeats hash verification before submission. Scratch free bytes are recorded rather than used as a numerical output-space admission threshold; the reviewed allocated stage independently checks node-local staging space. Exclusive local and remote intents precede the single `sbatch`; a timeout or ambiguous result retains evidence and does not trigger retry. Final packet/review/controller commit remains a future prerequisite. These controls do not cure the required baseline comparison above.

## Exact expected input and submission bindings

| Artifact | SHA-256 |
| --- | --- |
| Outer `packet.json` | `a50cfb5bee794ec20b5452c0183059cdc13c3f3038ed310b9d987912032b0838` |
| Inner `training.json` | `390336e2af6d239f310f7d99560c1e6342b0a3b868f7ed4eae59501cc447f460` |
| `job.sbatch` | `a5347ba3011efe334c6bb388c346bb5b035156f3581b00bc422731967712ce1f` |
| Qualifier | `4665d03e2b910921e5b7e9bc2d74e24d07278852d9d7d5b89ec1cc56932d2b56` |
| Worker | `ecf9d7099d2a038fba01f2eddf93d93680fe675396a7d21d248ad81f2892c53d` |
| Prepared file | `6ec85a1608d3e64c914b51b81605248be1a11dfbe879e0d463a9cc09b4da3712` |
| Canonical training plan | `0807b705f2bad022630629629e99a36f519b7e0f9e980464ec9a7b3385c10ded` |
| CPU capacity acceptance | `69446fb29e24c34cff070d2f9f699a5d1d8d3e9a478c9d438adc01be7a290e59` |
| Saved tokenization | `fb663a035eef20420fe461bc4dfa4fa6afa8462b38e32e1262c5661f666ff72c` |
| Worker review | `dd5d1f8b6b31022b69bed0aecaa2cf92cfeaf1bf8c6733d53520f7e64804e1f5` |
| Supervisor review | `aba5341108399e69d3364500571da60e856b0008de68ce87620a8e256056039d` |
| Actual-input review | `c97e82d34dfbb6dd425da0a69a09fd6903f43f40dd239ad1f8208f2cb8c30167` |
| Builder | `1ddead11830b8a08a8f7ec605281e27abb8f88be661f0701c83926a33a268bd1` |
| Monitor | `ef77ffbd46143ad1b83a14ad01220ed50caac6efaf4f3a455bb51b1a4cfcc651` |
| Terminal collector | `e4cd620b5d9d8b24023deeca814b4b975126414c8bcce6628fe6b7d3941a3080` |

All remaining input hashes are bound by the outer packet above. The canonical digest of the complete expected 39-entry submission `pins` map (`new_input_files` plus outer packet, sorted compact JSON) is `e8341ef97a6053492196e7d11362b57b7b642b22f6ae5ae601bf9cb818c815b0`. Future local intent must bind that map, outer packet, corrected launcher hash, final concrete-review hash, exact remote root, resource/admission objects and eventual packet commit. Its `source_commit` field currently denotes the eventual launch commit, distinct from the packet's component source commit. The remote result must bind the remote intent's exact bytes and a resolved job ID; local and remote intent bytes differ because the remote intent adds fresh observations. No actual submission-intent/result hashes or job ID exist for this unsubmitted review, and none are invented here.

## Copied monitor/terminal collector scope

The monitor is observational and records exclusive live receipts; it neither submits nor cancels. The copied terminal collector at `e4cd620b…` and `collector-preparation.json` are present in both mirrors; the collector source was read statically, not executed. No additional required collector source defect was identified within this custody-only scope, conditional on correcting the launch baseline above and committing the final collector with the packet.

The collector requires independent job ID, outer-packet SHA, local-intent SHA, result SHA and eventual launch commit. It checks committed transfer bytes, launcher, review and collector identity, claims both terminal roots exclusively, and preserves the full evolving local launch directories. It requires terminal accounting and target absence from the queue, inventories the entire new remote root twice, transfers every bounded regular file, and compares both local raw copies against both inventories. Bounds are 4096 files, 64 MiB per file, 512 MiB per inventory; exceeding them retains failure evidence rather than skipping outputs. Partial transfers are retained/mirrored when possible, with no automatic retry or job action.

Fresh census checks compare the exact 38-row/78-journal extension against the remote intent's 37/76 baseline and the two new submission journals. Actual owner/name/account/partition/QOS/request/allocation/time/no-requeue evidence and CPU/GPU counts are reconciled; uncertainty/conflicts prevent silently releasing the reservation or closing cumulative accounting. Complete checkpoints, private artifacts, logs and failure records are included by whole-root inventory, not a success-only whitelist. All training, runtime-adoption, GPU-qualification and scientific flags remain false pending independent actual-results acceptance. The collector does not repair a historical baseline accepted incorrectly at launch.

**Optional suggestions: none.** Launch remains pending the one required coordinator correction, a focused recheck and the final packet/review/controller commit. This review is not permission to submit.
