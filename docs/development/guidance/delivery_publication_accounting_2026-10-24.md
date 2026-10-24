# Publication accounting integration review — October 24, 2026

**The frozen through-replay12 allocation total reconciles exactly: 128,690 CPU-seconds (35.7472222222 allocated core-hours).** The original B chain and two pilots contribute 118,925 seconds; eleven later runtime-preparation/replay allocations contribute 9,765 seconds (2.7125 core-hours). These are 57 distinct allocations: 44 original B allocations, two pilots, two environment preparations, and nine replays. There is no unexplained allocation difference. This is the identified B/reproduction chain, not total project process CPU.

This new, local integration reads the latest execution ledger, publication-completion plan, final original accounting and preserved scheduler/attempt receipts. It inherits original B and replay12 acceptance. It does not repeat the accounting implementation/tests, scientific audits, model execution or complete custody audits. No remote operations or Git commit occurred. Source paths, hashes, raw parent rows, separate measurement scopes and arithmetic are in [incremental_accounting.json](/Users/pat/code/ACE/results/delivery_publication_accounting_20261024/incremental_accounting.json).

**Report01 is separate and pending.** The preserved [launch journal](/Users/pat/code/ACE/results/delivery_prospective_reporting_20261024/launch01/submission_result.json) records one submission, job **34864950 at 20:44:10 UTC**, with 1 CPU, 3 GiB and a 30-minute cap. Its actual allocation cost and reporting outcome are pending and excluded from 128,690. Reservation becomes **93.368333/150 core-hours**, including its 0.5-hour cap; that cap is not actual cost. The cutoff remains completed replay12 job34863006 at 20:18:31 UTC. The earlier reporting-preparation document is not evidence of report01 completion.

**Original B baseline.** The [final 44-allocation receipt][original-allocation] matches the parent/task rows in [final-sacct.txt][original-raw] (SHA256 `91fd2b333b077446eff5e79a4225b1564c4a08caf6709ad751cac6feba5489fc`). Qualification, collection, 20 small-world fit allocations, 20 large-world fit allocations, evaluation and audit contribute respectively **35 + 10 + 14,751 + 103,728 + 132 + 24 = 118,680** seconds. Failed pilot33505661 adds **126** and successful pilot33507335 adds **119**, giving **118,925**. Child `.batch`/`.extern` rows are excluded; array-task raw/display identities are retained, including task19 rather than an additional array-parent total.

The [final original fit/phase accounting][original-fit] (SHA256 `ba3427cd5866df063d61ed577124e510ce1988c47e55c0ab4776baecd1a4e5c3`) records **640 completed fits, 40 world journals, 640 distinct fit attempts and zero failed fit attempts**. Its **110,802.107301787 fit process CPU-seconds** and **113,596.44237109905 summed fit elapsed seconds** supersede the old 578-fit accounting snapshot. They overlap allocation and are not added to it. Original qualification/collection/evaluation/audit supervised elapsed times are 28.133447/4.311393/127.555273/20.427793 seconds; their whole-child process CPU remains null in that receipt. Failed supplemental jobs do not relabel accepted original fits as failed.

**Incremental attempt disposition.** Every amount below comes from the identified job's preserved scheduler row, corroborated by its terminal disposition. All eleven allocations used one CPU and 3 GiB, with no GPU. The JSON retains each raw row, receipt reference, wrapper elapsed time and separately labeled Slurm TotalCPU.

- **Replay03, 33539781: +182 → 119,107 seconds.** Failed before checkpoints: imported NumPy 2.2.5 disagreed with required 2.2.6; Python 3.10.19 was below the supplemental minimum. [Raw scheduler][r03]; [disposition][d03].
- **Environment preparation01, 33556219: +8 → 119,115.** Base-interpreter guard failed before installation; no preparation commands or environment creation. [Raw scheduler capture][env01]; [disposition][env01d].
- **Environment preparation02, 33560677: +169 → 119,284.** Bootstrap/install/pip-check completed; fileless `torch._C._dynamo.autograd_compiler` rejected qualification. No qualified replay. [Receipt with raw `scheduler_rows`][env02].
- **Replay05, 33625983: +60 → 119,344.** `Popen E2BIG` prevented child creation; no target imports. [Raw terminal accounting][r05]; [disposition][d05].
- **Replay06, 33640933: +69 → 119,413.** Fileless `torch._VF` provenance rejection before checkpoints. [October 9 raw scheduler][r06] and [October 23 preserved scheduler confirmation][r06later]; [receipt][d06].
- **Replay07, 34821094: +548 → 119,961.** Compute-node directory alias/source-pin comparison failed before checkpoints. [Raw scheduler capture][r07]; [disposition][d07].
- **Replay08, 34824953: +299 → 120,260.** Delegated `torch.backends.cudnn` provenance rejected before full inventory/checkpoints. [Raw scheduler capture][r08]; [disposition][d08].
- **Replay09, 34835547: +299 → 120,559.** Backend correction passed; `cython_runtime`/`six.moves` rejected before full runtime qualification/checkpoints. [Raw scheduler capture][r09]; [disposition][d09].
- **Replay10, 34837216: +1,766 → 122,325.** Initial runtime qualification passed; supervised deadline ended child with −9 and complete cleanup. No first-checkpoint/full-replay receipt; exact interrupted subphase unknown. [Raw scheduler capture][r10]; [disposition][d10].
- **Replay11, 34861753: +3,576 → 125,901.** Same recorded acceptance boundary and supervised deadline failure; exact interrupted subphase remains unknown. [Raw scheduler capture][r11]; [accounting][d11].
- **Replay12, 34863006: +2,789 → 128,690.** COMPLETED 0:0, full runtime/first-checkpoint/numerical replay and cleanup accepted: 640 checkpoints, 240 primary cells and 48,000 cached responses; zero new fits, optimizer updates or responses. [Raw scheduler capture][r12]; [acceptance][accepted12]; [accounting][d12].

Thus **182 + 8 + 169 + 60 + 69 + 548 + 299 + 299 + 1,766 + 3,576 + 2,789 = 9,765**. The two environment failures contribute 177; eight failed replays contribute 6,799; accepted replay12 contributes 2,789. The through-replay11 increment is 6,976, independently recovering **118,925 + 6,976 = 125,901**, then **+2,789 = 128,690**. Unsubmitted packet/preflight versions and duplicated custody copies are not additional allocations.

**Measured phases and preparation limits.** Replay12 separately reports **852.158 Slurm TotalCPU seconds**, **2,770.269435 wrapper elapsed seconds**, and **110.613683783 replay-function CPU / 1,460.476566117024 replay-function elapsed seconds**. Final diagnostic child cumulative CPU is 123.578901723 seconds. These scopes overlap. Batch MaxRSS is 617,620 KiB; supervisor sampled peak tree RSS is 739,577,856 bytes, a different memory scope.

Raw `runtime_phase_005/006`, `011/012`, `014/015` and `018/019` under terminal12/raw respectively measure **689.536479** initial-inventory elapsed seconds (19:35:08–19:46:37 UTC), **841.329041** CLI-to-first-checkpoint-entry seconds (19:46:42–20:00:43), **705.658691** live-inventory seconds (20:00:43–20:12:29), and **279.868732** first-checkpoint-authorization-to-CLI-completion seconds (20:13:10–20:17:49). Exact endpoints, CPU deltas and raw hashes are in the JSON. These selected intervals omit intervening work and cannot establish pure inference cost or retroactively locate replay10/11 interruptions. Diagnostics alone are not acceptance evidence.

Local runtime/source preparations around03/05/06 and07–12 are separately represented by thirteen preserved receipt references in the JSON. Some record only a final fabricated-check subprocess: for example05 records 0.322943 child CPU / 0.467788208 elapsed seconds,06 0.353608 / 0.473118292, and duration11 preparation 0.059026 / 0.063659834. Backend09 and compatibility10 preparation CPU are explicitly unmetered. These selected measurements are not a complete preparation inventory or additional Slurm allocation. No old check was rerun.

**Remaining evidence limits for manuscript integration.**

- Preserve **unknown confirmation whole-process CPU and unknown complete preparation/review/transfer/monitoring CPU**, including this review. The inherited confirmation record has 36 refit elapsed intervals totaling 6,123.024536 seconds, overlapping its stage elapsed intervals; its nested charged/reserved 11,943.253200 seconds is not CPU. Prior A/C fit/function CPU scopes also remain separate. The [existing bounded accounting receipt][prior-accounting] supplies those inherited scopes; its partial B values are superseded above.
- Replay06 reports `TotalCPU=00:00:00` in both preserved scheduler captures despite documented child runtime/provenance work. Retain the raw zero with measurement uncertainty; do not infer zero work or construct a complete incremental process-CPU total from it. Its 69-second allocation is corroborated.
- Replay10/11 lack finer interruption timing. Replay12's mixed diagnostic intervals do not repair that gap. Terminal replay12 `scontrol` had expired; captured `sacct` supplies terminal state/allocation, while earlier execution/allocation evidence supports the accepted guards.
- Original final fit accounting and final raw scheduler evidence are in external local custody, with paths and hashes recorded here. Later publication packaging must bind the required accounting evidence; this review does not itself package or republish originals.
- Through-replay12 reservations are **86.868333 original + 6.0 later caps = 92.868333 core-hours**; report01 raises that to93.368333. Neither reservation figure is actual usage. Append report01's eventual terminal evidence outside this frozen review; do not substitute its 1,800-second cap.

Suggested manuscript accounting wording: “The identified Stage B and reproduction chain used 128,690 allocated CPU-seconds through the accepted saved-artifact replay (35.7472 allocated core-hours), comprising 118,925 seconds for the original study and pilots and 9,765 seconds for subsequent environment preparation and replay attempts, including failures. The original 640 fits separately recorded 110,802.1073 process CPU-seconds. These overlapping scopes are not summed. Whole-project process CPU, confirmation process CPU and complete preparation overhead remain unmeasured. The subsequently submitted report-only job is outside this cutoff and awaits terminal accounting.”

[original-allocation]: /Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_allocation_and_replay_proposal_20261008.json
[original-raw]: /Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-complete-custody-20261008T0417Z/final-sacct.txt
[original-fit]: /Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-original-compute-final-20261008-01/accounting.json
[r03]: /Users/pat/code/ACE/results/delivery_prospective_replay_preparation_20261008/failed33539781/scheduler_final.txt
[d03]: /Users/pat/code/ACE/results/delivery_prospective_replay_preparation_20261008/failed33539781/failure_disposition.json
[env01]: /Users/pat/code/ACE/results/delivery_runtime_recovery_20261008/delivery-runtime-preparation-job33556219-failed-20261008-01/terminal_inventory.json
[env01d]: /Users/pat/code/ACE/results/delivery_runtime_recovery_20261008/delivery-runtime-preparation-job33556219-failed-20261008-01/failure_disposition.json
[env02]: /Users/pat/code/ACE/results/delivery_runtime_recovery_20261008/failed33560677/failure_disposition.json
[r05]: /Users/pat/code/ACE/results/delivery_qualified_replay_preparation_20261008/failed33625983/terminal_custody_inventory.json
[d05]: /Users/pat/code/ACE/results/delivery_qualified_replay_preparation_20261008/failed33625983/failure_disposition.json
[r06]: /Users/pat/code/ACE/results/delivery_descriptor_replay_20261009/terminal06/sacct.txt
[r06later]: /Users/pat/code/ACE/results/delivery_vf_remote_preparation_20261023/remote01/failed06.json
[d06]: /Users/pat/code/ACE/results/delivery_descriptor_replay_20261009/terminal06/receipt.json
[r07]: /Users/pat/code/ACE/results/delivery_vf_replay_20261023/terminal07/remote_inventory.json
[d07]: /Users/pat/code/ACE/results/delivery_vf_replay_20261023/terminal07/disposition.json
[r08]: /Users/pat/code/ACE/results/delivery_vf_replay_20261023/terminal08/remote_inventory.json
[d08]: /Users/pat/code/ACE/results/delivery_vf_replay_20261023/terminal08/disposition.json
[r09]: /Users/pat/code/ACE/results/delivery_backend_replay_20261023/terminal09/remote_inventory.json
[d09]: /Users/pat/code/ACE/results/delivery_backend_replay_20261023/terminal09/disposition.json
[r10]: /Users/pat/code/ACE/results/delivery_compatibility_replay_20261023/terminal10/remote_inventory.json
[d10]: /Users/pat/code/ACE/results/delivery_compatibility_replay_20261023/terminal10/disposition.json
[r11]: /Users/pat/code/ACE/results/delivery_duration_replay_20261024/terminal11/remote_inventory.json
[d11]: /Users/pat/code/ACE/results/delivery_duration_replay_20261024/terminal11/terminal_accounting.json
[r12]: /Users/pat/code/ACE/results/delivery_diagnostic_replay_20261024/terminal12/remote_inventory.json
[d12]: /Users/pat/code/ACE/results/delivery_diagnostic_replay_20261024/terminal12/accounting_and_phases.json
[accepted12]: /Users/pat/code/ACE/results/delivery_diagnostic_replay_20261024/terminal12/replay_acceptance.json
[prior-accounting]: /Users/pat/code/ACE/results/delivery_release_preparation_20261007/compute_accounting_preparation_20261008.json
