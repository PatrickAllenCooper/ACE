# Campaign GPU01 concrete packet review — 2026-11-08

**Required findings: 0.** Bounded static review of the active `launch01/build_packet.py`, `launch_once.py`, concrete packet/job, resource admission, and root/journal integration. Existing GPU wrapper/phase and CPU03 execution acceptance are reused; their closed reviews and component suites were not reopened. The separately retained, unexecuted builder prototype is not the active artifact.

Reviewed identities:

- Source commit: `12c7322e5046cd3e9b38cb59a1c01ed87054b71b`.
- GPU packet: `970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976`.
- Builder: `00c5f97c622695576798d52feff6fab79e2e9e6e14c6b4733cbb93bb422543ae`.
- Launcher: `190e222faa7e7ef13c0336ab56f52ddf2a7b96f0694a865741b66f5b2aaa81ed`.
- CPU03 acceptance: `11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f`; review receipt: `35100504f8279b2f96006706c6f440e5db1d92d35b65e2255ba336431ca1c3d7`.

Local read-only checks found all 79 input files and all nine metadata-pinned root artifacts consistent with their pins and the external `ACE-LLM-intervention-campaign-gpu-20261108-01` mirror; metadata also matches across copies. Original CPU inputs are byte-identical to retained CPU03 custody. The acceptance retains its exact 16 fields and matches its original authority file. The CPU review receipt records zero required findings and binds the unchanged review. The five added source files match the stated source commit, including outer `58f64d3e…` and phase `c0783ad8…`.

The new GPU launch root owns inputs, logs, readiness evidence and submission journals. Actual GPU output remains `ace_llm_intervention_campaign_artificial_20261108_03/gpu03`; all three frozen contracts retain their original CPU input paths and exact worker/launch/stage/binding destinations. No original contract or CPU input relocation is introduced.

The builder derives the 34-job/70-journal predecessor census from CPU03's retained lineage. Independently recomputed closed costs are 9,702 CPU-seconds and 1,347 GPU-slice-seconds. Adding 3,600 preparation CPU-seconds and the new maximum reservations of 4,800 CPU-seconds/600 GPU-slice-seconds gives **18,102/1,947**, within **57,600/7,200**. The job requests one `a100_3g.20gb`, four CPUs with admission reserving up to eight, 15,360 MiB, 600 seconds, `ucb736_asc2/aa100/gpu-testing`, and no requeue. Recorded resource rationale and the closed implementation's first-work/progress/cleanup bounds are reused, not requalified.

Launcher control flow holds both existing controller locks through readiness, transfers, verification and submission. It requires committed local/external packet bytes and an independently pinned committed concrete review; checks exact related terminal-job rows and journal hashes; checks account/QOS, partition memory ratio, configured GPU type and scratch write/read/cleanup; reuses strict CPU admission and full original input comparison; and reads back every transferred pin before submission. It rechecks admission before exclusively writing the remote intent. The new remote root has no submission journal during predecessor reconciliation, so it does not contaminate the expected 70-journal census. There is one `sbatch` call and no retry loop. Submission/transport uncertainty leaves intent and failure evidence for reconciliation, and existing-attempt/root guards prevent blind reuse.

No builder, launcher, project validator, tests, remote commands, stages, models, GPU work, response generation or private scoring were executed in this review. Only local source/data reads, hashes, comparisons, arithmetic and syntax parsing were performed; only this review document was written. Remote readiness, transfer/readback and scheduler admission are future launcher gates, not observations made here. This disposition closes the concrete static-review blocker only; it does not qualify GPU execution or authorize a successor/finalization.
