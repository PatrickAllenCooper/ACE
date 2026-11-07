# Runtime generated-module diagnostic launch02 packet review

Date: 2026-11-07

Disposition: **0 required corrections.** The actual launch02 packet is consistent with the reviewed v2 source and the bounded CPU-only diagnostic. No missing saved evidence prevents this packet review from closing. No attempt was launched by this reviewer.

Scope: read-only review of the generated packet, job script and changes to the preparer, launcher, monitor and collector relative to reviewed launch01. Local source parsing, pin checks and custody comparisons only. No source/package execution, tests, models, remote operations, commits or modifications to prior evidence.

The separate source review is `docs/development/guidance/reviews/ace_llm_intervention_runtime_generated_review_2026-11-07.md`, SHA-256 `d257a11a8df45f24f84642707fbf069ee672e76e1cf5f9a81213e614a89eaaf5`. It closes the six new saved delta methods with zero required corrections. The original eight methods were not rerun.

## Exact packet pins

All ten files below match byte-for-byte between `results/ace_llm_intervention_runtime_metadata_20261107/launch02` and `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-20261107-02`. Their membership was identical at review. The packet's five input pins, diagnostic source pins and job command arguments agree.

| File | SHA-256 |
| --- | --- |
| `packet.json` | `d59a578560a6fbe4ce68f1403414300c0208d054b1c8669eb18d4faf7e4214db` |
| `diagnostic.json` | `80107304e27dca7a1ff328487d3189210cd5509bdf2051e26641ab7c28dff151` |
| `job.sbatch` | `6281efc002c7070920a69736ca9eab2227c4ee03e14f094b7c26a934f72e28b3` |
| `qualify_llm_intervention_runtime_metadata.py` | `31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329` |
| `llm_intervention_runtime_archive.py` | `7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e` |
| `probe.py` | `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951` |
| `prepare_packet.py` | `c9fdd9bf3665afc06da8586c503b4c23d9aff2074cb4a7d37fb3bbb85023d3ac` |
| `launch_once.py` | `48d6cfb76e552e8702f46febee07901d1f9512536135d31b867f3090c98aabf5` |
| `monitor_once.py` | `32ba4f0df0cec34d1dfe23383a2ec440e664c232526830b5a239291c107a2a9c` |
| `pull_terminal.py` | `85d5c8f10751595e6d6d37f2055103bf6f599808db3eacedc41752ef2c4944f5` |

The final preparation manifest is `41f61393c50dfc331376ad6a1d73429ca7fb7a3f256ece9ab4a580476f02141e`. The current source, test and guide match its stable closeout pins. The diagnostic version is v2. The outer packet schema retains its existing launch-v1 schema name, which is separate from the diagnostic version and does not select v1 execution.

## Reviewed boundaries

- The new remote root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_02`. Packet sources, output, Slurm logs, monitor and terminal collector consistently use attempt02. Terminal custody uses new `terminal02` and external `ACE-LLM-intervention-runtime-metadata-terminal-20261107-02` paths.
- The preparer requires the tested source pin and predecessor acceptance pin `5dcae41ee7da34503abddca8802b44b9c90e311b31b32dcc873d7c2b8b18dc23`. That acceptance identifies failed job `35345616`, and its eight cited evidence pins match. This accepts failure custody and accounting, not runtime qualification.
- The bootstrap authenticates the exact captured source bytes before compiling those same bytes. Its parsed command passes the reviewed source, diagnostic, helper and probe pins. No source was executed during this review.
- The archive and manifest remain pinned to `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98` and `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a`. Prelaunch checks hash the manifest and stat the 8,438,650,880-byte archive. Full archive authentication remains inside allocated staging before the copied runtime executes. This reviewer did not redundantly hash the bulk archive.
- Resources agree across packet, diagnostic, script and runtime allocation guard: `ucb736_asc2`, `acpu/cpu-normal`, one node, one task, one CPU, 3 GiB, ten minutes, zero GPUs and no requeue. Supervisor, success, outer-child and cleanup deadlines remain 580, 180, 185 and five seconds. Process RSS limits remain 512 MiB and 1.5 GiB. Prior near-cap batch RSS is disclosed without claiming cgroup headroom or an OOM cause.
- Saved reconciliation supports the new totals: 16 prior jobs and 34 journals, followed by job `35345616` and its two journals, give 17 jobs and 36 journals. Its 72 allocated CPU seconds increase 6,215 to 6,287. Admission is **6,287 + 3,600 + 600 = 10,487 CPU seconds**, below 57,600. Prior GPU accounting remains 836 seconds with no new GPU request. Reservations, allocation, process CPU and preparation-test costs are not added as though disjoint actual consumption.
- The launcher retains both controller locks, commit checks, exclusive local and remote journals, fresh queue/accounting/association reconciliation, scratch readiness, remote input readback and one `sbatch` call. It refuses existing intent or ambiguous submission and does not retry. The reviewed 17/36 counts are expected prelaunch values, not a claim of a fresh remote check by this reviewer.
- Monitor changes are attempt-path changes only. Collector changes update attempt paths and the 6,287-second base. It retains terminal-state gating, fresh remote inventory, local byte comparison, packet/input binding and an independent-review-pending disposition. `planned_runtime_archive_reuse` and `observed_staging_scope` preserve the earlier correction and do not infer completed staging from intent.

At inspection launch02 had no submission intent or result. The parent still must satisfy the launcher's commit and live reconciliation gates. Actual imports, generated-module observations, arithmetic, cleanup, scheduler outcome and complete terminal custody remain future evidence. This packet review neither adopts a new scientific runtime nor qualifies a model service, and it does not alter the frozen failed predecessor.

Required corrections remaining: **0**.
