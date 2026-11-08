# Collection01 terminal collector handoff

Complete standalone sidecar `pull_terminal.py`, derived from the actual reviewed realization02 collector SHAdeebd55f... . The accepted builder86e29..., launcherfd190..., and monitorb621... are unchanged. Their corrected required realization-inventory input and four passing fabricated tests remain in `handoff.md` and `inventory-fix04/`. No preparation01/02 or launch01 output was touched. Parent owns custody of the separate failed preparation capture.

## Invocation after actual terminal job

Required arguments: `--job-id CURRENT_JOB --packet-sha256 OUTER_PACKET_SHA --submission-result-sha256 RESULT_SHA`. The collector is independently reviewed/captured as a sidecar; the stable builder deliberately still packages its original three authoring scripts. Run this collector separately after its review, using independently supplied actual pins. Collector copies its own source into both terminal capsules and records its SHA in the collector intent. No current job/result/default acceptance is fabricated.

`--packet-sha256` means collection launch01 `packet.json` / remote `ROOT/packet.json`, not `ROOT/inputs/packet.json`. Exact local submission intent and pinned result join the remote intent; remote fields must equal the local object plus only `remote_utc` and `fresh_readiness`, as emitted by this collection launcher. Existing recovery namespace claim pin is shared, not regenerated.

Fixed paths:

- ROOT `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01`.
- Launch repo `results/ace_llm_intervention_scientific_collection_20261108/launch01`, external `ACE-LLM-intervention-scientific-collection-20261108-01`.
- Terminal repo `results/ace_llm_intervention_scientific_collection_20261108/terminal01`, external `ACE-LLM-intervention-scientific-collection-terminal-20261108-01`.
- External BASE `/Users/pat/ACE_Study_Results/2026-10-peter-baseline`.
- Output `ROOT/collection`, wrapper task `ROOT/collection/session/worker/task`, charged collector tree `.../task/collection`.

## Custody, source, lineage and accounting

Both existing coordinator locks and exclusive terminal destinations. Requires current job conclusively terminal and no active ACE job before remote tree transfer. Captures full current ROOT including all outputs, source/input captures, runtime/cleanup evidence, logs, failures and journals. No suffix/prefix sampling or success filtering. Full new local/external launch histories are copied. Original and recovery namespace claims are read-only and copied from local/remote locations. Whole-root copies preserve every hardlink name as independent bytes; original link groups are metadata. Symlink aliases fail closed.

First and second remote inventory/readback compare all raw file bytes/names/directories, original hardlink metadata, full accounting rows/journals and training-only upstream pins. Two independent local copies are compared and sealed. Transfer/query timeouts preserve available stdout/stderr, with no retry. Partial custody on exceptions remains, with explicit failure/inventory evidence. A missing collection output on a terminal Slurm failure is preserved, not converted into successful collection.

Baseline comes from independently pinned packet `baseline.json`: actual40 jobs/82 journals,zero outstanding,3600 prep. No invented next-baseline pin or fixed successful job. Exact old rows/journals plus current two journals must produce41 jobs/84 journals. Actual CPU is allocatedCPUs×elapsed for all outcomes; final cumulative costs derive from the admitted baseline, not reserved or success-only cost. Unexpected GPU allocation is charged when interpretable and explicitly flagged; unknown telemetry remains unknown. Requested and allocated profile must match2CPU7680MiB900s,zeroGPU,ucb736_asc2/acpu/cpu-normal. Full steps/MaxRSS/TotalCPU and scontrol data are retained for independent runtime/cleanup review.

Upstream realization acceptance, packet/result, namespace and inventory transport pins are read from current immutable packet metadata. Remote upstream reads are an exact allowlist: realization02 packet, two submission journals, task terminal, and world000..031. No realization-root recursion or world032..055 reads/copy. All training originals are fingerprinted before and after transport; their immutable bytes already reside in current collection packet descriptor inputs. Existing accepted realization custody remains preserved separately. Metadata inventories may include all56 hashes, never development coefficients.

## Public counts versus acceptance

After complete raw custody, only the public wrapper task terminal is decoded. It must bind original collection packet/admission and saved collection terminal/inventory/plan byte pins. Reported complete totals must be160 records/3584 retained pairs, responses<=9216 with matching known/accepted returns,zero unknown returns,total charged cost10240,zero new worlds/updates/tokenizer/model calls and no development evaluation. This is a reported-summary check, not recomputation of all private records or independent scientific acceptance. All raw160 records,3584 pairs and every charged failure/attempt are retained for the later independent audit.

Missing or failed terminal counts remain absent/unresolved; never infer zero charges from missing output. Original collection journals/tree remain the accounting authority. Disposition always sets `acceptance=false`, `collection_accepted=false`, `scientific_qualified=false`. Resource/lineage/count disagreements preserve full available bytes and fail closed. Successful custody remains review-pending.

## Validation

AST parse/compile only, including embedded remote observation source. Exact source/derivation snapshots and stdout/stderr/receipt are retained under `collector-static01/`. Collector was not imported or executed; no tests, remote access, models, science, namespace/output claim, actual build/submission, repository edit or commit. No old test/review repeated. Actual collection run, runtime success and acceptance remain unqualified.

Collector SHA256: `754efa824d7589795f127adf07f66a46baa9f281f13758ca98ba3ad7a28ba83e`.

Static receipt SHA256: `d7f8589836c8c9189c4dfebbf6c55ed7b12dea69ca601580e751903b1d012a96`.
