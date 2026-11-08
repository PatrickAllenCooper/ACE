# Scientific GPU training terminal collector — authoring handoff

Stable source for independent Godel review, not executed or independently accepted. Source is standalone `pull_terminal.py` in this exclusive directory. Original GPU core remains unchanged; separate R1-corrected core is `/tmp/ace-scientific-training-gpu-core-r1-w52r46o9` (builderee9c6561…, launcher6971ac25…, monitor3fc1fee1…). Production freeze156026/sourcea934 and trained worker are untouched.

Derivation: reviewed artificial screen-training GPU collector supplies resource/count/cost handling; reviewed token recovery collector8a710320 supplies both-lock custody, exact intent joins, two inventories/copies, namespaces and complete upstream preservation. Both predecessor source copies, original reviews and core snapshots are under `predecessors/`; diffs against both collector predecessors are supplied. `derive.py` and its text fragments preserve the initial authoring transformation only; final source also includes subsequent focused edits and must not be regenerated from that draft.

## Exact invocation interface

Required CLI arguments (no defaults for future evidence):

- `--job-id`: actual current GPU training Slurm job.
- `--packet-sha256`: exact outer GPU `launch01/packet.json`.
- `--submission-result-sha256`: exact original local/remote result bytes.
- `--local-intent-sha256`: original local intent; result independently pins augmented remote intent.
- `--packet-commit`: full40-character commit containing all immutable packet bytes, packet review, standalone collector source and its independent review.
- `--packet-review PATH --packet-review-sha256 SHA`: original concrete packet review, matching local intent and full commit.
- `--collector-sha256 SHA --collector-repository-path REPO_RELATIVE_PATH`: independently reviewed standalone collector bytes, matching that path in the full packet commit.
- `--collector-review PATH --collector-review-sha256 SHA`: independently reviewed collector report, also matching the full packet commit.
- `--cpu-terminal-custody PATH --cpu-custody-inventory-sha256 SHA`: complete independently pinned accepted TOKEN01 terminal capsule; the existing final custody-inventory self-exclusion/copy-map schema is reused.

The collector is deliberately separate from the unchanged core's `new_input_files` roster. Parent must preserve it and its review in Git before the eventual packet commit, and supply the explicit repository-relative source path. No new core member, packet key, production schema or acceptance record is required. A concrete packet/review commit may contain these standalone files without changing builder/launcher/monitor bytes. No runnable command with fake future pins is supplied.

Current GPU ROOT is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01`; output ROOT/training; worker artifacts ROOT/training/session/worker/artifacts; fitted data under artifacts/fit. Outer packet is ROOT/packet.json; supervisor packet ROOT/inputs/training.json. Both external/local launch copies and committed bytes must match. Exact intent extensions allow only remote_utc/fresh_readiness and preserve every original local field.

Terminal destinations (must both be absent) are `results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01` and `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-training-gpu-terminal-20261108-01`.

## Custody and accounting

Both existing controller locks cover exclusive capsule claiming and remote observations/transfers. No command submits, retries, requeues, cancels, installs, invokes scientific code or mutates upstream roots. The exact job must be conclusively terminal and no related ACE job active before raw traversal. Terminal failure/missing output does not become acceptance.

Whole-root raw inventory/copy labels are `remote/launch` (GPU training), `remote/cpu` (TOKEN01), `remote/common` (collection02), and `remote/failed01` (failed collection01). Every source/input, log, checkpoint/optimizer/adapter byte, candidate/failure receipt, runtime/provenance, parent/child telemetry, cleanup and journal file is retained, including empty directories. No favorable output filter is applied. Model/adapter files are streamed as opaque bytes, never deserialized. Both separate namespace registries are preserved and compared. Original CPU/collection inputs are additionally checked by the core's explicit hash roster; no realization directory/development descriptor traversal occurs.

The complete original accepted TOKEN01 local terminal capsule is independently authenticated, retained as `upstream-cpu-custody`, and compared before/after copying. It carries both original inventories, coordinator histories, cleanup/failure evidence, all original upstream raw maps and receipts. Existing CPU acceptance stays unchanged; its exact packet/prepared/plan/freeze/namespace/job bindings are required. Full CPU transport map, disposition, capsule roster and current remote roots must agree. Parent supplies actual accepted evidence; this collector manufactures none.

Two remote inventories must match full file/directory/hash/size maps, original hardlink metadata, parent/all-related accounting, journals and input readbacks. External copy must match the first inventory; independent repository copy must match all bytes. Hardlink names become separate local files, with original alias metadata retained. Symlinks/changing files reject. Transport timeout partial stdout/stderr and partial raw transfers remain, with best-effort second-copy preservation and explicit errors; no overwrite or blind retry.

Exact43 accepted prior rows/all88 journals must extend to44 rows/90 journals. All failed charges remain included. Allocation charges use actualCPU×elapsed and actualGPU-count×elapsed; aggregate and typed GPU TRES are not double-counted, and unknown GPU quantities never become zero. Requested/allocated4CPU15360MiB, onea100_3g.20gb,2700sec,ucb736_asc2/aa100/gpu-testing,no-requeue/restarts are compared from raw Slurm request/allocation evidence plus saved/public allocation fields. Unexpected allocation still retains raw evidence and actual observed charge. Full scheduler step MaxRSS/TotalCPU, requests, job fields and both observations are preserved; overlapping CPU metrics are not summed.

## Public GPU evidence boundary

`public-gpu-evidence.json` selects bounded public metadata only, after complete raw custody: owned PID/NVIDIA process-memory rows, parent elapsed/progress records, CUDA forward/event interval, memory peaks/device-memory totals where present, cleanup scalars, phase/step counters and disposal-memory observations. A lexical top-level selector skips score/outcome values without JSON-decoding them; adapter/optimizer tensors are never decoded. Original scored receipts/terminals remain complete opaque bytes. Worker/outer terminal markers are hashed, not interpreted as accepted training outcomes.

Independent signal types are reported as observed metadata, never as training/runtime qualification. Memory MiB, CUDA event intervals and progress do not measure SM-utilization percentage; `utilization_percent` remains null, and missing/failed telemetry remains unresolved rather than proof of idleness. The typed20GB MIG request is distinguished from actual framework total-device bytes and15GiB host RAM. Downstream independent audit must assess startup240,progress180,telemetry90, inherited deadlines, runtime/source admission and complete cleanup using the retained originals.

Final disposition is `ace-scientific-training-gpu-terminal-custody-v1`: successful-looking markers produce only terminal_custody_review_pending; failures/partials/conflicts are explicit. All acceptance, training_execution_qualified and scientific_qualified flags remain false. Whole scored evidence is preserved, but no utility or numeric-training validation is attempted.

## New evidence and limitations

`test01` preserves source snapshots, stdout/stderr, cost receipt and all fabricated fixtures. Four NEW tests pass: exactGPU/CPU/MIG/time/no-requeue and unknown-charge negatives;44/90 preservation including a failed252CPU attempt; score-skipping selector plus independent CUDA/NVIDIA signals/unknown utilization; remote-intent semantic mismatch and hardlink-custody rejection. Deliberate fixture-only hardlinks at test01/fixtures/links/{original,alias} test rejection; they are not valid custody artifacts. No old tests, models, scientific draws, packet build, remote execution or commit ran. static01 is syntax-only compilation of current source and embedded remote text.

This local authoring/test evidence does not establish operational success, live resources, future CPU/GPU acceptance or scientific utility. Godel source review and later concrete invocation/evidence pins are still required.

Exact collector SHA256: `9346be8d3bc3f9060f1bbc24c9dbe6285c4bddfdd11e5a08951024ed84e5b55a`

Test source SHA256: `d7dbbcc9c03c194299d46761e4eca4b39da1655eec1778d4845ea49e222e0c10`
