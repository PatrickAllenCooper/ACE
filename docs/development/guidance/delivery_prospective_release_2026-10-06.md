# Stage B release and failure reconciliation

Authority and ceilings remain in delivery_paper_execution_2026-10-06.md.
This is an execution procedure, not a new scientific agenda. A/C are complete;
their fits and the immutable attribution gate must not be rerun or overwritten.

## Current gate

Pilot33505661, sourcebefa0bd8, is queued on acpu/cpu-normal, accountucb736_asc1,
1CPU/3GiB/15min. It remains PENDING at21:41UTC2026-10-06. Its five worker files
remain byte-identical. Do not edit the remote pilot bundle, duplicate it or move
partitions to avoid the queue. No selected-system responses have been generated.

The new batch implementation covers40worlds,80shared training histories and
640fits, including initialization sensitivity. It requires30 target-runtime
checks before any selected-system collection. Local checks pass in the separate
torch2.5.1 environment; those results do not qualify the CURC torch2.9.1 runtime.

## Release order

1. Reconcile pilot job state, original submission/bundle hashes and the complete
   qualification/execution/model/input receipts. Pull the complete pilot into
   exclusive external custody. Run audit_delivery_prospective_pilot.py. It
   independently recomputes the all640fit timing estimate with3x margin; no test
   losses are opened. If it exceeds150CPU-core-hours or parity fails, preserve
   everything and stop for a resource/implementation decision. Do not trim
   systems, omit sensitivity fits or change a scored recipe to fit the ceiling.
2. Package the committed source with package_delivery_prospective.py. It checks
   every worker/generator against Git bytes, checks the immutable Stage A gate
   and original learner, and copies only the40world/action descriptors. Preserve
   bundle_receipt.json locally. No responses or allocations are produced.
3. Recheck transport, bounded scratch stat/df and a dedicated ACE write/read/
   cleanup probe. Transfer the bundle to a new exclusive ACE scratch root and
   verify its SHA256 before extraction. Never overwrite the pilot's source.
4. In the pinned existing CURC ace environment, call delivery_prospective_batch.py
   freeze with the transferred draft, project, original runner, accepted pilot,
   immutable gate, full committed source revision and a new scratch study path.
   This metadata-only operation reproduces every world descriptor, validates
   held-out action blocks and freezes sources/dependencies/protocol/resources.
   No selected-system response is evaluated. Preserve registration.json and its
   SHA256 locally before launch.
5. Run delivery_prospective_slurm.py prepare against that registration. Review
   plan.json and all five generated scripts. Rounded wall-time requests for all
   40world tasks, qualification, collection, evaluation and the original pilot
   must together remain within150CPU-core-hours. Every task uses1CPU/3GiB,
   accountucb736_asc1, acpu/cpu-normal and explicit scratch logs. No GPU.
6. Submit exactly once using delivery_prospective_slurm.py submit. The exclusive
   submission_started.json precedes every sbatch attempt. The journal records
   commands, outcomes and job IDs, including partial or ambiguous submission.
   Record IDs/source/account/output immediately in the execution ledger.

## Worker barriers and allocations

The dependency chain is qualification -> collection -> two arrays -> evaluation.
Each graph-size array has20world tasks and concurrency2, at most4CPU world tasks
in total. A task fits the16registered cells of its world in sequence. World
wall ceilings come from the measured pilot and include3x margin, subprocess
startup and rounding; a requested total above the cap is rejected.

Qualification runs only development fixtures, analytic checks and selected
world/action metadata checks. It never generates selected-world responses.
Successful qualification receipts bind the source/runtime/logs before collection.
Collection charges32000training calls before generation, including failures.
The first50paid rows calibrate identical SCM normalizers. Nonfinite or zero
training target variance stops the study before fitting; no replacement world.
The80immutable input/journal receipts bind the exact640cell matrix.

Fits cannot open world/action coefficients or held-out responses. Every cell
uses the fixed recipe and initialization; no development overrides are permitted.
Model and receipt hashes, head/update/row counts and all40world attempt ledgers
must validate before any test response. A fit failure stops new cells across
the arrays and blocks evaluation. Already running cells may finish; retain them.
Evaluation then acquires16000shared held-out responses once, uses predicted
parents, and requires every score, including the retained ablation/sensitivity
cells. The240primary cells feed system-level paired statistics and four-test
Holm correction. No complete-case filtering or scored initialization selection.

Qualification/collection/evaluation have process-tree memory telemetry and
watchdog ceilings; Slurm independently limits every allocation. Fit children
also have wall/RSS guards, and the parent Slurm allocation limits combined memory.
Reported fit CPU excludes imports/parent work; reconcile it separately with
requested CPU time, phase elapsed time and sacct accounting. Keep raw weights,
predictions, failures and journals in external custody; commit compact receipts.

## Reconciliation rather than blind retries

If submission times out, inspect the exclusive journal and ACE-only scheduler
jobs before doing anything else. A missing job ID is an unknown outcome, not
proof that no job exists. The submitter refuses a second launch. Do not delete
claims, reuse paths or resubmit the entire chain. Preserve dependency failures
and request a genuinely new decision when the frozen study cannot complete.
Never modify unrelated jobs or environments. Leave valid pending jobs queued.

Scientific scope is controlled generalization of a noise-disabled root-action
delivery recipe. Measured intermediate supervision and fitting compute differ
from the root-target flat control. Root-only support may not constrain mechanism
behavior under perturbed intermediates; neither bounding boxes nor small local
residuals certify chain accuracy. No arbitrary internal-action identification,
external physical superiority, equal-compute or foundation-model claim follows.
