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
640fits, including initialization sensitivity. It requires39 target-runtime
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
   plan.json and all six generated scripts. Rounded wall-time requests for all
   40world tasks, qualification, collection, evaluation, a fifteen-minute final
   audit and the original pilot
   must together remain within150CPU-core-hours. Every task uses1CPU/3GiB,
   accountucb736_asc1, acpu/cpu-normal and explicit scratch logs. No GPU.
6. Submit exactly once using delivery_prospective_slurm.py submit. The exclusive
   submission_started.json precedes every sbatch attempt. The journal records
   commands, outcomes and job IDs, including partial or ambiguous submission.
   Record IDs/source/account/output immediately in the execution ledger.

## Worker barriers and allocations

The dependency chain is qualification -> collection -> two arrays -> evaluation -> independent audit.
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

Qualification/collection/evaluation/audit have process-tree memory telemetry and
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

## Independent result acceptance and manuscript exports

Before any prospective outcomes, the read-only auditor and exporter are part of
committed source custody. The final audit requests1CPU/3GiB/15min and depends
on successful evaluation. Its900seconds are included in the rounded150core-hour
allocation gate, along with every fit and the original pilot. An audit failure
is preserved and blocks publication; it does not authorize a retry or rescue.

Run audit_delivery_prospective_results.py against the complete raw study and
its exact frozen project/original runner. It validates all640cells,80training
journals,40world attempt ledgers, phase telemetry and the fit-before-test barrier;
replays every checkpoint using predicted parents; independently recomputes
continuous errors, training-only variance and the four primary Holm contrasts.
No simulator responses or fits occur. Cached predictions remain the scored
endpoint. Checkpoint replay tolerance was specified before outcomes:
rtol1e-6/atol1e-7 for CPU float32 roundoff, exact root clamps, with every maximum
replay discrepancy retained. Copied custody retains canonical remote paths.
Acceptance requires the exact pinned target dependencies and source hashes.

The acceptance receipt additionally reports the short-fit ablation, collection
history and initialization sensitivity descriptively, retaining the entire
matrix and fixed deployment init0. These add no significance tests or scored
model-selection rule. export_delivery_prospective_claims.py requires both the
independent acceptance and successful audit supervisor, then writes exclusive
LaTeX tables/macros and a claim-to-receipt index. It refuses changed scores and
incomplete results. The manuscript is edited only after actual accepted results;
analytic test fixtures are never manuscript evidence.

Thirty-nine local development checks pass (models5, response custody4,
batch/resources7, theory10, primary statistics5, independent audit5, export3).
Target-runtime qualification must run those39 checks before any selected-world
responses. The five original queued pilot workers are unchanged. New batch
source must be committed, packaged and transferred into a new exclusive root;
the earlier f9640718 candidate and befa0bd8 pilot remain preserved. No new
prospective fits, responses, tables or scientific acceptance exist yet.

### Current blocker: pilot qualification failed before timing

Pilot33505661 started22:32:29UTC and failed22:34:35UTC2026-10-06 (126allocated
CPU-seconds, one CPU). Four of five checks passed. The fifth could not import
experiments.large_scale_scm because the launch environment omitted the frozen
project root from PYTHONPATH. No timing fits/development inputs, projection or
selected-system responses exist. Original artifacts and five worker hashes
remain preserved; complete failed raw custody and Slurm accounting were pulled.
Receipt: results/delivery_prospective_preparation_20261006/pilot33505661_failure.json.

The full batch environment now supplies absolute project AND worker import
roots. Thirty-nine local tests pass from /private/tmp, including an import
regression independent of repository cwd. A bounded target-runtime import-only
probe resolves the original frozen generator; this is not runtime qualification
or timing acceptance. The proposed replacement pilot uses the same five unchanged
befa0bd8 workers, a new exclusive output,1CPU/3GiB/15min, explicit import roots,
and a new submission journal. Reviewable proposed script:
results/delivery_prospective_preparation_20261006/pilot_importfix_proposal.sbatch.
No replacement was submitted or registration overwritten.

The new resource gate carries126historical failed allocated CPU-seconds inside
the150core-hour ceiling, in addition to the900second next pilot reserve and
900second independent result audit. The committed failed receipt is packaged
and bound into final registration. No new allowance or matrix trimming.
Per the user qualification-failure stop instruction, another pilot requires
an implementation/resource decision. Timing/launch remain blocked; original
A/C scientific outcomes and manuscript claims are unchanged.


New candidate source5e8d6e48dae220ef0f20320f7fc81a14a34bdea0 is committed/pushed.
A Git-byte-verified bundle is prepared in exclusive local external custody:
delivery-prospective-audit-release-bundle-20261006, SHA256
dff8d1a4df7b1f0b6ae94c6ba950fd6b939d1c9f562453ca71583facb50c95c9.
It includes the historical failed-charge receipt and all39qualification checks.
No transfer/final freeze/replacement submission yet; the previous remote candidate
and failed pilot remain immutable. Compact bundle/release-status receipts are
in results/delivery_prospective_preparation_20261006. The hourly automation stops
at the explicit qualification decision gate; restart it only with the next
implementation/resource authorization. Shared transport is working; authentication
is not the blocker. No new manuscript result or outcome-based configuration change.


### Recovery decision resolved; one unchanged-science pilot running

The newer owner coordinated-review instruction allows routine agreed execution
within established scope/budget and reserves signoff for material deviations.
After preserving/diagnosing the import-only failure, the coordinator resolved
this implementation decision. Fresh custody/storage/runtime metadata checks
passed; source/models/fit settings remain identical. Submitted33507335 exactly
once22:50:38UTC; start22:50:39UTC andRUNNING verified22:51:23UTC on c3cpu-e2-u2.
Accountucb736_asc1,acpu/cpu-normal,1CPU/3GiB/15min; output
/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_importfix_20261006/pilot.
Original33505661 and126charged seconds remain preserved. Corrected registration
SHA99a5ff5a8b630d7c7f5329237a5edc0c78a6ad764be5db835ec665b800ce9bf6.
Receipt/journal: results/delivery_prospective_preparation_20261006/pilot_importfix_*.
No new allowance, favorable recipe change, full-study release or selected-world
response. Full gate still needs successful runtime/timing receipts and independent
640fit projection, then final frozen39check protocol and allocations. Individual
hourly automation stays paused; coordinated review handles further progress.


### Accepted timing and pre-response numerical metadata correction

Pilot33507335 completed0:0,119allocatedCPU-seconds (22:50:39–22:52:38UTC).
All five target checks pass; all eight timing model/cost/input artifacts are
in exclusive local custody and independently accepted. Conservative640fit
projection79.77947core-hours; rounded full allocation86.86833core-hours including
126historical failed seconds,900s successful-pilot reserve and900s final audit.
No test losses or selected-world responses. Compact acceptance/accounting in
results/delivery_prospective_preparation_20261006/pilot_importfix_*.

The first metadata freeze of5e8d6e48 stopped before creating the full study:
local NumPy2.4.6 versus CURC2.2.6 regeneration differs by one adjacent float64
value in92coefficients across23of40worlds (largest absolute4.440892098500626e-16).
No seed, graph or noncoefficient metadata difference. Diagnostic and decision
receipts are preserved. The validator now permits only adjacent finite float64
coefficient regeneration and records each hex-valued discrepancy. Original
hashed descriptors remain byte-identical and are the only coefficients used
for collection/evaluation. This changes neither simulator outcomes, recipes,
menus, models nor timed kernels. Larger coefficient drift, noise or graph/seed
drift fails. Forty local checks pass, including these rejections; final target
qualification must run40checks before any selected response. Old candidate is
preserved, new committed source/bundle required before final freeze. No new
allocation or timing retry is needed for this metadata-only correction.


Final pre-submission review also found the Slurm scope guard needed to resolve
both sides of CURC's /scratch->/gpfs alias. Corrected both prepare/submit guards;
an actual symlink preparation test accepts canonical ACE scratch and rejects
outside paths. Forty-one local checks pass from unrelated cwd. The previous
40-checkcandidate remains preserved without any full study/selected response;
latest committed source/bundle and41target qualification are required for
release. No model, scientific setting, original descriptor or timing worker
changed. Rounded86.86833CPUh allocation and150ceiling remain unchanged.


### Stage B released once; target qualification and shared collection accepted

Frozen source `45ebeb89d2c76daa97a55f07728239245e0c4f60` and bundle
SHA256 `2449e67882f57b28ec3ce92e0d571cdb33f5e2740cab49ae64982585e939ede7`
passed committed-byte transfer validation at 23:04:39 UTC. Full registration
SHA256 `e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5`
was frozen before any selected response. Six generated scripts were pulled and
reviewed against the exact registered plan before the exclusive submission.
Output: `/scratch/alpine/paco0228/ACE/results/delivery_prospective_full_20261006`.
Account `ucb736_asc1`, `acpu/cpu-normal`, one CPU and 3 GiB per task. The rounded
total request is **86.86833 CPU-core-hours**, including pilot/failure reserves,
all forty world tasks and qualification/collection/evaluation/audit, within 150.

Submitted once at 23:08:11 UTC: qualification **33507418**, collection **33507419**,
five-node array **33507420**, thirty-node array **33507421**, evaluation **33507422**,
and independent audit **33507423**. Each twenty-world array permits concurrency
two; at most four world tasks run simultaneously. Original pilots and prior
candidate bundles are preserved. No frozen-source modification or duplicate job.

Qualification passed **41 target-runtime checks** and finished successfully at
23:08:44 UTC (28.13 supervisor seconds; peak process-tree RSS 749,678,592 bytes).
Shared collection finished at 23:08:53 UTC (4.31 supervisor seconds; peak RSS
73,662,464 bytes). All **32,000 charged training responses**, eighty histories,
240 raw input/journal/receipt artifacts, and exact **640-cell matrix** are now in
exclusive local custody. Local verification recomputed every history artifact
hash, checked 400 paid rows per history, qualification log/execution hashes,
registration/matrix bindings and per-size/arm/init membership. This is input
and execution acceptance, not prospective scientific acceptance.

At **23:14:04 UTC**, four world tasks were running and **12 completed fit receipts**
were verified against model hashes, input bindings and recorded optimizer counts:
30,000 updates per head for delivery/flat, 48,000 for online, and 100 for the
short-fit ablation. No fit generated or read evaluation responses. These initial
receipts are from two five-node systems; thirty-node fits remain in progress.
All completed supervised attempts had successful exits and memory below 3 GiB.
No stop flag; evaluation/audit remain dependency-pending. Held-out responses and
prospective scores have not been generated or opened. Evaluation requires all
640 fit receipts and forty completed world journals, followed by independent
result audit and successful audit-supervisor validation before any paper claims.

Compact evidence: `results/delivery_prospective_preparation_20261006/full_*`
and `final_*` receipts. Raw custody:
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-full-20261006`.
Training inputs are complete; live fit custody remains explicitly partial.
The original A/C conclusions and root-support limitations remain unchanged.
Individual hourly automation remains paused; coordinated review advances work.
