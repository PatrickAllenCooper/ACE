# ACE delivery study handoff

Prepared 7 October 2026 for the next researcher or implementation agent. The
publication target is **Transactions on Machine Learning Research (TMLR)**.
The active question is how retaining and refitting a completed experimental
history changes the delivered causal surrogate. Historical foundation-model
and acquisition agendas are stopped under their existing gates.

The credible result comes from data retention and fitting effort, assessed with
matched histories, explicit eligibility and deployment semantics, strong simpler
controls, and independent receipts. It does not establish a new architecture,
foundation-model benefit, acquisition superiority or unrestricted identification.
Prospective independent-system scores remain pending.

## Start here

Use these records in this order:

1. [Delivery execution authority](/Users/pat/code/ACE/docs/development/guidance/delivery_paper_execution_2026-10-06.md).
2. The newest dated entry in the [execution ledger](/Users/pat/code/ACE/docs/development/guidance/research_execution_2026-09-25.md).
3. [Prospective registration](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json)
   and the completed source, input and custody receipts. Earlier sections of the
   execution documents describe superseded pilot blockers; do not treat those as
   the current state.
4. [Active manuscript](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex),
   [claim index](/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json),
   and this handoff. The directory name remains historical; the target is TMLR.
5. [Review and completion register](/Users/pat/code/ACE/docs/development/guidance/delivery_review_register_2026-10-07.md),
   which tracks independent review findings and the remaining submission gates.

No new experiment, environment installation or resource expansion is authorized
by this handoff. Keep the frozen study and original weights unchanged. Queue
waiting is acceptable. Do not duplicate submitted jobs or resume old lanes to
keep resources busy.

## Scientific evidence already accepted

### Completed confirmation

Twelve acquisition histories of one fixed deterministic emulator were compared
with their unchanged online chain weights. All eligible paid observations were
refit for 30,000 epochs. The geometric mean snapped exact-level error ratio was
**0.183**, with 95% log-scale t interval **0.102–0.327** and exhaustive two-sided
sign-flip **p = 0.00146484**. Eleven histories improved.

The numerator is the median error of three **scored** initializations. That is
a sensitivity summary, not a deployable selection rule. The evaluation grid
was exposed and includes acquired inputs. There are twelve histories of one
system, not twelve independently parameterized causal systems or 36 independent
models. Additional fitting compute is part of the treatment.

History **124753321** worsened from **0.196339** to **0.395658** snapped error,
ratio **2.015177**. It remains in every relevant report. No rescue, exclusion or
alternative scored initialization is permitted.

Evidence: [scores](/Users/pat/code/ACE/results/delivery_final_history_20261006/scores.json)
and [independent statistics](/Users/pat/code/ACE/results/delivery_final_history_20261006/independent_statistics.json).

### Stage A attribution

All **480 archived-history fits** are complete and independently accepted.
No new simulator responses were acquired. The matrix separates three retained
row sets and two scratch fitting budgets, with neural, classical and matched-CPU
controls. Other initializations are retained as sensitivity.

For primary initialization zero, all-paid SCM fitting for 30,000 epochs has
geometric mean continuous NMSE ratios of:

- **0.522** against the original online weights.
- **0.551** against the development-selected flat 30,000-epoch refit.
- **0.423** against the separately matched-CPU flat fit.
- **0.096** against final-buffer-only long SCM fitting.
- **0.034** against all-paid 100-epoch SCM fitting.
- **0.980** against long fitting on the rows actually admitted online.

Within the scratch-fit matrix, long retained-history fits outperform buffer-only
or short fits. These contrasts include training-derived scaling changes and do
not partition the gain over online weights with persistent optimization state.
Unused rows give little aggregate continuous improvement, but 10/12 histories
improve, ratios span 0.644–2.874 and the snapped ratio is 0.469. These are
exploratory results on the same exposed grid with no prospective superiority
claim. Structured and flat fits
differ in usable intermediate-intervention rows and supervision. The matched
CPU budget includes all SCM heads, including the communication-cost head that
does not feed the scored target.

The selected prospective recipe is **SCM / all paid / 30,000 / init 0**; the
simpler control is **flat / all paid / 30,000 / init 0**; the decisive ablation is
**SCM / all paid / 100 / init 0**. Keep the selection gate immutable.
The full matrix used **10.87 fit CPU-core-hours**, excluding startup and evaluation.

Evidence: [immutable gate](/Users/pat/code/ACE/results/delivery_paper_implementation_20261006/attribution_gate.json),
[attribution summary](/Users/pat/code/ACE/results/delivery_attribution_20261006/summary.json),
and [complete receipt](/Users/pat/code/ACE/results/delivery_attribution_20261006/complete.json).

### Stage C physical boundary

All eleven previously unscored `lt_malus_v1` conditions are complete and accepted;
`white_64` is development only. No physical responses were newly acquired.
Delivery improves row-weighted continuous NMSE over rolling fitting in **7/11**
conditions, the relative-angle physics fit in **3/11**, and Fourier regression
in **0/11**. This is one apparatus and one response mechanism, not eleven
independent worlds or a test of DAG factorization. Retain every condition.

The correct physics basis is an intercept plus
`cos(relative polarizer angle)^2`, with degrees converted to radians. The earlier
product-form attenuation control is misspecified for this task and must not be
reintroduced. The Fourier control uses 25 tensor-product features built from
`1, sin(2θ), cos(2θ), sin(4θ), cos(4θ)` at each angle. Domain structure beats the
generic neural recipe here.

Identical commands remain in the same joint 30-degree action block. Conditional
bootstrap intervals resample eight held-out blocks within a condition; they are
not population intervals across apparatuses. Block-weighted ratios differ from
row-weighted primary NMSE when readings per block differ. The accepted study used
**0.053 fit/evaluation CPU-core-hours**, excluding imports and startup.

Evidence: [physical acceptance](/Users/pat/code/ACE/results/delivery_chambers_20261006/acceptance.json),
[protocol](/Users/pat/code/ACE/results/delivery_chambers_20261006/protocol.json),
and [all condition scores](/Users/pat/code/ACE/results/delivery_chambers_20261006/scores.json).

## Why the original framing failed

The [metric audit](/Users/pat/code/ACE/docs/development/guidance/metric_audit_2026-09-10.md)
and [historical handoff](/Users/pat/code/ACE/docs/development/guidance/handoff_2026-09-25.md)
record four material confounds:

- Reward-shaped root weights leaked into one reported score, while baseline
  root weights differed.
- Nonroot validation distributions differed: broad parent contexts versus
  observed parents. Those endpoints do not answer the same question.
- The large baseline generator redrew coefficients per response while the ACE
  adapter used fixed mechanisms. The apparent comparison crossed stationary
  and changing systems.
- Students differed in capacity and fitting budget. Correcting metrics alone
  did not correct this asymmetry.

After learner matching, the earlier five-node acquisition advantage did not
survive as the advertised headline. Larger-scale DPO results were unfavorable.
Later failed semantic-prior, modular-repair, PEV and direct partial-identification
lanes retain their stop gates. Do not transplant an old favorable snapshot into
the delivery paper.

The useful change was to hold paid histories fixed, separate the online state
from a final fit, and ask which data and optimization differences mattered.
The lesson is to match mechanisms, learner, usable information, metric,
deployment inference and experimental accounting before interpreting an effect.

## The technical stack that made the delivery result interpretable

### Mechanism learning and predicted-parent inference

PyTorch implements each nonroot mechanism as
`Linear(d,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)`.
The parameter count is `64*d + 4289` per head. Full-batch Adam uses learning
rate `0.002`. Scratch heads have separate optimizers; rolling training preserves
its optimizer states. Count updates per head and all fitted heads, not epochs
as simulator calls or a whole-SCM epoch as a single flat update.

Fitting uses measured parents. Evaluation must traverse the DAG in topological
order and use **predicted** parents, with intervened roots clamped exactly.
Observed-parent residuals are a separate diagnostic. Teacher-forced accuracy
cannot stand in for deployed chain accuracy.

The prospective worker directly calls the archived
`ACEOracle._train_node_mlps_on` kernel; it does not approximate the online
baseline with a newly invented update. That retains 20 fast updates plus 100
consolidation updates per arriving row, persistent Adam, and the duplicated
new observation in replay. A 400-row prospective history therefore records
48,000 online optimizer updates per head. The comparison is retrospective after
a shared 50-row calibration prefix, not a strict stream before that prefix exists.

Primary code: [model kernels](/Users/pat/code/ACE/scripts/research/delivery_prospective_models.py)
and [input and evaluator primitives](/Users/pat/code/ACE/scripts/research/delivery_prospective_io.py).

### Observation logs and replay reconstruction

The final buffer is **50 entries**, not 50 experimental steps. Paid query IDs
distinguish repeated responses to the same action. Stage A reconstructs the
actual online-admitted union from the frozen event sequence: seeds enter the
buffer, a selected probe triggers fitting, and observation refreshes enter after
the fit. Clone-only probe fits are not admissions to the live learner.

Consolidation duplicates the selected row. Scratch attribution uses each query
ID once and records this weighting difference. The online fast path can train
a directly intervened head; the twelve confirmation histories have zero selected
clamped-node fast updates, so this latent code limitation is not an observed
explanation of their gain. Joint-root prospective interventions avoid that path
for modeled nonroots.

Eligibility excludes a row from a mechanism when that mechanism was directly
replaced by the intervention. A root-to-target flat predictor without action
inputs excludes intermediate-intervention rows. The phrase "same paid history"
must therefore not imply identical usable rows in Stage A.

Primary code: [attribution and replay reconstruction](/Users/pat/code/ACE/scripts/research/delivery_attribution.py),
[independent attribution audit](/Users/pat/code/ACE/scripts/research/audit_delivery_attribution.py),
and [selection analysis](/Users/pat/code/ACE/scripts/research/analyze_delivery_attribution.py).

### Normalization and the estimand

Stage A scratch input scales use eligible training minima and maxima, with width
one for a constant input. Stage B fixes structured input ranges from the first
50 paid rows and declared root support `[-3,3]` before retrospective updates.
Evaluation NMSE divides by the full corresponding training target variance.
No held-out response chooses a scale. Stage C also centers and scales neural
outputs using training responses only.

The prospective training and evaluation estimand is the **same noise-disabled
structural map**. Composing conditional means through a nonlinear mechanism is
not generally its noisy interventional expectation: if `X=±1` equiprobably and
`Y=X²`, composing the mean for X predicts zero while `E[Y]=1`. A future noisy
study needs an explicitly matched distributional or expectation estimand for
every arm; do not silently reuse the present evaluation.

### Simpler controls and honest compute comparison

Installed scikit-learn supplies ridge (`alpha=1`), degree-three polynomial ridge
(`alpha=1`) and extra trees (256 trees, leaf size two, seed zero, one worker).
The earlier development history selected the flat MLP control before scoring
the twelve histories. Equal epochs and matched process CPU are separately
labeled comparisons; neither matches intermediate supervision.

One CPU is sufficient for the current small MLP workload. GPU allocation is
unnecessary. Count fit process CPU, full supervised wall time, imports, peak
process-tree memory, and Slurm allocated CPU-seconds separately. A resource
reservation is a ceiling, not measured scientific cost.

### Independent custody and executable acceptance

The experiment pipeline separates metadata preparation, collection, fitting,
evaluation, audit and claim export. Exclusive destinations and journals preserve
failed or interrupted attempts. A charged response ledger records an attempt
before generation. Cached-response replay in an audit adds no acquisition.

Source, dependency versions, world coefficients, actions, input rows, model
states, optimizer counts and outputs are hash-bound. Fit workers cannot access
evaluation/mechanism files or make network calls. Every one of the 640 models
must seal before prospective held-out generation. Independent auditing then
replays checkpoints, recomputes scores and system-level statistics, and validates
the successful audit supervisor. A partial fit receipt or exit code alone is
not scientific acceptance.

Primary code: [batch barriers](/Users/pat/code/ACE/scripts/research/delivery_prospective_batch.py),
[full result audit](/Users/pat/code/ACE/scripts/research/audit_delivery_prospective_results.py),
[system-level analysis](/Users/pat/code/ACE/scripts/research/delivery_prospective_analysis.py),
and [claim export](/Users/pat/code/ACE/scripts/research/export_delivery_prospective_claims.py).

## Theory and the failure that must remain visible

The [theory source](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex)
and [analytical checks](/Users/pat/code/ACE/scripts/research/test_delivery_theory.py)
establish conditional mechanism eligibility, deterministic DAG propagation,
quantization margins and counterexamples. They are explanatory propositions,
not certified empirical bounds or new unrestricted identification results.

- Under the stated support and coordinate Lipschitz assumptions,
  `e_i ≤ ε_i + Σ_j L_ij e_j`; correctly clamped nodes have zero error. Path gains
  and depth can amplify small local errors.
- Root interventions can leave parents on a manifold. With `M=R`, mechanisms
  `R+M` and `R+M+K(M−R)` agree on every root-only observation but differ sharply
  after an upstream prediction shift or an internal intervention. Coordinate
  bounding-box inclusion does not resolve this ambiguity.
- Snapped accuracy depends on decision margins. An error smaller than the
  nearest quantization-boundary margin preserves the level, but similar MSE
  alone does not imply similar exact-level error.
- More observations or optimization need not improve test performance.

For worsening history 124753321, the primary target observed-parent MSE is
`8.07e-5`, free-running MSE `2.86e-4`, and propagated-shift MSE `2.09e-4`.
Only `0.020%` of evaluated parent vectors fall outside the coordinate training
box. These diagnostics do not isolate a unique cause or certify joint support.
Local residual and prediction-shift MSEs do not simply add: their squared-error
decomposition has a cross term. Keep both the continuous and snapped failure.

## Local and CURC integration findings

### Import roots must be explicit

Development-only pilot **33505661** failed before timing because the frozen
project root was absent from `PYTHONPATH`. Local repository-cwd checks had not
qualified the target launch context. Preserve its 126 allocated CPU-seconds and
all original files. The replacement **33507335** supplied absolute project and
worker import roots, passed the unchanged scientific kernels, and completed
with 119 allocated CPU-seconds. Regression checks run from an unrelated cwd.
An import-only diagnostic is not full timing or scientific qualification.

### Verify the target numerical runtime

The target environment is the existing
`/projects/paco0228/miniconda3/envs/ace/bin/python`: torch **2.9.1**, NumPy
**2.2.6**, SciPy **1.15.3**, pandas **2.3.3**, SymPy **1.14.0**, PyYAML **6.0.3**.
Local torch 2.5.1 checks did not qualify target torch 2.9.1. Login-base Python
also lacks the required package metadata; invoke the pinned environment for
diagnostics without changing it or installing packages.

Descriptor regeneration differed between local NumPy 2.4.6 and target 2.2.6
by one adjacent finite float64 in 92 coefficients across 23 worlds, maximum
absolute difference `4.44e-16`. The recorded validator admits only that adjacent
coefficient regeneration difference. **Original hashed descriptors remain
authoritative and unchanged for every response.** Graph, seed, noise or larger
drift fails. This is not a tolerance for altered outcomes or score selection.
Final qualification passed 41 target-runtime checks before training collection.

### Filesystem and submission discipline

SSH transport success does not establish scratch readiness. Earlier bounded
scratch `stat`/`df` calls timed out. Later dedicated ACE write/read/cleanup probes
passed before release. Recheck readiness before a new submission or large
transfer. Use the `curc-access` helper's shared master; if expired, ask Patrick
to reconnect privately and continue local development.

CURC aliases `/scratch` to `/gpfs`; both candidate and allowed-root paths must be
resolved before scope comparison. The symlink regression check accepts the
canonical ACE destination and rejects outside paths. Do not weaken that guard.
An ambiguous `sbatch` response requires journal/scheduler reconciliation,
not blind resubmission. `sstat` array aliases may not resolve; use the concrete
task ID reported by read-only `scontrol` before reading telemetry.

All authorized study tasks use account **ucb736_asc1**, `acpu/cpu-normal`,
**one CPU and 3 GiB per task**, explicit ACE scratch logs, and measured wall
limits. The two world arrays permit two concurrent tasks each. No Azure,
closed-source API or GPU is part of this campaign. Never modify or cancel
unrelated mono-s2s, foundation, DeFAb, ANI or other jobs and environments.

## Stage B resume checkpoint

This is the last committed observation at **7 October 2026, 07:49 MDT
(13:49 UTC)**, not a promise about the live queue when this handoff is read.

- Frozen source: `45ebeb89d2c76daa97a55f07728239245e0c4f60`.
- Registration SHA256:
  `e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5`.
- Remote output:
  `/scratch/alpine/paco0228/ACE/results/delivery_prospective_full_20261006`.
- Qualification **33507418** and collection **33507419** completed.
- Five-node array **33507420** completed all twenty systems.
- Thirty-node array **33507421** completed fifteen systems; indices 15–19
  remained pending under the maintenance reservation.
- Evaluation **33507422** and audit **33507423** remained dependency-pending.
- **560/640 fits and 35/40 worlds** completed, with no failed attempts, stop flag,
  source drift or target dependency drift in that observation.
- All **32,000 training responses** and the completed model/receipt pairs are
  locally hash-verified. No full fit seal or held-out score existed.
- Accrued chain plus both pilots: **24.540833 allocated CPU-core-hours**.
  Rounded total reservation **86.868333** remains within the **150** ceiling.

Evidence: [milestone](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_milestone_20261007T134904Z.json),
[source and accounting](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_source_accounting_20261007T134926Z.json),
and [custody](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_milestone_custody_20261007T134904Z.json).

Raw local custody root:
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline`.
The full prospective folder is `delivery-prospective-full-20261006`; fit custody
currently combines `fit-snapshot-20261006T235619Z` with
`fit-new-20261007T134904Z`. Do not incorrectly assume all models are in the
top-level full-study folder. Reconcile snapshot manifests before materializing
a complete local audit tree; never overwrite a conflicting artifact.

Next steps, in order:

1. Reconcile the six existing jobs and receipts through shared SSH. Leave valid
   maintenance-pending work queued. Do not alter frozen source or submit again.
2. Require all forty world journals and 640 model receipts, then the fit seal.
3. Let the existing evaluator generate 16,000 **shared** held-out responses,
   bringing the full charged collection/evaluation total to 48,000.
4. Require independent checkpoint/score/statistical recomputation and successful
   audit execution; pull important raw artifacts to exclusive local custody.
5. Apply the four frozen tests: delivery versus online and flat within each
   graph-size stratum. Average the two history log ratios within each system.
   Each stratum has twenty independent systems. Holm covers all four tests.
   A superiority claim requires ratio ≤0.8, upper marginal 95% t interval <1,
   and adjusted p<0.05. Report numerical-floor activations and failures.
6. Export prospective claims only after complete acceptance. Report init 1/2
   sensitivity and the short-fit ablation separately; select no scored model.
7. Integrate results into the TMLR draft and narrow the conclusion if controls
   erase the distinction. No favorable follow-up sweep is authorized.

## Manuscript preparation and editor integration

The same open `paper.tex` now uses the official anonymous TMLR style, a revised
introduction and related work, explicit delivery objective and endpoints,
completed attribution and physical findings, the frozen prospective analysis,
full theory, and implementation/system/split/provenance appendices. A visible
working notice states that prospective results remain pending. AI assistance
is disclosed in a first-page title footnote. Human scientific review remains
required; no submission has occurred.

The built-in desktop compiler accepts one source and originally failed at
`delivery_claims.tex`. The fix is a generated `filecontents*` companion bundle
inside **the existing document**, not a replacement PDF or an environment install.
The official `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` bytes remain unchanged;
their revision and hashes are in
[style provenance](/Users/pat/code/ACE/paper/aistats_ace_2027/tmlr_style_provenance.json).
The previous fancyhdr file is archived. Claims, tables, proofs and bibliography
remain separate authoritative tracked files and are embedded verbatim for the
single-source compiler.

After regenerating numerical claims or editing a companion:

```bash
cd /Users/pat/code/ACE
/Users/pat/code/ACE-Runner/.venv311/bin/python scripts/research/sync_delivery_manuscript.py --write
/Users/pat/code/ACE-Runner/.venv311/bin/python scripts/research/sync_delivery_manuscript.py --check
```

The check verifies all nine embedded files, including the receipt-generated
per-history attribution table, and official style hashes.
Use `generate_delivery_claims.py` for accepted A/C claims; do not hand-edit
numeric macros or generated tables. Use `export_delivery_prospective_claims.py`
only with fully accepted B custody and an exclusive destination, then extend
the companion list and insert the generated B tables into the same document.
Do not overwrite the current A/C claim index with a B-only index.

After any source edit, call the built-in `compile_latex_document` tool on
`/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex`. Keep its current editor
open; do not create or open a replacement document or separately compile a PDF.
Successful native compilation verifies LaTeX, not scientific readiness or a
completed visual page-by-page review.

Remaining publication work is prospective result acceptance and integration,
receipt-backed figures and full compute/failure reporting, independent scientific
review, and an anonymized reproducibility bundle. Author names, affiliations,
funding, conflicts and final approval must be supplied or verified by the human
authors. A two-to-three-week submission target remains conditional on those
gates. Do not fill pending results with projections or describe queue completion
as guaranteed.

## Boundaries for the next session

Preserve original weights, the immutable selection gate, failed pilots, earlier
bundles, historical manuscripts and all unfavorable conditions. New manuscript
commits do not authorize changing the frozen experimental source. Avoid
complete-case filtering, replacement worlds, test-selected initializations,
metric swaps and descriptions of empirical residuals as certified bounds.

The pre-existing `.DS_Store` change is unrelated user state and is excluded from
the manuscript/handoff commits. The personal archive search did not respond
during bounded connector and CLI checks. The archive drive is mounted; both
CLI `stat` and FTS-only search exceeded fifteen-second diagnostic deadlines.
The cause is unconfirmed. No archive process was stopped or database modified.
The findings above are grounded in the linked committed project records.

## Resumed hourly execution and independent review

Patrick explicitly renewed hourly execution on October 7. The existing
`ace-hourly-curc-research-progress` heartbeat is now **ACTIVE**, hourly, attached
to this chat; its stale pilot-failure prompt has been replaced with the released
Stage B chain and manuscript/review work. The saved status, schedule, thread and
new protocol references were read back and verified. Wakeups advance local
development, theory, analysis and writing even while CURC is maintenance-pending.

A fresh **14:57:39 UTC** read-only check confirms 560 completed fit receipts,
35 completed worlds, the unchanged frozen registration and 18 worker, 19 original
learner and two generator hashes. No full fit seal, evaluation start/directory,
complete study or stop flag exists. The last five thirty-node jobs remain pending
for maintenance and evaluation/audit remain dependency-pending. This lighter
queue check does not replace the earlier full input/model/optimizer custody audit.
Evidence: [fresh queue receipt](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_queue_check_20261007T145739Z.json).

Two bounded review agents completed scientific/statistical/theory and
implementation/reproducibility reports. Their reports and adjudication
are linked from the review register. The main agent corrected novelty positioning,
conditional scratch-fit attribution, empirical quantizer/scaling definitions,
true-parent diagnostics, eligibility precision and telemetry/audit scopes, and
added generated per-history findings. The explicit archived equations match all
390,625 saved responses, with only machine-precision differences in link values
and no target disagreement. Native compilation passed after these edits.
Additional
experiments require a specific unresolved concern, preserved prospective
protocol boundaries and an existing explicit resource authorization. Do not
alter the running study to satisfy a review request. Venue acceptance is not a
guaranteed completion condition; human final scientific and submission approval
remains necessary.

### October 7 anonymous release follow-up

The failed GitHub push recovered; `39fe413e` reached remote main. At 16:07 UTC,
CURC still has the same five maintenance-pending tasks, 560 completed fits, no
frozen source drift and no evaluation/seal/stop flag. The composite inventory
independently reconciles all 560 model/receipt pairs without opening losses.

New release machinery is described in
`delivery_anonymous_release_2026-10-07.md`, with a distinct bounded review in
`reviews/delivery_release_review_2026-10-07.md`. The current private A/C candidate
contains 1,152 files and 2,005 digest bindings; eight focused tests pass. Immutable
original digests are distinguished from newly derived protocol metadata digests.
No blind path replacement occurred and nothing was publicly uploaded.

A read-only relocated C adapter reconstructs all retained neural and linear
predictions and continuous/bootstrap statistics with zero cached discrepancy.
Its receipt is `results/delivery_release_preparation_20261007/verification.json`.
The original physical acceptance is unchanged; this strengthens later release
verification, not the scientific scope or B readiness. Two failed adapter attempts
were retained. Technical lessons: disable Python bytecode writes inside a verified
bundle, and preserve Python-float versus NumPy-scalar casts when reproducing
float32 outputs. The wrong scalar type shifted one condition's predictions by
2.27266e-5; matching frozen arithmetic restored exact outputs without tolerance
relaxation, new fits or responses.

Still required: complete B acceptance/integration, A/B scientific replay entry
points and full frozen-worker provenance, confirmation/claim/attempt accounting,
deliberate anonymity/redistribution review, and human final submission approval.
The same open manuscript compiled after the scoped provenance update.

## October 7 follow-up — full attribution replay (17:14 UTC)

The fresh17:00UTC CURC receipt remains560/640fits with five maintenance-pending
thirty-node worlds, no source drift, full seal, evaluation or stop flag. Existing
jobs and sources are unchanged. No new Slurm submissions or fitted studies.

R7 advanced with a read-only attribution adapter: all480accepted configurations
across all12histories, including124753321 and every classical/matched-CPU control,
reconstruct scores, mechanism diagnostics and quantization margins with maximum
absolute numeric discrepancy zero. It runs from an actually renamed package and
unrelated cwd with exact recorded dependencies, fixed comparison tolerances and
one CPU thread. No optimization, response acquisition, favorable rescue or
historical acceptance edit. The final replay used158.927602childCPUseconds,
159.795762elapsedseconds and1,415,118,848bytes peak childRSS. Smoke and earlier
full replay are retained; total measured childCPU across the three A replay
attempts is350.041653seconds (0.097234corehours). This excludes packaging,
checksum/projection verification, tests and source inventory construction.

An independent bounded adapter review found three robustness/reporting issues:
exact cohort identity, cached learner imports, and score-only discrepancy
reporting. All were fixed and three focused tests pass; eight release tests also
pass. Review/dispositions: reviews/delivery_attribution_release_review_2026-10-07.md.

The latest private A/C candidate contains1,153files/2,005bindings; manifestSHA
63caf4b16757f6aa79c5cf9b1d4452c0f8dfd5b2c45c377af68fdef9605ed28c. All1,152
previously verified A/C artifacts are unchanged; only the new A adapter was added.
C's existing scientific replay therefore retains the exact same numerical/model/
protocol/source byte identities. The prior candidates/receipts remain preserved.
One preparation configuration failure (wrong physical acceptance path) occurred
before candidate creation; it is recorded separately, with no fit attempt.

Preserved24frozen source bindings in exclusive private custody directly from
immutable Git objects: A/C worker and guard,18B workers,2generators. Eleven files
contain screened identifiers; no silent source rewrites, guard bypass or public
upload. The inventory is a source-provenance record, not a complete anonymous
execution adapter. The same open manuscript compiled after the scoped verification
paragraph; nine companion/style files remain synchronized.

Remaining: complete original B acceptance and all registered analyses; B scientific
replay/analysis packaging using complete conflict-rejecting composite custody;
full confirmation/claim/attempt accounting; explicit anonymous worker relocation,
redistribution/anonymity review and final human submission approval. Additional
fits remain unnecessary for the current scoped recipe/accounting claims.
Evidence: results/delivery_release_preparation_20261007/attribution_verification.json.


## October 7 follow-up — confirmation, accounting and claim reconstruction

Fresh CURC evidence at18:01UTC remains560/640fits and35/40worlds, with the
last five thirty-node worlds maintenance-pending. No full fit seal, evaluation,
stop flag or drift. The original queued chain and frozen sources were unchanged;
no new submissions, fits or responses. Queue evidence:
`results/delivery_prospective_preparation_20261006/full_queue_check_20261007T180135Z.json`.

R7 now includes all240sealed original confirmation case files, online weights,
three delivery initializations, original/prior accounting projections and the
separate interrupted journal. A read-only relocated replay reconstructed48model
sets/288neural heads and the paired statistics with maximum numeric discrepancy
5.551115123125783e-17 at unchanged rtol1e-10/atol1e-12. This is floating-point
roundoff, not an exact-zero claim. The original worsening history and all scientific
results remain unchanged. Replay used40.519136childCPUseconds,41.587096elapsed
seconds and463,060,992bytes peak childRSS, including imports/checksums. No A/C
model replay, optimization or new acquisition was repeated.

A distinct original-journal audit clarified60,962charged attempts versus57,636
persisted complete responses and3,326interrupted reservations. Returned responses
for the interrupted attempt are unknown. Thirteen distinct acquisition journals
are counted once; copied retained cases, overlapping summary counters and refit
initializations add no charges. The manuscript explicitly discloses the separately
authorized continuation for the same registered final seed before whole-matrix
scoring. Original hash-bound receipts and gates remain unchanged.

The latest private candidate09 contains1,423files/2,533bindings; manifestSHA
f49733578564e3b8e2e302bc1e41c282dead7a630beaf8764bf1226cbdb5a4fa. All280
confirmation inference/input/source/dependency objects remain unchanged from the
qualified candidate08. Its renamed unrelated-directory metadata check reconstructs
all35empirical macros and verifies complete generated LaTeX macro bytes against
the bound claim index. This used24.032717childCPUseconds,24.043367elapsedseconds
and81,362,944bytes peak childRSS. It does not verify every prose statement/table
or B claims. The current manuscript compiled successfully; all nine companion
and official style digests remain synchronized.

Two distinct bounded reviews are complete: original confirmation accounting and
new release implementation. The accounting ambiguity and continuation disclosure
were corrected; implementation review found no remaining required fixes. Three
synthetic journal/accounting tests pass. Reviewer did not rerun inference; dedicated
replay/macro rejection fixtures remain a testing limitation. Reports are in
`reviews/delivery_confirmation_accounting_review_2026-10-07.md` and
`reviews/delivery_confirmation_release_review_2026-10-07.md`. Evidence is bound by
`results/delivery_release_preparation_20261007/confirmation_verification.json`
and `confirmation_claims_preparation.json`. Integrity and numerical reconstruction
do not establish historical freeze timing, authentication of the supplied digest,
model refitting or final anonymity. This is private preparation, not release readiness.

Remaining independent work: B scientific replay/analysis relocation using exact
registered runtime, explicit frozen-worker/generator/guard handling and complete
conflict-rejecting composite custody; all-sprint compute/attempt disposition;
anonymous worker relocation and redistribution/anonymity review. After the original
640fit seal, evaluation and successful independent audit supervisor, integrate all
registered B outcomes via the claim exporter, then review proofs/prose/tables,
visual layout and submission package. Human final submission approval remains.
Additional fitted experiments remain unnecessary for the current scoped claims.
