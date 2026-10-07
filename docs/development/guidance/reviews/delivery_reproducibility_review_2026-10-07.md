# Delivery manuscript: reproducibility and implementation-methods review

Reviewed 7 October 2026. Scope: implementation semantics, accepted A/C custody,
frozen B design, compute reporting, leakage boundaries, and anonymous artifact
readiness. Manuscript reviewed after `\documentclass`, starting at line 2404;
bundled style was not reviewed as scientific prose. Reviewed `paper.tex` SHA256:
`2817ad6b7426f4cbeb75b5218b90bf17ea236c34efab5cdb197a670c1dd864b1`.
Line anchors below refer to that snapshot; the main agent may subsequently edit it.

No on-disk AGENTS.md was found in ACE or its applicable ancestor directories.
The user-supplied shared-GPU instructions and bounded review instructions applied.
This review used local source and already validated receipts only. No experiments,
fits, environments, remote access, closed APIs, frozen-worker changes, original
outcome changes, or unopened Stage B outcome inspection were performed. Only this
review file was written; no commit was made.

## Assessment

The accepted evidence and main deployment semantics are internally consistent
with the draft. I found no evidence of an observed replay-mask or predicted-parent
inference bug invalidating the accepted A/C findings. The concrete issues below
are methods-definition omissions, interpretation limits of the frozen study,
and release requirements. They do not authorize changing the frozen workers or
rerunning scored studies. B remains pending at the recorded 560/640 checkpoint.

## Findings requiring manuscript clarification

### R1 — P2: Specify the Stage A NMSE denominator separately from B/C

**Classification: methods omission, not a scoring bug.**
The endpoint paragraph defines *prospective* NMSE using training target variance,
but the Stage A results and table use NMSE without stating their different
denominator. Stage A calls the archived `score(pred, truth.y, levels)`, whose
`_r2_parts` divides by the population variance of the exposed evaluation grid's
target, not any retained training set. A reader implementing all reported NMSEs
with the training-variance definition will obtain different absolute Stage A
NMSEs. Within-history delivery/control ratios are unaffected because the grid
denominator is shared; this is not evidence of outcome-dependent training scaling.

Evidence:

- [Endpoint definition](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2542) and [Stage A NMSE claims](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2652).
- [Stage A scorer call](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:268).
- [Frozen grid variance implementation](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source/ace/grid_eval.py:517), whose source digest matches the B learner binding at [registration source hashes](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json:75).

For the retained worsening history's `all_paid-scm-30000-i0` score, `mse/nmse`
is `0.06259199408610418`. State explicitly that Stage A uses exposed-grid target
variance, whereas B uses the full corresponding training history and C uses
condition-specific training variance. Preserve existing scores.

### R2 — P2: Define the actual exact-level quantizer, including extrapolation

**Classification: methods mismatch; no demonstrated corruption of existing scores.**
The draft describes mapping to the nearest target level. The archived scorer
instead extends the level lattice beyond both endpoints using the adjacent end
spacing, and resolves midpoint ties downward. A nearest-*finite-level*
implementation would clip extreme predictions to an endpoint and can incorrectly
count them as exact. For levels `{0,1}`, truth `0`, prediction `-1`, finite-level
snapping counts exact; the recorded extended-lattice scorer counts an error.

Evidence:

- [Manuscript quantizer definition](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2547) and [confirmation endpoint](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2601).
- [Frozen quantizer contract and implementation](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source/ace/grid_eval.py:481), and [exact-score use](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source/ace/grid_eval.py:557).
- [Theory's finite-level quantizer](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex:109) and [diagnostic's internal-boundary construction](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:262).

Add the scorer's endpoint and tie conventions to the empirical endpoint. Explain
whether the theoretical finite-level proposition is a separate example or uses
the empirical quantizer. For true values exactly at the recorded levels, the
missing outer boundaries have the same nearest distance as the adjacent inner
boundary, so this review does **not** establish a wrong archived margin fraction.
The issue is an underspecified empirical definition and the scope of the general
quantization statement; do not silently replace the scorer.

### R3 — P2: Retained-row contrasts also change the input normalizer

**Classification: frozen existing-study attribution limitation.**
The scratch matrix fixes architecture, seed scheme, optimizer and masks, but
normalization is recomputed separately for each retained row set and each head.
Consequently the all-paid/final-buffer and all-paid/admitted contrasts change
both observations and the numerical input representation. The same parameter
initialization under different scales does not give the same initial function
of physical parent values. The wording that retention explains the improvement
should include this qualification rather than imply an isolated row-count effect.

Evidence:

- [Factorial description](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2627), [attribution interpretation](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2659), and [normalization statement](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2892).
- [Per-cell eligible-row range calculation](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:177).

Read-only reconstruction from the sealed input for history `124753321` gives
these target-parent (`iftu_score`) ranges and eligible counts:

- Final buffer: 50 rows, `[0.2321354166666667, 0.8352916666666668]`.
- Online-admitted union: 1,178 rows, `[0.07374999999999998, 0.8960000000000001]`.
- All paid: 4,803 rows, `[0.07011249999999998, 0.9534759566531541]`.

The input location and digest are bound by the [Stage A registration](/Users/pat/code/ACE/results/delivery_attribution_20261006/registration.json:4)
and input protocol. Add a sentence that the retention contrasts include changes
in training-derived scaling. The all-paid 100/30,000 contrast holds its row set
and scale fixed. A scale-controlled retention study would be future evidence;
it is not a reason to alter this accepted matrix or its selection gate.

### R4 — P2: Label the 0.020% box statistic as an observed-parent diagnostic

**Classification: interpretation ambiguity, not an evaluator bug.**
The failure paragraph says evaluated target parent values lie outside the
training box. The implementation computes this fraction using `truth.nodes`
for the target's parents. It does not compute the fraction of **predicted**
parents presented to the deployed target head. Readers could therefore infer
that deployed inputs were almost entirely within the training box, which the
reported statistic does not establish.

Evidence:

- [Failure paragraph](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2685).
- [Observed-parent box calculation](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:250), contrasted with [free-running parent inputs](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:249).

Change the prose to “evaluated **observed-parent vectors**” and retain the joint
support caveat. No predicted-parent outside-box statistic was established by
this review. Any later diagnostic must be clearly additional analysis of the
unchanged saved model, rather than a substituted score or a causal explanation
of the failure.

### R5 — P2: Make the numerical kernels reproducible without reverse engineering

**Classification: methods omission; current implementations are consistent.**
The appendices omit two consequential numerical details. In B, calibration
precedes retrospective updates, but the first 50 rows are then replayed from an
initially empty buffer as part of **all 400** updates; they are not merely seeded
into replay followed by 350 training arrivals. Each arrival has 20 fast updates
and 100 consolidation updates per nonroot head, giving **48,000 updates/head**.
Consolidation uses up to 50 buffer entries plus the duplicated current row.
In C, neural angle inputs are divided by 90, while outputs are standardized;
the manuscript currently states only the output transform.

Evidence:

- [Rolling comparator description](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2907) and [physical implementation description](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2965).
- [B prefix calibration](/Users/pat/code/ACE/scripts/research/delivery_prospective_models.py:52), [empty context](/Users/pat/code/ACE/scripts/research/delivery_prospective_models.py:101), and [all-row replay/counts](/Users/pat/code/ACE/scripts/research/delivery_prospective_models.py:133).
- [Archived fast/consolidation kernel](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source/ace/oracle.py:1263).
- [C angle and response transforms](/Users/pat/code/ACE/scripts/research/chambers_delivery_validation.py:128).

State these details directly. Specify affine min–max scaling as
`(x-lo)/(hi-lo)` without input clipping, and retain separate Stage A/B/C seed
schemes in the artifact. The existing B initialization offsets are described
accurately; they should not be retrospectively imposed on A or C. Adam defaults,
float32 neural arithmetic and stage-specific dependency versions should accompany
the executable release. No kernel change is requested.

### R6 — P2: Qualify per-fit telemetry and publish the actual compute scopes

**Classification: reporting overstatement and unfinished release evidence.**
The prose says CPU time, wall time and memory accompany each fit. A has head/fit
CPU and wall timing plus process RSS; B has fit CPU/wall and per-attempt supervised
memory. C instead records CPU/updates/parameters per method and memory at the
**whole fit-process** level; it does not record method-specific wall time or peak
memory. A universal “per fit” statement overstates the granularity of C's receipts.

Evidence:

- [Universal accounting claim](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2803) and [appendix per-fit claim](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2882).
- [C method costs](/Users/pat/code/ACE/scripts/research/chambers_delivery_validation.py:144), [phase totals](/Users/pat/code/ACE/scripts/research/chambers_delivery_validation.py:180), and [fit-process RSS receipt](/Users/pat/code/ACE/results/delivery_chambers_20261006/fit_execution.json:5).
- [B requested allocation](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json:150), [failed-pilot charge](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/pilot33505661_failure.json:8), and [successful-pilot accounting](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/pilot_importfix_sacct.txt:2).

Keep A's 10.87162054 fit CPU-core-hours and C's 0.05285207 fit/evaluation
CPU-core-hours with their existing exclusions. Describe memory as measured at
fit or phase granularity as available. Final compute reporting should distinguish
measured fit CPU, supervised elapsed execution, startup/imports, development
pilots, and scheduler allocated CPU-seconds. The 86.86833333 B hours are a
rounded reservation, not measured completed cost. The failed 126-second and
successful 119-second pilot allocations must survive final reporting. The
existing B work is CPU-only; no GPU justification or new allocation is needed.

## Anonymous artifact readiness and audit limits

### R7 — P1 before submission: Anonymization must preserve verifiable custody

**Classification: acknowledged release prerequisite, not a current study bug.**
The manuscript promises an anonymous reproducibility copy but contains no
executable relocation/anonymization contract. Original receipts contain personal
and account paths, and these bytes are covered by registration, matrix, model
receipt and claim-index hashes. Simply replacing paths in JSON invalidates the
existing custody chain. Preserving all JSON unchanged in a publicly accessible
anonymous package exposes those identifiers. This needs a deliberate derived
release representation before submission.

Evidence:

- [Anonymous release requirement](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3001).
- [Identifying B paths](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json:5), [A paths](/Users/pat/code/ACE/results/delivery_attribution_20261006/registration.json:4), and [C paths](/Users/pat/code/ACE/results/delivery_chambers_20261006/protocol.json:3).
- [Canonical matrix paths and protocol binding](/Users/pat/code/ACE/scripts/research/audit_delivery_prospective_results.py:45) and [physical protocol binding](/Users/pat/code/ACE/scripts/research/chambers_delivery_validation.py:43).
- [Split local fit snapshots](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_milestone_custody_20261007T134904Z.json:11).

Create a derived, independently hashed anonymous manifest with relative artifact
locations and an explicit mapping to the immutable source digests. A verification
entry point must resolve local paths independently of original machine names,
while checking unchanged model/data bytes and preserving original custody
privately. Do not edit frozen workers or original receipts to achieve relocation.
Include the archived learner source as well as the ACE project workers: the ACE
checkout alone does not provide `ace.oracle` or the recorded exposed-grid file.
Reconcile the two fit snapshots without overwriting conflicts. An unrelated-cwd,
offline verification of the derived package remains future release evidence.
The handoff already lists an anonymous bundle as unfinished; do not label the
current working draft submission-ready.

### R8 — P3: Preserve the distinction between hash acceptance and checkpoint replay

**Classification: existing audit limitation; not grounds to revoke acceptance.**
A's independent auditor checks source, eligibility digests, models, matrix
membership and score-file bindings, but does not independently run saved models
against the grid. C independently recomputes NMSE/bootstrap from sealed prediction
arrays and checks physics predictions against saved coefficients; it does not
replay the saved neural models or reconstruct Fourier predictions from their
saved coefficients. B's prospective auditor explicitly requires all 640 checkpoint
replays. These are different evidence strengths.

Evidence:

- [A model and score bindings](/Users/pat/code/ACE/scripts/research/audit_delivery_attribution.py:123) and [complete-score acceptance](/Users/pat/code/ACE/scripts/research/audit_delivery_attribution.py:145).
- [C prediction-array and physics checks](/Users/pat/code/ACE/scripts/research/audit_chambers_delivery.py:55).
- [B export checkpoint gate](/Users/pat/code/ACE/scripts/research/export_delivery_prospective_claims.py:19).

The current paper's scoped “audit of ... score files” is supportable. Avoid
expanding it to independent checkpoint reproduction for every completed study.
Any future verification should replay unchanged checkpoints, never refit them or
overwrite accepted scores. Similarly, the Python audit hooks are scoped guards
for the trusted workers, not an operating-system sandbox: A blocks audited socket
connects and `.npz/.npy` opens; B adds registered outcome/mechanism path guards.
C loads the full archive in its fitting process, but indexes only training rows
for objectives and scales and seals predictions before held-out losses. No
actual leakage was found; describe these boundaries precisely.

## Checks that support the current draft

- Replay reconstruction uses paid query IDs, excludes clone-only fits, duplicates
  the selected observation in consolidation, and admits refreshes after fitting:
  [reconstruction](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:32).
  The checked worsening history's selected clamped fast-update count is empty;
  the full accepted summary records zero selected clamped-node updates. The
  latent unmasked fast path is a frozen code limitation, not an observed
  explanation of the A gain. B joint-root histories keep nonroot masks vacuous.
- [Eligibility](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:91)
  removes a head's own clamp and excludes intermediate-clamp rows from the flat
  predictor. The paper correctly warns that a matched paid history need not mean
  identical usable rows or supervision.
- [B evaluation](/Users/pat/code/ACE/scripts/research/delivery_prospective_io.py:134)
  composes predicted parents; measured parents are used only in separately named
  diagnostics. [Existing fixtures](/Users/pat/code/ACE/scripts/research/test_delivery_prospective_io.py:38)
  test this by changing observed test intermediates. Checks were read, not rerun.
- A's matrix is 12 × `(3 row sets × 2 neural arms × 2 budgets × 3 inits +
  3 classical + 1 matched-CPU)` = 480. B's eight cells/history produce
  40 × 2 × 8 = 640, with 240 primary cells and the ablation separate.
  [A membership audit](/Users/pat/code/ACE/scripts/research/audit_delivery_attribution.py:93)
  and [B frozen cells](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json:112)
  agree with the text. The [B analysis](/Users/pat/code/ACE/scripts/research/delivery_prospective_analysis.py:69)
  averages history log ratios within systems and applies Holm to four tests.
- [C partition](/Users/pat/code/ACE/scripts/research/chambers_delivery_validation.py:61)
  groups repeated commands in joint 30-degree blocks. Training variance, eight
  held-out blocks, all eleven conditions, the relative-angle physics basis and
  the 25-feature Fourier basis agree with the accepted receipts. Row-weighted
  primary errors and equal-block bootstrap ratios are correctly distinguished.
- The read-only companion synchronization check passed for all eight embedded
  files and checked official style hashes. The anonymous author/PDF-author
  declarations and pending-results notice are present. This was a source and
  bundle review, not a native compile, visual page review or venue-policy audit.

## Verification performed and outstanding evidence

Executed only `python3 -B scripts/research/sync_delivery_manuscript.py --check`
and standard-library read/hash checks. Confirmation score/statistic hashes,
the claim-generator digest, attribution gate and all twelve score hashes,
physical acceptance and all its receipt hashes matched `claim_index.json`.
All B registered worker and generator hashes matched the local source, and the
archived learner source matched the B source binding. The B registration matched
`e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5`.
These checks establish byte consistency, not a newly repeated scientific audit.

The last authorized recorded checkpoint remains 35/40 worlds, 560/640 fits,
32,000 charged training responses, zero generated held-out responses, no failed
attempts, and no scientific acceptance. Its [custody receipt](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_milestone_custody_20261007T134904Z.json:7)
explicitly marks partial custody as insufficient. No live status was queried.
Require the original full fit seal, shared 16,000 evaluation-response custody,
all 640 checkpoint replays, four registered contrasts, successful audit execution,
and unchanged-init sensitivity/ablation reporting before exporting B claims.
Final compute and the derived anonymous executable bundle remain outstanding.
No additional fitted control or experiment is recommended within this review.

Review complete. Main-agent actions are prose clarification and release planning;
manuscript, automation, frozen study execution and result acceptance remain with
the main agent.
