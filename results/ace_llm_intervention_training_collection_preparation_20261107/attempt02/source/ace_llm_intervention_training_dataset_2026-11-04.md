# Single-world training-data assembly primitive

November 4, 2026. CPU implementation and artificial qualification only. The
frozen 32-training/24-development-world schedule, scientific namespace admission,
actual collection and policy training remain future, separately authorized work.
This primitive does not generate worlds, train policies, load models or evaluate
scientific performance. Purpose follows the agenda and first-screen design:
give both learned families identical paid states, supervised choices and measured
preference labels, while keeping all failures and training costs visible.

`assemble_training_dataset` in `scripts/research/llm_intervention_training_dataset.py`
is a callable with no CLI or scientific defaults. Supply a `WorldDescriptor` and
independent expected descriptor pin, matching `scope` (`artificial` or
`policy_train`), graph/menu, `PredictionTask` and its independent pin, fresh
`LinearGaussianSCM`, fixed initial action IDs and total trajectory budget. The
learner's noise/prior assumptions are explicit public inputs, not copied from
private mechanisms. The caller's learner stays unchanged; a private fork receives
only paid trajectory receipts.

Also supply the behavior method, a tuple of `Snapshot(acquisition_cost_units,
teacher_method, candidates)`, replicate IDs, planning draws, tie parameters,
`structural_scores`, public objective/notes, request namespace and exclusive output
path. Behavior and teacher methods are `uniform_random`, `menu_coverage`,
`local_information`, or `target_risk`. Their deterministic policy streams use the
world helper's separate `policy` namespace and fixed role/index. Assignment to a
method precedes all outcomes; subsequent behavior decisions can use paid history.
Only public task envelopes reach these numerical selectors.

The durable plan precedes every response and records all assignments, candidate
sets, budgets, query weights/scale/target, streams, source hashes and expected
feedback counts. Initialization is included in the trajectory budget. Milestones
count acquisition cost **after initialization**: zero fires immediately after the
fixed initialization; every other milestone fires at the first completed response
whose cumulative acquisition cost reaches or exceeds it. One expensive response
can cross multiple milestones; every one is retained with the actual crossing cost
and its own branch/scoring namespace. No threshold, teacher or candidate set is
selected from observed gains. Milestones are strictly increasing and bounded by
the acquisition budget. Unreached milestones fail rather than vanish.

Every trajectory response uses `ChargedExecutor` and a reservation-verifying
`ResponseAdapter`. Initial and acquisition streams are separate. At each snapshot,
the full public envelope, selection receipt and teacher choice are saved before
measured feedback. Nonempty candidate sets must contain at least two exact,
affordable menu members; none are pruned. The reviewed replicated-branch helper
forks the same pre-action learner for every candidate/replicate and shares separately
paid task scores across candidates. Scores never update the trajectory or branch
learners. `structural_scores=True` explicitly denotes charged, privileged training
structural labels; false uses noisy training scores. Neither is private evaluation.

An explicitly planned empty candidate set is valid only when no trajectory action
is affordable. It yields a terminal stop SFT example and no preference pairs or
feedback calls. Negative gain never supplies a stop label. Design candidate sets
and milestones to remain feasible after cost overshoot; the driver fails if that
fixed design cannot be honored. It does not adapt the candidate set for convenience.

Each snapshot retains complete raw branch/scoring evidence and replicate-aware
losses/gains, all candidate pairs including ties and negative gains, a hashed
supervision receipt, and the exact shared `record.json` consumed by `matched_views`.
Saved language/numerical views and alignment refer to the same record. Ties stay
in that record/alignment; the existing matched-view consumer exposes only strict
pairs as DPO/ranking targets. Its fixed tie rule is heuristic, not an inference
certificate. No new training or stop-inference rule is introduced.

`terminal.json` reports actual trajectory/branch/scoring charges and response
counts separately, source/task/method scope, completed records and learner-update
counts. Every durable reservation remains charged. A prepared adapter outcome is
not proof of a returned response. Failure retains `failure.json`, partial records,
raw invalid returns, unknown attempts, journals and pending snapshot indices;
there is no fallback, retry, resume, complete-case filtering or authorization to
train on a failed world's completed subset. Output creation is exclusive.

Technical bounds are the public schema's 8 nodes/32 menu actions/64 queries/80
trajectory responses, plus at most 16 snapshots, 16 replicates, 4,096 feedback
responses and 2–256 planning draws. All acquisition costs are positive integers;
trajectory budget is at most 1,000,000 units. These are representation/work caps,
not a scientific exposure schedule or wall-time supervisor. The trusted sequential
driver assumes responsive storage; it is not a malicious-same-user sandbox.

Descriptor binding and deterministic stream separation do not establish honest
public text, scientific independence, split cleanliness or protocol admission.
Observed source hashes are recorded after imports, not independent executable
authentication. All outputs retain `provenance_qualified=false` (or the equivalent
scientific field). The parent owns runtime/source admission, namespace governance,
outer resource limits, full-study scheduling and later training/integration.

New artificial tests only: two-node analytic descriptors, hand-computed gains and
charges, threshold overshoot, teacher-before-response custody, exact matched views,
negative/tied pairs, isolated learner updates, invalid plan/scope/pin rejection,
failed/invalid returns, infeasible/unreached snapshots and exclusive output. The
new result directory preserves exact commands, test source, logs, fixtures and
test-process CPU/wall measurements. Existing suites are not rerun. Local test CPU
is not whole-cycle CPU and does not alter scheduler accounting.

Repository custody: `results/ace_llm_intervention_training_dataset_preparation_20261104`.
Exclusive external custody:
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-dataset-preparation-20261104-01`.
