# CPU world generator and response adapter

November 4, 2026. Generator capability and artificial qualification only. This is
not a scientific descriptor set, first-screen freeze, provenance qualification,
or authorization to generate policy-training/development/confirmation worlds.
The agenda, execution ledger and first-screen design review remain authoritative.

`scripts/research/llm_intervention_world.py` uses the standard library and the
existing contract. It supplies a known DAG, never discovers or selects one.
`LinearMechanism(node, intercept, coefficients, noise_variance)` and
`WorldDescriptor(scope, world_id, graph, mechanisms)` are frozen, tuple-backed
objects. Mechanism and ordered parent membership must match the topological graph
exactly. Limits are eight nodes, 64 actions/queries and 80 callback requests.
Finite Gaussian parameters do not guarantee finite floating-point compositions:
overflow terminates and preserves the charged failure instead of resampling.

`generate_world(scope=..., world_id=..., graph=..., intercept_means=...,
coefficient_means=..., noise_variances=..., intercept_sd=..., coefficient_sd=...)`
realizes one explicit independent Gaussian parameter law. Tuple rows follow the
supplied graph/parent order. Zero SDs encode analytic fixtures without parameter
draws. There are no defaults for scientific families, namespaces, menus, task
scales, world counts, exposure schedules or selection criteria.

`keyed_seed(scope, world_id, stream, *key)` domain-separates `mechanism`, `initial`,
`measurement`, `policy`, `train_score` and `private_eval`. Keys use canonical
structured serialization, with per-node local `random.Random` Gaussian draws.
No global Python/NumPy RNG is changed. `train_score` is rejected outside
`artificial`/`policy_train`. This versioned key scheme deliberately differs from
the older contract helper, which rejects artificial training scores. Do not mix
the two schemes. Scientific namespace admission remains a future runner/freeze
obligation. Deterministic pseudorandom stream separation is not evidence of
independent real-world systems or contamination-free data.

`world.private_payload()` serializes the complete private descriptor;
`WorldDescriptor.from_private_payload(payload, expected_sha256=pin)` checks its
schema and expected digest. Pins must come from the trusted frozen inputs, not
from an untrusted replacement payload. `world.public_design()` returns only a
detached graph. Keep descriptors, seeds, pins, adapters and their journals out of
the policy process. Send the existing validated public task envelope to the
persistent policy service. Known noise supplied to the learner must be an
explicit public modeling assumption in the later protocol. Hash checks and
Python access conventions are not a sandbox against malicious same-user code.

## Paid response integration

Construct `ResponseAdapter(world, expected_sha256=pin, menu=menu, stream=stream,
replicate_id=rid, request_namespace=snapshot_or_episode_key,
executor_journal=charged_path, journal=private_new_path, max_requests=cap,
structural=False)` before opening the existing executor. It is a context manager
and callable as `adapter(action, journal_request_id) -> dict[str, float]`.
The journal parent must already exist. Both journals are exclusive; the adapter
does not create, reserve, refund or resume a charged executor.

On each call the adapter verifies the bounded executor journal hash chain,
graph/menu binding and current matching durable reservation. A private mapping
precedes evaluation; outcomes/failures are flushed and fsynced. Duplicate calls,
foreign actions, wrong pins, exhausted caps and incomplete/tampered journals
fail closed. An outcome records that the subset was prepared, not that the
executor accepted it: a crash between persistence and return stays ambiguous.
The existing charged journal remains the sole cost and acceptance authority.
Use one trusted writer per adapter/executor; concurrent/reentrant use and
malicious journal forgery are not supported.

Hard intervention replaces the clamped equation and its noise. Every other
equation retains independent structural noise and uses its generated parents.
Only the exact measured subset is returned. Noise keys include scope/world,
stream, request namespace, replicate, request ID, action ID, intervention and
node. Measurement subsets do not change a structural draw. Reusing those keys
intentionally pairs responses; use a new predeclared namespace for independent
episodes/snapshots. Changing menu/candidate iteration order alone does not
change branch noise. A repeated paid action uses a fresh request ID.

For a single episode, bind separate `initial` and `measurement` adapters to its
same charged journal and dispatch by the fixed initialization request schedule.
Both must use the complete executor menu. Do not choose stream boundaries from
observed outcomes. Policy randomization uses `policy`, never environment RNG.

## Training and private evaluation handoff

`training_callbacks(stack, world, expected_sha256=pin, menu=menu,
candidates=candidate_ids, task=task, replicate_ids=replicate_ids,
branch_output=new_driver_output, journal_directory=existing_private_directory,
request_namespace=snapshot_key, structural_scores=True_or_False)` returns:

- `branch_respond(replicate_id, candidate_id, action, journal_request_id)`;
- `score_respond(replicate_id, action, journal_request_id)`.

Pass these directly to `build_replicated_training_preferences`, inside the
caller-owned `contextlib.ExitStack`. Use exactly the same graph/menu/task,
candidate and replicate order and output path. The helper binds the driver's
existing filenames; the driver still owns all reservations, shared pre-action
learner forks and journal-to-learner request mappings. Keep its plan, receipts,
failures and all private adapter journals together. Branches use noisy
`measurement`; shared scores use separate `train_score`. Explicit
`structural_scores=True` gives charged, privileged noise-disabled training
supervision; `False` gives independent noisy scoring. Scores never update the
branch learner. Do not reinterpret structural-score equality across replicates
as independent score noise. Branch-outcome variation remains present.

`private_structural_means(world, queries, expected_sha256=pin)` computes true
noise-disabled interventional means independently of the learner, training
scores and measured rows. It returns a tuple of exact requested subsets. This
pure private computation makes no environment reservation and samples no RNG;
a future noisy evaluator must use `private_eval`. Its computation and output
custody belong in the future evaluator ledger. Never expose it as a free
deployment measurement or stopping oracle. Training structural responses use
the charged adapter, not this function as a bypass.

The learner's conjugate local factors require known noise and observed or
clamped parents. Omitting rows with genuinely missing parents is not exact
inference over partially observed worlds. This module makes no such claim.

## Preparation custody and validation scope

New tests are only `scripts/research/test_llm_intervention_world.py`: analytic
chain/fork/collider means, clamp cuts, nonintervened noise, subsets, immutability,
serialization, stream/order/RNG separation, invalid membership/pin/reservation
rejection, bounds, overflow failures, and integration with the existing paid
replicated driver. Existing test suites are not repeated. No model, download,
installation, remote operation, GPU or scientific descriptor generation occurs.

Repository custody: `results/ace_llm_intervention_world_preparation_20261104`.
Exclusive external original:
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-world-preparation-20261104-01`.
Exact sources/dependencies, command/environment, outputs, fixture journals,
failures and measured child CPU/wall time are preserved there. Read-only
inspection and implementation are not fully metered; test process CPU is not
whole-cycle CPU. This local work does not change scheduler allocation totals or
expand the existing cumulative preparation allowance. Independent review follows.

## Custody clarification after independent review (November 4)

The independent world review found no required generator/adapter logic
correction within the documented scope. Its custody finding is addressed here
and in the new `custody_clarification01/receipt.json`; this is a documentation
disposition, with no test rerun or implementation/test change.

The tamper fixture's final `tampered/charged.jsonl` is a **valid post-failure
journal**, not a retained corrupted input. It contains the preserved 1,190-byte
genuine prefix followed by a 351-byte executor failure record, totaling 1,541
bytes. The executor's original open handle overwrote the separately appended
malformed suffix when recording failure. The original closeout field named
`intentional_corrupted_executor_journal` therefore misdescribes that final file.
This clarification supersedes that description; the original closeout remains
unchanged.

Original evidence retained: `before_corruption.jsonl`, the pinned fixture source
containing the exact injection literal, both recorded hash-chain rejections,
and the valid final charged journal. **The exact rejected corrupted input was
not separately captured.** It can be reconstructed from the prefix plus the
source literal, but no reconstructed file is produced or represented as an
original observation here. Reservation/failure accounting and the twelve-method
result are unchanged. All original 133 attempt files remain byte-identical in
repository and external custody. A future test claiming exact malformed-input
custody must save injected bytes separately before invoking the callback; that
future fixture change has not been made or executed.
