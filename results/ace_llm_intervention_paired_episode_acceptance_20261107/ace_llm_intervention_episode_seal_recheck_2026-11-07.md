**Independent narrow sealer correction recheck — November 7, 2026.**

**Required findings remaining: 0. The original P2 terminal-posterior
reconciliation requirement is resolved** for sealer SHA-256
`91b7a40829157cc21d586fd22ae17eee6556d1b73da1dfbf2f4b104564f39811`.
This addendum supplies the new component-review disposition. The
[original review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_episode_seal_review_2026-11-07.md)
remains byte-for-byte unchanged, including its historical P2 finding. This
recheck does not duplicate or certify Sagan's separate paired-finalization work.

Reviewed only the posterior correction in
[sealer lines 298–400](/Users/pat/code/ACE/scripts/research/llm_intervention_episode_seal.py:298),
the necessary disposition/provenance wiring, updated fixture helper, four new
focused methods, guide and correction capsule. The old journal/accounting/design
review remains closed. AST comparison confirms the unchanged original functions
outside `_state`, `_disposition` and `seal_episode`; the three added functions are
`_solve_precision`, `_posterior` and `_posterior_check`. The diff in disposition
and sealing propagates reconciliation evidence and updates provenance rather than
changing the existing atomic-prefix selection or bundle accounting rules.

**Why the correction closes the finding.** `_posterior` independently builds
each node's batch precision and natural mean vector from the pinned prior/noise
configuration and the accepted receipt prefix:
`P = I/prior_variance + sum(phi phiᵀ/noise_variance)` and
`b = sum(phi*y/noise_variance)`. It omits hard-intervened nodes, requires a
measured response and all available parents, and uses both measurements and
known clamps for covariates. No private mechanism parameter or existing learner
update enters this computation. The Cholesky factorization and forward/backward
solves give `P⁻¹ b`; solving each basis vector gives all entries of `P⁻¹`.
The matrix indexing and column-to-row assembly are consistent.

The existing schema checks establish one correctly ordered, finite, dimensioned
summary per node before comparison. Lines 345–373 compare every mean and every
covariance entry with the fixed rule
`abs(observed-reconstructed) <= 1e-10 + 1e-8*abs(reconstructed)`;
eligible-row counts must match exactly. All nodes and entries must match the
same whole permitted prefix. It cannot combine a full-prefix mean with a
short-prefix covariance or choose different prefixes independently by node.
Nonfinite arithmetic, nonpositive factors and failed reconstruction do not
trigger floors, regularization or relaxed tolerances. A candidate failure is
recorded, after which an already-permitted shorter atomic prefix can still be
checked. The routine never creates a new prefix exception to rescue a mismatch.

The original disposition logic still limits the last-unapplied alternative to
the documented driver-failure boundary. Available failed states can reconcile
against either permitted whole prefix while their emitted terminal remains
failed. Initial accepted-but-unapplied state uses the prior prefix when required
by the initialized count. An explicit unavailable failed state still skips
reconstruction, records `unavailable_failed_state` and has
`posterior_replay_performed=False`; a complete source cannot use this exception.
Matched available states instead record `True`, permitted/matching prefix
lengths and per-node comparison flags. Private scoring and scientific acceptance
remain false. The guide accurately states these meanings and numerical limits.

**Retained focused evidence.** Inspected all seven saved successful sealing
results and eleven saved `SealError` rejections. Independently checked their raw
input/output hashes, canonical encodings, terminal hash inventories, posterior
flags, prefix comparisons and mutated input fields. This was saved-record
inspection, not execution of the sealer or tests.

For the original hand case, the retained terminal now has X mean `1/8`, variance
`1/2` and one eligible row, correcting the exact counterexample in review one.
For Y, the two rows give precision
`[[3, 5/4], [5/4, 33/16]]` and natural mean vector `[3, 15/8]`.
Its independently calculated mean is `[123/148, 15/37]`, covariance
`[[33/74, -10/37], [-10/37, 24/37]]`, with two eligible rows. These values
match the saved positive fixture. The heterogeneous control uses prior variance
two and noise variances three/five: X becomes mean `1/10`, variance `6/5`;
Y becomes mean `[267/463, 150/463]` and covariance
`[[570/463, -200/463], [-200/463, 720/463]]`. Changing only the pinned prior
after that control rejects. Empty initialization with a clamped root retains
that root's prior rather than applying an observational update.

All three mean-entry and five covariance-entry perturbations are retained as
posterior rejections. The atomic acquisition controls permit `[2, 1]` and match
only their declared full or short state; both remain failed with two accepted,
charged receipts. The short Y posterior is mean `[8/11, 2/11]` and covariance
`[[17/33, -4/33], [-4/33, 32/33]]`, with one eligible row. The mixed full-mean /
short-covariance control and wrong eligible count reject. Initial-unapplied
matches prefix zero while retaining one charged accepted receipt. Unavailable
failure preserves its failed status and charges without claiming replay.

The retained receipt explicitly selects only four
`PosteriorCorrectionTests` methods and records exit zero, Python 3.9.6, cwd
`/tmp`, `Ran 4 tests in 0.038s`, child CPU `0.11304499999999999` seconds and
elapsed `0.12865291699999998` seconds. Timestamps are
`2026-11-07T17:56:04.462889+00:00` through
`2026-11-07T17:56:04.591741+00:00`; stdout is empty. These are the retained
author-run measurements, not new reviewer test executions. AST comparison
independently confirms all six original test method bodies are unchanged;
their shared fixture helper now uses exact rational small-matrix inverses.
The original six methods were not part of the recorded correction invocation.

**Custody verification.** Recomputed all 270 correction manifest members and
verified the complete **271-file / 447,441-byte** tree against its exclusive
local custody copy at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-episode-seal-correction-20261107-01`.
All seven source/dependency/context files in the receipt match current files,
`source02/` and closeout pins. Also recomputed the original 877 manifest members
and verified the entire **878-file / 9,480,193-byte** original capsule and its
local custody copy remain unchanged. No original artifact was rewritten.

SHA-256 pins:

- Corrected sealer (34,494 bytes):
  `91b7a40829157cc21d586fd22ae17eee6556d1b73da1dfbf2f4b104564f39811`.
- Current tests (28,852 bytes):
  `7f6ca5e8099fb16fbfcc4fa0203d2390d17597935aa2b2de57ebc0145c90125a`.
- Current guide (9,712 bytes):
  `5abb2770774c23feba0801cec2d43142fcfc6f301d1437999180e46735990439`.
- Correction `manifest.json`:
  `a297e98fdb2f0bfd9a24d6baf74f835adfb136c1605281dcf6144bf83b326ba6`.
- Correction `review_request.md`:
  `5e3510f25844fd34ea69b50885ec1ce122908cfc0f70b50bfa9a5b9a9f24b248`.
- Correction `test-attempt01/receipt.json`:
  `3f72b0c00a23cc96c0fb46ae7610ba24c56bc92adafd8ce61da6fb9cf0710a6b`.
- Correction `test-attempt01/stderr.txt`:
  `19b4b73556720d0ce767e78017f9d69aeace7c94611c1e6659e5d4d3d7475cd2`.
- Correction `closeout.json`:
  `6ab5a19ea9a6bbbe767cdf9f1f13364c99ff5bbdbd1c2f34d9962c50e37d65ca`.
- Preserved original review:
  `57135c276ebc6d2d5a554e4a6cc44f4db99d2c3bc3483de82ce7cda7836fe46d`.
- Preserved original manifest:
  `aa049adf744ca164902ac4777ce6f0cb94e30700e65f50d99a9ead6f3294e05f`.

Correction-relative paths are under
`results/ace_llm_intervention_episode_seal_correction_20261107/`.

**Remaining caller boundary.** This closes numerical terminal consistency for
available states under the declared tolerance; it does not authenticate original
source/runtime honesty, independently establish input pins, reconstruct policy
process inputs, authenticate the outer receipt or prove physical process closure.
Those original obligations, complete roster retention and shared initialization
across arms remain with the caller. Unavailable failed states remain explicitly
unverified numerically. The saved focused tests use one- and two-parameter
mechanisms; larger bounded dimensions and reconstruction-error handling were
inspected statically, not empirically exercised here. Ill-conditioned inputs can
fail closed. No statistical or scientific certificate follows from these checks.
Parent/Sagan owns the separate affected finalization on saved raw outputs.

This recheck performed only source/AST comparison and standard-library inspection
and arithmetic on retained artifacts. It ran no sealer, old/new suite, learner,
model, policy/environment, private scorer, reporter, remote command, install or
commit. No counterexample probe was needed. This addendum is the only file written
for the sealer recheck; the separate account review was completed independently.
