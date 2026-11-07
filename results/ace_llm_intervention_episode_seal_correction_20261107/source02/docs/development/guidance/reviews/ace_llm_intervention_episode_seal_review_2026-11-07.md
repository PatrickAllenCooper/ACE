**Independent bounded episode-sealer review — November 7, 2026.**

**Status: changes required for the requested acceptance criterion. Required
findings: 1 P2; 0 P0/P1/P3.** The sealer does not independently reconcile terminal
learner parameters with accepted receipts. Its documentation and provenance flags
honestly disclose this limitation, but disclosure does not satisfy the current
review request's independent terminal-state replay requirement. No additional
required defect was identified in the bounded journal/accounting adapter review.

Reviewed Mencius's complete new
[sealer](/Users/pat/code/ACE/scripts/research/llm_intervention_episode_seal.py),
[test file](/Users/pat/code/ACE/scripts/research/test_llm_intervention_episode_seal.py),
[guide](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_episode_seal_2026-11-07.md),
and [review request](/Users/pat/code/ACE/results/ace_llm_intervention_episode_seal_preparation_20261107/review_request.md).
Consulted the old contract/executor, episode driver, learner equations,
descriptor/capture interfaces and cost-reporter input schema as source text only.
Did not inspect or duplicate the separate paired-runner review or its integration
execution. No conclusion here certifies that component.

**[P2] Reconcile terminal posterior values before sealing a consistent episode.**
Location: [sealer lines 308–324](/Users/pat/code/ACE/scripts/research/llm_intervention_episode_seal.py:308),
especially the schema construction at 317–318 and count-only comparison at
320–324. `_state` validates summary dimensions, finite entries, noise settings and
eligible-row counts. It never checks `parameter_mean` or
`parameter_covariance` against a fresh replay of the pinned learner and accepted
receipt prefix. `prior_variance` is not used in this state comparison. Any finite,
correctly shaped parameter values can therefore pass these checks while
contradicting the retained paid history.

The already-saved `success-initial-1` fixture demonstrates the issue without a
new probe. Its learner has unit prior and observation-noise variance. Accepted
receipt zero observes `X = 1/4`; receipt one intervenes on X, so it cannot update
X's mechanism. The scalar update equations in the read-only SCM source give
`gain = 1/(1+1) = 1/2`, terminal mean `1/8`, and variance `1/2`.
Nevertheless, the saved original final state has X mean `[0]`, covariance
`[[1]]`, and `eligible_rows = 1`. The saved output terminal is `complete` with
reason `policy_stop`. The test deliberately constructs placeholder means and
covariances at test lines 140–147, so its successful result does not establish
terminal learner consistency.

The concrete evidence is under
`results/ace_llm_intervention_episode_seal_preparation_20261107/test-attempt01/fixtures/test_success_schema_exact_receipt_pins_encoding_and_empty_initial/success-initial-1/`:

- `input/episode-disposition.json`:
  `e47d130818e972a3c6dcfaa5e699090f532337ddffd19c7fd82666d4e42ca08e`.
- `input/learner.json`:
  `855b35bf9f936d1950deb1e827adc43c7c6291e26612f2d172b18f6276c2f577`.
- `bundle/receipt-000.json`:
  `8dca94e76e24a2c46a143adfcb7ffbadaef677dc407e204cbb78298eb9d15e45`.
- `bundle/receipt-001.json`:
  `0b00ec9ed90eced2a8c78024505db753dcc4385c99ea52dcf2d84ed282fc59dd`.
- `bundle/terminal.json`:
  `bf3907218e8d6b24324956108f41aee25b718248cb2e71c06d6adafa0dca66f1`.
- `result.json`:
  `81343d89632827bdeada6da6084261ba068d476e8d41a4619516fcec9edbea17`.

Required correction: compare all terminal posterior means and covariance entries
with a fresh, bounded replay derived from the pinned learner configuration and
accepted receipts, with an explicit numerical comparison rule. Preserve the
documented atomic learner-update failure alternatives and explicit unavailable
failure state; do not turn those failures into successful episodes. Add focused
checks that change only a terminal mean or covariance, plus a positive fixture
whose terminal values follow the receipt equations. Alternatively, a separately
specified and independently verified integration gate would have to discharge
this requirement before acceptance; it is not established by this review.
The current guide's schema/count-only boundary and
`posterior_replay_performed=False` are accurate, so this finding is an unmet
requested validation requirement, not a claim of dishonest provenance flags.

Other requested checks, with line references to the reviewed sealer:

- **Pins and bounded parsing (65–86, 386–391):** the exact input/pin inventory,
  admitted filenames, raw-byte types, total size and every SHA-256 are checked
  before any JSON decoding. Duplicate fields, nonfinite constants and overflowing
  literals reject. Source identity/cap checks precede descriptor construction.
  The descriptor call validates supplied data; it does not generate or score a
  world. No fitting or private scoring path is called.
- **Journal and proposal state machine (142–245):** every record's semantic
  hash, previous link and monotonic timestamp are checked, then event-specific
  exact fields are enforced. Begin binds graph/menu/budget. Proposal/request IDs,
  fixed initial actions, decision indices, parse/budget rejection, stop,
  reservation action/cost/completion and accepted-response identity/clamps agree.
  Reservations establish charges before response handling. Missing dispositions,
  dangling reservations and continuation after a terminal reject. Policy-state
  hashes are checked for format, not reconstructed or independently authenticated.
- **Returned versus prepared evidence (253–293):** accepted responses require a
  matching callback wrapper and captured finite values equal to the executor's
  float-converted receipt. A prepared adapter outcome is not a callback return.
  Captured valid but unaccepted JSON and invalid JSON remain distinct, with
  canonical JSON bytes in their return evidence. No capture plus an explicitly
  unknown driver return retains the reservation as `unknown`. A known driver
  return without its wrapper, an unreserved wrapper or a non-JSON Python return
  rejects without fabricating accepted measurements or return bytes.
- **Shared initialization (130–138, 174–179, 280–281):** the full caller-supplied
  initial receipt-pin tuple must match the fixed schedule length, including for a
  partial arm. Every accepted initial receipt matches its tuple position using
  canonical JSON plus newline. No failed initialization is retried or filled in.
  Empty initialization is supported. Cross-arm establishment of those shared
  pins remains outside this single-episode API.
- **Source and outer disposition (327–373):** source counters, terminal history,
  charged attempts and exception text are compared with the sealed journal.
  Complete outer status requires both clean flags and the matching journal stop
  reason. Failure flags require the corresponding episode/adapter-close reason
  prefix. A complete source episode followed by adapter-close failure emits a
  failed terminal while retaining accepted history and costs. A source failure
  after stop is allowed without inventing a journal exception. Explicit
  `state_unavailable` remains a failure with journal-based accounting. The outer
  provenance receipt's hash is retained and format checked, not authenticated.
- **Cost-bundle interface and honest scope (394–425):** emitted source, plan,
  shared-initial, history, receipt/return and terminal members use canonical JSON
  plus LF. Original byte pins remain separately retained. The new plan binds
  normalized source hashes; history binds the plan; terminal binds all other
  bundle members and reconciles initial/acquisition, attempt, accepted and known
  return counters. The acquisition cap is checked. Scientific acceptance,
  scientific provenance, posterior replay and private scoring flags remain false.
  This was source/schema comparison, not a downstream reporter execution.

Inspected all **17 saved bundles and 49 saved `SealError` rejections**. Independently
checked raw input and output pins, canonical-LF encoding, journal hash chains,
plan/history/terminal bindings, evidence inventories, receipt extraction, return
classification, shared initial pins and charge totals using standard-library
JSON/hash/arithmetic inspection. There are seven complete and ten failed saved
terminals. Across these repeated artificial fixtures, the 29 attempts comprise
23 accepted, one invalid return, one unaccepted return and four unknown returns.
These are test artifacts, not independent scientific observations. All retained
rejection messages are consistent with their intended boundary; rejected cases
have no saved sealed result.

Independently verified all **877 manifest members / 9,272,129 bytes**, with no
missing or extra members. Including the manifest, the preparation tree contains
**878 files / 9,480,193 bytes**, agreeing with the requested inventory. The local
custody copy at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-episode-seal-preparation-20261107-01`
has the same complete inventory, sizes and hashes. All five current source files
listed in the test receipt match their `source01/` copies and closeout pins; all
five read-only reference copies match their declared pins and current source.
This establishes present byte equality, not the authenticity of the original
runtime or execution history.

The retained receipt/stderr records one successful six-method invocation, Python
3.9.6, cwd `/tmp`, exit zero, `Ran 6 tests in 0.097s`, child CPU `0.144719`
seconds and elapsed `0.16237649999999998` seconds. Its timestamps are
`2026-11-07T17:44:25.117158+00:00` through
`2026-11-07T17:44:25.279700+00:00`. Stdout is empty; the exact command and
fixture-output environment override are retained in the receipt. These are
author-run measurements; this review did not rerun the six methods.

Reviewed SHA-256 pins:

- Sealer (29,662 bytes):
  `b614744cecdd055720b8ec2fb35f245a75db569acacada38e0b9e1f2403f8478`.
- Tests (21,606 bytes):
  `462bc8709fc041730049a3db3910c350c344156777dbebd925decb536b8b765c`.
- Guide (6,813 bytes):
  `1ade118deef478533727444815c4e1ba17034d1d39a867cd8541c077dee332f2`.
- Contract/executor dependency (18,065 bytes):
  `ad89a8b72c8251c5a658bf45c27874353b161df5863b61929af40f5077254636`.
- World descriptor dependency (20,021 bytes):
  `f2d447d20c7fe64af9132e2d571e7731a91bed2c854b6a8a59b7467eac1225a8`.
- Preparation `review_request.md` (2,494 bytes):
  `3a00922e0c020e8d637c35bda8e30505cb3288b0e3795938b909624e83ca74b0`.
- Preparation `manifest.json` (208,064 bytes):
  `aa049adf744ca164902ac4777ce6f0cb94e30700e65f50d99a9ead6f3294e05f`.
- Preparation `closeout.json` (2,882 bytes):
  `2e9447080a707755f796e033b67074e2a8ed12a5071b64f5bac3da6a944a82cf`.
- Preparation `test-attempt01/receipt.json` (1,953 bytes):
  `2c77bbdeecf929b7e626c163f38d819e4e52c4b26a016980bbe9cda5b1ac80bb`.
- Preparation `test-attempt01/stderr.txt` (631 bytes):
  `1681441870f7bd09cb752b1a94480eb8e624353f9e411ad8c423d9b9b9579a34`.

Preparation-relative paths refer to
`results/ace_llm_intervention_episode_seal_preparation_20261107/`.

**Remaining caller boundary.** Independent pin custody, original raw evidence,
honest callback/runtime behavior, policy-process/public-state bindings, original
source qualification, outer service/cleanup receipt authentication and process
closure remain caller-owned. The sealer's hashes cannot establish those facts.
The paired finalizer must establish the prospective full roster and shared input
pins across arms and preserve failed or rejected arms at their common cap, with
raw actual-cost evidence retained. Sealing rejection never authorizes dropping
an arm or charging zero. Numerical terminal-state reconciliation is presently
missing as described in the required finding; a flag alone does not discharge it.
No component acceptance here extends to scientific evidence, source/split
qualification, private evaluator isolation or the separate parent integration.

This review used source reads and standard-library inspection only: no project
module import/execution, old suite, reporter, model, world generation, private
truth evaluation, remote operation, install or commit. A read-only search for a
guessed standalone executor file exited 2; the actual executor was then inspected
in the contract source. Existing saved evidence sufficed for the finding, so no
new counterexample probe or review folder was created. This review document is
the only file written; implementation, tests, guide, capsule and ledger were not
edited.
