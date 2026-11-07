# Independent offline cost-curves review

November 7, 2026. **0 required defects.** The reviewed implementation is suitable
as the bounded, private, prospective replay/reporting primitive described in its
guide. This disposition does not qualify scientific inputs, original paid
execution, a paired study roster, or future integration.

## Scope and method

Read the new reporter, its seven artificial test methods, its guide,
`results/ace_llm_intervention_cost_curves_preparation_20261107/review_request.md`,
and saved receipts, reports, rejections and source manifests. Consulted only the
existing contract, world, task and SCM interfaces needed to trace these calls;
did not review earlier review documents. Review operations were local text reads
and standard-library JSON/hash/size inspection of retained bytes. No reporter or
project module was imported or executed, no tests were run, and no model,
scientific generation, remote operation, submission or commit occurred. This
review document is the only file written.

## Findings by requested check

- **Independent bindings:** reporter lines 82–119 authenticate every raw input
  against the separately supplied pin dictionary before constructing world,
  task or learner objects. The plan binds world/task/learner/shared-initial bytes;
  scope/world identity and learner graph must agree. Learner noise/prior come
  from the independently pinned learner configuration, not private mechanisms.
  The world semantic digest is calculated after raw-byte authentication, so its
  local construction is not substituted for that authentication. Lines 130–142
  bind the ordered shared-initial receipt pins, history to plan, and terminal to
  every other input. Pins supplied alongside fabricated fixtures demonstrate
  consistency checks, not independent execution provenance.
- **Paid accepted replay only:** lines 149–198 require consecutive reservation
  IDs, the fixed initial schedule, legal action identity, exact cost, exact
  measured-variable membership, finite values and consistent clamps. Only the
  `accepted` branch appends replay receipts. All evidence files must be referenced,
  terminal counters reconcile, and continuation after an unaccepted outcome is
  rejected. Lines 247–254 update a newly constructed learner only from those
  accepted receipts. No callback, generator, policy, live learner snapshot or
  private truth is used to synthesize an update.
- **Fixed error endpoint and crossing:** lines 104–114 enforce fixed target
  queries, equal weights, positive public scale and agreement between task
  tolerance and plan threshold. Lines 234–254 compare composed learner means
  with independent noise-disabled structural means. The curve includes complete
  shared initialization at acquisition cost zero, including an empty initial
  schedule, then each accepted acquisition at its cumulative integer cost.
  Lines 258–274 select the first inclusive `error <= threshold` crossing; later
  regression does not erase it. Attainment exactly at the cap remains separately
  identifiable from nonattainment. This is replay of the existing learner's
  eligible local updates, not a claim of exact latent partial-observation inference.
- **Failures, unknown returns and actual spend:** lines 180–214 distinguish
  known invalid returned bytes, valid but unaccepted returned bytes, and unknown
  return status. All retain their reservation cost, none update the learner, and
  all force a failed terminal. A failed terminal may also represent failure
  between accepted responses. Lines 256–275 preserve a numerically evaluable
  prefix and the full verified charge accounting; operational/evaluation failures
  and nonattainment receive the common acquisition cap. Actual initial,
  acquisition, outcome and known-return costs remain separate. Partial failed
  initialization has no comparable initial curve point. A failure after crossing
  retains that prefix attainment but receives the cap.
- **No filtering or private feedback:** malformed or contradictory bundles raise
  `CurveInputError`; they are not converted into successful or zero-cost reports.
  Lines 218–225 and the guide explicitly require the caller to keep unresolved
  capped roster rows and original charge evidence. The reporter has no policy
  callback or stopping path. Its output is explicitly private, disallows
  complete-case filtering, and leaves source/split/execution qualification to the
  caller (lines 276–280). Enforcement across policies and processes remains a
  parent integration obligation, not something these output flags establish.

## Independent hand checks and retained evidence

For the main fixture, true `Y = 2 + 2X`, unit learner noise/prior, and initial
receipt `(X=0, Y=2)` give posterior mean `(intercept, slope) = (1, 0)`.
After `k` acquisition receipts `(X=1, Y=4)`, the means are
`((6k+2)/(3k+2), 6k/(3k+2))`. Equal queries at `X=0,1`, normalized by scale two,
therefore have error `5/(3k+2)^2`. This independently gives
`[5/4, 1/5, 5/64, 5/121, 5/196]` at acquisition costs `[0,1,3,4,6]`, matching the
saved report within floating-point rounding and crossing 0.04 at cost six.

For the constant fixture, the error after `k` acquisitions is `1/(k+2)^2`;
the cost-three crossing is retained when a subsequent `Y=-10` receipt raises
error to `49/36`. Its unknown-return failure variant instead receives cap five
while retaining six total spent units. The retained initial-equality fixture
crosses at zero; the empty-initial prior error is `5/2`. The three unaccepted
return classes each retain three total units and only the initial curve point.
The numerical-overflow fixture retains its initial point, both paid units, null
final error and cap six. These were calculations and saved-record inspection,
not additional executions.

The saved stderr records seven methods passing once; the receipt records exit
zero, 0.191142 child CPU seconds and 0.236475375 wall seconds. That measurement
covers the original test subprocess, not implementation, custody or this review.
Inspected all 12 saved reports (five operational failures and one evaluation
failure) and all 21 saved rejection messages; the rejections reached their
intended checks. Verified all 414 receipt-inventory members and all 420 outer
manifest members against their retained sizes and hashes. The local preparation
tree contains exactly 421 files / 525,796 bytes, with no missing or extra members.
The three current reporter/test/guide files exactly match the tested saved copies.

## Reviewed pins

SHA-256 values:

- `scripts/research/llm_intervention_cost_curves.py`:
  `f9420b77823f9becc7a3885cd258283ceda406655b4ba077ecbefb19fd0896e8`
- `scripts/research/test_llm_intervention_cost_curves.py`:
  `aae8ac71dfa9ef1ac02f887690bcac8c1fcbe7fcc0099b6c45e1d4e0ca96d97a`
- `docs/development/guidance/ace_llm_intervention_cost_curves_2026-11-07.md`:
  `18da5e5006397693b9cf9a151061444029e7008a567b68d06760c1df5aa892ff`
- Preparation `attempt01/receipt.json`:
  `2c9472058a23cc5cb269ca26f633885675e931b502da4e13cf23f410933e622f`
- Preparation `attempt01/stderr.log`:
  `a911f65b108f907efd825150219030d4d8d1fc91eccf8ff1ea0242e7bfad2757`
- Preparation `closeout.json`:
  `371e65d79146e6b154c4d760f9935599cd6a823ceb94e9c6bac8d3b6f423b561`
- Preparation `manifest.json`:
  `23bb6cfb8e7829b0173a7ffdeb1addd7c3538b5331ed715bd1b569e307727082`

Parent integration still owns trusted pin establishment and original-journal
reconciliation, the common predeclared world/task/learner/initial schedule/cap
and complete roster across policies, retention of rejected inputs, evaluator
isolation, shared-initial bill deduplication and scientific provenance. No
additional primitive changes are required by this review.
