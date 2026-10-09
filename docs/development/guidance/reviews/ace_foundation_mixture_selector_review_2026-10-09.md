# Foundation mixture selector: bounded code review

Date: 2026-10-09. Static review of selection semantics and the six artificial test definitions against the prospective mismatch protocol. No modules/models were imported, no tests or worlds were executed, and no experiments were performed. The six passing tests are user-reported; this review inspected their source. Only this review file was written.

## Exact reviewed inputs

- `scripts/research/foundation_mixture_selection.py` (101 lines): SHA-256 `13c107f88b965b7c6cb05997ba2ff53a8869a4a8ae945f51c09f0edc14636a87`.
- `scripts/research/test_foundation_mixture_selection.py` (47 lines): SHA-256 `a5135db132f353efe7388db4d7f3659432e85c7d7e1a82e74722b85c0340bb7c`.
- `docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md` (57 lines): SHA-256 `c7034cc5839a70196e02b2a1ffd876224192fa557a7bb02514b358a6ee72a9bf`.

All three were untracked when inspected. These hashes identify the reviewed contents; no committed-source or execution qualification is implied.

## Required findings

None in the requested selector scope.

## Basis for that conclusion

- **Split and eligibility:** selector lines 10–12 and 55–68 implement the protocol's exact 24/8 index split. The fitting split contains 15 natural-M rows and 24 natural-Y rows. Calibration requires the declared two observational, three X-clamped, three M-clamped layout. Only the first five natural-M rows supply root X and observed Y to the mixture score. The three M-clamped rows remain outside that score.
- **Measured versus predicted parents:** both selectors use observed root X and observed noisy Y for calibration. Neither substitutes measured M into a composed forecast. `mechanism_choice` feeds the mixed predicted M into both Y heads. Measured-parent fitting and local probe evaluation remain caller responsibilities, as appropriate for this selection-only module.
- **Terminal mixture:** lines 71–76 and 92–95 compose Grammar M→Y and PFN M→Y separately, then blend their terminal outputs. This matches the protocol and does not compose diagnostic local mixtures.
- **Mechanism mixture:** lines 79–89 and 98–101 blend the M predictions, evaluate both Y mechanisms on that shared blended parent, and blend their outputs. Selection and forecast implement the same operation. All 25 pairs are scored on composed Y MSE.
- **Exact ties:** the five weights are ordered ascending. Terminal candidates follow that order; mechanism candidates use M weight outermost and Y weight innermost. Python's first-minimum selection therefore implements the specified exact-tie rule without an undeclared tolerance or a different secondary criterion.
- **Recorded selection evidence:** each `Choice` retains the selected weights, selected calibration MSE, and the complete ordered candidate-score list. Prediction lengths and finite values are checked. The selectors perform prediction only; they contain no fit/refit, model load, environment call, or private-evaluation access.

The six fixtures exercise split counts/disjointness, complementary terminal errors, predicted-parent composition and invariance to measured M, exact ties, the distinction between terminal and local composition, and invalid inputs. Their artificial M-clamped rows also make accidental inclusion in the scored calibration set consequential. No additional broad test pass is requested by this review.

## Integration boundary

This module cannot prove calibration provenance or that supplied expert callables were fitted only on the 24-row split and remain unchanged afterward; its header explicitly acknowledges that boundary. The later adapter/runner review must establish those facts, measured-parent training eligibility, correct local diagnostics, and inference accounting under the existing protocol gate. Those are not newly discovered selector defects, and this review does not qualify the not-yet-frozen scientific pipeline.

## Implementer follow-up and current verification

After the reviewed bytes, `mse` was hardened to reject empty/nonfinite input vectors before computing differences and to reject overflowed squared residuals as ValueError. The corresponding invalid-vector fixture now covers empty vectors and overflow. This validation-only change was not independently re-reviewed; the original reviewed hashes above remain historical rather than being relabeled.

A final current-source run of all six artificial fixture methods passed from `/tmp` using the system Python. Current selector SHA294dd5c00f5fdd4b532aea837e5d5e09f3ee0e8b226700ca1049045a227e75df; test SHA2c9fd8706e2e22206eec53d21016fe1b969cf1103ecbcf60c3f2e4a9ff667efa. Exclusive receipt/logs: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-mixture-preparation-20261009-01`. Single final child0.02872CPU seconds/0.030765875elapsed/14811136peakRSS bytes excludes earlier fixtures, preparation and review. This establishes selector behavior on artificial inputs only; it does not qualify model adapters or scientific execution.
