# Retention selector and flexible control: implementation review

Date: October 9, 2026. **Disposition: 0 required issues within the bounded prototype review.** This uses the current design's standalone **RBF24**, matched to PFN24's fitting split, with nine reported arms.

## Scope and reviewed bytes

Read the two implementations, current design and eight artificial test methods. No implementation, test, scientific generator or model was executed or imported during this review. Only this report was written. The author's reported passing tests are not an independently reproduced execution result.

SHA-256:

- [foundation_retention_selection.py](/Users/pat/code/ACE/scripts/research/foundation_retention_selection.py): `6dc543f22766a43b749712ed066d347023fcf52446222f351e4dd95c895a3705`.
- [foundation_flexible_control.py](/Users/pat/code/ACE/scripts/research/foundation_flexible_control.py): `75764cd932768a79c2f95617aa29f87663a19875a33f3e5afd7ace438f45dd49`.
- [Current retention design](/Users/pat/code/ACE/docs/development/guidance/ace_foundation_retention_design_2026-10-09.md): `8c189cfe9c86f090347b80daad321070267ee3e34160e9bceb32b34d56af6c65`.
- [test_foundation_retention.py](/Users/pat/code/ACE/scripts/research/test_foundation_retention.py): `e4c05ee6da9ba54d9aa5290bf81087e3013c80254105aec703ef2ba2e976ffe3`.

## Selector disposition

- **Intervention semantics:** `training_layout()` requires the 24-row fit layout and eight-row calibration layout. M intervals use only the 15 natural-M fit parents; Y intervals use measured M from all 24 fit rows. Local-M and root-composed calibration use the five natural-M rows. Local-Y uses all eight measured-parent rows, including the three M interventions. Calibration does not expand either interval, and natural values are not clipped.
- **Predicted-parent propagation:** every admitted pair scores Y at its candidate M prediction. It never substitutes the observed calibration M into composed scoring. `forecast()` applies the same selected effective heads and passes predicted M into the downstream gate, so a Y gate uses the actual composed parent rather than a marginal local-probe coordinate.
- **Gates:** boundaries are inclusive; a degenerate interval includes its single value. Updated heads receive only inside points and retained heads only outside points; results return in original input order. Retained candidates remain ungated. With the declared deterministic pointwise-head contract, this implements exact outside retention identity without evaluating an unused updated branch.
- **Local constraints and ties:** effective, gated heads supply local scores in constrained interval mode. Updated candidates require strictly smaller local MSE; retained remains feasible. All admitted pairs are scored. Ranking is composed MSE, number of replaced heads, then fixed M/Y candidate order. Thus ties prefer retention as specified, and the selected composed calibration loss cannot exceed the feasible retained-pair loss. These are calibration inequalities, not private-risk guarantees.
- **Ablations:** raw/local/interval/combined implement the declared two switches. `mode='combined', include_pfn=False` implements the three-candidate combined-no-PFN rule. Complete callable dictionaries are required; PFN cannot silently remain in the no-PFN set. The returned selection records all local errors, admitted candidates, pair errors, intervals, chosen pair and local calibration inside counts.
- **Failures and finite values:** finite nonempty vectors and prediction lengths are checked. Overflow/nonfinite squared losses invalidate selection. Exceptions propagate; no candidate is silently removed and no exception is converted into a favorable retained selection. An intentionally unused gate branch need not be evaluated. Private partition errors, endpoint failure ledgers and aggregate treatment belong to the future evaluator/reporter, not this selector.

## RBF24 control disposition

`RBFControl.fit()` takes paired scalar arrays and uses only those supplied fit rows. It computes parent mean/population standard deviation, replaces an exactly zero standard deviation by 1, centers targets on their fit mean, and fits `KernelRidge(kernel='rbf', gamma=1, alpha=0.01)`. Prediction reuses those stored statistics and restores the target mean. It performs no calibration search or prediction-time normalization update. A successfully fitted control rejects refitting; malformed/nonfinite fit inputs, unfitted calls and nonfinite predictions raise.

The current RBF24/PFN24 match is achieved by supplying the same eligible 24-split data—15 M labels and 24 Y labels—to their respective adapters. This generic numerical control does not receive intervention tags and cannot itself authenticate that eligibility or provenance. Its scalar one-dimensional input contract is explicit; the future adapter must pass eligible scalar arrays. Fixed parent standardization/target centering here is distinct from the future evaluator's shared training-variance NMSE normalizer. Neither normalization uses private outcomes.

## Relevant artificial test adequacy

The eight methods cover the principal prototype contracts:

- Gate partition/order, inclusive endpoints, outside identity and skipping an unused updated branch.
- Strict local admission, retained feasibility, predicted-parent composition and a forecast whose updated M moves outside the Y gate.
- M-clamped calibration exclusion from M/composed losses, while those rows contribute to local-Y loss.
- Retention/candidate-order ties, four mode settings and the exact three-candidate set.
- Fit-only eligible intervals and calibration points outside both intervals.
- Length errors, nonfinite/overflow outputs, candidate exceptions and malformed layouts/modes/intervals.
- A two-point analytic RBF oracle and invariance of stored normalization under new prediction inputs.
- Constant-parent handling, invalid fits, unfitted calls and refit rejection.

The RBF oracle is substantive: standardized parents −1 and +1 give off-diagonal kernel value `exp(−4)`. For centered targets −1 and +1, fitted endpoint predictions are ±`(1−exp(−4))/(1.01−exp(−4))`, matching the test's independently specified formula. The affine-parent/target-translation check also agrees with the declared normalization rule. The test's `pfn` candidates are fabricated linear functions, not pretrained-model calls or scientific worlds.

This is adequate evidence for the inspected pure selector/control contracts; no additional required fixture was identified in this scope. It does not authenticate an eventual PFN head's deterministic pointwise behavior, response provenance, private-evaluation boundary or runtime resources. The design explicitly assigns those checks to the future runner and qualification stage. This review is **prototype qualification only**, not a scientific source/runtime freeze or launch approval.
