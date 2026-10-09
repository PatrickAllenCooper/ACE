# Reliability selector: bounded implementation review

Date: 2026-10-09. **Required findings: 0.** This disposition covers the pure fixed-family selector and its artificial fixture source, against the supplied prospective design. It does not qualify a scientific runner or independently approve the separately reviewed design.

## Exact inputs and scope

- `scripts/research/foundation_reliability_selection.py` SHA256: `796eb9923419ae701e58d16616981fb2b08f58c3f7e93ab1f190837864d03bbc`.
- `scripts/research/test_foundation_reliability_selection.py` SHA256: `dde8c99fe2d496ab18cfe3c08eed72875e0eec5c2cf0254062b327d90b421f05`.
- `docs/development/guidance/ace_foundation_selection_reliability_2026-10-09.md` SHA256: `522ed433f962607536c5cfbf370f5a817c897e366cdcc7e4ee173b9373f04733`.

Source inspection only; the nine previously passed artificial tests were read, not rerun. No selector execution, model imports, responses, scientific seeds, refits, prior tests, original-source edits or commits. A stdlib calculation checked the three declared bound widths. No additional counterexample was needed.

## Implementation disposition

**Range and arithmetic — consistent (selector lines 15–29, 49–62, 78–85).** Scalars reject booleans, nonfinite values and unsupported types. Squared scales must be positive. The loss routine compares the absolute residual with the square-root scale before squaring, avoiding overflow in a large squared residual; overflow of subtraction between finite extreme floats correctly saturates at one. Below saturation, the normalized residual cannot exceed one. Loss input sequences must have exactly the registered prefix length and values in [0,1]; therefore paired differences are bounded in [-1,1]. `math.fsum` avoids ordinary accumulation error, and all accepted losses and bounded sums remain finite at these sizes. The design's fit-only variance floor and clipping/floor counts are caller responsibilities, not silently estimated by this module.

**Multiplicity and look boundaries — consistent (lines 9–11, 41–44, 72, 93).** The module pins 15 nonreference pairs × three objectives × three fixed looks = 135 and delta .05. Its one-sided difference radius is `sqrt(2 log(135/.05)/n)`, appropriate to the declared difference range. The widths are approximately 1.405437 at 8, 0.702718 at 32 and 0.351359 at 128. The eight-observation simultaneous rule cannot accept a nonidentity composed candidate even at mean difference -1. Empirical mode applies zero margin and is explicitly labeled by `rule`; it is not the simultaneous guarantee. The no-PFN mode conservatively retains 135 rather than narrowing the bound. Per-world delta is labeled as such, with no across-world coverage claim.

**Retained identity versus sample ties — consistent (lines 63–71, 81–87).** Local head losses must be invariant to the other head's identifier. Exact-zero shortcuts apply only to the reference pair or to the retained identifier for that local objective. Any contradictory retained losses fail. An updated head with coincident sample losses still receives the simultaneous margin; equal loss samples do not create predictor identity. Nonreference composed objectives always retain their margin. The module explicitly delegates authentication of actual predictor identity, paired inputs and fixed candidate construction to the future runner.

**Failure-family closure — consistent within this interface (lines 45–62).** With PFN, all 16 declared pairs and all three objectives are required; without PFN, exactly the nine retained/grammar/RBF pairs are required. Missing, extra, malformed or nonfinite candidate data invalidate the call before selection, including otherwise uncompetitive candidates. This prevents dropping a failed required arm to make selection pass. A PFN-only failure may leave the independently declared nine-pair family usable; retained/grammar/RBF failures cannot be removed from that family. The API accepts losses rather than expert-status records, so the future runner must propagate failures instead of fabricating replacement loss arrays; the source documents that boundary.

**Admission, choice and fallback — consistent (lines 84–93).** Both local bounds must be nonpositive and composed improvement strictly negative. Among admissible pairs, ranking is composed empirical bounded loss, then fewer replaced heads, then the declared retained/grammar/RBF/PFN order. A tied reference is not replaced: it is excluded as an update and, absent a strictly improving admissible pair, returned with an explicit fallback reason. No-PFN ordering is the same restricted order.

## Existing fixture adequacy

The nine source fixtures directly cover loss saturation and invalid scales, eight-row impossibility versus empirical selection, a large-margin update with retained-local identity, sample ties without identity, local-harm rejection, retention/name tie ranking, missing and malformed candidate data, inconsistent local-head losses, and unchanged no-PFN multiplicity. These are artificial loss arrays with no model or environment imports. Their reported previous pass is not represented as a rerun by this review.

No required implementation or fixture correction was found. Fixed fits/scales, IID stream provenance, identical effective retained heads, nested-prefix isolation, failure journals and cross-call family consistency remain duties of a separately reviewed future runner, as already specified by the design and module contract.
