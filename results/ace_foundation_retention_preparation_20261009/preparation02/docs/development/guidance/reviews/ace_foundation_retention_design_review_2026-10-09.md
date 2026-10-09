# Retention design: bounded theory and comparison review

Date: 2026-10-09. This reviews a candidate design, not a scientific freeze or implemented runner.

Reviewed source: `docs/development/guidance/ace_foundation_retention_design_2026-10-09.md` (69 lines).

Exact current-source SHA-256: `8c189cfe9c86f090347b80daad321070267ee3e34160e9bceb32b34d56af6c65`.

The source was untracked at review time. The initial version was SHA-256 `adf8187b6fd301cc110eea6b6cb68f69ed82281ef5b3c0ee5fdc0fd55d0b4222`. During review, the author replaced standalone RBF32 with RBF24 and added direct PFN24/RBF24 reporting. The current document was reread and this review updated to the current-source hash above. Supporting context was the completed mismatch results note, SHA-256 `bc21696671f25b215f87553a6e208b2836172795940659a38e82c925dd7fdb8e`, and the inherited protocol's history/split/endpoint definitions. Completed results were read as reported evidence; no private outcomes were reopened or recomputed, and no previous review was repeated.

## Required findings

None in the requested design scope. The elementary guarantees, comparison counts, and statistical/causal qualifications are consistent. The direct-comparison and no-pretraining-attribution language is adequate for the stated prospective development questions.

## Checks supporting that conclusion

**Guarantees.** Outside a node's fixed fit interval, the gated head is identically the retained function at the same parent input. Equality of squared local error follows for any fixed local target. This is a construction identity, not a statistical risk bound. In constrained modes, strict improvement admits updated heads while the retained head remains feasible; consequently each selected head's local calibration loss is no greater than retained. The retained pair also remains available to composed selection, establishing the composed calibration inequality. These statements require the declared finite, deterministic, fixed predictions. The text correctly limits them to supplied calibration observations and does not infer private or population improvement.

The composition warning is correct: changing upstream predictions changes downstream inputs even when the downstream function is retained exactly. Retaining stale functions outside the interval can also prevent needed adaptation. Neither mechanism-change identification nor a global no-harm guarantee follows. The noisy-composed-surrogate versus noise-disabled-target distinction is explicitly retained.

**Fit/calibration semantics.** The inherited split supplies 15 eligible natural-M fit labels and 24 Y fit labels. Local M and composed Y use the five calibration rows where M was generated naturally; local Y may use all eight rows with measured M, including the three internal-intervention rows. Those internal rows are not used as root-composed forecasts. No post-selection refit changes the selected functions. Fit-only RBF scaling/target centering and fit-only gate intervals avoid calibration/private leakage in the stated design. Standalone RBF24 and PFN24 now share exactly the same fitting rows and natural-mechanism eligibility. Grammar32 uses 20 eligible M labels and 32 Y labels, a disclosed pipeline difference.

**Comparisons and attribution.** The four selector ablations implement the stated local-constraint × interval-gate design. Direct combined/raw, combined/local, and combined/interval comparisons can describe the relevant paired pipeline changes; they do not establish an additive effect or absence of interaction. Combined versus combined_no_pfn tests adding the PFN candidate to this selector. It includes the consequences of the enlarged selection set and is not an isolated causal effect of pretraining. In particular, its larger feasible set can only improve or preserve the minimum calibration objective when all candidates are valid; that alone is not evidence of private benefit. The added direct PFN24/RBF24 comparison removes the previous unequal-fit-label issue for that standalone contrast. Its remaining architecture, regularization, optimization, and prior differences are explicitly acknowledged, so the no-pretraining-attribution language remains adequate. The design also discloses repeated use of tiny calibration samples and does not present the fixed RBF control as optimal.

**Counts and estimands.** Four fixed predictors plus five selectors give nine methods. Six base seeds × four variants × nine methods gives 216 cells; three endpoints give 648 records. Shared histories require 6 × (32 + 4 × 32) = 960 training responses. Private evaluation requires 6 × 4 × 768 = 18,432 responses, without multiplying by methods. The unrestricted candidate grids have 16 pairs, or nine without PFN; constraints may reduce them. Six systems per stratum, rather than 24 independent worlds, is the correct comparison unit. Common post-change training-label normalizers, explicit undefined aggregates, signed local harm, and separate local-probe/composed gate rates prevent the stated denominator and support confusions. The reservation arithmetic 6,900 + 120 + 900 = 7,920 seconds is consistent and remains explicitly prospective.

**Evidence and references.** The reported mismatch evidence supports pursuing conditional target benefit together with unchanged-head harm, while retaining coefficient-change and null counterevidence. It does not validate the proposed retention selector. The design makes that distinction and labels the progression as adaptive development on fresh systems. Its limited EWC analogy matches the original description of slowing updates to important weights, without claiming to implement EWC or inherit its results. [Kirkpatrick et al.](https://arxiv.org/abs/1612.00796). The stated kernel-ridge role and explicit alpha/gamma/kernel interface agree with the versioned documentation. [Scikit-learn 1.6.1 KernelRidge](https://scikit-learn.org/1.6/modules/generated/sklearn.kernel_ridge.KernelRidge.html). This review did not inspect the installed runtime.

## Optional improvements — not required corrections

1. **State that the M interval gate is inactive over this screen's evaluated root domain.** The fitting indices include root-intervention rows 8–16, whose menu contains both −1 and +1; all other fit X values lie within that range. Thus C_M is exactly [−1,1]. All planned root actions and local-M probes lie inside it. The gate's outside-interval behavior is therefore directly exercised at Y, not M, although Y gating can indirectly alter which M candidate wins joint selection. An empty outside-M diagnostic is expected, not a defect. The current empty-stratum rule already handles it; making this structural fact explicit would sharpen later interpretation without changing the experiment.

2. **Optionally list combined/RBF24 if a claim about the complete selector beating standalone RBF will drive the next-stage decision.** The new direct PFN24/RBF24 contrast is adequate for the stated standalone learner comparison. Existing per-world errors also make combined/RBF24 available without another arm or new fits. If that additional contrast is desired, compute each world's combined NMSE / RBF24 NMSE before aggregation; dividing arithmetic mean Grammar32-relative ratios is not the arithmetic mean of the desired paired ratios. Apply the same zero/failure rules. This is optional reporting clarification, not a defect in the current direct contrasts or no-PFN interpretation.

## Scope

Static design and analytical checks only. No implementation/design edits, tests, model loads, scientific-seed generation, experiments, or empirical recomputation. Only this review was written. The future runner must still establish the stated source/runtime provenance, pointwise fixed-candidate behavior, selection/evaluation separation, accounting, and failure handling; this review does not authenticate that unwritten integration or authorize execution.

## Scoped clarification delta — 2026-10-09

New design SHA-256: `ac40cba6b69fbca9f749db4c8239103db8919570a80c51b49b8a16d32c3892c0`.

Only the additions at design lines 32 and 63 were checked. Both optional suggestions are addressed: the text explicitly identifies C_M=[−1,1], expected empty outside-M diagnostics, and the absence of an empirical outside-M gating-benefit claim; it also lists combined/RBF24 and requires within-world ratios before aggregation, rather than a quotient of arithmetic means. No required finding arises from these two changes.

This is a delta disposition, not a repeated full review of the new document. Prior source hashes and conclusions retain their stated scopes. Only this paragraph block was appended; no code/design edits, tests, models, or experiments were performed.
