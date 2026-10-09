# Fixed scalar mechanisms: distinct theory/design review

Date: 2026-10-09.

## Required findings

**None in this proposed design.** The sealed-table pointwise argument and conditional interpolation bound are sound. The document correctly treats this as a new predictor representation requiring future implementation, provenance checks and qualification, without certifying an earlier model or result.

Reviewed document: `docs/development/guidance/ace_foundation_fixed_mechanisms_2026-10-09.md`.

Exact SHA-256: `a8eb8df955b0bac8030a90929c08d941e4f9ed18dea711f3aae24814b92e33fb`.

Scope: the new mathematical construction, teacher-context limitations, approximation/boundary behavior, representation comparison and future provenance obligations. No source implementation, previous pilot/review, model, scientific world or test suite was examined or executed. Only this report was written. Existing empirical and resource figures were not re-audited.

## Checked conclusions

**Pointwise construction.** Conditional on the sealed grid, copied finite values, retained family/coefficients and evaluator semantics, the scalar evaluator is a fixed function of its scalar argument. Applying that same evaluator separately to each input preserves each result under batch splitting, permutation or extension. The conclusion needs neither pointwise nor batch-invariant behavior from the original teacher: its complete grid response is part of construction, finished before validation. A stochastic teacher is also compatible when its resulting table is fixed before independent validation. This establishes the student's function identity, not equivalence to future teacher predictions or universal numerical success.

**Reference identity and provenance.** Using the same copied retained expression for the reference and every outside-range branch supports exact local identity within the new evaluator. It does not imply bitwise equality with historical NumPy evaluation or composed identity after upstream changes. Copying primitive coefficients and table values and removing teacher references addresses accidental subsequent model mutation. The separation between immutable artifacts and authentication of their origin/phase is appropriate: a digest cannot prove that the supplier avoided private data. Binding evaluator code and runtime remains necessary for reproducible finite arithmetic.

**Grid and edge cases.** The nondegenerate grid has 129 knots and 128 cells. Exact endpoint/knot returns, strict ordering, rejection of collapsed or nonfinite grids and a separate singleton case define an unambiguous piecewise function. A singleton interval specifies the stored value only at that point and retention elsewhere; it has no positive-width interpolation-error claim and can be discontinuous at the singleton. The document appropriately treats nonfinite arithmetic as failure rather than silently replacing a failed candidate with retention. The statistical use still requires a well-defined predictor on the declared deployment law, including composed predicted-parent inputs; finite qualification points alone cannot establish this.

**Approximation remainder.** For a genuine scalar function with continuous second derivative satisfying `|f''| <= B` on a cell `[a,b]`, the exact two-knot remainder is `f''(xi)*(u-a)*(u-b)/2`. The product magnitude is at most `(b-a)^2/4`, giving `B*h^2/8`. For `f(u)=u^2`, the interpolant exceeds the true midpoint value by exactly `h^2/4`. These conclusions agree with [NIST DLMF 3.3.5](https://dlmf.nist.gov/3.3.E5) and its explicit [linear-interpolation constant 3.3.15](https://dlmf.nist.gov/3.3.E15). The bound applies to each cell's actual width; floating-point construction/evaluation error is additional. It is not justified merely because finite teacher values can be interpolated, and the document correctly excludes an unsupported curvature claim for a context-dependent teacher.

**Boundary and composition limitations.** Inside interpolation preserves stored endpoint values, while the one-sided outside limit is the retained expression. Unequal values create a jump, so neither a global Lipschitz claim nor a smooth interpolation remainder across that switch follows. Even small upstream approximation error can move a composed input across such a jump. These costs are explicitly disclosed rather than hidden by the pointwise proof; fixedness alone is not accuracy or risk control.

**Comparison design.** Tabulating all three updated providers on the same fit-derived grid avoids silently comparing a compiled PFN with an unmodified numerical learner. Retaining the exact copied reference is an intentional baseline definition. Equal representation does not imply equal approximation error across providers or isolate pretraining; results must concern the newly named table pipelines. The document's new identifiers, unchanged frozen originals and requirement for a new scientific specification/freeze adequately express that distinction. Teacher grid calls are inference cost, not additional environment responses, provided they query only models; no actual speedup, accuracy or response savings has been established.

**Literature scope.** The compression analogy is appropriately limited relative to [Hinton, Vinyals and Dean (2015)](https://arxiv.org/abs/1503.02531): this proposal does not optimize a neural student or inherit a distillation guarantee. The linked [TabPFN Nature article](https://www.nature.com/articles/s41586-024-08328-6) could not be retrieved through the browsing tool, and its PMC copy returned a browser challenge; that paper-specific interface statement was therefore not independently confirmed from full text here. No installed-model batch-dependence claim is needed for the construction, and the draft explicitly declines to make one.

## Optional clarification for future qualification

Specify the timing, teacher state and query context for the proposed off-grid teacher/table discrepancy diagnostic. Separate artificial qualification calls must not become live teacher calls after sealing a scientific predictor, and qualification activity must not silently mutate a teacher that will subsequently supply its grid. For a batch-dependent teacher, an off-grid diagnostic under a different query context measures both context change and representation discrepancy, not pure interpolation error. This is a nonblocking qualification-specification clarification; the current fixed-student proof and its stated approximation limits remain valid.

## Disposition

No required design correction. Implementation conformance, full-domain arithmetic checks, model qualification and any scientific execution remain future work and are outside this review.

Scoped paragraph recheck: the added separate-artificial-attempt requirement, freshly initialized fitted teacher with recorded query context, prohibition on scientific-teacher mutation or post-seal calls, and distinction between context change and interpolation error resolve the optional clarification while preserving one scientific fixed-grid query per head; no required findings. Current design SHA-256: `5fa2052bb48e7f2adaed258861bb202c148b22f599a97898fd2ecb378d5122c2`; only this clarification was rechecked, with no repeated full review or execution.
