# Fixed scalar mechanisms from pre-validation prediction tables

October 9, 2026. Proposed refinement of the unexecuted [selection-reliability design](ace_foundation_selection_reliability_2026-10-09.md). **Pure adapter preparation only: no pretrained inference, model fitting, scientific responses or new allocation.** This does not change an accepted study or retroactively qualify any earlier model.

## Why this is useful

The reliability theorem conditions on fitted, fixed pointwise functions. Calling a generic model repeatedly on different validation batches does not establish that premise; a one-input wrapper also fails if the model changes state between calls. We can instead finish all teacher inference before validation, copy its finite outputs into a small numerical object, and evaluate that object alone thereafter.

[TabPFN's primary paper](https://www.nature.com/articles/s41586-024-08328-6) describes inference conditioned on labelled training data and unlabelled test inputs. This motivates explicitly controlling the inference interface, but does not establish batch dependence in our installed configuration. We have not found a demonstrated batch-dependence failure in the previous pilots. Finite compatibility checks are evidence at their tested inputs, not a proof of all future behavior.

The broad idea resembles teacher-to-student compression, as in [Hinton, Vinyals and Dean (2015)](https://arxiv.org/abs/1503.02531): preserve predictions in a smaller representation. Here there is no neural student training, distillation loss optimization or inherited accuracy guarantee. The student is a piecewise-linear scalar table. This is a standard numerical representation, not a claim of a novel distillation algorithm.

## Exact new method

For each new head, construct C=[a,b] from only its eligible fit parents, before accessing validation or private inputs. For a<b use exactly129 equally spaced points `a+(b−a)k/128`, k=0,…,128, with endpoints set exactly to a,b. Reject nonfinite spans, collapsed floating-point grid points or invalid fit parents. For a=b use a single point; its inside interval is that point only. Grid count is fixed; do not adapt it to validation or private errors.

Call the teacher once on that entire grid. Require exactly one finite scalar output per grid input, copy all outputs into tuples, then retain no callable/model reference. Teacher preprocessing, weights, fit data, runtime, random state, grid and returned values must be source/receipt-bound by a future runner before validation. If the teacher is stochastic, its randomness and resulting fixed table are part of pre-validation construction. No later teacher calls, query-dependent updates, mutable caches or validation-aware grid refinement are allowed.

For u∈C, return the linear interpolation of neighboring stored values, preserving knot values exactly. For u outside C, return a fixed retained grammar head specified only by family and copied coefficients. Supported retained families are the existing linear, quadratic and tanh definitions. Reference and every updated head use the same copied retained object for the corresponding node; identity branches therefore share the same pointwise arithmetic. Retained is used globally for the reference predictor. Returned functions use only scalar arithmetic/binary search and no random draw, model, batch statistic, file or network access.

Apply this SAME table representation to post-change numerical grammar, RBF and PFN heads. Do not quietly leave one candidate exact while tabulating another. Use identifiers `table_grammar24`, `table_rbf24`, `table_pfn24` and `retained_parameters24` in eventual reports; selectors contain these heads. This is a new pipeline, not a reproduction of PFN24/RBF24/Grammar24. Scalar evaluation of copied retained coefficients is mathematically the same family expression but not asserted bit-identical to historical NumPy evaluation. Original source/checkpoints remain preserved separately.

The pure adapter can bind the exact table bytes and fit-parent digest, but cannot authenticate who supplied them, when they were constructed, or whether a teacher saw private data. Those remain source/phase/receipt checks for the future runner. Frozen Python tuples/dataclasses deter accidental mutation; they are not a security boundary against hostile code or object.__setattr__.

## Deterministic pointwise property

Condition on the completed tables, retained coefficient tuples and authenticated evaluator code. For any scalar u, every branch depends only on u and those fixed values. Thus splitting, permuting or extending a batch leaves each scalar result unchanged when the same finite arithmetic is used. The teacher may have used the full grid as context: the guarantee concerns the sealed numerical student, not how the teacher would predict a new input. With independent IID validation after this construction, batch-context dependence of the original teacher no longer enters the validation losses.

Nonfinite values or arithmetic overflow are explicit failures. The implementation does not clip predictions to hide them or silently choose retention after a failed inside prediction. A statistical theorem requiring a predictor on the full deployment law additionally needs finite well-defined predictions there; artificial finite checks alone do not prove all possible noisy-support behavior. Local/root parent domains in the proposed screen are bounded, but composed predicted parents also need an authenticated domain/arithmetic check. No completed empirical result becomes certified by this property.

## Approximation cost and what remains unknown

[NIST DLMF §3.3, equation3.3.5](https://dlmf.nist.gov/3.3.E5) gives the interpolation remainder. If a genuine scalar teacher f has a continuous second derivative bounded in absolute value by B on a cell of width h, its exact two-knot interpolant has error≤Bh²/8: the remainder is f''(ξ)(u−a)(u−b)/2 and the product magnitude is at most h²/4. This is a conditional approximation result, not a measured bound for PFN or a teacher whose grid outputs lack a common pointwise interpretation. Floating-point error is additional. For f(u)=u² the midpoint error is exactly h²/4, an elementary oracle for the adapter.

A small parent-grid error need not give a small composed prediction error. In particular, the outside-retention switch can make a jump at either fit boundary, so a global Lipschitz claim for the effective head would be unjustified. Do not use a smooth interpolation bound across that switch, infer certified curvature from finite differences, or change grid density after scientific results. Report teacher/table discrepancy on separately declared fit-derived artificial qualification points, interpolation interval widths and boundary jumps without calling them risk guarantees.

Potential benefits are explicit function identity, cheap repeated scalar evaluation and easy byte-level reproduction. Costs are altered predictions, approximation error, grid inference overhead and limited applicability to the current one-parent chain. A129-point tensor grid is not a viable prescription for arbitrary high-dimensional parents. No speedup, accuracy gain or intervention savings is established in this preparation.

## Integration obligations before a new scientific attempt

The previous proposed312-cell validation-budget screen is still UNFROZEN and UNEXECUTED. This proposal changes its candidate representation and must be adopted in a new exact implementation/specification before a fresh freeze; it does not silently amend the earlier design digest. Scientific seeds94000–94005 remain ungenerated. All methods, response charges and noise/structural endpoint distinctions would remain reported, with table-prefixed identities and all original teacher metadata preserved.

A future runner must (1) isolate fit data from unused rows, later prefixes and truth; (2) authenticate original teacher source/runtime/weights; (3) build/seal all candidate tables and retained parameters before validation; (4) test the compiled artifact without live teachers; (5) bind exact stream namespaces, prefix isolation, model-failure propagation and an independent saved-prediction reporter; and (6) measure an actual-runtime artificial qualification under a separately reserved CPU cap before scientific admission. There is no remaining-time justification for skipping these steps before22:00UTC. No extra allocation is authorized merely by this implementation.

Current prototype tests use hand-written arithmetic teachers only. No scientific seed, accepted model, PFN checkpoint or external response is accessed. Existing conservative reservations7920/28800CPU seconds, measured model-attempt90.044439childCPU+.108012supervisorCPU seconds, GPU0 remain unchanged; preparation/test/review overhead is unmetered and excluded from those model-attempt totals.
