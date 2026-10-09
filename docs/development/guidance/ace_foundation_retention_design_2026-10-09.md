# Retain or adapt: a prospective mechanism-selection design

Status: method prototype and proposed next screen; **no scientific freeze or execution**. Authored after the completed mismatch screen. This is an explicitly adaptive research progression using fresh future systems, not a preregistration predating earlier results. Accepted studies and both completed foundation pilots remain unchanged.

## Scientific motivation and literature

The completed mismatch screen found conditional target improvements and unchanged-mechanism harm together. Its detailed evidence and all four strata remain in [the results note](ace_foundation_mismatch_results_2026-10-09.md). In particular, the flexible pretrained estimator beat the deliberately inadequate grammar on missing families, while losing on coefficient changes. Those comparisons do not isolate pretraining from representation. A useful next method must decide **which mechanism to replace, and where to use its replacement**, as well as optimize the final target.

[Kirkpatrick et al. (2017)](https://arxiv.org/abs/1612.00796) preserve earlier-task competence by slowing changes to important neural parameters. That supplies a related continual-learning motivation for retention. Our construction below instead retains an entire old predictor and chooses among fixed candidate functions; it does not implement EWC or inherit its empirical results. We do not claim novelty from this comparison.

Kernel ridge regression supplies a flexible, non-pretrained comparator: nonlinear kernels yield nonlinear input-space predictions with squared-error fitting and regularization. We use the [installed scikit-learn 1.6.1 interface](https://scikit-learn.org/1.6/modules/generated/sklearn.kernel_ridge.KernelRidge.html). Its role is to test whether a generic nonlinear numerical learner can also recover missing families. A fixed RBF configuration is not the best possible non-pretrained learner; either outcome remains specific to these pipelines. Comparing PFN with RBF does not by itself isolate the causal contribution of pretraining: architecture, regularization and optimization differ too.

## Inputs and immutable candidates

The learner sees known X→M→Y, one pre-change history and one post-change history, and a declared phase boundary. It never sees the variant label, changed node, true equations or private probes. Keep the prior screen's measured-parent fitting, intervention-label eligibility and exact 24/8 stratified split. Fifteen natural-M fit labels and24Y fit labels construct each new candidate. Five natural-M calibration rows provide local-M and composed-Y losses; all8calibration rows provide measured-parent local-Y losses. The3internal-M calibration rows now help local-Y selection and remain excluded from root-only composition scoring. All responses are charged whether or not a particular selector uses their labels.

For each node keep four fixed candidates, in this tie order:

1. `retained`: the pre-change Grammar24 head, fitted once and reused across variants.
2. `grammar`: post-change Grammar24, with the existing three-family learner unchanged.
3. `rbf`: post-change RBF24, defined below.
4. `pfn`: post-change PFN24, with the existing pinned TabPFN-v2 configuration unchanged.

The RBF control standardizes its scalar parent using **only its eligible fit rows**, mean and population standard deviation (replace an exactly zero standard deviation with1). Center the target on its fit-row mean. Fit `KernelRidge(kernel='rbf', gamma=1, alpha=0.01)` and restore the target mean at prediction. No sine-specific feature, calibration hyperparameter search, private tuning or post-selection refit. These fixed settings are a modest generic control, not an optimized benchmark. The standalone RBF24 and PFN24 controls use exactly the same24-row fitting split and eligibility rules. Grammar32 remains the full-paid-data reference, so comparisons against it are pipeline comparisons with different supervision use.

## Explicit selector and ablations

For node i, let r_i be its retained head and C_i the closed interval between the minimum and maximum **post-change eligible fit parents**. These are X values for M and measured M values for Y. Calibration never expands the interval. Natural values are not clipped. Define a candidate's interval-gated head

`h_i^k(u) = f_i^k(u)` for u∈C_i, and `h_i^k(u) = r_i(u)` otherwise.

The retained candidate is always r_i. The interval includes both endpoints, and a degenerate interval contains just one point. An interval is a range check, not evidence of dense coverage, joint support, stationarity or reliable interpolation. For these scalar-parent systems it may bridge large holes. Gates can create discontinuities at the boundary; report their cost rather than smoothing them after scoring. In the proposed screen, the eligible M fit parents include root-menu endpoints−1 and1, so C_M=[−1,1]. All planned root actions and local-M probes lie inside it. Outside-M diagnostics are therefore expected to be empty; this screen directly tests outside-interval retention at Y, while Y gating may indirectly change which M candidate is selected. It cannot establish an empirical benefit of gating outside the M range.

The **local constraint** admits an updated head only when its measured-parent calibration MSE is strictly smaller than the retained head's MSE on the same node/rows. There is no fitted margin, significance threshold or risk certificate. Retained is always admitted. Compute local errors using the actual effective heads, including the interval gate when enabled.

For every admitted pair, propagate predicted M into Y and score composed Y MSE on the five natural-M calibration rows. Choose the minimum. Exact ties prefer fewer replaced heads, then the fixed candidate order for M and Y. Keep all candidate local scores, admissibility decisions, pair scores, intervals and counts of local calibration points inside intervals. No candidate is fitted again after selection. Candidates must be deterministic, pointwise and fixed during this process; source/runtime integration must verify those requirements rather than assuming the selector authenticates them.

Proposed nine reported methods:

- Grammar32, RBF24, PFN24, retained Grammar24: fixed complete predictors.
- `raw`: all four candidates per node, no local constraint or interval gate.
- `local`: local constraint only.
- `interval`: interval gate only.
- `combined`: both local constraint and interval gate.
- `combined_no_pfn`: same combined rule with exactly retained/grammar/rbf candidates.

The four selector ablations form a2×2design for the local constraint and interval gate. The no-PFN comparison tests the incremental value of including that pretrained candidate in this specific selection procedure. These are separate fixed descriptive comparisons, not a search for whichever method wins. Sixteen pair scores per unrestricted four-candidate selector, nine for unrestricted no-PFN; constraints can reduce those numbers. Reuse fitted candidates across selectors but record overlapping compute scopes, inference calls and charged responses separately.

## Elementary guarantees and their limits

**Pointwise retention identity.** At any fixed parent input u outside C_i, every interval-gated candidate equals r_i(u). Consequently its squared error equals the retained head's squared error against any fixed true local target at that input. The proof is the definition's outside branch. This requires the same parent input and deterministic pointwise heads; it is not a bound inferred from data.

**Finite calibration constraints.** If all predictions/losses are finite, retained is feasible. Each selected local head in a constrained mode has calibration MSE≤retained calibration MSE. The selected composed calibration loss is also≤that of the retained pair because that pair is feasible. These are deterministic inequalities on the supplied calibration observations, with no population or private-risk guarantee. Selection uses the same tiny calibration sample repeatedly; it is not an independent test. Noisy composed calibration also targets a different conditional mean than noise-disabled composition under nonlinear downstream mechanisms, as already disclosed in the mismatch protocol.

**Composition remains vulnerable.** Even exact downstream retention does not preserve the final output if the upstream head changes its input. For a concrete example, retained r_M(x)=x and r_Y(m)=m, a new M head f_M(x)=2x and unchanged Y give final predictions2x versus x wherever the M update is active. Y is identical at every fixed parent, yet the target changes. Which prediction is better depends on the true current mechanisms. Likewise a true changed mechanism outside C_i may need adaptation precisely where the gate retains a stale function. The proposed gate trades a specified form of extrapolation change for possible under-adaptation; it does not certify safety or identify which mechanism changed.

The selection module accepts callables and rows, so it cannot prove their origin, absence of private leakage, or lack of hidden mutation. Its new runner must authenticate model/source/runtime closure, keep evaluation behind a durable selection seal and preserve every failure. A candidate exception/nonfinite output invalidates that selector; it is never silently dropped or converted into a favorable retained fallback. Endpoint failures do not become complete-case aggregates.

## Proposed fresh screen and go/no-go evidence

Use six new base seeds93000–93005, the same four paired null/coefficient-M/missing-M/missing-Y variants, Gaussian noise and legal action domains as the completed mismatch protocol. Retain its exact RNG stream layout, history allocation, private endpoint definitions, training-only normalizers, zero/failure rules and6system pairing within each stratum. No scientific seed is generated until the new full implementation is reviewed and frozen. This is216method×variant×seed cells,648endpoint records,960shared training and18432private responses. Variants sharing a base system are not24independent worlds. Do not reuse92000–92005 for selection, fitting or rescoring.

Report all three private endpoints and all nine methods. For each endpoint/stratum publish all six paired ratios, arithmetic and geometric paired summaries versus Grammar32, with undefined full aggregates on any missing/zero/nonfinite pair. Report direct fixed comparisons PFN24/RBF24, combined/raw, combined/local, combined/interval, combined/combined_no_pfn and combined/RBF24 under identical rules. Compute each direct ratio within each world before aggregation; a quotient of arithmetic means of Grammar32-relative ratios is not the arithmetic mean of the desired paired ratios. Publish signed local MSE harm against retained on both changed and unchanged mechanisms, all choices/calibration losses, and per-head inside/outside private local-probe errors and counts using that world's frozen fit intervals. Empty strata are undefined, never zero. Also record predicted-parent Y-gate rates for composed forecasts; a marginal local-Y gate rate is not a composed rate. These support partitions are diagnostic, not a means of dropping unfavorable probes or choosing methods.

The key next-experiment decision is whether an observed target gain survives comparison with RBF and whether local constraints/range gating reduce unchanged-mechanism harm without unacceptable changed-mechanism loss. All four strata must be considered. This screen still cannot establish intervention savings. Only subsequent separately frozen estimator×acquisition comparisons with random/coverage/uncertainty baselines, identical menus and charged budgets can test that claim.

## Execution boundary

The current deliverable is the pure selector, fixed numerical control, artificial contract fixtures and reviewed design. A new runner/supervisor/reporter and source/model/runtime/response/resource freeze are still required. Proposed caps remain120seconds for one actual-runtime artificial qualification and900seconds for one scientific attempt, CPU only, within the22:00UTC cycle. These are **unallocated proposals**, not measured requirements or submission records. Existing6900second conservative reservations would become7920seconds if both were admitted within the28800second ceiling. Measure and check the artificial fixture before scientific admission; no retry or scaling just to consume budget. Preserve all preparations and failed attempts. No accepted artifact is refitted and no delivery replay is authorized by this design.
