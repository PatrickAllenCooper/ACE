# Selection reliability: mathematical and prospective-design review

Date: 2026-10-09.

## Required findings

**None in the reviewed version.** The paired Hoeffding bound, multiplicity, identity shortcut, structural-error decomposition and proposed response/output counts are correct under the stated assumptions. The design appropriately remains unfrozen and unexecuted.

Reviewed document: `docs/development/guidance/ace_foundation_selection_reliability_2026-10-09.md`.

Exact SHA-256: `522ed433f962607536c5cfbf370f5a817c897e366cdcc7e4ee173b9373f04733`.

Scope: this new mathematical argument and prospective design only, plus its linked primary literature. No previous review, test suite, pilot, model, fit, response call or world generation was repeated. A standard-library calculation checked only the displayed constants and integer accounting. This report is the sole file written; the design and source remain unchanged.

## Mathematical checks

- **Finite-selection example:** the stated expected future risk is exactly 0.18 versus reference risk 0.1, while the two-candidate minimum observed loss is zero. This correctly distinguishes empirical minimization from future risk.
- **Hoeffding constant:** each paired difference has range length 2. The one-sided lower-tail deviation of its empirical mean is therefore bounded by `exp(-n*t*t/2)`. Substitution of `sqrt(2*log(J/delta)/n)` gives `delta/J`; no additional factor of two is needed for this one-sided assertion. The sample size is n per objective, not 2n from pooling the two streams.
- **Multiplicity and selection:** `(16-1)*3*3=135` covers every nonreference candidate, objective and declared prefix. Shared root responses, duplicate local heads and nested prefixes do not invalidate the union bound. The no-PFN family is a subset of the same family, so using the same J is conservative and does not require another multiplicity factor. Outcome-dependent choice among candidates passing the simultaneous inequalities is covered. The empirical rule and arbitrary additional looks are not thereby certified.
- **Identity branch:** a fixed retained local head, evaluated at exactly the reference input with the same loss, has identically zero paired difference and may use upper bound zero. An observed zero mean or numerical sample tie cannot establish that identity. Retaining Y alone does not establish identity of a composed forecast when M changes. A completely identical composed predictor cannot satisfy the strict composed-improvement requirement.
- **Risk claim:** on the simultaneous event, each selected local population loss difference is nonpositive and the selected composed population loss difference is strictly negative. This is a high-probability statement over validation conditional on fixed predictors and the specified laws; it does not certify realized private losses, raw MSE, all worlds jointly or individual predictions. Averaging conditional coverage over fitting data preserves the coverage probability.
- **Distribution shift:** the error identity and `(a+b)^2 <= 2a^2+2b^2` yield the stated pointwise bound. Integrating under the root-input law and applying `dQ/dP <= rho` gives `2*rho*E_P[e_Y^2] + 2*L^2*E[e_M^2]`. These are structural squared-error terms, distinct from the clipped noisy losses of the preceding theorem. Required input-domain Lipschitz control and distributional domination are stated; fit-range inclusion alone cannot supply either. The quadratic noisy-versus-structural mean gap is exactly `c*sigma^2` under the stated zero-mean independent noise. Clipping does not transfer this argument into an untruncated-MSE guarantee.

## Prospective design and accounting

Immutable fitting, fit-only scales/gates, two independent IID validation streams and independent evaluation streams provide the intended separation. Root responses support both local-M and composed-Y objectives without doubling their charge; internal interventions support local-Y. The broader local-Y validation law, clipped loss, changed rule and fit-only denominators are material differences from the prior screen, and the document appropriately disallows attributing cross-screen differences solely to sample count. Within this new design, nested counts hold fitted candidates fixed. The no-PFN contrast tests candidate inclusion in this rule, not pretraining in isolation or equal compute.

The counts reconcile:

- Four fixed outputs plus three rules at three prefixes give 13 outputs per world and 312 cells over 24 paired variant/world instances.
- Shared prehistory: `6*32=192`; posthistory generation: `24*32=768`; validation: `24*(128+128)=6,144`; total training/validation: 7,104.
- Private evaluation: `24*(512+768)=30,720`; total scientific responses: 37,824. The unused eight posthistory rows remain charged. Shared prehistory is not generated four times per base seed.
- Available post-change prefix budgets are 48, 96 and 288. They are information-availability budgets within a dataset generated through the largest prefix, not measured stopping costs or intervention savings.
- Epsilon values are 1.4054365026560627, 0.7027182513280313 and 0.35135912566401567. At n=8, even empirical paired difference -1 gives a positive nonidentity upper bound, so certification is impossible. Reporting that planned negative control without changing thresholds is coherent.

Failures, undefined ratios, all strata, fallback reasons, clipping, raw losses and separate structural diagnostics remain mandatory. Six base systems and their paired variants do not supply independent repeated validation experiments establishing nominal coverage. RNG namespaces, exact generation order, candidate identity enforcement and artificial resource qualification remain explicitly deferred implementation/freeze requirements, not completed qualifications.

## Primary literature check

The descriptions of selection-criterion overfitting in [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html), risk control through multiple testing in [Learn then Test](https://arxiv.org/html/2110.01052v5), and sequential e-process testing in [Zecchin, Park and Simeone (2025)](https://proceedings.mlr.press/v267/zecchin25a.html) agree with those primary sources. LTT's high-probability risk definition and Bonferroni construction support the stated connection. The document correctly avoids claiming implementation of adaptive LTT or an identified cause of earlier selection errors. The linked [Hoeffding publisher page](https://doi.org/10.1080/01621459.1963.10500830) was inaccessible through the browsing tool; the range-length substitution and union-bound algebra above were independently checked, without claiming inspection of that original full text.

## Optional clarification

For later reporting, an explicit sentence could prevent confusion between coverage and conditional-on-update reliability: the guarantee bounds the probability of selecting an update that violates a claimed risk inequality by delta; it does **not** generally bound the probability of a violation conditional on an update having occurred by delta, nor promise nonpositive risk difference after averaging over bad validation events. The current phrase “on that simultaneous event” is already correct, so this is not a required correction. In a hierarchical formulation, take the underlying world's SCM as fixed (or condition on it as well) when invoking conditional IID sampling.

## Scoped author-clarification disposition

Current design SHA-256: `b2103056d86731cad527004fa78a0a3847dd7d1199ada5e0ac91d6c8fbd74147`.

Reviewed the added paragraph under “Systems and immutable fits” and its adjacent interface text only. **No new required findings.** Explicitly withholding truth, variant labels, future prefixes and private endpoints preserves the intended learner boundary. Requiring a fixed pointwise predictor conditional on fitting correctly addresses the theorem's sampling premise: a predictor that adapts to the whole validation batch generally makes the per-observation losses dependent and is not justified by this IID Hoeffding argument. The clarification correctly states that finite five-input checks alone cannot prove the needed property.

A one-input adapter is sufficient only together with the paragraph's overarching fixed-function requirement: fitted state must remain unchanged and repeated calls cannot depend on validation history. A justified batch-invariant implementation must preserve the same property. This remains an implementation obligation, not a completed qualification. The earlier mathematical/count disposition remains applicable. No pure-module code, tests, models, scientific inputs or execution were reviewed or run for this delta.

Scoped clarification recheck: fixing or conditioning on the SCM before fitting, and distinguishing joint violation probability from conditional-on-update probability and expected risk over exceptional events, correctly resolve both optional clarifications; no required findings. Current design SHA-256: `df36d704ff509fad56f2027843ccd262b362e7657aaf16d440e34e619c2123ce`; only those two clarifications were rechecked, with no full review or execution repeated.
