# Selection reliability before intervention efficiency

October 9, 2026. New theory and prospective design after the completed retention pilot. **No new scientific run, performance claim, or execution freeze.** Earlier protocols, accepted results and failures remain unchanged.

## Evidence motivating the question

The reviewed [retention results](ace_foundation_retention_results_2026-10-09.md) establish conditional candidate benefits and unreliable selection. PFN beats the fixed RBF control on all six missing-Y systems but loses on five of six missing-M systems. Combined selection loses to Grammar32 in five of six null systems and all six coefficient-M systems. Its calibration feasibility inequalities therefore do not imply private improvement. All four selector ablations have identical composed predictions in missing-Y; that benefit cannot be assigned to the guards. These observations motivate a new design; they do not identify the cause of the errors.

## Literature and specific implications

[Cawley and Talbot (2010), JMLR](https://www.jmlr.org/papers/v11/cawley10a.html) explain that optimizing a finite-sample model-selection criterion can itself overfit, with selection-criterion variance playing an important role. Here, choosing among up to sixteen mechanism pairs on five composed calibration responses creates a relevant concern. This is an analogy supported by their general analysis, not proof that overfitting caused our observed failures.

[Angelopoulos, Bates, Candès, Jordan and Lei, Learn then Test](https://arxiv.org/abs/2110.01052) formulate predictive risk control through multiple testing after learning. The proposed ACE adaptation below uses a simple simultaneous upper bound for paired loss differences. It keeps model fitting separate from validation and permits an outcome-dependent choice within a predeclared family. This is an elementary specialization motivated by that framework, not a novelty claim or a wholesale implementation of their methods.

[Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830) supplies the concentration inequality for independent bounded variables used below. Ordinary squared error with Gaussian noise is unbounded. A distribution-free bounded-loss argument therefore requires a specified bounded loss; it must not be advertised as a bound on our existing untruncated NMSE.

[Zecchin, Park and Simeone (2025), Adaptive Learn-then-Test](https://proceedings.mlr.press/v267/zecchin25a.html) develop sequential testing with e-processes to reduce testing rounds while maintaining risk control. That is a relevant future connection when interventions are costly. The initial proposal here uses three fixed sample counts and a union bound; it neither implements their sequential method nor permits arbitrary repeated looks. Adaptive acquisition also changes the sampling law and needs a separate argument.

## 1. Finite selection is not a population comparison

Fix the underlying world (or condition on its SCM), then condition on all fitting data and pretrained weights. Let H contain K fixed complete SCM predictors, including reference r. For a validation sample, enlarging H cannot increase its minimum observed loss. It can nevertheless increase the selected predictor's expected loss on a new observation. The minimum comparison alone contains no generalization statement; a new candidate may win through sampling variation.

For a concrete exact example, take Bernoulli labels with P(Y=1)=0.1 and a single validation label. Predictors r=0 and h=1 use squared loss. The reference has risk0.1. Selection chooses h when the validation label is1, giving expected selected risk0.9×0.1+0.1×0.9=0.18. Its observed minimum loss is always0. Adding a candidate improves the observed minimum while worsening expected future risk. This is an illustrative calculation, not a fitted simulation or an explanation established for TabPFN.

## 2. Conditional update rule with a precise, limited guarantee

For each objective j and fixed candidate h define a loss in[0,1] and paired difference

`D_hj(Z) = loss_hj(Z) − loss_rj(Z) ∈ [−1,1]`.

Fitting, candidate construction, interval gates and loss scales must be independent of validation. Within each objective the validation observations are IID from its declared deployment law conditional on fitting. Different candidates/objectives/budget prefixes may be dependent; the union bound does not require independence across those tests. Predeclare K candidates, q objectives and L sample counts, and use `J=(K−1)qL`. For n_j observations, set

`eps(n_j) = sqrt(2 log(J/delta) / n_j)`;

`U_hj = mean(D_hj) + eps(n_j)`.

For the reference itself, D is identically0, so its upper bound is exactly0. The same exact0 shortcut is valid for a candidate's local objective only when it uses the identical retained head at the identical input with the identical loss, established by predictor identity before validation; a coincidental sample tie is insufficient.

**Claim.** With probability at least1−delta over validation, simultaneously for every predeclared candidate/objective/budget, `E[D_hj] <= U_hj`. A rule that returns r or selects a candidate satisfying both local upper bounds≤0 and composed upper bound<0 consequently has nonpositive expected bounded local risk difference and strictly negative expected bounded composed risk difference whenever it updates, on that simultaneous event.

**Proof.** For one nonidentity difference, Hoeffding applied to independent variables in[−1,1] yields `P(E[D]−mean(D)>t) <= exp(−n t²/2)`. Substituting eps gives delta/J. Summing over at most J events gives delta. On the complement, every accepted inequality transfers to its corresponding expected loss difference. Reference and identity branches are exact equalities. The simultaneous event covers selection using the same validation values. No independence among candidates or objectives is used. Conditioning can be removed by averaging if these conditions hold for every fitting-data realization.

The probability of selecting an update that violates a claimed risk inequality is at most delta. This is not a delta bound conditional on an update having occurred, and it does not guarantee nonpositive risk after averaging over the exceptional validation events.

This is a conditional theorem, not an empirical certificate for any completed pilot. It protects only the declared bounded losses under their respective future laws. It does not bound untruncated MSE, noise-disabled structural risk, unknown changes, arbitrary distribution shift or every individual prediction. If every candidate fails, report retention and its adaptation failure explicitly; abstention is not evidence of recovery. Separate six-system empirical comparisons are not a test of the theorem's nominal coverage.

## 3. Why local validation and composed prediction can disagree

For deterministic truth `y=f_Y(f_M(x))`, let e_M=h_M−f_M and e_Y=h_Y−f_Y. At each x,

`h_Y(h_M(x))−f_Y(f_M(x)) = e_Y(h_M(x)) + f_Y(h_M(x))−f_Y(f_M(x))`.

If f_Y is L-Lipschitz over the needed inputs, then squared composed error is at most

`2 e_Y(h_M(x))² + 2 L² e_M(x)²`.

The downstream error is evaluated under Q, the distribution of predicted parents h_M(X), not under the measured-parent validation law P. If Q is absolutely continuous with respect to P with density ratio≤rho, then

`R_composed <= 2 rho R_Y,local(P) + 2 L² R_M`.

This follows by integrating the pointwise inequality and using `E_Q[e_Y²] <= rho E_P[e_Y²]`. It requires square-integrable errors, finite L and a genuine distributional domination constant. Fit-range inclusion supplies neither rho nor L; atoms, support gaps or shifted density can invalidate domination. We do not estimate a certified rho or Lipschitz bound from the current pilot. This explains why preserving a local predictor at the same input is logically weaker than preserving final outputs.

There is also a distinct target-law mismatch. With natural Gaussian upstream disturbance epsilon_M, noisy root-action observations have conditional mean `E[f_Y(f_M(x)+epsilon_M)]`, not necessarily `f_Y(f_M(x))`. For quadratic f_Y(m)=a+bm+cm², zero-mean epsilon_M of variance sigma² produces exact difference c sigma². More IID noisy validation cannot remove that difference. With bounded/clipped loss, its risk-optimal predictor need not even be the conditional mean; specify the loss and deployment law together.

## 4. Concrete fresh validation-budget proposal

This is a design for implementation/review, not an allocation or frozen scientific protocol. Its question is narrow: **holding every fitted candidate fixed, how do validation count and a simultaneous update rule change selection and local harm?** It is not an intervention-acquisition competition or an equal-total-data comparison to learners refitted on all paid validation rows.

### Systems and immutable fits

Reserve fresh base seeds94000–94005, four paired variants null/coefficient_M/missing_M/missing_Y with the inherited coefficient distributions and Gaussian noise SD0.05. Do not generate these seeds before a new implementation/source/model/runtime/endpoint freeze. Reuse no prior scientific artifact as training data. Keep one new32-row prehistory per seed and one new24-row post-fit history per variant: obtain the24fit rows by generating a new32-row inherited-menu history and designating its existing24-row fit split; the other8 rows remain paid but are unused by this new selector. Thus charge32 post-fit-generation responses, not24. This deliberately retains exact15natural-M/24Y expert fitting eligibility and fixed RBF/PFN configurations. The eight unused rows must be persisted and disclosed, never repurposed after outcomes.

Fit retained Grammar24 from prehistory and post-change Grammar24/RBF24/PFN24 once, all before validation. The learner must not receive variant labels, true equations, future prefixes or private endpoints. Every effective predictor must be a fixed pointwise function conditional on fitting: a future runner needs a one-input-at-a-time adapter or a justified batch-invariant implementation. Finite five-input compatibility checks alone do not prove that assumption; adapting predictions to the whole validation input batch would require a different proof. No statistical execution qualification is implied by this proposal. Form all16 head pairs with the fixed interval-retention transformation of the prior design. Include the retained pair. All pairs remain available: do not use a small preliminary calibration sample to remove heads. Freeze fit-only intervals, target scales and predictor hashes before generating validation. No candidate refit at larger validation budgets.

### Two validation streams and three budgets

For each world generate independent IID root interventions `do(X=x), x~Uniform[-1,1]` and independent IID internal interventions `do(X=0,M=m), m~Uniform[-2,2]`, both with the ordinary independent Gaussian disturbances enabled. Each returned vector is one charged response, not one charge per objective. Root responses supply natural M for local-M loss and Y for composed loss; internal-M responses supply Y for local-Y loss. This deliberately matches local-Y assessment to its declared broader intervention distribution; it changes validation support relative to the previous screen. The experiment must not attribute differences versus the prior screen solely to sample size.

Use nested prefixes n=8,32,128 **from each stream**. The full dataset is generated once and prefix reuse does not create additional responses. RNG namespaces, generation order and saved stream hashes must be specified by the future implementation and independently frozen. Selection at a smaller prefix must not access a later prefix. K=16,q=3,L=3 gives J=135 and delta=.05 per world across all its three prefixes/objectives/candidates. This is not a5% familywise guarantee across24worlds; no such claim is planned.

For objective j use `loss=min((prediction−observation)²/s_j²,1)`, where `s_j²=max(variance of corresponding eligible post-fit labels,1e−12)` with population variance. Y-local and composed use the same Y fit scale. The scale is fixed before validation; unlike earlier NMSE denominators it does not use the full32 generated rows. Record floor and clipping activations explicitly. Clipping limits the guarantee to this surrogate and can hide large raw errors, so raw errors remain mandatory outputs.

### Three decision rules, plus fixed references

At each prefix report: (a) empirical constrained selection, requiring local mean paired differences≤0 and composed mean<0; (b) simultaneous-bound selection using U as above; (c) the same simultaneous rule excluding PFN, using the conservative sameJ=135 to avoid confounding the comparison by multiplicity width. For all rules, choose the admissible pair of smallest composed empirical bounded loss, then fewer replaced heads, then retained/grammar/rbf/pfn order. If none is admissible, return the retained pair with an explicit fallback reason. The empirical rule is a new specified rule, not the old combined selector under a renamed label.

All candidate failures invalidate rules that require them; never silently drop an arm. A separately declared no-PFN rule may remain valid if only PFN fails. Identity shortcuts rely only on frozen head identity; no sample-zero shortcut. Fixed reported predictors are retained, interval-gated Grammar24, interval-gated RBF24 and interval-gated PFN24. Their finite-sample statistical bounds are not claimed unless they are selected by the tested rule. Four fixed predictors +3rules×3prefixes =13outputs/world,312scientific output cells; all16candidate-pair private predictions may be retained as explicitly evaluator-only diagnostics after selection seals, never routed back into rules.

### Evaluation and decision criteria

After all prefix selections are durably sealed, use independent256-response root and256-response internal-M streams with Gaussian noise enabled. Report all three bounded deployment losses, paired differences from retention, untruncated MSE and training-scale NMSE, clipping/floor rates, every selection/U value and fallback frequency. Separately retain the previous three256-response noise-disabled diagnostic endpoints (composed-Y,local-M,local-Y) to expose objective mismatch. These diagnostics do not enter selection or the bounded-risk theorem. Declare which losses and noisy streams overlap so counts are not multiplied by methods.

Private evaluation response charge per world is512noisy+768structural=1280;24worlds=30720. Training/validation charge is6×32prehistory+24×(32posthistory+128root+128internal)=7104. Total proposed scientific responses37824, shared across methods. At any reported prefix the available post-change budget is32+2n (48,96,288); prehistory32is separately disclosed perworld but generated once perbase seed. No interventions-to-threshold claim follows from comparing these curves alone.

Prespecified descriptive comparisons: simultaneous versus empirical at each prefix; with-PFN versus no-PFN simultaneous at each prefix; each rule at128 versus8; and each output versus retained. Report every system/variant/endpoint, raw paired differences, defined ratios and fallback reason. Zero-reference/failed ratios remain undefined in full summaries; no complete-case filtering or score-based replacement. Larger n is not declared a win merely because it certifies more updates: it must also be considered with raw/structural harm and actual paid cost. Permanent fallback, near-universal clipping, or the reference failing after a change are informative negative outcomes, not successes to hide.

The bound may be too conservative at these sample counts. For J135 and delta.05, eps is about1.405 at8,0.703 at32 and0.351 at128; no nonidentity composed candidate can pass at8 because the most favorable observed difference is−1. This analytic impossibility is part of the design, not a reason for a post-hoc threshold change. A future sharper bound or sequential test is a different method requiring a new pre-outcome specification. Do not increase n until something passes.

### Implementation and resource gate

Next authorized local work is a pure bounded-loss decision module and artificial boundary tests, with a separate mathematical/design review. No fresh scientific generation in this pass. A future runner must bind exact source/candidates/checkpoint/runtime and stream namespaces; preserve all attempts; qualify one actual-runtime artificial integration before admission. Existing scientific model-attempt usage remains90.044439childCPU+.108012supervisorCPU seconds; conservative reservations7920/28800,0GPU. No extra reservation or new allocation is created by this document. Before any new run, measure artificial resource demand and reserve against remaining budget and the22:00UTCcycle deadline. No GPU is indicated by these small CPU candidate fits.
