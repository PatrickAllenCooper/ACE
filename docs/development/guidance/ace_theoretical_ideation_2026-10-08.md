# ACE: explanations, derivations and candidate methods

Date: October 8, 2026. Author: Codex, AI research collaborator.

Status: **theoretical ideation for Patrick's review**. Exact statements below have explicit assumptions. Explanations of our empirical results are hypotheses. Proposed methods have not been shown superior. This note uses accepted A/C/F evidence and historical acquisition studies; it uses no unqualified prospective B performance and changes no frozen experiment.

## Executive position

My proposed organizing principle is:

> **Choose experiments and fitting procedures to reduce error in the model we will actually deploy, under the actions we intend to use.**

That requires coordinating four objects: usable observations, the deployment distribution, the learner's remaining fitting error, and propagation through the graph. A policy can collect informative measurements while its online learner leaves their information unused. A final refit can reduce local error while making the composed prediction less stable. An uncertainty score can be mathematically sophisticated while ranking the same actions as a simpler rule.

The strongest near-term candidate is **pooled mechanism fitting with a deployment loss and explicit propagation regularization**, selected by an independently reserved validation protocol. A subsequent acquisition candidate should reduce **target-relevant posterior risk per resource cost**, with systematic exploration retained. Neither candidate requires changing or rescuing the accepted studies.

## 1. Evidence that an explanation must accommodate

- **Delivery helps substantially in a bounded setting:** 12 histories of one deterministic emulator, geometric paired exact-level error ratio 0.183, interval [0.102, 0.327], 11/12 improved. This compares additional all-paid fitting with unchanged online weights, on an exposed grid, using the median of three scored fitting initializations. It does not isolate data reuse from optimization. History 124753321 worsens by a factor 2.015177. [Claims and scope](/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json).
- **Long fitting and data retention matter in exploratory attribution:** fixed initialization 0 continuous-error ratios include 0.096 against final-buffer fitting and 0.034 against 100-update fitting. The all-paid versus admitted-long ratio is much closer, 0.980. These are selected recipe comparisons on one emulator; they do not prove a population decomposition or equal-compute effect. [Generated values](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_claims.tex).
- **Acquisition improves over a specified random rule, with important limits:** historical shifted-mechanism PEV error 0.015430 versus random 0.043898, 19/20 wins, secondary paired p=0.00750. Its primary comparison against ordinary coverage fails at p=0.05108. Simpler variance scoring is unresolved relative to PEV. [Study](/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md).
- **Value design can explain some apparent acquisition gains:** endpoint coverage beats PEV on three reused development systems, but the ranking reverses on 20 fresh systems. A different sparse-DAG pilot fails its promotion gate. [Development control](/Users/pat/code/ACE/results/research_pev_extreme_value_dev/README.md), [fresh replication](/Users/pat/code/ACE/results/research_pev_endpoint_replication_v1/README.md), [graph shift](/Users/pat/code/ACE/results/research_pev_random_dag_dev/README.md).
- **Topology and the strength of the control matter:** posterior-risk pair selection passes a matched factorial-control gate in the 40-system binary-tree study and the 40-system random-recursive study; it fails the 40-system fanout gate. These sequential studies do not establish a statistically confirmed topology interaction. [Binary tree](/Users/pat/code/ACE/results/local_connected_factorial_pair_confirmation_20260929/README.md), [recursive](/Users/pat/code/ACE/results/local_connected_random_recursive_pair_20261001/README.md), [fanout](/Users/pat/code/ACE/results/local_connected_fanout_pair_confirmation_20260929/README.md).
- **A suitable simple model can dominate neural delivery:** physical delivery beats Fourier regression in 0/11 conditions. These are correlated conditions of one apparatus, not independent worlds. [Physical boundary](/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json).

No one mechanism has yet been causally established as explaining all these patterns. The useful question is which explanations yield distinguishable predictions.

## 2. Explanation A: measurements and usable information are different

### Mechanism eligibility comes first

For an invariant mechanism `V_i = f_i(Pa_i) + U_i`, a row in which node i is clamped is not a natural observation of f_i. Other naturally operating mechanisms can still provide usable labels, provided their measured parents and the retention/noise assumptions are valid. One paid response can therefore support several mechanism regressions; usable label count is different from response count and from independent-system count.

The existing eligibility proposition correctly requires conditional mean-zero noise after retention. Predictable actions and all-paid retention help, but response-dependent terminal selection, confounding or changing mechanisms can break that condition. [Existing theory](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex).

### Exact illustration: additional information lowers Bayesian variance

Assume a correctly specified linear Gaussian mechanism, a proper Gaussian prior with precision Λ0 ≻ 0, independent observation noise variance σ² > 0, and an action policy whose choice depends on recorded history rather than unknown parameters beyond that history. Natural eligible labels yield

$$
\Sigma_D=\left(\Lambda_0+\sigma^{-2}\sum_{t\in D}x_tx_t^\top\right)^{-1}.
$$

Adding eligible observations produces a positive semidefinite precision increment, so `Σ_(D∪E) ≼ Σ_D`. For a fixed linear prediction gᵀθ, posterior squared-error risk `gᵀΣ_Dg` cannot increase. This is a statement about correctly specified posterior uncertainty, not realized frequentist test error or an arbitrary neural optimizer.

Adaptive collection does not automatically invalidate this Bayesian update: a known policy's action probabilities factor out of the parameter likelihood conditional on history. Conversely, it does not justify conditioning on the final design as if ordinary least-squares observations had been collected passively. Adaptivity can affect estimator bias and inference; see [Deshpande et al., Accurate Inference in Adaptive Linear Models](https://arxiv.org/html/1712.06695).

### Empirical interpretation and discriminating prediction

All-paid retention could recover information discarded by short buffers or response-selected admission. However, the exploratory 0.980 all-paid/admitted-long ratio is consistent with much of the available information already being retained by that particular admitted set. The much larger short-fit/long-fit difference makes optimization a serious competing explanation.

**Prediction:** under a future matched fitting budget, retention should help primarily when it increases informative design directions or covers deployment contexts. Duplicating highly redundant rows should have much less effect than adding informative directions. More rows alone cannot remove an unobserved direction of a mechanism.

## 3. Explanation B: delivery closes an optimization deficit

An online model follows a changing sequence of objectives, batches and optimizer states. The delivered model instead optimizes a fixed pooled objective. Their difference bundles observation eligibility, row weighting, optimizer history, normalization and extra computation. It is not automatically a new learning principle.

There is also an information conservation principle. Under a joint probabilistic model, if the delivered estimator uses only saved observations D and independent algorithmic randomness, `Θ → D → θ̂` is a Markov chain and the data-processing inequality gives `I(Θ; θ̂) ≤ I(Θ; D)`. Additional fitting does not create environmental information; it can extract more useful predictions from information already recorded. This statement does not identify a measured mutual-information gain from ACE's MSE change. A model selector using test labels has an additional input and is outside this saved-data-only premise.

Let F̂_D be one fixed empirical objective and let θ_on and θ_del belong to the same hypothesis class. If delivery achieves an approximate empirical minimum with tolerance δ,

$$
\widehat F_D(\theta_{del})\leq\widehat F_D(\theta_{on})+\delta.
$$

If, additionally, `sup_θ |R(θ) − F̂_D(θ)| ≤ ε` for the *same deployment risk*, then

$$
R(\theta_{del})-R(\theta_{on})
\leq \widehat F_D(\theta_{del})-\widehat F_D(\theta_{on})+2\epsilon
\leq\delta+2\epsilon.
$$

Proof: add and subtract the two empirical risks. No such uniform deployment bound has been verified for ACE. In particular, a sum of observed-parent losses is not the composed deployment risk, and adaptive histories do not provide a free iid generalization theorem.

For a quadratic empirical loss with Hessian H and gradient g, one gradient step gives the exact identity

$$
\widehat F(\theta-\eta g)-\widehat F(\theta)
=-\eta\|g\|^2+\frac{\eta^2}{2}g^\top Hg.
$$

If `0 < η < 2/λmax(H)`, a nonzero gradient decreases this empirical loss. This explains how additional computation can release information already in the data. It does not prove that additional Adam updates improve held-out or snapped error.

**Prediction:** if optimization deficit dominates, an equal-compute pooled comparator should remove much of delivery's advantage. If structure matters beyond this, a remaining difference should survive equal usable supervision, matched objective and matched fitting resources. Current comparisons do not identify that residual effect.

**Method implication:** maintain an explicit pooled objective and schedule bounded consolidation while collecting, instead of letting the last online state silently become the deliverable. Reserve compute for final consolidation as part of the protocol. Any consolidation schedule comparison is future work, not an amendment to the frozen study.

## 4. Explanation C: small local errors can become large deployment errors

### Exact signed propagation identity

Consider a deterministic known DAG under identical correctly clamped interventions. Let true values be v, predictions v̂, signed errors `e = v̂ − v`, and local residuals `r_i = f̂_i(v_Pa(i)) − f_i(v_Pa(i))`. Assume each learned function is continuously differentiable along the segment joining its true and predicted parent vectors. Define

$$
H_{ij}=\int_0^1 \partial_j\widehat f_i\big(v_{Pa(i)}+t e_{Pa(i)}\big)\,dt
$$

for parent j of i, and zero otherwise. Set clamped rows and residuals to zero. The fundamental theorem of calculus gives

$$
e=r+He,\qquad e=(I-H)^{-1}r
=\sum_{k=0}^{d-1}H^kr.
$$

The inverse is a finite path sum because H is strictly triangular in topological order. For terminal target T, let b_T be its coordinate basis vector and `qᵀ = b_Tᵀ(I−H)⁻¹`; then the scalar terminal error is `e_T = qᵀr` exactly. For an affine learned DAG, H is constant and the identity is directly computable. For nonlinear learned models it depends on the unknown true values as well as predictions; it is explanatory, not a deployed certificate.

The smoothness assumption is sufficient rather than a claim about every implementation. Piecewise-linear neural heads require compatible derivatives along each segment and treatment of kinks; a single autograd Jacobian at a sampled point is not the integral H or a certified bound.

Every eigenvalue of a strictly triangular DAG Jacobian is zero. Its spectral radius therefore cannot diagnose amplification: arbitrarily large path gains can coexist with spectral radius zero. Target path sums and operator norms are the relevant quantities, with absolute path sums supplying conservative versions when cancellation is fragile.

This sharpens the manuscript's absolute Lipschitz bound: signed paths can reinforce or cancel. If q is treated as fixed under a justified linearization, the mean squared target error is

$$
(q^\top b)^2+q^\top C_rq,
$$

where b and C_r are residual mean and covariance. Dropping cross-mechanism covariance requires a separate assumption. In a nonlinear model q can depend on r, so plugging unconditional b and C_r into this expression is generally unjustified.

### Geometry creates an identification limit

Root interventions can constrain parent vectors to a manifold. On `M=R`, mechanisms `R+M` and `R+M+K(M−R)` agree for every root intervention, yet have arbitrarily different sensitivity normal to that manifold. More root data on the same manifold cannot determine K. The existing proposition already proves this failure inside a coordinate bounding box.

**Prediction:** a worsening history can have good measured-parent fits but elevated free-running error, high normal sensitivity, or correlated residual amplification. These are competing diagnostic patterns; none is established as the cause of history 124753321 merely by this argument.

### Candidate method: fit the observed mechanisms and the composed prediction

For future eligible measurements and supported recorded actions, consider

$$
\mathcal L(\theta)=
\sum_i\lambda_i\widehat R_{i,eligible}(\theta_i)
+\gamma\widehat R_{terminal,free}(\theta)
+\rho\widehat\Omega_{propagation}(\theta).
$$

The terminal term compares the free-running prediction under each *actual recorded action* with its measured target. It need not query a new environment merely to reuse such a label. The regularizer penalizes excessive target-relevant path sensitivity on supported training contexts. Weights, normalization and admissible clamps must be fixed before evaluation; mechanism labels overwritten by interventions remain excluded from local fitting.

One possible regularizer uses an absolute local Jacobian A=|J| on supported training rollouts and penalizes `‖b_Tᵀ(I−A)⁻¹‖²`. Its finite DAG inverse avoids imposing a nonexistent cyclic spectral-radius condition. This penalty can discourage genuinely large physical gains too; it must be checked against mechanism fit and validation. If a training-only tangent estimate is reliable, a separate normal-gradient penalty can target underconstrained directions. Units and scaling matter in both penalties.

This is a hybrid prediction objective. Its end-to-end term can trade local mechanism fidelity for terminal fit, so the resulting heads must not automatically be called identified causal mechanisms. Normal-sensitivity regularization chooses a stable extension; it does not learn the true off-manifold extension from missing data.

If mechanism fidelity is a central requirement, an alternative is constrained fitting: minimize terminal loss plus a propagation penalty subject to each eligible mechanism's empirical loss remaining within a predeclared tolerance of its local-fit baseline. This turns an implicit weighting tradeoff into an explicit fidelity allowance. It still controls only observed losses and estimated sensitivities; it supplies no missing causal identification or guarantee on unseen contexts. A future comparison should distinguish this constrained version from the weighted objective rather than silently switching between them.

**Critical restriction:** perturbing a parent input and attaching its old child label is generally invalid. Reusing an observed root action/target pair for a composed loss is legitimate prediction supervision; inventing labels for unobserved internal contexts is a new assumption. DAgger motivates attention to prediction-induced inputs, but its oracle labeling and regret assumptions do not transfer automatically to ACE. [Ross, Gordon and Bagnell](https://proceedings.mlr.press/v15/ross11a.html).

## 5. Explanation D: sophisticated uncertainty scores can become redundant

The implemented PEV score sums descendant integrated covariance reduction over synthetic parent-reference points. Its simpler variant sums candidate variance. The historical three-member ensemble gives a centered covariance rank of at most two. A bootstrap ensemble is not automatically a calibrated Bayesian posterior. [Implementation](/Users/pat/code/ACE/baselines.py:1235).

### Exact rank-one reduction

For one mechanism, suppose the centered ensemble prediction at every input is `a(x)z`, for the same centered member vector z. Write `v(x)` for member variance and assume a fixed ν > 0. Then

$$
C(x,u)^2=v(x)v(u),\qquad
S_{IVR}(x)=\frac{v(x)}{v(x)+\nu}\int v(u)\,d\mu(u).
$$

Proof: covariance is `a(x)a(u)Var(z)`; squaring yields the product of variances. The score is strictly increasing in v(x) whenever integrated reference variance is positive. It therefore ranks points identically to ordinary variance. At ν=0 and v(x)>0 it saturates to a constant. Different mechanisms, reference weights, context averages or noise denominators can break global ranking equivalence. Nearly rank-one disagreement predicts similarity, not an exact identity for the complete historical policy.

This gives a concrete explanation to investigate for the PEV/variance tie. Other explanations remain possible: shared misspecification, insufficient covariance calibration, different action sequences reaching similarly informative designs, or score integration over the wrong distribution.

The archived shift study already records different executed sequences: the policies selected the same target in only 95/640 paired steps, with 18 exact target/value matches. Therefore the analytic rank-one result cannot be promoted to an assertion that the complete historical policies were identical. It concerns a conditional scoring geometry that might contribute to similar final performance.

### Two implementation-specific concerns

1. PEV draws independent reference parent coordinates uniformly from a box. Actual feasible deployment parent vectors can occupy a very different correlated distribution. Reducing variance on unreachable combinations can waste effort for a feasible prediction objective.
2. The denominator's `noise_var` is an exponential average of residual squared error of the ensemble mean, floored at 1e−4. That mixes observation noise with approximation and optimization error; it is not a separately measured noise variance. The candidate simulator also composes mean mechanisms without internal noise, which need not reproduce a noisy interventional distribution through nonlinear mechanisms. [Residual update and simulation](/Users/pat/code/ACE/baselines.py:1135).

Cohn and colleagues' variance-reduction argument explicitly integrates over a specified prediction distribution and uses model assumptions for the variance update. ACE's ensemble score is an approximation to that idea, not an inherited optimality guarantee. [Primary paper, NeurIPS version](https://papers.nips.cc/paper/1011-active-learning-with-statistical-models.pdf).

**Predictions:** if low rank explains redundancy, reference/candidate covariance should have a dominant spectral direction and the two scores should agree in ranking where their objectives coincide. If integration mismatch dominates, replacing only the reference distribution with a predeclared feasible distribution should change selection meaningfully. If uncertainty is mainly common bias, both scores can confidently fail despite low disagreement.

## 6. Candidate acquisition method: reduce target-relevant risk

### A solvable version

Suppose current parameter uncertainty is Gaussian with covariance Σ ≻ 0. A candidate natural observation is `Y=xᵀθ+ε`, with known design x and independent Gaussian noise `ε∼N(0,σ²)`, σ² > 0. The target of interest is a fixed linear functional `gᵀθ`. Conjugate conditioning gives

$$
\Sigma'=\Sigma-
\frac{\Sigma xx^\top\Sigma}{\sigma^2+x^\top\Sigma x},\qquad
\Delta_T(x)=\frac{(g^\top\Sigma x)^2}{\sigma^2+x^\top\Sigma x}.
$$

An observation with high parameter variance but zero covariance with the target has zero target variance reduction. For example, with `Σ=diag(9,1)`, target `g=(0,1)` and σ²=1, measuring `(1,0)` reduces target variance by 0; measuring `(0,1)` reduces it by 0.5. Maximum observation variance prefers the former. Thus target-aware design can be strictly better for this specified posterior prediction objective; that does not establish superiority for ACE's nonlinear experiments.

For multiple eligible labels conditioned on a known observation design, assume `Y_a=Φ_aθ+ε_a` with independent Gaussian noise vector `ε_a∼N(0,R_a)`, R_a ≻ 0, and use the joint update:

$$
\Sigma_a'=(\Sigma^{-1}+\Phi_a^\top R_a^{-1}\Phi_a)^{-1}.
$$

For a predeclared distribution ν of supported deployment actions and fixed target gradient g(u), let `M_T=E_ν[g(u)g(u)ᵀ]`. A candidate score is

$$
\frac{\operatorname{tr}[M_T(\Sigma-\Sigma_a')]}{c(a)}.
$$

This is exact for the stated linear Gaussian target/observation model. For nonlinear composed SCMs, target gradients, induced observation features, covariance and noise are estimated; the score is a local surrogate. Joint observations require their joint update, rather than blindly multiplying a one-observation gain by batch size. Clamped mechanisms contribute no natural labels.

An intervention can produce random measured parent features, so its design matrix is generally not known before execution. Prospective utility is the current target risk minus **expected posterior target risk**, averaging over the full predictive response/design distribution under that action. The displayed conditional covariance formula can be averaged over random designs only when its conditional model remains valid; if the feature values themselves reveal parameter information, their likelihood must also be included. Substituting simulated mean parent features is a surrogate even for linear mechanisms. This is a substantive distinction between selecting a known regression design and selecting an action that induces uncertain designs.

Prediction-aware information matching is established related work, including [Kurniawan et al.](https://arxiv.org/html/2411.02740v5). The possible ACE contribution is a carefully specified implementation linking action-induced eligible labels, deployed graph propagation and separate measurement/compute costs—not the invention of prediction-oriented optimal design.

### Geometry and topology predictions

With no sample in a design direction, that component can remain prior-dominated even after many redundant measurements. More actuator magnitude increases linear information in an already excited direction but does not create a missing direction. In the scalar known-intercept model, endpoint values ±a give per-sample information `a²/σ²`, while uniform values on [−a,a] give expected information `a²/(3σ²)`. This factor of three is an information illustration, not a claim of threefold error improvement in finite nonlinear runs.

Joint interventions may rotate the design into directions that root-only data cannot excite. They also cost more and can overwrite the very labels needed to learn a targeted mechanism. Their value depends on the eligible information matrix, not on actuator count alone.

A fanout hub can affect many descendants and make a strong static schedule nearly sufficient. A recursive graph can leave uneven information bottlenecks that reward adaptation. These hypotheses fit the historical pattern, but the sequential graph studies cannot identify a causal topology effect. A future test must vary topology and information geometry under one frozen comparison.

### Resource allocation should include learning from existing data

The next action may be a measurement, a fitting step, or stopping and delivering. Compare expected deployment-risk reduction with separate measurement and computation prices:

$$
\widehat\Delta R(s)-\lambda_{env}c_{env}(s)-\lambda_{cpu}c_{cpu}(s).
$$

Prices are chosen resource tradeoffs, not constants discovered by the existing experiments. Maintain hard budgets and account for acquisition simulation, optimization, measurements and evaluation separately. A myopic score need not be globally optimal. In particular, a hypothetical experiment's information is only useful if the remaining fitting budget can convert it into the delivered model.

## 7. Candidate delivery method: validate stability and exploit simple structure

### Models should match the task's functional structure

Fourier regression's physical advantage is consistent with a suitable periodic representation having lower approximation error and a simpler fitting problem. It does not prove the true physics is exactly in that basis. A causal graph specifies allowed dependence, not a unique neural functional form or the best predictor for every target.

A future delivery procedure should include predeclared simple head families—linear, periodic or other domain-supported bases—and a neural residual only when validation supports it. Complexity penalties and family selection introduce additional decisions and must be included in their own protocol. Validation must be separated from final reporting; the historical scored initialization median is not such a deployment rule.

### Snapping changes the objective

An error can cross a target-level boundary even when aggregate MSE is small. The manuscript's margin proposition proves that target distance to those boundaries matters. Lower MSE alone cannot order two models by exact-level error. Candidate delivery should therefore state whether it serves continuous prediction, snapping or decision cost, and evaluate that same objective on reserved validation data.

### A future validation-based fallback, with limited guarantees

For M fixed predictors **including the online baseline**, all independent of the validation set, and n iid validation observations drawn from the declared deployment distribution, assume each loss lies in [0,1]. Hoeffding plus a union bound gives, with probability at least 1−α, simultaneous deviations at most

$$
\epsilon=\sqrt{\frac{\log(2M/\alpha)}{2n}}.
$$

Selecting the empirical minimum has true risk at most `min_m R_m + 2ε` on that event. If replacing the online baseline is allowed only when a candidate's validation loss is lower by more than 2ε, the replacement cannot have greater true risk on the same event. This elementary guarantee requires fixed candidates independent of that validation set; hyperparameter tuning on the set invalidates the stated M. MSE needs bounded loss or a different concentration argument.

The iid validation rows are not the independent-world units of scientific reporting, and validation measurements must be paid/accounted for. Correlated actions, adaptive reuse or conditions of one apparatus do not satisfy this guarantee automatically. The existing archived exposed grid cannot retroactively serve as a pristine prospective validation set.

## 8. Priorities and falsifiable work sequence

### Priority 1: explain delivery with supported diagnostics

Keep the accepted results intact. After the already-required exact-runtime replay, use separately source-bound, read-only diagnostics on saved artifacts to distinguish: local fitting error, free-running target error, support geometry, residual correlation and path sensitivity. A measured Jacobian is a diagnostic, not a uniform Lipschitz certificate. Exploratory diagnosis of the known worsening history requires disclosure; it must not select a rescue recipe.

Decisive future comparison: same eligible observations, fixed initialization rule and matched fitting budget, with explicit objective ablations: pooled mechanism loss; mechanism plus terminal loss; and mechanism plus terminal loss plus propagation penalty. Predeclare all cases, model families and validation rules. This comparison tests the objective changes. A separate architecture comparison must hold the objective and usable supervision fixed. If the benefit disappears at equal fitting resources, revise the explanation toward optimization rather than architecture.

### Priority 2: determine whether the uncertainty machinery adds information

A proposed diagnostic can inspect covariance rank, score ranking agreement and residual calibration without new measurements, but only if the relevant snapshots exist. If score state was not saved, recovering it requires new inference and a separately bounded scope. Do not present a derivation as evidence that the historical ensemble was rank one.

Decisive future comparison: ordinary variance, existing IVR, feasible-reference IVR, and target-risk design with the same action menu, value support, learner, response budget and charged computation. Keep endpoint/factorial coverage as strong controls. Prediction distributions must be fixed without final-test outcomes.

### Priority 3: test mechanism-specific predictions, then external relevance

Vary target sensitivity separately from graph size; vary reachable parent rank separately from sample count; and vary periodic versus misspecified functional bases separately from neural capacity. Use analytically known environments first to make the causal question legible. Any actual fitted experiment requires a new frozen protocol and resource scope; this note authorizes none.

Only after those distinctions are clear should a prospectively chosen external task support a broader usefulness claim. Historical foundation-model agendas remain stopped. Current submission preparation still requires its original replay/reporting/integrated review gates; theory ideation is useful parallel work, not a substitute.

## 9. What is proved, what is proposed, what remains unknown

**Elementary derivations and standard implications in this note:** saved-data information conservation; Gaussian variance reduction; empirical-risk comparison under a stated uniform bound; quadratic fitting descent; exact signed DAG propagation; rank-one IVR/variance monotonicity; target-functional observation gain; scalar endpoint information; and the bounded-loss validation selection guarantee.

**Hypotheses about ACE:** optimization deficit explains much of delivery; off-manifold sensitivity explains some failures; low-rank disagreement or reference mismatch explains the acquisition ablation; suitable basis structure explains Fourier's advantage; target-relevant information geometry explains topology-dependent acquisition gains. These are not established diagnoses.

**Method proposals:** pooled consolidation, mechanism-plus-terminal fitting, propagation regularization, feasible/target-aware acquisition, separately measured noise, structured model alternatives and reserved validation fallback. Their superiority is open.

**Most valuable unresolved question:** can we reduce deployment error while preserving mechanism fidelity and avoiding unsupported extensions, under an explicit measurement-and-fitting budget? That question connects the successful results and the failures without requiring a universal claim.

## 10. Scope of verification and literature

The companion [algebra checks](/Users/pat/code/ACE/scripts/research/check_ace_ideation_algebra.py) contain only fabricated finite mathematical witnesses: covariance updates/target gain, rank-one scores, signed DAG paths, correlated-error cancellation, quadratic descent, endpoint information and a missing design direction. No learner is fitted, environment queried, checkpoint loaded or empirical outcome added. These checks supplement the proofs; they do not prove that assumptions hold for ACE.

Primary literature checked for this note: [Cohn et al., Active Learning with Statistical Models, NeurIPS version](https://papers.nips.cc/paper/1011-active-learning-with-statistical-models.pdf); [Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html); [Deshpande et al., adaptive regression inference](https://arxiv.org/html/1712.06695); and [Kurniawan et al., information matching, v5](https://arxiv.org/html/2411.02740v5). These establish precedents and relevant assumptions. No novelty claim or direct inheritance of their guarantees is made.
