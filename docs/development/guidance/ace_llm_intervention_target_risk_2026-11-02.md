# A target-aware numerical reference for ACE

Status: isolated CPU implementation with artificial analytic fixtures. No scientific worlds, trained policy comparison or new CURC allocation.

## Why this reference is necessary

The question is whether a trained LLM chooses interventions that help an experimentalist reach a useful answer at lower cost. A numerical control should therefore care about that answer. Maximizing information about every mechanism can waste a budget on an uncertain variable that does not affect the target.

The new reference estimates the expected reduction in uncertainty about the target's response, divided by the intervention cost. It uses the same public graph, paid history and learned mechanism summaries as the other policies. It never calls the environment while planning. Its simulated possible measurements are labeled fantasy responses and counted separately from paid responses.

This complements uniform random, menu coverage and local parameter information. It is a stronger comparator in the sense that its objective matches target prediction. Empirical dominance, global optimality and efficient engineering decisions have not been established. A matched learned numerical policy remains required to distinguish language pretraining from amortized numerical acquisition.

## Public prediction task

Before collecting data, define a target node, a finite set of legal intervention queries, positive probability weights summing to one, and a positive fixed error scale. These describe where the experimentalist wants accurate predictions. They contain no measured evaluation outcomes. Private evaluation responses must be generated from a separately frozen stream.

Every policy, including each LLM arm, must receive the same task definition. The isolated implementation currently accepts it as a function argument. It does not yet add a shared structured task descriptor to the LLM prompt or the process interface. That integration and its private-data exclusion check are required before a scientific comparison. Giving the numerical policy an undisclosed evaluation set would violate the design.

The current reference predicts the noise-disabled structural response. It does not count irreducible measurement noise as reducible uncertainty. Its observation fantasies retain the learner's declared noise variances, so acquisition still accounts for noisy measurements. Changing the deployment task to noisy outcomes or an operating-decision regret requires a new explicit objective.

## Exact moments for the current linear SCM

Condition on the paid data retained by the local learner. Each mechanism has an intercept and parent coefficients with posterior mean \(m_i\) and covariance \(V_i\). The local coefficient vectors are independent under this factorized working posterior. For a fixed intervention query, let \(\phi_i=(1,Y_{\mathrm{pa}(i)}^\top)^\top\).

The node mean is

\[
\mathbb E[Y_i]=m_i^\top\mathbb E[\phi_i].
\]

Its second moment is

\[
\mathbb E[Y_i^2]
=\operatorname{tr}\!\left[(V_i+m_i m_i^\top)
                         \mathbb E[\phi_i\phi_i^\top]\right].
\]

For an earlier node \(j\) in topological order,

\[
\mathbb E[Y_iY_j]
=m_{i,0}\mathbb E[Y_j]
 +\sum_{p\in\mathrm{pa}(i)}m_{i,p}\mathbb E[Y_pY_j].
\]

The reason is that the current node's coefficients are independent of earlier nodes. The recursion retains shared-ancestor correlations and covariance between coefficients in the same head. A clamped node is a constant with zero variance. Algebraically, subtracting the outer product of the means gives the full composed covariance. The implementation instead uses the equivalent centered recurrence to avoid cancellation when the mean is large:

\[
\begin{aligned}
\operatorname{Cov}(Y_i,Y_j)
  &=\sum_p m_{i,p}\operatorname{Cov}(Y_p,Y_j),\\
\operatorname{Var}(Y_i)
  &=\operatorname{tr}(V_i C_{\phi_i})
    +\mu_{\phi_i}^{\top}V_i\mu_{\phi_i}
    +m_i^{\top}C_{\phi_i}m_i.
\end{aligned}
\]

Here \(\mu_{\phi_i}\) and \(C_{\phi_i}\) are the feature mean and covariance. The implementation checks finite arithmetic and positive semidefiniteness within a declared numerical tolerance. Only roundoff-scale negative diagonal values can be clipped.

Under squared error, the posterior mean minimizes working-model posterior risk. The weighted target variance across the public query law is its structural prediction risk. These calculations are exact moments of the specified factorized linear model. They do not certify accuracy under misspecification, unknown noise, a wrong graph, or discarded missing-parent data. The latter is not full latent-variable inference.

## One-response lookahead

For each affordable intervention, draw possible noisy measurements from the current learned model. Each draw samples mechanism coefficients and propagates sampled parent values. Make a separate learner copy for each possible receipt. Update only measured, unclamped mechanisms whose parent inputs are measured or known hard clamps. Then recompute the integrated target risk.

For intervention \(a\), the estimated score is

\[
\widehat U(a)=
\frac{R(D)-K^{-1}\sum_{k=1}^{K}R(D\cup z_{a,k})}{c(a)}.
\]

The default is 32 planning draws per affordable action, with a hard implementation bound of 256. The public query law is capped at 64 points. Candidate-specific random streams depend on a fixed decision seed and the action identifier, so changing candidate iteration order does not change the estimates. Candidate failure invalidates the call. It cannot silently remove an inconvenient arm.

The policy chooses the largest score with a fixed identifier tie rule. Negative estimated gains are retained. Monte Carlo standard errors describe the planning approximation, not a confidence interval for experimental performance or a comparison adjusted for selecting the largest score. The policy stops only when the common budget admits no action or the episode is terminal. This is not a calibrated autonomous stopping rule.

The next scientific runner must record the declared query law, scale, decision seed, draw count, scores, planning costs and output. Action-value estimates are model-based. If the working SCM is wrong, they may be wrong. That is one reason to evaluate transferred LLM behavior and numerical controls on fresh regimes rather than assume this reference is an oracle.

## What the new fixtures establish

Six new methods pass from an unrelated working directory. An independent exhaustive coefficient enumeration checks a three-node graph with shared ancestors and within-head covariance. A direct regression calculation checks the exact target risk and gain per cost. Other checks cover clamped targets, weighted queries, scale, missing-parent eligibility, candidate order, affordability, invalid laws and unmodified negative estimates.

One artificial example has an uncertain nuisance variable disconnected from the target. The local-information policy selects that variable. The target-risk policy selects an informative target-parent intervention. This illustrates an objective mismatch, not a scientific ACE victory or an empirical comparison to an LLM.

The initial fixture command used 0.117156 child CPU seconds, 0.161120417 elapsed seconds and 44023808 peak child RSS bytes. These are local preparation measurements, not whole-cycle costs. Exact source and logs are preserved in `results/ace_llm_intervention_target_risk_preparation_20261102/preparation01` and the corresponding exclusive external directory. No pretrained model, optimization or scientific response was used.

A separate arithmetic probe then exposed cancellation in the original second-moment subtraction. At a mean of one trillion and a true variance of one, it returned zero variance. That failed probe and original source remain in preparation02. The centered recurrence fixes the defect. One new large-offset regression and three affected analytic methods pass in preparation03. The corrected command used 0.085137 child CPU seconds, 0.092640208 elapsed seconds and 41189376 peak child RSS bytes. The unaffected methods were not repeated. The initial receipt's `models: 0` field means pretrained models, not the absence of artificial linear learners or fantasy posterior updates.

## Remaining integration

The shared public task descriptor, process wrapper and matched learned numerical ranker remain open. So do frozen training-world supervision, adapter reload and multi-episode inference, independent development and confirmation namespaces, actual resource qualification and engineering transfer. This module advances the comparison needed by the agenda. It does not authorize a scientific result or replace those steps.

## Review disposition

Independent mathematical and implementation review found one required numerical issue, the cancellation defect above. The centered correction has zero required findings remaining. The review covers public-law semantics, eligibility, branch isolation, RNG ordering and the scope of the objective. A subsequent optional fixed-fantasy oracle checks downstream risk variation and its Monte Carlo standard-error calculation without changing the reviewed production module. It passed alone, using0.085011childCPU seconds,0.088084958elapsed seconds and41451520peakRSS bytes. This is not a scientific RNG qualification. There are eight distinct passing fixture methods across the preserved attempts. See [review](reviews/ace_llm_intervention_target_risk_review_2026-11-02.md).
