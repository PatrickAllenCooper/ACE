# A useful pretrained expert need not be the best standalone predictor

Date: 2026-10-09. Author: Codex. These elementary identities and the counterexample are prospective method analysis, not new empirical results or a novelty claim. They motivate the next design after the fixed component pilot.

## Fixed terminal predictions: the relevant quantity is complementary error

Let g and f be two fixed scalar predictors under a declared target distribution, with squared risks Rg and Rf. Define errors eg=g−Y and ef=f−Y, cross moment C=E[eg ef], and D=E[(f−g)^2]=Rg+Rf−2C. For a constant mixture hλ=(1−λ)g+λf,

`R(λ) = (1−λ)^2 Rg + λ^2 Rf + 2λ(1−λ)C`

`       = (1−λ)Rg + λRf − λ(1−λ)D.`

This follows by expanding the square; no independence assumption is required. If D>0, the best population constant mixture weight in[0,1] is `clip((Rg−C)/D,0,1)`. The derivative at0is `2(C−Rg)`. Therefore an expert with worse standalone risk Rf>Rg can still help at a small weight if C<Rg: it corrects enough of the baseline's errors. A pretrained expert that merely repeats or amplifies those errors cannot earn its place by confidence alone. If D=0the two predictors coincide almost surely under the specified distribution.

The identity applies to fixed predictors and a common target distribution. It does not authorize choosing λ using private evaluation outcomes. A learned λ requires separate training-only selection/calibration and fresh evaluation; arbitrary adaptive data do not supply an iid generalization guarantee. Input-dependent weights also require a different analysis: they cannot simply be substituted into the constant-weight formula using averaged λ.

## Why mixing SCM heads is not the same operation

Consider a deterministic two-head chain with true values M=1 and Y=M=1 at one supported input. Predictor G emits M=0.9 and uses the Y mechanism2M; predictor F emits M=2 and uses the Y mechanism0.5M.

- G's composed prediction is1.8; squared error0.64.
- F's composed prediction is1; squared error0.
- Blending terminal predictions equally gives1.4; squared error0.16, consistent with convex squared risk.
- Blending each mechanism equally gives M=1.45 and Y mechanism1.25M. Composition gives1.8125; squared error0.66015625, worse than both complete predictors.

Thus a per-mechanism mixture can lose beneficial cross-head cancellation and need not inherit the terminal-mixture risk identity. This is an exact counterexample, not a proposed training trick. It also warns against interpreting cancellation as correctly identified mechanisms. Preserve both measured-parent and composed-target evaluation.

## Consequence for the next prototype

The pilot's TabPFN standalone loss is insufficient to justify either unconditional replacement or a claim that a mixture cannot help. A sensible restricted next method would retain numerical grammar, add the pretrained predictor as an optional alternative, and choose a small finite weight set using only supported training histories. It must then score the resulting *composed model* on fresh held-out worlds/actions; it cannot substitute local validation success for composed benefit.

Two different proposed treatments should remain separate:

1. **Terminal ensemble:** convexly blend full composed forecasts. This has the exact fixed-predictor identity above, but the blend is an output ensemble rather than a newly identified set of local mechanisms.
2. **Mechanism mixture:** blend eligible natural mechanisms and compose the blended graph. This retains an explicit SCM-style factorization but must be evaluated for propagation and unchanged-mechanism harm; convex local mixtures supply no automatic terminal improvement.

A cheap training-only numerical selector should remain in both treatments. Charge the pretrained inference needed to form every proposal or validation prediction. Language proposals remain a separate typed-interface experiment because the first pilot's six raw outputs were all invalid. No new execution is authorized by the algebra alone; the next experimental stage still requires its own frozen protocol within the approved envelope.

## Evidence separation

The empirical statement is only that the fixed TabPFN adapter was worse than the correctly specified numerical grammar in five of six composed pilot tasks. No residual covariance, optimal mixture weight or mixture gain has been estimated from those private outcomes here. The identities tell us what a new training-only selection mechanism would need to estimate; they do not certify it has done so.

## Review

An independent bounded mathematical review found zero required corrections to the identity, derivative, constrained optimum, exact counterexample or limitations. The rational counterexample was checked without model execution:169/256>16/25. This review does not revalidate empirical pilot claims; those have a separate saved results review.
