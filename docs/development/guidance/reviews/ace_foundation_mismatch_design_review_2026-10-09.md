# Foundation mismatch protocol: independent design review

Date: October 9, 2026. Scope: [ace_foundation_mismatch_protocol_2026-10-09.md](/Users/pat/code/ACE/docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md) only. This is a prospective design review; no worlds, models, tests, fits or experiments were executed, and only this review was written.

**Disposition: two required clarifications before generator/adapter freeze.** Neither requires more worlds, methods, responses or compute.

## 1. Specify the mixture functions and weight direction

Location: **Fixed methods**, Terminal mixture24 and Mechanism mixture24.

The protocol distinguishes terminal blending from composing blended heads, but it does not define which expert receives λ. The grid alone does not resolve this: reversing its meaning also reverses the expert favored by the declared first-weight tie rule. The implementation and selected-weight interpretation therefore remain underdetermined.

Required correction: declare the equations, including the local diagnostics. For example, let `gM, gY` be Grammar24 heads and `pM, pY` be PFN24 heads, and define λ as PFN weight:

- Terminal forecast: `tλ(x) = (1−λ) gY(gM(x)) + λ pY(pM(x))`.
- Terminal local diagnostics: `dMλ(x) = (1−λ) gM(x) + λ pM(x)` and `dYλ(m) = (1−λ) gY(m) + λ pY(m)`.
- Mechanism mixture: `hM(x) = (1−λM) gM(x) + λM pM(x)`, `hY(m) = (1−λY) gY(m) + λY pY(m)`, with composed forecast `hY(hM(x))`.

Specify calibration scores as the mean squared residual over exactly the five declared natural-M calibration rows, using these same forecast functions. Then λ=0 means grammar, λ=1 means PFN, and the existing tie order favors grammar. Either weight convention is possible, but one must be fixed. The terminal local diagnostics describe blended head predictions; they are not the heads whose composition produces the terminal forecast and cannot be presented as its mechanistic decomposition.

## 2. Name the noise-free estimand and the noisy calibration surrogate

Location: **Shared histories and selection split**, **Private evaluation and harm reporting**.

Private evaluation disables disturbances, while selection uses noisy observed Y on natural-M rows. With a nonlinear downstream mechanism, these are different prediction targets, not simply the same target observed with independent zero-mean output noise.

For a variant with structural functions `fM, fY`, the declared noise-free composed target is

`y0(x) = fY(fM(x))`.

Under the stated natural disturbances, the conditional mean of a calibration response is instead

`E[Y | do(X=x)] = E[fY(fM(x) + εM)]`,

with zero-mean independent εY integrated out. This generally differs from `y0(x)`. For a quadratic Y mechanism with quadratic coefficient c, the difference is `c × 0.05²`; tanh and sine also generally introduce a gap. Consequently, a weight can improve the noisy calibration score while moving away from the private noise-free target, even before finite-sample selection error is considered.

Required correction: explicitly define the primary composed estimand as the noise-disabled structural response, if retaining the current design, and state that noisy calibration Y MSE is a **selection surrogate for that endpoint**, not an unbiased estimate of its private risk. The five-row calibration comparison then tests that fixed selection pipeline on the independently scored noise-free endpoint. A failure cannot isolate prior mismatch or the superiority of terminal versus mechanism blending independently of this objective mismatch. No new probes or responses are needed for this clarification.

Also distinguish the local estimands as `fM(x)` on the fixed X probes and `fY(m)` under legal M interventions. These local targets integrate no upstream natural disturbance; the root-action composed target and an actual noisy-intervention mean must not be conflated. Switching the primary endpoint to the latter would be a substantive design change, not a clarification of the existing noise-disabled evaluation.

## Accounting and validity points that are already specified adequately

- **144 cells:** six base worlds × four variants × six post-change methods. Variant pairing is within base seed; the four six-world strata are not 24 independent sampled systems.
- **960 training responses:** six shared pre-change histories ×32 =192, plus 24 post-change histories ×32 =768. Reusing the pre-change predictor across variants does not create additional paid responses.
- **18,432 private responses:** 24 variants × three separate 256-response evaluation sets. Fixed draws shared across variants do not require merging the distinct response evaluations. The response ledger must preserve this declared three-set accounting.
- The fit indices contain 24 rows and calibration indices eight, without overlap: M has 15 eligible fit labels and five natural-M calibration rows; Y has 24 eligible fit labels. The three reserved M-clamped calibration rows remain paid but excluded from root-only mixture selection. Grammar32 uses all 20/32 eligible labels. The protocol correctly disclaims identical supervision use and compute.
- Private outcomes and variant/changed-node labels are excluded from fitting and selection. The experts remain fixed after selecting weights; the no-update reference retains its pre-change predictor. Exact RNG streams still need to be pinned, as the protocol already requires, before generation.
- Using the same full post-change training-variance denominator for every method within an endpoint/variant avoids phase-dependent ratio comparisons. Local harm against the retained predictor should be read as the signed adapted-minus-retained MSE difference, and its normalized counterpart as that difference divided by the same declared variance (with its declared floor). Positive differences indicate harm. Different variant normalizers do not make cross-stratum normalized magnitudes a common absolute harm scale.
- Fixed parent probes prevent upstream input-distribution movement from being mislabeled as local mechanism change. The explicit Y-probe extrapolation limit and separate changed-head/composed outcomes are appropriate. The null variant compares unnecessary adaptation with retention; it is not a change-detection test.
- Private evaluation protects the reported comparisons from calibration selection optimism, including the larger 25-candidate mechanism grid. This remains a comparison of the specified pipelines, with different selection flexibility, rather than isolated architecture attribution. No additional sweep or significance test is required for this development screen.

After the two clarifications, the design can answer its bounded question: whether these fixed post-change prediction and mixture pipelines improve the declared private endpoint over Grammar32 while reporting local retention harm. It cannot establish causal identification, contamination-free foundation-model generalization, recovery time, unknown change detection or intervention efficiency.

## Scoped correction recheck — October 9, 2026

Reviewed protocol SHA-256: `92de4a08e7a933bedb69fe3c2993b327c84ddba4ee6111f6511496f5805e5318`.

**Disposition: both required findings closed; 0 remaining required issues in this recheck scope.** The original findings above are retained as review history.

- **Finding 1 closed:** λ is explicitly PFN weight, with endpoints 0=grammar and 1=PFN. Terminal forecasts, terminal diagnostic heads, mixed mechanism heads and their composition now have explicit equations. Calibration uses the respective composed predictor on exactly five natural-M rows. The declared grid/tie order remains consistent with the weight convention; terminal diagnostics are explicitly excluded from a mechanistic decomposition of the terminal predictor.
- **Finding 2 closed:** the primary target is explicitly `fY(fM(x))` with disturbances disabled, and local targets are `fM(x)` and `fY(m)`. The noisy calibration conditional mean, nonlinear discrepancy, quadratic gap `c×0.05²`, surrogate status and attribution limits are now stated correctly. The clarification preserves the planned endpoint and response budget.
- **Adjacent domain clarification accepted:** legal intervention domains X∈[−1,1] and M∈[−2,2] are distinguished from the narrower observed training menu. Natural M remains unclipped; its realized or composed values need not lie inside the legal M-intervention domain. This is consistent with the stated structural functions and with the separate support/extrapolation limits on parent probes.
- **Adjacent harm clarification accepted:** local harm is adapted-minus-pre-change MSE, positive for harm, with the same difference divided by the declared common, floored normalization denominator for normalized harm. The explicit warning about differing variant denominators prevents interpreting normalized differences as a common absolute harm scale across strata.

This recheck covered only the two findings and these adjacent clarifications. It did not repeat the broader design review, review implementation or authorize execution. No tests, models or scientific runs were executed; only this disposition was appended.
