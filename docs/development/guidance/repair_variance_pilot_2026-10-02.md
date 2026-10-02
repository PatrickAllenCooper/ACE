# Frozen independent-block variance pilot

One fresh three-node, one-motif system at seed 830217, using existing connected_motif sampler. This is engineering/variance development, not a synthetic confirmation sweep. Two evaluator cases: unchanged conditional mechanism and added .85*tanh(1.7*x1+.8*x2). Same learner code, no change labels supplied to fit or gate.

Source: 32 observational rows, Gaussian prior mean zero/covariance identity, noise .15. Target fitting: 16 independent uniformly selected pair actions from the four {-2,+2} corners, four responses per action (64 rows). Fixed RBF width1.5, precision10; no model search. Compare posterior linear warm baseline with RBF residual fit. Diagnostic: 128 independent randomized pair-action blocks, four fresh responses each. All predictors fixed before diagnostics. Noise and action RNG streams separate from fitting. For each block compare mean min(squared error,1) losses; scale1 fixed now. Report raw MSE separately. Gate K=2, alpha=.05, using existing Hoeffding implementation. Record block variance, acceptance, counts, and raw arrays. No new sample size will be chosen by repeatedly peeking at this pilot.

Total generated rows: 32 source + 2*(64 fit +512 diagnostic)=1184. Diagnostic distribution is the four-corner menu only, not arbitrary parent support. One system/two cases cannot verify population safety. No outside-support fallback or identification of unknown change locations is tested. Two-minute local CPU cap; no GPU. Stop after the fixed pilot regardless of outcome.

## Outcome and explanation

The fixed pilot generated exactly 1184 rows in ~.018 seconds. Both cases abstained. Candidate-minus-baseline mean loss was +.000281 unchanged and +.000173 changed; raw errors likewise show no candidate advantage. These results do not motivate relaxing the gate.

The action menu explains the lack of a repair task. Any odd function h on the four corners (±2,±2), satisfying h(-x,-y)=-h(x,y), is representable by ax+by there. Write u=h(2,2),v=h(2,-2); choose a=(u+v)/4 and b=(u-v)/4. Then all four values match. The chosen tanh change is odd, so although globally nonlinear it is exactly representable by the baseline on this diagnostic distribution. This is a support-identifiability failure, not evidence that the change is globally linear or that repair succeeds.

Stop this pilot. Before another experiment, choose an independently justified action distribution that distinguishes the competing mechanisms (e.g. interior points in addition to corners), verify representational identifiability analytically, and freeze the primary distribution before observing scores. This must remain a bounded mechanism diagnostic, not another favorable synthetic confirmation sweep. The conservative gate remains unchanged.
