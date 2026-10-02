# Frozen bounded CPU prediction screen

Freeze before outcome scoring. Dataset: verified lt_malus_v1 archive SHA-256 584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9. Engineering/development screen on one apparatus, not confirmation across independent systems.

Use white_64 only for this bounded first screen. Predict vis_3 using commanded pol_1 and pol_2 only. No measured angles or other downstream features. Finite outcomes and positive training variance required; otherwise stop. No outcome-dependent exclusions. Keep every recorded row.

Divide each commanded angle into six 30-degree regions over [-90,90). For region indices i,j, test blocks satisfy (i+2*j)%5==0; all other blocks train. Identical pairs therefore stay together. Holdouts mix interior/exterior blocks; report this as blocked-angle prediction, not pure extrapolation or universal compositional generalization. Other eleven conditions remain unscored.

Fixed least-squares controls, no tuning: constant; additive Fourier basis with sin/cos harmonics 2 and 4 of each angle; full tensor product of the two five-dimensional Fourier bases; physical basis [1, cos(theta1)^2*cos(theta2-theta1)^2]. Angles in radians. Fit coefficients on training data only. Record matrix rank, row counts, MSE divided by training outcome variance, and exact prediction hashes. Physics basis is explicit privileged domain knowledge, not a generic learner. Do not use test errors to tune basis, regularization or split.

This screen diagnoses available headroom. It authorizes no neural scaling: a low-error physical control calls for studying robustness/transfer only with a separate frozen task. Keep negative results. Resource cap: one local CPU process, two minutes. No new physical query or model API call.
