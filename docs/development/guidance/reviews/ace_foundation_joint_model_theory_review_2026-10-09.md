# Joint-model averaging: bounded theory review

Date: 2026-10-09.

Reviewed document: `docs/development/guidance/ace_foundation_mixture_theory_2026-10-09.md`.

Exact whole-file SHA-256: `dc57de443328202c68fb95cc240057b534b51e01530ad70330753b69f85199ab`.

Reading scope was restricted to the newly appended “Coherent model averaging preserves dependence between mechanisms” section (lines 47–59), plus the existing two-head counterexample (lines 17–26) needed to interpret its numbers. Hashing the whole file does not extend the substantive review to its other sections. The previous mixture/empirical reviews were not repeated.

Primary reference checked: Hoeting, Madigan, Raftery and Volinsky (1999), *Bayesian Model Averaging: A Tutorial*, Statistical Science 14(4), equation (1), printed page 383 (PDF page 2). [Original article](https://www.stat.colostate.edu/~jah/papers/statsci.pdf).

## Required findings

None in the requested theory scope.

## Basis and qualifications

The attribution is accurate: equation (1) combines model-conditional posterior distributions with posterior model probabilities. Applying that decomposition to an interventional prediction is the document's own conditional SCM construction; the text correctly distinguishes this application from a causal result established by Hoeting et al. [Equation (1) and its surrounding definitions](https://www.stat.colostate.edu/~jah/papers/statsci.pdf#page=2).

For a fixed intervention chosen from the already conditioned-on history D, before its outcome is observed, retaining weights p(z | D) is appropriate: that design choice supplies no additional model evidence beyond D. Each component must itself define the interventional law and integrate its remaining parameter and disturbance uncertainty. The appended section states these requirements, including the absence of an identification or truth-in-model-class guarantee.

The linear argument is correct for posterior mean predictions at fixed X. With equal weights on coefficient pairs (a,b)=(0.9,2) and (2,0.5), E[ab]=1.4, E[a]=1.45, E[b]=1.25, and E[a]E[b]=1.8125. Hence Cov(a,b)=−0.4125 and the mean-prediction difference is Cov(a,b)X. This is dependence across the complete models' paired coefficients; it is not a claim about dependence between structural disturbances. The earlier counterexample supplies exactly these predictions at X=1.

Drawing one model index and retaining its joint parameter realization through propagation preserves the represented cross-head dependence. Independently averaged heads generally do not. The nonlinear qualification is appropriate: the linear covariance identity is not a general correction for nonlinear composition, and a full predictive distribution is not specified by its mean alone.

The section adequately distinguishes posterior model weights from calibration-selected predictive weights, and marginal pretrained uncertainty from a joint SCM posterior. A separately calibrated ensemble can be a predictive construction without acquiring a Bayesian-posterior interpretation. The text makes no claim that the current selector implements such a posterior or that this algebra establishes acquisition gains.

## Work performed

Static reading and analytical algebra check only. No code edits, models, experiments, private outcomes, data analysis, or numerical runs. Only this separate review was written. This does not review a future likelihood, calibration procedure, model-assembly algorithm, or acquisition implementation.
