# ACE: theory, an intuitive example and evidence

Date: October 8, 2026. Prepared by Codex for Patrick.

## Purpose and format

A ten-slide, editable presentation for an interested general audience with light technical background. Slides 1–4 introduce theory. Slides 5–6 use a worked example. Slides 7–10 present the strongest supported findings and a consequential boundary. A restrained 16:9 research design uses large type, diagrams and editable evidence charts. Speaker notes carry derivations, source references and reporting details.

The presentation selects strong results for explanation, not a new scientific success criterion. It keeps failed primary gates, the worsening delivery history and the physical comparator result visible. It reports no performance from prospective Stage B while its supplemental numerical replay remains unqualified.

## Slide sequence

1. **ACE and the prediction we will use.** Organizing theory: an experiment has value when its usable observations can reduce error in the deployed prediction. Introduce acquisition and final fitting as distinct choices. The tagline, “Make every experiment count,” expresses an aspiration.
2. **Information and fitting effort.** Under a correctly specified linear Gaussian model, eligible observations reduce posterior prediction variance. A neural optimizer need not realize that benefit. Final fitting can reuse saved observations, with an explicit computation cost.
3. **Error propagation through a causal model.** In a two-stage linear chain, output error equals the learned downstream slope times intermediate error plus its residual at the true intermediate value. Signed errors can amplify or cancel. This does not diagnose any particular ACE failure.
4. **The value of an experiment depends on the target.** A linear Gaussian observation reduces target variance by a squared target–observation covariance divided by observation variance. Show the intuition without claiming the nonlinear ACE policy inherits optimality.
5. **A two-stage heater.** Illustrative, dimensionless control input u sets an intermediate response M = 2u. Final response Y = 3M. At u = 1, the true response is 6. This is a toy calculation, not a laboratory experiment.
6. **A small intermediate error changes the final output.** An intermediate prediction of 2.2 produces final prediction 6.6 even with a perfect downstream rule. Measured-parent fitting can hide that composed error. Reveal the arithmetic step by step in speaker notes.
7. **Primary gains in two synthetic families.** Show the separate 40-system confirmation against systematic coverage, including paired differences, 95% intervals and Holm-adjusted p-values. Both primary gates passed. Retain known-graph, fixed-learner and unresolved scoring-ablation limits. These results are not pooled with the shifted-mechanism study.
8. **Final fitting improved 11 of 12 histories.** Show the accepted geometric exact-level error ratio 0.183 and its 95% interval [0.102, 0.327]. Display the retained worsening history, ratio 2.015. State one deterministic emulator, exposed grid, median of three scored refit initializations and unequal fitting computation.
9. **An uncertainty policy beat a specified random policy.** Show all four arithmetic mean errors in the 20-system shifted-mechanism study. Display 19/20 wins and the descriptive 64.85% lower group mean error against random. Explicitly retain secondary comparison status and the failed primary coverage comparison. Do not call this a unique scoring-formula effect or equal-compute result.
10. **Simple structure can be the strongest model.** Show physical delivery wins of 7/11 against rolling, 3/11 against physics regression and 0/11 against Fourier regression. These are conditions of one apparatus. Conclude with the proposed direction: fit for the deployed target, compare strong simple models and use reserved validation. Broader superiority remains open.

## Demonstration script

Ask the audience to predict the heater’s output at u = 1. Reveal M = 2, then Y = 6. Next keep the downstream rule exact but replace the intermediate estimate with 2.2. The output becomes 6.6. The audience can see why good local predictions alone may not yield the best final output.

Explain that the same recorded input and measured final output can supervise a composed prediction objective. They do not justify assigning the old output label to an invented intermediate intervention. A mathematical illustration motivates a method; an independent experiment must establish whether that method improves ACE.

## Source basis

- [Evidence assessment](/Users/pat/code/ACE/docs/ACE_evidence_and_public_claims_2026-10-08.md).
- [Theory and method proposals](/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md).
- [Accepted delivery claims and scope](/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json).
- [Shifted-mechanism completed study](/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md).
- [Frozen shifted-mechanism protocol](/Users/pat/code/ACE/docs/development/guidance/protocol_pev_shift30_mean_confirmation.json).
- [Graph provenance correction](/Users/pat/code/ACE/docs/development/guidance/erratum_shift_graph_provenance_2026-09-27.md).

Detailed evidence and primary literature appear in the relevant slide notes. No new fit, environmental response, hypothesis test or empirical result is needed to construct this presentation.
