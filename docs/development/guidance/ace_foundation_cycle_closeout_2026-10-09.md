# ACE foundation-model / SCM cycle: research recommendation

Author: Codex. October 9, 2026. The authorized cycle ended at **22:00 UTC**. Closeout began on the 22:24 UTC wakeup; the automation is **paused**. No new experiment or allocation was started during closeout. Further execution requires new user direction.

## Recommendation

**Keep a compact numerical SCM as the reference, and treat pretrained mechanisms as alternatives whose inclusion must be justified on independent validation.** The next useful experiment concerns reliable selection, before an acquisition-efficiency comparison. The evidence supports conditional predictive benefits; it does not support replacing every mechanism with a foundation model.

Three separately frozen development screens completed: 30, 144 and 216 cells. These are method/world cells, not 390 independent systems. Each screen has six base systems; the latter two contain four paired variants per system. All comparisons below concern the known chain X → M → Y and fixed shared histories. They are descriptive findings, not confirmation or evidence of fewer interventions.

## What the experiments establish

**A suitable compact prior is a strong baseline.** In the [component screen](ace_foundation_component_results_2026-10-09.md), the numerical grammar contained the true mechanism families. TabPFN's composed-target NMSE geometric paired ratio to grammar was **2.64323**, with higher error in five of six systems. All six language proposals were invalid and used grammar fallback. Matching grammar through fallback establishes no language benefit; the failed proposals remain preserved.

**Broader predictors can recover missing functions.** In the [mismatch screen](ace_foundation_mismatch_results_2026-10-09.md), TabPFN/Grammar32 ratios were **0.015875** when the M family was omitted and **0.014407** when Y was omitted, with six wins in each stratum. Its null ratio was **1.052381**, and its coefficient-change ratio **3.591699**, with six losses in the latter stratum. The large improvements occur against a deliberately inadequate grammar. They motivate a broader candidate class, without isolating a benefit of pretraining.

**The numerical flexibility control changes the interpretation.** In the fresh [retention screen](ace_foundation_retention_results_2026-10-09.md), both fixed RBF regression and TabPFN beat inadequate grammar in all six systems of both missing-family strata. TabPFN/RBF composed ratios, in null / coefficient-M / missing-M / missing-Y order, were **0.666338 / 0.561791 / 1.696999 / 0.472055**. TabPFN won all six missing-Y comparisons but lost five of six missing-M comparisons. This is a promising conditional advantage for the tested pipeline, not a general or causally isolated pretraining effect. Same fitting eligibility does not make the models otherwise equivalent.

**Good candidates do not ensure a good selector.** Combined selection with versus without PFN had ratios **0.997629 / 1.153268 / 1.382026 / 0.815129** in those four strata. Adding PFN helped three and tied three missing-Y systems, but harmed three and tied three missing-M systems. Combined selection lost to Grammar32 in five of six null systems and all six coefficient-change systems. All four selector ablations produced identical composed forecasts in missing-Y; that gain cannot be credited to their guards.

**Target accuracy and mechanism preservation are different objectives.** Combined selection still worsened unchanged-Y local error relative to retention in four of six null systems and two of six coefficient-change systems. Local-Y probes include extrapolation; they do not describe every root-reachable deployment law. All local-M probes were inside the fitting interval, so an outside-M protection effect was not tested. A retained downstream function can also receive different predicted parents after an upstream update.

Ratios here are geometric means of within-system NMSE ratios; lower is better. They are not ratios of pooled errors, universal wins or significance claims. Complete reports retain arithmetic summaries, absolute errors, every method, all strata, failures and local harms. The existing interactive component, mismatch and retention canvases remain the detailed visual companions; no new deck or canvas replaces them at closeout.

## Theoretical synthesis and methodological direction

The reviewed [mixture analysis](ace_foundation_mixture_theory_2026-10-09.md) separates complete-predictor averaging from averaging individual mechanisms: nonlinear composition and discarded cross-model dependence can make those operations disagree. This motivates preserving complete candidate SCMs and evaluating their composed behavior explicitly. It does not establish an acquisition advantage.

The reviewed [selection-reliability analysis](ace_foundation_selection_reliability_2026-10-09.md) connects finite validation selection to Cawley–Talbot and Learn-then-Test. A larger candidate set can reduce observed calibration loss while worsening new-observation risk. Small calibration size and noisy-calibration versus noise-disabled-target mismatch are plausible contributors to the pilot failures, not identified causes.

Its proposed simultaneous bounded-loss rule conditions on fixed fitted candidates and IID validation under declared laws. It bounds the joint probability of choosing a violating update, not the probability of harm conditional on updating, raw MSE, structural error or unconditional expected gain. With the specified multiplicity, eight validation observations cannot approve a nonidentity update at all. Permanent fallback must therefore be reported as potential adaptation failure, not advertised as success.

The reviewed [fixed scalar mechanism adapter](ace_foundation_fixed_mechanisms_2026-10-09.md) supplies a possible fixed-function representation: seal teacher predictions on a fit-only grid before validation, then evaluate copied interpolation tables. Its arithmetic fixtures establish scoped implementation properties. It has **not** been qualified with real pretrained teachers, and changes the predictor being evaluated. Conditional interpolation error bounds require genuine smoothness and curvature assumptions; they are not certified PFN bounds. The one-parent construction does not automatically scale to larger parent sets.

These are complementary pieces: candidate flexibility, explicit composition, independent selection and fixed prediction semantics. None retroactively certifies the completed empirical studies.

## Next cycle, only if newly authorized

1. **Qualify the proposed representation on artificial systems.** Measure actual teacher/table discrepancy, boundary behavior and resource cost for every provider under explicit query context. Authenticate source, model, runtime, fit eligibility and the seal before validation. Keep all failure attempts. No scientific private outcomes may tune the grid, thresholds or methods.
2. **Implement and independently review the fresh validation-budget protocol.** The reserved 94000–94005 seeds remain ungenerated; the 312-cell proposal is unfrozen and unallocated. Adopt exact table-prefixed candidate identities explicitly. Bind independent random streams, prefix isolation, all response charges, failed-candidate rules and an independent saved-prediction reporter. Compare empirical versus simultaneous selection and with-PFN versus no-PFN; preserve the numerical and retained references. Report bounded deployment loss alongside raw/structural error, local harm, clipping and fallback rates. Accept negative results without raising the budget until a rule passes.
3. **Then test acquisition separately.** Cross fixed estimators with random, coverage and uncertainty acquisition under identical legal menus, charged budgets and independent stopping/evaluation. Measure learning curves or interventions-to-target directly. Better prediction on fixed histories is insufficient evidence of intervention savings.

Do not hardwire node-specific winners from these private results, scale models merely to consume budget, or revive the failed language interface without new artificial output-contract qualification. No immediate new fitted study is launched by this recommendation.

## Attempts, resources and operational disposition

The [closeout reconciliation](../../../results/ace_foundation_cycle_closeout_20261009/reconciliation.json) matches nine local model-attempt terminal receipts to exclusive originals: four compatibility attempts, three scientific pilots and two artificial integration attempts. All nine have terminal process status complete/exit0. This status does not make the superseded compatibility predecessors scientifically qualified or turn invalid language proposals into successes. Preparation02 was not an allocation; preserved development failures and source corrections remain evidence outside these nine model attempts.

Measured totals are **90.044439 child CPU seconds** and **0.108012 supervisor CPU seconds**, with **zero GPU use**. Preparation, unit tests, reviews, reporting, hashing, Git and closeout are partly unmetered and excluded. Whole-sprint CPU is unknown. Conservative reservations are **7,920 / 28,800 CPU seconds**, including a preparation allowance; reservation is not measured consumption. Overlapping phase timers are not additive, and unused reservation is not authorization to extend the cycle.

The shared local controller lock was acquired without blocking and released without launch; the process snapshot found no matching local scientific Python worker/supervisor. The external attempt inventory matches the nine terminal receipts. The final shared SSH check returned255 because the control socket was absent, so no fresh remote scheduler state is claimed. No remote submission, cancellation, reconnect attempt or environment change occurred during closeout.

Accepted original A/B/C remain immutable. Separately authorized delivery replay06 job33640933 retains its earlier failed disposition; its one-attempt approval is consumed. It is outside the nine local foundation-model attempts and has no new scheduler verification here. No retry, qualified Stage B export or publication follows. Presentation v6 and its timed ten-minute script remain complete. The remaining paper reproduction and human submission gates remain open.

The hourly foundation synthesis automation is paused at this closeout. Any future execution requires a new scope, resource approval and deadline; the expired envelope does not roll forward.
