# Prospective prior-mismatch and recovery screen

Status: runner and reporting implementation prepared; review and technical qualification are required before freeze. **Not frozen, generated, fitted or scored.** This screen addresses a concern exposed by component pilot01: the compact grammar contained every generating family, so a broader pretrained prior had little representational advantage to offer. The completed pilot remains unchanged. This is a new development stage; no confirmatory or acquisition-efficiency claim follows.

## Question

Can a pretrained alternative earn predictive weight when the numerical grammar misses a mechanism, while preserving accuracy when the grammar is adequate and on unchanged mechanisms? Does selecting mixtures on composed validation forecasts help compared with blending local mechanisms? Compare entire pipelines at the same charged response budget; do not claim identical supervision use or identical compute.

## Systems and four paired variants

Use six fresh base seeds92000–92005, known X→M→Y, fully observed, independent Gaussian disturbances SD0.05. Use the component generator's fixed coefficient ranges and family cycle, but these new seeds. Freeze the new generator implementation before calling it. Each base world supplies one shared pre-change history; then create four variants using the same base coefficients:

1. **Null:** no mechanism changes.
2. **Coefficient change:** M's intercept increases by0.2 and every non-intercept M coefficient is multiplied by1.5; Y stays unchanged.
3. **Missing M family:** replace M with `old_intercept + old_linear_coefficient * sin(pi * X)`; Y stays unchanged.
4. **Missing Y family:** retain M and replace Y with `old_intercept + old_linear_coefficient * sin(pi * M)`.

Declare legal intervention domains X∈[-1,1] and M∈[-2,2]. The training intervention menu is the narrower fixed set stated below; legal actions outside that menu are not thereby observed training support. Natural M values are not clipped to its intervention domain.

There is no result-dependent family/frequency/seed choice. Sine is absent from the declared numerical grammar. This creates an explicit model-class mismatch but does not establish that sine was absent from TabPFN pretraining. All methods know the phase boundary; none receives the changed node or variant label. **No change-detection claim.** Variants are paired within base seed, not24independent sampled systems. Report each six-world stratum separately, with no favorable pooling or omission.

## Shared histories and selection split

Each pre- or post-change history contains32responses:8observational root draws,12root interventions cycling−1,−0.5,0,0.5,1, and12internal M interventions on the same menu with independently observed roots. Exact RNG construction is `default_rng(SeedSequence([seed,stream]))`: coefficients stream0; shared pre-change history stream1; post-change histories stream10+variant_index in the listed variant order; private composed-root, local-M and local-Y probes streams100,101,102. History draws are observational roots, internal-intervention roots,32M disturbances,32Y disturbances in that order. Coefficients follow the original intercept/linear/quadratic draw order. Private parent coordinates are shared across variants; post-change training randomness is separate by variant. The six pre-change histories are generated once and reused by all four variants; post-change histories are shared across every method within a variant. This is192pre-change plus768post-change responses,960total training responses, not multiplied by the number of methods. No extra environment calls during weight selection.

For the24-row fitting split, use zero-based indices0–5,8–16,20–28. Reserve indices6,7,17–19,29–31for calibration:2observational,3root intervention,3internal intervention rows. M has15eligible fit rows; Y has24. The full-data grammar control has20eligible M rows and32Y rows. Fit with measured parents and exclude clamped natural-mechanism labels exactly as before.

Mixture selection uses only the **five calibration rows where M is natural** (observational or X intervention), with measured root X and noisy observed Y. The three M-clamped calibration rows cannot be silently scored as root-action composed forecasts. Keep them in the paid-response ledger; this design reserves them but does not use them for root-only weight selection. No optimization or refit occurs after mixture selection: the two base experts remain fitted to24rows. That prevents changing the predictor to which a chosen weight applies. Grammar32 is the comparator that uses all eligible paid labels.

## Fixed methods

Use the existing pinned TabPFN-v2 model/runtime/configuration, CPU and one estimator, and the existing three-family numerical grammar selector. The numerical selector's internal75/25training split and ridge remain as previously implemented; its source pin is required. No new model version or weight sweep.

Report these six methods in every post-change variant:

- **Grammar32:** numerical grammar fitted to all32post-change rows. Main reference.
- **Grammar24:** same learner fitted to the24-row split. Selection-cost diagnostic.
- **PFN24:** TabPFN mechanisms fitted to the same24-row split.
- **Terminal mixture24:** blend complete Grammar24 and PFN24 composed forecasts; choose λ from0,0.25,0.5,0.75,1 by calibration Y MSE. Exact ties choose the first listed weight. Local diagnostics use the same λ to blend observed-parent predictions, but **the composed forecast is the mixture of complete forecasts**, not the composition of those diagnostic local mixtures.
- **Mechanism mixture24:** choose `(λM,λY)` from the same ordered5×5grid by calibration composed Y MSE. Compose mixed M with mixed Y. The order is lexicographic, λM outer, λY inner; exact ties favor smaller weights. Every candidate prediction is charged as inference. No substitution of measured M into a supposedly composed calibration score.
- **Pre-change grammar24:** retain the24-row pre-change predictor without adaptation. A reference for no-update performance, not a claim of successful recovery or permitted abstention.

Let gM,gY denote Grammar24 heads and pM,pY PFN24 heads. Every λ is the **PFN weight**:0 selects grammar and1 selects PFN. The forecast definitions are

- Terminal: `tλ(x)=(1−λ)gY(gM(x))+λpY(pM(x))`.
- Terminal local diagnostics: `dMλ(x)=(1−λ)gM(x)+λpM(x)` and `dYλ(m)=(1−λ)gY(m)+λpY(m)`. These diagnostic heads do not constitute the terminal predictor's mechanistic decomposition.
- Mechanism mixture: `hM(x)=(1−λM)gM(x)+λM pM(x)` and `hY(m)=(1−λY)gY(m)+λY pY(m)`, composed as `hY(hM(x))`.

Calibration MSE is the average squared residual of the respective composed forecast over exactly the five natural-M calibration rows. The specified first-weight tie order therefore favors grammar. Local mechanism-mixture diagnostics evaluate hM and hY directly on the declared parent probes.

No language arm is included. Its previous six invalid requests remain failures. A future language treatment must first qualify a constrained output interface on separate artificial inputs and receive its own frozen protocol; it cannot be inserted into this screen after seeing outcomes.

## Private evaluation and harm reporting

For each variant, evaluate256common noise-disabled root interventions on[-1,1] for composed Y. Separately evaluate256common M-parent probes X∈[-1,1] and256common Y-parent probes M∈[-2,2], with fixed draws shared across methods and variants within seed. The Y probes explicitly include possible extrapolation; do not call all probe support observed or reachable through root actions. Use legal internal interventions to create those Y probes; all parents are observed. This is768private evaluation responses per variant,18432total; keep them separate from the960training responses and do not use them in selection.

For structural functions fM,fY, the primary composed estimand is the **noise-disabled response** `y0(x)=fY(fM(x))`. Local targets are `fM(x)` and `fY(m)` on their respective probes. No natural upstream disturbance is integrated into those targets. Noisy calibration responses instead have conditional mean `E[fY(fM(x)+εM)]`, since independent εY has zero mean. For nonlinear fY this generally differs from y0; for quadratic coefficient c the gap is `c×0.05²`. Calibration MSE is consequently a fixed **selection surrogate**, not an unbiased estimate of private noise-disabled risk. Its selection can fail through this objective mismatch as well as sampling error or inadequate priors. This screen compares the specified pipelines and cannot isolate those causes. No extra noise-averaging responses or endpoint changes are authorized by this clarification.

Record M-local, Y-local and composed-Y absolute MSE and NMSE. Normalize every method by the full32post-change eligible training-label variance for the corresponding node, floor1e−12with indicator. Use the same normalizer for pre-change reference predictions; do not compare ratios normalized on different phases. On a fixed private parent probe set, compare each adapted method with the pre-change reference on the unchanged mechanism, so upstream input-distribution movement cannot masquerade as a local mechanism change. Retain the changed-mechanism result and composed outcome separately. The null variant measures unnecessary adaptation harm.

There are144planned post-change method×variant×seed cells. Report all cells and all four six-world strata. Main ratios are candidate NMSE / Grammar32 NMSE, lower is better; report per-world values, arithmetic means of paired ratios and geometric means. A missing/failed/zero/nonfinite pair makes that six-world aggregate undefined; do not drop the world or add a ratio floor. Also report adapted-minus-pre-change differences for the two local probe MSEs; positive means harm. Divide that same difference by the declared common variance for normalized harm. Different variant denominators do not establish a common absolute harm scale across strata. No hypothesis tests and no best-stratum headline. Report selected weights and all calibration scores; zero weight is an observed selection, not a failed cell. All failures and unattempted cells remain in the ledger.

## Gates and resources

No scientific execution until generator, adapters, evaluator, supervisor and reporter are implemented, independently reviewed for intervention/phase leakage and source-bound in a new freeze with model/runtime/protocol pins. Existing successful component compatibility receipts can inform resource sizing but cannot authenticate new worker bytes. A bounded artificial integration fixture exercises all four variants/six methods on seed123456 with fixed linear M coefficients[0.1,0.8] and quadratic Y coefficients[0.2,1.1,0.3], with otherwise identical histories/probes. It must never call a scientific base seed. Its160training/3072private artificial responses are accounted separately from the scientific960/18432. Fabricated stub-model unit checks precede this actual-runtime fixture; they are not model qualification. Preserve failed preparations.

Proposed maximum:one CPU process,15minutes elapsed/900childCPU seconds, no GPU or installs. A successful120-second artificial fixture must first demonstrate peakRSS≤6GiB; this is a sizing gate, not a claimed OS-enforced memory limit. Both modes have whole-child CPU/wall supervision and an absolute22:00UTC stop. The scientific launch must independently pin a successful fixture with identical source/model/runtime/protocol. Shared controller lock and exclusive attempt directories prevent concurrent duplicate launches; incomplete attempts require reconciliation, never automatic rerun. This is justified as a conservative cap from the completed component pilot's14.53CPU seconds for6worlds, but must be checked against an artificial integration run; it is not a runtime promise. An additional120-second technical fixture plus900-second pilot would bring the current5880-second conservative reservation to6900seconds, within28800. Neither new allocation nor fit has occurred. Retain prior28.436977childCPU seconds separately from reservations and unmetered preparation; do not infer a complete sprint total.

The screen tests one fixed recovery history, not a recovery-time curve, unknown change detection or intervention efficiency. Only a concrete benefit after checking unchanged-mechanism harm would motivate a subsequent crossed estimator×acquisition protocol with separate random, coverage and variance baselines. An unfavorable outcome stays unfavorable; do not add favorable families or increase compute to rescue it.

## Implemented boundary and attempt records

The runner records every training or evaluation response block as reserved before generation and returned after persistence. If interrupted between those records, returned count for that block is unknown, not zero. Pre-change predictors are fitted once and reused across variants; post-change Grammar24/PFN24 experts are fitted once and reused by both mixture methods. This reduces repeated computation without changing a compared predictor. All experts remain fixed after selection; prediction counters include calibration and private inference and pre-change counters are cumulative across variants. No scientific fit/model from an accepted study is reused or modified.

The learner-facing constructor receives only returned training rows and fixed learner configuration. The evaluator owns variant identity/truth; private probe generation occurs only after a durable selection seal. This is an audited in-process interface boundary, not adversarial process isolation. The worker uses authenticated captured helper source; inherited grammar and TabPFN configuration remain unchanged. Each planned method gets a terminal disposition, including dependent failure if its expert failed. Resources for shared fits/selection, per-cell evaluation and the variant are overlapping scopes; do not add them. The supervisor's whole-child CPU is the attempt cost. Reporting independently reconstructs saved errors and validates cell/response/selection closure without fitting models or generating new scientific responses.
