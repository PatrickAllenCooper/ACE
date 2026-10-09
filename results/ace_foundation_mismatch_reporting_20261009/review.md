# Foundation mismatch results: independent review

Date: October 9, 2026. **Disposition: 0 required factual, numerical or scope corrections to the reviewed draft.**

## Scope and exact evidence

Reviewed [the results draft](/Users/pat/code/ACE/docs/development/guidance/ace_foundation_mismatch_results_2026-10-09.md), the frozen protocol, both saved freezes, fixture/scientific terminal and completion records, plans, preflight records, verified summaries, individual cell JSON, response journals, selection seals and resource records under [the repository results copy](/Users/pat/code/ACE/results/ace_foundation_mismatch_20261009). Independently calculated aggregates from saved cells using only Python's standard library. No model was loaded, no scientific responses were generated, and no fitting, research program or test was executed. Only this review was written.

Independently checked SHA-256:

- Results draft: `bc21696671f25b215f87553a6e208b2836172795940659a38e82c925dd7fdb8e`.
- Scientific freeze: `c867457ac7746cefc46a3e39404f52137ca198922fce6cba664a78137ca197d6`.
- Scientific terminal: `0ea6a1eecf73cc50acd5fcb5394c1f2aae32e8578710a093c349f68e1da3f6d3`.
- Scientific verified summary: `e8a6961679bdadd73ded1116e2da49c46bf0f09f3856187aec2b98e0512f70df`.
- Fixture freeze: `e09beef39ee26e58a174be41bc26d50ec8ffc477c50f4668c3539f154453657a`.
- Fixture terminal: `4a4cca6530ab1f548529b7598efd8a8ca1953fdd818801c213466ecac3508a19`.
- Fixture verified summary: `619e84aa3b8ea74f8deada9ebd780f7d83b14331a3ec6af7f6354f71cb59ce69`.
- Protocol: `4b3f59584f0f8b1ab15202c35b37b7fe2949374b85f9b5ae95e65e345ad8b025`.
- Custody manifest: `a8b7d966d97130ad28eef60237b29a0b953b4bcf1632224dea6920c34c47325a`.

All **809 listed local files**, totaling 3,040,717 bytes, match the custody manifest's sizes and hashes; membership is exact, excluding the manifest itself. Frozen source copies match their six source pins. Both freezes record revision `1dd4d93b26c1e2126931fa32802678b8c20f3983`, identical source/protocol pins, runtime and checkpoint identity. The scientific freeze's three fixture bindings match the copied fixture freeze, terminal and summary. Preflight records agree with the nine dependency versions, Python version and checkpoint hash. This checks saved provenance and local custody, without rereading checkpoint weights or independently observing runtime execution.

## Numerical headline and all four strata

The scientific plan, terminal, completion, summary and **144 individual cells** agree on the full ordered six-seed × four-variant × six-method matrix. Every cell is complete; none is omitted or replaced. Independently recomputed all **60 endpoint comparisons**, including each six-world paired ratio, arithmetic mean and geometric mean, and all **240 signed local-harm records**. They agree with the saved summary.

The PFN24 composed-Y ratios to Grammar32 are:

- Missing M family: geometric `0.01587462035062029`, arithmetic `0.018799095911653033`, **6/6 lower errors**.
- Missing Y family: geometric `0.014406637097847915`, arithmetic `0.017431068056643854`, **6/6 lower errors**.

`100 × (1 − geometric ratio)` gives **98.412538%** and **98.559336%**, matching the draft's rounded 98.4%/98.6%. These are reductions represented by geometric means of paired error ratios, not significance estimates, intervention savings or a common percentage reduction in each world.

All 20 displayed composed-Y geometric ratios match the draft. In its method order—Grammar24, PFN24, terminal24, mechanism24, prechange24—the values rounded to six decimals are:

- Null: `0.958028, 1.052381, 0.942609, 0.900716, 0.613099`.
- Coefficient M: `2.141619, 3.591699, 1.296435, 1.966393, 318.678426`.
- Missing M: `0.877368, 0.015875, 0.015875, 0.013304, 0.847947`.
- Missing Y: `0.876406, 0.014407, 0.016145, 0.009431, 0.989602`.

The unfavorable coefficient-change result is preserved: PFN24 is worse than Grammar32 in all six worlds. The null PFN arithmetic ratio `1.968568`, terminal ratio `1.179996` and mechanism ratio `1.175227` also match. The draft correctly distinguishes these arithmetic results from geometric improvements and makes no uniform-improvement claim.

## Weights, normalization and unchanged-head harm

Every saved selection equals its summary entry; all five/25 candidate grids and first-minimum tie choices validate. In missing-M worlds, terminal24 selects PFN weight 1 in all six cases, explaining its exact equality with PFN24. Mechanism24 also selects M weight 1 in all six, with Y weights `[1, 0, 0.5, 1, 0, 1]` in seed order. In missing-Y worlds, mechanism24 selects Y weight 1 throughout, with M weights `[1, 1, 0.75, 0.5, 0.5, 0]`; terminal weights are `[1, 1, 1, 1, 1, 0.75]`. These choices are calibration results, not private-outcome selections or mechanism-identification evidence.

All **432 endpoint records** have finite positive errors/variances and consistent MSE-to-NMSE arithmetic. Every method, including retention, uses the same recorded post-change normalizer within each seed/variant/endpoint. No floor activates; the minimum recorded eligible-label variance is approximately `0.120630`. Fit-row records are 20/32 for Grammar32 and 15/24 for the other fitted experts; retained family/coefficient records remain unchanged across variants.

Signed harm is adapted-minus-retained MSE on common local probes, with its normalized difference using the same floored denominator. The unchanged-head flags and all draft harm claims match:

- PFN unchanged Y: worse **6/6** in null, coefficient-M and missing-M strata; mean increases `0.1483520116`, `0.1265221904`, `0.1306726911`.
- Mechanism mixture unchanged Y: worse **6/6 null** and **5/6 coefficient-M**; mean increases `0.0672076924`, `0.0823043448`.
- Missing-M terminal diagnostic harm equals PFN's because its weight is 1.
- Missing-Y PFN unchanged M: worse **3/6**; mean increase `0.0002746119`.

These validate harm on the declared broad parent-probe distribution. They do not isolate error outside each learner's observed parent range, attribute all harm to extrapolation, or establish the same harm on every root-reachable distribution. The draft retains those distribution limits and correctly separates terminal diagnostic heads from a mechanistic decomposition.

## Responses, qualification and cost arithmetic

Saved reservation/return counts, array hashes and journal/selection timestamps close successfully:

- Scientific: **960 training + 18,432 private = 19,392** reserved and validated returned responses.
- Artificial fixture: **160 training + 3,072 private = 3,232**, separately accounted; **24/24** cells complete.

The fixture reports **5.443610 child CPU seconds**, **6.751167 elapsed seconds** and **729,710,592 bytes RSS**. The scientific terminal reports **23.050243 child CPU seconds**, **23.276169 elapsed seconds**, **735,477,760 bytes RSS** and **0.022578 supervisor CPU seconds**. These match the draft and are below the respective 120/900-second limits; fixture RSS is below the 6-GiB sizing gate. Both launch records specify one CPU thread and no GPU; both terminals record zero GPU seconds.

For the cumulative-cost statement only, read the four earlier compatibility terminal receipts and the first component terminal; no old tests or runs were repeated. Their totals are **28.436977 child CPU seconds** and **0.029621 supervisor CPU seconds**. Adding this fixture and scientific pilot gives **56.930830** and **0.063201**, respectively. The draft correctly excludes unmetered preparation/reporting/review costs and avoids summing overlapping timing scopes. The separate **6,900-second reservation** is consistent with 5,880 +120 +900 and remains below 28,800; it is not measured consumption.

## Scope disposition

The draft supports a conditional predictive advantage for this fixed pretrained alternative over a deliberately misspecified compact grammar. It explicitly leaves **pretraining versus representational flexibility unresolved**, keeps all unfavorable strata visible, identifies the noisy-calibration/noise-disabled-target mismatch, and limits mixture attribution given five versus 25 candidates. Wide-probe harm prevents a preservation/no-harm claim. Known graphs, shared fixed histories, paired variants and six-world development strata do not establish confirmation, real-world transfer, causal discovery or intervention efficiency.

The proposed flexible non-pretrained comparator and retention-aware future selector are labeled prospective and are not inserted into the completed study. No required correction to the reviewed draft was found; no draft or scientific source was edited.
