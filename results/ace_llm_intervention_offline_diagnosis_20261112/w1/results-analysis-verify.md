# Independent verification of the 24-system re-analysis (label: results-analysis-verify)

Prepared 2026-11-12. Read-only verification. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was modified. No model, experiment, SLURM or network activity. Only `/usr/bin/python3` offline analysis of saved CSV/JSON files.

## Scope and method

- Inputs (raw): `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/cells.csv` (192 runs), `curves.csv` (4,540 points with the 8 target predictions per point), `policies.csv`, `pairs.csv`, `actions.csv`, `strata.csv`.
- Cross-check source: the original per-run reports under `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque/cpu/cpu/worker/artifacts/campaign/private/` (`summary.json` and 192 `world-XX/arm-YY/cost-report.json`).
- Code read for definitions only: `/Users/pat/code/ACE/scripts/research/llm_intervention_screen_protocol.py` (lines 22-33 regimes/arms, 63-92 menu, queries, task, 145-152 per-arm noise bindings), `/Users/pat/code/ACE/scripts/research/llm_intervention_scm.py` (lines 82-85 intervened nodes are not updated, 121-130 composed prediction), `/Users/pat/code/ACE/scripts/research/llm_intervention_world.py` (lines 306-314 noise key includes the arm namespace).
- All verifier code is fresh and lives in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/` (`v00`..`v18` scripts, `log_*.txt` outputs, `out_*` intermediates). I did not open the earlier agent's `results-analysis/` directory or scripts.
- Short system names: `wNN` = world id `2026110701NN`.
- Every statistic here is descriptive, post hoc, development-data. No confirmatory claim.

## Bottom line

Almost every number claimed by the earlier agent reproduces exactly from the raw CSVs. I found no arithmetic error. Four statements need correction or qualification:

1. **"TR and local_info open with X at the farther extreme in 23/24 and 24/24 runs" is worded too strongly.** 23/24 and 24/24 hold only for "the first pure X-extreme action that appears anywhere in the run". The literal first action is the far pure X extreme in 17/24 (target risk) and 22/24 (local information) runs.
2. **The shuffle baselines (claimed 0.38 and 0.47) do not reproduce as a pair.** The agreement values reproduce (0.409 and 0.570). With action order shuffled inside each run I get baselines 0.337 and 0.478. With cell labels shuffled across systems I get 0.381 and 0.587. The claimed 0.38 matches only the label shuffle and the claimed 0.47 only the order shuffle. The qualitative conclusion (sequences are far from fixed) holds. Two refinements: step position carries a modest but real signal (order-shuffle p 0.003 and 0.0015), cell membership carries none (label-shuffle p 0.11 and 0.81), and target risk's action mix varies strongly between systems inside a cell.
3. **"An oracle at the floor would have had power 0.68" is a normal-theory number that understates detectability of a dominating policy.** The realized TR-minus-floor contrast gives exact sign-flip p = 0.0039, and a system bootstrap gives power 0.84 (t-test) to 0.93 (sign-flip). The formula is roughly calibrated for realistic two-sided noisy differences.
4. **"1 of 70 pairwise tests below 0.05" among the five weak arms cannot be reproduced exactly** because the seven endpoints were not specified. With my nine endpoints I get 0 of 90. Scanning fixed-cost endpoints at every cost 9..32 gives 1 of 240 (frozen vs supervised at total cost 12, p = 0.043). The substance (weak arms indistinguishable) is confirmed.

## A. Data integrity (confirmed)

Script `v01_consistency.py`, log `log_v01_consistency.txt`.

- Recomputed first crossing of 0.04, capped total and acquisition cost (cap 32), final error, ever/final attainment and actual total cost from `curves.csv` for all 192 runs. **0 of 192 inconsistent** with `cells.csv`.
- `policies.csv` reproduces (max abs difference 4.5e-17). `pairs.csv`: 28 of 28 rows reproduce (difference vectors, means, win/tie/loss; "wins" = left has lower cost). `actions.csv`: 38 of 38 rows reproduce.
- Totals: 4,348 acquisitions, 5,884 accepted responses (4,348 + 8 x 192), 6,097 cost units. Acquired actions per policy: random 487, coverage 480, local information 570, target risk 560, ranker 576, frozen 576, supervised 523, DPO 576.
- Extra check not requested: `cells.csv` equals the 192 rows of the original `summary.json` exactly (with round-trip float parsing), and all 4,540 `curves.csv` NMSE values equal the `curve` entries of the 192 original `cost-report.json` files exactly.
- `strata.csv` has 72 rows = 9 strata x 8 policies.

## B. Systems at target after initialization (confirmed)

Scripts `v02_init.py`, `v03_truth.py`.

- Initial state (total cost 8) is identical across the 8 arms inside every system (zero spread in NMSE and in all 8 predictions).
- **10 systems are at target after initialization: w00, w01, w02, w06, w07, w08, w09, w15, w18, w20.** All 8 policies have capped cost 8 on exactly these 10.
- By graph: chain 5/8, fork 5/8, collider 0/8. By coefficient law: matched 7/12, shifted 3/12. By cost menu: uniform 6/12, heterogeneous 4/12. (Also: low noise 4/12, high noise 6/12; matched_low 3/6, matched_high 4/6, shifted_low 1/6, shifted_high 2/6.)
- Initial NMSE: min 0.000284 (w06), median 0.044010, max 0.544896 (w19).
- "Three more start within 12.5 percent of the threshold": w14 0.041121 (+2.80%), w05 0.043019 (+7.55%), w04 0.045001 (+12.503%). True to rounding. Strictly, w04 is 12.503% above.
- Threshold sweep, systems at target after initialization: 3 at 0.01, 5 at 0.02, 10 at 0.04.
- True target means reconstructed per system by least squares from (prediction, NMSE) pairs. Max absolute NMSE residual 2.9e-15. Recovered truth satisfies the graph constraints (chain q6 = q7, fork q4 = q5 = q6 = q0 and q7 = q1).
- **All-zero prediction already meets 0.04 on 3 systems**: w02 (0.02319), w06 (0.02436), w18 (0.03914). All three are among the 10 at-target systems.
- New observation: on w02, w03 and w21 the initialized SCM is *worse* than the all-zero (prior mean) prediction.

## C. Headroom against target risk (confirmed)

Script `v04_headroom.py`.

| Quantity | Claimed | Verified |
| --- | ---: | ---: |
| Target-risk total capped cost | 279 | 279 (mean 11.625) |
| Best-of-8 envelope total | 230 | 230 (mean 9.5833) |
| Envelope saving vs target risk | 49 | 49 (2.0417 per system) |
| Saving on top five systems | 42 | 42 (w12 10, w23 10, w19 9, w16 7, w21 6) |
| Trivial floor (8 if at target, else 9) | 206 | 206 |
| Ceiling, target risk minus floor | 73, 3.04/system | 73, 3.0417/system |
| Systems where some arm beats target risk | 8 | 8 (w05, w11, w12, w16, w17, w19, w21, w23) |

- Target-risk acquisition cost on the 14 informative systems, sorted: 1,1,1,1,1,2,2,3,6,7,9,14,15,24. Confirmed. It never attains on w19 only.
- Arms that deliver the envelope: w12 ranker, w23 local information, w19 local information, w16 supervised LLM, w21 local information and frozen LLM, w11 random, w17 coverage/local information/supervised, w05 ranker. Systems where each arm strictly beats target risk: random 2, coverage 3, local information 5, ranker 4, frozen 4, supervised 2, DPO 2.
- Threshold sweep [at target after init / target risk never attains / target risk minus floor per system]: 0.01: 3 / 4 / 7.2917. 0.02: 5 / 1 / 4.7500. 0.04: 10 / 1 / 3.0417. Confirmed.

## D. Paired SD, power and sample sizes (confirmed, with caveats)

Scripts `v05_power.py`, `v14_power_extra.py`, `v16_bootstrap_power.py`.

Policy minus target risk, capped total cost, 24 systems (mean, SD, lower/tie/higher, exact two-sided sign-flip p):

| Policy | Mean | SD | -/0/+ | Sign-flip p | d = mean/SD |
| --- | ---: | ---: | --- | ---: | ---: |
| Random | +2.625 | 5.428 | 2/12/10 | 0.0112 | 0.484 |
| Coverage | +0.833 | 3.074 | 3/13/8 | 0.2256 | 0.271 |
| Local information | +0.333 | 6.211 | 5/13/6 | 0.8398 | 0.054 |
| Ranker | +1.958 | 6.630 | 4/14/6 | 0.1934 | 0.295 |
| Frozen LLM | +2.000 | 6.763 | 4/11/9 | 0.1841 | 0.296 |
| Supervised LLM | +3.000 | 6.058 | 2/12/10 | 0.0220 | 0.495 |
| DPO LLM | +3.542 | 6.434 | 2/13/9 | 0.0068 | 0.551 |

All match the claim. Holm over these 7 tests: DPO 0.0479, random 0.067, supervised 0.110. Confirmed.

- **Planning SD.** The "median paired SD 6.2" is the median of the seven policy-minus-target-risk SDs = 6.2113 (it is the local-information contrast). The median over all 28 pairs is 6.977 (range 3.074 to 9.035). The planning SD is therefore specific to contrasts against target risk and dominated by weak arms.
- **Minimum detectable difference at 24 systems** (paired t, two-sided 0.05, 80% power, noncentral t): 3.709. At 90% power 4.293. Confirmed, and above the 3.0417 ceiling.
- **Systems needed at SD 6.2113** (80% / 90%): 1 unit 305 / 408. 2 units 78 / 104. Noninferiority margin 1 with true difference 0 (one-sided 0.025) 305 / 408. All confirmed. (3.04 units: 35 / 46.)
- **Replicated design**, 8 replicates per arm, replicate share r of paired-difference variance, variance factor (1 - r) + r/8, 1 unit, 80%: r = 0.5: 173 at SD 6.2113 and 172 at SD 6.2. r = 0.8: 93. Claim "172 to 93" confirmed to rounding of the SD.
- **Log-area planning.** Median of the seven log-area SDs = 0.4781 (the ranker contrast). 10% lower NMSE (0.0458 log10 units): 859 / 1150. 20% lower: 193 / 258. Confirmed.
- **Oracle at the floor.** Target risk minus floor: mean 3.0417, SD 5.909. Noncentral-t power 0.676. Claim 0.68 confirmed as a formula value. Using SD 6.211 instead gives 0.632.

Caveats that matter for planning:

1. The oracle contrast is one-sided and zero-inflated (15 zeros, 9 positive differences). On the realized data its exact sign-flip p is 0.0039. Resampling systems with replacement (n = 24, 2,000 draws) gives power 0.84 with the t-test and 0.93 with the sign-flip test. So a policy that dominated target risk system by system would very probably have been detected. What the screen could not detect is a noisy policy with a better mean.
2. For realistic two-sided contrasts the formula is roughly calibrated. Bootstrap vs formula power at n = 24: random 0.76 vs 0.62, coverage 0.23 vs 0.25, ranker 0.22 vs 0.28, frozen 0.21 vs 0.28, supervised 0.72 vs 0.64, DPO 0.88 vs 0.73. A constructed mean-1 population reproduces the 305-system answer (bootstrap power 0.85 at n = 305).
3. Sample size is very sensitive to which SD is used. 1 unit at 80% power: 77 systems at SD 3.07 (the coverage contrast, the arm most like target risk), 305 at 6.21, 361 at 6.76, 384 at 6.98 (median of all pairs), 643 at 9.03.
4. Restricting to the 14 systems not at target after initialization: differences scale by 24/14 and the median SD is 7.53. The same underlying effect (1 unit averaged over the current mix = 1.714 units on informative systems) needs 154 informative systems instead of 305 mixed systems. Dropping at-target systems by design roughly halves the number of systems.
5. Replicate share is not measured for target risk. The only replicate evidence is the DPO/supervised twin pair at cap 18 (below). Variance of the twin difference is 7.82. That is 53% to 101% of the variance of six of the seven policy-minus-target-risk differences at cap 18, and 221% of the coverage contrast (so replicate noise is policy dependent). The assumed range 0.5 to 0.8 is plausible and may be low.

## E. Curve-area ranking and alternative endpoints (confirmed, ranking is definition-sensitive)

Scripts `v06_area.py`, `v07_endpoints.py`.

The claimed per-policy "log-area" values reproduce to within 0.0005 with this definition: mean of log10 NMSE on the integer total-cost grid 9..32 (24 points), step curve with last observation carried forward to 32.

| Policy | Capped cost | Sustained cost | Log-area | Geo-mean final NMSE |
| --- | ---: | ---: | ---: | ---: |
| Random | 14.250 | 18.417 | -1.7090 | 0.01093 |
| Coverage | 12.458 | 15.125 | -1.8423 | 0.00721 |
| Local information | 11.958 | 17.750 | -1.8610 | 0.00945 |
| Target risk | 11.625 | 15.667 | -1.8920 | 0.00849 |
| Ranker | 13.583 | 15.708 | -1.7605 | 0.01276 |
| Frozen LLM | 13.625 | 17.000 | -1.6998 | 0.01054 |
| Supervised LLM | 14.625 | 16.208 | -1.7898 | 0.01152 |
| DPO LLM | 15.167 | 17.000 | -1.7597 | 0.01084 |

- **Log-area ranking (best to worst): target risk, local information, coverage, supervised, ranker, DPO, random, frozen.** Confirmed. The same order holds for grids 8..32, 8..31 and trapezoid integration. Ranker and DPO differ by 0.0008, which is a tie.
- The ranking is not robust to the summary. Raw-NMSE area (grid 9..32): local information 0.0298, target risk 0.0336, coverage 0.0355, supervised 0.0371, frozen 0.0415, ranker 0.0452, DPO 0.0508, random 0.0608. Median log-area (grid 9..32): target risk, random, supervised, local information, coverage, ranker, DPO, frozen.
- New (script `v19_area_split.py`): target risk's first place depends on the 10 systems that were already at target. On the 14 informative systems the mean log-area order is local information -1.755, target risk -1.624, coverage -1.598, ranker -1.539, supervised -1.481, DPO -1.474, random -1.457, frozen -1.441. Local information minus target risk there is -0.132 (lower on 9 of 14, sign-flip p 0.32). On the 10 at-target systems it is +0.259 (p 0.088) and local information is last of the eight. Neither difference is resolved.
- Policy minus target risk, log-area (mean, SD, sign-flip p): random +0.183, 0.490, 0.080. Coverage +0.050, 0.351, 0.502. Local information +0.031, 0.490, 0.761. Ranker +0.132, 0.478, 0.190. Frozen +0.192, 0.385, 0.023. Supervised +0.102, 0.356, 0.174. DPO +0.132, 0.481, 0.189. All confirmed.
- Sustained attainment (first cost from which the run stays at or below 0.04 through its final state, else 32): order coverage 15.13, target risk 15.67, ranker 15.71, supervised 16.21, frozen 17.00 = DPO 17.00, local information 17.75, random 18.42. Confirmed: coverage first, DPO tied fifth. No arm is separated from target risk on this endpoint (all p >= 0.17).
- Standardized effect, random vs target risk: capped cost 0.484, log-area 0.374, capped cost at threshold 0.02 0.676. Confirmed. Fixed-cost log10 NMSE at cost 10/12/16/20/24/28/32: 0.39/0.30/0.34/0.31/0.35/0.11/0.17.
- Qualification: "curve area is less sensitive than capped cost" holds for random, coverage, supervised and DPO. It does not hold for frozen (0.50 on log-area vs 0.30 on capped cost) and is about equal for ranker and local information.
- Coverage vs target risk: +0.833 at 0.04 (p 0.226), log-area +0.050 (p 0.502), +2.167 at 0.02 (p 0.035). Confirmed. New: at 0.01 it is +3.875 (p 0.0007).
- New: tighter thresholds separate target risk from more arms (post hoc). Sign-flip p vs target risk at 0.02: random 0.0010, coverage 0.035, local information 0.20, ranker 0.0015, frozen 0.0062, supervised 0.018, DPO 0.0004. At 0.01 all seven are below 0.05 (local information 0.013). Mean capped cost at 0.02: target risk 13.54, local information 15.50, coverage 15.71, supervised 16.92, ranker 17.92, random 18.04, DPO 18.25, frozen 19.29.
- Final NMSE separates nothing (all p >= 0.23 vs target risk).

## F. How fixed are the strong policies' action sequences (partially confirmed)

Scripts `v08_actions.py`, `v09_fixedness.py`, `v10_fixedness_variants.py`, `v11_fixedness_more.py`, `v15_mix_heterogeneity.py`.

The estimated natural X mean was recovered from saved predictions as (q0 - q2)/(q3 - q2), valid for all three graphs because the 8 target queries are the 8 menu actions (`llm_intervention_screen_protocol.py` line 80).

Confirmed:

- Observation share: target risk 215/560 = 38.4%, local information 318/570 = 55.8%.
- Y-involving share of target-risk actions: uniform menu 122/288 = 42.4%, heterogeneous menu 16/272 = 5.9%. (Local information: 10.4% and 2.1%.)
- Seven systems have estimated X mean below zero, so X = -1 is the near side: w03, w04, w07, w10, w12, w15, w20.
- The recovered X-mean estimate is constant in 24/24 DPO runs and 24/24 ranker runs (0/24 for random, coverage, local information, target risk and frozen, 13/24 for supervised). Consistent with `llm_intervention_scm.py` lines 82-85.

Needs correction, the "opens with" statement:

| Definition | Target risk | Local information |
| --- | ---: | ---: |
| Literal first action is the far pure X extreme | 17/24 | 22/24 |
| Literal first action has X at the far extreme, counting set_xy as X = -1 | 22/24 | 22/24 |
| First X-setting action anywhere in the run (incl. set_xy) is at the far extreme | 24/24 | 24/24 |
| First pure X-extreme action anywhere in the run is the far one | **23/24** | **24/24** |

The claimed 23/24 and 24/24 match only the last row. Target risk opens with set_xy in 5 runs (w05, w08, w14, w17, w22) and with a Y action in 2 (w16, w20). The single exception in the last row is w17, where the estimate is +0.015. Over all steps, target risk uses the far extreme 182 times and the near extreme 23 times. Local information 207 and 9.

Needs correction, the fixed-sequence agreement:

- Definition that reproduces the claimed agreement: cells = graph x cost menu (6 cells of 4 systems), predicted action at each step = positional mode of the other three systems, X extremes relabelled far/near by the initial X-mean estimate. Leave-one-out agreement: **target risk 0.409, local information 0.570** (claimed 0.41 and 0.58).
- Within-run order-shuffle baseline under the same definition (2,000 shuffles): **target risk 0.337 (p 0.003), local information 0.478 (p 0.0015)** (claimed 0.38 and 0.47). The target-risk baseline does not reproduce under this definition.
- Cell-label shuffle baseline (permute which cell each system belongs to, 500 permutations, script `v20_fixedness_labelshuffle.py`): **target risk 0.381 (p 0.11), local information 0.587 (p 0.81)**. This reproduces the claimed 0.38 for target risk but not the claimed 0.47 for local information. The two claimed baselines therefore match two different shuffle definitions. Reading: knowing the graph x cost-menu cell adds nothing to step-by-step prediction of either policy. Step position adds a little (order shuffle +0.07 and +0.09).
- Sensitivity of the order-shuffle result: over 48 variants (36 = 3 cell definitions x 3 token schemes x 4 tie rules, each with two shuffle schemes, plus 12 = 4 cell definitions x 3 token schemes with ties broken by overall frequency) the target-risk agreement ranges 0.32 to 0.49 and the order-shuffle baseline 0.26 to 0.37. Excess over the order shuffle is +0.03 to +0.13.
- A constant "always the cell's most frequent action" guess scores 0.35 for target risk and 0.558 for local information under the same cells. For local information a per-cell fixed sequence is no better than "always observe".
- All 24 target-risk sequences are distinct. Mean pairwise positional agreement is 0.29 within cells and 0.25 across cells. Agreement is highest in the first three steps (0.49) and about 0.37 mid-run.
- Menu coverage is a truly fixed cycle: agreement 1.000 across all 24 systems.
- New, against "an action mix plus one sign bit": target risk's action mix varies strongly between systems inside a cell. Chain/uniform: observation share 0.25, 0.25, 0.71, 0.54 and Y share 0.29, 0.58, 0, 0. Collider/heterogeneous: observation share 0.11, 0.61, 0.00, 0.67. Fork/uniform: Y share 0.46 to 0.83. Heuristic chi-square p is below 0.01 in 4 of 6 cells. Target risk is state dependent beyond graph, menu and sign. What the outcome data show is that this state dependence did not buy a measurable advantage over the fixed coverage cycle at threshold 0.04.

## G. Replicate noise, twins, selection (confirmed, inference plausible)

Script `v12_noise_twins.py`.

- 293 policy pairs share the same first action inside a system. 0 of 293 have identical post-action predictions. Initial predictions are identical across arms. Code agrees: each arm has its own measurement namespace (`llm_intervention_screen_protocol.py` lines 149-150, `llm_intervention_world.py` line 308).
- DPO and supervised share an all-set_x_neg prefix of at least 10 actions in every system (min 10 on w11, max 22).
- At cap 18 their capped costs differ on 7 of 14 informative systems (differences -2, 7, -5, -1, 8, 1, -6). SD across systems of the log10 NMSE difference at total cost 9..18: 0.351 to 0.580. That implies a single-run replicate SD of 0.25 to 0.41 log10 units.
- Best of two identical-policy runs gains 0.625 units per system over their mean at cap 18. The envelope gains 1.2917 over target risk at cap 18. Both confirmed.
- The "largely selection" reading is an inference. Scaling the twin gap by normal order statistics gives an expected single-run-minus-best-of-8 gain of about 1.6, which is above the observed 1.29, so selection alone could produce the envelope. It rests on one twin pair of a weak always-X policy and exchangeability that does not hold across the 8 arms.

## H. Remaining claims

Scripts `v12_noise_twins.py`, `v13_strata_sign.py`, `v17_weakpairs_scan.py`, `v18_nearside_specificity.py`.

- Error rose after 1,909 of 4,348 acquisitions (43.91%). 61 of 172 crossing runs were later above target. On the 10 at-target systems 20 of 80 runs left it. Confirmed. 15 crossing runs end above target.
- Supervised LLM stopped in 23 systems (not w17) after 21 to 23 actions at total cost 29 to 31, 47 units unspent. Stops above target: w10 0.0493, w11 0.0472, w12 0.0917, w19 0.1991, w21 0.0583. Pearson correlation of stop cost with log10 final error 0.143. Confirmed.
- DPO minus target risk by stratum: chain +1.125, fork +2.5, collider +7.0. matched_low +5.5, matched_high +3.667, shifted_low +2.167, shifted_high +2.833. Uniform +2.5, heterogeneous +4.583. Confirmed and equal to `strata.csv`.
- DPO minus target risk log-area: +0.535 on the 7 near-side systems, -0.033 on the other 17. Exact permutation over all 346,104 label sets: two-sided p = 0.00547. Confirmed.
- New qualification: the same split appears for supervised (+0.359 vs -0.004, p 0.020) and for the frozen LLM, which never intervenes (+0.426 vs +0.096, p 0.055). DPO minus frozen does not split (p 0.32). DPO minus coverage does (p 0.0054). The pattern separates arms that never set X at the far extreme from arms that do. It is not specific to DPO. It is a post hoc split on 7 systems.
- The ranker's majority action equals the far X extreme in only 10 of 24 systems. Its X choice does not follow the sign rule.
- Weak arms (random, ranker, frozen, supervised, DPO), 10 pairs: 0 of 90 sign-flip tests below 0.05 over nine endpoints (capped cost at 0.04, 0.02, 0.01, sustained cost, log-area, log final NMSE, log NMSE at cost 16 and 24, raw area). Minimum p 0.061. Scanning log NMSE at every cost 9..32: 1 of 240 below 0.05 (frozen vs supervised at cost 12, p 0.043). The claimed "1 of 70" is plausible but not exactly reproducible without the endpoint list.

## Verdict table

| Claim | Verdict |
| --- | --- |
| CSVs internally consistent, 0 of 192 runs inconsistent | Confirmed |
| 10 systems at target after initialization, identity and strata | Confirmed |
| Three more within 12.5% of threshold | Confirmed to rounding (third is 12.503%) |
| All-zero prediction meets 0.04 on 3 systems | Confirmed (w02, w06, w18) |
| Headroom 279 / 230 / 206, 49 and 73, 8 systems | Confirmed |
| Median paired SD 6.2, MDD 3.7, 305/408, 78/104, log-area 859/1150 and 193/258 | Confirmed |
| Replicated design 172 to 93 | Confirmed to rounding (173 at unrounded SD) |
| Oracle at floor power 0.68 | Confirmed as formula. Bootstrap gives 0.84 to 0.93 |
| Sign-flip p-values and Holm vs target risk | Confirmed |
| Log-area values and ranking | Confirmed. Ranking changes under raw area, medians and on the 14 informative systems |
| Sustained-cost values and order | Confirmed |
| Curve area less sensitive than capped cost | Partially. True for random, not for frozen |
| Open with far X extreme 23/24 and 24/24 | Partially. Literal first action 17/24 and 22/24 |
| Observation and Y-involving shares | Confirmed |
| Leave-one-out agreement 0.41 / 0.58, shuffle 0.38 / 0.47 | Partially. Agreement 0.409 / 0.570. Order shuffle 0.337 / 0.478. Cell-label shuffle 0.381 / 0.587 |
| Coverage within noise of target risk at 0.04, worse at 0.02 | Confirmed |
| DPO strata, near-side split p 0.0055, constant X-mean estimate | Confirmed |
| Non-monotone error counts | Confirmed |
| Supervised stop facts | Confirmed |
| Arms do not share noise, twin facts | Confirmed |
| Envelope largely selection | Numbers confirmed. Inference plausible, untested |
| 1 of 70 weak-arm tests below 0.05 | Could not check exactly. 0 of 90 and 1 of 240 in my sets |

## Implications for the agenda (my inferences)

- The earlier summary sentence "this screen could neither support nor refute an LLM benefit" should be narrowed. The screen could have detected a policy that dominated target risk on most informative systems. It could not have detected a noisy policy with a modestly better mean, and the available headroom at threshold 0.04 (3.04 units, of which 42 of 49 envelope units sit on five systems) is about the size of the 80%-power detectable difference.
- Three design levers have direct support in the saved data: remove systems that are at target after initialization (10 of 24 here, halves required systems), replicate runs inside systems (replicate noise is at least half of the paired-difference variance at cap 18), and use a tighter or multi-threshold endpoint (0.02 and 0.01 separate target risk from more arms, post hoc).
- Sample sizes of roughly 300 to 400 unreplicated systems for a 1-unit difference follow from SD 6.2. They fall to roughly 90 to 175 with 8 replicates, and the planning SD itself ranges from 3.1 to 9.0 depending on the contrast.
- Target risk is state dependent (its mix differs across systems inside a cell), yet a fixed coverage cycle is within noise of it at 0.04. Whether state-dependent choice has measurable value on this task family is unresolved at this threshold and looks resolved in target risk's favor only at 0.02 and 0.01, post hoc.
