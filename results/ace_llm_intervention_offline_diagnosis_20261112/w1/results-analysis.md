# Quantitative re-analysis of the saved 24-system comparison (label: results-analysis)

Written November 12, 2026 by a read-only analysis agent. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was modified. No model, episode, SLURM or network operation was run. Everything below is offline arithmetic on saved files with `/usr/bin/python3` (numpy 2.0.2, pandas 2.3.3, scipy 1.13.1).

**Status of every number here: descriptive, post hoc, development data.** The 24 systems were a development screen. No endpoint other than capped cost to target at NMSE 0.04 was predeclared. All p-values are uncorrected unless a correction is named, and many endpoints were examined. Do not use them to select a policy or as confirmatory evidence.

## Top-line summary

1. **The saved data are internally consistent and every number in the handoff and the results report reproduces.** No numerical disagreement between documents and data.
2. **Ten of 24 systems are at target after initialization**: 5 of 8 chains, 5 of 8 forks, 0 of 8 colliders; 7 of 12 matched-regime and 3 of 12 shifted-regime systems; 6 of 12 uniform-menu and 4 of 12 heterogeneous-menu systems. They are weak-effect systems. On three of them an all-zero prediction already meets the target.
3. **There is almost no measurable room above target risk on the predeclared endpoint.** The absolute ceiling is 3.04 cost units per system (73 in total). The hindsight best-of-8 envelope is 2.04 (49 in total, 42 of them in five systems) and is plausibly mostly selection among noisy runs. The minimum detectable difference at 24 systems is 3.7 units at the typical paired SD.
4. **Single runs carry large measurement-noise replicate variance.** Arms do not share noise. dpo and sft ran the identical action prefix for at least 10 actions in every system, and their first-crossing costs differ on 7 of the 14 informative systems. Replicate SD of log10 NMSE at a fixed cost is about 0.3.
5. **On the predeclared endpoint target risk is separated only from dpo (sign-flip p 0.0068, Holm 0.048), random (0.011) and sft (0.022).** The ranker and the three LLM arms are not separated from random or from each other beyond chance (1 of 70 pairwise tests over seven endpoints is below 0.05). Target risk and local information cannot be ranked.
6. **The curve-area and fixed-cost endpoints are not more sensitive than capped cost.** A stricter threshold (0.02) is.
7. **Power:** about 305 systems (80 percent) or 408 (90 percent) for a 1-unit improvement over the planner with single runs, 78 or 104 for 2 units, the same 305 or 408 for noninferiority with a 1-unit margin. For the log-area endpoint about 193 or 258 systems for a 20 percent improvement and 859 or 1,150 for 10 percent. Replicates could cut the 1-unit figure to about 93 to 172.
8. **Good behavior here is a mix, an opening rule and one sign bit, not a fixed sequence.** Both strong policies open with X at the extreme on the far side of the natural X mean (23 of 24 and 24 of 24 runs), then interleave observation (38 and 56 percent of actions) and use Y only when it is cheap. A fully fixed coverage cycle is within noise of the planner at the predeclared threshold.
9. **The constant x- policy loses most on colliders (+7.0 cost units) and where x- is the low-leverage side** (log-area +0.54 against target risk on 7 systems, -0.03 on the other 17). It can never update the X root mechanism and cannot separate intercept from slope in X's children.
10. **Error curves are strongly non-monotone.** Error rose after 43.9 percent of acquisitions. 61 of 172 runs that crossed the target later left it.
11. **The supervised arm's stop is step-linked, not accuracy-linked.** 23 stops after 21 to 23 actions at total cost 29 to 31, five of them above target.
12. Recommendations are in section 10. Open questions are in section 13.

## 0. Inputs, scripts and conventions

Inputs (all read-only):

- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/cells.csv` (192 rows), `curves.csv` (4,540 rows), `policies.csv`, `pairs.csv`, `strata.csv`, `actions.csv`.
- Scope of the earlier arithmetic review: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/numerical-review01/review.md`. That review reconstructed counts, pair differences and group means, and checked the CSV exports and the static canvas (21,045 assertions). Line 23 states its limits: saved errors were taken as inputs, no ground truth was re-evaluated, and (line 15) no significance conclusion was drawn. It contains no paired tests, no alternative endpoints, no power analysis, no action-sequence analysis and no variance decomposition. Everything in sections 3 to 9 below is new relative to that review.
- Design facts used for interpretation: `/Users/pat/code/ACE/scripts/research/llm_intervention_screen_protocol.py` (regimes at lines 27-28, graphs at 57-61, menu and costs at 64-71, queries, weights, scale 2 and tolerance 0.04 at 76-77, 8 initial observations and 24 further units at 87-88, arm-specific measurement namespaces at 148-152), `/Users/pat/code/ACE/scripts/research/llm_intervention_scm.py` (update rule at lines 82-98) and `/Users/pat/code/ACE/scripts/research/llm_intervention_target_risk.py`.

Scripts and their saved outputs are in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis/` (`common.py`, `a01_sanity.py` ... `a26_weakpairs.py`, each with `*_out.txt`; table generators `make_tables.py`, `make_tables2.py`, `make_tables3.py` and `assemble_notes.py`; generated tables in `tables.json`; CSV exports `persystem.csv`, `paired_endpoints.csv`, `key_contrasts.csv`, `threshold_sweep.csv`, `sft_stops.csv`, `nonmonotonicity.csv`, `fixed_sequence_agreement.csv`).

Conventions:

- Systems are labeled w00 to w23 = `world_id - 202611070100`. Policies: random, coverage, local_info, target_risk (TR), ranker, frozen, sft, dpo.
- "Informative 14" = the 14 systems whose error after initialization was above 0.04. "At target after init" = the other 10.
- The error curve is treated as a right-continuous **step function of paid cost**: e(c) is the error after the last accepted response with total cost at most c. After a policy stop the last SCM state is carried forward. Under this convention the error at cost 32 equals `final_error` for every run (checked).
- "log-area" = mean of log10 NMSE at the 24 integer total costs 9..32 (area under the log error curve over the acquisition phase, divided by 24). Lower is better. A difference of 0.0458 is a 10 percent lower geometric-mean error and 0.0969 is 20 percent lower.
- Paired difference = policy minus reference per system. Negative means the policy is better. "-/0/+" counts systems with negative, zero and positive difference.
- "sign-flip p" = exact two-sided paired randomization p-value for the mean difference, enumerating all sign patterns (validated against brute force in `a20_validate.py`). "Wilcoxon p" = scipy signed-rank with zeros dropped (normal approximation when ties are present).
- Actions: obs = observe_all, x- / x0 / x+ = set X to -1 / 0 / 1, y- / y0 / y+ likewise for Y, xy = set X=-1 and Y=1. In the heterogeneous menu y-, y0, y+ and xy cost 2.

## 1. Sanity checks (script `a01_sanity.py`)

All passed.

- The policy summary recomputed from `cells.csv` equals `policies.csv` in every column.
- All 192 runs are internally consistent between `cells.csv` and `curves.csv` (first crossing, capped cost, ever and final attainment, final error, paid cost, initialization row).
- Every cost increment in `curves.csv` equals the declared menu price of the recorded action (0 of 192 runs disagree).
- The post-initialization prediction vector and error are identical across the eight arms in every system.
- `actions.csv` and all 28 rows of `pairs.csv` are reproduced. In `pairs.csv`, difference = left minus right and "wins" = left lower.
- Totals: 4,540 curve points, 5,884 accepted responses, 6,097 paid cost units. Acquisitions per policy: random 487, coverage 480, local_info 570, TR 560, ranker 576, frozen 576, sft 523, dpo 576.

| policy | mean capped total cost | ever /24 | final /24 | mean final NMSE | median final NMSE | geo-mean final NMSE | paid cost | stops |
|---|---|---|---|---|---|---|---|---|
| random | 14.25000 | 21 | 20 | 0.04113 | 0.01329 | 0.01093 | 768 | 0 |
| coverage | 12.45833 | 22 | 21 | 0.01974 | 0.00836 | 0.00721 | 768 | 0 |
| local_info | 11.95833 | 23 | 18 | 0.01800 | 0.00822 | 0.00945 | 768 | 0 |
| target_risk | 11.62500 | 23 | 21 | 0.01722 | 0.00996 | 0.00849 | 768 | 0 |
| ranker | 13.58333 | 21 | 20 | 0.03572 | 0.01299 | 0.01276 | 768 | 0 |
| frozen | 13.62500 | 22 | 20 | 0.02008 | 0.01259 | 0.01054 | 768 | 0 |
| sft | 14.62500 | 21 | 19 | 0.02708 | 0.00829 | 0.01152 | 721 | 23 |
| dpo | 15.16667 | 19 | 18 | 0.03642 | 0.01065 | 0.01084 | 768 | 0 |

The mean final error is driven by a few systems. By median or geometric mean the ordering changes: sft has the second lowest median final error (0.0083), and dpo, frozen and random have nearly the same geometric mean (0.0108, 0.0105, 0.0109).

## 2. Per-system table (script `a02_persystem.py`)

Capped total cost to target per policy. "init NMSE" is the error after the eight shared initial observations. best8 is the per-system minimum over the eight arms.

| sys | graph | regime | menu | init NMSE | random | coverage | local_info | target_risk | ranker | frozen | sft | dpo | best8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w00 | chain | matched_low | uni | 0.0298 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w01 | chain | matched_low | het | 0.0060 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w02 | fork | matched_low | uni | 0.0377 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w03 | fork | matched_low | het | 0.0681 | 18 | 10 | 9 | 9 | 11 | 24 | 26 | 32 | 9 |
| w04 | collider | matched_low | uni | 0.0450 | 10 | 10 | 9 | 9 | 9 | 10 | 13 | 11 | 9 |
| w05 | collider | matched_low | het | 0.0430 | 13 | 13 | 17 | 10 | 9 | 13 | 11 | 18 | 9 |
| w06 | chain | matched_high | uni | 0.0003 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w07 | chain | matched_high | het | 0.0367 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w08 | fork | matched_high | uni | 0.0059 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w09 | fork | matched_high | het | 0.0379 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w10 | collider | matched_high | uni | 0.1250 | 10 | 11 | 11 | 9 | 12 | 32 | 23 | 13 | 9 |
| w11 | collider | matched_high | het | 0.1107 | 10 | 21 | 15 | 14 | 32 | 11 | 21 | 32 | 10 |
| w12 | chain | shifted_low | uni | 0.5219 | 32 | 32 | 16 | 23 | 13 | 25 | 32 | 32 | 13 |
| w13 | chain | shifted_low | het | 0.3160 | 32 | 9 | 10 | 9 | 9 | 14 | 10 | 9 | 9 |
| w14 | fork | shifted_low | uni | 0.0411 | 9 | 9 | 9 | 9 | 9 | 11 | 9 | 9 | 9 |
| w15 | fork | shifted_low | het | 0.0129 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w16 | collider | shifted_low | uni | 0.0916 | 16 | 11 | 26 | 17 | 16 | 11 | 10 | 22 | 10 |
| w17 | collider | shifted_low | het | 0.1115 | 17 | 9 | 9 | 11 | 21 | 14 | 9 | 10 | 9 |
| w18 | chain | shifted_high | uni | 0.0154 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w19 | chain | shifted_high | het | 0.5449 | 32 | 32 | 23 | 32 | 32 | 32 | 32 | 32 | 23 |
| w20 | fork | shifted_high | uni | 0.0209 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| w21 | fork | shifted_high | het | 0.0778 | 21 | 21 | 9 | 15 | 11 | 9 | 32 | 12 | 9 |
| w22 | collider | shifted_high | uni | 0.4139 | 14 | 12 | 32 | 10 | 32 | 25 | 19 | 20 | 10 |
| w23 | collider | shifted_high | het | 0.2332 | 28 | 19 | 12 | 22 | 30 | 16 | 24 | 32 | 12 |

The ten systems already at target after initialization are w00, w01, w02, w06, w07, w08, w09, w15, w18 and w20. All eight policies score 8 on each of them, and no policy scores 8 anywhere else.

| stratum | at target after init | of |
|---|---|---|
| chain | 5 | 8 |
| fork | 5 | 8 |
| collider | 0 | 8 |
| matched_low | 3 | 6 |
| matched_high | 4 | 6 |
| shifted_low | 1 | 6 |
| shifted_high | 2 | 6 |
| uniform | 6 | 12 |
| heterogeneous | 4 | 12 |

- By graph: 5 of 8 chains, 5 of 8 forks, **0 of 8 colliders**. Every collider is informative.
- By regime: matched_low 3 of 6, matched_high 4 of 6, shifted_low 1 of 6, shifted_high 2 of 6. So 7 of the 12 matched (edge mean 0) systems and 3 of the 12 shifted (edge mean 1.5) systems are uninformative.
- By menu: uniform 6 of 12, heterogeneous 4 of 12.
- Initial error over the 24 systems: minimum 0.00028 (w06), median 0.0440, mean 0.1228, maximum 0.5449 (w19).
- Three more systems start within 12.5 percent above the threshold: w14 (0.0411), w05 (0.0430), w04 (0.0450). On w14 seven of the eight policies cross with their first response. In practice about 11 of 24 systems cannot separate policies on the predeclared endpoint.
- Number of systems at target after initialization for other thresholds: 0.005: 1, 0.01: 3, 0.02: 5, 0.03: 7, 0.04: 10, 0.06: 13, 0.08: 15.

Why these systems are easy (script `a21_scale.py`; uses reconstructed true target means, see section 11.3): the median standard deviation of the eight true target means is 0.49 on the ten at-target systems and 1.73 on the others (Spearman correlation of initial error with true target spread 0.70). On three systems (w02, w06, w18) the prior-mean prediction of zero for every query already meets the 0.04 target. On six systems (w02, w03, w06, w07, w08, w18) a single constant equal to the mean of the eight true targets meets it. The fixed public scale of 2 makes weak-effect systems trivial regardless of intervention choice.

## 3. Headroom above target risk (scripts `a03_headroom.py`, `a16_sensitivity.py`)

"floor" is the trivial lower bound: 8 if at target after initialization, otherwise 9 (one unit-cost response).

| subset | target_risk mean | best-of-8 mean | floor mean | TR - best8 (sum) | TR - best8 (mean) | TR - floor (sum) | TR - floor (mean) | systems where some arm beats TR |
|---|---|---|---|---|---|---|---|---|
| all 24 | 11.625 | 9.583 | 8.583 | 49 | 2.042 | 73 | 3.042 | 8 |
| informative 14 | 14.214 | 10.714 | 9.000 | 49 | 3.500 | 73 | 5.214 | 8 |

- With hindsight, picking the best of the eight realized runs per system saves **49 cost units in total relative to TR** (2.04 per system over 24, 3.50 per system over the informative 14). The absolute ceiling for any policy is 73 units (3.04 per system, 5.21 on the informative 14).
- The hindsight saving is concentrated: w12 10 (ranker), w23 10 (local_info), w19 9 (local_info), w16 7 (sft), w21 6 (local_info and frozen), w11 4 (random), w17 2, w05 1. Five systems hold 42 of the 49 units. On 16 of 24 systems (6 of the informative 14) TR is already at the envelope.
- Winners on those eight systems: local_info alone on w19 and w23 and tied on w17 and w21, ranker on w05 and w12, sft on w16 and tied on w17, random on w11, coverage tied on w17, frozen tied on w21. No single alternative policy realizes the envelope: the best single policy mean is TR's own 11.625.
- Best-of-5-numerical envelope mean 9.625. Best-of-3-LLM envelope mean 11.958. The best LLM arm is strictly below the best numerical arm on one system (w16, sft 10 versus coverage 11), tied on 14 and above on 9.

**Most of the envelope is probably selection among noisy runs (inference).** The envelope is a minimum over eight noisy first-passage times. Section 11.1 shows that two runs of the *same* constant policy on the same system with independent measurement noise differ in capped cost by a mean absolute 1.25 units (2.14 on informative systems) when capped at total cost 18, so best-of-two of one policy already "saves" 0.63 units per system (1.07 on informative systems) with no change in policy. At the same cap 18, the best-of-8 envelope saves 1.29 units over TR (2.21 on informative systems). A best-of-eight selection effect of the same policy would plausibly be as large as the whole observed envelope gap.

**Is there measurable room above target risk? Barely, and not with 24 systems.**

- Minimum detectable mean difference for a paired t-test with 24 systems (two-sided 0.05, 80 percent power) at the median observed paired SD of 6.2 cost units is **3.7 cost units**. That exceeds the absolute ceiling of 3.04. Across the seven observed SDs the minimum detectable difference ranges from 1.8 (coverage, SD 3.07) to 4.0 (frozen, SD 6.76).
- A hypothetical oracle that hit the trivial floor on every system would differ from TR by mean -3.04, SD 5.91, t = -2.52, with plug-in power of only 0.68 in this 24-system screen.
- A new policy that captured half of the floor headroom (1.5 units) with the observed between-policy SD would have had power 0.21 at 24 systems and would need about 133 systems for 80 percent power.

## 4. More sensitive endpoints from the curves (scripts `a06_endpoints.py`, `a16_sensitivity.py`, `a19_contrasts.py`)

Definitions are in section 0. "sustained cost" = the smallest cost from which the error stays at or below 0.04 for the rest of the run (32 if the final state is above target). "geo NMSE@c" = geometric mean over systems of the error at total cost c. Ranks in brackets, 1 = best.

All 24 systems:

| policy | capped cost | sustained cost | log-area | raw area | geo NMSE@12 | geo NMSE@16 | geo NMSE@24 | geo NMSE@32 (final) |
|---|---|---|---|---|---|---|---|---|
| random | 14.250 (6) | 18.417 (8) | -1.709 (7) | 0.0608 (8) | 0.0289 (7) | 0.0272 (8) | 0.0178 (8) | 0.0109 (6) |
| coverage | 12.458 (3) | 15.125 (1) | -1.842 (3) | 0.0355 (3) | 0.0233 (4) | 0.0165 (1) | 0.0118 (4) | 0.0072 (1) |
| local_info | 11.958 (2) | 17.750 (7) | -1.861 (2) | 0.0298 (1) | 0.0213 (2) | 0.0179 (3) | 0.0097 (2) | 0.0095 (3) |
| target_risk | 11.625 (1) | 15.667 (2) | -1.892 (1) | 0.0336 (2) | 0.0192 (1) | 0.0171 (2) | 0.0097 (1) | 0.0085 (2) |
| ranker | 13.583 (4) | 15.708 (3) | -1.761 (5) | 0.0452 (6) | 0.0267 (6) | 0.0183 (4) | 0.0135 (5) | 0.0128 (8) |
| frozen | 13.625 (5) | 17.000 (5) | -1.700 (8) | 0.0415 (5) | 0.0374 (8) | 0.0236 (7) | 0.0168 (7) | 0.0105 (4) |
| sft | 14.625 (7) | 16.208 (4) | -1.790 (4) | 0.0371 (4) | 0.0231 (3) | 0.0188 (5) | 0.0115 (3) | 0.0115 (7) |
| dpo | 15.167 (8) | 17.000 (5) | -1.760 (6) | 0.0508 (7) | 0.0245 (5) | 0.0204 (6) | 0.0163 (6) | 0.0108 (5) |

Informative 14:

| policy | capped cost | sustained cost | log-area | raw area | geo NMSE@12 | geo NMSE@16 | geo NMSE@24 | geo NMSE@32 (final) |
|---|---|---|---|---|---|---|---|---|
| random | 18.714 (6) | 22.429 (8) | -1.457 (7) | 0.0899 (8) | 0.0627 (7) | 0.0529 (8) | 0.0282 (7) | 0.0158 (3) |
| coverage | 15.643 (3) | 19.500 (2) | -1.598 (3) | 0.0531 (3) | 0.0434 (3) | 0.0266 (2) | 0.0203 (4) | 0.0161 (4) |
| local_info | 14.786 (2) | 19.214 (1) | -1.755 (1) | 0.0378 (1) | 0.0361 (2) | 0.0255 (1) | 0.0100 (1) | 0.0098 (1) |
| target_risk | 14.214 (1) | 20.714 (4) | -1.624 (2) | 0.0517 (2) | 0.0336 (1) | 0.0298 (3) | 0.0200 (3) | 0.0137 (2) |
| ranker | 17.571 (4) | 19.857 (3) | -1.539 (4) | 0.0680 (6) | 0.0516 (5) | 0.0298 (4) | 0.0197 (2) | 0.0199 (6) |
| frozen | 17.643 (5) | 21.857 (6) | -1.441 (8) | 0.0617 (5) | 0.0802 (8) | 0.0436 (5) | 0.0294 (8) | 0.0199 (5) |
| sft | 19.357 (7) | 21.643 (5) | -1.481 (5) | 0.0576 (4) | 0.0483 (4) | 0.0455 (7) | 0.0221 (5) | 0.0204 (8) |
| dpo | 20.286 (8) | 22.000 (7) | -1.474 (6) | 0.0792 (7) | 0.0550 (6) | 0.0448 (6) | 0.0261 (6) | 0.0199 (7) |

Observations:

- TR is first on capped cost and log-area over all 24, but it is **not robustly the best numerical policy**. On the informative 14, local_info is first on log-area (-1.755 versus -1.624 for TR) and on the error at costs 16, 24 and 32. TR and local_info are statistically indistinguishable on every endpoint (capped cost difference 0.33, SD 6.2, sign-flip p 0.84).
- The ordering of the weak arms depends on the endpoint. dpo is 8th on capped cost, 6th on log-area, 5th on geometric-mean final error. frozen is 5th on capped cost and 8th on log-area. On sustained cost the order is coverage 15.1, TR 15.7, ranker 15.7, sft 16.2, frozen 17.0, dpo 17.0, local_info 17.8, random 18.4, and no pair is distinguishable (paired SDs 7 to 11).
- Per-system winners are close to uniform across policies. Lowest log-area: TR 4, frozen 4, dpo 4, random 3, ranker 3, coverage 2, local_info 2, sft 2. Lowest final error: TR 4, random 4, dpo 4, local_info 3, sft 3, coverage 2, ranker 2, frozen 2. Single-run system-level outcomes are noise dominated.

### 4.1 Paired differences against target risk

Capped cost to target (predeclared endpoint):

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 2.6250 | 5.4279 | 2/12/10 | 0.0122 | 0.0112 | 4.5000 | 6.5604 | 2/2/10 | 0.0112 |
| coverage | 0.8333 | 3.0740 | 3/13/8 | 0.2402 | 0.2256 | 1.4286 | 3.9752 | 3/3/8 | 0.2256 |
| local_info | 0.3333 | 6.2113 | 5/13/6 | 0.9658 | 0.8398 | 0.5714 | 8.2529 | 5/3/6 | 0.8398 |
| ranker | 1.9583 | 6.6298 | 4/14/6 | 0.2754 | 0.1934 | 3.3571 | 8.5269 | 4/4/6 | 0.1934 |
| frozen | 2.0000 | 6.7631 | 4/11/9 | 0.3757 | 0.1841 | 3.4286 | 8.6976 | 4/1/9 | 0.1841 |
| sft | 3.0000 | 6.0577 | 2/12/10 | 0.0210 | 0.0220 | 5.1429 | 7.2839 | 2/2/10 | 0.0220 |
| dpo | 3.5417 | 6.4335 | 2/13/9 | 0.0068 | 0.0068 | 6.0714 | 7.5291 | 2/3/9 | 0.0068 |

Holm correction over these seven comparisons gives dpo 0.048, random 0.067, sft 0.110, all others above 0.7. So on the predeclared endpoint TR is separated from dpo, random and sft at nominal level and from nothing else. 11 to 14 of the 24 differences are exact zeros in every comparison.

log-area (mean log10 NMSE over costs 9..32):

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.1830 | 0.4898 | 7/0/17 | 0.1280 | 0.0801 | 0.1661 | 0.4551 | 3/0/11 | 0.1969 |
| coverage | 0.0497 | 0.3514 | 7/0/17 | 0.2076 | 0.5023 | 0.0254 | 0.3432 | 4/0/10 | 0.8192 |
| local_info | 0.0310 | 0.4903 | 12/0/12 | 0.5838 | 0.7610 | -0.1319 | 0.4683 | 9/0/5 | 0.3188 |
| ranker | 0.1315 | 0.4781 | 10/0/14 | 0.2405 | 0.1900 | 0.0849 | 0.4494 | 7/0/7 | 0.4912 |
| frozen | 0.1922 | 0.3847 | 9/0/15 | 0.0395 | 0.0227 | 0.1821 | 0.4676 | 6/0/8 | 0.1689 |
| sft | 0.1023 | 0.3562 | 10/0/14 | 0.2076 | 0.1737 | 0.1422 | 0.3990 | 4/0/10 | 0.2074 |
| dpo | 0.1324 | 0.4812 | 8/0/16 | 0.1688 | 0.1892 | 0.1498 | 0.5137 | 4/0/10 | 0.2885 |

log10 NMSE at total cost 12:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.1759 | 0.5964 | 11/0/13 | 0.2897 | 0.1631 | 0.2706 | 0.4649 | 6/0/8 | 0.0502 |
| coverage | 0.0831 | 0.3819 | 8/0/16 | 0.2768 | 0.2950 | 0.1107 | 0.3392 | 5/0/9 | 0.2418 |
| local_info | 0.0437 | 0.4361 | 10/0/14 | 0.6033 | 0.6242 | 0.0307 | 0.4454 | 7/0/7 | 0.7955 |
| ranker | 0.1417 | 0.5183 | 10/0/14 | 0.2522 | 0.1971 | 0.1859 | 0.4139 | 5/0/9 | 0.1166 |
| frozen | 0.2881 | 0.5138 | 8/0/16 | 0.0249 | 0.0122 | 0.3775 | 0.5953 | 5/0/9 | 0.0364 |
| sft | 0.0789 | 0.4345 | 10/0/14 | 0.3748 | 0.3876 | 0.1578 | 0.4791 | 6/0/8 | 0.2428 |
| dpo | 0.1049 | 0.5808 | 8/0/16 | 0.1515 | 0.3916 | 0.2136 | 0.4940 | 4/0/10 | 0.1294 |

log10 NMSE at total cost 16:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.2010 | 0.5925 | 11/0/13 | 0.1600 | 0.1097 | 0.2496 | 0.4923 | 7/0/7 | 0.0817 |
| coverage | -0.0160 | 0.5168 | 12/0/12 | 0.9218 | 0.8804 | -0.0496 | 0.4461 | 8/0/6 | 0.6793 |
| local_info | 0.0198 | 0.5340 | 8/0/16 | 0.5838 | 0.8578 | -0.0679 | 0.5309 | 5/0/9 | 0.6439 |
| ranker | 0.0286 | 0.4775 | 12/0/12 | 0.9218 | 0.7705 | 0.0002 | 0.4604 | 7/0/7 | 0.9979 |
| frozen | 0.1388 | 0.6857 | 9/0/15 | 0.2897 | 0.3301 | 0.1649 | 0.6682 | 6/0/8 | 0.3700 |
| sft | 0.0414 | 0.5349 | 11/0/13 | 0.7048 | 0.7090 | 0.1841 | 0.4505 | 5/0/9 | 0.1495 |
| dpo | 0.0765 | 0.5714 | 10/0/14 | 0.3596 | 0.5207 | 0.1769 | 0.4846 | 4/0/10 | 0.1962 |

log10 NMSE at total cost 24:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.2638 | 0.7590 | 9/0/15 | 0.1688 | 0.0962 | 0.1501 | 0.4517 | 5/0/9 | 0.2405 |
| coverage | 0.0828 | 0.5117 | 9/0/15 | 0.2643 | 0.4406 | 0.0063 | 0.4407 | 5/0/9 | 0.9679 |
| local_info | 0.0001 | 0.9262 | 10/0/14 | 0.7048 | 0.9998 | -0.3003 | 0.8429 | 8/0/6 | 0.2194 |
| ranker | 0.1414 | 0.7482 | 11/0/13 | 0.4223 | 0.3648 | -0.0063 | 0.6413 | 7/0/7 | 0.9702 |
| frozen | 0.2385 | 0.5018 | 8/0/16 | 0.0564 | 0.0288 | 0.1679 | 0.5055 | 6/0/8 | 0.2334 |
| sft | 0.0742 | 0.4769 | 10/0/14 | 0.6033 | 0.4532 | 0.0441 | 0.4403 | 6/0/8 | 0.7133 |
| dpo | 0.2248 | 0.5960 | 7/0/17 | 0.0564 | 0.0774 | 0.1161 | 0.6432 | 5/0/9 | 0.5038 |

log10 final NMSE (cost 32):

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.1096 | 0.6415 | 7/0/17 | 0.1208 | 0.4212 | 0.0606 | 0.7510 | 3/0/11 | 0.7823 |
| coverage | -0.0713 | 0.7370 | 8/0/16 | 0.6431 | 0.6554 | 0.0708 | 0.5888 | 3/0/11 | 0.6719 |
| local_info | 0.0466 | 0.6588 | 10/0/14 | 0.6231 | 0.7327 | -0.1449 | 0.6745 | 8/0/6 | 0.4333 |
| ranker | 0.1767 | 0.7151 | 10/0/14 | 0.1974 | 0.2381 | 0.1631 | 0.7180 | 6/0/8 | 0.4132 |
| frozen | 0.0939 | 0.6369 | 11/0/13 | 0.5457 | 0.4849 | 0.1612 | 0.7122 | 6/0/8 | 0.4247 |
| sft | 0.1325 | 0.5778 | 10/0/14 | 0.2768 | 0.2700 | 0.1729 | 0.6487 | 5/0/9 | 0.3289 |
| dpo | 0.1062 | 0.8359 | 8/0/16 | 0.4559 | 0.5370 | 0.1631 | 0.9289 | 5/0/9 | 0.5150 |

Sustained cost:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| random | 2.7500 | 9.4604 | 4/8/12 | 0.1297 | 0.1774 | 1.7143 | 9.6191 | 3/2/9 | 0.5430 |
| coverage | -0.5417 | 8.9733 | 4/10/10 | 0.8077 | 0.7917 | -1.2143 | 11.8398 | 4/1/9 | 0.7236 |
| local_info | 2.0833 | 10.7417 | 5/9/10 | 0.3028 | 0.3629 | -1.5000 | 10.6681 | 5/4/5 | 0.6270 |
| ranker | 0.0417 | 7.2740 | 6/10/8 | 1.0000 | 1.0000 | -0.8571 | 9.2890 | 6/2/6 | 0.7466 |
| frozen | 1.3333 | 9.7341 | 6/9/9 | 0.6387 | 0.5260 | 1.1429 | 12.6604 | 6/2/6 | 0.7559 |
| sft | 0.5417 | 9.4408 | 6/8/10 | 0.5282 | 0.7994 | 0.9286 | 12.3691 | 5/1/8 | 0.7954 |
| dpo | 1.3333 | 10.6757 | 6/8/10 | 0.4637 | 0.5606 | 1.2857 | 13.2342 | 5/1/8 | 0.7305 |

### 4.2 Paired differences against uniform random

Capped cost to target:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -1.7917 | 5.9416 | 6/16/2 | 0.1953 | 0.1719 | -3.0714 | 7.6305 | 6/6/2 | 0.1719 |
| local_info | -2.2917 | 8.6148 | 8/11/5 | 0.2439 | 0.2156 | -3.9286 | 11.1525 | 8/1/5 | 0.2156 |
| target_risk | -2.6250 | 5.4279 | 10/12/2 | 0.0093 | 0.0112 | -4.5000 | 6.5604 | 10/2/2 | 0.0112 |
| ranker | -0.6667 | 9.0346 | 6/13/5 | 0.6377 | 0.7363 | -1.1429 | 11.9927 | 6/3/5 | 0.7363 |
| frozen | -0.6250 | 7.6090 | 6/13/5 | 0.5771 | 0.7197 | -1.0714 | 10.0954 | 6/3/5 | 0.7197 |
| sft | 0.3750 | 6.9458 | 5/13/6 | 0.6377 | 0.8242 | 0.6429 | 9.2288 | 5/3/6 | 0.8242 |
| dpo | 0.9167 | 7.9176 | 3/13/8 | 0.5771 | 0.6045 | 1.5714 | 10.4787 | 3/3/8 | 0.6045 |

log-area:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -0.1333 | 0.5692 | 14/0/10 | 0.4908 | 0.2735 | -0.1407 | 0.4107 | 10/0/4 | 0.2311 |
| local_info | -0.1520 | 0.6870 | 14/0/10 | 0.4559 | 0.2924 | -0.2979 | 0.6193 | 11/0/3 | 0.0973 |
| target_risk | -0.1830 | 0.4898 | 17/0/7 | 0.1280 | 0.0801 | -0.1661 | 0.4551 | 11/0/3 | 0.1969 |
| ranker | -0.0515 | 0.5639 | 11/0/13 | 0.8115 | 0.6656 | -0.0811 | 0.6665 | 7/0/7 | 0.6587 |
| frozen | 0.0092 | 0.5678 | 11/0/13 | 0.7898 | 0.9372 | 0.0160 | 0.6345 | 8/0/6 | 0.9249 |
| sft | -0.0808 | 0.5814 | 12/0/12 | 0.7469 | 0.5050 | -0.0239 | 0.5638 | 8/0/6 | 0.8782 |
| dpo | -0.0507 | 0.6845 | 9/0/15 | 0.6634 | 0.7228 | -0.0163 | 0.6609 | 4/0/10 | 0.9362 |

log10 NMSE at total cost 12:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -0.0928 | 0.5242 | 12/0/12 | 0.8553 | 0.3967 | -0.1598 | 0.4939 | 9/0/5 | 0.2639 |
| local_info | -0.1322 | 0.6819 | 12/0/12 | 0.4559 | 0.3518 | -0.2398 | 0.5734 | 8/0/6 | 0.1451 |
| target_risk | -0.1759 | 0.5964 | 13/0/11 | 0.2897 | 0.1631 | -0.2706 | 0.4649 | 8/0/6 | 0.0502 |
| ranker | -0.0341 | 0.5685 | 13/0/11 | 0.8334 | 0.7710 | -0.0847 | 0.6336 | 7/0/7 | 0.6237 |
| frozen | 0.1123 | 0.4813 | 12/0/12 | 0.4732 | 0.2644 | 0.1070 | 0.3896 | 7/0/7 | 0.3257 |
| sft | -0.0970 | 0.6142 | 12/0/12 | 0.6231 | 0.4569 | -0.1128 | 0.5059 | 7/0/7 | 0.4274 |
| dpo | -0.0710 | 0.5364 | 10/0/14 | 0.8334 | 0.5330 | -0.0570 | 0.5452 | 5/0/9 | 0.7473 |

log10 NMSE at total cost 16:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -0.2170 | 0.6501 | 14/0/10 | 0.2522 | 0.1162 | -0.2992 | 0.4856 | 10/0/4 | 0.0387 |
| local_info | -0.1812 | 0.7800 | 12/0/12 | 0.5088 | 0.2700 | -0.3174 | 0.7083 | 8/0/6 | 0.1190 |
| target_risk | -0.2010 | 0.5925 | 13/0/11 | 0.1600 | 0.1097 | -0.2496 | 0.4923 | 7/0/7 | 0.0817 |
| ranker | -0.1724 | 0.6693 | 13/0/11 | 0.4061 | 0.2210 | -0.2494 | 0.6812 | 9/0/5 | 0.2002 |
| frozen | -0.0622 | 0.7325 | 14/0/10 | 0.7683 | 0.6871 | -0.0846 | 0.5505 | 10/0/4 | 0.5671 |
| sft | -0.1596 | 0.6395 | 13/0/11 | 0.5088 | 0.2489 | -0.0655 | 0.4012 | 7/0/7 | 0.5537 |
| dpo | -0.1245 | 0.6777 | 13/0/11 | 0.6231 | 0.3794 | -0.0727 | 0.5931 | 7/0/7 | 0.6827 |

log10 NMSE at total cost 24:

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -0.1810 | 0.7320 | 14/0/10 | 0.4061 | 0.2433 | -0.1438 | 0.4918 | 10/0/4 | 0.2961 |
| local_info | -0.2637 | 0.9212 | 16/0/8 | 0.2643 | 0.1756 | -0.4504 | 0.8437 | 11/0/3 | 0.0684 |
| target_risk | -0.2638 | 0.7590 | 15/0/9 | 0.1688 | 0.0962 | -0.1501 | 0.4517 | 9/0/5 | 0.2405 |
| ranker | -0.1224 | 0.7112 | 12/0/12 | 0.4732 | 0.4106 | -0.1564 | 0.7615 | 7/0/7 | 0.4652 |
| frozen | -0.0253 | 0.7318 | 13/0/11 | 0.9888 | 0.8665 | 0.0179 | 0.7519 | 8/0/6 | 0.9327 |
| sft | -0.1896 | 0.6945 | 15/0/9 | 0.1974 | 0.1955 | -0.1060 | 0.6560 | 10/0/4 | 0.5526 |
| dpo | -0.0390 | 0.7631 | 10/0/14 | 0.7683 | 0.8044 | -0.0339 | 0.7312 | 6/0/8 | 0.8717 |

log10 final NMSE (cost 32):

| policy | mean | SD | -/0/+ | Wilcoxon p | sign-flip p | inf14 mean | inf14 SD | inf14 -/0/+ | inf14 sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| coverage | -0.1809 | 0.8302 | 12/0/12 | 0.5838 | 0.3266 | 0.0102 | 0.4942 | 6/0/8 | 0.9393 |
| local_info | -0.0630 | 0.7308 | 13/0/11 | 0.7469 | 0.6752 | -0.2055 | 0.6973 | 9/0/5 | 0.2915 |
| target_risk | -0.1096 | 0.6415 | 17/0/7 | 0.1208 | 0.4212 | -0.0606 | 0.7510 | 11/0/3 | 0.7823 |
| ranker | 0.0671 | 0.6689 | 11/0/13 | 0.6634 | 0.6283 | 0.1024 | 0.7870 | 7/0/7 | 0.6323 |
| frozen | -0.0157 | 0.9267 | 12/0/12 | 0.8553 | 0.9343 | 0.1006 | 1.0610 | 8/0/6 | 0.7196 |
| sft | 0.0229 | 0.7919 | 9/0/15 | 0.8996 | 0.8888 | 0.1123 | 0.9081 | 5/0/9 | 0.6503 |
| dpo | -0.0034 | 0.9639 | 11/0/13 | 0.8774 | 0.9862 | 0.1024 | 1.0482 | 6/0/8 | 0.7168 |

**Against random, only TR is nominally better on the predeclared endpoint (sign-flip p 0.011).** The ranker, frozen, sft and dpo are not separated from random on any endpoint examined (over ten endpoints the smallest sign-flip p is 0.09, sft on the natural-scale area, and the smallest Wilcoxon p is 0.19). The screen can barely tell its best numerical policy from random.

### 4.3 Contrasts that carry the scientific claims

| contrast | endpoint | mean | sd | -/0/+ | signflip_p | why |
|---|---|---|---|---|---|---|
| dpo - frozen | capped_cost | 1.5417 | 7.6328 | 5/11/8 | 0.3545 | effect of the whole training recipe |
| dpo - frozen | log_area | -0.0598 | 0.5243 | 13/0/11 | 0.5797 | effect of the whole training recipe |
| sft - frozen | capped_cost | 1.0000 | 6.2485 | 7/11/6 | 0.5076 | effect of supervised stage |
| sft - frozen | log_area | -0.0899 | 0.3641 | 14/0/10 | 0.2365 | effect of supervised stage |
| dpo - sft | capped_cost | 0.5417 | 6.3038 | 4/13/7 | 0.7227 | effect of preference stage |
| dpo - sft | log_area | 0.0301 | 0.3517 | 11/0/13 | 0.6760 | effect of preference stage |
| dpo - ranker | capped_cost | 1.5833 | 7.0212 | 2/14/8 | 0.3184 | LLM vs numerical ranker, same supervision |
| dpo - ranker | log_area | 0.0009 | 0.5423 | 15/0/9 | 0.9938 | LLM vs numerical ranker, same supervision |
| sft - ranker | capped_cost | 1.0417 | 8.4672 | 5/12/7 | 0.5708 | supervised LLM vs numerical ranker |
| sft - ranker | log_area | -0.0292 | 0.5209 | 14/0/10 | 0.7847 | supervised LLM vs numerical ranker |
| ranker - target_risk | capped_cost | 1.9583 | 6.6298 | 4/14/6 | 0.1934 | learned numerical vs planner |
| ranker - target_risk | log_area | 0.1315 | 0.4781 | 10/0/14 | 0.1900 | learned numerical vs planner |
| ranker - random | capped_cost | -0.6667 | 9.0346 | 6/13/5 | 0.7363 | learned numerical vs random |
| ranker - random | log_area | -0.0515 | 0.5639 | 11/0/13 | 0.6656 | learned numerical vs random |
| frozen - random | capped_cost | -0.6250 | 7.6090 | 6/13/5 | 0.7197 | observe-only vs random |
| frozen - random | log_area | 0.0092 | 0.5678 | 11/0/13 | 0.9372 | observe-only vs random |
| coverage - target_risk | capped_cost | 0.8333 | 3.0740 | 3/13/8 | 0.2256 | fixed open-loop cycle vs planner |
| coverage - target_risk | log_area | 0.0497 | 0.3514 | 7/0/17 | 0.5023 | fixed open-loop cycle vs planner |
| local_info - target_risk | capped_cost | 0.3333 | 6.2113 | 5/13/6 | 0.8398 | information vs target-risk planner |
| local_info - target_risk | log_area | 0.0310 | 0.4903 | 12/0/12 | 0.7610 | information vs target-risk planner |
| coverage - random | capped_cost | -1.7917 | 5.9416 | 6/16/2 | 0.1719 | fixed cycle vs random |
| coverage - random | log_area | -0.1333 | 0.5692 | 14/0/10 | 0.2735 | fixed cycle vs random |

None of the learned-arm contrasts (training recipe, supervised stage, preference stage, LLM versus numerical ranker) is distinguishable from zero on any of the four endpoints computed for them (capped cost at 0.04 and at 0.02, log-area, final error; smallest sign-flip p 0.24). Their sign also flips with the endpoint: dpo minus frozen is +1.54 cost units at threshold 0.04, -1.04 at threshold 0.02 (in `key_contrasts.csv`) and -0.06 on log-area. "DPO was worst" is a statement about the mean of one endpoint, not a robust ordering among the ranker, the three LLM arms and random. Over all ten pairs among those five arms and seven endpoints (script `a26_weakpairs.py`), 1 of 70 sign-flip tests is below 0.05 (sft has lower error than frozen at cost 12, p 0.043) and 2 are below 0.10, which is what chance alone would give.

### 4.4 Which endpoint is most sensitive?

Standardized paired effect d_z = mean / SD of (policy minus TR), all 24 systems. Larger means the endpoint separates that policy from TR with fewer systems (n for 80 percent power is about 7.85 / d_z^2 + 2).

| endpoint | random | dpo | frozen | sft | ranker | coverage | local_info |
|---|---|---|---|---|---|---|---|
| capped cost, threshold 0.04 (predeclared) | 0.48 | 0.55 | 0.30 | 0.50 | 0.30 | 0.27 | 0.05 |
| capped cost, threshold 0.02 | 0.68 | 0.69 | 0.61 | 0.48 | 0.60 | 0.45 | 0.28 |
| capped cost, threshold 0.01 | 0.70 | 0.50 | 0.62 | 0.62 | 0.48 | 0.72 | 0.55 |
| log-area, costs 9..16 | 0.43 | 0.23 | 0.47 | 0.32 | 0.30 | 0.16 | 0.09 |
| log-area, costs 9..32 | 0.37 | 0.28 | 0.50 | 0.29 | 0.28 | 0.14 | 0.06 |
| raw area, costs 9..32 | 0.40 | 0.43 | 0.23 | 0.16 | 0.30 | 0.10 | -0.09 |
| log NMSE at cost 12 | 0.29 | 0.18 | 0.56 | 0.18 | 0.27 | 0.22 | 0.10 |
| log NMSE at cost 16 | 0.34 | 0.13 | 0.20 | 0.08 | 0.06 | -0.03 | 0.04 |
| log NMSE at cost 24 | 0.35 | 0.38 | 0.48 | 0.16 | 0.19 | 0.16 | 0.00 |
| log final NMSE (cost 32) | 0.17 | 0.13 | 0.15 | 0.23 | 0.25 | -0.10 | 0.07 |

**The curve-area and fixed-cost error endpoints are not more sensitive than capped cost in this data.** They remove ties (no zero differences) but their standardized effects are smaller (random versus TR: 0.37 for log-area against 0.48 for capped cost). The largest standardized effects come from capped cost at a stricter threshold (0.02: 0.68 for random, 0.69 for dpo, 0.61 for frozen, 0.60 for ranker). This is a post hoc comparison of endpoint properties and must not be used to pick a threshold because it favors a policy.

## 5. Power planning (script `a14_power.py`)

Formulae, for a paired design with d_i = policy_i - reference_i, paired SD s_d and n independent systems:

- Superiority, two-sided alpha: n = ((z_(1-alpha/2) + z_(1-beta)) * s_d / delta)^2. With alpha 0.05 the multiplier is 2.8016 for 80 percent power and 3.2415 for 90 percent.
- Noninferiority with margin M, one-sided alpha 0.025 and assumed true difference D0: n = ((z_(1-alpha) + z_(1-beta)) * s_d / (M - D0))^2.
- The tables use the exact paired-t version (smallest n whose noncentral-t power reaches the target), which adds 2 to 3 systems to the z formula.
- With R replicate episodes per (system, policy) averaged within system, and replicate noise making up a share rho of the single-run paired variance: s_d(R)^2 = s_d^2 * ((1 - rho) + rho / R).

Observed paired SDs of (policy minus TR):

| policy - target_risk | cost mean | cost SD (24) | zeros of 24 | cost SD (inf 14) | log-area mean | log-area SD (24) | log-area SD (inf 14) |
|---|---|---|---|---|---|---|---|
| random | 2.625 | 5.428 | 12 | 6.560 | 0.183 | 0.490 | 0.455 |
| coverage | 0.833 | 3.074 | 13 | 3.975 | 0.050 | 0.351 | 0.343 |
| local_info | 0.333 | 6.211 | 13 | 8.253 | 0.031 | 0.490 | 0.468 |
| ranker | 1.958 | 6.630 | 14 | 8.527 | 0.132 | 0.478 | 0.449 |
| frozen | 2.000 | 6.763 | 11 | 8.698 | 0.192 | 0.385 | 0.468 |
| sft | 3.000 | 6.058 | 12 | 7.284 | 0.102 | 0.356 | 0.399 |
| dpo | 3.542 | 6.433 | 13 | 7.529 | 0.132 | 0.481 | 0.514 |

Capped cost to target, systems needed:

| paired SD used | superiority 1 unit, 80% | 1 unit, 90% | 2 units, 80% | 2 units, 90% | noninferiority margin 1, true diff 0, 80% | margin 1, 90% | margin 1, true diff +0.33, 80% |
|---|---|---|---|---|---|---|---|
| 3.07 (coverage, smallest) | 77 | 102 | 21 | 27 | 77 | 102 | 169 |
| 6.21 (local_info = median of 7) | 305 | 408 | 78 | 104 | 305 | 408 | 684 |
| 6.43 (dpo) | 327 | 437 | 84 | 111 | 327 | 437 | 733 |
| 6.76 (frozen, largest) | 361 | 483 | 92 | 123 | 361 | 483 | 810 |
| 7.53 (median, informative-only systems) | 447 | 598 | 114 | 151 | 447 | 598 | 1004 |

Curve-area endpoint (log-area), systems needed to detect a 10, 20 or 50 percent lower geometric-mean error over the budget:

| paired SD used (log10 units) | 10% lower, 80% | 10%, 90% | 20% lower, 80% | 20%, 90% | 50% lower, 80% |
|---|---|---|---|---|---|
| 0.351 (coverage, smallest) | 465 | 622 | 106 | 141 | 13 |
| 0.478 (ranker = median of 7) | 859 | 1150 | 193 | 258 | 22 |
| 0.490 (local_info, largest) | 904 | 1209 | 203 | 271 | 23 |

Effect of replicate episodes (cells are systems needed for 1 cost unit at SD 6.2 / systems needed for a 20 percent area improvement at SD 0.478, both 80 percent power):

| replicate-noise share of single-run paired variance | R=1 | R=2 | R=4 | R=8 | R=16 |
|---|---|---|---|---|---|
| 0.5 | 304 / 193 | 229 / 146 | 191 / 122 | 172 / 110 | 163 / 104 |
| 0.8 | 304 / 193 | 183 / 117 | 123 / 79 | 93 / 60 | 78 / 50 |

Notes on ties and assumptions:

- 11 to 14 of 24 cost differences are exactly zero, ten of them by construction. The difference distribution is a zero-inflated, heavy-tailed mixture bounded in [-24, 24]. At the sample sizes in the table the mean is close to normal, so the t-based numbers are reasonable for a test of the mean. A Wilcoxon test drops zeros, so its effective sample is the nonzero count (10 to 13 of 24 systems here).
- If a successor screen contains only informative systems, the per-system SD rises (median 7.5) but so does the per-system effect, because a mean saving of 1 unit over a 24-system mix equals 1.71 units on its 14 informative systems. Required informative systems for that same policy improvement: (2.8016 * 7.53 / 1.71)^2 = about 152, against 305 mixed systems of which about 178 are informative. Removing uninformative systems about halves the number of systems to run, but the number of informative systems needed stays about the same.
- The agenda's planning range of 30 to 50 systems per family across three families (90 to 150 systems, agenda line 159) gives a single-run minimum detectable difference of about 1.4 to 1.8 cost units at SD 6.2. That is 47 to 60 percent of the absolute ceiling of 3.04 units in this screen.
- Bootstrap check (resampling the 24 observed differences, two-sided 0.05): effects of the size actually observed between TR and random (2.6 units) reach about 0.76 to 0.82 power at 24 systems and 0.99 at 48. Effects the size of ranker minus TR or frozen minus TR (2.0 units, larger SD) reach about 0.2 at 24 systems and 0.9 at 96 with the t-test (0.5 to 0.7 with the Wilcoxon test). For the log-area endpoint, the TR versus random effect has power 0.44 at 24 systems and 0.96 at 96.
- The replicate share rho is not known precisely. Section 11.1 gives a direct but rough estimate: for the log-area endpoint over costs 9..18, the variance between two same-policy runs is 0.47 of the dpo-minus-TR paired variance (bootstrap 95 percent range 0.32 to 0.94). For the capped cost endpoint at cap 18 the point estimate is 0.81 but the range is 0.18 to 2.9. Treat rho = 0.5 as a planning value and measure it properly with deliberate twin arms.

## 6. Action structure of the strong policies (scripts `a07_actions.py`, `a08_actions2.py`, `a13_seqs.py`, `a18_perm.py`)

### 6.1 Distributions

Target risk, by graph and menu:

| cell | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| chain / heterogeneous | 39 | 37 | 1 | 13 | 0 | 0 | 2 | 1 | 93 |
| chain / uniform | 42 | 21 | 0 | 12 | 0 | 0 | 18 | 3 | 96 |
| collider / heterogeneous | 30 | 26 | 0 | 14 | 0 | 0 | 5 | 8 | 83 |
| collider / uniform | 24 | 15 | 1 | 20 | 10 | 1 | 20 | 5 | 96 |
| fork / heterogeneous | 64 | 16 | 0 | 16 | 0 | 0 | 0 | 0 | 96 |
| fork / uniform | 16 | 14 | 0 | 1 | 19 | 15 | 17 | 14 | 96 |

Target risk, by regime:

| cell | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| matched_high | 56 | 28 | 1 | 20 | 6 | 7 | 17 | 8 | 143 |
| matched_low | 34 | 36 | 1 | 25 | 9 | 1 | 21 | 9 | 136 |
| shifted_high | 66 | 35 | 0 | 6 | 9 | 8 | 14 | 3 | 141 |
| shifted_low | 59 | 30 | 0 | 25 | 5 | 0 | 10 | 11 | 140 |

Target risk, by step index:

| steps | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 1 | 25 | 0 | 11 | 2 | 0 | 2 | 7 | 48 |
| 3-4 | 10 | 14 | 1 | 9 | 1 | 0 | 7 | 6 | 48 |
| 5-8 | 34 | 25 | 1 | 11 | 4 | 3 | 16 | 2 | 96 |
| 9-12 | 44 | 18 | 0 | 10 | 5 | 3 | 11 | 5 | 96 |
| 13-16 | 43 | 18 | 0 | 11 | 7 | 5 | 8 | 4 | 96 |
| 17-24 | 83 | 29 | 0 | 24 | 10 | 5 | 18 | 7 | 176 |

Local information, by graph and menu:

| cell | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| chain / heterogeneous | 56 | 34 | 0 | 6 | 0 | 0 | 0 | 0 | 96 |
| chain / uniform | 51 | 32 | 0 | 11 | 0 | 0 | 2 | 0 | 96 |
| collider / heterogeneous | 51 | 28 | 0 | 5 | 0 | 0 | 5 | 1 | 90 |
| collider / uniform | 40 | 13 | 0 | 15 | 7 | 0 | 17 | 4 | 96 |
| fork / heterogeneous | 60 | 12 | 0 | 24 | 0 | 0 | 0 | 0 | 96 |
| fork / uniform | 60 | 30 | 0 | 6 | 0 | 0 | 0 | 0 | 96 |

Local information, by regime:

| cell | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| matched_high | 93 | 34 | 0 | 11 | 6 | 0 | 0 | 0 | 144 |
| matched_low | 64 | 41 | 0 | 23 | 1 | 0 | 11 | 1 | 141 |
| shifted_high | 101 | 33 | 0 | 5 | 0 | 0 | 3 | 2 | 144 |
| shifted_low | 60 | 41 | 0 | 28 | 0 | 0 | 10 | 2 | 141 |

Local information, by step index:

| steps | obs | x- | x0 | x+ | y- | y0 | y+ | xy | n |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 2 | 28 | 0 | 12 | 1 | 0 | 4 | 1 | 48 |
| 3-4 | 9 | 22 | 0 | 11 | 1 | 0 | 4 | 1 | 48 |
| 5-8 | 54 | 23 | 0 | 12 | 1 | 0 | 6 | 0 | 96 |
| 9-12 | 63 | 18 | 0 | 9 | 2 | 0 | 3 | 1 | 96 |
| 13-16 | 66 | 20 | 0 | 7 | 1 | 0 | 1 | 1 | 96 |
| 17-24 | 124 | 38 | 0 | 16 | 1 | 0 | 6 | 1 | 186 |

What good behavior looks like in this screen:

1. **Open with X at an extreme.** The first three actions are x- or x+ in 50 of 72 cases for TR and 58 of 72 for local_info.
2. **Choose the extreme on the far side of the natural X mean.** The posterior mean of X after initialization can be recovered from the saved initial predictions as (q_obs - q_do(X=0)) / (q_do(X=1) - q_do(X=0)) (an inference from saved predictions, exact for all three graphs). The first X-extreme action is on the side farther from that estimate in **23 of 24 TR runs and 24 of 24 local_info runs** (the single exception, w17, has an estimated mean of 0.015). Within a run the sign is stable: 19 of 24 TR runs and 21 of 24 local_info runs use a single sign. The estimated mean is positive in 17 of 24 systems, so x- is the far side in 17 and x+ in 7 (w03, w04, w07, w10, w12, w15, w20).
3. **Then interleave observation.** obs is 38 percent of TR's actions and 56 percent of local_info's. For TR it is 1 of 48 in steps 1-2, 35 percent in steps 5-8 and 45 to 47 percent afterwards. For local_info it is 2 of 48 in steps 1-2 and 56 to 69 percent from step 5.
4. **Use Y only when it is cheap.** Y-involving actions (y-, y0, y+, xy) are 42 percent of TR's actions in the uniform menu and 6 percent in the heterogeneous menu (10 percent and 2 percent for local_info).
5. **Graph matters.** In uniform-menu forks TR uses mostly Y and joint actions (65 of 96). In uniform-menu colliders it mixes x+, y+, y- and obs. local_info uses Y-involving actions almost only in colliders (34 of its 36).
6. **Observe more when the known noise is high.** The observation share is 39 and 47 percent for TR in matched_high and shifted_high against 25 and 42 percent in the two low-noise regimes. For local_info it is 65 and 70 percent against 45 and 43 percent.

Neither constant LLM policy can express this mix. Always-obs (frozen) matches the 38 to 56 percent observation share and none of the interventions. Always-x- (dpo) matches the X-extreme share only where x- is the far side.

### 6.2 How close to a fixed sequence?

Share of decisions equal to the modal action at the same step among the systems of the same cell (4 systems per graph-by-menu cell, 8 per graph). The in-sample modal share has a floor (random scores 0.40), so the leave-one-out (LOO) column is the honest one: each system's action is predicted from the other systems of its cell.

| policy | graph_x_cost_modal | graph_x_cost_loo | graph_modal | graph_loo | global_constant_action_share |
|---|---|---|---|---|---|
| random | 0.402 | 0.098 | 0.335 | 0.103 | 0.138 |
| coverage | 1.000 | 1.000 | 1.000 | 1.000 | 0.175 |
| local_info | 0.672 | 0.514 | 0.639 | 0.570 | 0.558 |
| target_risk | 0.563 | 0.375 | 0.473 | 0.395 | 0.384 |
| ranker | 0.891 | 0.859 | 0.884 | 0.870 | 0.755 |
| frozen | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| sft | 0.936 | 0.902 | 0.927 | 0.914 | 0.920 |
| dpo | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Per graph-by-menu cell, TR: chain/het 0.62 (LOO 0.48), chain/uni 0.58 (0.43), collider/het 0.58 (0.44), collider/uni 0.45 (0.20), fork/het 0.73 (0.63), fork/uni 0.42 (0.08). local_info: 0.73 (0.60), 0.64 (0.49), 0.68 (0.58), 0.55 (0.33), 0.70 (0.49), 0.74 (0.59).

- After merging mirror-image signs (x- with x+, y- with y+) the LOO agreement is 0.41 for TR and 0.58 for local_info, against 0.23 for random.
- A within-run shuffle of step order (which keeps each run's action mix) gives LOO 0.38 (95 percent range 0.34 to 0.44) for TR and 0.47 (0.41 to 0.53) for local_info on the sign-merged sequences. So TR's step-by-step agreement is explained by its action mix. Only local_info shows shared ordering beyond the mix, and that is the opening block of X interventions.
- Conclusion: the strong policies are **not** close to one fixed step sequence per graph and menu. Their shared structure is a mix plus an opening rule plus one sign bit. The conjugate posterior does not depend on the order of responses, so step-level disagreement overstates how state dependent the useful content is.

### 6.3 Does state dependence pay in this screen?

- menu coverage is a fully state-independent cycle (x-, x+, x0, xy, y-, y+, y0, repeated; identical in all 24 systems; it never observes). At the predeclared threshold it is within noise of TR on every endpoint: capped cost +0.83 (SD 3.07, sign-flip p 0.23), log-area +0.05 (p 0.50), final error slightly lower than TR (geometric mean 0.0072 against 0.0085).
- A difference appears only at a stricter threshold: at 0.02 coverage minus TR is +2.17 cost units (uncorrected sign-flip p 0.035, post hoc threshold).
- Reason (inference from the learner code): in `llm_intervention_scm.py` lines 87-92 the posterior covariance update uses only the regressor vector, not the measured outcome. With known noise and clamped parents the information from a design is fixed in advance. State enters only through naturally varying parents and through posterior means in composed predictions. In this model family a good static design is nearly optimal, so there is little for any adaptive policy, numerical or language, to gain.

### 6.4 The learned arms for comparison

- ranker: only x- (141) and x+ (435). 10 of 24 runs are exactly "x- seven times, then x+ seventeen times" and 3 runs are x+ throughout. x- never occurs after step 12. It never observes and never touches Y. Its majority sign is on the far side in only 10 of 24 runs.
- sft: x- on 481 of 523. x+ exactly once in 23 of 24 runs, at step 12 to 19 (nine times after exactly 12 paid acquisition units). y- once in all 8 colliders (steps 11 to 13) and in 3 chains, never in forks. obs only in three colliders after 18 or more paid units. Then stop after 21 to 23 acquisitions.
- dpo: x- on 576 of 576. frozen: obs on 574 of 576 (two x0 in w15).
- The numerical ranker trained on the matched examples also collapsed to X-only interventions with a step-indexed sign pattern. So the absence of observation and of state-dependent sign choice is shared by all three learned arms, language or not.
- This analysis did not examine the training data. Sibling notes that did (`/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training.md` lines 36, 45, 131 and 142, and `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training-forensics.md` lines 10-12; not verified by this agent) report that the 128 nonterminal supervised labels do contain observation (24) and x+ (29) next to x- (32), that the modal label changes with snapshot index (x- at the first decision, x+ at the second, observation at the third and fourth, stop at the fifth), that the ranker's training loss stayed near its uniform value ("barely trained"), and that the adapters already emitted near-constant commands on their own training states. If that is right, the shared collapse is a failure to fit state-dependent supervision rather than a property of the labels, and the ranker arm is not a meaningful "learned numerical" control.
- The positions of the sft deviations in deployment (x+ after about 12 paid units, observation after 18 or more, stop after 21 to 23 actions) line up with the training snapshot milestones 0, 6, 12, 18, 24 declared in `llm_intervention_screen_protocol.py` line 32 and with the label schedule reported in the sibling notes. sft appears to follow a budget-position schedule learned from 160 records at five budget positions, not a function of the posterior (inference). sft does respond to the graph (y- in colliders, not in forks), so it is not input-blind.

## 7. Where the constant x- policy (dpo) loses most and least (scripts `a10_dpo_strata.py`, `a17_xroot.py`, `a23_sign.py`, `a24_sign2.py`)

| stratum | n | informative n | DPO cost | TR cost | DPO-TR | DPO-TR (informative) | DPO-random | DPO-frozen | log-area DPO-TR | log final DPO-TR | DPO never attains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 8 | 3 | 14.125 | 13.000 | 1.125 | 3.000 | -2.875 | 0.250 | 0.099 | 0.099 | 2 |
| collider | 8 | 8 | 19.750 | 12.750 | 7.000 | 7.000 | 5.000 | 3.250 | 0.184 | 0.096 | 2 |
| fork | 8 | 3 | 11.625 | 9.125 | 2.500 | 6.667 | 0.625 | 1.125 | 0.114 | 0.123 | 1 |
| matched_high | 6 | 2 | 12.833 | 9.167 | 3.667 | 11.000 | 4.167 | 0.333 | -0.033 | -0.256 | 1 |
| matched_low | 6 | 3 | 14.167 | 8.667 | 5.500 | 11.000 | 3.333 | 2.333 | 0.377 | 0.342 | 1 |
| shifted_high | 6 | 4 | 18.667 | 15.833 | 2.833 | 4.250 | 0.167 | 2.333 | 0.287 | 0.433 | 2 |
| shifted_low | 6 | 5 | 15.000 | 12.833 | 2.167 | 2.600 | -4.000 | 1.167 | -0.101 | -0.094 | 1 |
| heterogeneous | 12 | 8 | 17.417 | 12.833 | 4.583 | 6.875 | 0.500 | 3.667 | 0.164 | 0.056 | 4 |
| uniform | 12 | 6 | 12.917 | 10.417 | 2.500 | 5.000 | 1.333 | -0.583 | 0.101 | 0.156 | 1 |

Graph by menu:

| cell | n | informative n | DPO cost | TR cost | DPO-TR | log-area DPO-TR | DPO never attains |
|---|---|---|---|---|---|---|---|
| chain / heterogeneous | 4 | 2 | 14.250 | 14.250 | 0.000 | 0.057 | 1 |
| chain / uniform | 4 | 1 | 14.000 | 11.750 | 2.250 | 0.142 | 1 |
| collider / heterogeneous | 4 | 4 | 23.000 | 14.250 | 8.750 | 0.051 | 2 |
| collider / uniform | 4 | 4 | 16.500 | 11.250 | 5.250 | 0.316 | 0 |
| fork / heterogeneous | 4 | 2 | 15.000 | 10.000 | 5.000 | 0.384 | 1 |
| fork / uniform | 4 | 1 | 8.250 | 8.250 | 0.000 | -0.156 | 0 |

- **Loses most:** colliders (+7.0 cost units against TR, all 8 informative, never attains on 2), heterogeneous-menu forks (+5.0), the matched regimes on informative systems (+11.0 in both, n = 3 and 2), heterogeneous menu overall (+4.6 against +2.5 uniform). dpo never attains on w03, w11, w12, w19 and w23.
- **Loses least:** chains (+1.1 overall, 0.0 in heterogeneous chains), uniform forks (0.0), shifted_low (+2.2). It beats TR on w17 (10 against 11) and w21 (12 against 15). Against random it is better on chains (-2.9) and in shifted_low (-4.0).
- Strata differ in how many systems are informative and use different parameter draws, so these are descriptions, not effects of graph or price.

### 7.1 Mechanism-level explanation

What set_x_neg can and cannot identify, from the learner's update rule (`llm_intervention_scm.py` lines 82-98):

| mechanism | effect of a response under do(X = -1) |
|---|---|
| X root (natural mean of X) | never updated, because the clamped node is skipped (lines 83-85) |
| children of X (Y in chains and forks, Z in forks and colliders) | updated, but every new regressor has X = -1, so only the child's value at X = -1 is learned. Intercept and slope are not separated beyond what the prior and the eight initial observations give |
| Z given Y in chains | updated with naturally varying Y around its X = -1 level, so the Z-on-Y coefficient is learned, with leverage limited by the noise in Y |
| Y root in colliders | updated (Y is not clamped), and the Z-on-Y coefficient is learned from natural Y |

Which queries need what:

| query | chain X -> Y -> Z | fork Y <- X -> Z | collider X -> Z <- Y |
|---|---|---|---|
| q0 observe | natural X mean and both slopes | natural X mean, intercept and slope of Z on X | natural X and Y means, all of Z's coefficients |
| q1 do(X=-1), q7 do(X=-1, Y=1) | q1: exactly what x- samples. q7 equals q6 | both equal what x- samples | q1 and q7 are learned (value at X = -1 plus Z-on-Y) |
| q2, q3 do(X=0), do(X=1) | need the slope of Y on X | need the slope of Z on X | need the slope of Z on X |
| q4-q6 do(Y=y) | only Z given Y (learned) | identical to q0 (natural X mean) | natural X mean times the slope of Z on X |

Direct check of the first row: the X-mean estimate recovered from saved predictions is **constant across all steps, to floating-point roundoff, in 24 of 24 dpo runs and 24 of 24 ranker runs** (median range 1e-15), in 13 of 24 sft runs, and in 0 of 24 runs of each of the other five policies, where its median range is 0.21 to 0.31 (script `a17_xroot.py`).

Empirical decomposition. Per-query error contributions were computed from reconstructed true target means (section 11.3). Mean contribution to NMSE at the final state, times 1000, by graph (8 systems each). "X-mean-dependent share" is the share of the row total carried by the queries whose true value depends on the natural X mean (q0 in chains; q0 and q4-q6 in forks and colliders).

| state | graph | q0 obs | q1 X=-1 | q2 X=0 | q3 X=+1 | q4 Y=-1 | q5 Y=0 | q6 Y=+1 | q7 X=-1,Y=+1 | total | X-mean-dependent share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| initial (all arms) | chain | 24.40 | 67.09 | 23.96 | 44.90 | 6.35 | 5.02 | 6.08 | 6.08 | 183.87 | 0.13 |
| initial (all arms) | fork | 2.78 | 6.98 | 5.03 | 7.68 | 2.78 | 2.78 | 2.78 | 6.98 | 37.80 | 0.29 |
| initial (all arms) | collider | 7.08 | 24.40 | 11.20 | 12.90 | 8.53 | 13.13 | 23.11 | 46.39 | 146.74 | 0.35 |
| dpo | chain | 39.89 | 2.35 | 1.89 | 9.01 | 0.75 | 1.50 | 2.83 | 2.83 | 61.05 | 0.65 |
| dpo | fork | 2.10 | 0.33 | 1.77 | 6.11 | 2.10 | 2.10 | 2.10 | 0.33 | 16.93 | 0.50 |
| dpo | collider | 6.85 | 1.49 | 1.13 | 2.68 | 4.10 | 5.78 | 8.17 | 1.07 | 31.26 | 0.80 |
| sft | chain | 28.31 | 5.41 | 1.41 | 2.19 | 0.38 | 0.49 | 1.01 | 1.01 | 40.22 | 0.70 |
| sft | fork | 2.27 | 0.55 | 1.74 | 5.67 | 2.27 | 2.27 | 2.27 | 0.55 | 17.61 | 0.52 |
| sft | collider | 3.49 | 1.44 | 2.58 | 4.31 | 2.92 | 3.08 | 4.41 | 1.18 | 23.41 | 0.59 |
| ranker | chain | 42.99 | 4.76 | 1.50 | 3.40 | 0.83 | 0.88 | 1.67 | 1.67 | 57.69 | 0.75 |
| ranker | fork | 1.61 | 0.58 | 0.18 | 0.37 | 1.61 | 1.61 | 1.61 | 0.58 | 8.16 | 0.79 |
| ranker | collider | 4.28 | 3.49 | 1.06 | 0.85 | 3.23 | 4.96 | 9.95 | 13.49 | 41.31 | 0.54 |
| frozen | chain | 2.94 | 10.37 | 3.49 | 1.30 | 1.14 | 0.53 | 0.37 | 0.37 | 20.52 | 0.14 |
| frozen | fork | 0.71 | 1.32 | 1.57 | 2.98 | 0.71 | 0.71 | 0.71 | 1.32 | 10.01 | 0.28 |
| frozen | collider | 1.10 | 4.71 | 3.90 | 5.34 | 1.01 | 2.42 | 5.68 | 5.55 | 29.70 | 0.34 |
| target_risk | chain | 6.13 | 9.67 | 3.82 | 3.32 | 0.45 | 0.49 | 0.90 | 0.90 | 25.69 | 0.24 |
| target_risk | fork | 2.04 | 0.60 | 0.57 | 1.38 | 2.04 | 2.04 | 2.04 | 0.60 | 11.33 | 0.72 |
| target_risk | collider | 2.90 | 1.55 | 1.68 | 2.54 | 1.22 | 1.04 | 1.44 | 2.25 | 14.62 | 0.45 |
| local_info | chain | 9.41 | 2.94 | 1.24 | 2.17 | 1.40 | 1.57 | 2.05 | 2.05 | 22.83 | 0.41 |
| local_info | fork | 1.82 | 1.30 | 0.67 | 1.42 | 1.82 | 1.82 | 1.82 | 1.30 | 11.97 | 0.61 |
| local_info | collider | 2.40 | 4.10 | 1.74 | 1.41 | 1.88 | 1.95 | 2.63 | 3.05 | 19.18 | 0.46 |
| coverage | chain | 10.14 | 9.69 | 2.01 | 5.98 | 0.78 | 0.68 | 1.19 | 1.19 | 31.66 | 0.32 |
| coverage | fork | 0.50 | 3.36 | 0.67 | 1.04 | 0.50 | 0.50 | 0.50 | 3.36 | 10.42 | 0.19 |
| coverage | collider | 2.87 | 3.06 | 1.46 | 1.32 | 1.66 | 1.78 | 2.95 | 2.04 | 17.16 | 0.54 |
| random | chain | 11.42 | 51.33 | 17.02 | 9.98 | 1.87 | 1.32 | 1.66 | 1.66 | 96.27 | 0.12 |
| random | fork | 0.79 | 1.07 | 0.67 | 0.91 | 0.79 | 0.79 | 0.79 | 1.07 | 6.88 | 0.46 |
| random | collider | 1.70 | 3.03 | 2.62 | 4.34 | 0.65 | 1.09 | 1.98 | 4.83 | 20.24 | 0.27 |

- **Chains:** dpo ends at 61.1 against 25.7 for TR. 39.9 of dpo's 61.1 (65 percent) is the observational query, against 6.1 for TR. The X = -1 query is learned better by dpo than by TR (2.4 against 9.7).
- **Forks:** dpo ends at 16.9 against 11.3. The excess is in the X = 0 and X = +1 queries (7.9 against 2.0), which need the slope that data at X = -1 cannot give. The X-mean-dependent queries are no worse than TR's (8.4 against 8.2).
- **Colliders:** dpo ends at 31.3 against 14.6. The excess is in the do(Y) queries (18.1 against 3.7) and the observational query (6.9 against 2.9). Both need the natural X mean times the slope of Z on X. The two X = -1 queries are learned best of any policy (2.6 in total against 3.8 for TR).
- The ranker (X-only, both signs) shows the same signature in chains (43.0 of 57.7 in q0) but not the fork slope problem (0.6 in X = 0 and +1), because it samples both X = -1 and X = +1. frozen (observe only) is the mirror image: the smallest q0 error in chains and colliders, and in chains more than four times dpo's error at X = -1 (10.4 against 2.4).
- **These means are driven by a few systems.** w19 and w12 hold 98 percent of dpo's X-mean-dependent final error in chains. w11 and w23 hold 75 percent in colliders. w21 and w20 hold 68 percent in forks. These are the systems where X has a strong total effect on Z and the eight-observation estimate of the X mean is off.

Fraction of the initial error that remains at the final state, as a ratio of sums over the eight systems of each graph (below 1 = reduced):

| policy | graph | q1,q7 (X=-1) | q2,q3 (X=0,+1) | q0 (observe) | q4-q6 (do Y) | all queries |
|---|---|---|---|---|---|---|
| dpo | chain | 0.071 | 0.158 | 1.635 | 0.291 | 0.332 |
| dpo | fork | 0.047 | 0.619 | 0.755 | 0.755 | 0.448 |
| dpo | collider | 0.036 | 0.158 | 0.966 | 0.403 | 0.213 |
| sft | chain | 0.088 | 0.052 | 1.160 | 0.108 | 0.219 |
| sft | fork | 0.079 | 0.583 | 0.817 | 0.817 | 0.466 |
| sft | collider | 0.037 | 0.286 | 0.492 | 0.232 | 0.160 |
| ranker | chain | 0.088 | 0.071 | 1.762 | 0.193 | 0.314 |
| ranker | fork | 0.083 | 0.043 | 0.580 | 0.580 | 0.216 |
| ranker | collider | 0.240 | 0.079 | 0.604 | 0.405 | 0.281 |
| frozen | chain | 0.147 | 0.070 | 0.121 | 0.117 | 0.112 |
| frozen | fork | 0.189 | 0.358 | 0.254 | 0.254 | 0.265 |
| frozen | collider | 0.145 | 0.384 | 0.156 | 0.203 | 0.202 |
| target_risk | chain | 0.145 | 0.104 | 0.251 | 0.106 | 0.140 |
| target_risk | fork | 0.086 | 0.154 | 0.735 | 0.735 | 0.300 |
| target_risk | collider | 0.054 | 0.175 | 0.409 | 0.083 | 0.100 |
| local_info | chain | 0.068 | 0.049 | 0.386 | 0.288 | 0.124 |
| local_info | fork | 0.186 | 0.164 | 0.654 | 0.654 | 0.317 |
| local_info | collider | 0.101 | 0.131 | 0.339 | 0.144 | 0.131 |
| coverage | chain | 0.149 | 0.116 | 0.416 | 0.152 | 0.172 |
| coverage | fork | 0.481 | 0.135 | 0.179 | 0.179 | 0.276 |
| coverage | collider | 0.072 | 0.116 | 0.405 | 0.143 | 0.117 |
| random | chain | 0.724 | 0.392 | 0.468 | 0.278 | 0.524 |
| random | fork | 0.154 | 0.124 | 0.283 | 0.283 | 0.182 |
| random | collider | 0.111 | 0.288 | 0.241 | 0.083 | 0.138 |

The same as the median over systems of the per-system ratio (robust to the dominant systems):

| policy | graph | q1,q7 (X=-1) | q2,q3 (X=0,+1) | q0 (observe) | q4-q6 (do Y) | all queries |
|---|---|---|---|---|---|---|
| dpo | chain | 0.193 | 0.186 | 4.406 | 0.630 | 0.468 |
| dpo | fork | 0.061 | 0.537 | 0.739 | 0.739 | 0.382 |
| dpo | collider | 0.023 | 0.173 | 0.360 | 0.464 | 0.184 |
| sft | chain | 0.143 | 0.098 | 5.360 | 0.275 | 0.231 |
| sft | fork | 0.095 | 0.872 | 0.836 | 0.836 | 0.568 |
| sft | collider | 0.037 | 0.148 | 0.577 | 0.209 | 0.111 |
| ranker | chain | 0.211 | 0.260 | 4.738 | 0.282 | 0.434 |
| ranker | fork | 0.118 | 0.166 | 1.221 | 1.221 | 0.112 |
| ranker | collider | 0.200 | 0.108 | 0.938 | 0.644 | 0.243 |
| frozen | chain | 0.224 | 0.392 | 2.010 | 0.148 | 0.184 |
| frozen | fork | 0.099 | 0.340 | 0.372 | 0.372 | 0.224 |
| frozen | collider | 0.222 | 0.427 | 0.192 | 0.144 | 0.197 |
| target_risk | chain | 0.171 | 0.413 | 11.777 | 0.239 | 0.238 |
| target_risk | fork | 0.060 | 0.408 | 0.170 | 0.170 | 0.161 |
| target_risk | collider | 0.080 | 0.081 | 0.309 | 0.066 | 0.091 |
| local_info | chain | 0.250 | 0.479 | 2.385 | 0.896 | 0.480 |
| local_info | fork | 0.045 | 0.147 | 0.428 | 0.428 | 0.171 |
| local_info | collider | 0.070 | 0.085 | 0.252 | 0.111 | 0.118 |
| coverage | chain | 0.108 | 0.201 | 2.572 | 0.117 | 0.210 |
| coverage | fork | 0.169 | 0.165 | 0.176 | 0.176 | 0.164 |
| coverage | collider | 0.107 | 0.116 | 0.526 | 0.162 | 0.120 |
| random | chain | 0.429 | 0.650 | 7.356 | 0.188 | 0.704 |
| random | fork | 0.244 | 0.154 | 0.209 | 0.209 | 0.227 |
| random | collider | 0.074 | 0.362 | 0.010 | 0.058 | 0.181 |

- The X = -1 queries are the best learned part of dpo under both views (3.6 to 7.1 percent of the initial error remains in sums, 2 to 19 percent in medians).
- In forks dpo retains 62 percent of the X = 0 and X = +1 error in sums (TR 15 percent) and 54 percent in medians (TR 41 percent).
- For q0 the two views differ. In sums dpo retains 164, 76 and 97 percent (chain, fork, collider) against 25, 74 and 41 percent for TR. In medians q0 gets worse in chains for every policy (4.4 for dpo, 11.8 for TR, 2.0 for frozen) because q0 starts out accurate after eight observational samples, and in colliders dpo's median (0.36) is close to TR's (0.31). The q0 deficit of dpo is real in the systems where it matters and absent in the typical system.
- Size of the frozen-X-mean problem: if every other mechanism were known exactly, the residual error from the initial X-mean estimate alone would exceed 0.04 on w11 (0.041), w12 (0.043) and w19 (0.145), and exceed 0.01 on w10, w20 and w23 as well. dpo never attains on w11, w12 and w19. The rank correlation of this residual with final error is 0.84 for dpo, 0.36 for the ranker and 0.21 for TR. This is an asymptotic residual, not a strict lower bound for a finite run, because errors in different mechanisms can cancel (the ranker ends w12 at 0.032).

### 7.2 The sign bit

Split by whether X = -1 is on the near (low-leverage) side of the estimated natural X mean (7 systems) or the far side (17 systems):

| policy - target_risk | log-area, near (n=7) | log-area, far (n=17) | cost, near | cost, far | log final, near | log final, far |
|---|---|---|---|---|---|---|
| random | 0.096 | 0.219 | 2.857 | 2.529 | -0.014 | 0.160 |
| coverage | 0.163 | 0.003 | 1.857 | 0.412 | 0.150 | -0.162 |
| local_info | 0.064 | 0.018 | -0.714 | 0.765 | 0.298 | -0.057 |
| ranker | 0.241 | 0.086 | -0.714 | 3.059 | 0.466 | 0.058 |
| frozen | 0.426 | 0.096 | 5.857 | 0.412 | 0.479 | -0.065 |
| sft | 0.359 | -0.004 | 6.286 | 1.647 | 0.608 | -0.063 |
| dpo | 0.535 | -0.033 | 5.429 | 2.765 | 0.786 | -0.174 |

- On the 17 far-side systems dpo's log-area equals TR's (-0.03) and its final error is lower (-0.17 log10). On the 7 near-side systems its geometric-mean error over the budget is 3.4 times TR's (+0.54) and its final error 6.1 times TR's (+0.79). Exact permutation p for the near-far gap is 0.0055 on log-area and 0.0076 on final error (post hoc, n = 7 against 17).
- In absolute terms dpo's geometric-mean final error is 0.028 on near-side systems and 0.0073 on far-side systems, with similar initial errors (0.057 and 0.046).
- Caution: TR itself does unusually well on the near-side systems (geometric-mean final error 0.0046 against 0.0109), so every policy looks worse relative to TR there (frozen +0.43 on log-area). The absolute dpo numbers above are the cleaner evidence.
- On capped cost dpo still loses on far-side systems (+2.8, +4.7 on the informative ones), consistent with the missing observation share.

## 8. Non-monotonicity (script `a11_nonmono.py`)

| policy | acquisitions | error_rose | rose_share | rose_gt2x | runs_crossed | crossed_then_left | crossed_but_final_above | exit_events | first_cross_lost_next_step |
|---|---|---|---|---|---|---|---|---|---|
| random | 487 | 212 | 0.435 | 26 | 21 | 9 | 1 | 10 | 5 |
| coverage | 480 | 196 | 0.408 | 21 | 22 | 6 | 1 | 8 | 1 |
| local_info | 570 | 262 | 0.460 | 30 | 23 | 11 | 5 | 20 | 4 |
| target_risk | 560 | 242 | 0.432 | 26 | 23 | 8 | 2 | 10 | 1 |
| ranker | 576 | 246 | 0.427 | 20 | 21 | 7 | 1 | 11 | 2 |
| frozen | 576 | 268 | 0.465 | 19 | 22 | 10 | 2 | 13 | 4 |
| sft | 523 | 235 | 0.449 | 21 | 21 | 6 | 2 | 7 | 2 |
| dpo | 576 | 248 | 0.431 | 21 | 19 | 4 | 1 | 5 | 1 |

- Error rose after **1,909 of 4,348 accepted acquisitions (43.9 percent)**. The share is 41 to 47 percent for every policy. After the very first acquisition the error rose in 64 of 192 runs.
- **61 of the 172 runs that ever crossed the target later left it** at least once, and 15 ended above it. local_info is the least stable (11 leavers, 5 ending above, 20 exit events).
- On the ten systems already at target after initialization, 20 of the 80 runs left the target at some later point and 2 ended above it (both local_info, w07 and w20). "At target after initialization" is itself a noisy state.
- First crossing is therefore a first-passage time of a noisy process and rewards early lucky dips. The sustained-cost endpoint in section 4 changes the policy ordering substantially.

## 9. The supervised arm (script `a12_sft.py`)

| sys | graph | regime | cost | term | stop_total_cost | acquisitions | init_err | err_at_stop | at_target_at_stop | ever | capped_cost | tr_err_at_same_cost | non_xneg_actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w00 | chain | matched_low | uniform | policy_stop | 29 | 21 | 0.0298 | 0.0076 | True | True | 8 | 0.0015 | 13:x+,14:y- |
| w01 | chain | matched_low | heterogeneous | policy_stop | 30 | 21 | 0.0060 | 0.0013 | True | True | 8 | 0.0010 | 13:x+,18:y- |
| w02 | fork | matched_low | uniform | policy_stop | 30 | 22 | 0.0377 | 0.0033 | True | True | 8 | 0.0026 | 13:x+ |
| w03 | fork | matched_low | heterogeneous | policy_stop | 29 | 21 | 0.0681 | 0.0374 | True | True | 26 | 0.0040 | 14:x+ |
| w04 | collider | matched_low | uniform | policy_stop | 29 | 21 | 0.0450 | 0.0045 | True | True | 13 | 0.0049 | 12:y-,17:x+,19:obs,20:obs,21:obs |
| w05 | collider | matched_low | heterogeneous | policy_stop | 30 | 21 | 0.0430 | 0.0045 | True | True | 11 | 0.0095 | 13:y-,14:x+,19:obs |
| w06 | chain | matched_high | uniform | policy_stop | 29 | 21 | 0.0003 | 0.0075 | True | True | 8 | 0.0074 | 14:x+ |
| w07 | chain | matched_high | heterogeneous | policy_stop | 31 | 22 | 0.0367 | 0.0069 | True | True | 8 | 0.0152 | 13:x+,18:y- |
| w08 | fork | matched_high | uniform | policy_stop | 30 | 22 | 0.0059 | 0.0090 | True | True | 8 | 0.0104 | 13:x+ |
| w09 | fork | matched_high | heterogeneous | policy_stop | 30 | 22 | 0.0379 | 0.0016 | True | True | 8 | 0.0126 | 14:x+ |
| w10 | collider | matched_high | uniform | policy_stop | 31 | 23 | 0.1250 | 0.0493 | False | True | 23 | 0.0189 | 12:y-,17:x+ |
| w11 | collider | matched_high | heterogeneous | policy_stop | 30 | 21 | 0.1107 | 0.0472 | False | True | 21 | 0.0085 | 11:y-,12:x+ |
| w12 | chain | shifted_low | uniform | policy_stop | 30 | 22 | 0.5219 | 0.0917 | False | False | 32 | 0.0168 |  |
| w13 | chain | shifted_low | heterogeneous | policy_stop | 31 | 23 | 0.3160 | 0.0039 | True | True | 10 | 0.0269 | 13:x+ |
| w14 | fork | shifted_low | uniform | policy_stop | 30 | 22 | 0.0411 | 0.0064 | True | True | 9 | 0.0684 | 18:x+ |
| w15 | fork | shifted_low | heterogeneous | policy_stop | 30 | 22 | 0.0129 | 0.0126 | True | True | 8 | 0.0019 | 14:x+ |
| w16 | collider | shifted_low | uniform | policy_stop | 30 | 22 | 0.0916 | 0.0033 | True | True | 10 | 0.0060 | 12:y-,13:x+ |
| w17 | collider | shifted_low | heterogeneous | budget_exhausted | 32 | 23 | 0.1115 | 0.0132 | True | True | 9 | 0.0445 | 13:y-,15:x+ |
| w18 | chain | shifted_high | uniform | policy_stop | 29 | 21 | 0.0154 | 0.0038 | True | True | 8 | 0.0056 | 13:x+ |
| w19 | chain | shifted_high | heterogeneous | policy_stop | 30 | 22 | 0.5449 | 0.1991 | False | False | 32 | 0.1295 | 14:x+ |
| w20 | fork | shifted_high | uniform | policy_stop | 30 | 22 | 0.0209 | 0.0123 | True | True | 8 | 0.0003 | 19:x+ |
| w21 | fork | shifted_high | heterogeneous | policy_stop | 30 | 22 | 0.0778 | 0.0583 | False | False | 32 | 0.0317 | 19:x+ |
| w22 | collider | shifted_high | uniform | policy_stop | 30 | 22 | 0.4139 | 0.0304 | True | True | 19 | 0.0060 | 12:x+,13:y-,19:obs,20:obs,21:obs,22:obs |
| w23 | collider | shifted_high | heterogeneous | policy_stop | 31 | 22 | 0.2332 | 0.0349 | True | True | 24 | 0.0133 | 12:x+,13:y- |

- It stopped in 23 of 24 systems (not w17), at total cost 29 (5 systems), 30 (14) or 31 (4), leaving 1 to 3 units, 47 in total (768 - 721).
- The stop comes after 21 (8 runs), 22 (13) or 23 (2) acquisitions in both menus (means 21.75 uniform, 21.73 heterogeneous). It tracks the number of steps taken, not the error: correlation of stop cost with log error at the stop is 0.14.
- Error at the stop: geometric mean 0.0115, median 0.0076. 18 of 23 stops were at target.
- **Five stops were above target**: w10 (0.049), w11 (0.047), w12 (0.092), w19 (0.199), w21 (0.058). w10 and w11 had crossed earlier (capped costs 23 and 21). w12, w19 and w21 never crossed. TR's final error on those five systems was 0.021, 0.012, 0.021, 0.122 and 0.016.
- On average the sft error at its stop is 0.17 log10 units (48 percent) above TR's error at the same paid cost.
- The stop is a late, budget-linked behavior. It saves at most 3 units and is not evidence of an accuracy-aware stopping rule.

## 10. Recommendations (labeled as recommendations, not findings)

**R1. Fix the sensitivity of the headline endpoint before spending on a new policy.** Keep experimental cost to a target accuracy as the headline, as the agenda requires, but calibrate initialization size and threshold on training-namespace or artificial systems, and freeze them before any development outcome. Suggested acceptance rule for the calibration: at most about 10 percent of systems at target after initialization, and the reference planner reaches the target within budget on at least about 90 percent. In the saved data a threshold of 0.02 would leave 5 of 24 systems at target after initialization and 1 never reached by TR, against 10 and 1 at 0.04. Threshold sweep:

| threshold | at target after init | TR never attains | no policy attains | informative for TR | TR mean cost | best-of-8 mean | TR minus floor | random-TR mean | random-TR SD | random-TR d_z | local_info-TR SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0025 | 1 | 14 | 5 | 9 | 26.875 | 18.708 | 17.917 | -0.500 | 9.855 | -0.051 | 10.069 |
| 0.0050 | 1 | 9 | 3 | 14 | 23.542 | 16.083 | 14.583 | -0.875 | 9.018 | -0.097 | 9.529 |
| 0.0075 | 3 | 6 | 3 | 15 | 18.625 | 14.625 | 9.750 | 2.833 | 6.605 | 0.429 | 8.506 |
| 0.0100 | 3 | 4 | 3 | 17 | 16.167 | 13.833 | 7.292 | 4.458 | 6.400 | 0.697 | 7.442 |
| 0.0150 | 4 | 2 | 1 | 18 | 14.125 | 12.000 | 5.292 | 4.500 | 6.916 | 0.651 | 7.226 |
| 0.0200 | 5 | 1 | 0 | 18 | 13.542 | 11.583 | 4.750 | 4.500 | 6.653 | 0.676 | 7.012 |
| 0.0300 | 7 | 1 | 0 | 16 | 12.167 | 10.458 | 3.458 | 4.583 | 6.795 | 0.675 | 6.366 |
| 0.0400 | 10 | 1 | 0 | 13 | 11.625 | 9.583 | 3.042 | 2.625 | 5.428 | 0.484 | 6.211 |
| 0.0600 | 13 | 1 | 0 | 10 | 10.917 | 9.000 | 2.458 | 0.750 | 5.351 | 0.140 | 4.422 |
| 0.0800 | 15 | 0 | 0 | 9 | 9.500 | 8.750 | 1.125 | 1.500 | 5.167 | 0.290 | 2.430 |

Policy means by threshold (first crossing, capped at 32):

| threshold | random | coverage | local_info | target_risk | ranker | frozen | sft | dpo |
|---|---|---|---|---|---|---|---|---|
| 0.0025 | 26.375 | 27.500 | 26.417 | 26.875 | 25.667 | 28.417 | 28.042 | 26.792 |
| 0.0050 | 22.667 | 24.750 | 24.792 | 23.542 | 23.333 | 24.958 | 26.583 | 26.458 |
| 0.0075 | 21.458 | 21.167 | 23.083 | 18.625 | 22.792 | 22.792 | 24.000 | 23.750 |
| 0.0100 | 20.625 | 20.042 | 20.250 | 16.167 | 20.417 | 22.167 | 20.708 | 21.500 |
| 0.0150 | 18.625 | 17.042 | 16.833 | 14.125 | 18.667 | 20.125 | 18.125 | 19.292 |
| 0.0200 | 18.042 | 15.708 | 15.500 | 13.542 | 17.917 | 19.292 | 16.917 | 18.250 |
| 0.0300 | 16.750 | 13.708 | 13.375 | 12.167 | 16.083 | 15.542 | 16.208 | 15.708 |
| 0.0400 | 14.250 | 12.458 | 11.958 | 11.625 | 13.583 | 13.625 | 14.625 | 15.167 |
| 0.0600 | 11.667 | 10.958 | 10.000 | 10.917 | 12.250 | 11.125 | 12.042 | 13.250 |
| 0.0800 | 11.000 | 10.292 | 9.583 | 9.500 | 11.500 | 10.417 | 10.500 | 12.542 |

**R2. Shrink the initialization rather than only moving the threshold.** The frozen arm is in effect a larger observational initialization: after 12, 16, 24 and 32 total observations, 12, 16, 18 and 20 of 24 systems are at or below 0.04. More initial data makes more systems trivial. Fewer initial observations (for example 4) is the direction that creates headroom. This needs a qualification run on non-evaluation systems because the saved curves start at 8.

**R3. Replace the fixed scale by a public, predeclared per-family scale, or declare a public triviality rule.** With scale 2, three systems meet the target with an all-zero prediction and six with a constant. Do not drop such systems after the fact. Either normalize by a quantity known before data (for example the prior-predictive spread of the target under the declared parameter law), or stratify on it prospectively.

**R4. Replicate within system and pair the noise.** (a) Run R of about 8 independent measurement-noise replicates per (system, arm) and analyze the per-system mean, with systems as the unit. (b) Key the measurement noise by (system, action, occurrence index) instead of by arm, so two arms that take the same multiset of actions reach the same posterior. Today the arms use different noise namespaces (protocol lines 148-152) and 0 of 293 same-first-action arm pairs produced identical predictions. (c) Include one deliberate twin arm (a second seed of the reference planner) so the replicate share of variance is measured rather than inferred.

**R5. Endpoints for a successor.** Primary: capped cost to target on the recalibrated threshold, computed per replicate and averaged per system. Secondary, declared in advance: sustained cost to target, log-area over the budget, and error at one or two fixed budgets. Report medians or geometric means of final error next to means. The log-area endpoint is useful because it has no ties and no threshold, but in this data it is less sensitive than capped cost, so it should not be promoted to primary on the expectation of more power.

**R6. Sample size.** For a superiority claim of 1 cost unit over the planner with single runs and SD near 6.2: about 305 systems for 80 percent power and 408 for 90 percent. For 2 units: 78 and 104. For a noninferiority claim with a 1 unit margin: 305 and 408 if the policies are truly equal, and far more if the new policy is slightly worse. With R = 8 replicates and a replicate share of 0.5 to 0.8: about 93 to 172 systems for 1 unit. For the log-area endpoint: about 193 systems for a 20 percent improvement with single runs, 60 to 110 with R = 8. A 24-system screen should be described as a behavior and variance pilot only.

**R7. Choose a successor environment in which adaptivity can matter.** In this linear-Gaussian, known-noise, known-graph family a fixed cycle is within noise of the planner at the predeclared threshold, and the absolute ceiling above the planner is 3 cost units. A positive LLM claim against the planner is close to unattainable here by construction. The realistic claims in this family are (a) noninferiority to the planner at lower decision compute, which needs the sample sizes in R6, or (b) nothing. A superiority claim needs a setting where the planner's working model is wrong or incomplete, the action space is too large to enumerate, or semantic information that the numerical policy cannot use is informative.

**R8. Add cheap controls that this analysis shows are decisive.** Constant-action arms (always x-, always obs) as explicit named controls. A static heuristic "X at the far-side extreme three times, then alternate observation and far-side X". The fixed coverage cycle. An oracle design computed from true parameters for an honest upper bound. These separate "the LLM uses the state" from "the LLM emits a good default".

**R9. Use the sign bit as the first state-sensitivity probe.** The strong policies' most consistent state-dependent decision is one bit read from the public state: the sign of the posterior mean of X. A diagnostic that flips only that quantity in otherwise identical prompts, and checks whether the frozen, supervised and preference-trained command scores for set_x_neg and set_x_pos swap, tests state use directly and does not touch the 24 development outcomes. A second probe: does the score of observe_all rise once two or three X interventions are in the history.

**R10. Keep both TR and local_info as numerical references.** The saved data cannot rank them. Declaring only TR as "the strong baseline" risks a successor beating TR while losing to local_info on the informative systems.

## 11. Additional findings not requested but relevant

### 11.1 The arms do not share measurement noise, and same-policy twins exist by accident (scripts `a04_noise_twins.py`, `a05_twin_cost.py`, `a22_misc.py`)

- Among 293 pairs of arms that took the same first action in the same system, none produced identical post-action predictions. Arms are paired by system and initial data only.
- dpo and sft executed the **identical action prefix** (x- repeated) for at least 10 actions in every system and at least 12 in 18 systems (313 matched points). They are two independent-noise replicates of one policy up to total cost 18.
- Their log10 error differs with SD 0.58 after one identical action, 0.35 after two, 0.49 after four, 0.38 after eight. The implied single-run replicate SD of log10 NMSE at a fixed cost is about 0.25 to 0.40 (a factor of 1.8 to 2.5 in error). Pooled over all matched points it is 0.31.
- Capped cost with the cap moved to 18: twin differences on the informative systems are 0, -2, 7, -5, 0, 0, -1, 0, 8, 1, 0, -6, 0, 0. Seven of 14 differ. Examples inside the identical prefix: w05 crossed at 18 and 11, w16 not by 19 and at 10, w21 at 12 and not by 26, w10 at 13 and not by 19.
- Twin variance at cap 18 is 7.50 against 9.28 for dpo minus TR at the same cap (ratio 0.81, bootstrap range 0.18 to 2.9). The dpo arm's own replicate variance alone is 40 percent of that paired variance.
- For the log-area over costs 9..18, the twin SD is 0.31 against 0.31 to 0.54 for differences between different policies. Twin variance is 0.47 of the dpo-minus-TR variance (range 0.32 to 0.94). For close competitors (coverage minus TR SD 0.31, sft minus TR 0.33) the paired difference is no larger than pure replicate noise.

### 11.2 Budget use

- On the informative 14, TR's acquisition cost to target is 1, 1, 1, 1, 1, 2, 2, 3, 6, 7, 9, 14, 15, 24 (median 2.5). It crosses with its first response on 5 systems and within three units on 8. For most systems the endpoint is decided by one to three actions, which is a very coarse scale for a 1-unit effect.
- The executed budget (8 initial plus 24 further units) is much shorter than the design values in the November 2 agenda (16 plus 64, agenda line 153). The handoff states that the agenda's proposed counts are superseded.

### 11.3 True target means can be reconstructed from the saved files (script `a09_truth.py`)

Because 32 * NMSE = sum over queries of (prediction - truth)^2, every saved curve point gives one equation that is linear in the eight true means and their sum of squares. Least squares over the 174 to 199 points of each system recovers the eight true target means with maximum reconstruction error 2.9e-15 over all 4,540 points, and the recovered means satisfy the structural identities of each graph to 1e-12. This is an inference from saved outputs, not a read of private descriptors. It enabled the per-query decomposition in section 7. It also means the saved development reports are not blind with respect to target truth. That is harmless for a closed development screen, but a confirmation set should not expose per-step predictions together with per-step errors to anyone who can influence methods.

## 12. Consistency with the project documents

- All numbers quoted in `/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md` lines 25-42 and in `/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md` lines 13-38 were reproduced: the policy table, dpo against TR 2 lower / 13 tied / 9 higher, dpo against random 3 / 13 / 8, the action counts, the 23 sft stops with five above target, dpo worse than TR and than frozen in each of the four regimes. No numerical disagreement was found.
- Statements that this analysis qualifies rather than contradicts:
  - "Direct SCM planning performed better" holds for the mean of the predeclared endpoint. After Holm correction only dpo is separated from TR (0.048). On that endpoint TR is not separated from local_info, coverage, ranker or frozen (sign-flip p 0.18 to 0.84). On post hoc endpoints frozen is nominally worse than TR on log-area (0.023) and on the error at cost 12 (0.012), and the ranker (0.0015) and coverage (0.035) are nominally worse at threshold 0.02. local_info is never separated from TR and is better on the informative systems for error-based endpoints.
  - "DPO was worst" holds for mean capped cost at threshold 0.04 only (section 4.3).
  - "Ten systems already met the target ... limits how much the average endpoint can distinguish" is correct and understates the issue: at the typical paired SD the minimum detectable difference at 24 systems (3.7 units) exceeds the maximum possible improvement over TR (3.04 units).
- The only document-level difference noticed is the budget (agenda 16 + 64 design values against the executed 8 + 24), which the handoff already declares superseded.

## 13. Open questions

1. What is the replicate share of the paired variance for the strong policies? Only the constant x- policy has accidental twins.
2. How much of the best-of-8 envelope survives replication? Is there any real headroom above TR in this family?
3. Would a static "far-side X, then alternate with observation" design match TR? It cannot be evaluated from saved data.
4. Why did the ranker, sft and dpo all collapse to X-only schedules when (per the sibling training notes) the labels contain observation and both X signs? A numerical learner that demonstrably fits the supervision is needed before "LLM versus numerical learner" can be interpreted.
5. What initialization size and threshold make at least 90 percent of systems informative for the three graphs? Needs a qualification run on non-evaluation systems.
6. Does the near-side versus far-side split of dpo's deficit hold on fresh systems? It rests on 7 against 17 systems and was found post hoc.
