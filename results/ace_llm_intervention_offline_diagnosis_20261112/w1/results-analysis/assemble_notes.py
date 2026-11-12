import json
T = json.load(open('tables.json'))
SCR = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis'
OUT = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis.md'
D = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01'
doc = []
A = doc.append
A(f"""# Quantitative re-analysis of the saved 24-system comparison (label: results-analysis)

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

- `{D}/cells.csv` (192 rows), `curves.csv` (4,540 rows), `policies.csv`, `pairs.csv`, `strata.csv`, `actions.csv`.
- Scope of the earlier arithmetic review: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/numerical-review01/review.md`. That review reconstructed counts, pair differences and group means, and checked the CSV exports and the static canvas (21,045 assertions). Line 23 states its limits: saved errors were taken as inputs, no ground truth was re-evaluated, and (line 15) no significance conclusion was drawn. It contains no paired tests, no alternative endpoints, no power analysis, no action-sequence analysis and no variance decomposition. Everything in sections 3 to 9 below is new relative to that review.
- Design facts used for interpretation: `/Users/pat/code/ACE/scripts/research/llm_intervention_screen_protocol.py` (regimes at lines 27-28, graphs at 57-61, menu and costs at 64-71, queries, weights, scale 2 and tolerance 0.04 at 76-77, 8 initial observations and 24 further units at 87-88, arm-specific measurement namespaces at 148-152), `/Users/pat/code/ACE/scripts/research/llm_intervention_scm.py` (update rule at lines 82-98) and `/Users/pat/code/ACE/scripts/research/llm_intervention_target_risk.py`.

Scripts and their saved outputs are in `{SCR}/` (`common.py`, `a01_sanity.py` ... `a26_weakpairs.py`, each with `*_out.txt`; table generators `make_tables.py`, `make_tables2.py`, `make_tables3.py` and `assemble_notes.py`; generated tables in `tables.json`; CSV exports `persystem.csv`, `paired_endpoints.csv`, `key_contrasts.csv`, `threshold_sweep.csv`, `sft_stops.csv`, `nonmonotonicity.csv`, `fixed_sequence_agreement.csv`).

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

{T['policy_summary']}

The mean final error is driven by a few systems. By median or geometric mean the ordering changes: sft has the second lowest median final error (0.0083), and dpo, frozen and random have nearly the same geometric mean (0.0108, 0.0105, 0.0109).

## 2. Per-system table (script `a02_persystem.py`)

Capped total cost to target per policy. "init NMSE" is the error after the eight shared initial observations. best8 is the per-system minimum over the eight arms.

{T['per_system']}

The ten systems already at target after initialization are w00, w01, w02, w06, w07, w08, w09, w15, w18 and w20. All eight policies score 8 on each of them, and no policy scores 8 anywhere else.

{T['uninformative']}

- By graph: 5 of 8 chains, 5 of 8 forks, **0 of 8 colliders**. Every collider is informative.
- By regime: matched_low 3 of 6, matched_high 4 of 6, shifted_low 1 of 6, shifted_high 2 of 6. So 7 of the 12 matched (edge mean 0) systems and 3 of the 12 shifted (edge mean 1.5) systems are uninformative.
- By menu: uniform 6 of 12, heterogeneous 4 of 12.
- Initial error over the 24 systems: minimum 0.00028 (w06), median 0.0440, mean 0.1228, maximum 0.5449 (w19).
- Three more systems start within 12.5 percent above the threshold: w14 (0.0411), w05 (0.0430), w04 (0.0450). On w14 seven of the eight policies cross with their first response. In practice about 11 of 24 systems cannot separate policies on the predeclared endpoint.
- Number of systems at target after initialization for other thresholds: 0.005: 1, 0.01: 3, 0.02: 5, 0.03: 7, 0.04: 10, 0.06: 13, 0.08: 15.

Why these systems are easy (script `a21_scale.py`; uses reconstructed true target means, see section 11.3): the median standard deviation of the eight true target means is 0.49 on the ten at-target systems and 1.73 on the others (Spearman correlation of initial error with true target spread 0.70). On three systems (w02, w06, w18) the prior-mean prediction of zero for every query already meets the 0.04 target. On six systems (w02, w03, w06, w07, w08, w18) a single constant equal to the mean of the eight true targets meets it. The fixed public scale of 2 makes weak-effect systems trivial regardless of intervention choice.

## 3. Headroom above target risk (scripts `a03_headroom.py`, `a16_sensitivity.py`)

"floor" is the trivial lower bound: 8 if at target after initialization, otherwise 9 (one unit-cost response).

{T['headroom']}

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

{T['endpoint_means_all24']}

Informative 14:

{T['endpoint_means_inf14']}

Observations:

- TR is first on capped cost and log-area over all 24, but it is **not robustly the best numerical policy**. On the informative 14, local_info is first on log-area (-1.755 versus -1.624 for TR) and on the error at costs 16, 24 and 32. TR and local_info are statistically indistinguishable on every endpoint (capped cost difference 0.33, SD 6.2, sign-flip p 0.84).
- The ordering of the weak arms depends on the endpoint. dpo is 8th on capped cost, 6th on log-area, 5th on geometric-mean final error. frozen is 5th on capped cost and 8th on log-area. On sustained cost the order is coverage 15.1, TR 15.7, ranker 15.7, sft 16.2, frozen 17.0, dpo 17.0, local_info 17.8, random 18.4, and no pair is distinguishable (paired SDs 7 to 11).
- Per-system winners are close to uniform across policies. Lowest log-area: TR 4, frozen 4, dpo 4, random 3, ranker 3, coverage 2, local_info 2, sft 2. Lowest final error: TR 4, random 4, dpo 4, local_info 3, sft 3, coverage 2, ranker 2, frozen 2. Single-run system-level outcomes are noise dominated.

### 4.1 Paired differences against target risk

Capped cost to target (predeclared endpoint):

{T['paired_capped_cost_target_risk']}

Holm correction over these seven comparisons gives dpo 0.048, random 0.067, sft 0.110, all others above 0.7. So on the predeclared endpoint TR is separated from dpo, random and sft at nominal level and from nothing else. 11 to 14 of the 24 differences are exact zeros in every comparison.

log-area (mean log10 NMSE over costs 9..32):

{T['paired_log_area_target_risk']}

log10 NMSE at total cost 12:

{T['paired_log_e12_target_risk']}

log10 NMSE at total cost 16:

{T['paired_log_e16_target_risk']}

log10 NMSE at total cost 24:

{T['paired_log_e24_target_risk']}

log10 final NMSE (cost 32):

{T['paired_log_e32_target_risk']}

Sustained cost:

{T['paired_sustained_cost_target_risk']}

### 4.2 Paired differences against uniform random

Capped cost to target:

{T['paired_capped_cost_random']}

log-area:

{T['paired_log_area_random']}

log10 NMSE at total cost 12:

{T['paired_log_e12_random']}

log10 NMSE at total cost 16:

{T['paired_log_e16_random']}

log10 NMSE at total cost 24:

{T['paired_log_e24_random']}

log10 final NMSE (cost 32):

{T['paired_log_e32_random']}

**Against random, only TR is nominally better on the predeclared endpoint (sign-flip p 0.011).** The ranker, frozen, sft and dpo are not separated from random on any endpoint examined (over ten endpoints the smallest sign-flip p is 0.09, sft on the natural-scale area, and the smallest Wilcoxon p is 0.19). The screen can barely tell its best numerical policy from random.

### 4.3 Contrasts that carry the scientific claims

{T['contrasts']}

None of the learned-arm contrasts (training recipe, supervised stage, preference stage, LLM versus numerical ranker) is distinguishable from zero on any of the four endpoints computed for them (capped cost at 0.04 and at 0.02, log-area, final error; smallest sign-flip p 0.24). Their sign also flips with the endpoint: dpo minus frozen is +1.54 cost units at threshold 0.04, -1.04 at threshold 0.02 (in `key_contrasts.csv`) and -0.06 on log-area. "DPO was worst" is a statement about the mean of one endpoint, not a robust ordering among the ranker, the three LLM arms and random. Over all ten pairs among those five arms and seven endpoints (script `a26_weakpairs.py`), 1 of 70 sign-flip tests is below 0.05 (sft has lower error than frozen at cost 12, p 0.043) and 2 are below 0.10, which is what chance alone would give.

### 4.4 Which endpoint is most sensitive?

Standardized paired effect d_z = mean / SD of (policy minus TR), all 24 systems. Larger means the endpoint separates that policy from TR with fewer systems (n for 80 percent power is about 7.85 / d_z^2 + 2).

{T['dz']}

**The curve-area and fixed-cost error endpoints are not more sensitive than capped cost in this data.** They remove ties (no zero differences) but their standardized effects are smaller (random versus TR: 0.37 for log-area against 0.48 for capped cost). The largest standardized effects come from capped cost at a stricter threshold (0.02: 0.68 for random, 0.69 for dpo, 0.61 for frozen, 0.60 for ranker). This is a post hoc comparison of endpoint properties and must not be used to pick a threshold because it favors a policy.

## 5. Power planning (script `a14_power.py`)

Formulae, for a paired design with d_i = policy_i - reference_i, paired SD s_d and n independent systems:

- Superiority, two-sided alpha: n = ((z_(1-alpha/2) + z_(1-beta)) * s_d / delta)^2. With alpha 0.05 the multiplier is 2.8016 for 80 percent power and 3.2415 for 90 percent.
- Noninferiority with margin M, one-sided alpha 0.025 and assumed true difference D0: n = ((z_(1-alpha) + z_(1-beta)) * s_d / (M - D0))^2.
- The tables use the exact paired-t version (smallest n whose noncentral-t power reaches the target), which adds 2 to 3 systems to the z formula.
- With R replicate episodes per (system, policy) averaged within system, and replicate noise making up a share rho of the single-run paired variance: s_d(R)^2 = s_d^2 * ((1 - rho) + rho / R).

Observed paired SDs of (policy minus TR):

{T['paired_sd']}

Capped cost to target, systems needed:

{T['power_cost']}

Curve-area endpoint (log-area), systems needed to detect a 10, 20 or 50 percent lower geometric-mean error over the budget:

{T['power_area']}

Effect of replicate episodes (cells are systems needed for 1 cost unit at SD 6.2 / systems needed for a 20 percent area improvement at SD 0.478, both 80 percent power):

{T['power_reps']}

Notes on ties and assumptions:

- 11 to 14 of 24 cost differences are exactly zero, ten of them by construction. The difference distribution is a zero-inflated, heavy-tailed mixture bounded in [-24, 24]. At the sample sizes in the table the mean is close to normal, so the t-based numbers are reasonable for a test of the mean. A Wilcoxon test drops zeros, so its effective sample is the nonzero count (10 to 13 of 24 systems here).
- If a successor screen contains only informative systems, the per-system SD rises (median 7.5) but so does the per-system effect, because a mean saving of 1 unit over a 24-system mix equals 1.71 units on its 14 informative systems. Required informative systems for that same policy improvement: (2.8016 * 7.53 / 1.71)^2 = about 152, against 305 mixed systems of which about 178 are informative. Removing uninformative systems about halves the number of systems to run, but the number of informative systems needed stays about the same.
- The agenda's planning range of 30 to 50 systems per family across three families (90 to 150 systems, agenda line 159) gives a single-run minimum detectable difference of about 1.4 to 1.8 cost units at SD 6.2. That is 47 to 60 percent of the absolute ceiling of 3.04 units in this screen.
- Bootstrap check (resampling the 24 observed differences, two-sided 0.05): effects of the size actually observed between TR and random (2.6 units) reach about 0.76 to 0.82 power at 24 systems and 0.99 at 48. Effects the size of ranker minus TR or frozen minus TR (2.0 units, larger SD) reach about 0.2 at 24 systems and 0.9 at 96 with the t-test (0.5 to 0.7 with the Wilcoxon test). For the log-area endpoint, the TR versus random effect has power 0.44 at 24 systems and 0.96 at 96.
- The replicate share rho is not known precisely. Section 11.1 gives a direct but rough estimate: for the log-area endpoint over costs 9..18, the variance between two same-policy runs is 0.47 of the dpo-minus-TR paired variance (bootstrap 95 percent range 0.32 to 0.94). For the capped cost endpoint at cap 18 the point estimate is 0.81 but the range is 0.18 to 2.9. Treat rho = 0.5 as a planning value and measure it properly with deliberate twin arms.

## 6. Action structure of the strong policies (scripts `a07_actions.py`, `a08_actions2.py`, `a13_seqs.py`, `a18_perm.py`)

### 6.1 Distributions

Target risk, by graph and menu:

{T['actions_target_risk_graphcost']}

Target risk, by regime:

{T['actions_target_risk_regime']}

Target risk, by step index:

{T['actions_target_risk_step']}

Local information, by graph and menu:

{T['actions_local_info_graphcost']}

Local information, by regime:

{T['actions_local_info_regime']}

Local information, by step index:

{T['actions_local_info_step']}

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

{T['fixed_seq']}

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

{T['dpo_strata']}

Graph by menu:

{T['dpo_graphcost']}

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

{T['perq_abs']}

- **Chains:** dpo ends at 61.1 against 25.7 for TR. 39.9 of dpo's 61.1 (65 percent) is the observational query, against 6.1 for TR. The X = -1 query is learned better by dpo than by TR (2.4 against 9.7).
- **Forks:** dpo ends at 16.9 against 11.3. The excess is in the X = 0 and X = +1 queries (7.9 against 2.0), which need the slope that data at X = -1 cannot give. The X-mean-dependent queries are no worse than TR's (8.4 against 8.2).
- **Colliders:** dpo ends at 31.3 against 14.6. The excess is in the do(Y) queries (18.1 against 3.7) and the observational query (6.9 against 2.9). Both need the natural X mean times the slope of Z on X. The two X = -1 queries are learned best of any policy (2.6 in total against 3.8 for TR).
- The ranker (X-only, both signs) shows the same signature in chains (43.0 of 57.7 in q0) but not the fork slope problem (0.6 in X = 0 and +1), because it samples both X = -1 and X = +1. frozen (observe only) is the mirror image: the smallest q0 error in chains and colliders, and in chains more than four times dpo's error at X = -1 (10.4 against 2.4).
- **These means are driven by a few systems.** w19 and w12 hold 98 percent of dpo's X-mean-dependent final error in chains. w11 and w23 hold 75 percent in colliders. w21 and w20 hold 68 percent in forks. These are the systems where X has a strong total effect on Z and the eight-observation estimate of the X mean is off.

Fraction of the initial error that remains at the final state, as a ratio of sums over the eight systems of each graph (below 1 = reduced):

{T['retained']}

The same as the median over systems of the per-system ratio (robust to the dominant systems):

{T['retained_median']}

- The X = -1 queries are the best learned part of dpo under both views (3.6 to 7.1 percent of the initial error remains in sums, 2 to 19 percent in medians).
- In forks dpo retains 62 percent of the X = 0 and X = +1 error in sums (TR 15 percent) and 54 percent in medians (TR 41 percent).
- For q0 the two views differ. In sums dpo retains 164, 76 and 97 percent (chain, fork, collider) against 25, 74 and 41 percent for TR. In medians q0 gets worse in chains for every policy (4.4 for dpo, 11.8 for TR, 2.0 for frozen) because q0 starts out accurate after eight observational samples, and in colliders dpo's median (0.36) is close to TR's (0.31). The q0 deficit of dpo is real in the systems where it matters and absent in the typical system.
- Size of the frozen-X-mean problem: if every other mechanism were known exactly, the residual error from the initial X-mean estimate alone would exceed 0.04 on w11 (0.041), w12 (0.043) and w19 (0.145), and exceed 0.01 on w10, w20 and w23 as well. dpo never attains on w11, w12 and w19. The rank correlation of this residual with final error is 0.84 for dpo, 0.36 for the ranker and 0.21 for TR. This is an asymptotic residual, not a strict lower bound for a finite run, because errors in different mechanisms can cancel (the ranker ends w12 at 0.032).

### 7.2 The sign bit

Split by whether X = -1 is on the near (low-leverage) side of the estimated natural X mean (7 systems) or the far side (17 systems):

{T['nearfar']}

- On the 17 far-side systems dpo's log-area equals TR's (-0.03) and its final error is lower (-0.17 log10). On the 7 near-side systems its geometric-mean error over the budget is 3.4 times TR's (+0.54) and its final error 6.1 times TR's (+0.79). Exact permutation p for the near-far gap is 0.0055 on log-area and 0.0076 on final error (post hoc, n = 7 against 17).
- In absolute terms dpo's geometric-mean final error is 0.028 on near-side systems and 0.0073 on far-side systems, with similar initial errors (0.057 and 0.046).
- Caution: TR itself does unusually well on the near-side systems (geometric-mean final error 0.0046 against 0.0109), so every policy looks worse relative to TR there (frozen +0.43 on log-area). The absolute dpo numbers above are the cleaner evidence.
- On capped cost dpo still loses on far-side systems (+2.8, +4.7 on the informative ones), consistent with the missing observation share.

## 8. Non-monotonicity (script `a11_nonmono.py`)

{T['nonmono']}

- Error rose after **1,909 of 4,348 accepted acquisitions (43.9 percent)**. The share is 41 to 47 percent for every policy. After the very first acquisition the error rose in 64 of 192 runs.
- **61 of the 172 runs that ever crossed the target later left it** at least once, and 15 ended above it. local_info is the least stable (11 leavers, 5 ending above, 20 exit events).
- On the ten systems already at target after initialization, 20 of the 80 runs left the target at some later point and 2 ended above it (both local_info, w07 and w20). "At target after initialization" is itself a noisy state.
- First crossing is therefore a first-passage time of a noisy process and rewards early lucky dips. The sustained-cost endpoint in section 4 changes the policy ordering substantially.

## 9. The supervised arm (script `a12_sft.py`)

{T['sft']}

- It stopped in 23 of 24 systems (not w17), at total cost 29 (5 systems), 30 (14) or 31 (4), leaving 1 to 3 units, 47 in total (768 - 721).
- The stop comes after 21 (8 runs), 22 (13) or 23 (2) acquisitions in both menus (means 21.75 uniform, 21.73 heterogeneous). It tracks the number of steps taken, not the error: correlation of stop cost with log error at the stop is 0.14.
- Error at the stop: geometric mean 0.0115, median 0.0076. 18 of 23 stops were at target.
- **Five stops were above target**: w10 (0.049), w11 (0.047), w12 (0.092), w19 (0.199), w21 (0.058). w10 and w11 had crossed earlier (capped costs 23 and 21). w12, w19 and w21 never crossed. TR's final error on those five systems was 0.021, 0.012, 0.021, 0.122 and 0.016.
- On average the sft error at its stop is 0.17 log10 units (48 percent) above TR's error at the same paid cost.
- The stop is a late, budget-linked behavior. It saves at most 3 units and is not evidence of an accuracy-aware stopping rule.

## 10. Recommendations (labeled as recommendations, not findings)

**R1. Fix the sensitivity of the headline endpoint before spending on a new policy.** Keep experimental cost to a target accuracy as the headline, as the agenda requires, but calibrate initialization size and threshold on training-namespace or artificial systems, and freeze them before any development outcome. Suggested acceptance rule for the calibration: at most about 10 percent of systems at target after initialization, and the reference planner reaches the target within budget on at least about 90 percent. In the saved data a threshold of 0.02 would leave 5 of 24 systems at target after initialization and 1 never reached by TR, against 10 and 1 at 0.04. Threshold sweep:

{T['threshold']}

Policy means by threshold (first crossing, capped at 32):

{T['threshold_means']}

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
""")
open(OUT, 'w').write('\n'.join(doc))
print('written', OUT, len('\n'.join(doc)))
