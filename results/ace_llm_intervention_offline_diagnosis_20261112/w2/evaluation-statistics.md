# Evaluation and statistics agenda for ACE (skeptical-statistician lens)

Prepared 2026-11-12 for the expert panel. Read-only design work. Nothing under `/Users/pat/code/ACE` was changed, no model or episode was run, no network was used. The only computation was arithmetic on one saved table (script and output in Appendix A). Facts come from the state-of-evidence brief (`../w2/brief.md`) and the panel notes in `../w1/`; anything else is marked **Assumption**.

## 0. Bottom line

1. The November screen did not fail only because the policies were weak. Its endpoint could not have resolved a realistic benefit. The paired SD of first-crossing cost is about 6.2 units, the detectable difference at 24 systems is 3.7 units, and the headroom above target-risk planning (TR) was 3.04 units. The first thing to repair is measurement, before any training.
2. **The single largest repair is Rao-Blackwellized evaluation.** For synthetic classes the evaluator knows the generating law, so it can score each episode by the conditional expected error given what the policy saw (ERB), not by the realized error. On the saved data this cuts the systems needed for the same precision by roughly **8 to 96 times** (Appendix A, Table A2). It is unbiased under conditions I state and prove as theory item T1. It is *not* the learner's own posterior risk once side information or prior misspecification is present. In those cases the posterior risk is biased and can be gamed (T1, counterexample).
3. Common random numbers keyed by action occurrence, a relative cost-to-target band (theory T2 shows it is the declared endpoint averaged over thresholds), fresh namespaces and a pre-policy rule for difficulty remove the other defects: floor ties, unshared noise, threshold arbitrariness and leakage.
4. **No new policy should be trained before a numerical-only headroom audit (E3)** shows that a world class leaves detectable room above TR and that using feedback is identifiable from outcomes (theory items T4 and T5). In the development class neither held. Plan-once matched TR, so no policy could have shown it uses measurements.
5. The confirmation uses 96 fresh systems, one CRN replicate (plus a 25% second-replicate subset), 5 training seeds, a superiority test against TR, TOST equivalence as the pre-registered null inference, and a gatekept secondary family of contribution controls. Its LLM inference cost is about 3.6 GPU-slice-hours, about 6 with controls. The whole program needs roughly 15 to 21 GPU-slice-hours of inference plus training. That is a new ceiling Patrick must approve.

**Thesis.** *If results are positive:* on fresh systems from a world class qualified in advance by numerical-only criteria, a language-model policy trained to read an explicit SCM state reaches a Bayes-reliable answer at lower expected experimental cost than target-risk planning. The effect is replicated across five training seeds. It disappears when the policy's measurements are shuffled, and it exceeds a numerical student given the same information. All of this is measured with Rao-Blackwellized endpoints whose validity conditions are proved. *If results are null or negative:* with measurement repaired and the study powered for equivalence at a pre-declared margin, a trained small LLM policy is equivalent to, or worse than, target-risk planning. The headroom bounds, the open-loop bound and the fidelity diagnostics explain why. The paper then contributes an evaluation methodology for experimental-design policies that needs one to two orders of magnitude fewer systems.

## 1. What the saved data say about measurement

Verified facts (sources: brief §2, `../w1/numerics-protocol.md` §4 and §7, `../w1/results-analysis-verify.md` §C to E, `../w1/deployment-forensics.md` §1 and §6):

- 10 of 24 systems met the target at initialization, against 9.87 expected from the learner's own posterior. They were floor ties in all 28 pairings.
- Each arm drew its own acquisition noise (`paired_episode.py:114-117` forbids shared measurement namespaces). The SFT and DPO arms issued identical first 10 to 22 actions in every system, yet their capped costs differ in 11 of 24 systems.
- Paired SD against TR on the declared endpoint ranges from 3.07 to 6.63, with a median of 6.2. The detectable difference at n = 24 is 3.71 at 80% power, which is above the 3.04 ceiling. One cost unit needs 305 systems unreplicated, or 93 to 173 with 8 replicates.
- Realized NMSE given a state has a coefficient of variation of about 1.1. Error rose after 43.9% of single acquisitions.
- The decision-level TR-score regret ranks the arms sharply (TR 0.016 to DPO 0.700). It is the planner's own one-step objective, though, so it is circular for TR.
- No per-command LLM score was saved. Deployment used greedy decoding over a trie of command strings, which differs from taking the argmax over whole-command likelihoods (`../w1/mechanism.md` §3).
- The exact public state of every LLM decision can be recovered from the saved prompt tokens (1,698 decodes verified), so offline endpoint work covers all eight arms.

My computation on the saved CPU-arm table (`../w1/numerics-protocol/out_endpoints_cpu_arms.csv`, Appendix A) gives the planning parameters for this agenda:

| Endpoint (other arm minus TR, 24 systems, 1 episode, independent noise) | paired SD (LI / coverage / random / ranker) | standardized effect d (LI / cov / rand / rank) |
| --- | --- | --- |
| Declared: first crossing of realized NMSE 0.04, capped | 6.21 / 3.07 / 5.43 / 6.63 | 0.05 / 0.27 / 0.48 / 0.30 |
| Cost to learner posterior risk ≤ 0.04 | 1.79 / 3.27 / 3.61 / 4.32 | 0.09 / 0.64 / 0.68 / 0.59 |
| log final posterior risk | 0.103 / 0.192 / 0.259 / 0.278 | 0.53 / 1.66 / 1.34 / 1.88 |
| Mean log posterior risk over the cost grid (AUC) | 0.103 / 0.145 / 0.281 / 0.190 | 0.43 / 1.74 / 1.11 / 1.43 |
| log final realized NMSE | 1.52 / 1.70 / 1.48 / 1.65 | 0.07 / −0.10 / 0.17 / 0.25 |
| Mean log realized NMSE over the cost grid | 1.08 / 0.75 / 1.08 / 1.04 | 0.06 / 0.16 / 0.38 / 0.28 |

The ratio of arithmetic means of final error has an implied per-system SD of 0.16 to 0.38 (log scale) on posterior risk, against 1.13 to 1.57 on realized NMSE. The variance ratio is 8 to 96, and so is the factor by which the required number of systems shrinks. On TR's own runs the mean realized NMSE divided by the mean posterior risk is 0.995, which agrees on average with the Rao-Blackwell identity. Caveat: these "posterior" numbers use the learner's posterior, which differs from ERB in the 12 shifted systems. E1 recomputes everything with the evaluator's true-law posterior.

## 2. Measurement repairs required before any new policy is scored

| # | Repair | Why (defect it removes) | Size (from the numerics notes) |
| --- | --- | --- | --- |
| R1 | **Common random numbers keyed by action occurrence.** The noise for node v, at the j-th execution of action a in system s and replicate r, is a keyed draw `U[s,r,v,a,j]` shared by all arms. Initial responses stay shared as now. | Arms with identical actions then get identical data, so differences come only from design. Because conjugate updates do not depend on order, two arms that end with the same multiset of actions have the same final posterior. Each arm's marginal law is unchanged. | Medium: noise keying in `world.py:306-314`, the namespace rule in `paired_episode.py:114-117`, seal and custody records |
| R2 | **Evaluator-side Rao-Blackwellized error (ERB).** ρ_t = E_true[NMSE(f̂_t) \| everything public to the system up to t], computed under the true generating law: posterior variance under the true law plus squared bias of the shared learner's prediction, per query, using the existing `structural_moments` recursion. In matched regimes this equals the learner's posterior risk exactly. In shifted regimes it uses the true N(1.5, 1) edge prior. With text hints it conditions on the hints through their known generating law (Monte Carlo importance weights). | Removes the dominant noise source (CV ≈ 1.1 per realized endpoint) without the circularity of TR's objective. Unbiased under T1. | Small to medium: an evaluator-only conjugate head with prior means (the policy's learner stays unchanged), plus sealer and cost-curve versions |
| R3 | **Relative cost-to-target band.** The target is κ·ρ_0(s), where ρ_0(s) is the system's post-initialization ERB, which does not depend on any policy. The primary endpoint averages cost-to-target over log-uniform κ ∈ [κ_lo, 1/2] (T2). κ_lo is calibrated on numerical arms only so that TR_primary reaches it within budget in ≥ 75% of calibration systems. | No floor ties by construction. Removes between-system difficulty scale (target SD ranged 0.02 to 5.06 across systems) and the threshold arbitrariness of 0.04. T2 shows it is a generalization of the declared endpoint, not a replacement. | Small (analysis layer) |
| R4 | **Public stop rule and censoring.** All arms spend the full budget. The band endpoint is the restricted mean cost, capped at the budget. Report attainment curves per arm with the fraction censored at each κ. An arm with stop authority is secondary only, scored on cost at stop plus reliability at stop. | The SFT arm stopped early in 23 systems, 5 above target. Stopping must not be credited as efficiency. | Small |
| R5 | **Namespaces and retirement.** Disjoint keyed scopes for training, validation (a fresh shard per iteration), calibration (per class), confirmation and transfer. Hash lists go in the registration. The 24 development systems are retired from all decisions. | Leakage; no tuning on the 24 development outcomes | Small; three modules reject confirmation scope today |
| R6 | **Prompt and menu randomization.** For training and evaluation alike: per-system query laws, menu order and identifiers (as recommended in `../w1/numerics-protocol.md` §5), with some query laws held out for transfer. | A state-blind constant (`set_x_neg`) was optimal in 11 of 24 first states. Menu position, identifier and tie rule were confounded. | Small to medium |
| R7 | **Treatment-fidelity logging.** Per LLM decision: the full score vector over legal commands, the frozen decision rule, the prompt hash and the menu permutation. | Without scores the dev screen could not tell a state-blind policy from a state-dependent one whose margins never flipped. | Small |
| R8 | **Failure treatment fixed in advance.** Infrastructure failures are re-executed once with the identical keys, which is deterministic. Policy failures are scored as no further acquisitions (treatment-policy estimand). No fallbacks or fills. Every failure is logged. | Mission-1's "LLM arm" was 56 to 69% hand-coded fills | Small |
| R9 | **Blinded analysis.** Arm labels are coded until the analysis code is frozen and its hash registered. A/A and planted-effect tests run on calibration data. | Garden of forking paths; history of favorable publication gates | Small |
| R10 | **Paired-branch decision attribution (T3).** At states where the LLM and TR disagree, branch both actions under shared noise and continue with TR. The sum of these advantages equals the policy contrast in expectation. | A decision-level attribution that is anchored to truth, not to TR's score, and that tests whether the LLM contributes or narrates | Medium; CPU only |

## 3. Endpoints and estimands

Each system is the unit. Arms are paired within system and replicate through CRN.

- **Primary endpoint (band cost to a Bayes-reliable answer).** For each episode, C̄ is the mean over log-uniform κ ∈ [κ_lo, 1/2] of the first cumulative cost at which ρ_t ≤ κ·ρ_0, capped at the budget. By T2, C̄ = c_0 + (1/ln(1/(2κ_lo))) ∫ clip(ln(m(c)/ρ_0), ln κ_lo, ln ½) − ln κ_lo dc, where m is the running minimum of ρ. It reads as cost and has the precision of an area under a curve.
  - Per-system value: the mean over replicates and over training seeds.
  - Estimand: E_s[C̄_LLM − C̄_TR], with systems drawn from the class generator and equal stratum weights.
  - In well-specified classes with public information only, ρ is the learner's own posterior risk. The endpoint is then deployable, and by the optional-stopping corollary of T1 the answer at the stop is reliable in expectation for any policy.
- **Key secondary endpoints.**
  - Normalized Bayes risk at the full budget: the ratio of arithmetic means of ρ_B/ρ_0. This is valid under T1 because ρ_0 is known at time 0.
  - Reliability at stop: ERB and realized truth NMSE at the first time the public stop (posterior risk ≤ κ_mid·ρ_0) fires. Non-inferiority against TR applies in misspecified or side-information classes.
  - Cost to an absolute posterior-risk target of 0.04, for continuity. Systems with ρ_0 ≤ 0.04 are excluded from this secondary by a pre-policy rule.
- **Descriptive only (never confirmatory).** The declared realized first-crossing endpoint. Realized-truth curves. TR-256 relative regret, which is circular for TR and is used only as a futility screen. First-decision quality at the shared state. Fidelity metrics. T3 attribution.
- **Validity rule, fixed per class before any LLM is scored.** Use ERB where the generating law is computable (all synthetic classes). Where it is not computable (nonlinear truth without a tractable posterior, or physical hardware), use realized truth with R ≥ 8 replicates and size S from that variance, or scope the claim as a case study.
- **Reporting.** Stratified bootstrap (BCa) intervals, per-system win fractions, interquartile means and performance profiles in the rliable style, forest plots by graph, regime and menu, and full error-versus-cost curves.

## 4. Hypotheses, multiplicity, superiority versus non-inferiority

- **H1 (primary).** LLM recipe versus TR_primary on C̄: superiority, one-sided α = 0.025. The test is a stratified, studentized sign-flip test on per-system contrasts, with 100,000 flips, and a BCa interval.
  - TR_primary is the stronger of TR-32 and TR+ (256 draws, CRN across candidate actions). The choice is made in E2 on calibration systems with numerical arms only.
- **Null branch, pre-registered.** If H1 is not rejected, run TOST at ±δ (α = 0.05 each side).
  - Within ±δ: "no benefit larger than δ".
  - Adverse 95% interval: "harm".
  - Otherwise: inconclusive. S is not extended after the fact.
- **Second level, only if H1 is rejected; Holm at one-sided 0.025.**
  - H2: non-inferiority against the best numerical reference at margin δ, then superiority.
  - H3: better than the same LLM with shuffled feedback.
  - H4: better than the state-permuted LLM.
  - H5: better than the semantic-feature numerical student.
  - H6: better than the open-loop bound.
  - H7 (semantic class only): correct metadata better than anonymized.
- **Third level.** Transfer families (Holm over two families). Menu-permutation invariance (TOST at ±δ/2).
- **Margin δ.** The minimal important difference is declared before E3 runs: 5% of TR_primary's mean C̄. The planning default is d = 0.3 per-system SD units. The final S comes from the rule in §5.
- **Seeds.** H1 uses the seed-averaged contrast, treating the 5 trained seeds as fixed. Recipe-level claims also need at least 4 of 5 seeds individually favorable, plus a Hartung-Knapp random-effects interval over seeds, reported as secondary.

## 5. Sample sizes with their basis

Formulas, for paired per-system contrasts with standardized effect d = δ/σ:

- Superiority, one-sided 0.025, power 80%: S = (z_.975 + z_.8)²/d² + 2.
- TOST, true difference 0, α 0.05 each side, power 80%: S = (z_.95 + z_.9)²/d².
- Holm worst step with m = 5: replace z_.975 with z_(1−0.025/5).
- Variance with CRN and R replicates: σ² = σ_b² + σ_w²/R, where σ_w² = σ_A² + σ_B² − 2ρσ_Aσ_B.

| d | Superiority 80% | Superiority 90% | TOST 80% | Holm worst step 80% |
| ---: | ---: | ---: | ---: | ---: |
| 0.20 | 198 | 265 | 214 | 294 |
| 0.30 | 89 | 119 | 95 | 132 |
| 0.35 | 66 | 88 | 70 | 97 |
| 0.50 | 33 | 44 | 34 | 49 |

- **Confirmation default: S = 96.** It covers superiority (89) and TOST (95) at d = 0.3.
  - On the dev planning SDs of the mean-log-risk endpoint (0.10 to 0.28, one replicate, independent noise), d = 0.3 corresponds to 3 to 9% lower expected error over the budget. On the band cost that is at most 0.4 to 1.0 units.
  - CRN should shrink σ, so 96 is conservative.
- **Registered sizing rule:** S = max(48, min(150, ceil of the larger of the superiority and TOST formulas)). It uses σ̂, the upper 80% confidence limit of the LLM-versus-TR contrast SD from E5's validation data (an internal pilot on a separate dataset, so the confirmation's α is untouched), and δ from E3. If S would exceed 150, the confirmation is powered only for 2.8·σ̂/√150, and the paper says so.
- **Replicates.** Generating a system costs nothing, so with total episodes N = S·R fixed, Var = σ_b²R/N + σ_w²/N is minimized at R = 1. Within-system replication is kept only on a random 25% subset, to estimate σ_b against σ_w (heterogeneity reporting).
- **Seeds.** K = 5. With a true d of 0.3 and per-seed SE 1/√96, P(at least 4 of 5 seeds favorable) is 1.00, 0.93 or 0.79 for a seed SD of 0, 0.2 or 0.3. The random-effects interval with t on 4 df has low power. That is why recipe-level inference is secondary and the seed SD is reported.
- **Second-level controls at S = 96 (Holm worst step).** Detectable d = 0.35. For the shuffled-feedback control, T5 gives an effect of at least H_feedback + d_H1 whenever H1 holds, which is ≥ 0.45 under the E3 gate. That is approximate, because plan-once only bounds the open-loop optimum from above.
- **Validation screen (E5), 48 systems.** SE is 0.144. The rule "d̂ ≤ −0.15 and the one-sided 80% upper bound < 0" promotes a true d = −0.3 candidate with probability 0.85 and a d = 0 candidate with probability 0.15. Under the global null, the best of 3 candidates is promoted with probability 0.38. That costs compute, not α, because the confirmation is separate.
- **Transfer (E8), 72 systems per family, Holm over 2 families.** Detectable d = 0.36.
- **Concrete task (E9).** Stage A with 48 simulator instances detects d = 0.40. Stage B with 30 blocks detects 0.51 within-apparatus SD units. It is a case study, not a population claim.
- **Why the raw truth endpoint cannot be used.** At SD 1.1 to 1.6 per system, a 10% contrast needs about 970 to 2,000 single-episode systems (Appendix A).

## 6. The experiments, in order

### E1 Endpoint non-favorability and planning audit on saved development data (must)

- **Question.** On the saved 24-system data, do the repaired endpoints resolve policies better than the declared endpoint? Does the Rao-Blackwell identity hold? Would adopting the new endpoints favor the LLM arms?
- **Design.** Offline, with no new environment responses and no model execution.
  - Inputs: replayed states of the 5 CPU arms, decoded public states of all 1,698 LLM decisions (`../w1/deployment-forensics/out/decisions.pkl`), and the 192 cost reports.
  - For all 8 arms × 24 systems, compute:
    - (a) the declared endpoint;
    - (b) learner posterior-risk paths;
    - (c) ERB under the true generating prior: N(0, 1) in matched regimes, and N(1.5, 1) on edges in shifted regimes;
    - (d) band cost with κ ∈ [1/8, 1/2] and [1/4, 1/2];
    - (e) normalized Bayes risk at the budget;
    - (f) realized-truth versions;
    - (g) TR-256 regret (already computed).
  - Final post-acquisition states for the LLM arms: from archived receipts if they are retrieved, otherwise the last decoded state (flagged).
- **Sample size.** Fixed: 24 systems × 8 arms. Post hoc, used for planning only.
- **Decision rule.** An endpoint can be the registered primary only if all three hold:
  - (i) its median |t| over the 7 contrasts against TR is at least twice that of the declared endpoint;
  - (ii) no development LLM arm improves its position relative to TR or random compared with the declared endpoint (non-favorability);
  - (iii) the ratio of mean realized NMSE to mean ERB has a 95% interval containing 1 within the matched and shifted families. If not, the ERB code is fixed before E2.
- **What each outcome licenses.**
  - Positive: adopt the endpoint family. The paper documents the change with this audit.
  - Null: keep the declared family, but CRN, replicates and S ≥ 300 are then required.
  - Negative (an endpoint favors the dev LLM arms): that endpoint cannot be primary, and the fact is reported.
- **Cost.** Under 1 CPU core-hour locally, 1 to 2 analyst days. No approvals.
- **Main risk.** Final LLM states are incomplete locally. Planning SDs from 24 systems are uncertain by about ±30%.

### E2 Evaluation-harness repair and validation on calibration systems, numerical arms only (must)

- **Question.** Does the repaired harness (R1 to R10) give unbiased, correctly sized, lower-variance contrasts before any new policy is scored?
- **Design.** A calibration namespace of 48 systems from the dev generator (4 regimes × 3 graphs × 2 menus × 2), with query-law and menu randomization.
  - Arms:
    - TR-32
    - TR+
    - local information (LI)
    - coverage
    - random
    - plan-once
    - an A/A twin: TR-32 with a second planning seed
    - a planted-effect arm: TR forced to spend 25% of its budget on observe_all
  - Replicates: R = 8 for the cheap arms (ERB check); R = 4 for TR-32, LI, A/A and the planted arm; R = 2 for TR+ and plan-once. Independent-noise copies of TR-32, LI and coverage measure the CRN variance-reduction factor.
  - T3 attribution: LI versus TR and coverage versus TR on 24 systems, with M = 16 branches.
- **Checks (tolerances declared in advance).**
  - (a) CRN leaves marginals unchanged: |Δmean| < 2 SE and KS p > 0.01 per arm.
  - (b) A/A: the interval for the TR-32 versus twin contrast contains 0. The registered test's size, estimated on 2,000 sign-randomized resamples of the A/A per-system contrasts (a null by construction, which checks the studentization and stratification code), lies in [0.035, 0.065].
  - (c) Planted-effect power is within ±10 points of the formula.
  - (d) ERB is unbiased: the realized/ERB ratio of means lies in [0.92, 1.08], with a 95% interval covering 1, in each regime family.
  - (e) T3 agrees with the direct contrast within 2 SE.
  - (f) In matched regimes, mean realized NMSE at the public stop is ≤ κ·ρ_0.
  - (g) Variance components σ_b, σ_w and ρ_CRN are estimated per endpoint.
  - (h) κ_lo is calibrated.
- **Sample size basis.**
  - For the ERB check, the per-episode CV of about 1.1 over about 1,150 episodes per regime family gives SE ≈ 0.032, enough to detect about 9% bias.
  - 48 systems estimate σ_b to within about ±20%.
  - The earlier matched-regime ratio of 0.71 to 0.74 (12 systems) is within noise. This check settles it.
- **Decision rule.** Proceed to E3 only if (a) to (f) pass.
  - If ERB fails in a class: that class uses realized truth with R ≥ 8, and S is recomputed.
  - If the CRN variance-reduction factor is below 1.2: keep CRN (it adds no bias) but size without it.
  - If T3 fails (e): T3 is descriptive only.
- **What each outcome licenses.**
  - Positive: a "validated measurement" methods section and appendix.
  - Null or negative: no new policy is scored until the failure is fixed.
- **Prerequisites.** E1. Patrick's approval of the harness changes: shared CRN namespaces, evaluator-side prior means, in-process numerical policies, raised roster limits.
- **Cost.** About 5 to 30 CPU core-hours on the local Mac. The range is wide because in-process TR latency has not been measured. Add 1.5 to 3 weeks of code and review.
- **Main risk.** New harness code has repeatedly produced new guard failures. Latency may be higher than assumed.

### E3 Headroom and identifiability audit of candidate world classes, numerical references only (must)

- **Question.** In which classes can a public-information policy beat TR by a detectable margin, and is use of feedback identifiable from outcomes?
- **Design.**
  - Classes: C0, the dev generator, as a negative control. Plus at least 2 classes from the method panel, each with TR's handicap stated before the audit. Examples:
    - partial-measurement actions with cost ratios ≥ 4;
    - text hints with a known conditional law and 2 initial observations;
    - a stronger shift of the prior.
  - Per class: 48 new calibration systems, R = 2 (R = 1 for the non-myopic and clairvoyant arms).
  - Arms:
    - TR-32, TR+, LI, coverage, random
    - plan-once with local search
    - a static schedule tuned leave-one-system-out on training-generator systems by graph × menu
    - a non-myopic reference (two-step lookahead or rollout TR)
    - an informed reference (TR planning with the generating prior, or with text mapped by a fixed rule; prediction still by the shared, uninformed learner, so that only action choice differs)
    - a clairvoyant bound (design optimized with the true parameters)
  - ERB conditions every arm on the same system-level public information (all generated text, whether or not the arm reads it) plus that arm's own data.
  - Headrooms, all on the primary endpoint:
    - H_attain = TR_primary − best attainable reference;
    - H_ceiling = TR − clairvoyant;
    - H_feedback = plan-once − TR;
    - H_state = static schedule − TR.
- **Decision rule** (registered before E3 runs, with no LLM involved). A class qualifies if all four lower 80% bounds hold, in per-system SD units:
  - H_ceiling ≥ 0.6;
  - H_attain ≥ 0.3;
  - H_feedback ≥ 0.15;
  - H_state ≥ 0.3.
  - Qualifying classes are chosen in a pre-declared order. C0 is expected to fail, which serves as a positive control for the audit.
  - δ, σ̂ and S are fixed here.
- **What each outcome licenses.**
  - Positive: the class is frozen. The paper reports the full headroom table, including classes that failed.
  - Null: no superiority confirmation. At most 2 rounds of new classes, chosen by LLM-independent criteria. Otherwise the program goes to the null paper, with T4 as the explanation.
  - Negative (small ceiling everywhere): a theorem-backed statement that these classes leave no room for any policy.
- **Sample size.** 48 systems give a headroom SE of about 0.144 SD units, which separates 0.3 from 0.
- **Cost.** About 15 to 60 CPU core-hours across 3 or 4 classes, dominated by the non-myopic and clairvoyant arms. Class modules are small to medium each and come from the method panel.
- **Main risk.** T4 predicts that classes whose risk depends only on the design will all fail. A weak non-myopic reference understates H_attain; H_ceiling brackets it.

### E4 Treatment-fidelity standard, leakage audit and pre-scoring gates (must)

- **Question.** What does an LLM arm actually decide with? Does each candidate pass fidelity, learnability and leakage gates before any outcome scoring?
- **Design, retrospective part (a).**
  - Forward-only scoring of base, SFT and DPO on all 1,858 saved prompts, over every legal command: full-vocabulary and grammar-renormalized log-probabilities, whole-command argmax against greedy trie decoding, and length normalization.
  - 300 states × 4 menu permutations and identifier renamings.
  - Runtime check against the saved CURC scores: 2 × 1,184 legal-command scores on the 160 training states (SFT and final adapters) and 238 reference scores.
  - Metrics: decision-rule disagreement, the between-state share of score variance, MI(action; state), invariance, agreement with TR-256, and out-of-sample label NLL against the state-blind counter (1.93 nats).
- **Design, prospective gates (b).**
  - G1: score vectors logged for 100% of decisions.
  - G2: ≥ 90% of decisions unchanged under menu permutation when the top-2 margin exceeds 0.5 nats.
  - G3: state-permuted inference changes ≥ 20% of decisions, and the MI permutation test gives p < 0.01.
  - G4: leave-world-out label NLL at most the counter minus 0.1 nats.
  - G5: disjoint namespace hashes, no private fields in prompts, equal tuning budgets on validation data only.
- **Sample size.** All prompts. The invariance rate is estimated to ±2.9 points.
- **Decision rule.**
  - The decision rule for future arms is fixed on a priori grounds (TWOSOME, trie mass pooling), not from outcomes: grammar-renormalized, length-normalized whole-command argmax with randomized menus.
  - A candidate that fails G1 to G5 is not promoted.
  - If the local runtime deviates from the saved CURC scores by more than 1e-3 nats, run on CURC instead.
- **What each outcome licenses.**
  - Positive: a treatment-fidelity section and a score-level account of the dev collapse.
  - Null (metrics do not discriminate): revise the metrics before E5.
- **Prerequisites.** Patrick's approval of local inference, with a written protocol.
- **Cost.** About 1 to 2 hours locally (not measured), or 0.5 to 1.2 GPU-slice-hours on CURC (ledger basis: 0.12 to 0.4 s per command forward, with prefix sharing). 2 to 3 analyst days.
- **Main risk.** Version differences in the local runtime.

### E5 Validation screen with registered promotion and futility rules (must)

- **Question.** Does any candidate recipe, developed only on training and validation namespaces, earn a confirmation?
- **Design.** A fresh 48-system validation shard per iteration, R = 1 with CRN.
  - Candidates: at most 3 recipes × K = 3 seeds.
  - Arms:
    - the candidates
    - TR_primary
    - the best reference from E3
    - the tuned static schedule
    - a numerical student (same data, same tuning budget, K = 3)
    - the frozen base model
    - for the leading candidate: shuffled-feedback and state-permuted inference controls
  - Zero-cost pre-screens prune candidates first: first-decision quality and TR-256 regret.
- **Decision rule.** Promote the best seed-averaged candidate only if all hold:
  - G1 to G5 pass for at least 2 of 3 seeds;
  - against TR, d̂ ≤ −0.15 and the one-sided 80% upper bound is below 0;
  - d̂ ≤ −0.15 against both the shuffled-feedback and state-permuted controls.
  - If the candidate fails only against the numerical student, it may still be promoted, but "language contributes" is registered as exploratory.
  - Futility: if nothing is promoted after 3 iterations, stop development.
- **What each outcome licenses.**
  - Positive: the recipe is frozen, K = 5 seeds are trained, and the study is registered.
  - Null or negative: the null branch. Patrick decides whether to run E6 in null mode. Every candidate is listed in the failure ledger.
- **Sample size.** See §5: probability of promotion 0.85 for a true d of −0.3, 0.15 for a d of 0.
- **Cost.** About 3.1 GPU-slice-hours of inference per iteration (0.6 s per decision, no reasoning tokens) plus training of 9 policies per iteration, and 2 to 5 CPU core-hours.
- **Main risk.** Forking paths across iterations, mitigated by fresh shards and a cap of 3 iterations. Large seed variance.

### E6 Registered confirmation on fresh systems (must)

- **Question.** Does the frozen recipe reach a reliable answer at lower expected cost than TR on fresh systems (H1)? How does it compare with the best numerical reference (H2)?
- **Design.** Confirmation namespace, 96 systems stratified equally, R = 1 with CRN, plus a second replicate on 24 systems. Analysis is blinded.
  - Arms:
    - the LLM recipe × 5 seeds
    - TR_primary
    - TR-32 (continuity)
    - the best numerical reference
    - local information, coverage, random
    - the tuned static schedule
    - plan-once
    - the numerical student × 5 seeds (the standing ranker control under the new recipe)
    - the frozen base model
    - the supervised-only stage × 2 seeds
    - if the recipe has a stop action: the LLM with stop authority, as a secondary arm scored on cost and reliability at stop
  - Endpoints and tests as in §3 and §4.
- **Sample size.** §5: 89 for superiority and 95 for TOST at d = 0.3, so S = 96, adjusted by the registered rule.
- **Decision rule and what each outcome licenses.**
  - Positive (H1 rejected, at least 4 of 5 seeds favorable): "beats TR in class C" with its interval. H2 decides between "matches or beats the best numerical reference" and "beats myopic TR only".
  - Null (H1 not rejected, TOST passes): "no benefit larger than δ". This leads to the null paper. No replacement class is allowed. A redirected method needs a new registration and namespace, and must report this confirmation.
  - Negative (adverse 95% interval): harm. The claim is narrowed.
  - Inconclusive: reported as such, and S is not extended.
- **Prerequisites.** E2, E3, a promotion in E5 (or Patrick's null-mode decision), the registration, a compute ceiling, custody sharding and raised roster limits.
- **Cost.** About 3.6 GPU-slice-hours of inference, plus 20% for failures. 5 to 25 CPU core-hours, depending on whether numerical policies run in-process or under the current harness. Training of 5 + 2 seeds, depending on the recipe.
- **Main risk.** No candidate is promoted. Seed heterogeneity. Custody failures at scale.

### E7 Contribution-isolation controls (must)

- **Question.** Can any effect be attributed to measurements, causal structure and language, rather than to a schedule, extra compute, an identifier prior or narration?
- **Design.** On the E6 systems with the same CRN, using seeds 1 and 2.
  - c1, feedback-shuffled: post-initialization responses are replaced with coherent decoys from another system in the same stratum, and the posterior summaries are recomputed. All 96 systems.
  - c2, state-permuted: the whole public state comes from another system at the same step. All 96 systems.
  - c3, menu and identifier permutation, tested for equivalence. 48 systems.
  - c4, graph-ablated: trained and evaluated without the graph. 48 systems, 2 seeds.
  - c5, metadata ladder (semantic class only): anonymized, correct, irrelevant and misleading metadata, plus misleading × shuffled. 48 systems. Recovery cost is the first cost at which the misleading arm's ERB path comes within 10% of the anonymized arm's.
  - c6, the semantic-feature numerical student (from E6).
  - c7, compute-matched TR: draws or lookahead set to match the LLM's wall-clock per decision on the same hardware. All 96 systems.
  - c8, an argmax-of-input baseline if any planner statistic appears in the prompt.
  - c9, the open-loop bound: the best static design from the post-initialization state, found by local search. All 96 systems.
- **Tests.** Holm at one-sided 0.025 over c1, c2, c6, c7 and c9, plus c5 (correct versus anonymized) in the semantic class. c3 by TOST at ±δ/2. If H1 fails, all controls are reported with intervals as diagnostics.
- **What each outcome licenses.**
  - c1 or c2 equivalent to the LLM: "the benefit is not from using measurements".
  - c6 equivalent: "language adds nothing beyond its features".
  - c7 equivalent or better: "the benefit is extra computation".
  - c9 not beaten: no certificate of feedback use.
  - c3 fails equivalence: identifier dependence, a fidelity failure.
  - c5 recovery cost above half the budget: a robustness limit, stated.
- **Sample size.** Detectable d of 0.35 at 96 systems and 0.49 at 48 (Holm worst step). The c1 effect is at least H_feedback + d_H1 (T5).
- **Cost.** 2.3 GPU-slice-hours, plus 1.9 for the semantic ladder, plus 2 training runs for c4. 3 to 10 CPU core-hours.
- **Main risk.** Decoy prompts may be out of distribution. Mitigated by coherent, same-stratum decoys.

### E8 Transfer beyond the training generator (should)

- **Question.** Does the effect, or the null, hold outside the training generator?
- **Design.** 2 confirmatory families and 1 exploratory family, registered with E6 before unblinding and chosen by LLM-independent criteria:
  - graphs of 4 to 6 nodes;
  - different query laws and cost ratios;
  - (exploratory) nonlinear truth with the linear learner, if `world.py` supports it.
  - 72 systems per family, R = 1 with CRN.
  - Arms: the LLM (seeds 1 to 3), TR_primary, the best reference, LI, random, and the numerical student.
  - Endpoint: band cost on ERB. Where the law is not computable, realized truth with R = 8.
- **Decision rule.** Holm over the 2 confirmatory families at 0.025.
  - Positive: "transfers to family F".
  - Null: the scope is limited to the training generator.
  - Negative: harm outside the generator is stated in the abstract.
  - No families may be added after unblinding.
- **Sample size.** Detectable d of 0.36 per family.
- **Cost.** About 2.6 GPU-slice-hours (longer prompts × 1.5), about 1 more for the exploratory family, 3 to 10 CPU core-hours.
- **Prerequisites.** E6 registration. Harness limits raised: node counts, menu sizes, roster.
- **Main risk.** Prompt length grows with graph size.

### E9 Concrete experimentalist task: wind-tunnel simulator, then the real Remote Lab if approved (should)

- **Question.** Does the frozen policy give a useful, observable outcome at lower cost on a physical-style task with real costs and model misspecification?
- **Design, Stage A (simulator).**
  - The `causalchamber` wind-tunnel simulators, wrapped with time-based costs: a load change costs about 10 s, while repeat measurements are cheaper.
  - Task: predict downwind pressure at the registered operating points.
  - Learner: the linear SCM, which is knowingly misspecified.
  - 48 randomized simulator instances, R = 4.
  - Arms: the LLM (seeds 1 to 3), TR_primary, LI, random, and the best reference.
  - Endpoints: realized truth against the noise-free simulator grid, and cost at the public stop plus truth reliability at the stop. ERB is not available under misspecification.
- **Design, Stage B (hardware).** One Wind Tunnel Mk2. 4 policies × 30 randomized blocks × 16 interventions. A validation grid is measured with replicates at the start and end of each session. The analysis is block-paired.
- **Decision rule.**
  - Stage B runs if Stage A passes engineering qualification (latency, legality, cost accounting, drift protocol), regardless of how the policies performed.
  - Positive (non-inferior error at lower cost in Stage A, same direction in Stage B): "a useful outcome on one apparatus".
  - Null: no practical benefit shown.
  - Negative: harm, reported.
- **Sample size.** Stage A detects d = 0.40. Stage B detects 0.51. One apparatus means no population claim.
- **Prerequisites.** Approval to install `causalchamber`. For Stage B, sign-up, terms and spend.
- **Cost.** Stage A: 1 to 2 weeks and about 1.5 GPU-slice-hours. Stage B: about 5.3 hours of chamber time, about 270 to 700 EUR at academic rates.
- **Main risk.** Fidelity of the Mk1 simulators to Mk2 hardware is undocumented. Drift.

### E10 Blinded independent re-analysis and computational reproduction (should)

- **Question.** Can every registered number be reproduced from the sealed outputs and the registration alone? Is the pipeline deterministic?
- **Design.**
  - (a) Arm codes stay in place until the code hash is frozen.
  - (b) An analyst outside method development re-implements the endpoint, tests and intervals from the registration text. Ideally this is the coauthor.
  - (c) A random 10% of confirmation cells is re-executed and checked for episode equality: bitwise, or ERB within 1e-9.
  - (d) A failure ledger lists every recipe, iteration, crash and exclusion.
  - (e) A superseding statement addresses the public 70% DPO claim.
- **Decision rule.** Any discrepancy beyond tolerance blocks submission until it is resolved and documented.
- **Sample size.** Re-executing 10% of cells detects nondeterminism that affects at least 30% of cells with probability > 0.97.
- **Cost.** About 0.4 GPU-slice-hours, under 2 CPU core-hours, 2 to 3 analyst days.
- **Main risk.** A reviewer from inside the project is not fully independent.

## 7. Sequence, gates and parallelism

- **Stage 0, now, with no new responses.** Run E1 and E4(a) in parallel (E4(a) after the local-inference go-ahead). Draft the registration skeleton and the E3 qualification rule.
- **Stage 1.** Run E2. In parallel, the method panel builds the class modules, and E9 Stage A integration begins if the install is approved.
- **Gate G1.** E2's tolerances pass.
- **Stage 2.** Run E3.
- **Gate G2.** At least one class qualifies. Then freeze the class, δ, σ̂ and S. Otherwise allow at most 2 rounds of class redesign, then take the null path.
- **Stage 3.** Develop methods on the training and validation namespaces, and run up to 3 iterations of E5, each behind the E4(b) gates.
- **Gate G3.** On promotion: train 5 seeds, freeze the recipe and register E6 to E9 with the analysis code hash. On futility: Patrick decides whether to run E6 in null mode.
- **Stage 4.** Run E6 and E7 once, together, on the confirmation namespace. E8 and E9 Stage A run concurrently, because they are registered with E6 and do not depend on its outcome. E10 runs during and after unblinding.
- **Stage 5.** E9 Stage B, if approved.
- **Calendar.** About 8 to 14 weeks. ICML on January 22, 2027 is not realistic for the whole program. E1 to E3 and the theory items could be written up on their own as an evaluation-methodology paper.

## 8. Minimum evidence sets

- **Positive paper:**
  - E1 and E2 (validated, non-favorable measurement);
  - E3 headroom table, including failed classes;
  - E6 H1 with at least 4 of 5 seeds favorable, and H2 reported either way;
  - the E7 controls: shuffled feedback, state permutation, the open-loop bound, the numerical student and compute matching;
  - the ablations: frozen base (pretraining), ACE training, graph-ablated (causal structure), compute-matched TR (extra computation);
  - at least one E8 family and E9 Stage A;
  - E4 fidelity;
  - E10.
- **Null paper:**
  - E1 to E3, with T4 and T5 explaining the absence of room in the dev class;
  - E6 powered for TOST at ±δ, with 5 seeds;
  - E7 run as diagnostics, showing whether the policy uses feedback at all;
  - E4 and the saved-artifact forensics;
  - the numerical-student comparison, which shows whether the failure is specific to language;
  - E10.
  - Whether a null paper is acceptable is Patrick's decision.

## 9. Asks for the lead

1. Choose a venue and date. The program needs about 8 to 14 weeks.
2. Decide whether a rigorous null or diagnostic paper is acceptable. This decides whether E6 runs in null mode.
3. Approve the harness changes:
   - CRN namespaces shared across arms;
   - an evaluator-side posterior with true or informed priors;
   - numerical policies running in-process;
   - lifting the confirmation-scope restriction;
   - more than 8 policies and more than 64 worlds per roster;
   - sharded custody.
4. Approve local forward-only inference on the saved prompts under a written protocol, or about 0.5 to 1.2 CURC GPU-slice-hours instead.
5. Set a new compute ceiling:
   - about 15 to 21 GPU-slice-hours of inference across E4 to E10, plus training (5 seeds and up to 9 candidate policies per E5 iteration, depending on the recipe);
   - about 40 to 160 CPU core-hours, most of which the local Mac can cover;
   - a QoS choice: gpu-testing with a 1-hour wall, or a full A100.
6. Approve K = 5 training seeds for the confirmed recipe.
7. Approve installing `causalchamber` (E9 Stage A). Separately, approve Remote Lab sign-up, terms and about 270 to 700 EUR (Stage B).
8. Retire the 24 dev systems from all decisions, and approve the namespace plan.
9. Choose where to register (OSF, or a committed hash) and name an independent analyst for E10.
10. Reconcile the 70% claim in the public preprint and README before submission.

## 10. What not to do

- Do not use first crossing of realized NMSE at a fixed absolute threshold as the primary endpoint.
- Do not use the learner's posterior risk or TR-score regret as a confirmatory endpoint where side information or prior misspecification is present.
- Do not scale up the number of systems before the measurement is repaired.
- Do not run or tune anything on the 24 dev systems, and do not reuse a validation shard.
- Do not choose the class after seeing LLM results, add transfer families after unblinding, or extend S after the fact.
- Do not drop systems after scoring. Only the pre-policy rule may exclude systems.
- Do not let fallbacks or fills hide policy failures.
- Do not compare a tuned LLM against untuned numerical controls.
- Do not present single-seed results as evidence about the recipe.
- Do not put planner statistics in the prompt without the argmax-of-input baseline.
- Do not evaluate new arms with greedy trie decoding.
- Do not claim "uses feedback" where plan-once ties TR.
- Do not run a confirmation in a class that failed E3.

## 11. Biggest uncertainties

- Whether any tractable class has headroom over TR that survives the informed and non-myopic numerical references.
- The SD of LLM-versus-TR contrasts under CRN and ERB. The dev numerical contrasts suggest 0.10 to 0.39 log units, but the right S could lie anywhere from 48 to 150.
- Seed variance of the new recipes.
- The in-process cost of TR and of the non-myopic and clairvoyant references, which drives E2 and E3 CPU between about 5 and 60 core-hours.
- Inference cost if the recipe uses reasoning tokens: 10 to 50 times the basis of 0.6 s per decision.
- Validity of ERB in nonlinear or physical settings, where realized truth with low power is the fallback.
- Whether the harness changes introduce new failures.
- The venue date.

## 12. Theory items worth proving

- **T1. Rao-Blackwellized evaluation.** Setting: Π is the generating law of (θ, I), F_t is the policy's information, and f̂_t is any F_t-measurable predictor.
  - For every policy that acts on F only, and every F-stopping time τ: E[ℓ(θ, f̂_τ)] = E[ρ_τ], where ρ_τ = E_Π[ℓ | F_τ], and Var(ρ_τ) ≤ Var(ℓ_τ).
  - Corollary (a): if Π equals the learner's prior and I is trivial, ρ is the deployable posterior risk. The public stop at ρ ≤ c then gives E[ℓ at stop] ≤ c on attainment, for any policy.
  - Corollary (b): if I is informative, the learner's posterior risk is biased and can be gamed. Counterexample: a policy that does not measure when the hints indicate a large |θ| looks better than it is. The evaluator's ρ, conditioned on I, stays unbiased.
  - Corollary (c): conditioning on both arms' information jointly gives a contrast whose variance is no larger than the raw contrast's.
  - **Why it matters.** It makes model-based endpoints legitimate under stated conditions, answers the circularity objection, and cuts the number of systems needed by 8 to 96 times.
- **T2. Threshold-averaging identity.** Mean first-crossing cost over a log-uniform band, censored at the budget, equals c_0 plus a clipped log-AUC of the running-minimum error path. The declared endpoint is the degenerate band.
  - **Why it matters.** It shows the repair is a generalization of the declared endpoint, not a favorable replacement, and it removes the arbitrariness of the threshold.
- **T3. Paired-branch performance difference.** J(π) − J(TR) = E_π[Σ_t A^TR(s_t, a_t)]. The advantage is zero wherever π agrees with TR. The CRN branch estimator is unbiased.
  - **Why it matters.** It attributes outcomes to individual decisions, anchored to truth rather than to TR's score. It separates contributing from narrating.
- **T4. Adaptivity gap in conjugate, known-graph classes.** If every row that matters is deterministic and the target is linear, the optimal adaptive risk equals the optimal static risk. Otherwise the gap is bounded by the terms that depend on the means and the rows.
  - **Why it matters.** It explains why the dev class had no headroom, and it turns class selection into a criterion that does not involve the LLM.
- **T5. Open-loop lower bound.** Any policy that does not use feedback has risk at least R*_OL(F_0), for any endpoint that is an expectation.
  - **Why it matters.** Beating R*_OL certifies that the policy uses feedback. The shuffled-feedback control can be identified from outcomes only if TR is below R*_OL. It also gives the power argument for c1.
- **T6 (short lemma). Greedy trie decoding.** It equals the whole-command argmax only if, at every branch, the subtree with the most pooled mass contains the most probable leaf.
  - **Why it matters.** It formalizes the treatment-definition repair.

## Appendix A. Calculations

Script: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w2/calc/sample_sizes.py`. It uses the standard library only, reads one saved table read-only, and is run as `/usr/bin/python3 -I -B`.

Table A1 is §1's table.

Table A2, ratio of arithmetic means of final error (other arm / TR), with implied per-system SD of the log ratio (bootstrap SE × √24):

| Arm | Posterior: ratio, SD | Realized: ratio, SD | Variance ratio |
| --- | --- | --- | ---: |
| local information | 1.032, 0.160 | 1.045, 1.571 | 96 |
| coverage | 1.499, 0.383 | 1.147, 1.132 | 8.7 |
| random | 1.440, 0.259 | 2.389, 1.534 | 35 |
| ranker | 1.681, 0.360 | 2.075, 1.333 | 13.7 |

For a 10% contrast at 80% power with two-sided 0.05: on realized error, 7.84 × SD²/0.01 gives about 1,000 to 1,900 systems. On ERB-type error it gives about 20 to 115. For TR, mean realized NMSE over mean posterior risk is 0.995.

Sample sizes: the §5 table. Promotion operating characteristics: 0.85 at d = −0.3, 0.50 at d = −0.15, 0.15 at d = 0, and 0.38 for the best of 3 under the null. Seed replication: 1.00, 0.93 and 0.79.

## Appendix B. Cost arithmetic

LLM decisions are costed at 0.6 s each (brief: 0.4 to 0.7 s) with 24 decisions per episode and no reasoning tokens.

| Experiment | GPU-slice-hours |
| --- | ---: |
| E5, per iteration | 3.07 |
| E6 | 3.55 |
| E7 | 2.30 |
| E7, semantic ladder extra | 1.92 |
| E8 | 2.59 |
| E9 Stage A | 1.54 |
| **Total, one E5 iteration** | **15.0** |

The total is about 21 GPU-slice-hours with three E5 iterations. All of this is before training and before a 20% failure overhead.

Numerical costs use the ledger's figures of 47.7 s wall per episode in the current per-decision-process harness, and 95.5 allocated CPU-seconds. In-process latency has not been measured.

The current remaining envelope is 13,735 CPU-seconds and 2,742 GPU-slice-seconds. Everything from E2 onward needs a new ceiling.

## Appendix C. Registration checklist (frozen before E6 scoring)

- Class generator, with hash lists of the system IDs per namespace.
- Arms and artifact hashes: adapters and numerical code.
- Decision rule, prompt templates and randomization scheme.
- Endpoints, the κ band and the ERB law.
- Hypotheses, α allocation and margins.
- S, R and K.
- Pre-policy exclusion rule.
- Failure treatment.
- Blinding procedure and analysis code hash.
- Deviation log.
- Every earlier attempt, listed.
