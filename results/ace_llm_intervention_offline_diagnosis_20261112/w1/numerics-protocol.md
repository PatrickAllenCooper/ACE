# numerics-protocol: world generator, SCM learner, numerical policies, evaluation protocol, and the room for adaptivity

Prepared 2026-11-12 by the `numerics-protocol` subagent. Read-only on all project data. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was created, modified, staged or committed (checked with `git --no-optional-locks status --porcelain`, identical to the session snapshot). No repository module was imported or executed. All numbers marked "computed" come from my own scripts in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/` reading saved files.

Path shorthand used below:

- `SR/` = `/Users/pat/code/ACE/scripts/research/`
- `DG/` = `/Users/pat/code/ACE/docs/development/guidance/`
- `REP/` = `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/` (cells.csv, curves.csv, pairs.csv, policies.csv, actions.csv, strata.csv)
- `CPU/` = `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/campaign/` (policy-00..04 = uniform_random, menu_coverage, local_information, target_risk, learned_ranker; world-00..23; each has `world.json`, `learner.json`, `episode/decision-NNNN.input.json`, `episode/decision-NNNN.process.json`, `episode/terminal.json`, `episode/journal.jsonl`)
- `REAL/` = `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_realization_20261108/terminal02/remote/launch/realization/session/worker/task/` (world-000..055.json; 000..031 training, 032..055 development)
- `SCR/` = my scratch directory named above

## 0. What is verified, what is computed, what is inferred

Verified by reading code and documents: sections 1 to 3, 6, and the endpoint definition in 7.

Computed from saved files (reproducible with the scripts in `SCR/`):

- Paired endpoint statistics from `REP/cells.csv` and `REP/curves.csv` (`SCR/endpoint_stats.py`, `SCR/endpoints_all_arms.py`).
- Exact replay of the five CPU arms' saved histories with an independent re-implementation of the learner, structural moments and integrated risk (`SCR/acelib.py`, `SCR/state_replay.py`). Cross-checks: replayed NMSE equals every saved `curves.csv` value to 1.2e-14, replayed posteriors equal every saved public state to 2.9e-14, and my integrated risk equals the saved target-risk `risk_before` on all 560 decisions to 8.9e-16.
- Analysis of the saved target-risk and local-information decision receipts (stderr JSON inside each `decision-NNNN.process.json`) (`SCR/tr_decisions.py`).
- "Model-internal" quantities: expectations under the learner's own working model, starting from the saved post-initialization public state of each development system. Parameters are drawn from the saved public posterior, and hypothetical responses from the learner's predictive law. This is the same kind of computation `choose_target_risk` does one step ahead. It uses no private world descriptor, generates no environment response, runs no policy through the repository pipeline, and is not new experimental evidence (`SCR/static_designs.py`, `SCR/public_static_search.py`, `SCR/design_quality_all_arms.py`, `SCR/endpoint_power_model*.py`, `SCR/init_cross_prob.py`). The batch implementation was validated against the scalar one (exact agreement on a single draw, and the zero-row design reproduces the exact saved risk). Spurious numpy/Accelerate `matmul` RuntimeWarnings appeared; outputs were checked finite and moment-correct.

Local data limits: full states (responses, posteriors) exist locally only for the five CPU arms. For the three LLM arms only the cost reports are local (action sequence, NMSE and the eight predictions per curve point, in `REP/curves.csv`). Their receipts are in the CURC archive. Everything I compute for LLM arms uses their saved action sequences only.

Inferences are labelled "Inference".

## 1. (a) World generator, task, threshold, budget

### Graphs, mechanisms, noise

- Three variables X, Y, Z. Chain X→Y→Z, fork X→Y and X→Z, collider X→Z←Y. X and Y are intervenable within [−2, 2]; Z is not. `SR/llm_intervention_screen_protocol.py:57-61`.
- Mechanisms are linear with additive Gaussian noise: `value = intercept + Σ coef·parent + sqrt(noise_variance)·N(0,1)`. A hard intervention replaces the whole equation including its noise. `SR/llm_intervention_world.py:181-192`. Only `LinearMechanism` exists (`world.py:81-96`); the generator docstring says no nonlinear families, selection, rejection sampling or retries exist (`world.py:153-158`).
- Generic limits of the generator: 8 nodes, 64 actions, 80 requests (`world.py:26`).

### Regime names (exact meaning)

`REGIMES = (("matched_low", 0., .25), ("matched_high", 0., 1.), ("shifted_low", 1.5, .25), ("shifted_high", 1.5, 1.))` at `screen_protocol.py:27-28`. The tuple is (id, mean of every edge coefficient, noise variance of every node).

- Truth law per world (`screen_protocol.py:107-113`): intercepts ~ N(0, 1) for all three nodes; every edge coefficient ~ N(edge_mean, 1); noise variance equal on all three nodes; noise is public (`noise_is_known_public: True`).
- "matched" = edge mean 0, which equals the learner's prior exactly (`prior_relation: "exact constructor prior"`, line 113). "shifted" = edge mean 1.5 with the same SD, intercept law unchanged. The learner prior stays N(0, I). So "shifted" is a 1.5 SD prior-mean misspecification on edges only. It is not model-form misspecification.
- "low" = noise variance 0.25, "high" = 1.0. In all regimes the learner is told the true noise variances (`screen_protocol.py:80`; equality between world noise and public learner noise is enforced at `SR/llm_intervention_scientific_campaign.py:160-162`).
- Public prose is identical across regimes; matched and shifted systems with the same graph, cost menu and noise have identical public designs (`DG/reviews/ace_llm_intervention_screen_protocol_review_2026-11-07.md:11`).

### How the 24 systems were drawn

- One development world per regime × graph × cost menu: 4 × 3 × 2 = 24 (`screen_protocol.py:156-167`), IDs 202611070100 to 202611070123 in that nesting order (regime outermost, then graph chain/fork/collider, then uniform/heterogeneous). Cost strata are different world IDs and therefore different parameter draws (`screen_protocol.py:116`).
- Each parameter is one keyed Gaussian draw: `random.Random(keyed_seed(scope, world_id, "mechanism", node, "intercept" | "parent", p)).gauss(0, 1)` (`world.py:43-57`, `171-177`), called once for the whole 56-world roster by `SR/llm_intervention_screen_realization.py:94-103`. No selection or replacement.
- Computed check (`REAL/world-032..055.json`, identical to `CPU/policy-00/world-NN/world.json`): development matched edges n=24, mean 0.305, SD 0.882; shifted edges n=24, mean 1.370, SD 1.062; intercepts n=72, mean −0.012, SD 1.088. Training worlds: matched edges mean 0.074, shifted 1.346.
- Difficulty varies enormously across systems because the error scale is fixed: the SD of the eight true target values across queries ranges from 0.02 (world 6, matched_high chain, Z edge 0.01) to 5.06 (world 12, shifted_low chain).

### Action menu and costs

`screen_protocol.py:64-71`: `observe_all`, `set_x_neg` (X=−1), `set_x_zero`, `set_x_pos` (X=1), `set_y_neg`, `set_y_zero`, `set_y_pos`, `set_xy` (X=−1, Y=1). Every action measures X, Y and Z. Uniform menu: all cost 1. Heterogeneous menu: the last four (Y and joint) cost 2.

### Initial schedule and budget

- `initial_actions = ["observe_all"]*8`, initial cost 8, additional 24, total 32 (`screen_protocol.py:87-88`).
- Initial responses use a world-level key shared by all arms; acquisition responses use arm-specific keys (`screen_protocol.py:104-105, 148-150`; `SR/llm_intervention_paired_episode.py:189-199`). `prepare_plan` forbids two arms sharing a measurement namespace (`paired_episode.py:114-117`), so there are no common random numbers across arms after initialization.

### The eight target queries

`queries = tuple(Action(f"q{i}", a.kind, a.targets, ("Z",), 1) for i, a in enumerate(menu))` with weights 1/8, target Z, scale 2.0, tolerance 0.04 (`screen_protocol.py:76-77`). So the queries are the eight menu interventions themselves, asking for Z.

The private truth for a query is the noise-disabled structural mean: unclamped nodes take `intercept + Σ coef·parent` with no noise (`world.py:195-205`, used at `SR/llm_intervention_cost_curves.py:234`). For the observational query this is a product of parameters, for example a_Z + b_Z·(a_Y + b_Y·a_X) in the chain.

Computed fact (exact column equality over all rows of `REP/curves.csv`): the eight queries are not eight distinct quantities.

- Chain: q6 ≡ q7 (do(Y=1) equals do(X=−1, Y=1) for Z). Effective weight 2/8 on do(Y=1).
- Fork: q0 ≡ q4 ≡ q5 ≡ q6 (Y does not affect Z) and q1 ≡ q7. Effective law: observational mean of Z 4/8, do(X=−1) 2/8, do(X=0) 1/8, do(X=1) 1/8.
- Collider: all eight distinct.

The double weight on X=−1 (via q7) is a built-in asymmetry of the public task.

### Error, threshold and how they were chosen

- NMSE = (1/8) Σ_q ((prediction_q − truth_q)/2)² using the composed posterior-mean prediction (`cost_curves.py:236-245`). First crossing is the first curve point with NMSE ≤ 0.04, checked after initialization and after each accepted acquisition (`cost_curves.py:247-258`). 0.04 with scale 2 means RMS error 0.4 in Z units.
- Origin: the 2026-11-02 first-screen design review proposes "eight shared initial responses and 24 additional cost units. This bounds history length at 32 responses" and "normalized structural MSE at most 0.04 under a fixed public scale. This is a synthetic diagnostic threshold, not an engineering requirement" (`DG/reviews/ace_llm_intervention_first_screen_design_2026-11-02.md:11, 17`). The protocol guide says the settings are "prospective and unqualified; no outcomes were used to select them" (`DG/ace_llm_intervention_screen_protocol_2026-11-07.md:3-5`). The agenda had proposed 16 initial plus 64 additional (`DG/ace_llm_intervention_agenda_2026-11-02.md:153`); the handoff says those counts are superseded.
- I found no derivation, power calculation or attainment forecast for 0.04, scale 2, or 8+24 in the protocol guide, its review, the scientific contract, its science review, the execution design, or the ledger.

### Was "10 of 24 already meet the target after initialization" anticipated?

- Not quantitatively. No document predicts the fraction.
- Qualitatively the phenomenon was seen before the freeze: in the two-world artificial integration "every method crossed at initialization on one supplied system", and the decision was to keep the threshold and report ever/final attainment separately (`DG/ace_llm_intervention_screen_execution_design_2026-11-08.md:25`; ledger `DG/ace_llm_intervention_execution_2026-11-02.md:433`).
- It was predictable. Computed (`SCR/init_cross_prob.py`, 200,000 posterior draws per system from the saved post-initialization public posterior): the sum over the 24 systems of P(realized NMSE ≤ 0.04 at initialization) is **9.87, Poisson-binomial SD 2.20; observed 10**. Matched: expected 5.92, observed 7. Shifted: expected 3.95, observed 3.
- The learner's own expected NMSE after initialization is ≤ 0.04 in only **2 of 24** systems (world 3: 0.0290, world 15: 0.0322); mean 0.1074, median 0.0785. World 3 is one of those two and did not cross at initialization (realized 0.0681). World 6 had expected NMSE 0.0549 and realized 0.0003. So the ten initial crossings are mostly favourable realizations of a noisy error, plus systems with tiny effects. Given a state, realized NMSE has coefficient of variation about 1.1 (median over systems) and median about 0.65 of its mean.
- The ten initial crossers: 5 chain, 5 fork, 0 collider; matched_high 4, matched_low 3, shifted_high 2, shifted_low 1; uniform 6, heterogeneous 4. All eight arms have capped cost 8 on all ten.

## 2. (b) SCM learner

`SR/llm_intervention_scm.py`.

- One independent Bayesian linear regression per node: y_i = φ_iᵀβ_i + ε, φ_i = (1, parent values), ε ~ N(0, σ_i²) with σ_i² **known** and strictly positive (`scm.py:41-43`). Prior β_i ~ N(0, prior_variance·I), prior_variance = 1 (`scm.py:46`; `screen_protocol.py:80-82`). No prior means, no non-isotropic prior, no unknown-noise option.
- Update for a receipt (`scm.py:61-100`): a head is updated only if the node was not intervened, was measured, and all parents are measured or clamped (`scm.py:83-85`). Rank-one conjugate update (`scm.py:86-92`):
  `projected = V φ; denom = σ² + φᵀV φ; mean += projected·(y − φᵀmean)/denom; V −= projected projectedᵀ/denom`.
- **The covariance update contains φ only, never y.** Posterior covariance of a head is V_T = (I/τ² + Σ_t φ_t φ_tᵀ/σ²)⁻¹, a function of that head's design rows only.
- But a design row holds parent values taken from `available = dict(values, **targets)` (`scm.py:80, 87`). When a parent is clamped the row is deterministic. When a parent is merely observed the row contains a random response. Root heads (X always; Y in the collider) have φ = (1) and a fully deterministic variance 1/(1 + n/σ²).
- Prediction composes posterior-mean mechanisms through predicted parents (`scm.py:121-130`). Because heads are independent a posteriori, this composed prediction equals the posterior mean of the structural target, so it is the Bayes estimator under squared loss for the working model.
- The docstring is explicit that this is "the tractable reference in the agenda, not the final nonlinear learner" (`scm.py:1-9`).
- All arms in a system share this learner (`paired_episode.py:177`). The private evaluator replays accepted receipts into a fresh learner (`cost_curves.py:247-254`), and the sealer independently rebuilds the posterior from batch sufficient statistics (`SR/llm_intervention_episode_seal.py:322-342`).

Which design rows are random, by graph (derived from the eligibility rule and menu):

| Graph | Head that matters for Z | Deterministic row under | Random row under |
| --- | --- | --- | --- |
| chain | Y head (1, X) | set_x_* | observe |
| chain | Z head (1, Y) | set_y_*, set_xy (Z head only) | observe, set_x_* (Y is a response) |
| fork | Z head (1, X) | set_x_*, set_xy | observe, set_y_* |
| collider | Z head (1, X, Y) | set_xy | observe, set_x_*, set_y_* (one coordinate random) |

In the fork the Y head never matters for the target, `set_y_*` gives the same target information as `observe_all`, and `set_xy` the same as `set_x_neg`.

## 3. (c) Policies

### Target risk (`SR/llm_intervention_target_risk.py`)

- Risk functional: R(D) = Σ_q w_q · Var_post[Z | do(q)] / scale² (`target_risk.py:76-88`), where the variance is the exact posterior variance of the noise-disabled composed mapping (`structural_moments`, lines 23-61). The recursion at lines 46-54 is
  Var(Y_i) = tr(V_i C_φ) + μ_φᵀ V_i μ_φ + m_iᵀ C_φ m_i,
  so the variance depends on **posterior means** (μ_φ holds upstream predicted means, m_i holds this head's slope means) whenever a parent is not clamped.
- Because the composed posterior-mean prediction is the posterior mean of the target, R(D) is exactly the working-model expected value of the private NMSE.
- Score of action a (`target_risk.py:113-127`): `(R(D) − mean_k R(D ∪ z_{a,k})) / cost(a)` with 32 fantasy responses z drawn by `learner.simulate` (parameters sampled once per draw, noise included, sampled parents propagated), each followed by the same eligible-row update on a forked learner. Planning draws are fixed at 32 for the screen (`SR/llm_intervention_screen_policy.py:25, 147-148`).
- Each action uses its own RNG stream seeded by the decision seed and the action id (`target_risk.py:115-116`). There are no common random numbers across candidate actions.
- Myopic: exactly one response ahead. Cost enters only as the divisor and as the affordability filter `cost ≤ remaining_units` (line 108). No budget lookahead.
- Tie rule: largest score, then smallest `action_id` (line 130). Negative gains are retained. It never stops voluntarily; it returns stop only when terminal or nothing is affordable (lines 110-111).

Is it the exact one-step Bayes-optimal action for the declared endpoint under the learner's own model? **No**, for five separate reasons.

1. The declared endpoint is the capped cost of the first retrospective crossing of the realized private NMSE. Target risk maximizes expected reduction of posterior expected NMSE. These are different objectives; no arm can observe the endpoint.
2. It is a Monte Carlo estimate with 32 independent draws per action. Computed from the 560 saved receipts: the best-versus-second-best gap is below 1 MC SE in 51.6% of decisions and below 2 SE in 74.8% (median z 0.96). After the first six decisions 79 to 88% of decisions are within 2 SE. No decision had a negative best estimated gain.
3. Gain per cost is the ratio heuristic, not a budget-optimal rule.
4. It is one-step, not multi-step.
5. In shifted regimes the working prior is wrong, so "optimal under the working model" is not optimal under the generating law.

In the matched regimes the working model is the true generating law, so target risk is a noisy myopic Bayes rule and no policy of any kind can beat the exact Bayes-optimal sequential rule in expectation (the agenda says the same at `DG/ace_llm_intervention_agenda_2026-11-02.md:49`).

Practical size of reason 2 (computed, model-internal): the action multiset target risk actually chose has 1.4% higher expected final NMSE than a plan computed once from the post-initialization state with 4,000 common-random-number draws (t = 3.2 over 24 systems). The noise is real but cheap.

Saved decision time: mean 1.449 s, max 7.70 s per target-risk decision; local information mean 1.352 s (`elapsed_seconds` in the process receipts, includes subprocess start).

### Local information (`SR/llm_intervention_numerical_policy.py:67-87`)

For each affordable action and each of 32 fantasy draws: sum over eligible heads (not clamped, measured, parents available) of ½·log(1 + φᵀVφ/σ²) with φ built from the simulated parent values, divided by cost; score is the mean. It is expected information about all local mechanisms, target-agnostic. One RNG shared across actions. Tie rule as above.

### Menu coverage (`numerical_policy.py:62-66`)

Least-used affordable action, counting the whole public history including the eight initial `observe_all`, tie to smallest id. Consequence (verified in every saved history): it never observes during acquisition and follows one fixed sequence per cost menu: `set_x_neg, set_x_pos, set_x_zero, set_xy, set_y_neg, set_y_pos, set_y_zero` repeating (24 actions under uniform costs, 16 under heterogeneous). It is an open-loop schedule that depends only on the cost menu, not on the graph, the queries or the data.

### Uniform random (`numerical_policy.py:59-61`)

Uniform over affordable actions with a keyed per-decision seed (`screen_policy.py:118-124`).

### What the numerical arms actually did (computed from saved histories)

- Target risk totals: observe 215, x− 129, x+ 76, y+ 62, xy 31, y− 29, y0 16, x0 2 (matches `REP/actions.csv`). Under heterogeneous costs it used cost-2 actions in only 16 of 272 decisions; under uniform costs Y or joint actions were 122 of 288. So the heterogeneous menu mostly removes the Y actions and leaves an all-unit-cost problem. It created no budget structure.
- First target-risk action at the common post-initialization state: x− in 11 systems, x+ in 6, xy in 5, y+ in 1, y− in 1. Margins are often decisive (z up to 40). Between the two systems that share an identical public design (same graph, cost menu, noise; matched versus shifted) the stepwise sequences agree on only 33% of steps and 9.75 of about 23 actions differ.
- Saved one-step scores at the common post-initialization state, as a fraction of the best action's score, averaged over 24 systems: set_x_neg 0.740 (median 0.986, best in 11 of 24), set_x_zero 0.578, set_xy 0.513, set_x_pos 0.493, set_y_pos 0.410, observe_all 0.398, set_y_zero 0.320, set_y_neg 0.289. **set_x_neg is the best state-blind first action**, which follows from the double weight on X=−1 in the query law.
- The learned numerical ranker used only x− and x+ (141 and 435). In 10 of 24 systems its sequence is exactly seven x− followed by seventeen x+, and it starts with x− in 21 of 24. It is close to a schedule on the step index. Inference: both learned policies (ranker and LLM) collapsed toward state-blind schedules, which points at the supervision rather than at the language model specifically. This is for the training agents to test.

## 4. (d) The key analysis: value of adaptivity, and what the screen could resolve

### 4.1 Does "posterior covariance depends only on the design" apply?

Partly.

- It applies head by head (section 2).
- It fails as a statement about the task for two code-level reasons.
  1. **Random design rows.** A child head's row contains the parent's realized response whenever the parent is not clamped (table in section 2). The realized information matrix of the Z head then depends on outcomes.
  2. **The target functional is not linear in the parameters.** Composition multiplies coefficients across heads. The posterior variance of the target therefore contains mean-dependent terms: where the upstream node is predicted to sit (μ_φᵀ V μ_φ) and how large the downstream slope is (m² · upstream variance) (`target_risk.py:46-54`). This affects 4 of 8 queries in the chain (q0 to q3), the observational quantity carrying weight 4/8 in the fork, and 7 of 8 in the collider.
- The purely design-only components are: chain do(Y) queries (weight 4/8) when the Z head is probed with clamped Y; fork do(X) queries (weight 4/8) when probed with clamped X; collider q7 only.

So the optimal design is state-dependent in principle. The saved first decisions confirm it (which side of X to probe varies by system, with large margins).

Computed size of the mean-dependent part (`SCR/out_replay_states.csv`, per-query risk at the post-initialization state): the composed queries carry on average 70% of the posterior risk in chains (range 47 to 95%), 32% in forks (7 to 65%) and 78% in colliders (58 to 97%). Direct evidence of mean dependence along runs: the learner's posterior risk rose after a paid response in 81 of 2,673 replayed steps (largest relative rise 9%), which cannot happen for a design-only functional. For comparison the realized NMSE rose after a paid response in 1,158 of 2,673 steps (43.3%).

### 4.2 How much is adaptivity worth here? (computed, model-internal, from the 24 saved post-initialization states)

J(d) denotes the working-model expected final posterior NMSE after adding design d (24 cost units) to the saved initialization. "Plan-once" is a greedy open-loop design computed only from the post-initialization state and evaluated with an independent seed. Geometric means over 24 systems and ratios to plan-once (`SCR/static_designs.py`, `SCR/static_summary.py`, `SCR/public_static_search.py`):

| Design | Geo-mean final expected NMSE | Ratio to plan-once | Comment |
| --- | ---: | ---: | --- |
| Post-initialization risk (no acquisition) | 0.08465 | | |
| Target risk, realized adaptive trajectories | 0.01244 | 0.965 (SE 0.031 on log scale) | one realization per system |
| Plan-once from post-initialization state | 0.01290 | 1.000 | state-conditional, not sequential |
| Target risk's own action multiset replayed open-loop | 0.01309 | 1.015 (t = +3.2) | |
| Local information, realized | 0.01314 | 1.019 | |
| Shared schedule per graph × cost, optimized in-sample on the cell's 4 systems | 0.01402 | 1.087 (t = +6.3, 0/24 better) | optimistic for static |
| Shared schedule per graph × cost, leave-one-system-out | 0.01558 | 1.208 (t = +4.1, 0/24 better) | honest public-only static |
| Fixed 12 observe + 6 x− + 6 x+ for every system | 0.01577 | 1.223 | graph-agnostic |
| Target-risk multiset transplanted from the partner system with identical public design | 0.01672 | 1.296 | |
| Menu coverage schedule | 0.01763 (realized 0.01711) | 1.367 | |
| Random's realized multiset | 0.01861 | 1.443 | |
| Ranker's realized multiset | 0.02076 | 1.609 | |
| 24 observations (the frozen LLM's design) | 0.02263 | 1.755 | |
| 24 × set_x_neg (the DPO design) | 0.02744 | 2.127 | |

Two separate conclusions.

- **Sequential adaptation after initialization is worth about nothing.** Realized target risk versus plan-once: ×0.965, t = −1.2. A cleaner difference-in-differences that removes the common-mode offset shared by static arms on the same systems (realized over expected for target risk, minus the same ratio for coverage and random) gives **−0.7% ± 1.7%**. The model-expected declared endpoint is identical for target-risk sequences and plan-once sequences (11.932 versus 11.926 cost units).
- **Reading the post-initialization state once is worth roughly 10 to 20% of final expected NMSE** relative to a good schedule that depends only on graph and cost menu (in-sample bound 8.7%, leave-one-out 20.8%, hand-fixed 22%). The absolute gap is about 0.002 to 0.004 NMSE at the end of the budget.

Calibration of this machinery: standardized residuals (realized final posterior risk minus J, over the fantasy SD) have SD 0.86 to 1.04 for all five CPU arms in both regime families, and no static arm deviates in mean by more than about 2 SE.

Caveats. The plan-once and shared-schedule searches are greedy, so both are upper bounds on the respective optima. The exact Bayes-optimal sequential policy was not computed; that needs dynamic programming or nested rollouts, which I did not run because it would amount to executing adaptive episodes on fantasy systems. Inference: its advantage over target risk is at most a few percent, since a 4,000-draw plan-once design gains only 1.4% over target risk's own allocation and realized adaptive target risk is statistically indistinguishable from plan-once.

### 4.3 Could the 24-system screen have distinguished a state-adaptive policy from a good static schedule?

**Not on its declared endpoint.** Three independent lines of evidence.

1. Observed paired statistics (`SCR/endpoint_stats.py` from `REP/cells.csv`, n = 24, capped total cost):

| Contrast (left − right) | Mean | SD | SE | t | left lower / tie / higher | Bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| target risk − menu coverage | −0.833 | 3.074 | 0.627 | −1.33 | 8 / 13 / 3 | [−2.08, 0.29] |
| target risk − local information | −0.333 | 6.211 | 1.268 | −0.26 | 6 / 13 / 5 | [−3.00, 1.83] |
| target risk − uniform random | −2.625 | 5.428 | 1.108 | −2.37 | 10 / 12 / 2 | [−5.00, −0.79] |
| target risk − numerical ranker | −1.958 | 6.630 | 1.353 | −1.45 | 6 / 14 / 4 | |
| target risk − frozen LLM | −2.000 | 6.763 | 1.381 | −1.45 | 9 / 11 / 4 | |
| DPO − target risk | +3.542 | 6.433 | 1.313 | +2.70 | 2 / 13 / 9 | [1.25, 6.29] |
| DPO − uniform random | +0.917 | 7.918 | 1.616 | +0.57 | 3 / 13 / 8 | [−2.21, 3.96] |
| DPO − frozen LLM | +1.542 | 7.633 | 1.558 | +0.99 | 5 / 11 / 8 | [−1.46, 4.54] |

Median SE over all 28 pairs is 1.42 cost units. Target risk is not separated from the static coverage rule.

2. Headroom. Ten systems tie at the floor of 8 for every arm. The hard floor of the mean endpoint given those ten is 8 + 14/24 = 8.583. The best-of-eight-arms envelope is 9.583 (9.625 over the five numerical arms). Target risk is 11.625. So the largest conceivable improvement over target risk is 3.0 units and the envelope gap is 2.0 units, against a paired SE of 1.1 to 1.4.

3. Model-internal prediction of the declared endpoint for fixed action sequences, using only the saved post-initialization public states (`SCR/endpoint_power_model.py`, `endpoint_power_model2.py`, `endpoint_power_model3.py`). It reproduces the screen:

| Design (as a fixed sequence) | Predicted mean capped total | Observed | Predicted ever/24 | Observed | Predicted final-ok/24 | Observed | Predicted mean final NMSE | Observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-risk sequences | 11.93 | 11.625 | 22.9 | 23 | 21.5 | 21 | 0.0174 | 0.0172 |
| local-information sequences | 12.02 | 11.958 | 22.9 | 23 | 21.4 | 18 | 0.0176 | 0.0180 |
| coverage | 13.08 | 12.458 | 21.9 | 22 | 20.1 | 21 | 0.0251 | 0.0197 |
| random's sequences | 13.53 | 14.250 | 21.9 | 21 | 20.0 | 20 | 0.0255 | 0.0411 |
| ranker's sequences | 13.29 | 13.583 | 20.9 | 21 | 19.3 | 20 | 0.0281 | 0.0357 |
| 24 observations (frozen LLM) | 13.90 | 13.625 | 21.8 | 22 | 19.3 | 20 | 0.0283 | 0.0201 |
| 24 × set_x_neg (DPO) | 14.41 | 15.167 | 19.5 | 19 | 17.9 | 18 | 0.0385 | 0.0364 |

Predicted fraction crossing at initialization: 0.4097 × 24 = 9.8 (observed 10). The within-system SD of a single episode's capped cost is 5.6 to 8.1 units.

Predicted paired contrasts with the screen's own pairing structure (shared system and initialization, independent acquisition noise, one episode per system, 24 systems). n* is the number of systems for z = 2.8:

| Contrast | Declared endpoint: expected diff, SE(24), z, n* | log final realized NMSE: z, n* | log final posterior risk: z, n* | Observed capped diff ± SE |
| --- | --- | --- | --- | --- |
| target risk − coverage | −1.18, 1.23, −0.96, 204 | −1.00, 187 | −7.19, 4 | −0.83 ± 0.63 |
| target risk − random | −1.62, 1.29, −1.25, 120 | −1.12, 150 | −6.59, 4 | −2.63 ± 1.11 |
| target risk − DPO design | −2.51, 1.50, −1.67, 67 | −2.11, 42 | −7.43, 3 | −3.54 ± 1.31 |
| target risk − frozen design | −2.00, 1.51, −1.32, 107 | −1.46, 88 | −5.36, 7 | −2.00 ± 1.38 |
| target risk − local information | −0.11, 1.12, −0.09, 20,864 | −0.04, 133,994 | −0.99, 190 | −0.33 ± 1.27 |
| target risk − ranker | −1.37, 1.29, −1.06, 168 | −1.34, 104 | −7.50, 3 | −1.96 ± 1.35 |
| DPO design − random | +0.89, 1.56, +0.57, 579 | +1.02, 180 | +3.98, 12 | +0.92 ± 1.62 |
| DPO design − frozen design | +0.51, 1.78, +0.29, 2,293 | +0.53, 660 | +1.51, 83 | +1.54 ± 1.56 |
| **target risk − leave-one-out public static schedule** | **−0.91, 1.22, −0.74, 340** | −0.52, 688 | −3.31, 17 | not run |
| target risk − plan-once | +0.005, 1.10, 0.00, about 7.6 million | +0.05 | +0.74, 340 | not run |

Every observed contrast lies within about one predicted SE of its model-internal expectation.

Answer. With 24 systems, one episode per cell and the realized first-crossing endpoint, the expected z for a state-conditional policy against a good public-only static schedule was about 0.75; about 340 systems would be needed. Even target risk against the worst static design present (the DPO design) had expected z of only 1.7. The comparison "menu coverage 12.458 versus target risk 11.625" is exactly what the model predicts (13.08 versus 11.93) and is not statistically distinguishable. The same question asked with the learner's public posterior risk as the endpoint is easy: 17 systems for static versus state-conditional, 3 to 7 for the large contrasts.

### 4.4 Room for any adaptive policy to beat target risk in this world class

- On the declared endpoint: target risk's model-expected value (11.93) equals that of the best non-sequential state-conditional plan (11.93). The hard floor is 8.58. Any real improvement is bounded by what a better design can add to expected risk, which the evidence puts at a few percent, equivalent to at most a few tenths of a cost unit. That is two to three orders of magnitude below what 24 systems can detect.
- In matched regimes there is a theorem-level ceiling (exact Bayes-optimal rule under the true model). In shifted regimes every arm is scored through the same misspecified learner, so a policy can only change the data, not the predictor.
- Inference: this world class cannot show an LLM beating target risk except by chance.

## 5. (e) Conditions under which a learned or language policy could legitimately beat target-risk planning

Size estimates: small = a new protocol or binding module only; medium = a few generic modules plus new sealed-schema versions; large = most of the generic layer.

| Condition | Expressible now? | What blocks it | Change and modules |
| --- | --- | --- | --- |
| Non-myopic budget effects with heterogeneous costs | Costs: yes. Arbitrary positive integer `cost_units` (`SR/llm_intervention_contract.py:80-81`), enforced generically by the executor (`contract.py:286-288`), sealer and reporter. Setup or switching costs: no, cost is fixed per action. | The protocol hardwires costs 1 and 2 (`screen_protocol.py:70`) and the binding hardwires 8 observe + budget 32 (`screen_policy.py:68-70`). More important: in this linear all-measured class the heterogeneous menu produced no budget structure (target risk simply dropped cost-2 actions: 16 of 272), and plan-once equals sequential. | Small for larger cost ratios (new protocol + binding). Medium for history-dependent costs (`contract.py` ChargedExecutor, `episode_seal.py` reservation checks, `cost_curves.py:157-158`, `task.py:108-109`). A non-myopic numerical control is a small addition to `target_risk.py` plus `screen_policy.METHODS`. Needs a class with real complementarity (partial measurement, structure uncertainty) to matter. |
| Learner misspecification (nonlinear truth, linear learner) | No. | `world.py` has only `LinearMechanism` and a linear `_evaluate` (81-96, 181-192); descriptor schema is strict (125-148); the realization recipe passes only linear law fields (`screen_realization.py:49-53`). The evaluation target "noise-disabled structural mean" is no longer the expected response once mechanisms are nonlinear; `private_eval` is reserved but unimplemented (`world.py:197-200`). | Medium: `world.py` (+ tests), protocol truth law, realization recipes. The learner, sealer and cost curves can stay if the learner stays linear. Caveat: the endpoint scores the fixed linear learner's predictions, so the policy can only choose where the linear fit is good. Since the queries are the menu actions, sampling the query conditions themselves is close to ideal, which favours a static schedule again. |
| Unknown noise | No. | Known strictly positive noise is required and used in the update (`scm.py:41-43, 89`); `MechanismSummary` carries one `noise_variance` (`contract.py:155-175`); `restore` rebuilds the learner from it (`numerical_policy.py:20-43`); the sealer's independent replay is the known-noise formula (`episode_seal.py:322-342`); the campaign assembler enforces world noise = learner noise (`scientific_campaign.py:160-162`). | Medium to large: `scm.py` (normal-inverse-gamma heads), `contract.py`, `numerical_policy.py`, `target_risk.py`, `task.py` (features), `episode_seal.py`, `cost_curves.py:119`, `paired_episode.py:177`, prompt. Per-node noise heterogeneity is already expressible in the world (`world.py:151-152`). A cheap variant, wrong but "known" noise, needs only relaxing the equality check (small). This is a genuine adaptivity case: posterior uncertainty then depends on responses. |
| Model-family or graph uncertainty | No. | The graph is supplied and single everywhere: learner (`scm.py:29`), public state (`contract.py:335-337`), features (`task.py:192-263`), sealer (`episode_seal.py:384-386`), cost curves. | Large: mixture learner over candidate graphs (closed-form marginal likelihoods are available for conjugate heads), public state, features, `target_risk.py` moments under a mixture, sealer, reporter, prompt. The generator already supports any DAG up to 8 nodes. This is the classic setting where sequential adaptivity has large value. |
| Semantic prior knowledge not in the numeric SCM | Partly. Text fields exist: `objective`, `domain_notes`, node `name` and `unit` (`contract.py:52-56, 335`). | The protocol fixes constant prose and names (`screen_protocol.py:83-86`), and `validate_proposal` accepts only the exact proposal (230-246). The learner cannot take prior means (`scm.py:46`), so the fair control "target risk with the same knowledge encoded numerically" (required by `agenda:123`) cannot be built today. After eight observations the posterior is data-dominated, so knowledge adds little unless initialization is shortened. | Small for varied text (new protocol version). Small to medium for informed-prior controls: `scm.py` constructor, learner config schema in `paired_episode.py:120`, `cost_curves.py:115-119`, `episode_seal.py:121-125, 336-338`. The shifted regimes already contain exploitable knowledge (edges near +1.5) that nobody is told. |
| Large or continuous action spaces | Menus up to 32 are expressible (`task.py:21`, `paired_episode.py:79`, `cost_curves.py:22`). Continuous or novel actions are not. | The grammar accepts only `{"action_id": <menu id>}` or stop (`contract.py:128-142`); the adapter requires `action in self._menu` (`world.py:300`); features are padded to 32 actions. Numerical scoring is cheap here: about 1.4 s per decision including process start, 2,056 structural queries. | Small for menus up to 32 or graphs up to 8 nodes (protocol only). Medium to large for typed continuous actions: `contract.py`, grammar module, `world.py` adapter, `task.py`, sealer and reporter menu checks. Compute amortization is not a credible advantage at this scale. |

Notes on the table.

- In the executed screen every policy scores or chooses over the complete eight-action menu, so the candidate-omission term κ of the agenda's decomposition (`DG/ace_llm_intervention_agenda_2026-11-02.md:45`) is zero by construction. Only the value-error and selection-error terms were exercised.
- The eight shared observations already identify the direction of the informative intervention in most systems (first-decision margins up to 40 MC SE). Any design that wants prior knowledge, text or pretraining to matter has to start from a less informed state: fewer or no initial observations, higher noise, or more unknowns per observation.
- A successor should vary the query law across systems. With the present fixed law one constant action (set_x_neg) is the best state-blind first move in 11 of 24 systems, so a state-blind policy is not penalized enough for the difference to be visible.

Two further levers the task did not list.

- **Stopping.** Every numerical arm spends the full budget; the public posterior risk would support a stop rule today. An endpoint of actual paid cost with a public stop rule is expressible with a small change, and target risk plus that stop rule is then the fair control. Otherwise an LLM that stops early will look efficient or reckless by accident (the supervised LLM stopped early in 23 systems, 5 of them above target).
- **Measurement choice.** `Action.measure` already supports partial measurement and the learner skips heads with missing parents (`scm.py:83-85`). Cheap partial-measurement actions need only a protocol change and create real trade-offs.

## 6. (f) Reusable as-is versus hardwired

Reusable generic layer (bounded, not tied to X, Y, Z or eight actions):

| Component | File | Limits that bind a successor |
| --- | --- | --- |
| Action and state contract, charged executor, hash-chained journal, stop command | `SR/llm_intervention_contract.py` | menu-id grammar only |
| Linear-Gaussian generator, keyed seeds, response adapter | `SR/llm_intervention_world.py` | linear only; 8 nodes, 64 actions, 80 requests (line 26) |
| Learner | `SR/llm_intervention_scm.py` | known noise, zero-mean isotropic prior |
| Target risk and the three other controls | `SR/llm_intervention_target_risk.py`, `SR/llm_intervention_numerical_policy.py` | ≤ 64 queries, 2..256 draws |
| Public task envelope, prompt, numeric features | `SR/llm_intervention_task.py` | 8 nodes, 32 actions, 64 queries, 80 history rows (line 21); features not permutation invariant (196-198) |
| Episode loop | `SR/llm_intervention_episode.py` | none beyond the above |
| Pairing: one plan per world, shared initial keys, per-arm journals, all-arm finalization | `SR/llm_intervention_paired_episode.py` | scope artificial or development only (line 77); ≤ 8 policies (94); ≤ 32 menu actions (79); equal query weights (69-70); arm measurement namespaces must differ, so no common random numbers (114-117) |
| Sealing | `SR/llm_intervention_episode_seal.py` | same scopes (103); ≤ 80 requests (87); equal weights (115); posterior replay hardwired to this learner (322-342) |
| Cost curves | `SR/llm_intervention_cost_curves.py` | equal weights (111-112); this learner (119); structural-mean truth; threshold must equal task tolerance (113-114) |
| Roster reporting | `SR/llm_intervention_screen_reporting.py` | ≤ 64 worlds, ≤ 8 policies (19); ≤ 4 strata fields (79); development scope (74); descriptive only, no intervals |
| Campaign runner | `SR/llm_intervention_campaign.py` | ≤ 64 source plans (64) |

Per-world scale and tolerance are already supported by this layer (each paired plan carries its own task; the roster carries per-world `scale` and `threshold`, `screen_reporting.py:83-88`).

Hardwired to this screen:

- `SR/llm_intervention_screen_protocol.py`: the three graphs, the eight named actions and costs, queries = menu, weights 1/8, scale 2, tolerance 0.04, eight `observe_all`, budget 32, constant prose, the four regimes, ID ranges, arms, milestones. `validate_proposal` only accepts the exact regenerated proposal (230-246).
- `SR/llm_intervention_screen_policy.py`: development IDs limited to the 24 (line 52), exactly eight `observe_all` and budget 32 (68-70), exactly 32 planning draws (66), four methods, rejects training and confirmation scopes (47).
- `SR/llm_intervention_scientific_campaign.py`: exactly eight policies in fixed order (25-27, 83-86), three services of exactly 576 requests (32, 144), development scope (154).
- `SR/llm_intervention_screen_realization.py`: exactly 56 worlds (54).
- Launch and collection scripts are tied to their exclusive directories (handoff line 107).

Consequences for a successor: a new protocol module, binding module, assembler and realization helper are needed (each roughly 100 to 300 lines). A confirmation run needs the scope checks lifted in three reviewed modules. More than eight arms (for example a static-schedule control, a non-myopic planner, an informed-prior control, stop-rule variants) or more than 64 worlds per campaign need the roster limits raised. Any learner change needs new sealer and cost-curve versions.

## 7. (g) Endpoint weaknesses and more sensitive endpoints already supported by saved data

Definition: `crossing = first curve point with NMSE ≤ threshold`; `capped = cap if failed or never crossed else crossing cost` (`cost_curves.py:258-274`); total adds the declared initial 8 (`screen_reporting.py:252`).

Weaknesses (all computed from saved files unless cited):

1. **It is a first-passage time of a noisy private quantity.** Given a state, realized NMSE has CV about 1.0 to 1.2. Crossings are not stable: 15 of the 172 runs that ever attained ended above the threshold; there are 245 threshold direction changes over 192 runs and 60 runs with two or more. On the ten initial-crossing systems, 20 of 80 runs later left the band and 2 ended above it. Local information: 23 ever, 18 final.
2. **Floor ties from initialization.** 10 of 24 systems carry no information. This was predictable (expected 9.87 ± 2.2).
3. **Cap censoring.** World 19: seven of eight arms capped. Counts at the cap: DPO 5, random 3, ranker 3, SFT 3, coverage 2, frozen 2, target risk 1, local information 1.
4. **Fixed absolute scale.** Systems with tiny effects are trivially below 0.04 and systems with large shifted edges are far above it. Between-system difficulty dominates.
5. **No arm optimizes it and none can observe it.** The agenda and results documents already say it is not a deployable stopping rule (`DG/ace_llm_intervention_24_system_results_2026-11-10.md:85`).
6. **One episode per cell with independent acquisition noise across arms.**
7. **Duplicate queries** (section 1) make the effective query law graph-dependent and give X=−1 double weight.
8. **Fragility of the headline ordering.** On other realized endpoints the DPO arm is not distinguishable from target risk: sustained crossing (first cost after which the run stays below threshold) DPO − target risk +1.33 ± 2.18; log final NMSE +0.25 ± 0.39; log AUC of NMSE +0.23 ± 0.16. Sustained-crossing means: coverage 15.13, target risk 15.67, ranker 15.71, SFT 16.21, frozen 17.00, DPO 17.00, local information 17.75, random 18.42. Geometric-mean final NMSE: coverage 0.0072, target risk 0.0085, local information 0.0095, frozen 0.0105, DPO 0.0108, random 0.0109, SFT 0.0115, ranker 0.0128. Only the first-crossing endpoint gives DPO − target risk a t above 2.

More sensitive endpoints that need no new run:

A. **Public posterior expected NMSE** (the learner's `integrated_risk`), by replaying saved receipts. Available locally for the five CPU arms; for the LLM arms it needs the receipts from the CURC archive. Observed means over 24 systems:

| Arm | Final posterior risk | Geo-mean | Systems ≤ 0.04 at end | 8 + cost to posterior risk ≤ 0.04 |
| --- | ---: | ---: | ---: | ---: |
| target risk | 0.0173 | 0.01244 | 22 | 14.50 |
| local information | 0.0179 | 0.01314 | 22 | 14.67 |
| uniform random | 0.0249 | 0.01760 | 20 | 16.96 |
| menu coverage | 0.0259 | 0.01711 | 20 | 16.58 |
| numerical ranker | 0.0291 | 0.02098 | 18 | 17.04 |

Paired log final posterior risk, target risk minus other: coverage −0.319 (t = −8.1, lower in 24/24), random −0.347 (t = −6.6, 24/24), ranker −0.523 (t = −9.2, 24/24), local information −0.055 (t = −2.6, 16/24). The same contrasts on the declared endpoint have t = −1.3, −2.4, −1.4, −0.3. Posterior-risk threshold crossing is stable in practice (sustained crossing equals first crossing in all 120 CPU-arm runs, although the risk itself is not strictly monotone) and is a quantity every arm could legitimately use to stop. Limits: it is the learner's own belief. Realized over posterior ratio of means was 0.71 to 0.74 in matched and 1.27 to 1.33 in shifted systems (12 systems each, noisy), so it is optimistic under prior misspecification and would be wrong under model-form misspecification.

B. **Design quality of the realized action multiset** (model-expected final NMSE from the common saved post-initialization state). Needs only the saved action sequences, so it covers all eight arms now (`SCR/design_quality_all_arms.py`):

| Arm | Mean J | Geo-mean J | Ratio to target risk (paired, t) | Arm better in |
| --- | ---: | ---: | --- | ---: |
| target risk | 0.0173 | 0.0131 | 1 | |
| local information | 0.0177 | 0.0134 | ×1.02 (t = 2.8) | 5/24 |
| menu coverage | 0.0251 | 0.0176 | ×1.35 (8.6) | 0/24 |
| uniform random | 0.0255 | 0.0186 | ×1.42 (7.3) | 0/24 |
| numerical ranker | 0.0281 | 0.0208 | ×1.59 (8.6) | 0/24 |
| frozen LLM | 0.0282 | 0.0225 | ×1.72 (5.8) | 1/24 |
| supervised LLM | 0.0354 | 0.0246 | ×1.88 (8.2) | 0/24 |
| DPO LLM | 0.0384 | 0.0274 | ×2.10 (7.8) | 0/24 |

Across the eight arms, mean J correlates 0.92 with the headline mean capped cost. The whole ranking of the screen follows from which actions each arm chose.

C. **First-decision quality at the common state**, from the saved target-risk receipts: fraction of the best one-step score captured by each arm's first action. Target risk 1.00 by construction, local information 0.958, ranker 0.840, coverage, SFT and DPO 0.740 (all 24 start with set_x_neg), random 0.535, frozen 0.398 (all 24 observe). A zero-cost screen for any candidate policy.

D. Realized-error alternatives (AUC of log NMSE, NMSE at fixed cost, time below threshold) are available for all arms but are not materially more sensitive than first crossing, because they inherit the same CV of about 1 (target risk versus random: t = −2.3 for log AUC against −2.4 for capped cost).

Not computed, recommended for a successor's private evaluator: truth-conditional expected error of each realized design over many fresh noise replicates, replicate episodes per cell, and common exogenous noise across arms. I did not compute these because they need new responses from the development systems.

## 8. Disagreements between documents, and results that cut against the convenient story

- Agenda 2026-11-02 proposes 16 initial + 64 additional responses and "seven policies" (`agenda:153, 177`); the executed protocol is 8 + 24 with eight policies. The handoff (newest) says the agenda's counts are superseded.
- The world guide says "64 actions/queries" (`DG/ace_llm_intervention_world_2026-11-04.md:13`); the task envelope, paired plan and reporter limit menus to 32. The effective limit is 32.
- The handoff and results report describe target risk as "direct SCM planning" and "strong". It is a 32-draw one-step ratio heuristic whose late decisions are mostly Monte Carlo ties. It is nevertheless near the state-conditional optimum in this class, and local information is within 2 to 6% of it in expected final NMSE (design quality ×1.02, realized final posterior risk ×1.06). "Strong" is true here mainly because the class is easy.
- Evidence against the convenient reading "DPO was clearly worst": on realized endpoints other than first crossing, DPO is indistinguishable from target risk, random and frozen, and coverage has the best geometric-mean final NMSE. Evidence for it: DPO's static design has 2.1 times the expected final NMSE of target risk's, with 0 of 24 systems better.
- Evidence against "the LLM arms failed for mysterious reasons": their endpoint values are reproduced by treating them as the fixed schedules they were (frozen: predicted 13.90, observed 13.625; DPO: predicted 14.41, observed 15.167).
- Evidence that set_x_neg is a sensible state-blind choice, not an arbitrary collapse: it is the best first action in 11 of 24 systems and captures a median 99% of the best first-step gain. A constant-set_x_neg policy is wrong mainly because it never observes and never switches side.
- Realized over expected final posterior risk is slightly below 1 even for static arms in matched systems (random ×0.92, coverage ×0.97; n = 12). I attribute this to shared-system sampling variability plus Jensen (about −1.5%), not to a defect, but it is a 2 to 3.6 SE deviation for random and should be rechecked on fresh systems.

## 9. Scratch artifacts (all under `SCR/`)

Scripts: `acelib.py`, `endpoint_stats.py`, `state_replay.py`, `replay_analysis.py`, `tr_decisions.py`, `static_designs.py`, `static_summary.py`, `public_static_search.py`, `design_quality_all_arms.py`, `init_cross_prob.py`, `endpoints_all_arms.py`, `endpoint_power_model.py`, `endpoint_power_model2.py`, `endpoint_power_model3.py`.

Outputs: `out_capped_cost_by_world.csv`, `out_pairs_stats.csv`, `out_replay_states.csv` (2,793 replayed states with posterior risk, realized NMSE and per-query risk), `out_endpoints_cpu_arms.csv`, `out_endpoints_all_arms.csv`, `out_decision_receipts.csv`, `out_action_counts.csv`, `out_static_designs.csv`, `out_public_static_search.csv`, `out_design_quality_all_arms.csv`, `out_init_cross_prob.csv`, `out_endpoint_power_model.csv`.

Run with `/usr/bin/python3 -B <script>` from `SCR/`. Fixed seeds. Monte Carlo SE of each J is about 0.3% of its value (4,000 draws).
