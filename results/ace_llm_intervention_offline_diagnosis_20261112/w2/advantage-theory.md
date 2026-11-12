# Where a language model can beat a numerical planner, and where it cannot: an experimental agenda

Panel lens: Bayesian and causal experimental design theory. Prepared 2026-11-12. Read-only design work. Nothing under `/Users/pat/code/ACE` was modified, nothing was run except arithmetic in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w2/calc/calcs.py` (standard library only, no project code, no data).

Sources: the verified brief (`.../w2/brief.md`) and the notes in `.../w1/` (mechanism, numerics-protocol, results-analysis-verify, deployment-forensics-verify, literature-policy, literature-benchmarks, prior-evidence-core, training). Three repository facts were checked directly: the learner prior is zero-mean isotropic (`scripts/research/llm_intervention_scm.py:29-46`), the target-risk planner accepts 2 to 256 draws only (`scripts/research/llm_intervention_target_risk.py:102-103`), and the screen truth law has one scalar coefficient SD and per-edge means (`scripts/research/llm_intervention_screen_protocol.py:101-113`). Anything else not in the brief or notes is marked **[assumption]** or **[MEM, verify before citing]**.

---

## 0. Thesis

A language model can beat a numerical experimental-design planner only by supplying something the planner lacks. That can be decision-relevant information available only in language or pretraining, a better model class, candidates outside the planner's set, or a better approximation where exact planning is intractable. With equal information, a correct model and a tractable Bayes planner it provably cannot win. The completed 24-system screen sat in that regime: adaptivity is provably worthless in its design-only subclass and was measured at −0.7% ± 1.7% overall.

**If results are positive**, the paper claims the following. A trained LLM intervention policy that reads textual domain information (which causal links are absent, or how mechanisms bend) captured a calibrated share ρ of the computed information value on fresh systems. It beat the strongest numerical controls that lacked the text: the uninformed planner, the law-aware planner and a GO-CBED-style amortized policy. Its benefit scaled with the controlled information dose. It showed no benefit where theory predicts none, and it never exceeded the oracle-informed ceiling.

**If results are null or negative**, the paper becomes a theory-and-measurement contribution with five parts:

- a sequential value-gap decomposition with computable numerical references;
- theorems explaining why adaptive or learned policies cannot win in conjugate known-graph classes;
- a headroom atlas showing where an LLM could win;
- a calibrated instrument (endpoint, power and ceiling audit) that LLM-for-design claims should pass;
- the documented finding that the LLM did not turn text into better interventions at this scale.

If the compiler arm succeeds, the claim narrows to the LLM as an information source for a numerical planner.

---

## 1. The value gap, and where an LLM can legitimately sit in it

### 1.1 Decomposition (sequential, with computable references)

Let J(π) be the expected terminal loss of policy π under the true generating law P. In design-only mode, all arms share the conjugate learner and are scored on its final prediction against the noise-free truth. Let π_N be the numerical planner: one-step target risk with a working model M, information I, candidate set C, and a Monte Carlo approximation. The planner's gap to the best achievable policy telescopes into five terms. Each term is the difference between two numerical arms that can be computed with zero environment responses on synthetic systems:

| Term | Defined as | Numerical reference pair |
| --- | --- | --- |
| Δ_decision | approximation error of the planner itself (32 independent MC draws, ratio heuristic) | TR(32 draws) − TR(4,096 draws, common random numbers across actions) |
| Δ_myopia | one-step versus non-myopic planning under the same model, information and candidates | TR(exact) − rollout planner, or exact dynamic programming / multiset optimum where tractable |
| Δ_omission | candidate set versus the full feasible action set | planner on the menu − planner on the feasible set (for example the bounds ±2 that the screen menu omitted) |
| Δ_misspec | working model class versus the true model class | planner whose internal model is M − planner whose internal model is the true class (the shared learner is unchanged) |
| Δ_info | information the planner lacks (text, the law, the instance) | uninformed planner − oracle-informed planner (knows the side information numerically) |

The order of telescoping is a convention. Report the predeclared order (as listed) and the Shapley average over orderings as a sensitivity check (2^5 = 32 reference arms per cell, feasible in the model-internal machinery).

**Relation to the 2 November agenda.** The agenda's one-step bound (loss ≤ κ + 2ε + δ, `docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md` lines 44-48) covers omission (κ), value error (ε) and selection (δ) at one history. The decomposition above is sequential. It separates myopia from decision noise, and it adds the two terms that the literature says an LLM might supply: information (Δ_info) and model class (Δ_misspec).

### 1.2 Where an LLM can win, and where it provably cannot

| Term | Can an LLM policy capture it? | Can a numerical control with the same information close it? | Verdict for the paper |
| --- | --- | --- | --- |
| Δ_decision | Only by being a better approximator than more draws. | Yes, trivially, with more draws and common random numbers. | No legitimate LLM advantage over a properly computed planner. |
| Δ_myopia | Yes, if trained on multi-step returns (Kriz et al. show 0.6B can learn a non-myopic choice). | Yes where DP or rollouts are tractable. Otherwise an amortized policy (GO-CBED) can. | The LLM can beat only a myopic or intractable planner, and only by at most Δ_myopia. It must also beat GO-CBED-style. Not a headline route here. |
| Δ_omission | Yes, if it may propose non-menu actions. | Yes, by continuous optimization in low-dimensional spaces (for linear heads the optimum is at the bounds). | Legitimate only in combinatorial or unstructured action spaces. Not this codebase. |
| Δ_misspec | Yes, if pretraining or text knows the functional form. | Yes, if the planner is given the right class or averages over plausible classes. | Legitimate when the form is known only in language or pretraining. Bounded by Δ_misspec. |
| Δ_info | Yes, by reading text, names or pretraining knowledge. | Yes, if the information is given numerically (oracle) or can be parsed by a rule. | Legitimate when the information exists only in language and resists rule parsing. Bounded by Δ_info. |
| Same information, correct model, tractable exact planner | No. | n/a | **Provably impossible** (Section 2, T4). |

The completed screen's class had Δ_info = 0 in matched regimes (nobody, LLM included, was told anything the learner lacked). Δ_misspec was 0 in matched regimes. Δ_omission was 0 for the LLM-versus-planner comparison, because both used the same menu. Δ_myopia plus adaptivity was about 0, and Δ_decision was about 1.4%. The ceiling on any legitimate LLM advantage was therefore a few percent of final risk. That is two to three orders of magnitude below what 24 systems could detect on the first-crossing endpoint (numerics-protocol §4.4).

---

## 2. Formal results worth proving

Notation: conjugate linear-Gaussian heads with known noise σ² and prior N(m0, τ0² I). The design rows of a head are its parent values (clamped or observed). V is the posterior covariance, g(θ) the vector of noise-free interventional query means, W the query weights. Risk is Σ_q w_q Var(g_q)/scale². That risk is exactly what `target_risk.py:76-88` computes, and it equals the working-model expected NMSE (numerics-protocol §3).

**T1. Zero value of adaptivity (exact).** Assume:

- (A1) every head that any action updates is a root or has all its parents clamped by that action, so each action contributes fixed rows;
- (A2) the target is linear in the parameters (for example do(Y) queries of Z in the chain);
- (A3) the planner's loss is the learner's own posterior risk, or the truth law equals the learner prior.

Then:

1. V after any action sequence depends only on the action counts n (rank-one updates contain φ but never y: `scm.py`, numerics-protocol §2).
2. The expected terminal risk of every adaptive policy is an expectation of a deterministic function of n, so an open-loop multiset attains the adaptive optimum. The value of adaptivity is 0 and action order is irrelevant.
3. Under costs c and budget B the problem is min_n tr(W C V(n) Cᵀ) subject to cᵀn ≤ B, n integer. Its continuous relaxation is a convex Bayesian A/c-optimal design problem, because tr(A X⁻¹ Aᵀ) is convex in X ≻ 0 and X is affine in n (Chaloner and Verdinelli 1995, as read by the literature agent).

*Why it matters:* it explains the November null from first principles, turns "the LLM failed" into "this subclass cannot reward any policy", and gives a falsification check: a measured adaptive advantage in this subclass is a bug. Without A3, for example under a shifted prior mean, adaptivity regains value through learning about the bias. That is one of the "what changes that" levers.

**T2. Bounded adaptivity with random rows and composed targets (to prove; currently a conjecture with numerical support).** The completed class breaks A1 (observed parents make rows random) and A2 (composed queries are multilinear, and their posterior variance depends on posterior means: numerics-protocol §4.1). Claim to prove under A3:

> J_OL(s0) − J*(s0) ≤ EVPI_design(s0) + η(s0)

Here:

- J_OL is the best open-loop plan from the state s0;
- J* is the adaptive optimum;
- EVPI_design(s0) = E_{ψ|s0}[J(d*(s0); ψ) − J(d*(ψ); ψ)] is the expected value of perfect information about the design-relevant parameters ψ (upstream means and downstream slopes that enter the gradient of the composed queries);
- η is a noise-chasing remainder that vanishes under A1.

For a discrete design choice, EVPI_design ≤ (largest regret of a wrong design) × P_post(ψ lies across a design-switching boundary). It therefore shrinks quickly once initialization has fixed which side of each switch the system is on.

*Supporting facts:* first-decision margins after 8 initial observations reached 40 MC SE. Plan-once equals adaptive (difference-in-differences −0.7% ± 1.7%). Reading the post-initialization state once is worth 10 to 20% relative to a graph-and-cost-only schedule (numerics-protocol §3, §4.2). *Predicted dose-response:* adaptivity value rises as n_init falls and as switching parameters become poorly identified. E2 tests both directions.

**T3. Greedy near-optimality and what breaks it (known results to apply) [MEM, verify constants before citing].**

- For a monotone set function with submodularity ratio γ (Das and Kempe 2011), greedy over k steps attains at least (1 − e^{−γ}) of the best k-set.
- Under budgets, cost-ratio greedy alone can be arbitrarily bad, while the better of cost-ratio greedy and the best single affordable action attains a constant fraction (Khuller, Moss and Naor 1999; Leskovec et al. 2007).
- Under adaptive submodularity, adaptive greedy is within (1 − 1/e) of the optimal adaptive policy (Golovin and Krause 2011).

These are worst-case bounds. The class's measured myopia gap is far smaller (target risk's multiset is 1.4% worse than a 4,000-draw plan, and that 1.4% is MC noise). The useful content is the converse, which gives a map of where myopia can matter:

- complementary actions (γ ≪ 1);
- indivisible costs that leave budget remainders;
- stopping problems;
- structure or noise uncertainty, where information has delayed value.

Computing γ per system (E2) locates these regimes.

**T4. Equal-information ceiling (no free lunch).** Under A3 and equal information (the same public history and side information), the Bayes-optimal policy is weakly better than every other policy in expectation over the truth law. Hence for any LLM policy:

> J(π_N) − J(π_LLM) ≤ J(π_N) − J* = Δ_decision(π_N) + Δ_myopia(π_N)

Both terms are computable. *Why it matters:* every LLM claim gets a numerical ceiling. An observed advantage above the ceiling (beyond sampling error) is evidence of leakage or evaluation error and triggers an audit (the capture ratio ρ > 1 alarm). The theorem also shows the completed screen could not have produced a legitimate LLM win larger than about 2% of final risk.

**T5. Sequential decomposition (Section 1.1) as a formal identity.** It is exact by telescoping. The content is in the definitions: every term is an arm that can be computed with zero responses, and any LLM gap is bounded by the terms whose resources the LLM can access. For example, an LLM restricted to the same menu, model and information can only recover Δ_decision + Δ_myopia. The path dependence of the telescoping is resolved by a predeclared order plus a Shapley sensitivity analysis.

**T6. Design value of side information under a shared conjugate learner.** Take a linear target cᵀβ, rows X, a learner with prior N(m0, τ0²I), and truth β ~ N(μ, Σ). Writing δ = m0 − μ and Σ_eff = δδᵀ + Σ:

> b − β = V(m0 − β)/τ0² + V Xᵀε/σ²,  so  R_true(X) = cᵀ V Σ_eff V c / τ0⁴ + cᵀ V XᵀX V c / σ².

When Σ = τ0² I this is R_true = cᵀVc + (cᵀVδ)²/τ0⁴: the learner's own risk plus a squared-bias term. The uninformed planner minimizes cᵀVc and the informed planner minimizes R_true. Corollaries:

- **(a) Mean-shift information (C1) is second order.** The design value is at most (cᵀVδ)²/τ0⁴. Relative to the variance this is about σ²‖δ‖²/(τ0⁴ n). Calculation 4 below gives a squared-bias share of 1.7% at n = 32 with σ² = 0.25 and δ = 1.5, rising to 25% at n = 8 with σ² = 1. So the shifted regimes held a small but nonzero information term that no arm could use, because nobody was told the shift.
- **(b) Absent-edge information (C2) is first order even with a shared learner.** If a coefficient is truly 0 and equals the prior mean, never exciting it gives exactly zero error on that coordinate. The uninformed planner spends budget on it and injects noise. In the two-coefficient toy (calculation 5) the informed design's true risk is 0.26 to 0.36 times the uninformed design's.
- **(c) Prior-submission mode.** If the policy may submit a prior to the learner, the value becomes the full bias term plus the variance gain. This is a separate study, never mixed with design-only mode.
- **(d) Misleading text with a fully trusting planner gives unbounded harm.** A present coefficient that the text calls absent and that is never probed contributes c_j²β_j². Calibrated trust (a mixture prior with reliability r) makes exploration valuable again. This is a regime where adaptivity has value and the recovery delay is defined.

*Why it matters:* the dose-response curves an LLM's benefit should follow are predicted before any LLM is trained.

**T7. Corollary for the shuffled-feedback control.** Where the optimal policy is open-loop (T1) or nearly so (T2), a feedback-blind policy can be near-optimal. The loss from shuffling feedback is then at most the value of adaptivity plus the policy's own deviation. Gupta et al.'s shuffled-feedback control therefore has no power in the completed class. It must be run where E2 or E4 verify a positive value of adaptivity, and its expected effect size equals that value.

**T8. Cost-equivalent conversion.** When posterior risk scales as k/(n_eff) with n_eff ≈ n0 + n (plus prior pseudo-rows σ²/τ0²), a final-risk ratio r at budget B equals an equivalent cost saving of (n0 + B)(1/r − 1) units. For example, r = 0.9 at 8 + 24 rows is about 3.6 units. That is about the whole first-crossing headroom of the completed screen (3.04 units per system). The design-quality endpoint therefore answers the "lower experimental cost" question in cost units without first-passage noise. For reporting, use each system's computed expected error-versus-cost curve of the planner, not the asymptotic formula.

---

## 3. Supporting calculations

All of these were produced by `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w2/calc/calcs.py` (arithmetic only).

**1. Paired SDs implied by the screen's own t statistics (n = 24).** SD = |mean| × √24 / |t|.

| Contrast (source) | Mean log | SD | CV |
| --- | ---: | ---: | ---: |
| Design quality J, local information ×1.02, t = 2.8 (numerics-protocol §7B) | 0.020 | 0.035 | 1.75 |
| Design quality J, coverage ×1.35, t = 8.6 | 0.300 | 0.171 | 0.57 |
| Design quality J, random ×1.42, t = 7.3 | 0.351 | 0.235 | 0.67 |
| Design quality J, DPO ×2.10, t = 7.8 | 0.742 | 0.466 | 0.63 |
| Final posterior risk, TR − local information, t = −2.6 (§7A) | 0.055 | 0.104 | 1.88 |
| Final posterior risk, TR − coverage, t = −8.1 | 0.319 | 0.193 | 0.60 |

For comparison, first-crossing capped cost has a paired SD of about 6.2 units (results-analysis-verify §D), and realized final NMSE has a coefficient of variation of about 1 per run. Planning SDs for new contrasts: 0.10 to 0.25 on the log scale.

**2. Systems needed (paired, two-sided).** First table: α = 0.05, 80% power. Second table: Holm k = 3, 90% power.

| Effect | SD 0.035 | SD 0.10 | SD 0.15 | SD 0.20 | SD 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 5 | 33 | 75 | 132 | 207 |
| 7.5% | 2 | 16 | 34 | 61 | 94 |
| 10% | 2 | 9 | 20 | 35 | 55 |
| 20% | 1 | 3 | 6 | 10 | 15 |

| Effect | SD 0.10 | SD 0.15 | SD 0.20 | SD 0.25 |
| --- | ---: | ---: | ---: | ---: |
| 7.5% | 26 | 59 | 104 | 162 |
| 10% | 15 | 34 | 60 | 93 |

The brief's statements follow from these numbers: decision- and design-level endpoints need tens of systems, while first-crossing cost needs hundreds.

**3. Omission term in the screen's menu.** The prompt gives intervention bounds of [−2, 2] for X and Y, but the menu only sets ±1 and 0 (mechanism.md verbatim prompt). For one linear head with a balanced design at ±s, prediction variance at x_q is proportional to 1 + x_q²/s². At queries x ∈ {−1, 0, 1}, mean relative variance is 1.667 for s = 1 and 1.167 for s = 2, a ratio of 0.70. So the menu left about 30% of slope-dominated risk on the table for every arm. This is a property of the benchmark, not of any policy. Under a correct linear model a numerical optimizer captures it trivially. It becomes interesting only together with misspecification (boundary designs are risky under curvature), which is class C3.

**4. Mean-shift information (T6a).** One slope with unit-second-moment rows and τ0² = 1 gives squared bias / variance = V·δ², with V = 1/(1 + n/σ²).

| σ², n | δ = 0.75 | δ = 1.5 | δ = 3.0 |
| --- | ---: | ---: | ---: |
| 0.25, n = 2 | 0.062 | 0.250 | 1.000 |
| 0.25, n = 8 | 0.017 | 0.068 | 0.273 |
| 0.25, n = 32 | 0.004 | 0.017 | 0.070 |
| 1.0, n = 2 | 0.188 | 0.750 | 3.000 |
| 1.0, n = 8 | 0.062 | 0.250 | 1.000 |
| 1.0, n = 32 | 0.017 | 0.068 | 0.273 |

The design value is at most this share, and only part of it can be removed by design. Prediction: C1 qualifies in E4 only at small n_init, high noise and large δ.

**5. Absent-edge information (T6b).** Two coefficients with query weight on both. b2 is truly 0, which equals the prior mean. Each action probes one coefficient with the other clamped at 0. Budget n; the uninformed planner splits evenly, the informed planner puts everything on b1.

| σ² | n = 4 | n = 8 | n = 16 | n = 24 |
| --- | ---: | ---: | ---: | ---: |
| 0.25 (informed / uninformed true risk) | 0.280 | 0.265 | 0.258 | 0.255 |
| 1.0 | 0.360 | 0.309 | 0.280 | 0.270 |

This is a first-order effect, a factor of 3 to 4 in the toy. Realistic supergraphs will dilute it, and E4 measures by how much. For the "don't excite" option to exist, the menu must include actions that clamp the other parents at 0.

**6. LLM decision budget for a confirmation.** Using the brief's 0.4 to 0.7 s per decision: 120 systems × 2 replicates × 2 LLM arms × 24 decisions is 1.3 to 2.2 h. With 4 replicates it is 2.6 to 4.5 h. With 96 systems and 2 replicates it is 1.0 to 1.8 h.

---

## 4. Benchmark classes considered, by gap term

| Class | Gap term targeted | Dose variable (controlled) | Predicted-null anchor | Code change (from numerics-protocol §5 unless marked) |
| --- | --- | --- | --- | --- |
| C0 completed regime | none | n/a | the whole class | none |
| C1 prior-mean shift, text states the shift | Δ_info, second order | δ ∈ {0, 1.5, 3}, n_init ∈ {0, 2, 8}, σ² ∈ {0.25, 1} | δ = 0 | small (protocol); the informed control needs per-coefficient prior means in the planner's internal learner (small to medium) |
| **C2 known supergraph with sparse truth, text names absent edges** | Δ_info, first order; also law-awareness (structure learning) | fraction of absent edges ∈ {0, 1/3, 2/3}, reliability r ∈ {1, 0.8}, n_init ∈ {0, 2, 8} | no absent edges / uninformative text | **[assumption] small to medium**: the truth needs exact zeros on supergraph edges. Either a world-descriptor schema change (the coefficient SD is one scalar today) or a world graph distinct from the public graph. Whether the assembler enforces equality between world and public graph is unverified. The structure-aware control is a mixture over at most 8 subgraphs with closed-form marginal likelihoods. |
| C3 nonlinear truth, linear learner, queries off the menu | Δ_misspec (×Δ_omission) | curvature γ ∈ {0, 0.25, 0.5, 1} | γ = 0 (inward moves are harmful) | medium (nonlinear mechanisms, estimand, evaluator) |
| C4 indivisible costs, budget remainders | Δ_myopia | cost ratio κ ∈ {1, 2, 4, 8} | κ = 1 | small (protocol plus DP or rollout planner) |

All new classes vary the query law across systems, so state-blind schedules are penalized (the screen's fixed law made set_x_neg the best blind first move in 11 of 24 systems). Menus include boundary and clamp-at-zero actions where E1 or E4 show they matter, decided on LLM-independent grounds.

C2 is the most promising information class: the design value is first order, the change is small, it maps naturally to domain prose ("the inlet valve has no direct effect on outlet temperature"), and it has three clean planner levels:

- uninformed Gaussian planner (C1-control);
- law-aware planner that learns the structure from data, without text;
- text-informed planner.

The value of the text is the gap between the last two, and the LLM's legitimate advantage must come from there.

---

## 5. The experiments (ordered)

Common conventions:

- Statistical units are independent systems.
- Development and training namespaces are frozen and disjoint from confirmation.
- The primary endpoint for new work is the **truth-conditional expected log final NMSE of the realized design at a fixed budget**: the true parameters are known for synthetic systems, measurement noise is redrawn with 4,096 replicates, and the result is averaged over R replicate trajectories with common random numbers across arms. Secondary endpoints are realized error (sign-agreement check), public posterior risk, cost-equivalent saving (T8), public-stopping cost with reliability, and recovery delay.
- First-crossing cost is reported descriptively only.
- Design-only mode (one shared learner for all arms) is the default, so the LLM can contribute only through its choices.

### E1. Value-gap decomposition of the completed screen (must)

- **Question.** How large was each gap term between the target-risk planner and the best achievable policy in the completed class? That sets the largest advantage any policy, LLM or otherwise, could legitimately have shown.
- **Design.**
  - *Inputs.* Saved post-initialization states of all 24 development systems (identical across arms), the saved action sequences of all eight arms, the 560 target-risk receipts, and the private world descriptors (development data, post hoc).
  - *Reference arms*, all model-internal with zero environment responses:
    - R0: target risk as executed (32 draws);
    - R1: target risk with 4,096 draws and common random numbers across actions;
    - R2: plan-once multiset optimum (greedy plus local search);
    - R3: rollout planner (one-step lookahead over a target-risk base policy, 64 rollouts to the end of the budget);
    - R4: clairvoyant design (true θ plugged into the shared learner's risk);
    - R5: law-informed planner on the 12 shifted systems (knows the edge mean 1.5 and minimizes the truth-law risk of the shared learner);
    - R6: extended-menu planner (adds ±2 at the bounds);
    - R7: best leave-one-out state-blind schedule (graph × cost);
    - R8: EPIG criterion.
  - *Endpoints.* (a) Working-model design quality J from the common state. (b) Truth-conditional expected final NMSE. (c) For the LLM arms, decision-level relative regret against the 16,384-draw reference (deployment-forensics-verify), split into a state-blind component and a residual.
  - *Sample size.* n = 24, fixed by the completed screen. A paired log SD of 0.035 for near-equal designs gives SE ≈ 0.007, which resolves terms of 2% or more. The MC SE of J is about 0.3% at 4,000 draws.
- **Decision rule (predeclared).**
  - (i) "No-headroom verdict" if the upper 95% bound of (R0 − min(R1..R4))/R0 is below 5% on matched systems.
  - (ii) If R3 improves on R1 by 5% or more with a positive lower CI, a non-myopic control becomes mandatory in all later studies and T3 is narrowed.
  - (iii) If Δ_info (R5 versus R1 on endpoint b) is 5% or more, C1 is kept at n_init = 8 in E4. Otherwise C1 is tested only at low n_init.
  - (iv) Δ_omission (R6 versus R1) is reported. If it is 15% or more, successor menus include boundary actions, decided now on LLM-independent grounds.
  - *Positive* (no headroom): licenses "the November null was predicted; no policy could exceed about 2 to 5% over target risk in this class".
  - *Null or negative* (headroom of 5% or more): licenses "the null reflects endpoint and power, not the absence of headroom", with no rerun of the old study.
- **Prerequisites.** Saved files only (all local). Reuse the validated independent replay (numerics-protocol `acelib.py`, which matches the saved curves to 1e-14). No approvals.
- **Cost.** Zero responses and no CURC. About 1 to 2 local CPU-hours (rollouts dominate). 3 to 5 analyst-days.
- **Main risk.** The rollout and greedy references are themselves approximate, so the verdict can overstate the absence of headroom. Mitigation: also bound with the clairvoyant design R4.

### E2. Theory validation: adaptivity and greedy optimality in both directions (must)

- **Question.** Do the formal predictions hold numerically? Specifically: exactly zero adaptivity in the design-only subclass (T1); adaptivity at most EVPI_design and growing as initial information shrinks (T2); greedy near-optimal where γ ≈ 1 (T3); and the T6 formula matching simulation.
- **Design.** Fresh synthetic development systems from a frozen namespace, linear-Gaussian with known noise, model-internal, no LLM.
  - S0, design-only subclass: chain or fork; only actions that clamp the target head's parents; linear queries (do(Y) for Z in the chain, do(X) in the fork); budget 24 at unit cost; exact multiset optimum by enumeration (at most 2,925 multisets).
  - S1, completed-class mechanics: 3-node chain, fork and collider with the 8-action menu, n_init ∈ {0, 1, 2, 4, 8, 16}, σ² ∈ {0.25, 1}.
  - S2: 5-node random DAG, n_init ∈ {0, 2, 8}.
  - *Arms.* Target risk (4,096 draws, common random numbers), plan-once, rollout (adaptive non-myopic), clairvoyant, random, best leave-one-out static schedule.
  - *Quantities.*
    - V_adapt = (J_OL − J_rollout)/J_OL;
    - EVPI_design = (J_OL − J_clair)/J_OL;
    - greedy versus best multiset;
    - γ on small budgets;
    - analytic versus simulated Δ_info for linear targets.
  - *Sample size.* 48 systems per cell (S0: 48; S1: 12 × 48; S2: 3 × 48). SE ≈ 0.005 on the mean log ratio at SD 0.035, which resolves 1.5% at z ≈ 3.
  - *Endpoint.* Expected final posterior NMSE, exact under A3 here.
- **Decision rule.**
  - (P1) In S0, |J_adaptive − J_best multiset| ≤ 2 MC SE in at least 95% of systems → pass. A failure means a bug or a theory error, and all work stops until it is fixed.
  - (P2) V_adapt at n_init = 0 exceeds V_adapt at n_init = 8 by at least 3 points (paired lower 95% CI > 0), and V_adapt ≤ EVPI_design + 2 SE in at least 90% of systems.
    - If both hold, the "initial-information" explanation of the null is supported.
    - If V_adapt ≈ 0 at every n_init, the null is stronger: E4 must look to structure or misspecification.
    - If V_adapt is systematically above EVPI_design, the T2 bound is wrong and the theory section is narrowed to empirical statements.
  - (P3) Greedy ≥ (1 − e^{−γ}) of the best multiset everywhere; report the empirical ratios.
  - (P4) T6 analytic versus simulated within 10%.
  - *Positive* licenses the theory section with numerical verification. *Null or negative* narrows it.
- **Prerequisites.**
  - In-process vectorized planner (removes the one-process-per-decision artefact that makes a numerical episode take about 48 s).
  - Higher draw cap and common random numbers in target risk (capped at 256 draws today).
  - Small protocol module for n_init and the new graphs.
  - Rollout planner.
  - Written protocol with frozen namespaces.
- **Cost.** About 5 to 15 local CPU-hours [assumption: 10 to 50 ms per vectorized decision, rollouts 64 to 512 times that]. Engineering of about 1.5 to 2 weeks, shared with E3 and E4.
- **Main risk.** Optimization error in the references can confound the inequality checks. Mitigation: exact enumeration in S0 and small-budget exact DP on a subset.

### E3. Equal-information numerical control suite and its qualification (must)

- **Question.** Do the numerical controls that get exactly the LLM's information (state, the law, text in oracle or parsed form) reach their theoretically computable values? Only then are later LLM contrasts made against strong, correctly implemented controls.
- **Design.** Implement and qualify on development states:
  - C1, uninformed planner: target risk with 1,024 to 4,096 draws and common random numbers. This is "the planner" from here on.
  - C2, EPIG (Bickford Smith et al. 2023): the published prediction-oriented criterion.
  - C3, non-myopic planner: rollout, plus exact multiset or DP where T1 applies.
  - C4, oracle-informed planner: knows the side information numerically and minimizes the truth-law risk of the shared learner. For sparsity classes it draws fantasy truths from the true-subgraph learner while scoring the shared supergraph learner.
  - C5, law-aware planner: knows the generator law (for example spike-and-slab over at most 8 subgraphs) but not the instance and has no text.
  - C6, reliability-aware text planner: uses the text through its true reliability r as a mixture prior. This is the Bayes-optimal user of the text and also the privileged teacher for E7.
  - C7, rule-based parser plus C4: written from the training phrasing families only, before any LLM scoring.
  - C8, GO-CBED-style amortized goal-oriented policy: a transformer over the same numeric state, trained on the training namespace, in a no-text variant and a parsed-text variant. Use the public GO-CBED code too if the lead approves the download.
  - C9, semantic-feature numerical policy: numeric features plus a frozen-LLM embedding of the text, trained on the LLM's training data (the Gupta et al. control).
  - C10: best leave-one-out state-blind schedule, and random.
  - *Sample size.* 48 development systems × 8 states per qualifying cell. Basis as in E2.
- **Decision rule (qualification, predeclared).**
  - C1: median top-two gap of at least 2 MC SE in at least 75% of decisions.
  - C4: reaches the computed Δ_info ceiling within MC error.
  - C5 ≥ C1 on structure-uncertain cells.
  - C8: design quality within 5% of C1 (no text) on held-out development states at a budget of 1×, 4× or 16×, with the learning curve reported. If it fails at 16×, the paper labels it an under-trained comparator and may not claim "beats amortized numerical policies".
  - C9: at least 90% held-in text-to-structure accuracy, otherwise it is labeled under-powered.
  - Every failure is preserved and reported.
  - *Positive* licenses the strong-control requirement. *Negative* means the control is fixed or the claim is narrowed.
- **Prerequisites.**
  - E2 machinery.
  - Per-coefficient prior means and variances inside the planner's internal model only (the shared learner is unchanged).
  - Approval to download GO-CBED code, or an in-house reimplementation.
  - Local training or inference approval for C8 and C9 (C9 executes a frozen model).
- **Cost.**
  - 2 to 3 engineer-weeks.
  - 5 to 20 local CPU-hours.
  - C8 training: 1 to 4 GPU-hours or 10 to 30 local CPU/MPS hours [assumption; GO-CBED's own budget unverified].
  - C9 embeddings: one forward pass per text, under 1 h.
- **Main risk.** Reviewers may judge a reimplemented GO-CBED unfaithful. Mitigation: public code if approved, and the training-budget curve.

### E4. Headroom atlas and LLM-independent class selection (must)

- **Question.** In which expressible classes, and at which controlled doses, is there verified headroom between the strongest no-text numerical control and the ceiling reachable with information or a model the planner lacks? It must also be detectable at a feasible sample size.
- **Design.**
  - Numerical arms only (the E3 suite), development namespaces, design-only mode.
  - *Classes and doses.*
    - C1: δ ∈ {0, 1.5, 3} × n_init ∈ {0, 2, 8} × σ² ∈ {0.25, 1}.
    - C2: 4- or 5-node supergraphs with known topological order, absent fraction ∈ {0, 1/3, 2/3} × r ∈ {1, 0.8} × n_init ∈ {0, 2, 8} × σ² ∈ {0.25, 1}.
    - C4: κ ∈ {1, 2, 4, 8} × n_init ∈ {2, 8}.
  - Query laws vary across systems.
  - *Arms.* C1-control, C5, C8 (no text), static, random, and the ceilings C4, C6 and C3.
  - *Endpoints.* The primary endpoint, cost-equivalent saving, and the predicted n for the LLM contrast.
  - *Sample size.* 48 systems per cell (about 60 cells, about 2,900 systems), R = 2 trajectories with common random numbers. At the worst SD (0.24) the SE at n = 48 is 0.035, which resolves a 10% gap at z ≈ 2.7.
- **Decision rule.** Predeclared before any LLM is scored on any class, and independent of LLM performance. A class qualifies at a dose if all four hold:
  - (i) the ceiling gap G = log J(strongest no-text control) − log J(ceiling) is at least log 1.10, with a lower 95% CI of at least log 1.05;
  - (ii) the predicted n to detect G/2 (capture ρ = 0.5) at α = 0.05/3 two-sided and 90% power, using the cell's paired SD inflated by 1.5 for LLM noise, is at most 96 systems;
  - (iii) its zero-dose anchor has |G| < 2%;
  - (iv) G rises across doses (a dose-response exists).

  Choose at most two classes: largest G at feasible n first, then smaller code change, then relevance to the experimentalist task.

  If none qualifies, LLM superiority work stops. The paper becomes theory, atlas and diagnostics. The negative result narrows the claim to "in classes this codebase can express there is no information or model headroom an LLM could exploit". No other benchmark may be adopted without a new LLM-independent E4-type justification.

  *Predictions:*
  - C1 small, except at n_init ≤ 2, σ² = 1 and δ = 3;
  - C2 large;
  - C4 nonzero only for κ > 1 with remainders, and closed by DP. C4 can therefore host only a mechanism test against the myopic planner (E9), never a headline.
- **Prerequisites.**
  - E2 and E3 machinery.
  - Protocol modules for C1, C2 and C4.
  - The sparse truth law for C2 (see Section 4 on the unverified graph-equality check).
  - Frozen namespaces and a written protocol.
- **Cost.** 20 to 60 local CPU-hours [assumption: about 40,000 in-process episodes at 1 to 5 s; the law-aware and rollout arms dominate]. 1 to 2 engineer-weeks after E2 and E3.
- **Main risk.** Dilution: the first-order C2 effect from the toy may shrink in realistic supergraphs. The thresholds are judgment calls and must be frozen before running.

### E5. Instrument calibration with synthetic policies of known capture (must)

- **Question.** Do the planned endpoint, analysis and n recover a known information-capture ratio without bias and with nominal error rates, before any LLM is scored?
- **Design.** In the selected class or classes, on development systems:
  - π_ρ takes C4's action with probability ρ and C1's action otherwise, for ρ ∈ {0, 0.25, 0.5, 1}. Capture is defined on the value scale and the mapping from ρ to capture is calibrated here.
  - A text-trusting, state-blind schedule.
  - A leakage policy that reads the private truth and should exceed the ceiling.
  - A feedback-blind copy of C1, to measure the power of the shuffled-feedback control as a function of V_adapt (T7).
  - Run the complete planned E10 analysis: paired endpoint, R = 2 with common random numbers, dose-response regression of benefit on the E4-predicted gap, placebo non-superiority, Holm.
  - *Sample size.* The planned confirmation size, 156 systems, repeated over 20 independent namespace draws or 200 system-bootstrap resamples.
- **Decision rule.** Pass requires all of:
  - |ρ̂ − ρ| ≤ 0.1;
  - 95% CI coverage between 0.92 and 0.98;
  - power at least 80% at ρ = 0.5;
  - false positives at most 5% at ρ = 0 and in the placebo;
  - the leakage policy flagged in at least 95% of draws;
  - design-quality and realized-error endpoints agree in sign in at least 90% of draws at ρ ≥ 0.5.

  On failure, change n, replicates or endpoint now, never after LLM scoring. *Positive* licenses calibrated capture-ratio claims. *Negative* means the confirmation must be redesigned.
- **Prerequisites.** E3, E4 and frozen analysis code.
- **Cost.** 2 to 6 local CPU-hours and 3 to 5 analyst-days. No LLM and no CURC.
- **Main risk.** Decision-level mixing is nonlinear in value. That is handled by calibrating the mapping.

### E6. Zero-shot information-use probe on frozen local models (should)

- **Question.** Does any locally cached frozen model shift its choices toward what the side information warrants on states where that information changes the optimal action, and leave them unchanged otherwise?
- **Design.**
  - *States.* Offline states from the selected class (development namespace), 64 systems × 6 states.
  - *Text conditions.* Absent, correct, shuffled (another system's hint) and misleading. Plus content-free state controls, and menu-order and identifier permutations.
  - *Models.* Qwen3-0.6B, Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (all in the local HF cache).
  - *Prompt.* Compact, rounded, decision-relevant posterior summary plus text plus menu. No planner scores.
  - *Scoring.* Whole-command log-probabilities, length-normalized and raw, not greedy trie decoding. Thinking off versus on for Qwen3.
  - *Labels.* C1 versus C4 rankings at 4,096 draws with common random numbers. Switch states are those where the informed best action differs from the uninformed best by at least 2 MC SE.
  - *Endpoints.* Δ_switch = P(informed best | correct) − P(informed best | absent) on switch states, the same on non-switch states, and regret under the truth-law risk. Bootstrap by system.
  - *Sample size.* About 384 states per condition. Detecting Δ = 0.10 at binomial SD 0.5 with a clustering design effect of 2 needs 2 × ((1.96 + 0.84) × 0.5/0.1)² ≈ 392 states.
- **Decision rule.** A model "uses the information" if all of these hold:
  - Δ_switch ≥ 0.10 with a CI excluding 0;
  - misleading text moves choices away from the informed best by at least 0.05;
  - shuffled text gives |Δ| ≤ 0.05;
  - non-switch accuracy does not drop by more than 0.05.

  Predicted: null for 0.6B (literature and project history), possibly positive for 7B.
  - Positive at 0.6B: E7 can focus on decisions rather than reading.
  - Positive only at 7B: capacity is the bottleneck, and larger-model training becomes an ask.
  - Null for all models: training must teach reading, and the paper makes no zero-shot claims.
- **Prerequisites.** Approval for local forward-only inference, including the 7B weights. E4 selection. Frozen prompt format.
- **Cost.**
  - No CURC.
  - About 5,000 to 6,000 scorings at about 1 to 2 s (0.6B, 1.5B) and 3 to 6 s (7B) on the M5 Max is about 4 to 10 h wall [assumption; the GPU slice took 0.4 to 0.7 s per 0.6B decision on 2,000 to 4,000-token prompts].
  - 2 to 3 analyst-days.
- **Main risk.** Prompt-surface sensitivity, which has dominated every past LLM arm. Mitigation: permutations and whole-command scoring.

### E7. Trained LLM policy on the information channel: development dose-response (must)

- **Question.** Does a trained LLM action policy capture a predicted share of the information value? Its benefit should track the computed dose, vanish where theory says it must, show bounded harm and recovery under misleading text, and beat the strongest numerical control that lacks the text.
- **Design.**
  - *Training.* Training namespace of the selected class.
    - Targets: graded listwise softmax of the privileged teacher C6's scores, on states visited by a teacher-student mixture (DAgger-style relabeling).
    - Identifier and menu permutation per example.
    - Hint texts from human-written phrasing families, split into training and held-out families and frozen before training.
    - The recipe comes from the training workstream. Any recipe is acceptable if it passes the gates. Kriz et al.'s on-policy GRPO recipe is the literature baseline.
  - *LLM arms.*
    - L1: LLM policy (text plus numeric state, no planner scores).
    - L2: LLM re-ranker over C1's top 3 with scores (hybrid), paired with an argmax-of-prompt baseline.
    - L3: LLM compiler plus C4 (the LLM reads the text into a structure vector and the planner decides).
    - L4: L1 without text.
    - L5: L1 with shuffled text.
    - L6: the same architecture without language pretraining trained identically, or C9 if that is infeasible.
  - *Numerical arms.* C1, C5, both C8 variants, C7, the ceilings C4 and C6, static and random.
  - *Gates.*
    - G-state (no-text states): L1's decision-level relative regret below the best state-blind schedule's, capturing at least 25% of the plan-once-versus-static gap.
    - G-text: Δ_switch ≥ 0.20 on held-out phrasing families.
  - *Episodes.* E4 doses (zero, mid, high), a misleading-text cell and a no-information placebo cell. 48 development systems per cell, R = 2 with common random numbers. Shuffled-feedback control (L1 with permuted measured values) in the cell with the largest V_adapt (T7).
  - *Endpoints.* Primary as above. Secondary: cost-equivalent saving, realized error, and recovery delay (acquisitions until the first probe of a claimed-absent edge, compared with C6's).
  - *Analysis.* Regression of paired benefit (L1 versus C1, C5 and C8) on the E4-predicted gap.
- **Decision rule (development, predeclared).** Advance to E10 only if all of:
  - the benefit slope has a lower 90% CI above 0;
  - ρ̂ ≥ 0.3 at high dose;
  - zero-dose and placebo benefit ≤ +2% and at most the ceiling;
  - L1 (or the predeclared L2) beats the strongest no-text control at high dose by at least 5%;
  - misleading-text harm is at most that of a fully trusting planner, and recovery delay is at most twice C6's.

  Kill if ρ̂ ≤ 0.1, or if L1 is more than 5% worse than C1 at zero dose (it cannot even match the planner on numeric state).

  *What each outcome licenses:*
  - Positive: freeze and confirm.
  - L3 ≥ L1: confirms T4's prediction. The confirmable claim becomes "the LLM's contribution is reading information into the decision", and the lead's predeclared choice of role (L1, L2 or L3) goes to E10.
  - Null: "the LLM did not turn textual information into decisions at this scale". The claim narrows.
  - Negative (worse than C1 even with correct text): the method is redirected to L2 or a larger model. The benchmark is never changed.
- **Prerequisites.**
  - E3 to E5 passed. E6 is informative but not required.
  - A new GPU ceiling.
  - The training workstream's recipe.
  - Frozen text families and development protocol.
- **Cost.**
  - Training: 2 to 6 GPU-slice-hours. 8,000 listwise updates at 0.62 s is about 1.4 h. An on-policy stage of 250 updates with 128 single-decision rollouts adds 1 to 4 h [assumption: batched generation].
  - Development episodes: 48 × 5 cells × 2 reps × 3 LLM arms × 24 decisions ≈ 34,560 decisions at 0.4 to 0.7 s ≈ 3.8 to 6.7 h. This can be reduced by scoring L4 to L6 at decision level only.
  - Numerical arms: 5 to 20 CPU-hours in-process.
  - Total about 6 to 13 slice-hours, far above the remaining 2,742 slice-seconds, so a new ceiling is needed.
  - 3 to 4 weeks.
- **Main risk.** L1 fails G-state, as every past LLM arm failed to read numeric state. Then only L2 and L3 remain.

### E8. Misspecification: curvature dose-response and wind-tunnel simulator transfer (should)

- **Question.** When the planner's linear model is misspecified in a way that language, or pretrained physics, can describe, does the LLM capture a predicted share of Δ_misspec, with no benefit at zero curvature?
- **Design.**
  - *Stage A (numerical).*
    - Nonlinear truth (quadratic or saturating, γ ∈ {0, 0.25, 0.5, 1}) with a shared linear learner.
    - Operating-point queries off the menu (for example x* = 1.5) with a 9-level action grid on [−2, 2].
    - Arms: linear planner, oracle-informed planner (knows the form and the γ law), robust model-averaging planner (linear, quadratic and cubic bases, no text), and GO-CBED-style.
    - Same gate as E4.
    - Prediction: at γ = 0, extreme designs are optimal and inward moves are harmful (the negative control). At γ > 0, local designs win (the Box-Draper bias-variance trade-off) [MEM].
  - *Stage B (LLM, only if Stage A qualifies).* The E7 recipe trained on synthetic curvature families with physics-style text. Dose-response on γ, including γ = 0 with text saying "linear".
  - *Stage C (transfer).* `causalchamber` wind-tunnel simulators (affinity-law fan speed, cubic current, squared-speed pressure, linear hatch term), using the documented prose descriptions.
    - Controls: linear planner, a physics-basis planner using the published models (the well-specified ceiling), and the robust planner.
    - 48 simulated systems from randomized parameters within the documented ranges.
    - Claim scope: "one simulator family".
- **Decision rule.**
  - Stage A: as E4.
  - Stage B: as E7 (slope > 0, γ = 0 benefit ≤ +2%, ρ̂ ≥ 0.3).
  - Stage C: report ρ̂ relative to the physics ceiling with a system-bootstrap CI. Positive if the lower CI > 0 and the LLM beats the robust no-text planner.
  - *Positive* partly meets submission condition 4 (transfer beyond the synthetic generator).
  - *Null* means the decomposition is reported without an LLM claim.
  - *Negative at γ = 0* documents a calibrated-trust failure.
- **Prerequisites.**
  - Nonlinear mechanisms, the estimand (the expected response, because the noise-disabled structural mean is not the expected response under nonlinearity) and the evaluator: a medium change.
  - Basis learner for the planner's internal model.
  - `pip install causalchamber` (approval).
  - The E7 recipe.
- **Cost.**
  - 2 to 4 engineer-weeks.
  - Stage A: 10 to 30 local CPU-hours.
  - Stage B: about E7's GPU cost unless training is shared across classes.
  - Stage C: CPU only (numpy simulators).
- **Main risk.** A simple numerical locality rule may capture most of the misspecification headroom.

### E9. Non-myopic budget class: bounded benefit against the myopic planner, none against DP (could)

- **Question.** Can a trained LLM capture non-myopic value (budget remainders, indivisible costs) that the one-step ratio planner misses, with a benefit bounded by the DP control as T4 requires?
- **Design.**
  - C4 with κ ∈ {1, 2, 4, 8}, design-only subclass with exact DP over count vectors (T1, part 3).
  - Arms: myopic target risk, DP or multiset optimum, rollout, LLM trained on DP budget-to-go targets, LLM trained on one-step targets (ablation), and GO-CBED-style.
  - 48 development systems per κ. Endpoints as E7.
- **Decision rule.**
  - Predicted: all equal at κ = 1, and LLM_multi − myopic ≤ DP − myopic. One-step training should capture about 0.
  - "Learned budget-aware planning" may be claimed only if ρ̂ ≥ 0.5 with CI > 0 and ρ̂(one-step) < ρ̂(multi-step).
  - Superiority over DP is impossible. An apparent advantage beyond MC error triggers the leakage audit.
- **Prerequisites.** E4 shows C4 headroom against the myopic planner. DP planner. GPU.
- **Cost.** 2 to 4 GPU-slice-hours (reusing the E7 infrastructure) and under 10 CPU-hours.
- **Main risk.** Low payoff for the headline.

### E10. Fresh-system confirmation with dose and placebo strata (must)

- **Question.** On fresh systems with frozen methods, namespaces, endpoints and failure treatment, does the LLM role carried from E7 do all of the following?
  - beat the strongest no-text numerical control;
  - show benefit that tracks the predicted dose;
  - show no benefit where theory predicts none;
  - show no gain beyond the computed ceiling.
- **Design.**
  - *Namespace.* Fresh confirmation namespace in the selected class.
  - *Strata* (156 systems, R = 2 with common random numbers):
    - high dose: 36;
    - mid dose: 36;
    - zero dose: 36 (in-class placebo);
    - misleading text: 24;
    - completed-regime placebo: 24 fresh systems in the new pipeline (not a rerun).
  - *Arms* (at most 10, so the 8-policy roster limit must be raised):
    - the frozen LLM role from E7;
    - C1 at 4,096 draws with common random numbers;
    - C5;
    - C8, the best no-text variant fixed on development data;
    - C7;
    - ceilings C4 and C6;
    - static and random.
  - *Endpoints.* Primary as defined. Secondary as defined.
  - *Hypotheses* (Holm, k = 3):
    - H1: L beats the strongest no-text control on pooled high and mid strata.
    - H2: the dose-response slope is positive, and the zero-dose benefit has an upper 95% bound of 5% or less.
    - H3: placebo gain has an upper bound of at most the computed ceiling + 5%, and the leakage audit is clean.
  - *Sample size.* For a 10% effect (half of a verified gap of at least 20%) at paired SD 0.15 to 0.20 with Holm k = 3 and 90% power, 34 to 60 systems are needed. 72 are planned in the pooled informative strata. If E5 calibration says otherwise, n is increased before the freeze.
- **Decision rule.**
  - The superiority claim requires H1 and H2 after Holm and H3.
  - H1 passing with H3 failing: no claim until the leakage audit resolves.
  - H1 failing: the claim narrows to development-only evidence. No benchmark substitution. The paper reports theory, atlas and the failed confirmation.
  - In every case, failures are preserved, responses accounted for and an independent review done.
- **Prerequisites.**
  - E7 pass.
  - Custody redesign (brief §5).
  - Raised roster limits.
  - Common random numbers allowed by lifting the arm-namespace rule.
  - New ceiling.
  - Frozen protocol hash and analysis code.
  - An independent reviewer.
- **Cost.**
  - LLM decisions: 156 × 2 × 1 or 2 LLM arms × 24 ≈ 7,500 to 15,000 decisions at 0.4 to 0.7 s ≈ 0.8 to 2.9 slice-hours, plus service overhead.
  - CPU: the brief's 22 to 35 core-hours for 100 to 150 systems is the upper bound (it carries the subprocess artefact). In-process numerics are far cheaper.
  - 2 to 3 weeks including custody.
- **Main risk.** Shrinkage from development to fresh systems. Rankings reversed twice before in this project (prior-evidence-core F15).

---

## 6. Sequence and gates

1. **Week 1:** E1, plus writing T1 to T8. No approvals needed.
2. **Weeks 1 to 4:** E2 and E3 in parallel (shared in-process planner).
3. **Weeks 3 to 5:** E4, once E2's P1 passes.
   - **Gate A** (LLM-independent class selection). If no class qualifies, write the theory, atlas and diagnostics paper; ICML on 22 January 2027 is feasible for that version only.
4. **Weeks 5 to 6:** E5 and E6 in parallel. E8 Stage A and the numerical part of E9 may also run here.
   - **Gate B:** the instrument must be validated before any LLM episode is scored.
5. **Weeks 6 to 10:** E7.
   - **Gate C:** the development criteria.
6. **Weeks 10 to 13:** freeze, then E10. E8 Stages B and C are optional in parallel.

The full agenda fits NeurIPS 2027 (estimated May) or TMLR. UAI 2027 (estimated February) is tight.

---

## 7. Required comparators from the literature

- **Required, computed in-house:**
  - Exact Bayesian references: cost-constrained Bayesian optimal static design (Chaloner and Verdinelli 1995), plus DP or rollout references (Huan and Marzouk, verified against linear-Gaussian solutions).
  - The headroom ladder: random, coverage or static, best static, myopic, non-myopic, clairvoyant.
- **Required, published criteria:**
  - target risk;
  - EPIG (Bickford Smith et al., AISTATS 2023), the goal-oriented prediction criterion matched to the endpoint;
  - local information as the parameter-information foil.
- **Required, amortized goal-oriented causal design:** GO-CBED (Zhang et al., TMLR 2026). Its motivating example is a known-graph linear-Gaussian SCM. It needs no-text and parsed-text variants and a training-budget sweep, using the public code if the download is approved (stated public, not opened).
- **Required whenever text is present:** a semantic-feature numerical policy (Gupta et al. 2025), and the oracle-encoded, rule-parser and LLM-compiler controls.
- **Required LLM-side baseline:** Kriz et al. (arXiv October 2026) on-policy GRPO recipe on the same base model. Use EVOLvE-style algorithm-guided inputs only with an argmax-of-prompt baseline.
- **Optional hybrids:** LeGIT-style warm start then planner; LMABO-style LLM choice among acquisition rules.
- **Not required, with reasons:**
  - CAASL: structure-learning objective, no code found, about 3 days on 3 A100s per model. Required only if a structure-learning endpoint is adopted.
  - DAD and iDAD: only if BoxingGym's DAD tasks are used.
  - BED-LLM: question-asking with the LLM's own beliefs. Not applicable.

---

## 8. Asks for the lead researcher

1. Agree that numerical-only work (E1 to E5, E8 Stage A, E9 numerics) runs locally under written protocols with frozen namespaces (no CURC, no LLM).
2. Approve local forward-only inference with the cached Qwen3-0.6B, Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (E6, C9 embeddings). Using 7B counts as larger weights.
3. A new compute ceiling:
   - E7: about 6 to 13 GPU-slice-hours and 5 to 20 core-hours;
   - E10: about 1 to 3 slice-hours, with up to 22 to 35 core-hours and a custody redesign;
   - or approval to run these on the M5 Max.
4. Approve downloading the public GO-CBED code, or rule that only a reimplementation is used.
5. Approve `pip install causalchamber` (MIT) for E8 Stage C. Remote Lab sign-up is not needed in this agenda.
6. Before E7, decide which LLM role counts toward the submission condition: policy (L1), hybrid re-ranker (L2) or compiler plus planner (L3).
7. Approve the benchmark change: classes selected by E4's LLM-independent criteria replace the completed class, recorded before any LLM scoring.
8. Approve protocol changes:
   - more than 8 policies per roster;
   - common random numbers across arms;
   - truth-conditional design quality as the primary endpoint in place of first crossing;
   - query laws varied across systems.
9. Pick the target venue and date, accepting that 22 January 2027 can carry only the theory and diagnostic version.
10. Accept the theory, atlas and diagnostics paper as the outcome if E4 finds no class or E7 or E10 fail.
11. Decide who writes the hint-text phrasing families: a human, or Claude, frozen before training. It must never be the policy model.

---

## 9. What not to do

- No LLM superiority test in the completed class or in any class without E4-verified headroom.
- No rerun of the 24-system study or the old pipelines. The placebo stratum consists of fresh systems in the new pipeline.
- No first-crossing capped cost or single-run realized NMSE as a primary endpoint.
- No choosing classes, doses, thresholds or menus after seeing any LLM result, and no tuning on the 24 development outcomes.
- No planner scores in the L1 policy prompt, and no L2 without the argmax-of-prompt baseline.
- No comparing an LLM that sees text against numerical controls that do not, unless the oracle-informed, rule-parser and compiler controls are also run.
- No reading a gain beyond the computed ceiling as success. It triggers the leakage audit.
- No running the shuffled-feedback control where V_adapt ≈ 0 and treating its null as evidence of learning (T7).
- No letting arms use different learners in design-only mode. Prior-submission mode only as its own predeclared study.
- No treating the collapsed learned ranker as the amortized baseline. GO-CBED-style must be trained with a budget sweep.
- No carrying over the fixed query law, the 0.04 threshold, the 8 + 24 budget or the ±1-only menu without E1 or E4 justification.
- No using the policy model to write or paraphrase its own evaluation texts.

---

## 10. Biggest uncertainties

- Whether any expressible class has enough design-only information value once realistic supergraphs dilute the factor of 3 to 4 seen in the toy (E4 decides).
- Whether a 0.6B model can read both numeric state and text. Every past LLM arm collapsed. Kriz et al.'s success used categorical state.
- T4 predicts that compiler plus planner should match or beat the policy when the compiler's output format suffices. The honest headline may then be "LLM as information channel", which may not meet the stated condition.
- The design-quality endpoint ignores noise-chasing adaptivity. The realized-error secondary may disagree in classes where adaptivity has value.
- T2 may resist a clean proof and remain a conjecture with numerical support.
- Fidelity and training budget of a GO-CBED-style comparator.
- Text realism: templated hints are parseable by a rule, which removes the LLM's legitimate edge. Hand-written held-out phrasing families are laborious, and LLM-written ones risk shared biases.
- Engineering time (about 6 to 10 weeks to E7) against the venue dates.
