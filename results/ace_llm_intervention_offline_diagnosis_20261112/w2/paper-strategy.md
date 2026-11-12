# ACE experimental agenda from the paper-strategy lens

Prepared 2026-11-12 for the lead researcher. This is a design document only. Nothing under `/Users/pat/code/ACE` was modified, and nothing was executed beyond reading files and two read-only directory listings. Every number not derived here comes from the state-of-evidence brief (`w2/brief.md`) or the verified w1 notes, cited by note name. Derivations are in Section 11. Assumptions and judgments are marked **[A]**.

---

## 0. Bottom line

1. **What can be claimed is decided by where the decision-relevant information lives.** Suppose the planner and the LLM have the same correct model, the same information and the same objective. Then no policy, LLM or otherwise, beats the Bayes-optimal sequential rule in expectation (agenda line 49, numerics-protocol section 3). In the completed known-graph linear-Gaussian class, sequential adaptation is worth −0.7% ± 1.7% (brief, Finding A). An LLM can therefore win only through one of four sources:
   - information the planner cannot read (goals or mechanism knowledge stated in language),
   - the planner's approximation error,
   - misspecification,
   - compute amortization.

   At this scale only the first is a credible and novel source (literature-policy 4.2 and 5.3).
2. **Build a spine that pays off however the central question resolves, and gate everything expensive.**
   - The spine: E1 (headroom certification and endpoint freeze, numerical only), E2 (offline forensic scoring of the saved adapters) and E3 (imitation ladder with knockouts and matched numerical students). Together these produce a complete diagnostic paper.
   - The positive branch: E5 (development pilot), E6 (preregistered confirmation), E7 (external simulator) and E8 (hardware). These run only after an LLM-independent admission rule (E1) and a capability gate (E3) pass.
   - E4 is a conditional escalation.
   - E9 (delivery-study closure) runs in parallel and makes a separate, outcome-independent TMLR paper possible.
3. **Realistic outcomes [A].**
   - A rigorous diagnostic paper (framing F3) is close to certain. Its core evidence can be ready by early to mid January.
   - A confirmed positive LLM result (F1 or F2) is roughly a 10 to 20% outcome. It realistically targets NeurIPS 2027 (estimated May) or TMLR.
   - ICML 2027 (abstract January 14, paper January 22) is realistic only for F3.
4. **Correct the public record first.** arXiv 2602.02451 and `README.md` (lines 14 and 114, re-checked today) still advertise the "70-71% improvement" that the September audit invalidated. Correct both before any new submission or talk. This is the lead's decision (Section 7).
5. **Change where the work runs.** The November cycle spent about 39 active hours to run a 2.4-hour scientific kernel, and half of the failed jobs were the custody harness rejecting valid environments (ledger-ops). Do development locally with light custody. Reserve heavy custody for the single confirmatory run, or redesign it. Both options need the lead's approval.

---

## 1. Working backward: the target paper and its fallbacks

### 1.1 Thesis

The target paper asks where a trained language model can improve on explicit SCM experiment planning. It answers with three tools: benchmarks certified to have headroom before any LLM sees them, decision-level endpoints, and controls that hold information fixed.

- **If positive:** experiment classes exist in which the decision-relevant information (the experimentalist's goal, or knowledge about mechanisms) is available only in language. In such a class, an SCM-conditioned policy trained by privileged-teacher distillation recovers a measured share of an oracle planner's advantage. On fresh systems it lowers expected cost to a reliable answer compared with three controls: the strongest numerical planner without the text, a rule parser, and a numerical student given the same information. It uses measurements, which a shuffled-feedback control tests. It loses the gain when the text is removed. It transfers to an independently authored wind-tunnel simulator.
- **The equal-information ceiling, stated in either case:** the same paper proves and measures that when the planner has the same information, no policy beats it in the canonical class.
- **If null or negative:** the claim becomes a rigorous diagnosis.
  - The canonical class leaves no room for any learned policy.
  - First-crossing cost endpoints cannot resolve policies at feasible sample sizes. Decision-level endpoints need 3 to 17 systems for the same contrasts.
  - The collapse of small trained policies, language and numerical alike, has identified causes, and those causes are confirmed experimentally.
  - Either a repaired recipe restores state-dependent planning without exceeding the planner, or no recipe tested does.
- **In both cases:** every earlier adverse result is reported, and the delivery study is published separately.

### 1.2 Three framings and the evidence each needs

| Framing | Central claim | Evidence required | Meets the five submission conditions? | Venue fit |
|---|---|---|---|---|
| **F1. Positive, language-conditioned intervention policy** | An SCM-conditioned LLM action policy (direct, or adjudicating over planner scores) recovers ≥30% of the oracle advantage that is available only in text. It beats the uninformed planner (H1) and a rule parser (H2) on fresh systems. The gain disappears without the text, the policy uses measurements, and the result transfers to an independent simulator. The equal-information ceiling is proved and shown. | E1 admits a class. E3 (or E4) passes. E5 qualifies a direct or adjudicator variant. E6 confirms H1 and H2. E7 holds. E2 and E3 supply the anatomy section. E8 is optional. | Yes, provided the lead accepts that a benefit from information access satisfies condition 2 (ask 12). | NeurIPS 2027 (estimated May) main track, or TMLR. UAI 2027 (estimated February) only if E6 finishes by early February, which is unlikely. |
| **F2. Narrowed: LLM as a knowledge compiler for a numerical planner** | The direct LLM policy does not qualify, but the hybrid does: the LLM turns prose into an objective or prior, and the numerical planner acts on it. The LLM contributes decision-relevant input, and action selection stays numerical. | E1, E5 with the hybrid qualifying, E6 confirming H1 and H2 for the hybrid, E7. | Partly. Condition 1 has to be read as "the LLM is part of the decision mechanism" rather than "the LLM is the action policy". The lead must accept this. | UAI or NeurIPS 2027, or TMLR. |
| **F3. Diagnostic and negative** | (a) Known-graph linear-Gaussian design leaves no headroom: sequential equals plan-once, and no policy can beat target risk in matched regimes. (b) First-crossing cost endpoints cannot resolve policies at feasible n, and decision-level endpoints can. (c) The collapse has named causes, confirmed experimentally: chance-level labels from the replicate tie rule, hard labels, trie-greedy decoding, position and identity priors, a single unshuffled pass, and the same collapse in the numerical student. (d) Whether a repaired recipe restores state dependence, and whether the LLM ever beats the planner where text carries value (if E5 or E6 ran). | E1, E2, E3 (and E4 if triggered), saved November data, theory items T1 to T5, the full record of earlier attempts. | No. The lead must decide whether this outcome is acceptable (ask 1). | TMLR is the best fit, since its criterion is that claims are supported by evidence. ICML 2027 is possible if E1 to E3 finish by about January 10. Careful negative studies have appeared at ICML 2024 and in Findings of EMNLP 2025 (literature-policy 4.3). |
| **D. Delivery study (separate, independent of the LLM outcome)** | The value of a post-collection refit depends on task, comparator and collection history, and the five-node reversal is diagnosed. | E9 plus the panel's cheap fixes. | Not an LLM paper. | TMLR, the recommendation on record (manuscript note section 8). CLeaR (deadline November 16) is too soon. |

Portfolio choice (ask 1). Option (i) is one paper at NeurIPS 2027, with the framing chosen at Gate C. Option (ii) is F3 at ICML 2027 in January, followed by a separate F1 or F2 paper that cites F3. Both options include D at TMLR. Under option (ii) the later paper's contribution has to be clearly distinct: a new class and the positive claim, with the anatomy cited rather than repeated. **[A]** Standard dual-submission rules allow that sequence, but the lead should check the policy text before relying on it.

---

## 2. Claims-to-evidence map

| # | Claim | Existing evidence | New evidence | Robust to either outcome? | Used by |
|---|---|---|---|---|---|
| C1 | In the canonical class, sequential adaptation is worth about nothing, plan-once equals target risk, and no policy can beat target risk in matched regimes. | Model-internal analysis of the 24 saved post-initialization states (numerics-protocol 4.2). | E1 K0 on 60 fresh systems. Theory T1 and T2. | Yes | F1, F2, F3 |
| C2 | First-crossing cost cannot resolve policies at feasible n, and decision-level endpoints can. | Saved-data power analysis: 67 to 340 versus 3 to 17 systems (numerics-protocol 4.3, results-analysis-verify D). | E1 endpoint study, prospective on fresh systems. Theory T3. | Yes | All |
| C3 | The November collapse has identified mechanisms. | Training and deployment forensics. These are correlational: the token-bias model, the 1.93 versus 2.37 nats counter comparison, label noise of 69 to 73%. | E2 (decoding, position and posterior reading). E3 knockouts A3 to A7 and N2 (causal tests). Theory T4 and T5. | Yes | F3, and the motivation section of F1 and F2 |
| C4 | A repaired recipe lets a 0.6B LLM imitate state-dependent SCM planning. | None. Both trained adapters were worse than a state-blind counter. | E3, and E4 if triggered. | Informative either way | All |
| C5 | An explicit SCM posterior in the input matters. | None. | E3 A1 versus A2. E5 and E6 no-SCM ablations. | Either way | All |
| C6 | The LLM uses measurements. | None. The literature predicts that untrained models do not. | E5 and E6 shuffled-feedback controls. | Central | F1 |
| C7 | Where the information exists only in text, the LLM lowers cost or error relative to U, R and an equal-information numerical student. | None. | E1 admission, E5, E6. | Central | F1, F2 |
| C8 | The result transfers beyond the synthetic generator. | None. | E7, plus E8 as an illustration. | Central | F1, F2 |
| C9 | The delivery result is bounded, and its five-node reversal has a stated cause. | The completed registered study. | E9. | Yes | D |

C1 to C3 and C9 survive any outcome. That is why E1, E2 and E3 come first and why E9 runs alongside them.

---

## 3. Portfolio logic

### 3.1 Spine and branches

- **Spine (must):** E1, E2, E3. Each is cheap relative to what it settles, and none touches the 24 development outcomes.
- **Escalation (could, conditional):** E4. It runs only if E3 shows the LLM failing while the numerical student passes.
- **Positive branch (should, gated):** E5, then E6, then E7, with E8 optional.
- **Parallel and independent (could):** E9.

### 3.2 Kill criteria (all predeclared, all independent of how the LLM performed)

| Kill | Trigger | Consequence |
|---|---|---|
| K1 | E1 admits no class. | Every LLM-benefit branch closes. The paper is F3, with the claim that in the SCM classes the code can express, the text channel carries too little decision value or a parser can extract it. |
| K2 | After E3, and E4 if run, no LLM arm passes the imitation gates. | The direct LLM policy closes. The hybrid (F2) remains only if E1 admitted a class. Otherwise the paper is F3. |
| K3 | In E5, no LLM variant qualifies. | Stop. Report the development results in F3. No further class or benchmark is tried. |
| K4 | E6 fails H1. | Report the null. No benchmark is added afterwards to rescue it. |
| K5 | E7 fails. | The claim is restricted to synthetic classes. No search for a replacement simulator. |
| K6 | The E2 fidelity check falls below 99%. | Local LLM results do not stand in for the frozen runtime. |

### 3.3 What to cut

- Any further LLM-benefit screen in the canonical class. It has a ceiling at the level of a theorem.
- More DPO or preference training on labels built from replicate measurements and a tie rule.
- A 100 to 150 system confirmation of the November recipe.
- The old ACE GPU pipelines and the PEV ladder.
- BoxingGym, NeuronBench, CausalMan, Mission-1 root designs, the compressor (ThermoPower or pyCycle), the partial-identification canary and the action-language parser lane.
- Development iterations through the full CURC custody harness.
- A GO-CBED comparator, deferred unless a reviewer asks for it. It would need a download approval, and in small known-graph systems it is expected to be no better than U **[A]**.

---

## 4. Experiments (in execution order)

Common definitions:
- **U** is the strongest numerical planner without the text, chosen in E1: either 256-draw target risk with common random numbers (CRN) across candidate actions, or the rollout planner if that is more than 5% better.
- **O** is the same planner given the information that only the text carries (the true objective or the true prior). It is the oracle and the reference for equal information.
- **R** is a hand-written rule parser feeding U.
- **Primary endpoint:** frozen in E1, from the candidates listed there.
- **Replicates:** R = 4 replicate episodes per system and arm, with common random numbers keyed by (system, replicate, action, occurrence). This means two arms that take the same action receive the same response. In November they did not: SFT and DPO took identical actions in their first 10 to 22 acquisitions, yet their costs differed in 11 of 24 worlds.

### E1. Headroom certification and endpoint freeze (numerical only) [must]

**Question.** Which SCM classes the code can express contain decision value that only the public text carries, enough to detect at a feasible n? And which endpoint resolves policies? Both are decided before any LLM touches the new classes.

**Design.**
- **Classes.** Each is a new protocol module over the reusable generic layer, which allows at most 8 nodes, 32 actions and 64 queries (numerics-protocol section 6).
  - **K0, control.** The completed 3-variable generator, redrawn from a fresh declared development namespace. Not the 24 systems.
  - **K1, prose objective.**
    - Graphs: known-graph linear-Gaussian SCMs with 4 to 6 nodes.
    - Actions: single and pair clamps at 3 levels, plus observe and partial-measurement actions. Costs are heterogeneous, from 1 to 5.
    - Initial observations: 2.
    - Queries: the candidate set is every clamp, each asking for a designated outcome node. The true objective is a subset of 4 to 8 operating conditions with equal weights, since the generic layer cannot express unequal weights. The subset is described only in prose. U uses the uniform default query law.
  - **K2, prose mechanism knowledge.**
    - Graphs: the same.
    - Prose: describes edge signs and rough magnitudes.
    - Coefficients: drawn from the described ranges.
    - Priors: U uses the learner's prior N(0, I), and O uses the informed prior.
    - Admission is judged with a common, uninformed evaluation learner, so that it measures what the choice of actions contributes. Evaluation with each arm's own learner is recorded as a secondary, learner-side quantity and cannot by itself admit a class for policy claims.
- **Arms (numerical only).**
  - Uniform random and menu coverage.
  - A leave-one-out public static schedule.
  - Plan-once from the post-initialization state.
  - Target risk at 32 draws, as deployed in November.
  - U256: target risk at 256 draws with CRN across candidate actions.
  - A rollout reference: the top 4 candidates, each with 16 rollouts of a base policy, run on 20 systems with one replicate.
  - O.
  - R: written from paraphrase families A and B only, and scored on held-out family C (generated by a different template process) and family D (human-written, if the lead names an author).
- **Systems.** 60 per class from fresh declared development namespaces, with R = 4 and CRN. The threshold τ is calibrated on systems from the training namespace so that at most 10% of systems start at target.
- **Endpoint candidates.**
  - P1: fixed-budget, truth-conditional expected log NMSE. Each realized action sequence is replayed with 256 fresh noise draws under the true SCM through a common evaluation learner.
  - P2: expected cost to target, read from the truth-conditional curve averaged over replicates.
  - P3: paid cost under a common public stop rule (stop when posterior risk ≤ τ), plus reliability, defined as the share of stops at which the true NMSE is ≤ τ. All arms share the same stop rule.
  - P4: realized first-crossing cost. Kept for continuity only and never primary.

**Sample size basis.** The paired SD of log final posterior risk on the 24 saved systems is 0.19 to 0.28 natural-log units for contrasts between designs that differ (Section 11.1). With n = 60, the SE is 0.025 to 0.036. A 15% effect then has z between 4.5 and 6.6, and the SD that E5 and E6 use for power is known to about ±9%.

**Decision rule** (all LLM-independent; frozen and hashed before any LLM sees K1 or K2).
- **Endpoint freeze.** P2 is primary if two conditions hold: U256 beats random with z ≥ 3 at n = 60, and P2's z for O versus U256 is at least 70% of P1's. Otherwise P1 is primary and P2 secondary. P4 is never primary.
- **Numerical control.** U is the rollout planner if it beats U256 by more than 5% on the primary endpoint. Otherwise U is U256.
- **Admission of a class.** All four conditions must hold:
  - (a) ln(U/O) ≥ 0.1625 (O at least 15% better), with the 95% CI lower bound ≥ 0.05.
  - (b) ln(R/O) ≥ 0.105 on held-out families C and D: at least a 10% error reduction is left that the parser does not capture. Its CI must exclude 0.
  - (c) U beats random with z ≥ 3.
  - (d) At most 10% of systems are at target after initialization.
- **Positive** (at least one class admitted). Licenses LLM work in the admitted class only, and the paper statement "text carries Δ of decision value in class K, of which a parser recovers ρ".
- **Null** (no class admitted). Triggers K1 (Section 3.2).
- **Negative** (O not better than U, or U not better than random). Indicates a bug in the evaluator or the class. Audit before anything else. Nothing about LLMs is licensed.
- **K0, in every case.** Replicates or refutes C1 on fresh systems.

**Prerequisites.**
- Approvals: the lead's go-ahead for local, numerical-only simulation on new declared namespaces (ask 3), and a written protocol.
- Engineering:
  - New protocol, binding, assembler and realization modules for K1 and K2, roughly 100 to 300 lines each (numerics-protocol section 6).
  - Support for informed priors in the learner, with new sealer and cost-curve versions (small to medium).
  - A CRN keying change. The paired-episode module currently forbids shared measurement namespaces at lines 114 to 117.
  - A truth-conditional evaluator.
  - A rollout planner.
  - A raised roster limit, since the current limit is 8 arms.
  - Prose families A to C and the rule parser.
- People: a human author for family D (ask 9).
- No GPU and no CURC.

**Cost.**
- Engineering: about 1.5 to 2 weeks **[A]**.
- Compute: about 155,000 planner decisions (3 classes × 60 systems × about 9 arms × 4 replicates × about 24 decisions), at most about 1 process CPU-second each. That is the measured figure including interpreter start. The 256-draw planners cost more. Add the rollout subset at about 13 CPU-hours per class. The total is about 60 to 100 CPU-hours, or 5 to 9 hours of wall time on the Mac, assuming at least 12 usable cores **[A]**.

**Main risk.** The prose turns out to be solvable by the parser, so condition (b) fails. Or K2's value lies in prediction rather than choice. Either way every LLM branch closes. That would be the honest answer, but it leaves only F3.

### E2. Offline forensic scoring of the saved adapters [must]

**Question.** Which mechanisms produced the constant actions: token-greedy pooling, a position or identity prior, or the weights? And does any deployed adapter encode posterior information that its decoder failed to express?

**Design.**
- **Inputs, all already on the Mac.** The Qwen3-0.6B base (local cache). The SFT and final adapters, whose hashes match the deployed pins (583b1659…, 6f540dfe…). All 1,858 exact prompt token sequences: 1,698 from deployment and 160 from training.
- **Step 0, fidelity check.** Local constrained greedy decoding must reproduce the recorded command on at least 99% of the 1,698 deployment prompts and of the 322 saved training generations.
- **Census.** For every prompt, under each of the base, SFT and final weights, compute the full-sequence log-probability of all 9 legal commands. From these derive three choices: the token-greedy choice, the whole-command argmax, and the argmax after per-token normalization. Save the full score tables, which no LLM arm has today.
- **Probes on 240 separately declared artificial states** (10 decision indices × the 24 saved worlds). The adapters stay frozen and no outcome is used. Each state is altered in these ways:
  - menu order reversed, and menu order randomly permuted;
  - identifiers renamed to neutral codes of equal length;
  - a content-free state, with numbers replaced by placeholders;
  - posterior perturbations: one head's variance multiplied by 4 and by 0.25, and the sign of the X-mean estimate flipped, which flips the planner's preference between the far and near X setting.
- **Linear probes.** From the final-layer hidden state at the decision position, predict the saved 16,384-draw target-risk scores for each action and the sign of the X mean. Use ridge regression with world-grouped 6-fold CV. Controls are the same probe on the base model and a probe on prompt-surface features (decision index, action counts).

**Endpoints.**
- The share of deployed SFT and DPO choices that change under whole-command argmax.
- The share of permuted states whose modal choice follows list position.
- The total-variation (TV) change in the 9-way command distribution under each posterior perturbation.
- Probe R² under grouped cross-validation.

**Sample size basis.** The deployment and training prompts are a census. 240 probe states bound any rate to within ±6.3% (95% half-width at p = 0.5).

**Decision rule** (thresholds predeclared).
- **Thresholds.**
  - Token pooling is "material" if at least 20% of SFT deployment choices change under whole-command argmax.
  - A position prior is "dominant" if the modal choice follows list position in at least 70% of permuted states.
  - The adapter "reads the posterior" if either the TV change is at least 0.1 in the planner's direction in at least 25% of perturbed states, or probe R² for the X-mean sign is at least 0.3 beyond the surface-feature probe.
- **Positive** (fidelity at least 99% and at least one mechanism above its threshold). Licenses an anatomy section with effect sizes. E3's knockouts then become confirmatory tests of the named mechanisms. If the adapter reads the posterior but does not act on it, that is a knowing-doing gap, and the readout or decoding should be fixed before the supervision.
- **Null** (fidelity holds and nothing crosses a threshold). The collapse is in the weights, so supervision (E3) is the lever.
- **Negative** (fidelity below 99%). The local runtime is unfaithful. Report the agreement rate only. E3 and later stages then need a pinned runtime with its own fidelity check (K6).

**Prerequisites.**
- The lead's approval for local forward-only inference, which counts as a new model execution (ask 2), and a written protocol.
- No installs: torch 2.8 and transformers 4.57 are present, and the adapters are torch-only `.pt` files.

**Cost.**
- About 10,600 prompt forwards (1,858 × 3 plus 240 × 7 × 3), each with 9 short continuations.
- About 1 to 4 hours of local GPU, assuming 0.3 to 1.5 s per forward on a roughly 3,000-token prompt **[A]**. For comparison, the CURC slice measured 0.43 to 0.71 s per greedy decision.
- 2 to 4 calendar days. No CURC.

**Main risk.** Version drift between the local and frozen runtimes makes the fidelity check fail.

### E3. Imitation ladder with knockouts and matched numerical students [must]

**Question.** Can a Qwen3-0.6B policy trained with a repaired recipe read SCM state well enough to imitate target-risk planning on held-out systems? Which repairs are necessary? And does a numerical student trained on identical data do as well?

**Design.**
- **Class.** The completed 3-variable generator, drawn from a fresh declared training namespace outside every consumed range. No development system or outcome is used. The planner's choices in this class depend on state: first actions vary across systems with large margins, and the action mix varies within cells. So imitation is not trivial. The best state-blind first action has a relative regret of 0.257 at the first decision. Over whole episodes, the fixed coverage cycle scores 0.441 and the constant `set_x_neg` policy scores 0.699.
- **Data.**
  - 400 training worlds giving about 9,600 states, generated by a mix of behavior policies (random, coverage, local information, target risk).
  - Repetition-stress histories with 10 to 23 repeated actions. These cover the off-distribution states that DPO reached.
  - The 697 saved training envelopes, reused.
  - Labels: 4,096-draw target-risk scores for all 8 actions, which cost no environment responses.
- **Held out.** 80 more training-namespace worlds, evaluated on each student's own closed-loop rollouts.
- **Arms.**
  - **A1, full repair:**
    - targets: graded listwise KL to a softmax of planner scores, at a temperature fixed in the protocol;
    - state text: a rounded, decision-relevant SCM summary of about 400 to 700 tokens, with no scores per action;
    - identifiers: neutral, with identifiers and menu order permuted per example;
    - scoring: whole-command normalized scoring;
    - training: 2 shuffled epochs, rank-16 LoRA on the attention and MLP projections.
  - **A2:** A1 without the posterior summary, so the state is history only.
  - **A3:** A1 with hard argmax labels.
  - **A4:** A1 with labels made by the November rule (4 replicates, a 2-SE tie rule) on the same states, using simulated responses.
  - **A5:** A1 with a fixed menu order and the original identifiers.
  - **A6:** A1 with the raw November JSON envelope, which carries 17-digit floats.
  - **A7:** A1's weights decoded token by token greedily. Needs no training.
  - **N1:** an MLP on the same numeric summary, trained on A1's targets.
  - **N2:** N1 trained on A4's labels.
  - **References:**
    - a state-blind frequency counter;
    - target risk, local information, coverage and random;
    - the November ranker, SFT and DPO adapters, scored through E2's tooling.
- **Seeds.** 3 for A1, A2, N1 and N2. 2 for A3, A4 and A5. 1 for A6, on half the data.

**Endpoints.**
- **Primary:** mean relative regret on the held-out rollouts under the 16,384-draw target-risk score. A myopic criterion is appropriate here because the claim is imitation.
- **Secondary:** held-out cross-entropy relative to the counter, action entropy, TV distance under state shuffling, and the model-expected final NMSE J of each student's own rollouts relative to target risk.

**Sample size basis.** In November, DPO's mean regret was worse than random's in 20 of 24 worlds (Wilcoxon p ≈ 0.0005) at a gap of about 0.2. That implies a per-world SD of about 0.28. With 80 worlds the SE is about 0.031, so a regret difference of 0.10 gives z ≈ 3.2. Using 3 seeds answers the single-seed limitation that Kriz et al. flag.

**Decision rule.**
- **Gate.** An arm "learns state-dependent SCM planning" only if, in every seed, all four hold:
  - regret ≤ 0.154, the level local information reached in the screen;
  - cross-entropy at least 0.2 nats per example better than the counter;
  - state-shuffle TV ≥ 0.2, with world-cluster permutation p < 0.01;
  - J within 10% of target risk.
- **A1 passes and N1 passes.** The recipe is repaired and branch P may use it. The paper says that language pretraining is not needed for imitation, so any LLM benefit must come from text.
- **A1 passes and N1 fails.** An unexpected advantage from the language representation. Replicate it before claiming it.
- **A1 fails and N1 passes.** The 0.6B LLM is the bottleneck under offline distillation, so run E4. If E4 also fails, the direct LLM policy closes (K2).
- **Both fail.** The supervision or the state design is insufficient, and the paper is F3.
- **Knockouts.**
  - A component is necessary if removing it raises regret by at least 0.10 in every seed.
  - A4 and N2 reproduce the collapse if one action exceeds 80% of choices or the action entropy falls below 0.8 bits. If both collapse, the collapse comes from the supervision, not from language (C3).
  - A1 versus A2 is the paper's ablation of the SCM state (C5).

**Prerequisites.**
- Approval for local training under a pinned local runtime manifest, or a new CURC ceiling of about 7 to 11 GPU-slice-hours (training plus rollouts) and about 10 core-hours (ask 4).
- Code, about 1.5 to 2 weeks **[A]**:
  - a listwise loss (the training job already scores all legal commands);
  - a summary serializer;
  - permuted identifiers in the grammar and task modules;
  - multi-projection LoRA at rank 16 in the torch-only engine;
  - the numerical student;
  - the data generator.
- E2 informs this experiment but does not block it.

**Cost.**
- **Training.** 13 LoRA runs. A summary-prompt run is about 4,000 states × 2 epochs × 600 tokens × 1.5 for the listwise overhead, or about 7.2 M tokens. At the measured throughput of about 5,000 tokens per second on the slice, that is about 0.4 to 0.55 hours per run. The raw-envelope arm takes about 1.3 hours. Total: about 6 to 9 slice-hours. Only 0.76 slice-hours remain in the current envelope, and the `gpu-testing` QoS has a 1-hour wall per job.
- **Relabeling:** 5 to 10 CPU-hours.
- **Rollouts:** about 27,000 decisions, or 1 to 2 GPU-hours.
- **Calendar:** about 3 weeks.

**Main risk.** The summary text might quietly encode the planner's answer. This is mitigated by never including scores per action. Alternatively, a 0.6B model may fail whatever the recipe.

### E4. Escalation: model capacity versus on-policy training [could, conditional]

Runs only if E3 ends with A1 failing and N1 passing.

**Question.** Is the 0.6B LLM's failure due to model capacity or to offline supervision?

**Design.** The same data, held-out set and gates as E3.
- **(a) Capacity.** Qwen2.5-7B-Instruct, already in the local cache, with the A1 recipe, 2 seeds.
- **(b) On-policy training.** Qwen3-0.6B trained with on-policy GRPO, using the settings Kriz et al. report for the same base model: 16 states × 16 rollouts per update, 250 updates, temperature 2, an entropy bonus of 0.02, and permuted identifiers. The reward is the sampled action's 256-draw one-step target-risk gain per unit cost. 2 seeds.

**Decision rule.** An arm passes on E3's gates.
- **7B passes.** Capacity was the bottleneck. Branch P continues with 7B, and its higher inference cost goes into the compute ablation.
- **GRPO passes.** Offline distillation was the bottleneck. Branch P continues with GRPO.
- **Both fail.** K2: the direct policy closes. F2 remains only if E1 admitted a class. Otherwise the paper is F3.
- The paper reports the escalation in every case.

**Prerequisites.**
- Approval to use the cached 7B weights. No download is needed (ask 5).
- GRPO in the torch-only engine, about 1 to 2 weeks **[A]**.
- Compute approval.

**Cost.**
- GRPO: about 64,000 single-decision generations at 0.1 to 0.5 s, which is 2 to 9 GPU-hours plus backward passes **[A]**.
- 7B LoRA: 2 runs, each about 5 to 10 times as long as a 0.6B run, or 1 to 3 nights locally. The bf16 7B weights (about 15 GB) fit in 36 GB **[A]**.
- Calendar: 2 to 3 weeks.

**Main risk.** Memory and time limits on the Mac for the 7B model. A reward that leaks the planner's choice.

### E5. Development pilot in the admitted class [should, branch P]

Runs only if E1 admits a class and E3 or E4 produces a passing recipe.

**Question.** In the admitted class, does an SCM-conditioned LLM that reads the prose choose interventions with lower truth-conditional error than U? And does it beat the controls that have the same information?

**Design.**
- **Training.** 400 worlds from a fresh training namespace in the admitted class. Privileged-teacher distillation: the teacher is O, and the students see only the public prose and the SCM state. Prose comes from families A and B.
- **Systems.** 48 fresh development-namespace systems, with R = 4 and CRN. Prose comes from held-out family C, and from D if available.
- **Arms.**
  - U, O and R.
  - **L-dir:** the direct LLM policy, using the E3 recipe plus the prose.
  - **L-hyb:** the LLM produces the objective or prior as structured output, and U plans with it.
  - **L-adj:** the prompt adds U's scores for each action, and the LLM chooses.
  - **N-emb:** an MLP on the SCM summary plus a frozen LLM embedding of the prose, distilled from O. This is the numerical student with the same information.
  - **N-bow:** the same with bag-of-words text features and no pretrained model.
  - **F:** the frozen Qwen3-0.6B, zero-shot, with the same prompt.
  - Random and coverage.
- **Ablations, applied to every LLM variant on a prespecified 24 of the 48 systems with 2 replicates:**
  - text removed (anonymized);
  - misleading text (a wrong objective, or flipped signs);
  - shuffled feedback (responses permuted in the policy's input only);
  - no SCM state;
  - for L-adj, an argmax-of-prompt control. Literature-policy 7.2 says it must be reported, because otherwise the LLM may only be computing the argmax of numbers it was given.

**Endpoints.** The E1 primary. Secondary: oracle-gap closure, (U − L)/(U − O).

**Sample size basis.** With 48 systems the SD that E6 needs is estimated to within about ±10%. Gap closure is estimated with an SE of about 0.12, assuming a gap of 0.3 log units and a paired SD of 0.25, which is enough to apply a 30% qualification threshold.

**Decision rule** (development selection under a predeclared rule).
- **Qualification.** A variant qualifies only if it does all of the following:
  - closes at least 30% of the U−O gap;
  - loses at least 50% of its gain when the text is removed;
  - beats R by at least 10% of the gap on the held-out families;
  - degrades by at least 10% under shuffled feedback (L-dir and L-adj only);
  - for L-adj, deviates from argmax-of-prompt on at least 10% of decisions, with deviations beneficial on average.
- **Selection.** Take the qualifying variant with the highest gap closure. Freeze its weights, prompt, scoring and stop rule.
- **Positive.** Proceed to E6. If only L-hyb qualifies, the framing is F2. If only N-emb qualifies, the contribution is "LLM embeddings as features", a narrow F2.
- **Null.** K3: stop.
- **Negative** (the LLM variants are worse than U). Report the harm from the language input. The paper is F3.

**Prerequisites.**
- E1 admission, and a pass from E3 or E4.
- The prose generator, parser and oracle teacher from E1.
- Training approval or a compute ceiling.

**Cost.**
- 6 LoRA runs, about 3 to 4 slice-hours.
- About 23,000 LLM decisions at 0.1 to 0.6 s, or 1 to 4 GPU-hours. That covers L-dir, L-adj and F on all 48 systems × 4 replicates × 24 decisions, plus 8 ablation conditions for L-dir and L-adj on 24 systems × 2 replicates. L-hyb calls the LLM once per system, so its planning cost is CPU.
- Numerical arms: 10 to 20 CPU-hours.
- About 2 weeks of engineering. About 3 weeks of calendar.

**Main risk.** Choosing among several variants overfits the 48 systems. Two things limit the damage: only a single variant is selected, and the confirmation in E6 uses fresh systems.

### E6. Preregistered confirmation on fresh systems [should, branch P]

**Question.** On systems never scored before, does the frozen LLM variant L* reduce truth-conditional cost or error relative to U and R?

**Design.**
- **Registration, before any system is drawn.** It fixes:
  - the method hashes, prompts, scoring and stop rule;
  - namespaces from the reserved confirmation range [202611071000, 202611072000);
  - the endpoints and the Holm family;
  - failure treatment: failed cells are kept at the cap and no system is replaced;
  - the analysis code hash;
  - an independent review.
- **Arms.**
  - U, O, R, L*, N-emb and N-bow.
  - F, the frozen LLM.
  - A supervised LLM control (L-dir, if L* is not L-dir).
  - Random and coverage.
  - Together these keep the controls required by the standing rules: target risk, a numerical ranker or student, the frozen LLM and the supervised LLM.
- **Ablations on a prespecified random half of the systems.** L* with the text removed, with misleading text, with shuffled feedback, and without SCM state.
- **Systems.** n is the larger of the requirements for H1 and H2. H2 is powered at an effect of half the residual headroom ln(R/O) that E1 measured. With an SD of 0.19 to 0.28 and a minimum effect of 10%, at 90% power with Holm over two primaries, n = 40 to 88. **Plan 80** (90% power if SD ≤ 0.27, 80% if SD ≤ 0.31). The cap is 120 if E5's SD comes out larger. R = 4, with CRN.

**Endpoints.** The E1 primary. Secondary: the E1 secondaries, gap closure, reliability at stop, and compute per decision.

**Decision rule.**
- **H1:** the L*/U ratio ≤ 0.90, with Holm-adjusted p < 0.05.
- **H2:** the L*/R ratio < 1, with Holm-adjusted p < 0.05.
- **Positive (H1 and H2).** F1 if L* is the direct or adjudicator policy, F2 if it is the hybrid. The ablations attribute the effect:
  - text removed → the gain uses the language channel;
  - shuffled feedback → the policy uses measurements;
  - N-emb and N-bow → language pretraining;
  - F versus L* → ACE-specific training;
  - no SCM → causal structure;
  - the compute table → extra computation.
- **H1 only.** Access to the text helps, but a parser delivers it. The LLM's role is not established, and the result is reported in F3 or as a narrowed F2.
- **Null on H1.** A confirmed null after a positive development pilot. The paper is F3, with the pilot reported.
- **Negative** (L* worse than U, with the CI excluding 1). Report the harm. The paper is F3.
- No class or benchmark changes after this point.

**Prerequisites.**
- E5 frozen.
- The lead's decision on the runtime (ask 4). Either a pinned local runtime with light custody, or CURC with redesigned custody: sharded archives, numerical policies run in-process, and no repeated verification of the whole evidence tree.
- Raised roster limits.
- A compute ceiling.

**Cost.**
- About 38,000 LLM decisions, or 1 to 7 GPU-hours.
- About 54,000 numerical decisions, or 15 to 40 CPU-hours (more if U is the rollout planner).
- In the old harness the ledger projects 22 to 35 core-hours, 1.9 to 2.9 slice-hours and 6 to 10 days for 100 to 150 systems. Locally, about 3 to 5 days **[A]**.

**Main risk.** Effects shrink from development to fresh systems, or the SD is larger than planned.

### E7. External transfer on the Causal Chambers wind-tunnel simulator [should, branch P]

**Question.** Does the frozen method's benefit carry over to a physical simulator written by independent authors, with documented misspecification and variable descriptions written by others?

**Design.**
- **Simulator.** An adapter over the `causalchamber` 0.2.8 wind-tunnel simulators. `wt.SimA1C3` gives downwind pressure from `load_in`, `load_out`, `hatch` and ambient pressure. `wt.ModelB1` gives current.
- **Actions.** Actuators are discretized into at most 32 menu actions. Each action costs its settling time.
- **Graph and text.** The graph comes from the documentation. The prose comes from the official variable descriptions, plus experimentalist objectives written in held-out wording.
- **Learner.** The linear SCM learner, which is misspecified by construction: the true model has affinity laws, a cubic current and a pressure that scales with speed squared.
- **Instances.** 40 simulator instances, with physical parameters drawn from documented ranges. R = 4.
- **Arms.** U, O, R, L* (frozen, not retrained) and N-emb.
- **Scope of the claim.** One family of simulated apparatus.

**Endpoint.** The E1 primary, computed from the simulator's truth (noise-free outputs or many replicates).

**Sample size basis.** At SD 0.25, 40 instances give about 98% power for a 15% effect and about 76% for a 10% effect (two-sided, 0.05).

**Decision rule.**
- **Holds:** the L*/U ratio ≤ 0.90, with the 95% CI below 1 and a sign that matches E6.
- **Partial:** a point estimate ≤ 0.95 with a CI that includes 1.
- **Fails:** otherwise. The claim is then limited to synthetic classes (K5).

**Prerequisites.**
- Approval to install `causalchamber` 0.2.8, or to vendor its MIT-licensed, numpy-only wind-tunnel file (ask 6). Version 1.0.0rc1 needs Python 3.11, and the local Python is 3.9.
- The frozen method. The adapter can be built while E6 runs.

**Cost.** About 1 week of integration (literature-benchmarks estimate). About 13,000 decisions: at most 5 CPU-hours and 1 to 2 GPU-hours. 1.5 to 2 weeks of calendar.

**Main risk.** Instances that differ only in parameters are correlated, not independent systems. Under misspecification, stopping on posterior risk is miscalibrated. The reliability secondary detects this.

### E8. Remote Lab hardware demonstration [could]

**Question.** Does the frozen policy's ordering relative to U survive on the real Wind Tunnel Mk2?

**Design.**
- 4 policies (U, O, L* and random) × 30 episodes × 16 interventions.
- Randomized, blocked order, to handle drift and ambient pressure.
- The private evaluator is a separately measured validation grid with replicates, and its measurements are charged.
- Frozen policies only.

**Endpoint.** Error on the validation grid at a fixed budget, per episode.

**Sample size basis.** Set by the budget: 30 episodes per policy (the brief's arithmetic).

**Decision rule.** Descriptive only, because there is one apparatus. Report the ratio with an episode-bootstrap CI as an illustration. A reversal is reported as a limitation. This result is never used to rescue a synthetic null.

**Prerequisites.**
- E7 holds.
- A human signs up and accepts the terms.
- Spend approval (ask 7). The 1,000 free credits cover only about a 17-minute smoke test.
- Network access from the runtime.

**Cost.** About 5.3 hours of chamber time and about 270 EUR at the academic rate. 2 to 4 weeks of calendar, including sign-up.

**Main risk.** Drift, latency in the shared pool, and a single apparatus.

### E9. Closing the delivery study from saved checkpoints [could, parallel]

**Question.** Is the undiagnosed five-node reversal of delivery against rolling (2.46×) an error-propagation effect or an optimization artifact?

**Design.** Analyses on the 640 saved Stage B fits, with no new fits:
- teacher-forced versus free-running error for each head;
- a residual and shift decomposition;
- the sensitivity of the X3 head in the direction normal to the X2 = aX1 + b support, which is the situation Prop 8.3 describes;
- each of the above per initialization.

**Endpoint.** The share of the five-node log excess of delivery over rolling that comes from free-running propagation through the learned chain.

**Decision rule.**
- At least 70% of the excess comes from propagation: the reversal is explained, and the paper says so.
- Delivery's teacher-forced error is already worse: the cause is optimization or convergence, and the claims are narrowed.
- Neither: the reversal remains undiagnosed, but bounded.

Every outcome makes the delivery manuscript ready to submit to TMLR, either on its own or as a supporting section.

**Prerequisites.**
- The lead's permission for checkpoint-only diagnostics, which the manuscript notes leave as an open question, and a decision to unbundle the paper (ask 10).
- The checkpoints must be located. A bounded read-only search today of `ACE/results/delivery_prospective_*` and `/Users/pat/ACE_Study_Results` found only 8 pilot `.pt` files. The 640 fits are probably in the CURC `/projects` archive, so retrieval may need a CURC session with Duo.

**Cost.** Local CPU, 1 to 3 days of analysis, plus the retrieval.

**Main risk.** The checkpoints cannot be retrieved without CURC, or the diagnosis is inconclusive.

---

## 5. Sequence, gates and calendar

| When [A] | Work | Gate |
|---|---|---|
| Nov 12 to 16 | The lead's decisions (asks 1 to 4, 8, 11, 12). Correct the README. Write the protocols for E1 and E2. Declare all new namespaces. | n/a |
| Nov 17 to Dec 5 | E2 (2 to 4 days). E1 engineering and runs. E3 engineering. E9 retrieval and analysis. | **Gate A, about Dec 5:** the E1 admission decision, numerical control and endpoint are frozen and hashed before any LLM touches K1 or K2. The E2 report is done. |
| Dec 8 to 23 | E3 runs and gates. E5 engineering for the admitted class: prose families, parser, oracle teacher and the hybrid interface. | **Gate B, about Dec 23:** branch decision. Proceed to E5, run E4, or write F3. |
| Jan 4 to 22 (reduced by holidays) | E5, or E4. Draft F3 in parallel. | **ICML decision about Jan 8:** submit F3 only if the lead chose the portfolio route. |
| Jan 25 to Feb 28 | **Gate C:** E6 registration and independent review. Then E6. E7 engineering in parallel. | E6 outcome. The framing is fixed. |
| March | E7. E8 optional, March to April. | K5 |
| April to May | Writing, independent review and the coauthor's approval. NeurIPS 2027 (estimated) or TMLR. | n/a |

What can run in parallel:
- E1, E2, E3 engineering and E9.
- E7 engineering alongside E6.
- Writing F3 alongside E5.

Calendar risks: about 39 active hours per screen cycle. Human-gated waits, which cost 23.5 hours in November. Thanksgiving and the winter break. Moving development to the Mac removes most of the overhead from launchers, collectors and custody (34% of active hours in the scientific phase) and from guard failures (11 of 22 failed submissions).

---

## 6. Theory items

1. **T1, a ceiling on the value of adaptivity in conjugate linear-Gaussian design.**
   - Statement: with known graph, known noise and clamp interventions, each head's posterior covariance depends only on that head's design rows. The value of sequential adaptation is bounded by how much the mean-dependent composed terms vary under the posterior predictive. Those terms are μ_φᵀVμ_φ + mᵀC_φm.
   - Corollary: when every query probes clamped parents, plan-once is Bayes-optimal (Chaloner and Verdinelli 1995).
   - Why it helps: it turns Finding A from a measurement into a guideline for which classes can reward a learned policy. It also makes the K0 result general, which F1, F2 and F3 all need.
2. **T2, an equal-information impossibility and a decomposition of excess loss.**
   - Statement: under a correct model with the same information and objective, the Bayes sequential rule minimizes expected loss. Any policy's excess over the planner splits into four terms: information the planner lacks, the planner's approximation error (myopia, Monte Carlo error, the ratio heuristic), misspecification, and compute. This links to the agenda's κ + 2ε + δ.
   - Why it helps: it makes E1's admission rule principled rather than convenient, so a change of benchmark does not look like a favorable replacement. It also states in advance what an LLM win can mean.
3. **T3, efficiency of endpoints.**
   - Statement: consider first-passage cost on an error process with a coefficient of variation near 1 and no common random numbers. The variance of a paired difference has a floor set by within-system noise. Truth-conditional expected error with CRN removes that floor. The paper derives n under each endpoint, which reproduces the contrast of 67 to 340 versus 3 to 17 systems.
   - Why it helps: it is a reusable methodological contribution and justifies the primary endpoint.
4. **T4, noise in labels from a replicate tie rule.**
   - Statement: with r replicates and a k-SE rule, the probability of a false strict label is P(|t_{r−1}| > k). That is 0.139 at r = 4 and k = 2. Combined with the measured distribution of gains, this predicts that 69 to 73% of strict labels are at chance. Through the robust-DPO factor 1/(1−2ε), it predicts the failure of preference learning.
   - Why it helps: it explains the failure of DPO, and it gives a design rule for how many replicates are needed to reach a target label-noise rate.
5. **T5, bias of trie-greedy decoding.**
   - Statement: greedy prefix-constrained decoding picks, at the first point where commands diverge, the branch with the largest pooled probability. It departs from the whole-command argmax whenever max_c P(c) is smaller than the pooled mass of a sibling group. This biases it toward large command families, such as the 7 `set_*` commands against `observe`. Normalized whole-command scoring removes the bias.
   - Why it helps: it explains how SFT inflated `set_x_neg` (all 11 near-tie states). It is a general warning for policies that use constrained decoding.
6. **T6, the value of information that exists only in text.**
   - Statement: the value of the text is J(U) − J(O) ≥ 0 under the true objective. An LLM's gap closure is at most 1 in expectation. A rule parser bounds from below the share that needs no language understanding.
   - Why it helps: it bounds what a positive result can show, and it makes F1 and F2 readable as a statement about information rather than "LLM beats planner".

---

## 7. Decision for the lead: the public preprint and the README

**Facts.**
- **The preprint.** arXiv 2602.02451 (v1 February 2, 2026; v2 June 18, 2026) has the same title as the README. Its abstract claims a "70-71% improvement over baselines at equal intervention budgets" (p < 0.001, Cohen's d about 2). It also says the DPO policy "autonomously discovers" collider strategies (literature-policy 9.1). The author list was not checked, so the identity of the project rests on the identical title.
- **The README.** `README.md` line 14 and table row 114 carry the same claim, re-checked today with a read-only grep.
- **The audit.** The September 10 metric audit invalidated every ACE-versus-baseline number through four confounds:
  - a reward-shaped root weight leaked into the metric;
  - the validation distributions differed;
  - the baseline simulator was non-stationary at N ≥ 15;
  - the student networks were not matched.
- **The corrected result.** No advantage at 5 nodes once the learner is matched, and DPO hurts at 15 to 50 nodes.
- **What else is at risk.** A literature search found the preprint through a generic query, so reviewers of any new ACE paper will find it too. The v9 slide deck carries customer-facing numbers that have since been refuted (prior-evidence-core B7). An untracked PDF of the pre-audit draft sits next to the delivery source (manuscript note section 1).

**Options.**
- **A. Do nothing until the new paper.** Not recommended. Every framing contradicts the preprint, the credibility cost grows, and a public preprint on the same project may expose authorship.
- **B. Correct the README now.** Add a banner at the top saying the 70-71% results were invalidated by an internal audit in September 2026, with a pointer to a short audit summary, and strike the table rows. This takes minutes, can be undone, and involves no outside party.
- **C. B, plus an arXiv replacement (v3).** The replacement would be a short correction notice summarizing the four confounds and the corrected results. It needs the coauthor's approval, and no new experiments.
- **D. B, plus an arXiv withdrawal with a comment stating the reason.**

On both C and D, **[A]** arXiv generally keeps earlier versions visible. The lead should check arXiv's current policy pages before choosing between them.

**Recommendation.**
- Do B this week.
- Do C (or D) within about two weeks, and in any case before the next submission, talk or wider sharing.
- Check the external deck.
- Move the stale PDF to a clearly named archive location. Do not delete it.

This strengthens submission condition 5 (preserved failures), because the correction becomes part of the record the new paper cites. All of these are human actions. Nothing in this agenda changes the repository.

---

## 8. Asks for the lead

1. **Venue, strategy and acceptable outcomes.** The target venue and date. Option (i), one paper at NeurIPS 2027, or option (ii), F3 at ICML 2027 followed by a later F1 or F2 paper. Whether a diagnostic paper (F3) is an acceptable outcome if no LLM benefit emerges.
2. **Local forward-only inference for E2.** Approve it as a new model execution, under the attached protocol and its 99% fidelity check.
3. **Local numerical simulation.** Approve local, numerical-only simulation on new declared development and training namespaces (E1, and data for E3 and E5), or confirm that no approval is needed.
4. **Where training and confirmation run.**
   - Local LoRA training on the Mac with a pinned runtime manifest, or a new CURC ceiling. Rough needs: E3 and E5 together about 15 to 30 core-hours and 11 to 19 GPU-slice-hours; E6 15 to 40 core-hours and 1 to 7 GPU-hours. The remaining envelope is 3.8 core-hours and 0.76 slice-hours.
   - Whether the confirmatory run (E6) must use the CURC custody harness, which needs a redesign: sharded archives, in-process numerical policies, no repeated full-tree verification. The alternative is a pinned local runtime with independent review.
5. **The cached 7B weights.** Permission to use the Qwen2.5-7B-Instruct weights already cached locally in E4. No download.
6. **The `causalchamber` package.** Approval to install version 0.2.8 (MIT), or to vendor its numpy wind-tunnel simulator file, for E7.
7. **Remote Lab (E8 only, after E7 holds).** A human sign-up, acceptance of the terms, and a spend cap of about 300 EUR.
8. **The public record.** Correct the README now. Replace or withdraw arXiv 2602.02451 with the coauthor's approval. Check the external deck. Archive the stale pre-audit PDF.
9. **A human paraphrase author.** Name someone, for example the coauthor or a colleague, to write the held-out prose family D. Otherwise accept model-generated paraphrases from a different model family, with the caveat stated.
10. **The delivery study.** Permission for checkpoint-only diagnostics on the 640 saved fits (E9), including a CURC retrieval session if needed. A decision on whether the delivery study goes to TMLR independently.
11. **The admission rule.** Accept E1's admission rule as binding before any LLM sees K1 or K2. Accept that K0 stays in the paper whatever happens.
12. **What counts as a benefit.** Decide whether a benefit from information access counts toward condition 2: the LLM reads goals or knowledge that exist only in prose, and is compared with an oracle given that information numerically. The alternative standard is a win under equal information, which T2 rules out in matched regimes.

---

## 9. What not to do

- Do not train with DPO or any preference method on labels from replicate measurements with a 2-SE, 4-replicate tie rule. Do not scale the November recipe.
- Do not screen the canonical 3-variable known-graph class again for an LLM benefit. Use it only for the imitation gates in E3 and the headroom replication in E1.
- Do not choose, add or swap a benchmark after any LLM has run on it. Admission happens only under E1's rule and before any LLM run. K0 stays in the paper.
- Do not use realized first-crossing cost as a primary endpoint. Do not use regret under the planner's own score as the primary for a claim that the LLM beats the planner.
- Do not give the LLM scores per action without reporting the argmax-of-prompt control. Do not count agreement with that argmax as the LLM's contribution.
- Do not tune, select or check-point anything on the 24 development systems. Do not use their states for training. E2 uses them only for frozen-adapter forensics with no outcomes.
- Do not run the 100 to 150 system confirmation of the current policies. Do not run development iterations through the full CURC custody harness.
- Do not install OpenModelica or ThermoPower, sign up for the Remote Lab, or spend money before a positive result at the simulator stage. Do not revive BoxingGym, NeuronBench, CausalMan or Mission-1 root designs.
- Do not cite any pre-audit number. Do not submit or present new ACE work while the preprint and README still carry the invalidated claim uncorrected.
- Do not treat the delivery study as evidence about acquisition. Do not let local-runtime LLM results stand in for the frozen runtime without the fidelity check.

---

## 10. Biggest uncertainties

1. Whether any class the code can express carries decision value in text that a rule parser cannot extract (E1 condition (b)). Synthetic prose may simply be parsable.
2. Whether a 0.6B model can read numeric SCM state at all, even with repaired supervision. EVOLvE and Schmied et al. find that small models trail their teachers.
3. Local MPS throughput and fidelity to the frozen runtime. Neither has been measured. The local torch and transformers versions differ.
4. Whether the lead accepts an information-access claim, rather than an equal-information win, as meeting conditions 1 and 2.
5. Whether independently written paraphrases can be obtained without paid APIs. Without them, a reviewer can argue the text channel is artificial.
6. Calendar risk. The holidays, human-gated approvals, and past overhead of about 39 active hours per screen cycle make ICML infeasible for F1 and F2 and tight for F3.
7. Whether truth-conditional endpoints stay valid under the simulator's misspecification, and how well posterior-risk stopping holds up there (E7).
8. The subjective probabilities of the branches (Section 11.6), which drive the recommendation to front-load the spine.
9. Exactly how arXiv displays a replacement or withdrawal, and whether the coauthor is available in time.
10. Whether the 640 delivery checkpoints can be retrieved without a CURC session.

---

## 11. Supporting calculations

### 11.1 Paired SD for decision-level endpoints, from saved statistics

The paired SD comes from SD = |mean| × √n / |t|, with n = 24 and √24 = 4.899, applied to the replayed paired log final posterior risk (numerics-protocol 7A):

| Contrast (target risk minus other) | mean (ln) | t | SD |
|---|---:|---:|---:|
| coverage | −0.319 | −8.1 | 0.193 |
| random | −0.347 | −6.6 | 0.258 |
| ranker | −0.523 | −9.2 | 0.278 |
| local information | −0.055 | −2.6 | 0.104 |

These are natural logs. A check: the ratio of geometric means of final posterior risk for target risk against coverage is 0.01244 / 0.01711 = 0.727, and ln 0.727 = −0.319. Contrasts between designs that differ give SD 0.19 to 0.28. Similar designs give about 0.10. These SDs come from single realized episodes. Truth-conditional evaluation with R = 4 should lower them, so the figures are conservative.

### 11.2 Sample sizes

The formula is n = ((z₁₋α′ + z₁₋β) × s / δ)².
- E6 uses Holm over two primaries, so the first test is at 0.025 two-sided (z = 2.241).
- 90% power gives z = 1.2816, so the sum is 3.523. 80% power gives z = 0.8416, so the sum is 3.083.
- δ = ln(1/0.90) = 0.1054.

| s | n at 90% power | n at 80% power |
|---:|---:|---:|
| 0.19 | 40.4 | 30.9 |
| 0.25 | 69.9 | 53.5 |
| 0.28 | 87.7 | 67.1 |

With n = 80, power is 90% if s ≤ 0.268 and 80% if s ≤ 0.306. With n = 120, power is 90% if s ≤ 0.328.

**E1, n = 60.** The SE is 0.025 to 0.036. Suppose the true O/U ratio is 0.80 (ln −0.223) and the SE is 0.036. The probability that the point estimate is at or below 0.85 is then Φ((−0.1625 + 0.223)/0.036) = Φ(1.68) = 0.95, and the upper CI bound is about exp(−0.223 + 0.071) = 0.86 < 0.95. If the true ratio is 0.90, the probability of admission is Φ(−1.59) = 0.06.

**E7, n = 40, s = 0.25.** A 15% effect gives z = 0.1625 × 6.325 / 0.25 = 4.11, about 98% power. A 10% effect gives z = 2.67, about 76% power.

**E3, 80 worlds.** The SD of regret per world is about 0.2 × √24 / 3.5 ≈ 0.28, inferred from the November contrast of DPO against random (about 0.2, Wilcoxon p ≈ 0.0005). The SE at 80 worlds is 0.031, so a difference of 0.10 gives z ≈ 3.2.

**For contrast, the November endpoint.** The minimum detectable difference was 3.7 cost units at n = 24 against a ceiling of 3.04, and a 1-unit difference needs about 305 systems (results-analysis-verify D).

### 11.3 GPU cost model for training

- **Measured throughput.** One SFT update on the 20 GB slice took a median 0.595 s for a median of 3,014 input tokens, about 5,000 tokens per second for forward and backward passes at rank-8 LoRA with decoder recomputation (ledger-ops e). A DPO update took 0.994 s for 4,025 to 7,342 tokens, which is consistent.
- **An E3 summary-prompt run.** 4,000 states × 2 epochs × 600 tokens × 1.5 for the listwise overhead = 7.2 M tokens. That is 1,440 s, or about 1,900 s at rank 16 on 7 projections (a factor of 1.3 **[A]**), so 0.4 to 0.55 slice-hours.
- **The raw-envelope arm.** 2,000 states × 2 × 3,000 × 1.5 = 18 M tokens, or about 1.0 to 1.3 hours.
- **Total for 13 runs.** About 6 to 9 slice-hours. The remaining envelope is 2,742 slice-seconds (0.76 hours).
- **Inference.** The November decisions took 0.43 to 0.71 s with prompts of 2,000 to 4,000 tokens. Summary prompts of about 600 tokens are expected to take 0.1 to 0.2 s **[A]**.

### 11.4 CPU cost model

- **Numerical decisions.** The measured cost was about 1.0 process CPU-second per decision, including interpreter start (ledger-ops e). The cost inside a single process has never been measured.
- **E1.** About 155,000 decisions, or at most 43 CPU-hours at 32 draws. 256-draw planners and the rollout subset add about 20 to 60 CPU-hours. The total is about 60 to 100 CPU-hours.
- **Remote Lab.** 4 policies × 30 episodes × 16 interventions at about 10 s each (8 s to settle plus measurement) is about 19,200 s, or 5.3 hours. That is about 19,200 credits, or about 270 EUR at 0.014 EUR per credit. The 1,000 free credits are about 17 minutes (brief section 4).

### 11.5 Calendar model

- **The November chain** (ledger-ops a and g):
  - 181 calendar hours and about 39 active hours.
  - The scientific kernel was about 2.4 hours.
  - 34% of active hours in the scientific phase went to recorded launcher, collector and transfer time.
  - 11 of 22 failed submissions were guard failures in the custody harness.
- **Projections on CURC.** Retraining plus a screen takes 3 to 5 days. A confirmation of 100 to 150 systems takes 6 to 10 days.
- **The local route [A].** It removes the queue and the sbatch journals, the dual locks, the custody archives and the startup deadlines. That should cut the calendar for E1 to E5 roughly in half, while keeping the written protocols, frozen hashes, independent review and immutable evidence directories.

### 11.6 Subjective branch probabilities [A]

| Gate | Probability of passing (judgment) | Basis |
|---|---:|---|
| E1 admits at least one class | 0.6 to 0.75 | K1 has decision value by construction. The open question is whether a parser can extract it. |
| E3, A1 passes at 0.6B | 0.4 to 0.6 | Kriz et al. succeeded with the same base model on a categorical task. EVOLvE and Schmied et al. show gaps with numeric state. |
| E5, a variant qualifies, given the two gates above | about 0.4 | The rule-parser control is strict. Development pilots shrink. |
| E6, H1 and H2 hold, given qualification | about 0.6 | Shrinkage from development to fresh systems. |
| A confirmed positive result (F1 or F2), overall | about 0.1 to 0.2 | The product of the above, plus the hybrid paths. |

The spine yields F3 at close to probability 1, and its cost is small next to the positive branch. That is why E1 to E3 come first, and why the expensive steps (E6 to E8) sit behind gates that do not depend on the LLM.

---

## 12. Facts relied on (sources) and assumptions

- **The canonical class and its numbers.** The design, the 8 means, Findings A to E, and the relative regrets (TR 0.016 … DPO 0.700) are from the brief. The verified versions are in results-analysis-verify, deployment-forensics-verify, training-forensics-verify and numerics-protocol.
- **Code limits.**
  - The generic layer's limits: 8 nodes, 32 actions, 64 queries, 80 history rows, at most 8 policies, at most 64 worlds.
  - Query weights must be equal.
  - Arms cannot share measurement namespaces.
  - The learner allows only known noise and a zero-mean isotropic prior.
  - Source: numerics-protocol, sections 2 and 6.
- **Namespaces.** The reserved confirmation range [202611071000, 202611072000) and the consumed ranges are from ledger-ops c and prior-evidence-core G.
- **Operations and money.** The remaining envelope (13,735 CPU-s and 2,742 slice-s), the QoS limits, the unit costs and the projections are from ledger-ops. The Remote Lab facts and the `causalchamber` versions are from literature-benchmarks.
- **Literature.** Kriz et al., EVOLvE, Schmied et al., Paprika, GO-CBED and Gupta et al. are from literature-policy. The venue dates are from literature-benchmarks section 5.
- **The README claim.** Lines 14 and 114 were re-checked today with a read-only grep.
- **Delivery checkpoints.** A bounded read-only search of `ACE/results/delivery_prospective_*` and of `/Users/pat/ACE_Study_Results` (path patterns containing "rospective", maxdepth 6) found only 8 pilot `.pt` files.
- **Assumptions [A].**
  - local MPS throughput;
  - in-process planner cost;
  - Mac core count;
  - engineering durations;
  - the overhead of listwise training and rank-16 LoRA;
  - arXiv's display policy;
  - dual-submission rules;
  - all branch probabilities.
