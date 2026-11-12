# Judge assessment: area-chair lens (causal experimental design, LLM agents; ICML, UAI, TMLR)

Prepared 2026-11-12. This is a reading and design task. Nothing under `/Users/pat/code/ACE` was modified, and no experiments, inference, SLURM, network or installs were used. Facts were checked against the brief and the w1 notes; section 10 lists the checks.

## 1. Verdict

No single proposal would survive review as written, but together they contain a strong agenda. Any paper here has to answer one question: **where can a trained LLM legitimately beat an equal-information numerical planner, and did it?**

- **advantage-theory** and **paper-strategy** ask that question correctly.
- **evaluation-statistics** supplies measurement that makes either answer credible.
- **policy-repair** supplies the forensic and learnability gates that condition (1) needs. It overbuilds them in a class that has a theorem-level ceiling.
- **experimentalist-task** is the only proposal with a serious route to condition (4). It commits to hardware far too early.

The final agenda should take:

- its **decision tree** from paper-strategy;
- its **science** from advantage-theory: the equal-information ceiling, the value-gap decomposition, the control suite and the known-capture instrument;
- its **measurement** from evaluation-statistics: CRN, a truth-law (Rao-Blackwellized) evaluator, A/A and planted-effect validation, seeds, blinding and independent re-analysis;
- its **gates** from policy-repair, cut down hard: forensics, the CPU learnability ladder and one LLM imitation gate;
- its **transfer task** from experimentalist-task: the wind-tunnel simulator with an operating-point outcome first, and hardware only after a positive simulator result.

## 2. Scores (0 to 10, this lens)

| Designer | Score | Reason |
| --- | ---: | --- |
| advantage-theory | 7.5 | Best science and comparators, and an instrument-calibration step that is rare in this literature. It has no training recipe and no learnability gate, and its confirmation drops the standing controls. |
| paper-strategy | 7.0 | Best program architecture: decision tree, kill criteria, rule-parser residual in the admission rule, public-record correction first. The confirmation uses a single seed, there is no no-text law-aware control, and the evaluator is never validated. |
| evaluation-statistics | 6.5 | Makes any outcome credible, and the methodology could be a paper in its own right. It is method-agnostic, partly uses development LLM outcomes to pick the endpoint, and runs hardware regardless of results. |
| policy-repair | 5.5 | Excellent forensics, gates and lemmas, which make a strong diagnostic paper. Its positive thesis (imitate the planner in a capped class) is incremental. It ignores condition (4), and the advantage stage is only "should". |
| experimentalist-task | 5.0 | The only real answer to condition (4). The headroom premise is weak, hardware and sign-up come before any evidence, and the hybrid and robust-planner comparators are missing. |

## 3. Cross-examination

### 3.1 policy-repair

**Strongest ideas**

- **E1.** An in-process CRN planner that must reproduce all 560 saved target-risk receipts before use, followed by a precision ladder and an "infinite-data ceiling" of the old four-teacher label process. Together these say how much the old supervision could ever have taught.
- **E2.** Forward-only forensics behind a 99% re-decode fidelity gate:
  - the 1,858 x 9 score-table decomposition;
  - a whole-command versus trie-greedy decoding audit;
  - a counterfactual-sensitivity coefficient on posterior swaps, against invariant edits (history permutation, menu order, identifiers, budget, rounding);
  - hidden-state probes.
- **E4.** A CPU-only numerical learnability ladder (coarse N0 up to the equivariant posterior network N2, 32 to 2,048 worlds, stress trajectories). It decides cheaply whether the state-to-action map is learnable at all.
- **Gates G1 to G8 as mandatory acceptance.** They replace the legality-only gate that let the November collapse through. They include a closed-loop longest-run check.
- **Lemmas.**
  - Trie-greedy picks the branch with the largest subtree mass.
  - State-blind collapse under token-factorized logits.
  - A Gibbs soft-target regret bound.
  - With known per-action scores the exact expected-score gradient makes GRPO sampling unnecessary.

**Serious flaws**

1. **The positive thesis is imitation in a class with a theorem-level ceiling.** At best the paper says "Qwen3-0.6B matches target risk". The literature notes grade that as not novel: EVOLvE, DPT and Kriz et al. (literature-policy section 4.3). The research question (a benefit over strong numerical policies) is reached only in E8 to E10, all marked "should". Under the proposal's own priorities, the paper might never test it.
2. **Submission condition (4) is not addressed at all.** There is no experimentalist task and no transfer beyond the generator.
3. **E3's 2 x 2 x 2 factorial costs a great deal for an in-sample attribution on 128 states.** It needs 16 LLM runs, a new 800 to 1,200 line engine, and original-interface cells on 2 to 4k-token prompts. The single-pass roster-order cells re-execute the old SFT recipe, which comes close to breaking the "no old pipelines" rule. Most of the attribution value can be had from three knockouts inside the learnability gate.
4. **The GPU plan runs to about 35 slice-hours (E5 to E7) before any headroom evidence exists.** The 20 GB slice runs only under gpu-testing, capped at a 1-hour wall (ledger-ops lines 212-214).
5. **The E8 selection rule has a free branch.** It picks the largest H2 "if the hypothesized LLM contribution is semantic" (policy-repair.md:331). That hypothesis must be fixed now.
6. **The counterfactual-battery power calculation treats 300 states as independent.** They come from 56 worlds, so the analysis must be clustered by world. Posterior-swap envelopes are internally inconsistent, as the proposal itself notes.
7. **The E7(c) hidden-state linear action head would not count as an LLM action policy for condition (1).**

### 3.2 advantage-theory

**Strongest ideas**

- **T4 and T5.** The equal-information ceiling and the sequential value-gap decomposition (decision noise, myopia, omission, misspecification, information), each term a difference of computable numerical arms. A capture ratio above 1 triggers a leakage audit.
- **E1.** A zero-response decomposition of the completed screen (rollout, clairvoyant, law-informed and extended-menu references). It turns Finding A into a bound.
- **E3.** A qualified equal-information control suite:
  - a law-aware planner (C5);
  - a rule parser written before any LLM is scored (C7);
  - a GO-CBED-style amortized policy with a 1x, 4x and 16x budget sweep (C8);
  - a semantic-feature student (C9);
  - oracle-informed and reliability-aware ceilings.
  This is what a 2026 reviewer will ask for (literature-policy section 7.1).
- **E5.** Instrument calibration with synthetic policies of known capture (mixtures over rho, a leakage policy, a feedback-blind copy), run before any LLM is scored. I have not seen this done in the LLM-for-design literature, and it is decisive for credibility.
- **E7 and E10.** Dose-response with zero-dose and completed-regime placebo strata, recovery delay under misleading text, and the T7 argument for where shuffled-feedback controls have power at all.
- **The L3 compiler arm,** with the honest prediction that compiler plus planner will match or beat the policy.

**Serious flaws**

1. **There is no recipe and no learnability gate.** E7 imports "the recipe from the training workstream", yet every past LLM arm collapsed (brief, Finding E). The agenda is incomplete on exactly what condition (1) requires. It also includes no forensics of the saved adapters, so its null paper lacks the collapse anatomy.
2. **E10 drops the standing controls.** Its arms are the trained LLM role plus numerical controls (advantage-theory.md:527). There is no frozen base LLM and no supervised LLM, which the standing rules require, and no training-seed replication.
3. **"A legitimate LLM win was capped at about 2% of final risk" (advantage-theory.md:112) comes before the evidence for it.** The rollout and DP references have not been computed (numerics-protocol section 4.2). The cap holds in matched regimes against a Bayes rule. In shifted regimes, any policy trained on the generator learns the generator's law, which target risk does not know. The proposal's own T6 suggests this is second order, but it should be stated as a bound to be verified by E1, not as a fact.
4. **The engineering path is long.** E2 to E5 take about 5 to 8 engineer-weeks before any LLM training. The atlas has about 60 cells and 2,900 systems, and choosing the "largest G" among them invites a winner's curse. C2 needs a world-schema change, and whether the assembler enforces graph equality is unverified (advantage-theory.md:202).
5. **Templated hints can be read by a rule parser by construction.** The legitimate LLM edge then rests on a few human-written held-out phrasing families. Those families, not the systems, are the effective units for the claim.
6. **E9 (the non-myopic budget class) cannot show superiority by construction.** Most of T1 to T3 are known results (Chaloner and Verdinelli; Das and Kempe; Golovin and Krause). Present them as tools. The decomposition and the instrument are the actual contributions.

### 3.3 experimentalist-task

**Strongest ideas**

- **It designs for condition (4).** It uses:
  - an independently authored simulator;
  - billed time with a move-versus-replicate cost asymmetry;
  - prose semantics;
  - an observable useful outcome: an operating point that meets pressure and flow specifications at the lowest current, decided by a shared chance-constrained rule, so that arms differ only in the data they collect.
- **Device classes are held out of training** (fans, pumps, compressors, pressure sensing), so semantic transfer is tested out of distribution.
- **An LLM-independent headroom audit (E1)** runs before any LLM touches the task, with triviality checks and variance components.
- **The E6 control set is clean:**
  - anonymized semantics (L4);
  - misleading text (L5);
  - shuffled feedback (L6);
  - graph withheld (L7);
  - a frozen-embedding numerical student (N9);
  - a compute-matched rollout (N5);
  - an argmax-of-prompt straddle rule (N7);
  - Holm correction over three primaries.
- **Two useful theory items:** pool-replay unbiasedness for hardware, and CRN variance components.

**Serious flaws**

1. **The headroom premise is weak.** Fan physics is monotone and smooth, so constrained knowledge-gradient planning is likely to solve the task (the authors concede this). The fallback, up to two rounds of "levers" (experimentalist-task.md:158), searches for a configuration that has headroom. That is LLM-independent, but it is a forking path. Every configuration tried must be registered and reported.
2. **The semantic-transfer mechanism is speculative for a 0.6B model.** The model must map a device library onto fan affinity laws. There is a risk of leakage from the library (DC-motor laws) and of recall of the public Causal Chambers documentation. Fresh paraphrases and misleading text do not separate reading from recall.
3. **Hardware calibration (E2) and the roughly 400 EUR hardware run (E9) are marked "must" before any simulator evidence.** They need:
   - a human sign-up and acceptance of terms that include joint data ownership and use of the customer's name in marketing;
   - a Python 3.11 environment;
   - a check of what the package's call-home sends.
   That is operational risk and spend that nothing has yet justified.
4. **Comparators a reviewer will ask for are missing:**
   - an LLM that proposes a mechanism family or compiles a prior and feeds a numerical planner (the hybrid);
   - a robust model-averaging or model-selection planner without text.
   Without them the "generic cubic-basis" planner looks like a strawman.
5. **There is no anatomy of the November collapse, and the learnability check is minimal** (Q3: 500 steps on 2,000 states). A full SFT plus GRPO stack of about 20 slice-hours is built from scratch.
6. **The scope is one simulator family and one apparatus.** The 128 "systems" are parameter perturbations of a single simulator, and pyCycle has not been prototyped.
7. **The prompt gives per-candidate probabilities of meeting the specifications** (experimentalist-task.md:92). This sits close to narration for the final decision. N7 must be shown to be clearly worse than the LLM.

### 3.4 evaluation-statistics

**Strongest ideas**

- **A Rao-Blackwellized truth-law evaluator (ERB)** with validity conditions and a counterexample showing that the learner's posterior risk can be gamed under side information. This answers the circularity objection.
- **The harness is validated before anything is scored:** CRN keyed by action occurrence, A/A test size, planted-effect power, ERB unbiasedness, and agreement of paired-branch attribution.
- **A threshold-averaged band cost-to-target (T2)** that generalizes the declared endpoint instead of replacing it, with censoring and a public stop rule.
- **Confirmation discipline:**
  - K = 5 seeds, at least 4 of them favorable;
  - a TOST null branch at a pre-declared margin;
  - a Holm second-level family;
  - blinding;
  - fresh validation shards with a cap of three iterations;
  - a failure ledger;
  - independent re-analysis with 10% re-execution.
- **The open-loop bound (T5)** as an outcome-based certificate that the policy uses feedback.

**Serious flaws**

1. **It is method-agnostic.** Classes and recipes come from an unspecified "method panel", so the agenda alone cannot produce the paper.
2. **Endpoint adoption in E1 depends partly on how the development LLM arms placed** ("non-favorability", evaluation-statistics.md:137). The intent is protective, but it breaks the rule that a benchmark or measurement change rests on criteria independent of LLM performance. The test also has little bite, because those arms were constant schedules. The endpoint should be chosen by theory and numerical-only calibration, with the LLM positions reported only descriptively.
3. **Class thresholds are stated in per-system SD units.** Low-variance classes can meet them with trivial gains in natural units. Natural-unit minimum important differences are needed alongside.
4. **E9 Stage B runs hardware "regardless of how the policies performed".** It should be conditional.
5. **There are no law-aware or LLM-compiler controls,** and E8's synthetic transfer families come from the same codebase. ERB is unavailable under misspecification, so its efficiency gain disappears exactly where condition (4) is tested.

### 3.5 paper-strategy

**Strongest ideas**

- **A complete decision tree.** Each branch has a claim (F1 positive policy, F2 compiler, F3 diagnostic, D delivery), kill criteria K1 to K6 are set in advance, and venues are mapped. Correcting the README and arXiv record is a precondition.
- **The E1 admission rule** uses numeric thresholds that include a rule-parser residual on held-out phrasing families (condition (b)). That is the cleanest LLM-independent definition of "information that exists only in text". The endpoint is frozen before any LLM sees the new classes.
- **E3 imitation on fresh training-namespace worlds,** with matched numerical students (including N2 trained on November-rule labels) and knockouts tied to named mechanisms.
- **E5 tests three LLM roles** (direct, hybrid compiler, and an adjudicator with an argmax-of-prompt deviation test) against N-emb and N-bow, and promotes a single variant.
- **E6 has two primaries** (against the uninformed planner and against the parser). E7 is an external wind-tunnel transfer, and hardware is optional.
- **The calendar is realistic.**

**Serious flaws**

1. **The K1 "prose objective" class withholds the objective from the planner.** Reviewers will call it a strawman unless the text really resists parsing. That needs the human-written family D, which is not secured, and families A to C are probably solvable by a parser.
2. **There is no no-text law-aware planner and no no-text amortized student.** A trained LLM learns the training generator's coefficient law, so a gain in K2 could come from that law rather than from the text.
3. **The confirmation uses a single frozen L*, and E5 uses two seeds per variant,** so recipe variability goes unestimated (Kriz et al. flag this).
4. **No evaluator validation** (A/A, planted effect, known capture) runs before scoring.
5. **E3 is heavy** (13 LoRA runs, about 6 to 9 slice-hours, about 3 weeks). A6 (raw envelope) and A7 (greedy decoding) are forensic questions that E2 already answers.
6. **The transfer endpoint on the wind tunnel is NMSE, not an observable useful outcome.** E9 (the delivery study) is outside this agenda.

## 4. Cross-cutting issues a program committee will raise

1. **Direct policy versus compiler.** In every admissible class the LLM can win only by supplying information the planner lacks (T4). In that regime the natural strongest design is an LLM compiler feeding a numerical planner. If the compiler matches the direct policy, a reviewer asks why anyone would use the policy, and condition (1) ("an LLM action policy") is then only nominally met.
   - The compiler must be a mandatory comparator.
   - Patrick must rule, before development, which role counts toward the submission condition. Paper-strategy ask 12 and advantage-theory ask 6 both raise this.
2. **Learning the generator law.** Any amortized policy trained on the generator learns its law (for example, edge means of 1.5 in shifted regimes). Target risk uses the N(0, I) working prior. Without a law-aware (empirical-Bayes) no-text planner and a no-text amortized student, a "language" gain is confounded with "trained on the generator". Only advantage-theory has this control (C5).
3. **Text is the unit.** Claims about reading text generalize over texts, not systems. If hints come from a few phrasing families, the effective sample size for the language claim is the number of independently written texts. The analysis needs system x text crossed random effects, or many independently written texts. No proposal sizes for this.
4. **Winner's curse in class selection.** Choosing the cell with the largest headroom from roughly 60 inflates G. The selected class's gap must be re-estimated on fresh calibration systems before the confirmation is sized.
5. **Reading versus recall.** Anonymization and misleading text test whether text is used and whether it can do harm. They do not show that the model reads text rather than recalling pretraining. Counterfactual-law systems would, for example a modified exponent or a flipped sign described consistently in the text and implemented in the simulator.
6. **Reasoning tokens.** Kriz et al. report that direct amortization without reasoning does not transfer robustly across domains, and Schmied et al. tie fine-tuning gains to chain of thought. Every proposal assumes no reasoning, and none trains a reasoning arm. Condition (4) is a transfer claim, so the agenda needs a predeclared reasoning arm in the transfer test or an explicit scope statement.
7. **The endpoint must answer the research question.** The question is about cost to a reliable answer. The primary should therefore be expected cost to a reliability target under a shared public stop rule, read from truth-conditional (Rao-Blackwellized) curves, with fixed-budget truth-conditional error as the key secondary. Paper-strategy's E1 freeze rule is the right mechanism, applied with numerical-only calibration. Under misspecification (the wind tunnel) the stop rule is miscalibrated, so reliability at stop must be checked against simulator truth.
8. **Operations, not compute, are the critical path.** In November a 2.4-hour scientific kernel took about 39 active hours, and 26.4% of GPU slice-seconds were spent on failed jobs (ledger-ops line 24). Development should run on a light local in-process harness, with custody reserved for the confirmatory run.

## 5. What the final agenda must include

1. **Correct the public record and register a program-level decision tree before any new data.** Correct the README and supersede the arXiv preprint first. Hash a registration that maps each gate outcome to a claim and a venue and fixes every kill criterion. A new submission next to an uncorrected 70% claim from the same project is a credibility and integrity problem.
2. **Admit classes by a rule that is independent of the LLM.** The atlas has at most three named candidates:
   - structural knowledge given only in text (a sparse supergraph with absent edges or signs named in text);
   - misspecification on the independently authored wind-tunnel simulator;
   - optionally, the prose-objective class, only if human-written held-out text exists.
   The completed class serves as a negative control. Admission requires:
   - a ceiling gap over the strongest no-text controls (CRN target risk, rollout, a law-aware planner, an amortized no-text student);
   - a residual that the rule parser does not capture;
   - a sample size that is feasible;
   - a gap re-estimated on fresh calibration systems.
   This is the only defensible answer to the no-favorable-benchmark rule.
3. **Validate the measurement before scoring.**
   - The in-process CRN planner must reproduce all 560 receipts.
   - The truth-law (Rao-Blackwellized) evaluator is checked by A/A size, planted-effect power, ERB-versus-realized unbiasedness, and known-capture synthetic policies including a leakage policy.
   - The primary endpoint is fixed by theory plus numerical calibration only, and never by first-crossing realized cost.
4. **Run forward-only forensics on the saved adapters behind the 99% fidelity gate.** This covers:
   - the score tables;
   - whole-command versus trie decoding;
   - menu and identifier permutation;
   - content-free prompts;
   - counterfactual posterior edits, analysed with world-clustered inference;
   - probes;
   - plus frozen capacity probes on 0.6B, 1.5B and 7B over compact renderings.
   This is cheap and decisive, and it is the anatomy section of every paper branch.
5. **Pass a learnability gate before any direct-policy work in a new class.**
   - First the CPU numerical ladder.
   - Then one repaired recipe, 3 seeds: listwise planner targets, a compact state, permuted single-token codes, whole-command argmax, shuffled minibatches, rank-16 LoRA on all linear layers.
   - Alongside it: a label-permuted control, a numerical student on identical data, and at most three knockouts (hard labels, raw envelope, trie decoding).
   - One escalation, chosen in advance by the forensic result: GRPO for a knowing-doing gap, capacity if the state is not decodable.
6. **Use a full equal-information comparator suite in the admitted class:**
   - CRN target risk and a rollout planner;
   - a law-aware no-text planner;
   - an amortized no-text student with a training-budget sweep;
   - a rule parser plus planner;
   - an LLM compiler plus planner;
   - a semantic-feature numerical student;
   - the oracle-informed ceiling;
   - a compute-matched planner;
   - plus the standing controls: frozen LLM, supervised LLM, a numerical-ranker successor, and target risk.
7. **Run identification controls:**
   - text removed;
   - misleading text, with recovery delay measured;
   - shuffled feedback, only where adaptivity value is verified;
   - state permutation;
   - menu and identifier permutation;
   - argmax-of-prompt whenever any derived statistic is in the prompt;
   - dose-response with a zero-dose placebo.
8. **Run one preregistered fresh-system confirmation:**
   - methods, namespaces, endpoints and failure treatment all frozen;
   - CRN;
   - at least 3 training seeds (5 preferred) for the LLM and for the numerical student;
   - an evaluator coded independently, a named independent analyst, and blinding;
   - Holm over 2 to 3 primaries (against the strongest no-text control, and against the parser or compiler);
   - TOST for the null branch;
   - a leakage audit when capture exceeds the ceiling;
   - a failure ledger.
9. **Test transfer to the independently authored wind-tunnel simulator with an observable useful outcome** (an operating point within spec at the lowest current), including counterfactual-law variants.
   - If the wind tunnel fails admission, the transfer test is a predeclared non-inferiority (no-harm) test, not a superiority test.
   - The real Remote Lab is conditional on a positive simulator result and on Patrick's own sign-up and spend approval.

## 6. What must be dropped or changed

1. **No LLM-superiority testing in the completed known-graph class.** All five agree. Use it only as the learnability arena and as a negative control on fresh systems.
2. **policy-repair: make imitation a gate, not the thesis.**
   - Cut E3 to at most three knockouts inside the gate, and drop the single-pass roster-order cells.
   - Collapse E6 and E7 into one predeclared escalation.
   - Do not count a hidden-state linear head as an LLM action policy.
   - Fix the E8 "semantic" branch now.
3. **advantage-theory:**
   - Add the frozen base LLM, the supervised LLM, a numerical-student control and at least 3 seeds to E10.
   - Restate the "about 2% cap" as a bound pending the E1 rollout, DP and law-aware references.
   - Drop E9, or demote it to "could".
   - Cut the control suite to those named in section 5.6.
   - Re-estimate the selected G on fresh systems.
4. **evaluation-statistics:**
   - Remove development-LLM positions from the endpoint-adoption criteria and report them descriptively.
   - Add natural-unit minimum important differences to the class thresholds.
   - Make E9 Stage B conditional on a positive Stage A.
5. **experimentalist-task:**
   - Hardware E2, E9 and E10 leave "must" and become conditional on a positive simulator result.
   - Register the whole lever family and report every configuration tried.
   - Add the LLM-compiler and robust model-averaging planner controls.
   - pyCycle becomes "could".
6. **paper-strategy:**
   - The confirmation needs at least 3 seeds.
   - Add a no-text law-aware planner and an amortized student.
   - Admit K1 only with human-written held-out text, and otherwise lead with the structural-knowledge class.
   - Add evaluator validation.
   - Drop A6 and A7 from E3, because E2 answers them.
   - Use an operating-point outcome on the wind tunnel.
7. **Arms that see planner statistics are never primary.** Any arm given per-action planner scores (L2, L-adj, N4-style) is a secondary or upper-bound control, always reported with argmax-of-prompt.
8. **No Remote Lab sign-up, hardware spend, or 7B training before the gates that justify them.**

## 7. What all five missed

1. **Text is the statistical unit for language claims.** No power analysis models system x text crossed random effects, or the small number of phrasing families.
2. **Winner's curse in class selection.** None re-estimates the selected headroom on fresh calibration systems before sizing the confirmation.
3. **Counterfactual-law systems to separate reading text from recalling pretraining.** No proposal has them. Fresh paraphrases, anonymization and misleading text do not identify reading.
4. **A trained reasoning arm, or an explicit scope decision.** None includes either for the transfer claim, despite Kriz et al.'s direct-versus-reasoning transfer result.
5. **Amortization and total-cost accounting.** None says how training simulation and synthetic responses count against the "lower experimental cost" headline.
6. **A single hashed registration of the entire decision tree**, including the venue mapping and a global failure ledger across development iterations. Paper-strategy and evaluation-statistics each have part of this.
7. **World-clustered inference for state-level analyses** (counterfactual batteries, probes, gate statistics). States from the same world are not independent.
8. **A non-inferiority (no-harm) fallback for transfer** if the external simulator has no headroom. Condition (4) can then still be met with a correctly scoped claim.

## 8. Recommended spine

**Week 0. No compute and no approvals.**

- Correct the README and supersede the arXiv preprint.
- Register the program decision tree, the kill criteria and the class-admission rule.
- Obtain Patrick's rulings:
  - which LLM role counts (direct, compiler or adjudicator);
  - whether a diagnostic paper is acceptable;
  - the venue.

**Weeks 1 to 3. Three parallel tracks, none of which depends on any LLM result.**

- **(A) Measurement.**
  - Build the in-process CRN planner and confirm it reproduces all 560 receipts.
  - Validate the truth-law (Rao-Blackwellized) evaluator on calibration systems: A/A size, planted effects, known-capture policies.
- **(B) Forward-only forensics** of the saved adapters behind the 99% fidelity gate, plus frozen capacity probes on 0.6B, 1.5B and 7B.
- **(C) Theory and decomposition.**
  - The value-gap decomposition of the completed screen, with rollout and law-aware references.
  - The theory write-up: adaptivity and equal-information ceilings, endpoint efficiency, the decoding lemma, the label-noise bound and the side-information value.

**Weeks 2 to 5. Two more parallel tracks.**

- **The headroom atlas** over at most three named classes plus the completed-class negative control, scored only with numerical arms.
- **The learnability gate.**
  - The CPU ladder.
  - One repaired LLM recipe x 3 seeds, a label-permuted control, a numerical student, and at most three knockouts.
  - One escalation, chosen in advance from the forensic result.

**Gate A, about week 6.**

| Outcome | Next step |
| --- | --- |
| No class admitted | Write the theory, atlas, measurement and anatomy paper. ICML 2027 if ready by about January 10, otherwise TMLR. |
| LLM fails imitation while the numerical student passes | The direct-policy route closes. Only the compiler role continues, in an admitted class. |
| Otherwise | Continue to development. |

**Weeks 6 to 10. Development in the admitted class.**

- Training: privileged-teacher distillation plus DAgger.
- Arms:
  - the direct LLM policy;
  - an LLM compiler plus planner;
  - a semantic-feature student;
  - an amortized no-text student;
  - a rule parser plus planner;
  - a law-aware planner;
  - the oracle ceiling;
  - the frozen LLM and the supervised LLM;
  - a compute-matched planner.
- Controls: text removed, misleading text, shuffled feedback, state permutation, menu and identifier permutation.
- Dose strata include a zero dose.
- One variant is promoted by a fixed rule, on fresh validation shards, with at most three iterations.

**Weeks 10 to 14. One preregistered fresh-system confirmation.**

- Everything frozen; CRN.
- At least 3 seeds, 5 preferred.
- Independent evaluator code and analyst; blinding.
- Holm over 2 to 3 primaries; TOST for the null branch; leakage audit.

**Weeks 12 to 16. Transfer.**

- The frozen method runs on the wind-tunnel simulator family with an operating-point outcome and counterfactual-law variants.
- Superiority is tested if the wind tunnel was admitted, non-inferiority otherwise.
- The real Remote Lab is only an optional demonstration after a positive simulator result.

**Venues.** ICML 2027 (January 22) can carry only the diagnostic, theory and measurement paper. The positive program realistically targets NeurIPS 2027 (estimated May) or TMLR.

**Consolidated asks for Patrick.**

1. Local forward-only inference.
2. Local numerical simulation on new declared namespaces.
3. A staged compute ceiling. Training needs a longer-wall QoS on full A100s or local LoRA training, because the MIG slice is limited to gpu-testing with a 1-hour wall.
4. Use of the cached 1.5B and 7B weights.
5. Vendoring or installing the causalchamber 0.2.8 numpy simulator file.
6. The role-counting ruling.
7. Whether the diagnostic paper is acceptable.
8. The public-record correction.
9. An independent analyst.
10. Later, and only conditionally: Remote Lab sign-up and spend.

## 9. Claims that remain open to attack

- **Positive branch.**
  - "Any experimentalist would encode that text numerically." Answered by the parser and compiler controls and by human-written held-out text.
  - "It is a compressed planner." Answered by the numerical and semantic students.
  - "The gain comes from the generator law." Answered by the law-aware planner.
  - "It is memorized documentation." Answered by counterfactual-law systems.
  - "One simulator family, one apparatus." This scope limit cannot be removed and must be stated.
  - "Single seed." Answered by K of at least 3 to 5.
  - "Model-based endpoint." Answered by the truth-law evaluator plus agreement on realized error.
- **Null branch.**
  - "A null about a 0.6B model is uninformative." Answered by capacity probes, the escalation step, numerical students as positive controls, and the ceilings, which show what any policy could have gained.
  - "Known theorems." Answered by presenting them as tools and claiming the decomposition, the atlas and the instrument instead.

## 10. Fact checks performed

| Source | Lines or sections | What was checked |
| --- | --- | --- |
| `w2/brief.md` | in full | Findings A to E, assets, operations, venues |
| `w1/numerics-protocol.md` | 3, 4.2 to 4.4, 5, 7 | Plan-once versus target risk (×0.965, −0.7% ± 1.7%); state reading worth 10 to 20%; rollout and DP not computed; graph, prior and noise limits; per-arm namespaces (`paired_episode.py:114-117`) |
| `w1/deployment-forensics-verify.md` | in full | 560 of 560 receipts reproduced; regret table; prompts and adapters recoverable locally; menu order fixed everywhere; training run lengths of at most 10 |
| `w1/training-forensics-verify.md` | in full | Label quality; chance-level strict labels; token-greedy inflation; recoverable adapters |
| `w1/ledger-ops.md` | 24, 65, 212 to 215 | 26.4% of GPU slice time failed; 0.76 slice-hours remaining; gpu-testing MaxWall 1 hour; slice not allowed on gpu-normal |
| `w1/literature-policy.md` | 4.3, 6 to 8 | Novelty grading; reviewer-demanded comparators and controls; Kriz et al. caveats |
| `w1/literature-benchmarks.md` | in full | Causal Chambers terms; simulators; venue dates |
| `w2/advantage-theory.md` | 112, 202, 527 | The "about 2%" cap claim; the C2 schema dependency; E10 arms |
| `w2/paper-strategy.md` | 49 to 54, 259, 308 | Branch claims; seeds in E3 and E4 |
| `w2/experimentalist-task.md` | 92, 158 | Per-candidate probabilities in the prompt; lever rounds |
| `w2/evaluation-statistics.md` | 8, 38, 137 | Basis for the 8 to 96x efficiency; non-favorability criterion |
| `w2/policy-repair.md` | 331 | Conditional E8 selection rule |
