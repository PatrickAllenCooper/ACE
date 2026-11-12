# ACE experimental agenda after the 24-system screen

November 12, 2026. Proposal for Patrick Cooper and the coauthor. None of the proposed experiments has been run, approved or frozen. AI agents produced the supporting analyses and this draft, and no human has reviewed them yet. The agenda keeps the research question and the five submission conditions of the [November 2 agenda](ace_llm_intervention_agenda_2026-11-02.md) and replaces its work sequence. It follows the [November 12 handoff](handoff_ace_llm_research_2026-11-12.md).

## The question and the bar

**Research question.** Can a language model trained to use an explicit SCM choose interventions that give an experimentalist a reliable answer at lower experimental cost than strong numerical policies? The LLM must contribute to the decision rather than narrate a numerical optimizer's choice.

**Submission requires all five conditions:**

1. a reproducible LLM action policy with an explicit training and deployment mechanism
2. an independently evaluated benefit over a strong numerical acquisition policy on fresh systems
3. ablations separating language pretraining, ACE-specific training, causal structure and extra computation
4. a concrete experimentalist task with an observable useful outcome, plus transfer beyond the synthetic training generator
5. preserved failures, response accounting and independent review.

## In brief

**The November screen could not have shown an LLM benefit, and its learned arms learned almost nothing useful.** In three-variable linear-Gaussian systems with a known graph and known noise, a design planned once after initialization did as well as adaptive target-risk planning. A schedule that ignores the state was 9% to 21% worse. Both comparisons use greedy designs, and the Bayes-optimal sequential policy has not been computed, so "adaptivity is worth almost nothing here" is a bound that E1 and E5 will test. The declared endpoint was also too noisy to separate policies whose means differ modestly. Separately, the supervised and preference-trained adapters were trained on 160 mostly uninformative labels and 119 mostly noisy pairs. They emitted X = −1 on 108 and 123 of their 128 nonterminal training states before deployment. The numerical ranker trained on the same data collapsed too. The screen therefore says nothing yet about whether an SCM-conditioned language policy can help an experimentalist.

**Theory and the literature narrow where a benefit can exist.** No policy can beat a Bayes-optimal planner that has the same model, information and objective. A language policy can legitimately win only where the planner lacks something the language model has: knowledge stated only in text, a correct description of a misspecified mechanism, or an action space the planner cannot enumerate. The most careful published ablations find that language models supply priors and proposals but rarely adapt to measurements, and that students distilled from numerical planners trail their teachers.

**The agenda tests that narrowed claim so that every outcome produces a paper.** In order:

1. Preserve the evidence, correct the public record and register a decision tree.
2. Build and validate the measurement instrument.
3. Dissect the saved adapters with forward passes.
4. Establish learnability with numerical students.
5. Admit a new benchmark class only by a numerical headroom rule that never looks at a language model.
6. Test one repaired LLM recipe in the completed class.
7. Develop and confirm in the admitted class against controls that receive the same information.
8. Transfer the frozen method to a wind-tunnel simulator with a concrete operating-point outcome.

A positive result supports the paper Patrick wants. A narrowed result supports a paper about the language model as a knowledge compiler for a numerical planner. A null result supports a rigorous diagnostic and theory paper.

**Most of the first month is cheap.** Stages 1 and 2 need no CURC allocation and no CURC GPU. The first new model execution is a half-day local pilot: forward scoring of the saved adapters plus about 50 LoRA training steps on weights already on the Mac. It needs Patrick's go-ahead but no download.

## What the completed evidence says

**Provenance.** AI agents computed these figures in this session, and no human has checked them yet. The figures on the screen's outcomes and on the training and deployment records were computed from saved files by one agent and independently recomputed by a second, with no refutations. The model-based replay, the prompt-composition figures and the history of earlier studies rest on one agent, and the history figures were read from project documents. All of it is post hoc analysis of development data. It explains the November result and does not select a new method from the 24 development outcomes. This is not the independent review that condition 5 requires.

### The screen could not have shown an LLM benefit

**The world class left little room for adaptive choice.** Each mechanism had a conjugate Bayesian linear learner with known noise, whose posterior covariance depends on the design rows rather than on the measured responses. Adaptivity can enter only because unclamped parent values are random and because the target composes coefficients across mechanisms. Model-based replay from the 24 saved post-initialization states gives the following.

- Adapting after initialization was worth −0.7% ± 1.7% in expected final NMSE.
- A design planned once from the post-initialization state matched target-risk planning. That design reads the state once, so the comparison isolates sequential adaptation.
- A tuned schedule that depends only on graph and cost menu was 9% (in-sample) to 21% (leave-one-out) worse. Coverage was 37% worse, 24 observations 75% worse and 24 repetitions of X = −1 113% worse.
- These comparisons use greedy designs. The Bayes-optimal sequential policy was not computed.
- Target risk itself is a noisy 32-draw Monte Carlo heuristic. 74.8% of its decisions had a top-two gap under two Monte Carlo standard errors.
- In chain and fork graphs some of the eight target queries coincide. In the chain do(Y = 1) appears twice. In the fork the observational mean appears four times and do(X = −1) twice. In the collider all eight queries are distinct. X = −1 was the planner's best first action in 11 of 24 systems (13 counting exact ties with the joint clamp in fork systems) and the best state-blind first action overall.

**The declared endpoint could not resolve modest differences.**

- Ten of 24 systems met the 0.04 target after the shared initialization. The learner's own posterior predicted 9.9.
- The largest possible improvement over target risk was 3.04 cost units per system. A policy that reached that floor on every system would probably still have been detected (bootstrap power 0.84 to 0.93). What 24 systems could not detect is a noisy policy with a modestly better mean: the minimum detectable paired difference was 3.7 units, and the replay puts any real improvement at a few tenths of a unit.
- The median paired standard deviation against target risk was about 6.2 cost units (3.1 to 9.0 across all pairs). At that value, detecting a one-unit difference needs about 305 systems at 80% power (77 to 643 across the observed values). With eight replicate episodes per system it needs about 93 to 172, if replicate noise is 50% to 80% of the paired variance, which has not been measured.
- Arms did not share measurement noise. The supervised and DPO arms made identical choices for their first 10 to 22 interventions in every system, yet their capped costs differ in 11 of 24 systems, 8 of them decided by a crossing inside the identical prefix.
- The ordering depends on the endpoint. Under sustained attainment menu coverage ranks first (15.13) and DPO ties the frozen LLM (17.00). On the 14 systems not already at target, local information leads target risk on curve area (lower on 9 of 14, p = 0.32, not resolved).
- More sensitive endpoints exist, with caveats. The learner's public posterior risk would resolve the large contrasts with 3 to 7 systems, and static against state-conditional design with about 17 (model-internal predictions). It is biased whenever the generating law differs from the learner's prior, as in the shifted regimes. Decision-level regret under a 256-draw target-risk score orders the arms like the headline endpoint (Spearman 0.93) with far more resolution: target risk 0.016, local information 0.154, coverage 0.441, random 0.504, ranker 0.575, frozen 0.655, supervised 0.665, DPO 0.700. That regret is circular for the target-risk arm and is not validated against outcomes. The frozen LLM, which observed at 574 of 576 decisions, has regret 0.655 yet the fourth-best geometric-mean final NMSE.

### The learned arms were barely trained, and their supervision was broken

**Supervised data.** There were 160 examples: 32 training systems times 5 snapshots. Of these, 32 were terminal "stop" states where stop was the only legal command. Labels rotated across four teachers, and the prompt never said which teacher produced a label. A quarter came from uniform random selection. A quarter came from menu coverage, which ignores the SCM and emits X = −1 at every initial state through an alphabetical tie rule. Labels matched a high-draw planner optimum in only about 55 of 128 nonterminal states. Training was one unshuffled pass with batch size 1, learning rate 1e-4 and a rank-8 LoRA on q and v projections.

**The collapse was visible before deployment.** On its 128 nonterminal training prompts the supervised adapter generated X = −1 108 times and the final DPO adapter 123 times. They matched their teacher labels 36 and 32 times. Emitting X = −1 every time also matches 32. Acceptance checked legality only, so the collapse passed. Out of sample the supervised model never beat a state-blind label-frequency counter (2.37 against 1.93 nats per example).

**Preference labels were mostly noise.** Each of 128 states branched all eight actions with four replicates, giving 3,584 within-state comparisons. The tie rule amounted to a three-degree-of-freedom t-test at α ≈ 0.139. It declared 687 strict pairs, against 476 to 499 expected with no true differences, so about 70% of strict labels are what the rule passes by chance. The labels still carried state-specific signal: they agree with the same state's planner ordering 68.7% of the time, against 50.2% for another state's ordering. Only 119 pairs were used, at most one per state (9 of the 128 states had no strict pair), each once. Collecting the preference feedback cost 9,216 of the 10,240 training cost units.

**DPO did not learn preferences.** Mean loss was 0.7395, above ln 2. Pair agreement at first exposure stayed at 66 of 119. Chosen and rejected likelihoods both fell (by 4.69 and 6.68 nats on average), the pattern known as likelihood displacement. X = −1 rose 0.57 nats while X = +1 fell 3.84, although X = +1 was the most preferred action in the exposed pairs (23 chosen, 14 rejected, against 14 and 9 for X = −1). The per-command shifts followed the supervised model's command prior (r = 0.80 for the final change, 0.89 to 0.91 for the fitted online drift) far more than the labels (r = 0.44). A state-blind model with token-factorized biases tracks the DPO pass: r = 0.80 for the online margins (about 64% of their variance) and 0.93 for the final shifts. One bias per command does not fit (r = 0.16 to 0.36).

**The non-language learner collapsed too.** The numerical ranker trained on the identical 160 records and 119 pairs chose X = +1 435 times and X = −1 141 times. Action identity explains 98.1% of its score variance, and its losses stayed at chance. The failure is therefore not specific to the language model. Whether supervision and exposure caused it is tested prospectively in E4 and E6.

**The language interface carries known risk factors.**

- Deployment prompts ran 1,990 to 4,160 tokens (training prompts 1,998 to 4,199). Between 34% and 48% of tokens were digits of 16 to 17 digit floats. The raw history grew to 67% of the prompt while the posterior summary shrank to 10%. The 346 tokens nearest the answer were identical in every deployment prompt.
- One byte-identical task description and one fixed menu order appeared everywhere. The frozen model emitted the first menu entry. The trained models emitted the first listed intervention, which also wins every alphabetical tie. Position, identity and tie rule are confounded.
- Actions came from greedy token-by-token decoding over a trie of command strings, with thinking disabled. This is not whole-command argmax. After an action-versus-stop branch, the next branch pools seven `set_*` commands against observation. At late training states the supervised model's average probabilities put observation (0.367) level with X = −1 (0.345), yet greedy decoding produced X = −1 in 76 of 96 of those states. This stays an inference until per-state score tables are regenerated.
- No per-command scores were saved for any LLM arm. They are recoverable by forward passes alone. The base Qwen3-0.6B weights on the Mac match the SHA-256 pinned in the scientific freeze (`model_files/model.safetensors`, checked November 12). Both deployed adapters and the tokenizer files are hash-matched, and all 1,858 exact prompt token sequences are saved.
- The supervised adapter responds to coarse state: the budget counter and graph type. History length moves with the budget counter and was not separated from it. No adapter shows evidence of reading the posterior.

### Eleven months of the same pattern

No experiment in this project has shown an LLM proposer or policy beating a non-LLM control. Collapse onto one action recurs from January to November and tracks prompt surface features: the first node, the few-shot example and menu position. Hand-coded caps and fills masked it before. In the preregistered Mission-1 study the LLM arm's selected interventions were 56% to 69% hand-coded fills and 9% to 13% echoes of the prompt's own example. The September metric audit voided the original ACE-versus-baseline claims, and ACE with DPO lost to random at 15, 30 and 50 nodes. The repository README (lines 14, 114 and 116) still advertises a 70 to 71% DPO improvement. A public arXiv preprint with the README's exact title (2602.02451, v2 of June 18, 2026) makes the same claim. Its author list was not checked.

### What is solid

- Numerical SCM planning is the only acquisition signal with repeated support. Target risk led in November. Posterior-risk pair selection passed on two of three synthetic graph families in September. The propagated expected variance (PEV) planner beat random, although its primary contrast against coverage missed its gate (p = 0.051).
- The delivery study is a solid learner-side result on 12 histories of one deterministic emulator (confirmation ratio 0.182719, interval 0.101970 to 0.327410, 11 of 12 histories), with one undiagnosed adverse five-node prospective contrast (2.46).
- The execution and custody pipeline works. Its accounting, failure preservation and independent review are a strength reviewers will notice.
- The forensic diagnosis above is itself publishable once it is reproduced under a written protocol and checked by a human.

## What this means for the paper

**Compare at equal information.** Under the same correct model, information and objective, the planner's expected loss bounds every policy's loss from below. Any claimed LLM gain must therefore come from a planner handicap declared before the experiment: information only in text, misspecification, or an action space the planner cannot enumerate. Each handicap needs a numerical control that removes it, such as an informed-prior planner, a rule parser feeding the planner, or a numerical student on text embeddings. Otherwise a gain shows only that the information is valuable, which a numerical audit can show without any language model.

**Admit benchmarks by a rule that never looks at a language model.** A new class enters only if a numerical audit shows headroom between the strongest planner without the side information and an oracle-informed planner, at a sample size we can afford. The rule is hashed before any language model is trained, prompted or scored in the class. This is how the agenda changes benchmark without violating the rule against favorable replacements. The completed class stays in the paper as a predicted, explained null.

**Validate the instrument before scoring anything.** Use common random numbers keyed by system, replicate and action occurrence. Use an evaluator-side conditional expectation of the true error (Rao-Blackwellized evaluation), checked against realized error. Run an A/A test, a planted effect, synthetic policies of known information capture and a leakage canary. Realized first-crossing cost stays as a continuity endpoint, never as the primary.

**Earn the right to test for an advantage.** First show that some learner can recover planner-quality decisions from the public state, numerical students first. Then show that one repaired LLM recipe can, against a numerical student trained on identical data. Behavioral gates that would have caught the November collapse become mandatory acceptance.

**Write the paper for every outcome.** The theory, measurement repairs, headroom atlas and collapse anatomy are valid whatever the LLM outcome. The decision tree below maps each gate outcome to a claim and a venue.

## Week 0: decisions and the public record

These items come first because they cannot be made honestly after results exist.

0. **Preserve the evidence.** Done on November 12: the session analyses (scripts, notes, verification logs and the 16,384-draw reference table for all 4,371 decisions) were copied from the temporary session directory into [`results/ace_llm_intervention_offline_diagnosis_20261112/`](../../../results/ace_llm_intervention_offline_diagnosis_20261112/README.md) with a README and a SHA-256 manifest. Committing them is the remaining step. E1 and E2 start from them.
1. **Correct the public record.** Add a banner to the README headline and its results table (lines 14, 114, 116) pointing to the [metric audit](metric_audit_2026-09-10.md). Confirm that arXiv 2602.02451 is this project's preprint, then decide with the coauthor whether to replace or withdraw it under arXiv's policy. Check the v9 slide deck, which has uncommitted changes, and the stale `paper/aistats_ace_2027/paper.pdf`. No new ACE talk, deck circulation or submission goes out until the banner is live and the preprint decision is made. The preprint decision is due before Gate B, ahead of any ICML submission. Every new manuscript gets a short section relating it to the earlier public claims.
2. **Rule on what counts as the LLM contribution.** Decide before any LLM result in a new class whether a direct policy, a compiler feeding the planner, or both satisfy conditions 1 and 2, and whether a benefit that comes from reading text counts. An adjudicator that picks from the planner's own shortlist stays a secondary arm in every case, because it sees the planner's ranking.
3. **Register the program.** Hash a decision tree that maps each gate outcome to a claim and a venue and fixes every kill rule, the class-admission rule and the escalation rule for E6. Record it with an external timestamp. The registration opens one program-wide attempt ledger covering every class audited, redesign round, role, model size, recipe, seed and interface choice, and the paper reports its full count. Name a human independent reviewer for the evaluator, the registration and the freeze, because agents will write most of the code and analysis.
4. **Approve the operating model.** See [Operations](#operations): an in-process bench, two custody tiers, local model execution and compute tranches released at gates.

## The experiments

| ID | Stage | Question | Compute | Needs approval for |
| --- | --- | --- | --- | --- |
| E1 | 1 | Is the measurement instrument valid and which endpoint is primary? | Local CPU, 5 to 30 core-hours | Harness changes, calibration namespace |
| E2 | 1 | Why did the adapters emit one action, and do they read the posterior? | Local inference, hours | Local model execution, a 7B download for one probe |
| E3 | 1 | Which formal results explain the null and predict where gains can occur? | None | None |
| E4 | 2 | Can any numerical learner recover planner decisions from the public state? | Local CPU, 10 to 20 core-hours | Training namespace |
| E5 | 2 | Which candidate class, if any, has headroom only side information can close? | Local CPU, 20 to 100 core-hours | Namespaces, text authors, `causalchamber` (for C-mis) |
| E6 | 3 | Can one repaired LLM recipe imitate the planner as well as a numerical student? | GPU tranche 1, about 15 GPU-hours | Tranche 1 |
| E7 | 4 | In the admitted class, which LLM role and recipe beat equal-information controls? | GPU tranche 2, about 15 GPU-hours | Tranche 2 |
| E8 | 4 | Does the frozen method win on fresh systems? | Confirmation tranche | Registration, confirmation tranche |
| E9 | 5 | Does it transfer to the wind-tunnel simulator with a useful outcome? | Local CPU, some GPU | Registered with E8 |
| E10 | 5 | Optional: does the frozen method run on the real wind tunnel? | Remote Lab hours | Sign-up, terms, spend, Python 3.11 client |

### Stage 1. Instrument, anatomy and theory (no new training beyond the E2 pilot's 50 measured LoRA steps)

#### E1. In-process bench and instrument validation

**Question.** Can we measure policy differences without bias, at a feasible sample size, with an endpoint chosen without reference to any language model?

**Bench.** Promote the session's scratch replays, which already reproduce the saved artifacts, into one reviewed in-process module. It contains:

- a callable-policy episode loop
- a target-risk planner with common random numbers across candidate actions and more than 256 draws
- a rollout planner and the conjugate learner
- a truth-conditional evaluator
- a leakage guard that gives policies only the public envelope (graph, menu, costs, budget, history and posterior), hashes it at each decision and keeps the private world in a separate process.

Acceptance is fixed at registration: reproduce all 560 saved target-risk receipts, the saved public states of the numerical arms and all 4,371 decision-state NMSEs to numerical precision. The current harness cannot do this. Its episode loop accepts only subprocess policies, its planner caps draws at 256, its rosters cap at eight policies, and paired episodes forbid shared measurement namespaces.

**Validation.** Use 48 systems from a new calibration namespace drawn from the November generator, with per-system query laws and menu orders. Run numerical arms only: target risk at 32 and at 256+ draws, local information, coverage, random, plan-once, an A/A twin and a planted-effect arm that wastes a quarter of its budget. Use eight replicates for cheap arms and two to four for planners. Checks with tolerances fixed at registration:

- Common random numbers leave marginals unchanged.
- A/A test size lies in [0.035, 0.065].
- Planted-effect power is within ten points of the formula.
- The ratio of realized error to the Rao-Blackwellized expectation lies in [0.92, 1.08] in each regime family.
- The leakage canary is flagged.
- In matched regimes at least 90% of stops under a public posterior-risk stop rule are truly at target.

Also record the variance components that size every later study.

**Endpoint rule.** The primary endpoint is cost to target averaged over a band of thresholds, read from replicate-averaged truth-conditional curves, if it separates target risk from random with z ≥ 3. Otherwise it is fixed-budget truth-conditional log NMSE. Paid cost under the public stop rule, with its reliability, is secondary. Realized first-crossing cost is kept for continuity only. The endpoint is frozen at the end of E1, before the E5 audit starts. A descriptive value-gap decomposition of the completed screen, including the rollout reference, tests the "little room for adaptivity" bound. A disclosure of how the old arms would rank under each endpoint is reported but plays no part in the choice.

**Decision.** No new policy is scored until all six checks pass. E5 repeats the Rao-Blackwellized check in each candidate class. A class where it fails, and the misspecified C-mis family where it is unavailable, uses realized truth with at least eight replicates.

**Cost and prerequisites.** 1.5 to 3 engineer-weeks, mostly the bench. 5 to 30 local CPU-hours. No GPU. Approval of the harness changes and a written protocol.

#### E2. Forward-only forensics of the saved adapters

**Question.** Did the constant action come from a base-model prior, supervised training, the DPO stage or the decoder? Does any checkpoint respond to the SCM posterior, and does its hidden state encode it?

**Pilot first.** A half-day measured pilot on the Mac establishes forward-scoring speed and fidelity on saved prompts and runs about 50 LoRA steps on compact prompts. It decides where later training runs: locally, on a full GPU, or on the 20 GB slice with a checkpointed resume that has passed its own qualification. Every GPU figure in this agenda is unmeasured until then. The 1-hour `gpu-testing` limit rules out runs longer than about 45 minutes on the slice.

**Design.** Checkpoints: base Qwen3-0.6B, the supervised adapter and the final adapter, on the exact saved token ids of all 160 training and 1,698 deployment prompts. Scoring starts only after the registration, including the escalation rule below, is hashed.

- **Fidelity gate:** the local re-decode must reproduce at least 99% of the saved completions. Mismatches are allowed only where the branch margin is below 0.05 nats.
- **Score tables:** log-probabilities of all nine commands per prompt, decomposed into command, state and interaction shares.
- **Decoding audit:** trie-greedy against whole-command argmax and length-normalized argmax, with the change in regret.
- **Surface priors:** menu-order permutation, neutral identifiers and content-free prompts.
- **Counterfactual edits** on about 240 to 300 declared states. Swap in the posterior from a matched donor, scale the target-relevant covariance and flip the sign of the X posterior mean. Add edits the planner ignores, such as permuting history rows. The planner rescores every edited state.
- **Probes:** hidden-state probes for planner regrets, fitted on training-world states only. Frozen capacity probes over compact renderings on the cached Qwen3-0.6B and base Qwen2.5-1.5B. A Qwen2.5-7B-Instruct probe needs its weights downloaded first, which counts as new weights needing approval.
- **Inference:** clustered by world, because states nest in worlds.

**Decision.** The results form the collapse-anatomy section of every paper branch. The escalation for E6 follows a rule fixed in the registration:

- If probes decode planner regrets but the actions ignore them, the escalation is an objective change: on-policy reinforcement learning with an entropy bonus, as in the Qwen3-0.6B result of Kriz et al.
- If only larger models decode them, the escalation is capacity, Qwen2.5-1.5B before 7B.
- If no size decodes them from a compact rendering, the LLM interface must carry derived summaries, and the direct-policy claim narrows.

**Cost and prerequisites.** About 2 days of code and a few hours of local compute, to be measured by the pilot. Patrick's go-ahead for local model execution and a written protocol. No download for 0.6B and 1.5B, and no CURC.

#### E3. Theory

Written in parallel and valid in every branch. Known results (Chaloner and Verdinelli, Das and Kempe, Golovin and Krause) are presented as tools, not contributions.

- **Zero value of adaptivity.** In conjugate linear-Gaussian heads with known noise, clamped parents and a target linear in the parameters, the best open-loop design attains the adaptive optimum. This explains why the completed class could not reward adaptive policies and gives a bug-detecting check.
- **Bounded adaptivity.** With random design rows and composed targets the gap is bounded by the value of perfect information about the design-relevant parameters. This predicts how adaptivity value grows as initialization shrinks, which E1 and E5 test.
- **Equal-information ceiling and value-gap decomposition.** The planner's gap splits into decision error, myopia, candidate omission, misspecification and missing information. Each term is the difference of two computable numerical arms. The decomposition defines the required controls and triggers a leakage audit whenever an observed gain exceeds the ceiling.
- **Design value of side information.** Mean-shift knowledge is second order once a few observations exist. Knowledge that an edge is absent is first order (informed-to-uninformed true-risk ratio 0.26 to 0.36 in a two-coefficient toy). Misleading text with full trust gives unbounded harm, so calibrated trust matters. These are the dose-response curves an LLM benefit must follow, where dose is the amount of decision-relevant side information the text carries.
- **Shuffled-feedback power.** Where the optimal policy is nearly open-loop, a feedback-blind policy is nearly optimal. The standard shuffled-feedback control therefore has no power in the completed class and must be run where adaptivity value is verified.
- **Rao-Blackwellized evaluation and threshold averaging.** State when the learner's posterior risk is a valid endpoint, with a counterexample showing it can be gamed when side information exists. Prove that the band endpoint generalizes the declared single-threshold endpoint.
- **Training lemmas.** Trie-greedy decoding picks the largest pooled subtree, not the likeliest command. A state-blind model with token-factorized logits collapses to per-token pluralities. The replicate tie rule leaves about 0.1 bits per strict label at a 31% flip rate. A Gibbs soft target over regrets has bounded regret.
- **Cost-equivalent conversion.** A final-risk ratio r at budget B equals a saving of about (n0 + B)(1/r − 1) cost units. A 10% risk reduction at 8 + 24 is about 3.6 units, so precise design-quality endpoints can answer the cost question.

### Stage 2. Learnability and headroom (numerical only)

#### E4. Numerical learnability ladder

**Question.** Can any numerical learner recover planner-quality decisions from the public SCM state of held-out systems, and with how much data, which representation and which target?

**Design.** Fresh training-namespace worlds from the November generator. Use nested training sets of 32, 128, 512 and 2,048 worlds plus 256 validation and 256 test worlds. Trajectories mix ε-greedy target risk, the other numerical teachers and stress runs that repeat one action for 4 to 24 steps, because November's deployment states contained runs the training data never had. Every state is labelled by the high-precision planner at zero environment cost.

Learners:

- N0: coarse features covering everything the supervised adapter was found to respond to (budget counter, graph type, history counts).
- N1: the ranker MLP trained to convergence.
- N2: a permutation-equivariant network on posterior features.
- N3: N2 plus derived summaries.
- N4: per-action planner statistics, as an upper bound only.

Targets are hard labels, listwise soft targets and regret regression. Use three seeds per configuration.

**Decision.** Planner quality means closed-loop regret ≤ 0.15, a design-quality ratio whose 95% upper bound is ≤ 1.05, and regret at least 0.10 below N0 with a world-clustered 95% interval excluding zero. N* is the smallest training size at which N1 or N2 reaches planner quality. All selection uses validation worlds, and test worlds are scored once per frozen configuration.

- **Positive:** planner-quality decisions are learnable from the public state. The LLM gets N* worlds and the representation that passed.
- **Null:** only N3 passes. The LLM interface must carry derived summaries.
- **Negative:** nothing passes at 2,048 worlds. The direct-policy route closes in this class.
- **If N0 alone meets the regret and design-quality thresholds:** the class cannot discriminate posterior use, and only E2's counterfactual tests can.

**Cost.** About 1,000 lines of code, 5 to 7 days, and 10 to 20 local CPU-hours. Labelling about 166,000 states at 0.12 s each takes about 5.5 hours, 168 learner fits 3 to 10 hours and closed-loop evaluation 1 to 2 hours. No GPU.

#### E5. Headroom atlas and class admission

**Question.** In which expressible class is there headroom that only side information or a better model can close, large enough to detect at a feasible sample size?

**Declared-handicap table first.** For each class, write down before the audit what the uninformed planner lacks and which numerical control removes each handicap.

**Candidates.** At most three, plus a negative control.

- **K0, negative control.** The completed class redrawn from a fresh namespace. The prediction is no headroom, which replicates or refutes on fresh systems the finding that adapting after initialization was worth −0.7% ± 1.7%.
- **C-struct, structural knowledge in text.** A known supergraph of four or five nodes with sparse truth. Text names absent edges or signs, and the menu includes joint clamps. Theory predicts first-order value. It needs a world graph that differs from the public graph, which the current schema cannot express.
- **C-mis, misspecification.** The team-extended wind-tunnel simulator family (cubic fan current, squared-speed pressure) under the linear learner. From its first use here, whoever writes any training library must not read the simulator's mechanism code, or a frozen overlap audit applies.
- **C-obj, prose objective.** Operating conditions described only in prose while the default planner uses a uniform query law. Admit it only if human-written held-out text exists, because templated prose is parser-solvable by construction.

**Arms (numerical only).**

- random, coverage and a leave-one-out static schedule
- plan-once and target risk at 32 and 256+ draws
- a rollout planner and a law-aware empirical-Bayes planner without text
- an amortized no-text student (N2 trained on the class's training namespace)
- an oracle-informed planner (the ceiling)
- a rule parser feeding the planner and a bag-of-words text model, built to a stated effort standard by an author who does not develop the LLM.

Use 48 to 60 systems per cell with two to four replicates under common random numbers. The gap, the residual and the predicted n are computed on the primary endpoint frozen by E1.

**Admission rule.** All of the following must hold. The rule is frozen and hashed before any language model touches the class.

- The ceiling gap over the strongest no-text control is at least 0.16 in natural-log units (the oracle at least 15% better), with a lower 95% bound of at least ln 1.05, and corresponds to at least one cost unit under the conversion in E3.
- At least a 10% error reduction remains that the parser and the bag-of-words model do not capture on K human-written held-out text families.
- The strongest control beats random with z ≥ 3.
- At most 10% of systems are at target after initialization.
- The zero-dose anchor shows less than a 2% gap, and the gap rises with dose.
- The predicted n to detect half the gap with 90% power under Holm over the registered primaries is at most about 156 systems. K is set so that the text-family random effect still leaves 80% power.
- The gap is re-estimated on fresh calibration systems before sizing, to remove the winner's curse.

Name one primary class. Report every audited class, including failures. At most two redesign rounds are allowed, on LLM-independent criteria only.

If C-mis is the primary class, counterfactual-law systems join the E7 and E8 controls, and transfer uses a different family named at registration (for example a light-tunnel family from the same package, if it passes its own audit). If C-mis is admitted but is not primary, its E9 superiority test is registered as secondary to E8.

**Known-capture calibration.** In the admitted class, run the complete planned confirmation analysis on synthetic policies that mix the oracle and the uninformed planner at known rates. Add a leakage policy that reads the private truth and a feedback-blind copy. Pass requires recovering the capture ratio within 0.1, nominal coverage, at least 80% power at half capture, at most 5% false positives at zero capture, and the leakage policy flagged in at least 95% of draws.

**Decision.** If no class is admitted, the advantage program stops and the paper becomes the theory, atlas and diagnostic paper. If a class is admitted, LLM work proceeds in that class only.

**Cost.** 1.5 to 2 engineer-weeks after E1. 20 to 100 local CPU-hours. Human authors write the K held-out text families before the parser audit, and therefore before Gate A. A separate parser author. The `causalchamber` install or vendoring before the audit if C-mis is included.

### Stage 3. LLM imitation gate

#### E6. One repaired recipe in the completed class

**Question.** Can a repaired LLM recipe recover planner-quality decisions as well as a numerical student trained on identical data? The completed class serves only as a learnability arena on fresh training-namespace worlds. No advantage is claimed in it. E6 runs even if no class is admitted at Gate A, because it decides whether the diagnostic paper reports a working repaired recipe, unless Patrick prefers to stop GPU spend at that point.

**Recipe.**

- **Targets:** listwise soft targets from the high-precision planner, with the temperature set by the Gibbs bound.
- **State:** a compact, rounded, planner-sufficient rendering, with the raw history removed or summarized.
- **Actions:** permuted single-token action codes with randomized menu order, decoded by argmax over legal codes.
- **States:** sampled on-policy with planner relabeling (DAgger), plus stress states and no terminal-only stop labels.
- **Optimization:** shuffled minibatches over several epochs and a rank-16 LoRA on all linear layers.

**Arms.** Three seeds of the recipe. The frozen base model. The ranker successor N1 and N2, trained on identical states. A control whose labels are permuted across states. At most four leave-one-fix-out knockouts: hard labels, raw envelope, trie decoding and no permutation. If Patrick so rules in Week 0, one old-recipe anchor cell (one unshuffled pass at batch size 1 on the original interface) runs on the new engine as a control cell of the new design. Without that ruling the cell is dropped.

**Mandatory gates**, thresholds fixed at registration:

- G1: legality is 100%.
- G2: no action exceeds a 60% share, and entropy is at least half the planner's.
- G3: agreement beats the coarse-feature baseline by 0.15.
- G4: held-out cross-entropy is at least 0.20 nats below a state-blind counter.
- G5: a posterior-swap sensitivity coefficient of at least 0.3, with a world-clustered 95% lower bound above 0.1, and a total-variation change of at most 0.10 under edits the planner ignores (menu order, identifiers, history permutation).
- G6: the state-by-action share of the score table is at least half the planner's.
- G7: across three seeds, a standard deviation of validation regret of at most 0.03 and an action-histogram total variation of at most 0.15.
- G8: closed-loop regret of at most 0.20, with no runs of one repeated action longer than the planner's.

The gates were designed knowing the November failures. The paper may say they would have caught the collapse, but may not call that a validation.

**Decision.**

- **Pass:** all gates, and regret within 0.05 of N2. The recipe advances, and condition 1 is partly met: a reproducible LLM action policy that reads the SCM state.
- **A G1 failure** is an engineering defect, fixed before any scoring.
- **Any failure of G2 to G8:** apply the one escalation fixed by the E2 rule.
- **Language bottleneck:** the LLM still fails while N2 passes after the escalation. The direct route closes, and only the compiler role continues.

**Cost.** A new training-engine version (listwise loss, minibatches, all-linear LoRA, single-token codes, relabeling loop) of about 800 to 1,200 lines, built in Phase 1 so that E6 can start at Gate A. The whole gate takes 2 to 3 weeks including the engine. Each run then takes minutes to about an hour on one GPU at compact prompt lengths, for roughly 10 to 15 runs, inside a tranche capped at about 15 GPU-hours. The platform is set by the E2 pilot.

### Stage 4. Advantage test in the admitted class

#### E7. Development against equal-information controls

**LLM roles.**

- L1, direct policy: reads the text and the compact SCM state and chooses the action. It is primary unless Patrick rules otherwise.
- L2, compiler: reads the text and emits a numerical prior or objective that the planner then uses. It always runs, as the decisive alternative.
- L3, adjudicator: chooses among the planner's top candidates. It stays secondary even if Patrick rules that it counts, because it sees the planner's ranking. It must beat a family of argmax-of-prompt rules, and an L3-only success is reported as a narrowed claim that does not meet condition 1.

**Training.** Every learned arm, LLM or numerical, is trained from the same oracle-informed teacher on the same states with the same relabeling rounds. Every learned arm gets the same tuning budget, spent on validation shards only.

**Controls.**

- target risk with common random numbers, the rollout planner and the law-aware no-text planner
- an amortized no-text student with a training-budget sweep (N2 on the critical path, GO-CBED-style as a should-have)
- the rule parser and the bag-of-words model feeding the planner, and a numerical student on frozen-LLM embeddings
- the oracle ceiling and a compute-matched planner
- the standing frozen LLM, supervised LLM and ranker successor.

**Ablations.** Text removed. Misleading text, with recovery delay measured. Shuffled feedback on policy input, run only where adaptivity value is verified. State permutation. Menu and identifier permutation. No SCM state. Dose strata include a zero dose.

**Discipline.** At most three iterations, each on a fresh validation shard. One variant is promoted by a rule fixed in advance, and every attempt enters the program-wide ledger.

**Cost.** 2 to 3 engineer-weeks (roles, oracle teacher, text integration and controls). GPU inside a tranche capped at about 15 GPU-hours. 10 to 30 local CPU-hours.

#### E8. Registered confirmation

**Registration.** Before any confirmation system is generated, register:

- hashes of methods and adapters, the namespace, endpoints and failure treatment
- the hypotheses, n, equivalence margins, the seed policy and the analysis code
- the E9 design: family parameters, namespace, arms, n, endpoint, failure treatment and the non-inferiority margin.

Re-declare the reserved range [202611071000, 202611072000) for the new class or reserve a new one. Retire the 24 development systems and the 32 November training worlds from every evaluation role.

**Arms.**

- the promoted LLM variant, all seeds
- target risk with common random numbers, the rollout planner, the law-aware no-text planner and the amortized no-text student
- the parser, bag-of-words and embedding controls feeding the planner, and the compiler
- the oracle ceiling and the compute-matched planner
- the frozen base LLM, the supervised LLM and the ranker successor
- ablations: text removed, misleading text, shuffled feedback where adaptivity value is verified, and state permutation.

The ranker successor is N1 trained to convergence on the same data as the LLM. The supervised LLM is the supervised-only checkpoint of the promoted recipe.

**Hypotheses.** Two co-primaries, both required for a full positive:

- H1: the LLM beats the strongest planner without the side information.
- H2: the LLM beats the strongest numerical control given the same information: the parser, the bag-of-words model or the embedding student feeding the planner, whichever is strongest on the E7 validation shards, named at registration.

Use Holm over H1 and H2. A third primary, that the compiler beats the strongest equal-information control, is added only if Patrick's Week 0 ruling makes the compiler count toward condition 1, and E5's sizing then uses the same number of primaries. Use a TOST equivalence test with a registered margin for the null branch.

**Execution.** Common random numbers with two replicates if the Rao-Blackwellized evaluator passed in the admitted class, otherwise at least eight. All three to five trained seeds are confirmed, with recipe-level and instance-level claims kept separate. Failures are scored at the cap and no system is replaced. Arm labels stay blinded. The evaluator is coded independently and a named human analyst re-analyses. 10% of runs are re-executed. A leakage audit runs whenever a gain exceeds the computed ceiling.

**Sample size.** Fixed by the E1 and E5 variance components, including between-family text variance. The planning range is 96 to 156 systems crossed with the K human-written held-out text families, because claims about reading language generalize over texts, not only over systems. Deadlines may cut scope but never change a registered n, endpoint, gate or failure treatment. A confirmation that cannot reach its n is reported as incomplete.

**Decision.**

- H1 and H2 pass: the full positive claim.
- Only H1 passes: the text is valuable, but the LLM was not shown to read it better than a fixed rule.
- Only H2 passes: no benefit over the uninformed planner, reported as a failed confirmation.
- Neither passes: a failed confirmation with no benefit claimed. Equivalence is reported only if TOST shows it within the registered margin.

### Stage 5. The experimentalist task

#### E9. Wind-tunnel simulator transfer

**Why this task.** The Causal Chambers wind tunnel is the only candidate found that combines a validated causal graph, executable interventions, a real cost and prose variable descriptions. Its MIT-licensed `causalchamber` package ships numpy-only simulators of the published physics, with documented misspecification. Its causal chain (fan load to fan speed to pressure) has the same structure as the compressor teaching example in the v9 presentation (drive command to shaft speed to pressure rise), so the experimentalist story carries over.

**Task.** Choose the two fan loads to deliver a required pressure difference at minimum total fan current, on a grid of settings. A shared chance-constrained decision rule makes the final recommendation, so arms differ only in the interventions they buy. A change of fan load (an intervention) costs about 10 units (8 s settling plus measurement) and a replicate measurement about 2. The outcome is the true current gap at the recommended setting, with any specification violation counted as failure.

**Family.** The published simulators with per-system parameters around the published values. Every team addition (fan coupling, drift, hatch levels, costs) is declared with its documentary basis, and the family is described as team-extended, not independently authored. Counterfactual-law variants, where text and simulator consistently describe a modified exponent or flipped sign, separate reading from recall of the public documentation. Any training library must be written by someone who has not read the simulator's mechanism code, or be checked by a frozen overlap audit.

**Arms.** The frozen E8 method with all seeds, the strongest numerical controls from E8, and the standing frozen LLM, supervised LLM, ranker successor and target-risk planner. All LLM arms act without reasoning tokens, so the transfer claim is scoped to non-reasoning policies. One predeclared reasoning variant may be registered with E8 as a secondary transfer arm.

**Decision.** E9 runs only if E8 confirms at least H1. Its hypothesis is fixed at registration. It is superiority if the wind-tunnel family was admitted in E5 and both H1 and H2 passed, and non-inferiority within the registered margin otherwise. The result is registered with E8 and reported whatever the outcome. After a failed confirmation E9 does not run.

**Cost.** About 1 to 2 weeks of integration and 15 to 30 local CPU-hours. It uses the `causalchamber` 0.2.8 install or vendored simulator file approved for E5.

#### E10. Optional real-hardware demonstration

Run only after a positive simulator result and Patrick's own decisions on sign-up, terms and spend. The Remote Lab gives exclusive real-time sessions on the physical Wind Tunnel Mk2. Its academic rate of 0.014 EUR per second assumes academic eligibility, and the commercial rate is 0.038 EUR. Sign-up includes 1,000 free credits.

**Terms.** The terms forbid automated account registration, allow rate limits on automated access, make fees non-refundable, grant joint ownership of generated data and allow marketing use of the customer's name. The client is the pre-release `causalchamber` 1.0.0rc1, which needs Python 3.11 or newer in its own environment (the local Python is 3.9).

**Protocol.** Rehearse on a digital twin, randomize the order of policies within blocks to absorb drift, and score episodes intention-to-treat. Report the run descriptively as one apparatus, not a population. A design of 4 policies, 30 episodes and 16 interventions takes about 5 hours of chamber time, roughly 270 EUR, so a cap of about 350 to 400 EUR covers it with contingency.

## Controls every LLM claim needs

| Control | What it rules out | Where |
| --- | --- | --- |
| Strongest planner without side information (target risk with common random numbers, rollout) | A win from planner noise or myopia | All stages |
| Law-aware empirical-Bayes planner without text | A gain from learning the generator's coefficient law rather than the text | Admitted class |
| Amortized numerical student with a training-budget sweep | A learned-policy advantage unrelated to language | All |
| Rule parser and bag-of-words model feeding the planner | Text that is parseable without a language model | Text classes |
| Numerical student on frozen-LLM embeddings | Language features rather than language reasoning | Text classes |
| LLM compiler feeding the planner | A direct policy dominated by a simpler hybrid | Text classes |
| Oracle-informed planner and leakage audit | Gains above the information ceiling | All |
| Frozen LLM, supervised LLM, ranker successor | Required by standing rules | All |
| Text removed and misleading text with recovery delay | Text that is not used, or harm without recovery | Text classes |
| Shuffled feedback on policy input | A policy that ignores measurements | Only where adaptivity value is verified |
| State, menu and identifier permutation | Surface-feature priors | All |
| Argmax-of-prompt rules | Narration of statistics given in the prompt | Any arm that sees derived statistics |
| Compute-matched planner | Extra computation | All |
| Counterfactual-law systems | Recall of public documentation rather than reading | Transfer, and E7 and E8 if C-mis is primary |
| Three to five training seeds | Recipe variability | All trained arms |

Per-action planner scores never appear in the primary LLM input. A hidden-state linear head does not count as an LLM action policy. Every state-level interval or test (E2 edits and probes, E4 ratios, E6 gates) is clustered by world. Every test that can return a null (E8 TOST, E9 non-inferiority) has a registered margin. Simulated training responses, relabeling draws and training compute are reported separately from deployment cost. "Lower experimental cost" refers only to deployment responses, shown next to the planners' zero training cost.

## Gates, kill rules and the paper each branch produces

| Gate | Outcome | Next step | Paper |
| --- | --- | --- | --- |
| E1 instrument | Fails | Fix before scoring anything | None yet |
| E4 numerical ladder | Nothing passes at 2,048 worlds | Direct route closes in the completed class | Compiler or diagnostic |
| Gate A: E5 admission | No class admitted | Advantage program stops. E6 still runs unless Patrick stops GPU spend | **Diagnostic and theory paper**: why the benchmark could not reward adaptivity, how to measure policies, the atlas, and the anatomy of collapse in language and numerical students |
| Gate B: E6 imitation | LLM fails while N2 passes, after one escalation | Only the compiler role continues | Compiler paper if Patrick counts it, otherwise diagnostic |
| E8 confirmation | H1 and H2 pass | E9: superiority if the wind tunnel was admitted, otherwise non-inferiority | **Full paper**: an SCM-conditioned language policy that reads the experimentalist's knowledge, uses measurements, recovers from wrong text and transfers |
| E8 confirmation | Only H1 passes | E9 non-inferiority | Narrowed: text is valuable, LLM not shown better than a rule |
| E8 confirmation | Only H2 passes, or neither | No E9. Report the failure | Diagnostic paper with the failed confirmation |
| E9 transfer | Fails | Report it | The claim is scoped to the synthetic classes |

Every kill rule that depends on an LLM result can only narrow the claim. None adds a benchmark. The completed-class null, every failed gate and the attempt ledger stay in the paper. A controlled negative and diagnostic study has precedent at strong venues (Kristiadi et al., ICML 2024, and Gupta et al., Findings of EMNLP 2025), but it does not meet Patrick's stated submission condition, so it is the fallback and not the target.

**My expectation, labelled as judgment.** The structural-knowledge class is the likeliest to be admitted. Within it, the compiler is likely to match or beat the direct policy, because the published evidence favors language models as sources of priors inside numerical loops. A direct-policy win over equal-information controls is possible but should not be planned on. That is why the compiler arm and the diagnostic paper are built in from the start.

## Calendar and venues

| Phase | Dates | Work | Gate |
| --- | --- | --- | --- |
| Phase 0 (Week 0) | Nov 12 to 19 | Week 0 decisions, evidence preserved, README banner, registration hashed, bench and theory started, pilot if approved. E2 scoring starts only after the hash | Registration hashed |
| Phase 1 (Stages 1 and 2) | Nov 19 to about Dec 11 | E1 (endpoint frozen at its end), E2, E4, E5, training engine for E6, held-out text written | Gate A: about Dec 11, up to about Dec 17 at the upper estimates |
| Phase 2 (Stage 3) | Gate A to about Jan 8 | E6, diagnostic draft in parallel | Gate B, and the ICML go or no-go for the diagnostic paper |
| Phase 3 (Stage 4 development) | Jan 8 to Feb 5 | E7, E8 and E9 registration, independent review | Variant promoted |
| Phase 4 (Stage 4 confirmation) | February to March | E8 | Sealed results, human re-analysis |
| Phase 5 (Stage 5) | March to April | E9, E10 if approved | Transfer reported |
| Writing | April to May | Manuscript, independent review, coauthor approval | Submission decision |

Venue dates (checked November 12): ICML 2027 abstracts are due January 14 and papers January 22, 2027. That fits only the diagnostic paper, and only if it is ready by about January 8. A Gate A later than December 11 compresses that option. UAI 2027 is not announced (estimated February), which is too early for the confirmation. NeurIPS 2027 is not announced (estimated early to mid May) and fits the full program, as does TMLR, which is rolling. CLeaR 2027 closes November 16, 2026, too soon for anything new. The delivery study can go to TMLR separately whatever the LLM outcome.

## Operations

In November the scientific kernel was about 2.4 hours of 39 active hours. Half the failed submissions were custody guards rejecting a valid environment, and one storage decision left a 23.5-hour gap between commits, part of it overnight. Operations, not compute, were the critical path.

- **One in-process bench**, accepted once (E1). It replaces the per-decision subprocess design that made each numerical episode cost about 48 seconds. Cap new non-test harness code (workers, services, validators and guards, not learners or the training engine) at about 3,000 lines before Gate B, and ship every new guard with a fixture built from saved artifacts.
- **Two custody tiers.** Development uses a committed protocol, hash manifests with a provenance record (commit, environment hash, seeds) for every local run, append-only outputs, the attempt ledger and review at gates. Confirmation uses the full November regime: exclusive directories, once-only submission, controller locks, archives and independent acceptance.
- **Local execution.** The Mac (M5 Max, 36 GB) should be able to run Qwen3-0.6B scoring and LoRA training, and Qwen2.5-7B inference should fit in memory once its weights are downloaded. Speed and fidelity are unmeasured until the E2 pilot. Development can run locally. The confirmation runs every LLM arm on one pinned platform, so the comparison is not also a device comparison.
- **Compute in tranches released at gates.** Tranche 0 is local CPU and the E2 pilot. Tranche 1, released at Gate A, covers E6 and is capped at about 15 GPU-hours on a full A100 or H200 under `gpu-long`, or runs locally. Tranche 2, released at Gate B, covers E7 and is capped at about 15 GPU-hours. The confirmation tranche, released at registration, is sized from measured in-process costs. As a planning figure, about 156 systems × 2 replicates × 30 LLM arm-seed combinations × 24 decisions at 0.1 to 0.3 s per decision is roughly 6 to 19 GPU-hours. Eight replicates would make it about 25 to 75. Both scale linearly with the admitted class's episode length, and in-process planners add a few CPU-hours. The 2,742 GPU-slice-seconds left in the November envelope do not cover any tranche. The account's 450,000 SUs cover them easily, and a full A100 bills about twice the 20 GB slice.
- **CURC account.** Use `ucb736_asc2`. The `curc-access` skill still defaults to the expired `ucb736_asc1`.

## Decisions needed from Patrick

| # | Decision | Needed by |
| --- | --- | --- |
| 1 | Commit the preserved session evidence in `results/ace_llm_intervention_offline_diagnosis_20261112/` (Week 0 item 0) | Week 0 |
| 2 | The public-record correction: README banner now, preprint plan with the coauthor after confirming authorship, slide deck and stale PDF, and no talk or submission until it is done | Week 0 (preprint before Gate B) |
| 3 | Which LLM roles count toward conditions 1 and 2, whether a benefit from reading text counts, and whether the E6 old-recipe anchor cell is a control cell of the new design rather than a rerun of the old pipeline | Week 0 |
| 4 | Whether the diagnostic and theory paper may go to ICML 2027 on its own even if a class is admitted and the program continues, and whether it is an acceptable final output if no class is admitted or no benefit confirms | Week 0, with the go or no-go at Gate B |
| 5 | Local model execution on the Mac: the E2 pilot and forensics, then development training | Week 0 |
| 6 | The operating model: the in-process bench, two custody tiers and the harness changes listed below | Week 0 |
| 7 | Compute tranches: about 15 GPU-hours each for Tranches 1 and 2, and a confirmation tranche sized at registration | Week 0, released at gates |
| 8 | New calibration, training, validation, test and confirmation namespaces | Week 0, in the registration |
| 9 | Human authors for the K held-out text families, a parser author who does not develop the LLM, and a named human independent reviewer | Week 0 (text before the E5 parser audit) |
| 10 | Installing `causalchamber` 0.2.8 in an isolated environment, or vendoring its pinned simulator file | Before E5 if C-mis is audited |
| 11 | Use of the cached base Qwen2.5-1.5B for probes and the capacity escalation. Downloading Qwen2.5-7B-Instruct weights for frozen probes, with 7B training only if the pilot shows it is feasible. Any other new weights | Before E2's capacity probes |
| 12 | Whether the paid Remote Lab is compatible with "no paid APIs", the terms above, academic eligibility, a separate Python 3.11 environment for the pre-release client, and a spend cap of about 350 to 400 EUR | Only after a positive E9 |

The harness changes in item 6 are:

- shared random-number namespaces
- more than eight arms and more than 64 worlds
- in-process numerical policies and non-equal query weights
- an informed-prior learner
- a world graph that can differ from the public graph
- nonlinear simulator worlds
- confirmation scope in the bench.

## What not to do

- Do not test for LLM superiority in the completed linear-Gaussian known-noise class. Use it only for imitation and as a negative control on fresh systems.
- Do not rerun or scale the DPO recipe. Do not train with one unshuffled pass at batch size 1, except in the E6 anchor cell if Patrick rules it a control cell. Do not build preference labels from four replicates and a two-standard-error rule.
- Do not run another single-realization 24-system screen, or make realized first-crossing cost the primary endpoint.
- Do not choose endpoints, classes or thresholds from the 24 development outcomes or from those systems' private truth.
- Do not select recipes, training sizes, relabeling rounds or escalations on test worlds.
- Do not present a compiler, hybrid or adjudicator success as meeting condition 1 unless Patrick ruled so beforehand.
- Do not cite any pre-audit number, the README headline, the preprint's 70 to 71% or the stale `paper.pdf`.
- Do not run development iterations through full CURC custody. Do not sign up for, or spend on, the Remote Lab before a positive simulator result and Patrick's ruling.
- Do not let a deadline change a registered sample size, endpoint, gate or failure treatment.
- Do not reuse `ace_experiments.py` or the ACE-Runner pipelines. In this cycle, do not pursue gene-perturbation pools, Tennessee Eastman, the CausalMan probe menu or pyCycle.

## Main risks

- **No class is admitted.** This is the likeliest way the program stops, and the literature predicts it as a live possibility. It yields the diagnostic paper, which does not meet the submission condition.
- **Synthetic text is parser-solvable.** Only human-written held-out text can rescue the text classes, so its authors are on the critical path.
- **Qwen3-0.6B is too small.** The escalation to 1.5B and 7B is fixed in advance by the E2 rule. Local 7B training is unmeasured and its weights are not yet on the Mac.
- **New harness code fails its guards,** as in November. The bench acceptance test and fixture rule are the mitigation.
- **The calendar slips.** The gates are ordered so that each one leaves a finished paper behind it.

## Evidence and provenance

The diagnosis rests on these saved records, all read without modification:

- Comparison data: `results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/` (`cells.csv`, `curves.csv`, `actions.csv`).
- Training: `results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/` (per-step records, reference scores, generations, adapters).
- Training collection: `results/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/`.
- Deployment: `results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services/` (all 1,698 prompts and completions) and `results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/` (numerical decisions with per-action scores).
- Base-weight pin: `results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/inputs/scientific-freeze.json` (`model_files/model.safetensors`).
- Code in `scripts/research/`:
  - `llm_intervention_language_policy.py:132-184` (decoding)
  - `llm_intervention_screen_protocol.py:64-71` and `:124-135` (menu order and teacher rotation)
  - `llm_intervention_training_exposure.py:86-109` (pair selection)
  - `llm_intervention_scm.py:41-130` (learner: known noise and prior 41-46, conjugate update 61-100, composed prediction 121-130).

The session analyses that produced the numbers above are preserved in [`results/ace_llm_intervention_offline_diagnosis_20261112/`](../../../results/ace_llm_intervention_offline_diagnosis_20261112/README.md), not yet committed. They include scripts, notes, independent verification logs and a 16,384-draw reference table for all 4,371 decisions. Its README records what the agents did, the verification scope, known errata in the notes and a SHA-256 manifest. Forty regenerable binary caches were left out, and their hashes are listed there.

Primary literature cited here:

- Kriz et al. 2026 (arXiv 2610.20832) and GO-CBED (TMLR 2026, arXiv 2507.07359)
- CAASL (NeurIPS 2024), Paprika (ICML 2025) and EVOLvE (ICML 2025)
- Schmied et al. (ICLR 2026), Gupta et al. (Findings of EMNLP 2025) and Kristiadi et al. (ICML 2024)
- BoxingGym (arXiv 2501.01540) and LeGIT (arXiv 2503.01139)
- Razin et al. (ICLR 2025) and TWOSOME (ICLR 2024)
- Chaloner and Verdinelli (1995)
- the Causal Chambers paper and documentation.

Recheck every external number against the primary PDF before manuscript use.
