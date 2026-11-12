## What the completed evidence says

Every number in this section was computed from saved files by one analyst and independently recomputed by a second. No figure was refuted. All of it is post hoc analysis of development data. It explains the November result. It does not select a new method from the 24 development outcomes.

### The screen could not have shown an LLM benefit

**The world class left almost no room for adaptive choice.** The systems were three-variable linear-Gaussian SCMs with a known graph and known noise. Each mechanism had a conjugate Bayesian linear learner. In this class the posterior covariance of each mechanism depends mostly on the design, not on the measured responses. Model-based replay from the 24 saved post-initialization states gives the following.

- Adapting after initialization was worth about nothing: −0.7% ± 1.7% in expected final NMSE.
- A design planned once from the post-initialization state matched target-risk planning.
- A tuned schedule that depends only on graph and cost menu was 9% (in-sample) to 21% (leave-one-out) worse. Coverage was 37% worse, 24 observations 75% worse and 24 repetitions of X = −1 113% worse.
- Target risk itself is a noisy 32-draw Monte Carlo heuristic. 74.8% of its decisions had a top-two gap under two Monte Carlo standard errors.
- Two of the eight target queries are duplicates in each graph, so X = −1 carries double weight. X = −1 was the planner's best first action in 11 of 24 systems and the best state-blind first action overall.

**The declared endpoint could not resolve the policies.**

- Ten of 24 systems met the 0.04 target after the shared initialization. The learner's own posterior predicted 9.9.
- The largest possible improvement over target risk was 3.04 cost units per system. The minimum detectable paired difference at 24 systems was 3.7.
- The paired standard deviation was about 6.2 cost units. Detecting a one-unit difference needs about 305 systems at 80% power, or 93 to 172 with eight replicate episodes per system.
- Arms did not share measurement noise. The supervised and DPO arms issued identical actions for their first 10 to 22 acquisitions in every system, yet their capped costs differ in 11 of 24 systems.
- The policy ordering depends on the endpoint. Under sustained attainment menu coverage ranks first (15.13) and DPO ties the frozen LLM (17.00). On the 14 systems not already at target, local information leads target risk on curve area.
- More sensitive endpoints exist. The learner's public posterior risk resolves the large contrasts with 3 to 17 systems. Decision-level regret under a 256-draw target-risk score orders the arms like the headline endpoint (Spearman 0.93) with far more resolution: target risk 0.016, local information 0.154, coverage 0.441, random 0.504, ranker 0.575, frozen 0.655, supervised 0.665, DPO 0.700. This regret is circular for the target-risk arm and is not yet validated against outcomes. An all-observe policy has regret 0.654 yet the fourth-best final NMSE.

### The learned arms were barely trained, and their supervision was broken

**Supervised data.** There were 160 examples: 32 training systems times 5 snapshots. Of these, 32 were terminal "stop" states where stop was the only legal command. The labels rotated across four teachers, and the prompt never said which teacher produced a label. A quarter came from uniform random selection. A quarter came from menu coverage, which ignores the SCM and emits X = −1 at every initial state through an alphabetical tie rule. Labels matched the high-draw planner optimum in only 55 of 128 nonterminal states. Training was one unshuffled pass with batch size 1, learning rate 1e-4 and a rank-8 LoRA on q/v.

**The collapse was visible before deployment.** On its own 128 nonterminal training prompts, the supervised adapter generated X = −1 108 times and the final DPO adapter 123 times. They matched their teacher labels 36 and 32 times. Emitting X = −1 every time also matches 32. Acceptance checked legality only, so the collapse passed. Out of sample, the supervised model never beat a state-blind label-frequency counter (2.37 against 1.93 nats per example).

**Preference labels were mostly noise.** Each of 128 states branched all eight actions with four replicates, giving 3,584 within-state comparisons. The tie rule was a three-degree-of-freedom t-test at α ≈ 0.139. It declared 687 strict pairs, against 476 to 499 expected with no true differences. About 70% of strict labels are what the rule passes under no difference. The labels did carry state-specific signal: they agree with the same state's planner ordering 68.7% of the time, against 50.2% for another state's ordering. Only 119 pairs were used, one per state, once. Collecting the preference feedback cost 9,216 of the 10,240 training cost units.

**DPO did not learn preferences.** Mean loss was 0.7395, above ln 2. Pair agreement at first exposure stayed at 66 of 119. Both chosen and rejected likelihoods fell (4.69 and 6.68 nats on average), which is likelihood displacement. The shifts followed the supervised model's command prior (r = 0.91) rather than the labels. X = −1 rose 0.57 nats while X = +1 fell 3.84, although X = +1 was the most preferred action in the exposed pairs (23 chosen, 14 rejected against 14 and 9 for X = −1). A token-level bias model with no state input reproduces the DPO trajectory (r = 0.80 for margins, 0.93 for final shifts).

**The non-language learner collapsed too.** The numerical ranker trained on the identical 160 records and 119 pairs chose X = +1 435 times and X = −1 141 times. Action identity explains 98.1% of its score variance. Its losses stayed at chance. This points at data and exposure, not specifically at the language model.

**The language interface carries known risk factors.**

- Prompts ran 1,990 to 4,160 tokens. Between 34% and 48% were digits of 16 to 17 digit floats. The raw history grew to 67% of the prompt while the posterior summary shrank to 10%. The 346 tokens nearest the answer were identical in every prompt.
- One byte-identical task description and one fixed menu order appeared everywhere. The frozen model emitted the first menu entry. The trained models emitted the first listed intervention, which also wins every alphabetical tie. Position, identity and tie rule are confounded.
- Actions came from greedy token-by-token decoding over a trie of command strings, with thinking disabled. This is not whole-command argmax. The first branch pools seven `set_*` commands against observation. At late training states the supervised model's likeliest whole command was observation, yet greedy decoding produced X = −1.
- No per-command scores were saved for any LLM arm. They are recoverable by forward passes alone. The pinned Qwen3-0.6B revision, both deployed adapters (hash-matched), the tokenizer and all 1,858 exact prompt token sequences are on the Mac.
- The supervised adapter reads coarse state: budget counter, graph type and history length. No adapter shows evidence of reading the posterior.

### Eleven months of the same pattern

No experiment in this project has shown an LLM proposer or policy beating a non-LLM control. Collapse onto one action recurs from January to November and tracks prompt surface features: the first node, the few-shot example and menu position. Hand-coded caps and fills masked it before. In the preregistered Mission-1 study the LLM arm's selected probes were 56% to 69% hand-coded fills and 9% to 13% echoes of the prompt's own example. The September metric audit voided the original ACE-versus-baseline claims, and ACE with DPO lost to random at 15, 30 and 50 nodes. A public arXiv preprint (2602.02451, v2 of June 18, 2026) and the repository README still advertise a 70 to 71% DPO improvement that the project's own audit invalidated.

### What is solid

- Numerical SCM planning is the only acquisition signal that has replicated: target risk in November, the PEV family and posterior-risk pair selection on two of three synthetic graph families in September.
- The delivery study is a solid learner-side result (confirmation ratio 0.182719, interval 0.101970 to 0.327410, 11 of 12 histories), with one undiagnosed adverse five-node prospective contrast (2.46).
- The execution and custody pipeline works. Its accounting, failure preservation and independent review are a strength a reviewer will notice.
- The forensic diagnosis above is itself a publishable result if it is reproduced under a written protocol.
