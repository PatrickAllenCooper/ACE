# ACE state-of-evidence brief (November 12, 2026)

Every number below was derived from saved files by one analyst and independently re-derived by a second (no refutations). Analyses are read-only, post hoc, development-data only. Detailed notes with file paths live in the sibling directory `../w1/` (one `<label>.md` per topic: mechanism, training, numerics-protocol, results-analysis, results-analysis-verify, training-forensics, training-forensics-verify, deployment-forensics, deployment-forensics-verify, ledger-ops, prior-evidence-core, prior-evidence-portfolio, literature-policy, literature-benchmarks, manuscript).

## 1. The goal and the standing rules

Research question (Patrick's condition for submission): can a language model trained to use an explicit structural causal model (SCM) choose interventions that give an experimentalist a reliable answer at lower experimental cost than strong numerical policies? The LLM must contribute to the decision, not narrate a numerical optimizer's choice.

Submission requires all five: (1) a reproducible LLM action policy with explicit training and deployment mechanism, (2) an independently evaluated benefit over a strong numerical acquisition policy on fresh systems, (3) ablations separating language pretraining, ACE-specific training, causal structure and extra computation, (4) a concrete experimentalist task with an observable useful outcome plus transfer beyond the synthetic training generator, (5) preserved failures, response accounting, independent review.

Standing rules: a negative result must narrow the claim or redirect the method, never trigger a favorable replacement benchmark. No tuning against the 24 development outcomes. Freeze methods, namespaces, endpoints and failure treatment before scoring fresh systems. Keep target-risk planning, the numerical ranker, the frozen LLM and the supervised LLM as controls. No paid APIs. Installs, new model weights and any spend need explicit approval (they may be proposed as explicit asks). Do not rerun completed studies or the old pipelines. Independent systems are the statistical units.

## 2. What the November 24-system screen actually was

Design: 3-variable linear-Gaussian SCMs (chain, fork, collider), known graph, known noise, conjugate Bayesian linear learner with N(0, I) prior, 8-action menu (observe, set X to -1/0/1, set Y to -1/0/1, set X=-1 and Y=1), 8 initial observations plus 24 cost units, target = 8 queries on the noise-free mean of Z, endpoint = first crossing of NMSE 0.04 (capped at 32). Eight policies. Reported means: target risk 11.625, local information 11.958, coverage 12.458, numerical ranker 13.583, frozen LLM 13.625, random 14.250, supervised LLM 14.625, DPO LLM 15.167.

Finding A, the world class had almost no room for any learned policy.
- Sequential adaptation after initialization is worth about nothing (-0.7% +/- 1.7% in expected final NMSE). A design planned once from the post-initialization state equals target risk. A tuned schedule that depends only on graph and cost menu is 9% (in-sample) to 21% (leave-one-out) worse. Coverage is 37% worse, 24 observations 75% worse, 24 x set_x_neg 113% worse.
- Target risk is a noisy 32-draw one-step Monte Carlo heuristic (74.8% of its decisions have top-two gaps under 2 MC standard errors), yet it sits at the state-conditional optimum of this class. Local information is within 2 to 6%.
- The 8 target queries are not distinct, so X = -1 carries double weight. set_x_neg is the best state-blind first action (planner-best in 11 of 24 initial states).
- The code cannot express nonlinear truth, unknown noise, graph uncertainty, informed priors or non-menu actions. It can express graphs to 8 nodes, menus to 32, partial measurement, large cost ratios and varied text with a new protocol module.

Finding B, the declared endpoint could not resolve policies.
- 10 of 24 systems met the target after initialization (9.9 expected from the learner's own posterior). The ceiling above target risk is 3.04 cost units per system and the minimum detectable paired difference at n = 24 is 3.7. Paired SD is about 6.2 cost units, so a 1-unit difference needs roughly 300 systems (93 to 172 with 8 replicates per system).
- Arms did not share measurement noise. The SFT and DPO arms issued identical actions for their first 10 to 22 acquisitions in every world, yet their capped costs differ in 11 of 24 worlds. Error rose after 43.9% of single acquisitions.
- The ordering changes with the endpoint. Under sustained attainment coverage is first and DPO ties the frozen LLM. "DPO worst" is solid only on design quality (expected final NMSE 2.1x target risk, better in 0 of 24).
- A decision-level endpoint works far better: mean relative regret under a 256-draw target-risk score, on each arm's own states, is target risk 0.016, local information 0.154, coverage 0.441, random 0.504, ranker 0.575, frozen 0.655, SFT 0.665, DPO 0.700. Public posterior risk needs 3 to 17 systems for contrasts that first-crossing cost needs 67 to 340 for.

Finding C, the learned arms were barely trained and their supervision was broken.
- Supervised data: 32 training worlds x 5 snapshots = 160 examples (128 nonterminal, 32 terminal "stop" where stop is the only legal command). Labels rotate over four teachers with teacher identity hidden: uniform random (pure noise), menu coverage (ignores the SCM, emits set_x_neg at every initial state through an alphabetical tie-break), local information, target risk. Labels equal the high-draw planner optimum in only 55 of 128 states. One unshuffled pass, batch size 1, lr 1e-4, rank-8 LoRA on q/v, 160 updates.
- The supervised model never beat a state-blind label-frequency counter (2.37 versus 1.93 nats per example out of sample). On its own 128 training states it generates set_x_neg 108 times, and the final DPO adapter 123 times. The collapse was visible in files written by the training job and passed acceptance, because the gate checked legality only.
- Preference data: 3,584 within-state comparisons from 4 replicates each, with a tie rule equal to a 3-df t-test at alpha 0.139. 687 strict, of which about 70% are what the rule passes under no difference. Strict labels agree with the planner's ordering 68.7% of the time. Only 119 pairs were used (one per record, one pass). Collection cost 9,216 of 10,240 units.
- DPO did not learn preferences: mean loss 0.7395 (above ln 2), pair agreement unchanged at 66 of 119, both chosen and rejected likelihoods fell (likelihood displacement), and shifts follow the SFT command prior (r = 0.91). A token-level bias model with no state input reproduces the DPO trajectory (r = 0.80 to 0.93).
- The exposure-matched numerical ranker also collapsed (two actions only, 98.1% of score variance is action identity, losses at chance). So the failure is not specific to the language model.
- The whole LLM training run took 291.5 seconds on one 20 GB GPU slice.

Finding D, the language interface has known risk factors.
- Prompts are 1,990 to 4,160 tokens. 34 to 48% of tokens are digits (16 to 17 digit floats, one digit per token). Raw history grows to 67% of the prompt and the posterior summary shrinks to 10%. The 346 tokens nearest the answer are identical in every prompt. One byte-identical task payload and one fixed menu order everywhere.
- Actions are chosen by greedy token-by-token decoding over a trie of command strings, with thinking disabled. This is not whole-command argmax: at late-stage states the supervised model's likeliest whole command was observe_all, but the "set" branch pools seven commands and wins the first branch.
- The frozen model emits the first menu entry. The trained models emit the first listed intervention, which also wins every alphabetical tie-break. Position, identity and tie rule are confounded.
- No per-command scores were saved for any LLM arm. They are recoverable by forward passes only: base weights, both deployed adapters (hash-matched), the tokenizer and all 1,858 exact prompt token sequences are on the Mac.
- The supervised adapter does read coarse state (budget counter, graph type, history length). There is no sign that any adapter reads the posterior.

Finding E, history. No experiment in this project has shown an LLM proposer or policy beating a non-LLM control. Collapse onto one action recurs over eleven months and tracks prompt surface features (first node, few-shot example, menu position). It was previously masked by hand-coded caps and fills. In the preregistered Mission-1 study the "LLM arm" was 56 to 69% hand-coded fills and 9 to 13% echoes of the prompt's own example. The September metric audit voided the original ACE-versus-baseline claims, and DPO hurt at 15 to 50 nodes. A public arXiv preprint and the repository README still advertise a 70% DPO gain that the project's own audit invalidated.

What is solid: numerical SCM planning. Target-risk planning (November), the PEV family and posterior-risk pair selection (September, two of three synthetic graph families) are the only acquisition signals that replicated. The delivery (post-collection refit) study is a solid learner-side result with one undiagnosed adverse 5-node contrast.

## 3. Literature (primary sources checked on the web, not a systematic review)

- Prior art for trained LLM information-gathering policies: Paprika (ICML 2025, SFT then SFT plus multi-turn DPO, 8B/12B models, 17,181 trajectories and 5,260 pairs), Mazzaccara et al. 2024 (DPO on information-gain preferences), ASIG (GRPO), and Kriz et al. (arXiv October 2026), who train the same base model Qwen3-0.6B with on-policy GRPO and LoRA (250 updates, 16 states x 16 rollouts, entropy bonus, permuted identifiers) to a state-dependent non-myopic test-selection policy (75.0% versus 30.2%).
- Numerical amortized intervention policies already cover goal-oriented causal design: GO-CBED (TMLR 2026, transformer policy, known-graph linear-Gaussian example, code public), CAASL (NeurIPS 2024), Deep Adaptive Design.
- Careful ablations agree that LLMs supply priors and proposals and rarely adapt to measurements: Gupta et al. 2025 (shuffled feedback barely changes LLM agents, LinUCB on the same embeddings wins 93 versus 33), BoxingGym (priors do not reliably help), EVOLvE (cloning a numerical teacher from raw history gives 28.6, from computed per-action statistics 89.3, teacher 96.4), Schmied et al. (small models show frequency bias and a knowing-doing gap that RL fine-tuning halves), PhysGym (success falls from 66% to 31% as priors are removed).
- DPO misbehaves on short similar completions (likelihood displacement). Unnormalized joint token probability over multi-token action strings favors shorter strings (TWOSOME).
- Comparable papers use 3 to 10 environments or tens of tasks, 5 to 10 seeds, fixed-budget endpoints and rarely formal tests. Cost-to-target with censoring is rare.
- The regime of the completed screen (known graph, linear, known noise, small menu, anonymous variables) is where the literature predicts no LLM advantage.

## 4. Assets and environments

- Reusable generic layer: world/executor/charging, conjugate SCM learner, target-risk and local-information planners, paired episodes, sealed evaluation, cost curves, reporting (descriptive only, limits of 8 nodes, 32 actions, 64 queries, 64 worlds).
- Free data: 697 saved decision envelopes from the 32 training worlds, 4,371 saved deployment decisions with exact states, saved per-action planner scores for numerical arms. Planner relabeling of any state costs zero environment responses.
- Causal Chambers: self-service Remote Lab with a real-time Python API to physical wind and light tunnels (1 credit = 1 second, academic 0.014 EUR per credit, 1,000 free credits, sign-up and terms need a human). The pip package `causalchamber` (MIT, not installed) ships numpy-only wind-tunnel simulators with documented misspecification (affinity-law fan speed, cubic current, squared-speed pressure, linear hatch term). Graph edges validated by randomized experiments. Prose variable descriptions exist. Rough arithmetic: 4 policies x 30 episodes x 16 interventions is about 5.3 hours and 270 EUR.
- Compressor: pyCycle (Apache-2.0, pure Python, ships compressor maps, not prototyped) or ThermoPower through an OpenModelica container. Neither is integrated. The talk narrative uses a compressor rig (drive command to shaft speed to pressure rise).
- ACE-Runner (sibling repository) has five executable engineering simulators with do-semantics on internal nodes.
- BoxingGym adapters partly exist (no costs, not do-operations). bnlearn Gaussian networks are an option for supplied-graph systems.
- Local machine: Apple M5 Max, 36 GB, Python 3.9 with torch 2.8 and transformers 4.57. Local HF cache holds Qwen3-0.6B, Qwen2.5-1.5B(-Instruct) and Qwen2.5-7B-Instruct.

## 5. Operations

- The November cycle: 49 Slurm jobs (21 failed), 11.2 CPU core-hours and 1.24 GPU slice-hours, about 39 active hours over 8 days. The scientific kernel was about 2.4 hours. Half the failed submissions were the custody harness rejecting a valid environment.
- Remaining approved envelope: 13,735 CPU-seconds and 2,742 GPU-slice-seconds. A retrain plus a fresh 24 to 48 system screen needs roughly 8 to 15 core-hours and 0.8 to 1.3 slice-hours, so it needs a new explicit ceiling from Patrick. A 100 to 150 system confirmation needs roughly 22 to 35 core-hours, 1.9 to 2.9 slice-hours and a custody redesign. The account itself holds up to 450,000 SUs.
- Unit costs: LLM decision 0.4 to 0.7 s, SFT update 0.62 s, DPO update 1.07 s, numerical episode about 48 s wall (one process per decision, an implementation artefact).
- An offline diagnosis on saved artifacts needs no CURC, no locks and no envelope. Local forward-only inference is a new model execution, so it needs a written protocol and a go-ahead.

## 6. Venues and open human decisions

- ICML 2027: abstract January 14, paper January 22, 2027. UAI 2027 not announced (estimate February). NeurIPS 2027 not announced (estimate May). TMLR rolling. CLeaR 2027 closes November 16, 2026. No venue or date is recorded for the broader paper.
- Open decisions that only Patrick can make: target venue and date, a new compute ceiling, approval for local inference, pip installs (causalchamber, pyCycle), larger model weights, Remote Lab sign-up and spend, how to reconcile the public preprint and README, and whether a rigorous diagnostic or negative paper is acceptable if no LLM benefit emerges.
