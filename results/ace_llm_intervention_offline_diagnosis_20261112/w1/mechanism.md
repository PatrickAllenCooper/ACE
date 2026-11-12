# mechanism: how the LLM chooser sees the state and how its action is selected

Prepared 2026-11-12 by the `mechanism` agent for the ACE experimental-agenda workflow. Read-only analysis of `/Users/pat/code/ACE`. Nothing in the repository, ACE-Runner or ACE_Study_Results was created or changed. No model weights were loaded, no inference, training, SLURM or SSH.

Labels used below: **VERIFIED** = checked directly in a file or by a computation on saved files. **INFERENCE** = my reasoning beyond what files show.

Abbreviations: REPO = `/Users/pat/code/ACE`. GPU02 = `REPO/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch`. FIT = `REPO/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training`. CPU02 = `REPO/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/campaign`. SCR = `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/mechanism` (my scratch scripts and outputs).

## 0. What I used, and how far it can be trusted

- Code read in `REPO/scripts/research/`: contract, task, world, scm, episode, grammar, action_scoring, canonical_scoring, language_policy, persistent_policy, policy_cli, precision, screen_prompt_fixture, screen_realization, model_service, screen_policy, plus screen_protocol, learning, numerical_policy, target_risk, training_engine, scientific_campaign. VERIFIED: 14 of these (contract, task, scm, episode, grammar, language_policy, model_service, learning, canonical_scoring, training_engine, persistent_policy, screen_protocol, numerical_policy, target_risk) are sha256-identical to the frozen executed copies in `GPU02/inputs/`. My code reading therefore describes the executed bytes.
- Saved deployment records of the three LLM arms are retained locally: `GPU02/gpu/services/{frozen_llm,sft_llm,dpo_llm}/worker/generation-NNNN.json` (576, 546, 576 files). Each contains the exact `prompt_ids` and `sequence_ids`.
- VERIFIED: the Qwen3-0.6B tokenizer files in the local HF cache (`/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/`) match the deployment freeze pins byte for byte (`GPU02/inputs/dpo_llm.freeze.json`, `model_files`; tokenizer.json sha256 `aeb13307…dae4`). I decoded saved token IDs with this vocabulary only (`tokenizers.Tokenizer.from_file`, no network, no weights).
- VERIFIED: for all 1,698 saved deployment prompts the decoded user text hashes to the recorded `prompt_sha256`, the embedded envelope re-serializes canonically and hashes to the recorded `envelope_sha256`, and re-tokenizing my reconstruction gives exactly the saved `prompt_ids` (0 mismatches; `SCR/build_table.py`, `SCR/composition.py`). The same holds for the 160 training prompts in `GPU02/inputs/prepared.json` (160/160).
- The frozen target-risk planner (`GPU02/inputs/llm_intervention_target_risk.py`, copied to `SCR/frozen_src/`) was called as a pure function on the saved public states, with my own analysis seed 20261112 and the protocol's 32 draws. That is an offline computation on saved inputs, not a rerun of any arm.

## 1. (a) One complete real chooser prompt

Source: `GPU02/gpu/services/dpo_llm/worker/generation-0000.json` (DPO arm, service ordinal 0, first decision of development world 202611070100: chain graph, uniform cost menu, noise variance 0.25). The same envelope (`6c1798340be8…4c7c`) was the first state of all eight arms. Exact text files: `SCR/dpo_llm-0000.templated.txt` (with chat template) and `SCR/dpo_llm-0000.user.txt` (user content only).

Size (VERIFIED): user content 5,187 characters (5,187 UTF-8 bytes), 5,256 characters with the chat template, **2,012 tokens** (`prompt_tokens` recorded in the file). The model then emitted 9 tokens: `{"action_id":"set_x_neg"}` plus EOS.

Verbatim (the chat-template special tokens are literal):

```
<|im_start|>user
Choose the next experimental intervention using this public SCM, paid history and prediction task. The task queries and weights describe where accurate predictions are useful. They contain no observed answers. Predict the noise-disabled structural response. Error is squared error divided by scale squared. The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. Every environment response costs the listed units. Return exactly one JSON object containing an affordable action_id from state.menu, or {"stop":true}. Do not invent measurements or actions.
{"state":{"domain_notes":"Synthetic dimensionless variables. The supplied graph and hard interventions are known. The learner uses public known noise variances and zero-mean independent Gaussian parameter priors. The task tolerance is a goal, not evidence of attainment.","graph":[{"intervention_bounds":[-2.0,2.0],"name":"X","parents":[],"unit":"dimensionless"},{"intervention_bounds":[-2.0,2.0],"name":"Y","parents":["X"],"unit":"dimensionless"},{"intervention_bounds":null,"name":"Z","parents":["Y"],"unit":"dimensionless"}],"history":[{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":0,"targets":{},"values":{"X":1.2873731568992153,"Y":-0.6175376044829376,"Z":-0.270917213687895}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":1,"targets":{},"values":{"X":0.7160899140899049,"Y":-0.517849873854284,"Z":-1.37072559316845}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":2,"targets":{},"values":{"X":0.6291940211900355,"Y":-0.6632290952376003,"Z":-0.8544098851767415}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":3,"targets":{},"values":{"X":1.2216524991767816,"Y":0.9917474281085259,"Z":-3.0921736571241283}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":4,"targets":{},"values":{"X":0.915123226113694,"Y":0.060760665639840866,"Z":-1.2270375160493212}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":5,"targets":{},"values":{"X":1.404604457554113,"Y":-0.6614111288370578,"Z":-1.4814004641623564}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":6,"targets":{},"values":{"X":1.0365017106371108,"Y":-0.24262021606467354,"Z":-0.9668032267889245}},{"action_id":"observe_all","cost_units":1,"kind":"observe","request_id":7,"targets":{},"values":{"X":1.3823778070718178,"Y":0.789907760992661,"Z":-2.9622624638678667}}],"menu":[{"action_id":"observe_all","cost_units":1,"kind":"observe","measure":["X","Y","Z"],"targets":[]},{"action_id":"set_x_neg","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["X",-1.0]]},{"action_id":"set_x_zero","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["X",0.0]]},{"action_id":"set_x_pos","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["X",1.0]]},{"action_id":"set_y_neg","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["Y",-1.0]]},{"action_id":"set_y_zero","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["Y",0.0]]},{"action_id":"set_y_pos","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["Y",1.0]]},{"action_id":"set_xy","cost_units":1,"kind":"intervene","measure":["X","Y","Z"],"targets":[["X",-1.0],["Y",1.0]]}],"objective":"Predict the noise-disabled structural mean of Z under the supplied public query distribution.","remaining_units":24,"requests":8,"returned_responses":8,"scm":[{"eligible_rows":8,"node":"X","noise_variance":0.25,"parameter_covariance":[[0.030303030303030304]],"parameter_mean":[1.0415656718463846],"terms":["intercept"]},{"eligible_rows":8,"node":"Y","noise_variance":0.25,"parameter_covariance":[[0.2653213106466827,-0.2256394260066532],[-0.2256394260066532,0.21663485280449185]],"parameter_mean":[-0.5478667338366421,0.42589362595851954],"terms":["intercept","X"]},{"eligible_rows":8,"node":"Z","noise_variance":0.25,"parameter_covariance":[[0.031112650920001025,0.007764614191435232],[0.007764614191435232,0.07446603048155506]],"parameter_mean":[-1.609175737627502,-1.220565774823312],"terms":["intercept","Y"]}],"spent_units":8,"terminal":false,"version":"ace-intervention-contract-v1"},"task":{"normalized_mse_tolerance":0.04,"queries":[{"action_id":"q0","cost_units":1,"kind":"observe","measure":["Z"],"targets":[]},{"action_id":"q1","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["X",-1.0]]},{"action_id":"q2","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["X",0.0]]},{"action_id":"q3","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["X",1.0]]},{"action_id":"q4","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["Y",-1.0]]},{"action_id":"q5","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["Y",0.0]]},{"action_id":"q6","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["Y",1.0]]},{"action_id":"q7","cost_units":1,"kind":"intervene","measure":["Z"],"targets":[["X",-1.0],["Y",1.0]]}],"scale":2.0,"target":"Z","weights":[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]},"version":"ace-public-prediction-task-v1"}
Decision:
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

### Which spans vary

Class legend: **C** constant in every prompt of the study (all 160 training prompts and all 1,698 deployment prompts), **W** varies only between systems (constant inside an episode), **S** varies from step to step.

```
<|im_start|>user\n                                                        C  chat template (3 tokens)
Choose the next experimental intervention … measurements or actions.\n   C  instruction (108 tokens)
{"state":{"domain_notes":"Synthetic dimensionless … attainment.",         C
 "graph":[3 x {"intervention_bounds":C,"name":C,"parents":W,"unit":C}],    C except the three parents lists (chain/fork/collider)
 "history":[N x {"action_id":S,"cost_units":S,"kind":S,"request_id":S,
                 "targets":S,"values":{"X":S,"Y":S,"Z":S}}],               S  N = 8..31 rows, one new row per step, oldest first
 "menu":[8 x {"action_id":C,"cost_units":W,"kind":C,"measure":C,"targets":C}],   only the 8 cost integers vary (two menus)
 "objective":"Predict the noise-disabled structural mean of Z …",          C
 "remaining_units":S,"requests":S,"returned_responses":S,                  S  counters
 "scm":[3 x {"eligible_rows":S,"node":C,"noise_variance":W,
             "parameter_covariance":S,"parameter_mean":S,"terms":W}],      S  posterior numbers, W noise and term names
 "spent_units":S,"terminal":false,"version":"ace-intervention-contract-v1"},
 "task":{"normalized_mse_tolerance":0.04,"queries":[q0..q7],"scale":2.0,
         "target":"Z","weights":[0.125 x 8]},                              C  one identical task payload everywhere
 "version":"ace-public-prediction-task-v1"}
\nDecision:\n                                                             C
<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n                C  chat template (9 tokens), thinking disabled
```

VERIFIED facts about constancy: one distinct task payload (`task_sha256` `8f22ac43…0f71`), one objective string and one domain-notes string across all 160 training records and all 1,698 deployment prompts. Target queries, weights, scale and tolerance therefore never vary in this study. Costs vary only as two menus (Y and joint interventions cost 1 or 2). The matched/shifted regime is private and is not in the prompt.

### Token composition (VERIFIED, `SCR/composition.py`)

| Prompt | Tokens | history | scm | task | menu | graph | instructions | counters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First decision above (8 rows) | 2,012 | 718 (35.7%) | 378 (18.8%) | 332 (16.5%) | 288 (14.3%) | 120 (6.0%) | 108 (5.4%) | 41 (2.0%) |
| Same episode, last decision (31 rows) | 3,927 | 2,622 (66.8%) | 387 (9.9%) | 332 (8.5%) | 288 (7.3%) | 120 (3.1%) | 108 (2.8%) | 43 (1.1%) |
| Longest in campaign (frozen ordinal 143, collider, heterogeneous, 31 rows) | 4,160 | 2,811 (67.6%) | 430 (10.3%) | 332 (8.0%) | 288 (6.9%) | 121 | 108 | 43 |

Fraction of the prompt that is state-dependent, measured in tokens:

| Class | First decision | Last decision | All 1,698 prompts (min / mean / max) |
| --- | ---: | ---: | ---: |
| Constant boilerplate (C, including 12 chat tokens) | 48.3% | 24.8% | 23.4% / 33.8% / 48.8% |
| System-only (W) | 1.9% | 1.0% | 0.8% / 1.3% / 1.9% |
| Step-varying values (S) | 38.7% | 52.8% | 38.0% / 47.7% / 54.9% |
| of which numeric literals (measured values, posterior means and covariances) | 35.1% | 40.9% | 34.4% / 40.2% / 48.1% |
| Repeated key boilerplate inside history rows | 11.1% | 21.5% | 10.8% / 17.2% / 21.7% |

In characters the first-decision prompt is 66.5% constant, 1.5% system-only, 18.5% step-varying values (14.2% numeric) and 13.6% repeated history keys. The posterior (`scm`) block is 361 to 436 tokens (9.2% to 20.8% of the prompt). In the first-decision prompt it starts at token 1,269 and the constant task block starts at token 1,666, so **the 346 tokens nearest the answer position are identical in every prompt of the study**.

## 2. (b) Number serialization, history growth, truncation, context

- VERIFIED (code): the prompt is `task_prompt(envelope)` = fixed instruction + `canonical(envelope)` + `"\nDecision:\n"` (`llm_intervention_task.py:144-152`). `canonical` is `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)` (`llm_intervention_contract.py:23-24`). The state comes from `ChargedExecutor.state()` (`llm_intervention_contract.py:326-342`) and the SCM summaries from `LinearGaussianSCM.summaries()` (`llm_intervention_scm.py:102-110`).
- Precision and notation: Python shortest round-trip `repr` of binary64, no rounding. VERIFIED on the 1,698 saved envelopes: 191,836 float literals, of which 110,602 carry 12 to 17 significant digits (53,106 with 16 and 50,286 with 17). 53 use exponent notation, for example `4.5675812741832614e-05`. Clamp values, bounds, weights and noise appear as `-1.0`, `2.0`, `0.125`, `0.25`. Counters and costs are integers.
- Tokenization of numbers: VERIFIED one token per digit. `1.2873731568992153` is 18 tokens, `-0.6175376044829376` is 19, `0.030303030303030304` is 20, an exponent-form value is 21 to 22.
- Ordering: keys sorted alphabetically at every level, so the order is domain_notes, graph, history, menu, objective, remaining_units, requests, returned_responses, scm, spent_units, terminal, version, then task, then version. Arrays keep their construction order: graph X, Y, Z. History chronological, oldest first. Menu fixed as observe_all, set_x_neg, set_x_zero, set_x_pos, set_y_neg, set_y_zero, set_y_pos, set_xy. Within each SCM head the full covariance matrix (row-major, both symmetric halves) precedes the mean, and the `terms` list that names the coefficients comes after the numbers.
- History growth: yes. One complete row per accepted response, never summarized. VERIFIED 77 to 88 tokens per added row (DPO episode 0 grows 2,012 → 3,927 tokens over 23 added rows). The policy is called with 8 to 31 rows.
- Truncation: none on either side. `prepare_task_tokens` raises if `len(prompt_ids)+max_new_tokens > max_tokens` (`llm_intervention_language_policy.py:87-111`, check at 105-106). `validate_state` raises above 80 history rows (`llm_intervention_task.py:21,73-76`). The transport raises above 1 MiB (`llm_intervention_persistent_policy.py:407-408`).
- Context: cap 8,192 tokens with 32 reserved for generation (`llm_intervention_model_service.py:52-57`). VERIFIED maximum actually used at deployment: 4,160 prompt tokens + 9 generated. Minimum 1,990. Training prompts: 1,998 to 4,199 tokens. Model limit `max_position_embeddings` 40,960. The earlier CPU capacity qualification on 120 artificial long-spelling fixtures recorded `max_prompt_tokens` 4,744 (`REPO/results/ace_llm_intervention_screen_prompt_cpu_20261108/acceptance01/acceptance.json`).

## 3. (c) Exactly how the action is chosen

**It is constrained greedy generation, not scoring of each legal command.** VERIFIED in `llm_intervention_language_policy.py:132-184`: `generate_task_decision` calls `model.generate` with `GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=32, use_cache=True)`, `use_model_defaults=False`, and `prefix_allowed_tokens_fn` bound to a finite grammar (`llm_intervention_grammar.py:40-51,60-83`). At each step only tokens that extend the current suffix toward a legal command are allowed, and the highest-logit allowed token is taken. The stored model defaults (`do_sample: true, temperature 0.6, top_k 20, top_p 0.95`) are explicitly not used. The service recipe records `"decoding": "greedy_single_prompt_fresh_cache"`, `"thinking": false`.

- Sum or mean, length normalization: not applicable at deployment, because no command likelihood is ever computed. The grammar guidance states the consequence itself: "Greedy token choices need not maximize the probability of the whole command" (`REPO/docs/development/guidance/ace_llm_intervention_grammar_2026-11-02.md:19`).
- Training objectives, for contrast (VERIFIED): supervised loss = minus the sum of token log-probabilities of the command plus EOS, divided by the number of scored tokens, so a per-example token mean (`llm_intervention_canonical_scoring.py:73-76,99-102`). DPO uses unnormalized sequence sums for chosen and rejected against a fixed post-SFT reference, `-logsigmoid(beta*((pc-rc)-(pr-rr)))` with beta 0.1 (`llm_intervention_learning.py:94-103`, `llm_intervention_training_engine.py:428-430`). Both use the full-vocabulary softmax, not a grammar-renormalized one. Prompt tokens are masked from the loss.
- Chat template: applied. `tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)` (`llm_intervention_language_policy.py:96-97`). No system message. VERIFIED in saved tokens: prefix `<|im_start|>user\n` (151644, 872, 198), suffix `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271). Training prompts use the identical wrapper (160/160).
- Thinking mode: disabled by the pre-filled empty think block. There is no scratchpad of any kind. The second generated token already commits to act or stop.
- Sampling: none. Strict deterministic algorithms, FP32 output projection, fresh KV cache per request, weights frozen and checked unchanged before and after each request (`llm_intervention_model_service.py:299-379`).
- Ties: no explicit rule. `argmax` over the allowed-token logits, which on an exact FP32 tie returns the lowest token id. INFERENCE: exact ties essentially never occur.
- Legal set: every affordable menu action plus stop, sorted by action id (`llm_intervention_grammar.py:73-75`). VERIFIED 9 legal commands in 564 of 576 DPO and frozen decisions, 5 legal commands in the other 12 (heterogeneous menu with one unit left).

Exact command strings and token sequences (VERIFIED from `GPU02/inputs/prepared.json` → `training_plan.records[*].commands`, identical in 128 nonterminal records, reproduced by the local tokenizer). EOS is `<|im_end|>` = 151645.

| Command | Tokens incl. EOS | Token ids | Pieces |
| --- | ---: | --- | --- |
| `{"action_id":"observe_all"}` | 8 | 4913, 1311, 842, 3252, 48033, 5705, 9207, 151645 | `{"` `action` `_id` `":"` `observe` `_all` `"}` EOS |
| `{"action_id":"set_x_neg"}` | 9 | 4913, 1311, 842, 3252, 746, 3212, 28209, 9207, 151645 | `{"` `action` `_id` `":"` `set` `_x` `_neg` `"}` EOS |
| `{"action_id":"set_x_zero"}` | 9 | …, 746, 3212, 19359, 9207, 151645 | … `set` `_x` `_zero` `"}` EOS |
| `{"action_id":"set_x_pos"}` | 9 | …, 746, 3212, 6479, 9207, 151645 | … `set` `_x` `_pos` `"}` EOS |
| `{"action_id":"set_y_neg"}` | 9 | …, 746, 4178, 28209, 9207, 151645 | … `set` `_y` `_neg` `"}` EOS |
| `{"action_id":"set_y_zero"}` | 9 | …, 746, 4178, 19359, 9207, 151645 | … `set` `_y` `_zero` `"}` EOS |
| `{"action_id":"set_y_pos"}` | 9 | …, 746, 4178, 6479, 9207, 151645 | … `set` `_y` `_pos` `"}` EOS |
| `{"action_id":"set_xy"}` | 8 | 4913, 1311, 842, 3252, 746, 40920, 9207, 151645 | … `set` `_xy` `"}` EOS |
| `{"stop":true}` | 6 | 4913, 9495, 788, 1866, 92, 151645 | `{"` `stop` `":` `true` `}` EOS |

Shared prefixes: all nine share `{"`. The eight acquisition commands share `{"` `action` `_id` `":"`. The seven `set_*` commands also share `set`. Tokens `_neg`, `_zero`, `_pos` are the same ids after `_x` and after `_y`.

The decision is therefore at most four sequential greedy choices. Every other generated token is forced (one allowed token):

- B1, generated token 2: `action` versus `stop`.
- B2, generated token 5: `observe` versus `set`. `set` carries the mass of seven commands.
- B3, generated token 6: `_x` versus `_y` versus `_xy`.
- B4, generated token 7: `_neg` versus `_zero` versus `_pos`.

## 4. (d) Is the information needed to choose well present and usable

Present (VERIFIED in the prompt above): target `Z`, the eight target queries and their equal weights, scale and tolerance. Each action's `cost_units`. `remaining_units`. Graph. Full history. For each mechanism the posterior mean, full covariance, known noise variance and `eligible_rows`.

Sufficient in principle: yes. VERIFIED: the target-risk planner receives exactly this envelope and nothing else (`llm_intervention_screen_policy.py:140-155` → `choose_target_risk(state, **task.risk_arguments(), seed, draws)`, `llm_intervention_target_risk.py:91-136`). I ran that frozen function on the decoded contents of the LLM prompts (`SCR/planner_on_llm_states.py`, results in `SCR/planner_*.pkl` and `SCR/llm_decisions.csv`):

| Arm (its own visited states) | Planner-best action over those states | LLM action = planner best | Planner one-step gain of the LLM action / of the best | LLM action below best by more than 2 MC SE |
| --- | --- | ---: | ---: | ---: |
| DPO, 576 states | set_x_pos 173, observe_all 171, set_y_pos 119, set_x_neg 39, set_xy 35, set_y_neg 26, set_y_zero 13 | 39 (6.8%) | 0.00265 / 0.01146 = 0.23 | 515 of 576 |
| SFT, 523 acquisitions | observe_all 165, set_x_pos 138, set_y_pos 87, set_xy 83, set_x_neg 38, set_y_neg 33, set_y_zero 2 | 43 (8.2%) | 0.00302 / 0.01148 = 0.26 | 462 of 523 |
| Frozen, 576 states | set_x_neg 204, set_x_pos 158, set_xy 126, observe_all 49, set_y_pos 33, set_y_neg 6 | 49 (8.5%) | 0.00429 / 0.01637 = 0.26 | 491 of 576 |

For DPO the agreement is 27.1% in the first six acquisition steps and 0% afterwards. `set_x_neg` is the planner's best first action in 11 of the 24 initial states, and it ranks 7th or 8th among the feasible actions in 393 of the 576 DPO states. Caveat: the gain is the planner's own working-model, one-step, Monte Carlo estimate, not realized utility.

Usable form for a 0.6B model in one forward pass: INFERENCE, doubtful. Nothing decision-ready is stated. There is no per-action expected benefit, no predictive variance at the queries, no current risk estimate relative to the tolerance, no per-action count of past use. "Which mechanisms are still uncertain" exists only as raw covariance matrices and `eligible_rows`. Obtaining the planner's ranking requires propagating covariances through the composed mechanisms for eight queries and a rank-one posterior update per candidate, from digit-tokenized 16 to 17 digit numbers, with thinking disabled. The instruction also never states the selection criterion (expected reduction of target error per cost unit). It only defines the error.

A simpler usable signal is present: the history lists every past `action_id`. Menu coverage, which uses only those counts, reached 12.458 mean capped cost, better than all three LLM arms (13.625, 14.625, 15.167).

## 5. What the three LLM arms actually did (VERIFIED, `SCR/analyze1.py`, `SCR/analyze2.py`)

- Frozen: observe_all 574, set_x_zero 2 (both in episode 15, fork, heterogeneous, noise 0.25, decisions 7 and 10).
- SFT: set_x_neg 481, set_x_pos 23, set_y_neg 11, observe_all 8, stop 23. By decision index over 24 episodes: decisions 0 to 9 set_x_neg 24/24 every time. Decisions 10 to 13 are where set_x_pos and set_y_neg appear (decision 12: set_x_neg 12, set_x_pos 8, set_y_neg 4). observe_all appears at decisions 18 to 21, only in colliders. Stop at decisions 21 to 23 with 3, 2 or 1 units left (5, 14, 4 episodes).
- DPO: set_x_neg 576 of 576.
- Each arm received 576 (546 for SFT) distinct envelopes with growing histories. The constant output is not a stale-prompt artifact.

The SFT pattern tracks episode progress and weakly the graph. The supervised labels show the same progress profile (section 7).

## 6. (f) Are per-decision score vectors saved

- **LLM arms: no.** VERIFIED by the service source (`llm_intervention_model_service.py:347-378`) and by the union of keys over all saved files. Per decision the worker saved `request-NNNN.json` {envelope_sha256, identity, ordinal}, `raw-generation-NNNN.json` {shape, dtype, token_prefix, raw_sha256, …}, `generated-NNNN.unaccepted.json` and `generation-NNNN.json` {completion, prompt_ids, sequence_ids, prompt_sha256, envelope_sha256, task_sha256, prompt_tokens, legal_commands, longest_command_tokens, max_tokens, max_new_tokens, enable_thinking, elapsed_seconds, cuda_interval_ms, peak_gpu_allocated_bytes, …}. No logits, log-probabilities or scores. Path: `GPU02/gpu/services/{frozen_llm,sft_llm,dpo_llm}/worker/`.
- The only saved LLM likelihoods are from training: `FIT/reference-scores.json`, schema `{"scores": {"[\"<record_id>\",<pair_index>]": [chosen_logprob, rejected_logprob]}, "decoder_forwards": 238, "unique_pair_score_calls": 119, "scoring_layout": "ordered_unpadded_singletons", "signature": …}`. These are post-SFT sequence log-probabilities (sum over command tokens plus EOS) for the 119 scheduled DPO pairs. `FIT/step-NNNNN.{loss,complete}.json` hold all 279 step losses and gradient norms.
- Numerical arms: yes, locally. `CPU02/policy-0{0..4}/world-NN/episode/decision-NNNN.process.json`, field `stderr`, last JSON line. target_risk (policy-03): `planning_metadata.{risk_before, risk_after, risk_reduction_per_cost, monte_carlo_standard_errors}` per action. local_information (policy-02): `planning_metadata.scores`. learned_ranker (policy-04): `scores` (list of 9, stop last) and `action_ids`. The matching public envelope is `decision-NNNN.input.json`.
- Everything needed to compute the missing LLM score vectors offline is on this Mac (VERIFIED by sha256, nothing loaded): base weights `model.safetensors` 1,503,300,328 bytes sha `f47f7117…874b` equal to the freeze pin, the deployed SFT adapter `FIT/sft.adapter.pt` (`583b1659…3611`), the deployed DPO adapter `FIT/final.adapter.pt` (`6f540dfe…d623`), the pinned tokenizer, the token ids of all 1,698 deployment prompts and of the 160 training prompts. Only `sft` and `final` adapters exist as trained checkpoints (`checkpoint_steps` [159, 278]). There is no intermediate DPO checkpoint.

## 7. Offline measurements on saved training artifacts (all VERIFIED computations, scripts in SCR)

### 7.1 Supervised labels (`SCR/train_labels.py`, from `GPU02/inputs/prepared.json` and `proposal.json`)

- 160 records, one supervised update each, in world order then snapshot order (not shuffled). Batch size 1, AdamW, learning rate 1e-4, gradient clip 1.0, LoRA rank 8 alpha 16 on q/v of all 28 layers, seed 109.
- Labels: set_x_neg 32, stop 32, set_x_pos 29, observe_all 24, set_xy 12, set_y_pos 10, set_y_neg 10, set_x_zero 7, set_y_zero 4. So set_x_neg is 25% of the 128 nonterminal labels. The targets are not nearly constant.
- By snapshot (acquisition cost 0, 6, 12, 18, 24): cost 0: set_x_neg 16 of 32. Cost 6: set_x_pos 13, set_x_neg 7. Cost 12: observe_all 9, set_x_pos 8. Cost 18: observe_all 7, set_x_pos 6, set_xy 6. Cost 24: stop 32 of 32.
- The 32 stop labels are all terminal states (`terminal: true`, `remaining_units: 0`, stop is the only legal command, receipt `stop_label_basis: "no affordable trajectory action"`). The deployed policy is never called on such a state.
- Teachers rotate by snapshot: 32 nonterminal labels each from uniform_random, menu_coverage, local_information, target_risk. The teacher identity is not in the prompt. A quarter of the nonterminal labels are uniform random by construction.
- In-sample majority-label accuracy: constant 25.0%, by stage 35.2%, by teacher and stage 49.2%.
- Label token frequencies at the grammar branches: B1 action 128, stop 32. B2 set 104, observe 24. B3 `_x` 68, `_y` 24, `_xy` 12. B4 after `_x`: `_neg` 32, `_pos` 29, `_zero` 7. **The per-branch majority path spells set_x_neg**, although that command is only a quarter of the labels.

### 7.2 What the supervised stage learned (`SCR/train_scores.py`)

- Each record is visited once, so the loss at its visit is an out-of-sample (prequential) likelihood. Mean sequence NLL of the teacher's command on nonterminal states: 2.25 nats for steps 40 to 80, 2.57 for steps 80 to 120, 2.17 for steps 120 to 160. Uniform over 8 actions is 2.08. An in-sample stage-only label prior gives 1.57 to 1.79. The adapter's out-of-sample prediction of the teacher stayed at about the level of a uniform guess and worse than a stage-only prior. Caveat: the model's NLL is a full-vocabulary sequence likelihood, so it includes the format tokens and the act/stop branch. The terminal stop NLL fell to about 0.3 nats late in training, which bounds that overhead at roughly 0.3.
- Post-SFT sequence log-probability on 238 (training state, command) pairs from 119 records: **81.7% of the variance is explained by the command identity alone**, 93.5% by command and snapshot index, 95.2% adding graph. Means: set_x_neg −0.95, observe_all −1.09, set_y_neg −2.26, set_x_pos −3.22, set_xy −3.88, set_y_pos −4.33, set_x_zero −4.90, set_y_zero −5.56. Within-command SD 0.26 to 1.05, between-command SD 1.60.
- The prior is not the label marginal. set_x_pos has 22.7% of nonterminal labels but mean probability 0.05. set_y_neg and set_y_pos have equal label counts (10 and 10) but differ by 2.1 nats. INFERENCE: a preference for the token `_neg` is shared across the `_x` and `_y` contexts.
- Whole-command argmax versus branch-wise greedy, illustrated with the post-SFT stage means (geometric-mean probabilities over the training states where each command was scored, small n per cell, so INFERENCE until per-state vectors exist). Stage 0: set_x_neg 0.66, observe_all 0.16. Stage 1: set_x_neg 0.41, observe_all 0.30. Stage 2: observe_all 0.38, set_x_neg 0.31. Stage 3: observe_all 0.44, set_x_neg 0.26. In stages 2 and 3 the single most likely command is observe_all, but at branch B2 `observe` (0.38, 0.44) loses to `set`, the sum of seven commands (0.51, 0.50), after which `_x` (0.37, 0.34 against `_y` 0.12, 0.15) and `_neg` (0.31, 0.26 against `_pos` 0.05, 0.07) follow. Greedy decoding therefore yields set_x_neg where a whole-command argmax would yield observe_all. The eight actions sum to 0.89 to 0.97 at each stage, which is a consistency check on these means.
- Evidence of some state reading, to be preserved: the post-SFT log-probability of set_x_neg falls as set_x_neg accumulates in the state's history (correlation −0.74, −0.71 within stage, n = 23, means −0.66 with zero prior uses, −1.34 with three or more). The effect is about 0.7 nat against between-command gaps of 2 to 5 nats.

### 7.3 What the DPO stage did (`SCR/train_scores.py`, `SCR/train_scores2.py`)

Setup (VERIFIED): 119 updates, one strict pair each, chosen from 687 strict pairs. 2,897 of the 3,584 collected pairs are ties and are unused. Order is by world id, so the last 26 updates all come from the shifted_high regime. beta 0.1, optimizer state carried over from SFT, final iterate deployed.

Because each pair is visited once, the loss at its visit reveals `delta = (policy chosen − rejected margin) − (reference margin)` after training on the earlier pairs only: `delta = −log(exp(loss) − 1) / beta`.

- Mean prequential DPO loss 0.7395, above ln 2 = 0.6931 that an untouched reference would score. 53% of steps are above ln 2.
- Agreement with the measured preference at visit time: reference 66 of 119, evolving policy 66 of 119. Mean delta −0.39 nats (t = −0.91, p = 0.36). No generalization from earlier pairs to later pairs.
- In 92 of 118 pairs (78.0%, binomial p = 7.7e-10) the margin had moved further in the direction the SFT reference already leaned. Policy margin ≈ 1.74 × reference margin for steps 1 to 40, 2.61 × for 40 to 80, 2.42 × for 80 to 119 (correlations 0.75, 0.87, 0.83).
- A state-independent per-command linear drift explains R² = 0.60 of the deltas. The fitted drift correlates 0.91 with the post-SFT prior and 0.09 with the net chosen-minus-rejected label counts (set_x_pos +9, set_x_neg +5, set_y_neg +5, set_xy +4, set_y_zero +3, set_x_zero −6, observe_all −7, set_y_pos −13).
- The nine pairs in which set_x_neg was the rejected (worse) action show deltas of −0.2, −4.6, −6.4, +1.4, −8.2, −1.4, −6.7, −3.1, −7.8 (eight of nine negative: the policy increasingly preferred set_x_neg against the label). The fourteen pairs in which it was the chosen action show +0.7, −1.0, +1.6, −0.7, +4.2, +1.0, +3.6, +0.6, +1.9, +1.5, +4.1, +10.7, +1.5, +8.7 (twelve of fourteen positive). Across all pairs the deltas range from −13.8 to +15.9 nats.
- End of schedule: set_x_neg is the chosen side at steps 108, 109, 110 and is never the rejected side after step 97. observe_all, the SFT prior's second choice, is the rejected side at steps 112, 113, 115.
- Stop never appears in a DPO pair.

INFERENCE: with beta 0.1 the DPO loss is almost linear in the margin over ±10 nats, and Adam removes the beta scale from the step size. Scaling up the existing prior margins lowers the average loss whenever the prior agrees with the labels on average (mean reference margin +0.27 nats). That direction needs no state reading. The measurements above are consistent with it. They do not prove it.

### 7.4 The matched numerical ranker collapsed too (`SCR/numeric_scores.py`, from CPU02 policy-04)

- Learned ranker actions: set_x_pos 435, set_x_neg 141, nothing else (576 decisions). Mostly set_x_neg at decisions 0 to 6, always set_x_pos from decision 12.
- Over 564 full-menu decisions, the log-softmax over the eight actions has 98.1% of its variance explained by the action identity. Between-action SD 0.072 nats, within-action SD 0.010 nats. Top-1 minus top-2 gap: median 0.0031 nats, maximum 0.0059. Probability of the chosen action: median 0.136.
- This arm has no tokenizer, no prompt, no greedy trie, and receives 8,891 numeric features with the same labels and pairs. It also learned only a tiny, state-independent action bias.
- For reference, target_risk used observe_all 215, set_x_neg 129, set_x_pos 76, set_y_pos 62, set_xy 31, set_y_neg 29, set_y_zero 16, set_x_zero 2.

## 8. (e) Ranked hypotheses for a trained adapter emitting one command in every state

1. **The supervision and exposure could only teach a command prior.** One pass of 160 single-example updates on labels from four rotating teachers, one of them uniform random, with the teacher hidden. Support: 7.1, 7.2, and above all 7.4. Confirm or refute: (i) CPU-only learnability control on the saved 128 labels, any numeric model with leave-one-world-out validation against the 25% and 35% baselines, repeated for target_risk labels only. (ii) Full command score vectors for frozen, SFT and DPO on the 160 training prompts (section 6).
2. **DPO amplified the SFT prior instead of learning preferences.** Support: 7.3. This explains the step from 481 of 523 to 576 of 576. Confirm: score vectors of `sft.adapter.pt` and `final.adapter.pt` on identical states and compare margins. A deterministic replay from `sft.adapter.pt` and `sft.optimizer.pt` with intermediate checkpoints, shuffled order, larger beta or averaged iterates would separate recency, linear-regime and optimizer-carry-over effects. That replay is training and needs authorization.
3. **Greedy branch-wise decoding turns a diffuse, prior-dominated distribution into one deterministic command, and the branch majorities spell set_x_neg.** Support: section 3 and 7.1. Any argmax of a state-insensitive distribution is constant. The trie adds a subtree-mass bias toward `set` and `_x`. Confirm: with the score vectors, count decisions that change under whole-command argmax and under sampling.
4. **Serialization leaves the decision-relevant content unusable in one pass.** Digit-level tokens are 34% to 48% of the prompt, raw 17-digit floats, no derived features, constant text nearest the answer, no scratchpad. Support: sections 1, 2, 4. Against it as the sole cause: 7.4. Confirm: counterfactual edits of saved prompts (swap posterior blocks between states, permute or relabel history actions, change remaining_units, change the cost menu, change graph parents, reorder the menu) and measure the change in command log-probabilities against the change in planner scores for the same edits.
5. **On-policy distribution shift and self-reinforcing histories.** Training histories contain runs of one repeated acquisition of at most 10 (mostly 0 to 3). DPO deployment histories contain runs up to 23. The frozen model emits the only action id present in the initial history. Support: section 5 and `SCR` training-history tally. Against a pure copying story for SFT: the negative history-count effect in 7.2. Confirm: score saved late DPO states and edited histories.
6. **A token-level bias at a shared late token.** Shared prefix tokens are forced and cannot carry preference. The four branch tokens can. Support: the `_neg` preference common to `_x` and `_y` in 7.2. Confirm: branch-token distributions B1 to B4 from the score vectors.
7. **Nothing to learn from for objectives and almost nothing for costs.** One task payload in all 1,858 prompts. Two cost menus. DPO and frozen only ever chose cost-1 actions. This is a design limit on what could be learned or tested, not a cause of the constant action.
8. **Stop supervision artifact.** Explains the SFT early stops (23 of 24 episodes with 1 to 3 units left), not the constant acquisition. Support: 7.1.

Ruled out by saved evidence: truncation or context overflow. Identical or stale prompts. Sampling. A legal set reduced to one command. A training/deployment prompt-format mismatch. An adapter that was not loaded (pins match and the three arms behave differently).

Assessment of the task's example hypotheses: "supervised targets nearly constant" is refuted as stated (25%). "Prompt boilerplate dominating" is partly true (23% to 49% constant, and the last 346 tokens are constant). "Numbers tokenized into long digit strings" is verified. Whether the model ignores them is not established by any saved file.

## 9. (g) Stop rules

- LLM arms: `{"stop":true}` is always in the legal grammar, so the model may stop at any decision. No private error is available to it. If no action is affordable, the driver itself records a stop with provenance `budget_cap` and never calls the policy (`llm_intervention_episode.py:100-115`). VERIFIED outcomes in the 192 cost reports: frozen and DPO 24 of 24 `budget_exhausted`. SFT 23 `policy_stop`, 1 `budget_exhausted`, 721 units paid.
- uniform_random, menu_coverage, local_information, target_risk: return stop only for a terminal state or no affordable action (`llm_intervention_numerical_policy.py:54-57`, `llm_intervention_target_risk.py:108-111`). The screen wrapper raises if called with nothing affordable (`llm_intervention_screen_policy.py:116-117`). They always spend the budget. VERIFIED 24 of 24 `budget_exhausted`, 768 units each.
- Learned ranker: stop is one scored candidate row (`llm_intervention_policy_cli.py:42-44`, `llm_intervention_task.py:252-255`). It never won (mean stop score −7.78 against about −6.8 to −7.0 for actions).
- The 0.04 NMSE crossing is a private retrospective endpoint, never a stopping signal.

## 10. Documents versus code

No contradiction found between the newest documents and the executed code within this scope. Two points of wording matter:

- The handoff (2026-11-12, "Next research work" item 1) and the results report speak of comparing "command scores". No such scores exist for the LLM arms. Deployment never computes them. They must be newly computed.
- The agenda (2026-11-02) proposed supervised teachers "including uncertainty and information-based design". The executed teacher roster is uniform_random, menu_coverage, local_information, target_risk in equal shares.

## 11. Scratch artifacts for later agents (all under SCR)

- `dpo_llm-0000.templated.txt`, `dpo_llm-0000.user.txt`: the verbatim prompt.
- `llm_decisions.csv`: one row per LLM decision (arm, ordinal, episode, decision_index, graph, cost_menu, noise, history_rows, remaining_units, prompt_tokens, completion_action, planner_best_action, planner gains, envelope_sha256, source file).
- `decisions.pkl`: the same with the full decoded envelopes. `planner_{frozen,sft,dpo}_llm.pkl`: planner score vectors on those states.
- `numeric_decisions.pkl`: decisions and saved score metadata of the five numerical arms.
- `train_rows.json`, `train_scores.json`: per-record labels, teachers, histories. Per-step SFT losses. Per-pair reference scores, losses and deltas.
- Scripts: `decode_prompt.py`, `build_table.py`, `analyze1.py`, `analyze2.py`, `commands.py`, `composition.py`, `train_labels.py`, `train_scores.py`, `train_scores2.py`, `numeric_scores.py`, `planner_on_llm_states.py`, `planner_summary.py`.
