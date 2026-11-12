# literature-policy: prior art, failure modes, comparators and positioning for an SCM-conditioned LLM intervention policy

Prepared 2026-11-12. Label: `literature-policy`. Scope: web literature research plus a small number of read-only repository look-ups made only to align the literature with what was actually implemented.

## 0. How to read these notes

Verification legend used throughout:

- **[V-abs]** I opened the primary abstract page (arXiv, PMLR, ACL Anthology, ICLR/ICML/NeurIPS proceedings) in this session and the bibliographic facts and claim match.
- **[V-full]** I additionally read the full-text HTML (first ~100k characters unless noted) and extracted specifics. Numbers quoted come from that read.
- **[V-search]** Confirmed only through search-result snippets pointing at the primary page. Bibliographic facts are probably right. Details are less certain.
- **[MEM]** Stated from my own background knowledge and NOT opened in this session. Treat as a lead, not a verified citation.

All fetches went through a summarizing fetch tool, so "quoted" numbers are the tool's extraction from the page. Before any number below goes into a manuscript, re-open the paper PDF and check it. Nothing here was invented. Where I could not open a source I say so in Section 10.

No project names, people, repository paths or unpublished results were placed in any query or URL. One search with generic terms returned the project's own public arXiv preprint (Section 9.1). I then opened that public arXiv page by its identifier only.

Repository facts I relied on (read-only):

- Orientation: `/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md`
- Agenda and its literature section: `/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md` lines 27-39
- Results report: `/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md` (lines 32-46 on recorded decisions and scope, line 75 on training recipe)
- Per-policy action counts: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/actions.csv`
- Prompt rendering: `/Users/pat/code/ACE/scripts/research/llm_intervention_task.py:144-152` and serializer `/Users/pat/code/ACE/scripts/research/llm_intervention_contract.py:23-24`
- Command grammar: `/Users/pat/code/ACE/scripts/research/llm_intervention_grammar.py:60-83`
- Deployment decoding: `/Users/pat/code/ACE/scripts/research/llm_intervention_language_policy.py:87-111` and `132-184`
- DPO loss: `/Users/pat/code/ACE/scripts/research/llm_intervention_learning.py:94-103`
- Training loop (single pair or single row per step, LoRA rank check): `/Users/pat/code/ACE/scripts/research/llm_intervention_training_engine.py:240` and `428-430`

These are the current checkout, not the pinned executed bytes (frozen scientific source `a9346022...` per the handoff). Other agents own the forensic reading. I used them only to map literature onto the mechanism.

## 1. Implementation facts that determine which literature applies

Verified in the files above.

1. **Prompt = one instruction paragraph + compact canonical JSON of the whole public envelope + `"\nDecision:\n"`.** `task_prompt` concatenates fixed prose with `canonical(envelope)`, and `canonical` is `json.dumps(value, sort_keys=True, separators=(",", ":"))`. Floats therefore appear at Python `repr` precision inside a single long JSON string. There is no summarization of the history or posterior into decision-relevant statistics and no per-action score. (`llm_intervention_task.py:144-152`, `llm_intervention_contract.py:23-24`.)
2. **Action space = a closed set of short JSON strings.** Commands are `{"action_id":"<id>"}` for each affordable action sorted by `action_id`, plus `{"stop":true}`, each followed by EOS (`llm_intervention_grammar.py:73-82`). The executed identifiers are `observe_all, set_x_neg, set_x_pos, set_x_zero, set_xy, set_y_neg, set_y_pos, set_y_zero` (`actions.csv`). All commands share the prefix `{"action_id":"` and seven of eight share `set_`.
3. **Deployment = greedy, prefix-constrained decoding with thinking disabled.** `enable_thinking=False`, `do_sample=False, num_beams=1`, `prefix_allowed_tokens_fn` (`llm_intervention_language_policy.py:96-97`, `163-170`). The model never produces reasoning tokens. The chosen action is determined token by token, so the first diverging token decides between `observe…` and the pooled `set…` family.
4. **Training = full-sequence log-probability of command+EOS given the same chat prefix**, SFT then DPO with a fixed post-SFT reference. DPO is the standard loss `-logsigmoid(beta*((lp_c - lp_r) - (ref_c - ref_r)))`, default `beta=0.1` (`llm_intervention_learning.py:94-103`). The training loop feeds one row (SFT) or one chosen/rejected pair (DPO, `.reshape(1, 2)`) per optimizer step (`llm_intervention_training_engine.py:428-430`. The plan record at line 216 sets `sft_rows = sft_steps` and `dpo_rows = 2*dpo_steps`). The beta actually used is a recipe value I did not look up. The results report gives 160 supervised updates then 119 preference updates on 32 training systems, rank-8 LoRA on q/v (report line 75, handoff line 54, engine line 240).
5. **Scale, by the report's own numbers:** 160 SFT steps and 119 DPO steps. 3,584 preference comparisons were collected and retained, but the exposure rule produced 119 preference updates (handoff line 54). I did not verify how 3,584 maps to 119. That belongs to the training agents.
6. **Recorded action distributions** (`actions.csv`, accepted acquisitions over 24 runs):

| Policy | observe_all | set_x_neg | set_x_pos | set_x_zero | set_xy | set_y_neg | set_y_pos | set_y_zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_random | 66 | 59 | 64 | 52 | 60 | 67 | 55 | 64 |
| menu_coverage | 0 | 84 | 84 | 72 | 60 | 60 | 60 | 60 |
| local_information | 318 | 149 | 67 | 0 | 5 | 7 | 24 | 0 |
| target_risk | 215 | 129 | 76 | 2 | 31 | 29 | 62 | 16 |
| learned_ranker | 0 | 141 | 435 | 0 | 0 | 0 | 0 | 0 |
| frozen_llm | 574 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| sft_llm | 8 | 481 | 23 | 0 | 0 | 11 | 0 | 0 |
| dpo_llm | 0 | 576 | 0 | 0 | 0 | 0 | 0 | 0 |

Two things stand out for the literature mapping. The best planner (target_risk) uses all eight actions, with `observe_all` and `set_x_neg` the two most frequent. The **learned numerical ranker also collapsed**, onto two X interventions (435 `set_x_pos`, 141 `set_x_neg`, zero observations). The handoff and report text discuss collapse only for the LLM arms. A collapse shared by a non-language learner trained on matched examples points at the supervision signal or the training budget at least as much as at language-model numeracy. This is an observation from the counts, not a demonstrated cause.

## 2. Verification of the works already cited in the agenda

All ten exist and are described accurately. Corrections are minor.

| Agenda item | Verified facts | Status | Note |
| --- | --- | --- | --- |
| Deep Adaptive Design | Foster, Ivanova, Malik, Rainforth. ICML 2021, PMLR 139:3384-3395. https://proceedings.mlr.press/v139/foster21a.html | [V-abs] | Accurate. Abstract says "contrastive information bounds". |
| Learning to Ask Informative Questions | Mazzaccara, Testoni, Bernardi. Findings of EMNLP 2024, pp. 5064-5074. LLaMA-2-Chat 7B, 20 Questions, DPO on low-EIG vs high-EIG question pairs. https://aclanthology.org/2024.findings-emnlp.291/ | [V-abs] | Accurate. But the agenda's remark that this "does not establish ... ACE's proposed supervised-then-preference training sequence" overstates the gap. Paprika (Section 3.3) already uses SFT followed by SFT+DPO for information gathering. |
| ASIG | Hartmann, Harvey, Navott, Wang, Melo, Cipcigan, Zhang, Abate. arXiv:2607.03426, 3 Jul 2026. FoGen 2026 workshop at ICML 2026, non-archival. https://arxiv.org/abs/2607.03426 | [V-full] | Accurate. Abstract says "Expected" information gain. In the full text the EIG is estimated with the BED-LLM estimator. "Limited transfer evidence" is fair: MediQ accuracy 46.7→47.6 (7B) and 54.1→54.2 (14B), 5 episodes per setting. LoRA rank 32 on all attention and MLP projections, Qwen2.5-7B/14B, multi-turn GRPO with group size 8, 40 epochs × 100 entities. |
| Model-based Causal BO | Sussex, Makarova, Krause. ICLR 2023. https://arxiv.org/abs/2211.10257 | [V-abs] | Accurate. |
| LLAMBO | Liu, Astorga, Seedat, van der Schaar. ICLR 2024. https://arxiv.org/abs/2402.03921 | [V-abs] | Accurate. Abstract: no fine-tuning, benefit "especially" early when observations are sparse. |
| LMABO | Ngo, Phan Trong, Nguyen, (Sunil) Gupta, Venkatesh. ICLR 2026. Title "Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models". | [V-abs] | Accurate. Abstract says pre-trained LLM as "zero-shot, online strategist" choosing from an acquisition portfolio, 50 benchmark problems. |
| Model Discovery Agent | Kevin Murphy. arXiv:2608.09696, v5 27 Sep 2026, preprint. https://arxiv.org/abs/2608.09696 | [V-abs] | Accurate. LLM model proposals + value-of-information design + approximate Bayesian inference. ChemBench, GlucoseBench, neuron SDE. BoxingGym results in appendix. |
| LLMs for BO in Scientific Domains: Are We There Yet? | Rushil Gupta, Jason Hartford, Bang Liu. Findings of EMNLP 2025, pp. 15482-15510. https://aclanthology.org/2025.findings-emnlp.838/ and https://arxiv.org/abs/2509.21403 | [V-full] | Accurate. See Section 5.1 for details the agenda should use. |
| BoxingGym | Gandhi, M. Y. Li, Goodyear, Bhatia, L. Li, Bhaskar, Zaman, Goodman. arXiv:2501.01540 (v2 Oct 2025). | [V-full] | Accurate. Agenda omits two findings that matter: "Prior information does not improve performance" and the statistical-model-augmented agent "does not consistently improve" over the plain LLM. |
| Causal Chambers | Gamella, Peters, Bühlmann. Nature Machine Intelligence 7, 107-118 (2025). arXiv:2404.11341. | [V-abs] via arXiv | The nature.com URL redirects to an authorization flow. I did not follow it. arXiv record confirms the journal reference. |

## 3. What the agenda missed (closest prior art)

Each entry: authors, year, venue, URL, one-sentence claim, why it matters.

### 3.1 Newest and closest: LLM policies trained for sequential experimental design

- **Kriz, Berthon, Astorga, van der Schaar (2026). "Amortizing Sequential Experimental Design in Reasoning LLMs." arXiv:2610.20832, 14 Oct 2026, no venue listed.** https://arxiv.org/abs/2610.20832 **[V-full, partial]**. Claim: amortize sequential BED into an LLM by training the reasoning that guides action selection with information-gain rewards under a generative model kept separate from the LLM's own beliefs. Details: main policy Gemma-4-E4B-it, GRPO, LoRA rank 16 on q/k/v/o/gate/up/down, 250 steps, batch 128, 8 rollouts per state, lr 1e-5. Tasks: 20 Questions, MediQ, a natural-language adaptation of Cube-NM. Beats BED-LLM (inference-time) and ASIG (training-time) on most categories at lower cost. Ablation `AR-BED^Direct` (no reasoning) helps in distribution but "does not produce robust transfer across domains". **Critical for this project:** the Cube-NM mechanism check uses **Qwen3-0.6B with GRPO/LoRA, 250 updates, lr 1e-5, 16 states × 16 rollouts, temperature 2, entropy coefficient 0.02, randomly permuted test identifiers, no reasoning trace**, and the two-step-reward policy learns to pick the zero-immediate-value context test "with almost unit probability" (Bayesian MAP solve rate after five tests 75.0% vs 30.2% for the one-step-reward policy). Caveat stated by the authors: matched single training seeds. Why it matters: same base model as this project. It shows Qwen3-0.6B + LoRA can learn a non-trivial, state-dependent discrete selection policy when (a) the state is categorical text, not a long numeric JSON, (b) learning is on-policy with exploration pressure, and (c) identifiers are permuted so a label prior cannot be exploited. No numerical, causal or SCM experiment is included. This is now the single closest prior work and it is post-agenda.
- **Hartmann et al. (2026) ASIG.** Already cited. See Section 2.
- **Choudhury et al. (2025/2026). "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design." ICLR 2026. arXiv:2508.21184.** https://arxiv.org/abs/2508.21184 **[V-search]**. Claim: inference-time sequential BED for LLM question asking with a carefully designed EIG estimator from the LLM's belief distribution. Why it matters: the inference-time (no fine-tuning) baseline that both ASIG and Kriz et al. compare with. Kriz et al. report it is about 61× slower per game than their trained policy.
- **Tajwar, Jiang, Thankaraj, Rahman, Kolter, Schneider, Salakhutdinov (2025). "Training a Generally Curious Agent" (Paprika). ICML 2025. arXiv:2502.17543.** https://arxiv.org/abs/2502.17543 **[V-full]**. Claim: fine-tuning on synthetic interaction data from diverse tasks teaches in-context exploration that transfers to unseen task groups. Details: SFT on winning trajectories, then RPO = SFT loss + multi-turn DPO loss (α = 1.0) with the initial model as reference. Llama-3.1-8B-Instruct and Gemma-3-12B-IT. 17,181 SFT trajectories and 5,260 preference pairs. One pair per task because multiple pairs per task "led to unintended unalignment" (they cite Razin et al.). Leave-one-out improves 9 of 10 held-out groups. Bandit best-arm 42.25%→62.25% without bandit training, but gains shrink as arms grow. Appendix ablation says the RPO stage improves over SFT-only on six groups (figure only). Why it matters: **the supervised-then-preference sequence the agenda calls ACE-specific is prior art here**, with about 107× this project's supervised examples (17,181 vs 160) and 44× its preference pairs (5,260 vs 119) on models 13-20× larger.
- **Schmied, Bornschein, Grau-Moya, Wulfmeier, Pascanu (2025/2026). "LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities." ICLR 2026 per the ICLR proceedings listing. arXiv:2504.16078.** https://arxiv.org/abs/2504.16078 **[V-full]**. Claim: small and mid-size LLMs fail as decision makers through greediness, frequency bias and a knowing-doing gap, and RL fine-tuning on self-generated chain-of-thought rationales mitigates these. Numbers: Gemma2 2B "heavily suffers from repeated actions", frequent-action fraction 96%, versus 14% for 27B. The paper says "frequency bias is an artifact of supervised pre-training". For 27B, 87% of rationales were correct, yet among correct rationales the model chose the greedy action 58% of the time and the optimal one 21%. RL fine-tuning reduces the frequent-action fraction from 70% to 35% at 0-10 repetitions but it "remains elevated for high repetitions". "Without CoT, RLFT barely attains the performance of ICL w/ CoT." Why it matters: this is the named, measured version of the project's constant-action symptom, with a size gradient and a remedy that needs reasoning tokens.
- **Nie, Su, Chang, Lee, Chi, Le, Chen (2025). "EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration." ICML 2025 (PMLR 267). arXiv:2410.06238.** https://arxiv.org/abs/2410.06238 **[V-full]**. Claim: algorithmic knowledge can be injected into LLMs through inference-time algorithm-guided support and through fine-tuning on oracle (UCB/LinUCB) trajectories. Numbers (Gemini-1.5 Flash, win-rate style scores): contextual bandit, fine-tuned on raw history 28.6 versus fine-tuned with algorithm-guided support (per-arm exploitation value and exploration bonus given as text) 89.3. Teacher 96.4. Multi-armed bandit, fine-tuned raw history 65.6 versus teacher 90.6. Authors: "an optimality gap remains between LLMs and classical optimal algorithms". Gemma 2B and 9B "remain in the linear regret regime across nearly all methods". Inference-only baselines show little uncertainty-directed exploration. Why it matters: closest analogue of the project's supervised arm (behavior cloning of a numerical teacher). It predicts three things for this project: cloning from raw numeric context fails on contextual problems, supplying computed per-action statistics rescues it, and even then the student trails the teacher.
- **Krishnamurthy, Harris, Foster, Zhang, Slivkins (2024). "Can large language models explore in-context?" NeurIPS 2024. arXiv:2403.15371.** https://arxiv.org/abs/2403.15371 **[V-abs]**. Claim: across GPT-3.5, GPT-4 and Llama2, only GPT-4 with chain-of-thought and an externally summarized history presented as sufficient statistics explores satisfactorily. Raw history fails even with chain-of-thought. Why it matters: direct evidence that the input should be decision-relevant summaries, not a raw record.
- **Qiu, Sha, Allen, Kim, Linzen, van Steenkiste (2025/2026). "Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models." Nature Communications (arXiv:2503.17523, v3 Jan 2026).** https://arxiv.org/abs/2503.17523 **[V-full]**. Claim: fine-tuning LLMs to imitate a normative Bayesian model's predictions improves belief updating and generalizes to new tasks, and it beats "oracle teaching" on always-correct answers. Details: Gemma 2 9B, Llama 3 8B, Qwen 2.5 7B, full fine-tuning (LoRA "similar"), 6,240 examples. Authors' hypothesis: "the Bayesian model's educated guesses make for a stronger learning signal than the correct answers". Why it matters: supports distilling the planner's graded judgments rather than hard winner labels.
- **Menet, Terzić, Hersche, Krause, Rahimi (2025/2026). "Thompson Sampling via Fine-Tuning of LLMs" (ToSFiT). ICLR 2026. arXiv:2510.13328.** https://arxiv.org/abs/2510.13328 **[V-abs]**. Claim: avoid acquisition maximization in large discrete spaces by fine-tuning a prompt-conditioned LLM toward the posterior probability of maximality, with a regret bound. Why it matters: a principled route in which measured outcomes update LLM weights online. It targets optimization over huge unstructured candidate spaces, which is where an LLM proposal distribution has a natural role.
- **Ranković and Schwaller (2026). "Large Language Models as Generative Bayesian Policies." Spotlight at the ICML 2026 "AI for Science" workshop.** https://icml.cc/virtual/2026/73551 **[V-abs, workshop page]**. Claim: a GP critic on LLM features supplies acquisition-based preference signals that fine-tune the LLM as a generative policy for chemistry and peptide design. Why it matters: acquisition-derived preference fine-tuning of an LLM design policy already exists, at workshop level, for optimization. The page does not name the objective. I could not confirm DPO.
- **van Dijk and Vrkic (2026). "SciDesignBench." arXiv:2603.12724.** https://arxiv.org/abs/2603.12724 **[V-abs]**. Claim: 520 simulator-grounded inverse-design tasks, and an RL-from-simulator-feedback recipe raises an 8B model's single-turn success by 8-17 points in three domains. Why it matters: fine-tuning from measured simulator outcomes is established for inverse design. It is not intervention selection or cost-to-accuracy.

### 3.2 Amortized numerical design policies (the comparator family a statistics reviewer knows)

- **Blau, Bonilla, Chades, Dezfouli (2022). "Optimizing Sequential Experimental Design with Deep Reinforcement Learning." ICML 2022. arXiv:2202.00821.** [V-abs]. RL-trained design policies that handle discrete designs and black-box models.
- **Ivanova, Foster, Kleinegesse, Gutmann, Rainforth (2021). iDAD. NeurIPS 2021. arXiv:2111.02329.** [V-abs]. Policy-based design for implicit models.
- **Hedman, Ivanova, Guan, Rainforth (2025). "Step-DAD: Semi-Amortized Policy-Based BED." ICML 2025. arXiv:2507.14057.** [V-search]. Refines an offline policy during the live experiment. Why it matters: the direct numerical analogue of "the SCM adapted online, the LLM weights did not".
- **Huang, Guo, Acerbi, Kaski (2024). "Amortized Bayesian Experimental Design for Decision-Making" (TNDP). NeurIPS 2024. arXiv:2411.02064.** [V-abs]. Decision-aware amortized design with a transformer neural decision process.
- **Huang, Wen, Bharti, Kaski, Acerbi (2025). ALINE. NeurIPS 2025. arXiv:2506.07259.** [V-search]. One transformer amortizes inference and RL-trained acquisition with self-estimated information-gain reward.
- **Zhang, Huang, Kaski, Martinelli (2025). PABBO. ICLR 2025. arXiv:2503.00924.** [V-search]. Amortized preferential black-box optimization with a transformer neural process trained by RL.
- **Müller, Feurer, Hollmann, Hutter (2023). PFNs4BO. ICML 2023. arXiv:2305.17535.** [V-abs]. Prior-data fitted networks as BO surrogates, including learned non-myopic acquisition.
- **Annadani, Tigas, Bauer, Foster (2024). "Amortized Active Causal Induction with Deep Reinforcement Learning" (CAASL). NeurIPS 2024. arXiv:2405.16718.** [V-full]. Transformer intervention-design policy (targets and values) trained with SAC in its REDQ variant, reward = improvement in expected correct adjacency entries under a frozen pretrained AVICI posterior. Trained on linear-Gaussian SCMs with d = 10, budget T = 10. Claims zero-shot generalization to scale-free graphs, changed mechanisms, non-Gaussian and heteroskedastic noise, shift interventions and up to d = 30. Honest details from the text: the advantage over random shrinks as d grows, random "remains competitive" when intervention type changes on the gene simulator, each model took about 3 days on 3 A100 GPUs, and hyperparameter tuning took about 70,000 hours. No code link in the text I read. Why it matters: **a numerical amortized intervention policy that already claims the transfer behavior this project hopes language will supply**, and a benchmark for what "properly trained" costs.
- **Zhang, Dong, Liu, Huan (2025/2026). "Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning" (GO-CBED). TMLR 09/2026. arXiv:2507.07359.** [V-full, first 100k characters]. Non-myopic transformer policy that picks intervention target and value to maximize a variational lower bound on EIG about user-specified causal quantities. Its motivating example is a **fixed six-node linear-Gaussian SCM with known graph**, where the parameter-targeting policy scores higher on parameter EIG but clearly worse on the causal query. Main experiments use unknown graphs with d = 10 to 30 (50 in an appendix), linear and neural mechanisms, horizons T = 10 to 15. Baselines: random, nested Monte Carlo (from Toth et al.), CAASL, Soft-CBED, DiffCBED. Stated limits: assumes the true system is plausible under the training prior, synthetic and semi-synthetic data only. Code stated at https://github.com/HowIII/gocbed (not opened). Why it matters: **the closest numerical counterpart of "target-risk" intervention selection, peer reviewed two months ago, and its motivating example is nearly this project's setting.** It also documents the exact contrast between the project's local-information and target-risk baselines. It is the natural strong learned numerical comparator.
- **Lee, Xie, Pacchiano, Chandak, Finn, Nachum, Brunskill (2023). Decision-Pretrained Transformer. NeurIPS 2023. arXiv:2306.14892.** [V-abs]. Supervised prediction of the optimal action from a query state and an in-context dataset yields in-context posterior sampling with regret guarantees.
- **Laskin et al. (2023). Algorithm Distillation. ICLR 2023. arXiv:2210.14215.** [V-search]. Transformer trained on learning histories of an RL algorithm improves in context.
- **Li and Simchi-Levi (2026). "Transformers as Bayesian In-Context Experimenters." arXiv:2606.31184.** [V-abs]. Transformer policies imitate a Bayesian posterior Neyman teacher for adaptive treatment allocation. **Withdrawn by the author in v2 (2 Sep 2026) because a proof needed rewriting. Do not cite as established.**
- **Barlas and Salako (2025). "Performance Comparisons of RL Algorithms for Sequential Experimental Design." AAAI 2025 workshop. arXiv:2503.05905.** [V-abs]. Choice of RL algorithm changes generalization of design agents. Dropout and ensembles help.

### 3.3 Causal experimental design baselines and LLM + intervention work

- **Tigas, Annadani, Jesson, Schölkopf, Gal, Bauer (2022). "Interventions, Where and How?" (CBED). NeurIPS 2022. arXiv:2203.02016.** [V-search]. BOED selecting target and value.
- **Toth, Lorch, Knoll, Krause, Pernkopf, Peharz, von Kügelgen (2022). "Active Bayesian Causal Inference." NeurIPS 2022. arXiv:2206.02063.** [V-search, author list from memory]. Chooses interventions maximally informative about a target causal query.
- **Olko et al. (2023). "Trust Your ∇" (GIT). NeurIPS 2023. arXiv:2211.13715.** [V-search]. Gradient-based intervention targeting.
- **Zhang, Cammarata, Squires, Sapsis, Uhler (2023). "Active learning for optimal intervention design in causal models." Nature Machine Intelligence 5, 1066-1075. arXiv:2209.04744.** [V-search]. Causally informed acquisition for reaching a target mean, with consistency results for **linear causal models with known graph**.
- **Aglietti, Lu, Paleyes, González (2020). "Causal Bayesian Optimization." AISTATS 2020, PMLR 108:3155-3164.** [V-search, author list from memory].
- **J. Li, Chen, Liu, Cai, Liu, Han, Zhang, Xiong (2025). "Can Large Language Models Help Experimental Design for Causal Discovery?" (LeGIT). arXiv:2503.01139. Oral at an ICML 2025 workshop per search.** [V-full]. Frozen GPT-4o proposes intervention targets from variable names and descriptions for warm-up rounds (my derivation from stated hyperparameters: roughly 8-10 of 33 rounds), then GIT takes over. SHD on Alarm 17.40 vs GIT 19.60 vs random 38.80. Gains small and within overlapping deviations on Child and Asia. **No anonymized-name or no-semantics ablation is reported, and the benchmarks are classic public networks, so memorization is not excluded.** Why it matters: the LLM-for-interventions result most likely to be raised by a reviewer. It supports "semantic warm start then numerical", not "LLM replaces numerical".
- **Roy and Parbhoo (2026). "Why LLMs Fail at Causal Discovery and How Interventional Agents Escape." arXiv:2605.27567.** [V-abs]. Claims supervised fine-tuning, DPO and in-context learning cannot separate causal graphs with similar observational data (a "kernel obstruction theorem"), and proposes A-CBO with a frozen LLM as interventional oracle inside an external Bayesian loop. Preprint, 9 pages. I did not check the theorem. Why it matters: a published-on-arXiv argument that SFT/DPO are the wrong tools for causal decisions and that the decision should sit in a numerical loop.
- **Yang et al. (2026). "CausaLab." arXiv:2605.26029.** [V-abs]. Interactive causal-discovery environment with randomly sampled SCMs. GPT-5.2-high reaches 92% task accuracy but 0.471 all-edge F1 in the observational 6-node setting. Pure intervention strategies remain difficult. Premature stopping is a major weakness. Why it matters: independent benchmark candidate, and it reports the same early-stopping pathology the supervised arm showed (stopped early in 23 systems).
- **Chen et al. (2025/2026). "Auto-Discovery-Bench." arXiv:2502.15224 (v2 May 2026).** [V-abs]. Errors persist even with intervention selection removed. Long-range structured state tracking is the bottleneck.
- **Roohani et al. (2024/2025). BioDiscoveryAgent. arXiv:2405.17631 (ICLR 2025 per search).** [V-abs]. Claude 3.5 Sonnet agent claims +21% over BO baselines across six gene-perturbation datasets, including one unpublished dataset. Why it matters: the positive claim that Gupta et al. then stress-tested.
- Not found: a paper combining causal Bayesian optimization with an LLM-derived prior over graphs. One search only. Leads from snippets, unverified: GACBO (arXiv:2411.03028, uniform graph prior, no LLM), CausalGuard (arXiv:2605.21928), CauScientist (arXiv:2601.13614).

### 3.4 LLM as prior, surrogate, selector or program writer (frozen)

- **Kristiadi, Strieth-Kalthoff, Skreta, Poupart, Aspuru-Guzik, Pleiss (2024). "A Sober Look at LLMs for Material Discovery." ICML 2024. arXiv:2402.05015.** [V-abs]. LLMs help BO over molecules "only if they have been pretrained or finetuned with domain-specific data". LLM used as feature extractor for principled surrogates.
- **Requeima, Bronskill, Choi, Turner, Duvenaud (2024). "LLM Processes." NeurIPS 2024. arXiv:2405.12856.** [V-search]. Numerical predictive distributions conditioned on text, competitive with GPs.
- **Chen, Yuan et al. (2026). LABO. ICML 2026. arXiv:2605.22054.** [V-abs]. Treats LLM predictions as cheap evidence gated against real experiments, with a regret bound.
- **Karagözlü, Sonker, Gupta, Póczos, Schneider (2026). "Bayesian Optimization with Rich Auxiliary Information via LLMs." arXiv:2609.19437.** [V-abs]. LLMs digest unstructured side information. Authors find LLMs more useful when modeling likely maximizer locations than when predicting objective values pointwise.
- **Aglietti et al. (2024). FunBO. arXiv:2406.04824.** [V-abs]. An LLM writes acquisition functions as code, which generalize beyond training functions.
- **Li, Fox, Goodman (2024). "Automated Statistical Model Discovery with Language Models." ICML 2024. arXiv:2402.17879.** [V-search]. LLM proposes probabilistic programs and critiques them.
- **Capstick, Krishnan, Barnaghi (2025). AutoElicit. ICML 2025. arXiv:2411.17284.** [V-search]. LLM-elicited priors for linear models reduce sample needs, and beat in-context learning, which approximates Bayesian inference inconsistently.
- **Abhyankar et al. (2026). LLM-ACES. arXiv:2606.25039.** [V-abs]. LLM proposes operator priors that partition a symbolic hypothesis space, and candidate disagreement, not the LLM, selects which trajectories to collect.
- **Kambhampati et al. (2024). "Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks." ICML 2024, PMLR 235:22895-22907. arXiv:2402.01817.** [V-search]. Position paper for LLM-as-proposer with external model-based verifiers.
- **Chen, Song et al. (2022). OptFormer. NeurIPS 2022. arXiv:2205.13320.** [V-search]. Text-based transformer trained on large tuning logs imitates several HPO algorithms.
- **Song et al. (2024). OmniPred. TMLR 2024. arXiv:2402.14547.** [V-search]. Language models trained at scale on text-encoded (x, y) do precise numeric regression.
- **Nguyen and Grover (2025). LICO. ICLR 2025. arXiv:2406.18851.** [V-search]. Adds a separate embedding layer and prediction layer to an LLM for in-context molecular optimization.

## 4. Positioning map

### 4.1 Taken

| Territory | Representative work | Consequence |
| --- | --- | --- |
| Amortized numerical design policies, including non-myopic and decision-aware | DAD, iDAD, RL-BOED, Step-DAD, TNDP, ALINE, PABBO, PFNs4BO | "A learned policy replaces per-step optimization" is not a contribution. |
| Amortized numerical intervention design for causal models, including goal-oriented queries and zero-shot transfer claims | CAASL, GO-CBED, CBED, ABCI, GIT, Zhang et al. 2023 | A language policy must beat or complement these, not only random and coverage. |
| LLM fine-tuned for information gathering with EIG or preference signals | Mazzaccara (DPO), ASIG (GRPO), Kriz et al. (GRPO on reasoning), Paprika (SFT then SFT+DPO), BED-LLM (inference-time) | Preference or RL training of an LLM information-gathering policy, and the SFT-then-preference sequence, are prior art. |
| LLM fine-tuned from measured or simulated outcomes for design or optimization | ToSFiT, Ranković and Schwaller, SciDesignBench RLSF, LICO, Kristiadi | "Fine-tuning from outcomes" alone is not new. |
| Distilling numerical exploration algorithms into LLMs | EVOLvE, Schmied et al., Qiu et al., DPT and AD lineage | Behavior cloning of a planner into an LLM, and its shortfall against the teacher, are documented. |
| Frozen LLM as warm start, surrogate, acquisition selector, auxiliary-information reader, code writer | LLAMBO, LMABO, LeGIT, LABO, AuxBO, FunBO, LLM Processes, BioDiscoveryAgent | Zero-shot LLM help in early rounds is established and contested. |
| Careful negative results | Gupta et al., Kristiadi et al., BoxingGym, Krishnamurthy et al., CausaLab, Roy and Parbhoo | A weak or decorative LLM role will be recognized immediately. |

### 4.2 Open, to my knowledge after these searches

1. No paper found trains an LLM on **measured intervention outcomes in an explicit SCM state** and evaluates it against **goal-oriented numerical planners** on **experimental cost to a target accuracy**. The trained-LLM work is on 20 Questions, clinical dialogue, text games, bandits, a categorical diagnosis toy, molecules and inverse design.
2. No paper found shows a **trained** LLM design policy passing a **shuffled-feedback test** while also using semantic priors, and none measures **recovery delay after misleading metadata**. Gupta et al. show untrained open models fail the shuffled test. BoxingGym shows prior information does not help and is not revised.
3. Heterogeneous action costs with a learned stop decision in a causal intervention setting, for an LLM policy.
4. Whether **explicit SCM posterior input** (versus raw history, versus per-action planner statistics) changes a trained LLM policy. Krishnamurthy et al. and EVOLvE predict it does in bandits. Not shown for causal models.
5. LLM policy behavior under **SCM misspecification**, where Bayesian adaptive design is known to be fragile (Rainforth et al. 2024, Section 5.3).
6. A controlled diagnosis of constant-action collapse in a small fine-tuned LLM on numeric states with matched numerical students. Related phenomena are documented separately (frequency bias, likelihood displacement, linear regret of small models) but not joined in one controlled study.

### 4.3 Which claims would be novel if demonstrated (honest grading)

**Would be new, if the controls of Section 7 hold:**

- A trained LLM intervention policy that beats a goal-oriented numerical planner and a properly trained amortized numerical policy on cost to target accuracy, on fresh systems, in a regime where the reason for the planner's handicap is declared in advance (misleading or partial model class, semantic metadata the planner cannot read, objectives or constraints stated only in language, or an action space the planner cannot enumerate).
- A trained LLM policy that demonstrably uses measurements (performance drops under policy-input feedback shuffling) while also exploiting helpful metadata and recovering from misleading metadata, with recovery delay measured. No paper found shows all three together for any design task.
- The equal-information decomposition itself: frozen versus trained, language versus numerical student on identical data, with and without SCM posterior, with and without planner statistics, in one causal testbed. The individual contrasts exist separately in bandit and question-asking studies.

**Would be incremental:**

- Showing that SCM summaries or planner statistics in the prompt improve a cloned LLM policy (predicted by Krishnamurthy et al. and EVOLvE).
- Showing that on-policy RL with exploration pressure fixes collapse that offline DPO produced (predicted by Tajwar et al. 2024, Schmied et al., Kriz et al.).
- Transfer of a learned design policy across graphs or intervention types without a language-specific reason (CAASL and GO-CBED already claim this numerically).

**Not novel:**

- DPO or SFT-then-preference training of an LLM for information gathering (Mazzaccara et al., Paprika).
- Amortizing experimental design into a network, or into an LLM (DAD lineage, ASIG, Kriz et al.).
- An LLM matching a numerical planner it was distilled from. The agenda already says matching the teacher is not an advance (agenda line 88).

**A defensible alternative paper:** a controlled negative-and-diagnostic study in the style of Kristiadi et al. and Gupta et al., showing when and why a small SCM-conditioned LLM policy collapses and which of the documented mechanisms accounts for it, with numerical students as controls. Both of those papers were published at strong venues (ICML 2024, Findings of EMNLP 2025). This would not satisfy the stated submission condition of a demonstrated LLM benefit (agenda lines 161-169), so it is a fallback, not the target.

Scope caveat: "not found" reflects roughly 130 targeted searches and page fetches in one session. It is not a systematic review. The field moved between July and October 2026 (ASIG, Kriz et al., GO-CBED in TMLR, AuxBO, Model Discovery Agent v5), so a final check is needed before submission.

## 5. Where language models add value in experimental design: what the careful studies conclude

### 5.1 Gupta, Hartford, Liu (2025) [V-full]

- Models: LLaMA-3.1-8B-Instruct, Qwen-2-7B-Instruct, Qwen-2.5-14B-Instruct, Claude 4 Sonnet, GPT-4o-mini. None fine-tuned.
- Randomized-feedback control: replacing measured values or hit labels with random ones barely changed, and sometimes improved, cumulative hits (for example Llama on IFNG 44 with true feedback versus 51 with randomized feedback). Conclusion: early performance reflects priors, not use of feedback.
- Classical baselines on the **same LLM's embeddings**: LinUCB 93 versus the LLM agent 33.2 (Qwen-2-7B, IL2). LinUCB did not win everywhere (Llama, IL2: 35 versus 39.4).
- LLMNN: the LLM proposes five cluster centres, nearest unexplored candidates in an embedding space fill the batch. Replacing LLM centres with random centroids cut performance by more than 50% in some cases, so the LLM prior does carry information.
- Recommended controls: randomized-feedback ablation, classical baselines on identical embeddings, random-proposal ablation, matched budgets, several embedding choices, repeated runs, documented model versions.
- A table in the paper appears to repeat one GP row across models (possible table error noted during extraction). Do not quote the GP numbers without checking.

### 5.2 Other controlled findings

- **Kristiadi et al. (ICML 2024) [V-abs]:** LLMs help BO over molecules only with domain-specific pretraining or fine-tuning, and work best as features inside a principled Bayesian surrogate.
- **BoxingGym [V-full]:** "Prior information does not improve performance", models often did better with no prior after 10 experiments, the LLM sometimes made strong assumptions it did not revise, and adding an explicit statistical model "does not consistently improve" over plain GPT-4o. EIG regret is computed against the best of 100 random designs. The extracted text did not give a random-design baseline for the agents.
- **LLAMBO [V-abs]:** benefit concentrated early, when observations are sparse.
- **LeGIT [V-full]:** semantic warm-up helps most on larger graphs and low-data settings, no semantics ablation.
- **AuxBO [V-abs]:** LLM guidance is more useful as beliefs about where the optimum is than as pointwise value prediction.
- **EVOLvE [V-full]:** when the LLM is handed the algorithm's computed statistics it "only needs to perform addition and argmax", and that is when contextual performance becomes good.
- **Roy and Parbhoo [V-abs]:** argue that the decision should be in an external Bayesian loop with the LLM answering narrow interventional questions.
- **Falck, Wang, Holmes, "Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective", ICML 2024, arXiv:2406.00793 [V-abs]:** LLM in-context predictions violate the martingale property and their uncertainty does not shrink as a Bayesian learner's would. A statistics reviewer will cite this against any claim that an LLM "updates like a posterior" in context.
- **Anonymization evidence [V-search only, not confirmed on the primary pages]:** a search snippet reports that PhysGym (arXiv:2507.15550) has prior-knowledge levels ending in anonymized variable names, and that LLM-ACES found anonymizing ODEBench variables reduced exact recoveries to 2 of 63 systems. The abstract pages I opened do not state these details. Check before citing.

### 5.3 Synthesis by proposed LLM role

| Role | Evidence | Verdict |
| --- | --- | --- |
| Semantic priors and warm starts | LLAMBO, LeGIT, BioDiscoveryAgent, LLMNN, AutoElicit, LABO | Real but front-loaded, confounded with memorization on public benchmarks, and best consumed by a numerical loop. |
| Adapting to measurements in context | Gupta et al., Krishnamurthy et al., EVOLvE, Schmied et al., BoxingGym | Weak without training. Numerical methods dominate once data accumulate. |
| Hypothesis and mechanism-family proposals | Li, Fox, Goodman 2024, Model Discovery Agent, LLM-ACES, FunBO | Promising, and in each case numerical inference or disagreement does the selecting. |
| Natural-language objectives, constraints, unstructured side information | AuxBO, LLM Processes, LMABO | Plausible unique value because numerical planners cannot read this input at all. |
| Very large or unstructured action spaces | BioDiscoveryAgent (about 19,000 genes), ToSFiT, LICO, Kristiadi | Plausible value as a proposal distribution. An eight-action menu offers no such headroom. |
| Model misspecification | Rainforth et al. 2024 Section 5.3 on fragility of adaptive design, BoxingGym on LLMs choosing over-simple forms | Open. No evidence yet that an LLM policy is more robust. |

The consistent conclusion is that a numerical planner with a correct model and full numerical state is not where an LLM wins. The current screen (known graph, linear mechanisms, known noise, eight actions, constant prose, anonymous variables X, Y, Z) is exactly that case. The agenda already says an LLM cannot beat an exact Bayes-optimal rule under the same model and information (agenda line 49). The literature agrees and adds that distilled LLM students trail their numerical teachers (EVOLvE).

## 6. Failure modes that could explain one constant action, with diagnostics

Each item: primary source, why it could apply given Section 1, a concrete diagnostic that needs no new training unless stated, and a remedy. These are hypotheses. The counts in Section 1 do not establish any of them.

### 6.1 The policy learned the label marginal (majority-class or frequency collapse)

- Sources: Schmied et al. [V-full] (frequency bias, 96% frequent-action fraction at 2B, attributed to supervised pre-training). Menon et al., "Long-tail learning via logit adjustment", ICLR 2021, arXiv:2007.07314 [V-abs]. Zhao et al., "Calibrate Before Use", ICML 2021, arXiv:2102.09690 [V-abs] (the abstract names recency and common-token bias. "Majority label bias" is the paper-body term, [MEM]).
- Why plausible here: 160 single-example SFT steps on a 0.6B model. `set_x_neg` is the most frequent intervention chosen by both the target-risk and local-information policies on the development systems (129 and 149). The numerical ranker also collapsed.
- Diagnostics: (a) teacher label histogram and the entropy of the label given coarse state features on the saved training set. (b) **Prior-only baseline:** compute the SFT loss and DPO loss achieved by the best state-independent action distribution and compare with the trained model's loss on the same examples. Equality means only the marginal was learned. (c) **Content-free probe:** score the eight commands on a prompt whose state is replaced by a neutral placeholder (Zhao et al.). If the ranking equals the ranking on real states, the state is not being used. (d) Compare with a logistic or small MLP student on the same rows.
- Remedies: balanced sampling or logit adjustment (Menon et al.), soft targets (6.4), many more distinct states, report the prior-only baseline as a standing control.

### 6.2 DPO dynamics on near-identical short completions

- Sources: Razin et al., "Unintentional Unalignment: Likelihood Displacement in DPO", ICLR 2025, arXiv:2410.08847 [V-abs] (preferring "No" over "Never" raised "Yes". Displacement is driven by pairs with similar hidden embeddings, measured by the CHES score, and filtering such pairs helps more than adding an SFT term). Pal et al., "Smaug", arXiv:2402.13228 [V-abs] (DPO can lower preferred likelihood, worst when the edit distance between the pair is small. DPOP adds a penalty). Feng et al., arXiv:2404.04626, draft [V-abs] (DPO decreases dispreferred probability faster than it increases preferred). Yan et al., "3D-Properties", ICLR 2025, arXiv:2406.07327 [V-abs] (drastic drop of rejected likelihood, degradation into suppression, dispersion onto unseen responses). Cho et al., BDPO, arXiv:2506.12725 [V-abs] (rejected term dominates). Tajwar et al., ICML 2024, arXiv:2404.14367 [V-abs] (negative-gradient objectives are mode-seeking and move categorical mass fast). Azar et al., arXiv:2310.12036 [V-abs for existence. The claim that DPO overfits under near-deterministic preferences and that the identity variant is called IPO is [MEM]]. Xu et al., ICML 2024, arXiv:2404.10719 [V-abs: "may have fundamental limitations". The out-of-distribution exploitation detail is [MEM]].
- Why plausible here: every chosen/rejected pair is two JSON strings that differ in one to three tokens and share a prefix, which is the small-edit-distance and similar-embedding regime. With eight possible commands, mass removed from a rejected command must go somewhere, and DPO does not say where. One pair per step for 119 steps, one pass.
- Diagnostics: (a) from saved per-step artifacts, plot log-probability of chosen, rejected and each of the other six commands across DPO steps. Displacement shows as rising mass on a third command. (b) Tabulate chosen and rejected action identities over the 119 exposed pairs and over all 3,584 comparisons: if one action is chosen far more often than rejected, a state-independent policy lowers the loss. (c) CHES-style similarity for each pair. (d) Fraction of preference margins within replicate noise.
- Remedies: listwise objectives over all eight commands (LiPO, Liu et al., NAACL 2025, arXiv:2402.01878 [V-abs]. Plackett-Luce variant of DPO [MEM]). DPOP or bounded variants. SFT-anchored preference loss as in Paprika's RPO. Robust DPO for noisy labels (Chowdhury, Kini, Natarajan, arXiv:2403.00409 [V-abs], random flip rate ε, bound scales as 1/(1−2ε)). Paprika's practice of one pair per task. Or leave preference optimization for on-policy RL (6.7).

### 6.3 Behavior cloning pathologies

- Sources: Ross, Gordon, Bagnell, DAgger, AISTATS 2011, arXiv:1011.0686 [V-abs]. Wen, Lin, Darrell, Jayaraman, Gao, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories", NeurIPS 2020, arXiv:2010.14876 [V-search]. de Haan et al., causal confusion, NeurIPS 2019 [MEM]. EVOLvE [V-full].
- Why plausible here: the prompt contains the paid history, including past actions. If teacher actions are serially correlated, "repeat what the history shows most" is a cheap predictor. Students are deployed on states their own actions create, which the teacher data never covered (a constant policy creates histories with one action only).
- Diagnostics: (a) on saved teacher trajectories, accuracy of "copy previous action" and "copy most frequent action in history" as predictors of the teacher label. (b) Student accuracy split by whether the label equals the previous action. (c) Score the student on teacher states versus on its own deployment states.
- Remedies: DAgger-style relabeling of student-visited states by the planner, history summarization that removes the raw action sequence, EVOLvE-style algorithm-guided inputs.

### 6.4 Hard winner labels discard the planner's graded information

- Sources: Qiu et al. [V-full] (imitating a Bayesian model's probabilistic predictions generalizes better than oracle labels). Rusu et al., "Policy Distillation", ICLR 2016, arXiv:1511.06295 [V-search] (KL to temperature-scaled teacher Q-values). Lukasik et al., "Better autoregressive regression with LLMs via regression-aware fine-tuning", ICLR 2025 [V-search]. Hong, Dragan, Levine, Q-SFT, arXiv:2411.05193 [V-abs] (token probabilities trained to represent Q-values through a weighted cross-entropy, ICLR 2025 per search). LiPO [V-abs].
- Diagnostic: distribution of the planner's score gap between best and second-best action on training states. Many near-ties make hard labels close to noise.
- Remedy: train the eight-way command distribution toward a softmax of planner scores or measured cost-normalized gains (listwise KL), or regress scores.

### 6.5 Weak numeracy, digit tokenization and long raw numeric context

- Sources: Yang et al., "Number Cookbook", ICLR 2025, arXiv:2411.03766 [V-search] (LLMs fail frequently, worst on longer numbers and less common formats. Naive fine-tuning helps many but not all tasks, and special number techniques did not help fine-tuning of pretrained models). Singh and Strouse, "Tokenization counts", arXiv:2402.14903 [V-search]. Gruver, Finzi, Qiu, Wilson, "LLMs Are Zero-Shot Time Series Forecasters", NeurIPS 2023, arXiv:2310.07820 [V-abs] (GPT-4 can be worse than GPT-3 "because of how it tokenizes numbers"). Golkar et al., xVal, arXiv:2310.02989 [V-search]. Zausinger et al., "Regress, Don't Guess", ICML 2025, arXiv:2411.02083 [V-search]. Zhou et al., FoNE, ICLR 2026, arXiv:2502.09741 [V-search]. Liu et al., "Lost in the Middle", TACL 2023, arXiv:2307.03172 [V-abs]. Krishnamurthy et al. [V-abs]. EVOLvE [V-full]. Vacareanu et al., arXiv:2404.07544 [V-search] (frontier models regress well in context, which bounds the pessimism but concerns far larger models). Qwen digit-level tokenization: one secondary source for Qwen2.5 only, nothing for Qwen3. Unverified.
- Why plausible here: Section 1 item 1. A 0.6B model reads full-precision floats for posterior means and covariances inside one JSON line, with no reasoning tokens, and must emit a decision immediately.
- Diagnostics: (a) token count per prompt and tokens per number, from the saved `prompt_tokens` metadata. Confirm distinct states yield distinct token sequences (the handoff asks this). (b) **Linear probe:** from hidden states at the decision position, can a linear readout recover the planner's per-action scores or the largest posterior variance? Failure implicates representation. Success with a constant action implicates the objective or readout, which is the knowing-doing gap of Schmied et al. (c) Sensitivity curves: perturb one posterior variance or one cost in a saved prompt and record the change in the eight-way command distribution for frozen, SFT and DPO adapters.
- Remedies: summarized, rounded, decision-relevant features. Per-action statistics as text (EVOLvE). Learned numeric projector feeding soft tokens (LICO pattern). Number-aware losses or embeddings (xVal, NTL, FoNE). Note the Number Cookbook finding that add-on number techniques did not help fine-tuning of pretrained models, so favor input redesign.

### 6.6 Scoring, decoding and label-prior artifacts

- Sources: Tan et al., TWOSOME, ICLR 2024, arXiv:2401.14151 [V-full] (policy from joint token probabilities of action strings. Longer action strings get lower joint probability. No normalization gave fast early learning then "sudden performance drop in three of the four tasks". Word normalization fixed it). Carta et al., GLAM, ICML 2023, arXiv:2302.02662 [V-search]. Zheng et al., "LLMs Are Not Robust Multiple Choice Selectors", ICLR 2024, arXiv:2309.03882 [V-search] (prior mass on option identifiers, PriDe estimates it by permuting option contents). Park et al., "Grammar-Aligned Decoding", NeurIPS 2024, arXiv:2405.21047 [V-search] (constrained decoding distorts the model's distribution).
- Why plausible here: commands have different token lengths, training uses full-sequence log-probability, and deployment uses greedy prefix-constrained decoding, so the deployed choice need not be the highest-probability full command. `observe_all` is first in alphabetical identifier order (the order the grammar uses. I did not check the order shown in the prompt's menu) and is the only non-`set_` action, which is what the frozen model chose 574 of 576 times. `set_x_neg` is the first `set_` action in alphabetical order.
- Diagnostics: (a) for saved prompts, compare the greedy constrained choice with the argmax of full-command log-probability and with length-normalized variants. (b) Permute menu order and rename identifiers to neutral codes, then re-score with the frozen and trained adapters. An unchanged surface choice means identifier or position prior. (c) Count tokens per command.
- Remedies: score all commands and take the normalized argmax, single-token action codes of equal length, random identifier permutation in training and evaluation (as in Kriz et al.'s Cube-NM), contextual calibration.

### 6.7 Offline, tiny, exploration-free training on a very small model

- Sources: Tajwar et al. 2024 [V-abs] (on-policy sampling helps). Kriz et al. [V-full] (Qwen3-0.6B succeeds with on-policy GRPO, temperature 2 and an entropy bonus). Schmied et al. [V-full] (exploration bonus raised action coverage from 50% to 70%). Biderman et al., "LoRA Learns Less and Forgets Less", TMLR 2024, arXiv:2405.09673 [V-abs]. Scale comparison: this project 160 + 119 single-example steps with rank 8 on q/v. Kriz et al. 250 steps × 128 states × 8 rollouts with rank 16 on seven projection types. ASIG rank 32 on all attention and MLP projections. Paprika 17,181 trajectories + 5,260 pairs on 8B. Qiu et al. 6,240 examples with full fine-tuning on 7-9B. EVOLvE 50-200 teacher trajectories of 200-1000 steps each.
- Diagnostic: learning curves of held-out teacher-agreement and of state sensitivity (6.5c) versus number of updates and distinct states, using existing checkpoints if intermediate ones were saved. I did not check whether they were.
- Remedy, only after diagnosis: a predeclared scaling ladder over data, steps, LoRA capacity and model size, with the numerical student trained on identical data at every rung.

### 6.8 No reasoning tokens

- Sources: Schmied et al. ("without CoT, RLFT barely attains the performance of ICL w/ CoT"). Kriz et al. (direct amortization without reasoning "does not produce robust transfer across domains"). Krishnamurthy et al. (chain-of-thought plus summary was the only working configuration). Counterpoint: Kriz et al.'s Cube-NM policy on Qwen3-0.6B emits an identifier with no reasoning and still learns, on a categorical task.
- Diagnostic: frozen-model comparison with thinking on versus off on saved prompts (inference only, needs a GPU slot, so it is a later step).

## 7. What a strong reviewer at a statistics-leaning ML venue would demand

### 7.1 Comparators

1. **Bayes-optimal references for the tractable case.** With linear mechanisms, known graph, Gaussian noise of known level and a Gaussian prior, the posterior covariance of mechanism coefficients depends on the design and not on the observed responses. Chaloner and Verdinelli (1995), "Bayesian Experimental Design: A Review", Statistical Science 10(3):273-304 [read in a preprint PDF of the paper: Section 2.1 gives posterior covariance σ²(nM(η)+R)⁻¹ as "a function of the design η and of the prior precision matrix", Section 2.2 reduces expected Shannon gain to det{nM(η)+R}, and Section 1.4 covers per-observation costs under a fixed total cost]. **Inference, mine:** for covariance-based criteria in that setting an optimal design can be fixed in advance, so the value of adaptivity is limited to what breaks the conjugate picture (downstream regressors that depend on unknown upstream coefficients under observation, the normalized-error endpoint, heterogeneous costs, the stop decision). A reviewer will ask for (a) the cost-constrained Bayesian A-/c-/I-optimal static design per system, (b) a finite-horizon dynamic-programming or exhaustive lookahead reference where the eight-action menu and small budget make it feasible, and (c) the resulting **headroom table**: random, coverage, best static design, myopic planner, non-myopic reference. If the best static design already matches the planner, the screen cannot reward state-dependent behavior and should be redesigned before more training. Huan and Marzouk (arXiv:1604.08320 [V-abs]) verify a sequential design method against analytical linear-Gaussian solutions and treat batch and greedy designs as special cases of the dynamic program.
2. **Goal-oriented information criteria matched to the endpoint:** expected predictive information gain (Bickford Smith, Kirsch, Farquhar, Gal, Foster, Rainforth, "Prediction-Oriented Bayesian Active Learning", AISTATS 2023, arXiv:2304.08151 [V-search]), GO-CBED, ABCI-style query targeting. Parameter-information criteria alone are a weak comparator for a prediction endpoint.
3. **A properly trained amortized numerical policy** on the identical state and identical training outcomes: GO-CBED or CAASL-style transformer or a small MLP, trained with RL or listwise distillation and with a training-budget sweep. The current learned ranker collapsed and scored 13.583, worse than coverage (12.458), so it does not yet represent this family.
4. **Semantic-feature numerical baseline** whenever language carries information: LinUCB or GP on the same LLM's embeddings (Gupta et al.).
5. **LLM ladder:** frozen, frozen with planner statistics in the prompt, SFT, RL or preference tuned, each with and without reasoning, plus one larger base model, plus an inference-time BED-LLM-style baseline where applicable.
6. **Hybrid controls:** planner alone, planner + LLM tie-breaker, LLM selecting among numerical acquisition rules (LMABO-style), LLM warm start then planner (LeGIT-style).

### 7.2 Controls that identify what the LLM contributes

1. Shuffled or randomized feedback applied to policy input only (Gupta et al.).
2. State-blind and state-permuted fine-tuned controls, and the prior-only loss baseline (6.1).
3. Menu-order permutation and identifier renaming (Zheng et al., Kriz et al.).
4. Correct, anonymous, irrelevant and misleading metadata crossed with real and shuffled feedback, with recovery delay as an endpoint (already in agenda line 39). Use freshly generated systems and names so memorization is excluded (BioDiscoveryAgent used an unpublished dataset. LeGIT did not control this).
5. Graph ablation trained and evaluated without graph access (agenda line 125).
6. Equal-information accounting: any planner statistic given to the LLM must be given to numerical learners, and if per-action planner scores are in the prompt, an argmax-of-prompt baseline must be reported, because EVOLvE shows the LLM then "only needs to perform addition and argmax".
7. Training-seed replication. Kriz et al. flag single-training-seed results as not estimating variability. This project has one adapter per arm.

### 7.3 Statistical reporting

1. Independent systems as paired units, stratified bootstrap intervals, performance profiles and interquartile means rather than means alone (Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021, arXiv:2108.13264 [V-search]). Active-learning evaluation pitfalls: Lüth et al., NeurIPS 2023, arXiv:2301.10625 [V-search].
2. Treat capped cost as censored time-to-event: attainment curves, restricted mean cost, sensitivity to the 0.04 threshold and the cap of 32 [method suggestion, MEM]. Ten of 24 systems met the target after initialization (handoff line 36), so they carry no information about policies on the cost endpoint. Report effective sample size and choose thresholds or initial budgets that leave headroom.
3. Pre-specified primary contrast, multiplicity control, power analysis from development variance (agenda line 159).
4. Full error-versus-cost curves and final-error distributions, since ordering by cost and by final error can differ (the DPO arm had lower mean final error than random, handoff line 38).
5. Amortization accounting: training collection (10,240 cost units here) against per-episode savings, and inference cost (agenda line 53).
6. Misspecification stress tests and at least one independent benchmark (BoxingGym, CausaLab, Causal Chambers data).

## 8. Candidate method fixes, in the order the evidence supports

Diagnosis first (handoff item 1). The order below reflects strength of external evidence and the cost of testing.

1. **Fix scoring and identifiers before anything else** (cheap, deterministic): full-command normalized scoring instead of greedy prefix decoding, equal-length neutral action codes, random identifier permutation per example. Sources: TWOSOME, Zheng et al., Kriz et al., Park et al.
2. **Replace hard labels with graded targets:** listwise KL to a softmax of planner scores or measured cost-normalized gains over all eight commands. Sources: Qiu et al., Rusu et al., LiPO, Lukasik et al., Q-SFT.
3. **Rebalance and widen supervision:** balanced or logit-adjusted training, far more distinct states, DAgger relabeling on student-visited states. Sources: Menon et al., Ross et al., EVOLvE, Wen et al.
4. **Redesign the state text:** short, rounded, decision-relevant summaries (per-mechanism uncertainty, per-query predictive variance, costs, budget). Then a separate arm with per-action planner statistics. Sources: Krishnamurthy et al., EVOLvE. Keep the two arms separate because the second changes what the LLM contributes (7.2 item 6).
5. **If preference learning stays:** SFT-anchored loss (RPO as in Paprika), DPOP or BDPO, CHES filtering, robust DPO for noisy margins, one pair per state. Sources: Tajwar et al. 2025, Pal et al., Cho et al., Razin et al., Chowdhury et al.
6. **Prefer on-policy RL with exploration pressure for the feedback stage:** GRPO or PPO with entropy bonus and high-temperature rollouts, reward = measured cost-normalized risk reduction, short multi-step returns. Sources: Kriz et al. (same base model), ASIG, Schmied et al., Tajwar et al. 2024, Xu et al.
7. **Reasoning before action,** trained with the reward. Sources: Schmied et al., Kriz et al., Krishnamurthy et al.
8. **Numeric input pathway:** learned projector from the numeric state to soft tokens (LICO pattern) or number-aware embeddings and losses (xVal, FoNE, NTL). Lower priority given the Number Cookbook result.
9. **Capacity ladder:** LoRA on all attention and MLP projections at rank 16-32, then a 4-8B base. Sources: ASIG, Kriz et al., Biderman et al., Schmied et al. size gradient, EVOLvE small-model result.
10. **Change the LLM's job to one the literature supports:** semantic warm start then planner (LeGIT, LLMNN), acquisition-rule selection (LMABO), reading natural-language objectives, constraints and side information (AuxBO), proposing mechanism families under misspecification (Li, Fox, Goodman. Model Discovery Agent), writing the acquisition rule as code (FunBO), or narrow interventional-oracle queries inside a Bayesian loop (Roy and Parbhoo). Each needs the equal-information controls of 7.2.

## 9. Surprises and contradictions

1. **A public arXiv preprint from this project makes claims the completed screen does not support.** A generic search returned arXiv:2602.02451, "Active Causal Experimentalist (ACE): Learning Intervention Strategies via Direct Preference Optimization", v1 2 Feb 2026, v2 18 Jun 2026 [V-abs]. Its abstract claims a "70-71% improvement over baselines at equal intervention budgets", p < 0.001, Cohen's d about 2, across synthetic benchmarks, physics simulations and economic data, and says the learned policy "autonomously discovers that collider mechanisms require concentrated interventions on parent variables". The 24-system screen has the DPO policy last of eight with one constant action. The agenda (line 21) already says historical DPO numbers sit alongside evaluator audits and must not be recycled. Any new paper will be read next to this public record, so it needs an explicit reconciliation or a superseding statement. I did not read the preprint body and make no claim about why the results differ. The same headline is still in the repository's `/Users/pat/code/ACE/README.md` (line 14: "**70-71% improvement** over baselines at equal intervention budgets", and table rows at lines 114 and 116. Read-only `git log` shows the README last changed 2026-02-28). The preprint identifier does not appear in the guidance documents, the handoff or the results report (targeted grep of `docs`, `paper` and `README.md`).
2. **Collapse is not specific to the language model.** The learned numerical ranker used two actions (435 and 141) and never observed (`actions.csv`). The report text does not mention this.
3. **The agenda's novelty hedge on "supervised-then-preference" is out of date.** Paprika does SFT then SFT+DPO for information gathering.
4. **Same base model, opposite outcome.** Kriz et al. train Qwen3-0.6B with LoRA to a state-dependent selection policy. Differences from this project: categorical text state, on-policy RL with entropy bonus, permuted identifiers, 250 large-batch updates.
5. **The remedy with the strongest evidence conflicts with the project's stated requirement.** Supplying planner statistics makes cloned policies work (EVOLvE), but then the LLM "only needs to perform addition and argmax", which is the narrator role the handoff rules out (handoff line 13).
6. **Positive LLM-for-intervention evidence (LeGIT) lacks the anonymization control** that the agenda already plans, and its gains are small on small graphs.
7. **One on-topic 2026 preprint was withdrawn** (arXiv:2606.31184).
8. **Fine-tuning helps exploration but leaves a gap to the numerical teacher** in the cleanest study (EVOLvE), and frequency bias persists after RL fine-tuning at high repetition counts (Schmied et al.).

## 10. Items I could not verify or only partly verified

- Chaloner and Verdinelli: read from a preprint PDF hosted on a university course page, pages 1-10 only. Section numbers may differ from the journal version. The "adaptivity gives little in the conjugate linear case" statement is my inference from the verified formulas, not a quoted claim.
- Kriz et al. (arXiv:2610.20832): characters 0-140k of about 156k read. Figure-only numbers (for example the direct-versus-reasoning ablation) not extracted. No venue.
- Paprika: SFT-only versus RPO ablation numbers are in a figure. LoRA settings not extracted.
- Schmied et al.: the ICLR 2026 venue comes from the ICLR proceedings search hit, not the arXiv page.
- [V-search] items generally: bibliographic data from search snippets of the primary page. Author lists for ABCI and Causal Bayesian Optimization are from memory.
- [MEM] items: the "majority label bias" term in Zhao et al., IPO naming and the deterministic-preference overfitting argument in Azar et al., the out-of-distribution exploitation argument in Xu et al., de Haan et al. 2019, Plackett-Luce DPO, censored-cost analysis suggestions, knowledge-gradient and other classical non-myopic references (not listed above for that reason).
- PhysGym and LLM-ACES anonymization findings: search snippet only. Not on the abstract pages.
- Qwen3 digit tokenization: not verified. One secondary source for Qwen2.5.
- GP numbers in Gupta et al.: possible table error flagged at extraction.
- nature.com page for Causal Chambers: redirected to an authorization flow, not followed. Verified through arXiv instead.
- LMABO and "Generative Bayesian Policies": abstract pages only. For the latter the page does not name the fine-tuning objective.
- I did not search for or verify work on stopping rules, cost-aware BO, or physical-testbed transfer. Those belong to the benchmark-focused literature agent.

## 11. Bottom line for the agenda

1. The contribution "LLM fine-tuned by DPO from information or outcome preferences chooses experiments" is occupied for language tasks and partly for optimization. The unoccupied claim is narrower and harder: a trained LLM policy that uses an explicit SCM state and measured outcomes, passes feedback-shuffling controls, and beats goal-oriented numerical planners on cost to target in a regime where the planner is handicapped for a stated reason (semantics, misspecification, action generation, natural-language objectives).
2. The present screen is a regime where the literature predicts no LLM advantage, and it may offer little reward for state-dependent behavior at all. Compute the headroom table before training again.
3. The constant action has at least seven documented candidate explanations that can be separated offline on saved artifacts. The cheapest discriminating checks are the prior-only loss baseline, the pair-identity tabulation, the decoding-versus-scoring comparison, the identifier permutation test and the numerical-student comparison.
4. If training resumes, the evidence favors graded listwise targets, summarized inputs, on-policy RL with exploration pressure, reasoning tokens and more capacity, with a numerical student trained on identical data at every step.
