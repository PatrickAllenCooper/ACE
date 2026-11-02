# ACE research agenda for learned intervention selection

Date: November 2, 2026. Status: research design, not an executed or frozen experiment.

## The contribution we are trying to establish

**Can an LLM trained to use an explicit causal model choose interventions that give experimentalists a reliable answer with less experimental cost than strong numerical policies?**

The LLM must make a consequential choice in the experimental loop. It cannot merely narrate a decision already made by a numerical optimizer. The SCM supplies a representation of mechanisms, intervention effects and uncertainty. The environment supplies measured responses. Learning should improve the policy that chooses what to manipulate, at which value, what to measure and when to stop.

The proposed contribution is a transferable intervention policy that combines semantic task information with a numerical SCM and learns from experimental feedback. This is a hypothesis to test, not an established novelty or performance claim. The primary practical endpoint is experimental cost at a useful level of accuracy. A separate engineering evaluation asks whether the learned model supports a good operating decision.

Patrick has made this LLM mechanism and its demonstrated utility a condition of submission. The delivery paper is preserved as a component study. We will not submit it as if it established the broader ACE claim. A negative result in the new research must narrow the claim or redirect the method, not trigger a favorable replacement benchmark.

## What existing evidence contributes

The completed delivery work establishes that prediction after collection is a separate design choice. Refitting can help, but its benefit depends on the comparator, collection history, initialization and compute. The five-node rolling loss, mixed thirty-node balanced result, worsening recorded history and Fourier wins remain useful controls for the new system. They warn against attributing a better final predictor to a better intervention policy.

The earlier Mission-1 study is more direct evidence about LLM proposals. It accepted 44 runs across four arms, with eleven paired acquisition seeds on one emulator. After common refits at equal charged calls, the reported LLM/random-proposer error ratio was 1.0412, with a reported 98.75% interval of 0.8870–1.2221 and improvement in only two of eleven seeds. It detected no advantage. The LLM and no-LLM arms also differed in GPU/CPU execution, and the LLM trajectory was not bit-reproducible. This is a substantial adverse result for that pipeline, not a universal equivalence result. It must remain part of the renewed project's evidence. [Mission-1 statistics and execution disclosures](/Users/pat/code/ACE-Runner/studies/2026-10-peter-baseline/package/stats_appendix.md).

Historical policy-training traces establish that updates ran, but do not establish that the updates helped. Some shipped bundles were selected for beating Random, so their published success rates cannot serve as an unbiased confirmation sample. Earlier DPO comparisons also sit alongside evaluator and environment audits. We will use those records to identify defects and design new tests, not recycle their favorable numbers as validated headline evidence. [Publication selection policy](/Users/pat/code/ACE-Runner/docs/decisions/0002-baseline-publication-policy.md), [DPO default decision and caveats](/Users/pat/code/ACE-Runner/docs/decisions/0003-policy-update-default-none.md).

The October 9 exploration tested mechanism predictors on fixed histories. It did not test a trained LLM intervention policy. Six language proposals failed the declared output contract and used numerical fallback. That is evidence against that particular interface, not evidence that language models cannot help experimental design. TabPFN showed conditional benefits and harms, and flexible numerical regression explained some large gains over an inadequate grammar. TabPFN is not an LLM. [Component results](ace_foundation_component_results_2026-10-09.md), [retention results](ace_foundation_retention_results_2026-10-09.md), [cycle closeout](ace_foundation_cycle_closeout_2026-10-09.md).

We reuse the existing intervention masks, response accounting, fixed-world descriptors, graph execution, learned-chain evaluation and failure preservation. We do not rerun accepted studies or treat the old development worlds as fresh confirmation data.

## Where the literature leaves room for ACE

The relevant literature already contains learned experimental policies, LLM acquisition and causal experiment design. Combining their names is not a sufficient contribution. The opportunity is to show that a trained, SCM-conditioned language policy transfers useful experimental behavior, responds to contradictory evidence and reduces experimental cost beyond strong numerical alternatives. This positioning is a research judgment, not a claim that novelty has been established.

- **Deep Adaptive Design**, Foster et al. (2021), trains a numerical history-to-experiment network using simulated campaigns and information bounds. It establishes amortized experimental design without language. We need a numerical policy trained with comparable supervision, and DAD itself where its simulator and objective assumptions fit. [Primary paper](https://proceedings.mlr.press/v139/foster21a.html).
- **Learning to Ask Informative Questions**, Mazzaccara et al. (2024), uses information-gain preferences to train an LLM question policy with DPO. Preference-trained information gathering is therefore prior work. Its 20 Questions setting does not establish numerical intervention selection or ACE's proposed supervised-then-preference training sequence. [Primary paper](https://aclanthology.org/2024.findings-emnlp.291/).
- **ASIG**, Hartmann et al. (2026), trains sequential LLM information gathering through multi-turn GRPO and estimated information gain. It is a workshop paper with limited transfer evidence. ACE must test physical variables, costs and causal intervention semantics rather than repackage question selection. [Primary paper](https://arxiv.org/html/2607.03426v1).
- **Model-based Causal Bayesian Optimization**, Sussex et al. (2023), learns local mechanisms on a known graph and propagates uncertainty to choose interventions. It is a strong comparator for outcome optimization. Adapting its mechanism model to prediction-risk reduction changes its objective and does not inherit its published regret guarantee. [Primary paper](https://arxiv.org/abs/2211.10257).
- **LLAMBO**, Liu et al. (2024), uses frozen LLMs for warm starts, surrogate prediction and candidate sampling. **LMABO**, Ngo et al. (2026), uses a frozen LLM to select acquisition functions. These motivate separate direct-action and numerical-tool-selection treatments. Neither makes ACE-specific policy fine-tuning an established benefit. [LLAMBO](https://arxiv.org/html/2402.03921v2), [LMABO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1b7a29fe9759d4c9d23f78738e40b42-Abstract-Conference.html).
- **Model Discovery Agent**, Murphy (2026), is a preprint combining LLM mechanism proposals with numerical inference and experiment design. Hypothesis generation is an alternative LLM role, but it must be tested separately from learning the intervention policy. Simply adding generated mechanisms to a numerical selector is not a new general architecture. [Primary paper](https://arxiv.org/html/2608.09696v5).
- **LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?**, Gupta et al. (2025), uses shuffled feedback and embedding-informed controls to distinguish useful prior knowledge from adaptation to measurements. We should cross correct, anonymous and misleading metadata with real and shuffled feedback. Any shuffled-feedback arm must corrupt the declared policy input only, leaving the environment, true records and private evaluator intact. [Primary paper](https://aclanthology.org/2025.findings-emnlp.838/).

The focused hypothesis is: **a policy trained on measured intervention consequences can use semantic priors when they are helpful, abandon them when measurements disagree, and transfer this behavior to new mechanism families or legal action menus.** Measure recovery delay after misleading prior information alongside prediction and decision loss. Compare with a semantic-feature numerical policy so that language access is not secretly extra information. Benefits that survive shuffled feedback may reflect a useful prior, but do not by themselves show learning from experiments.

## Theoretical program

Treat the experimental state as a supplied graph, paid history, fitted mechanism distribution or ensemble, objective and remaining budget. The LLM is an approximate decision rule over that state. This separates three sources of error: the causal model may value an intervention incorrectly, the candidate generator may omit useful interventions, and the policy may choose poorly among the candidates it receives.

An elementary starting result makes this separation explicit. At one fixed history let V(a) be the true expected improvement in a declared loss, net of a fixed cost penalty. Let A be a finite legal action set and C a nonempty candidate subset. Suppose the learned value estimate has uniform error at most epsilon on C, and the chosen action has estimated value within delta of the largest estimate on C. If kappa = max over A of V minus max over C of V, then the one-step value loss is at most kappa + 2 epsilon + delta. To see this, compare the best legal action with the best candidate, replace the candidate's true value by its estimate, apply the delta selection condition, and replace the selected estimate by its true value. The two replacements each cost at most epsilon.

This is a conditional decomposition, not an ACE performance certificate. We do not know a uniform epsilon for an arbitrary learned SCM. It also concerns one-step utility, so greedily optimizing it does not establish the best finite-budget sequence. Training-world rollouts can estimate the three terms and reveal where more model capacity or planning helps. A later sequential result would need explicit transition, uncertainty and horizon assumptions.

The decomposition motivates three methodological tests. A broader proposal distribution may lower candidate omission but increase scoring computation. Better calibrated or more flexible mechanisms may reduce value error while changing the learner comparison. A trained policy may reduce decision error or approximate several steps of lookahead at lower inference cost. Each improvement must be measured independently. An LLM cannot beat an exact Bayes-optimal decision rule under the same correct model, information and objective by definition. Its plausible advantages lie in practical computation, useful transferable priors and recovery from an inadequate model.

Use paired training rollouts to teach action consequences at the same history and budget. Start with one-step feedback and compare a fixed short-horizon return only if the extra training cost is justified. Do not infer long-horizon planning from a one-step preference objective. The causal graph can make credit assignment more localized, but the training loss must still include the downstream scientific objective because local prediction gains can worsen composed predictions.

There is also a simple economic test for amortization. If reusable training costs C_train, numerical acquisition costs c_num per decision and the trained policy costs c_llm, compute savings require N times (c_num minus c_llm) to exceed C_train over N decisions, with comparable experimental performance. If c_llm is larger, no such compute break-even exists. Experimental savings can still matter when measurements are expensive, but those savings and compute costs must be reported separately rather than converted with an unstated exchange rate.

## The proposed mechanism

### Inputs and outputs

At decision step t, the policy receives:

- The experimental objective and a declared accuracy or decision tolerance.
- A known or explicitly supplied candidate graph, variable names, units and permissible intervention targets. Graph discovery is a later extension, not an assumption silently added to the first study.
- Legal ranges, prohibited combinations, measurement choices, response costs and remaining budget.
- The paid observation history and a fixed-size summary of fitted mechanisms, prediction uncertainty, residuals and joint support.
- Optional domain notes whose provenance and information content are controlled in the experiment.

The output is a typed action, such as `intervene(targets={speed: value}, measure=[pressure, temperature, power])`, `replicate(action_id)`, `observe(measurements)` or `stop`. Only capabilities implemented by the environment are legal. Internal interventions are permitted only when the experimental system can actually perform them. Observation and replication are charged too.

The executor validates target names, units, bounds and joint constraints before any experiment. A constrained action grammar prevents arbitrary generated code from becoming an intervention. Invalid proposals, fallback use and token costs remain visible. A fallback policy is a separate recorded behavior, not a successful LLM choice.

Parse only newly generated tokens. The existing Runner records a defect in which a prompt example could be returned as a leniently parsed action when generation contained no command. New action logs must bind the completion, parsed action and executed command. [Known parser defect](/Users/pat/code/ACE-Runner/docs/code_health.md:164).

The loop is:

1. Summarize the current SCM, observations, objective and legal actions.
2. Let the LLM propose or rank interventions. If a candidate menu is used, give every competing policy the same menu and numerical information.
3. Check feasibility, execute one selected intervention and record its full cost and response.
4. Update the eligible SCM mechanisms using the new observation.
5. Return the updated state to the policy. Stop only under a declared rule.

SCM prediction calls can help the policy compare hypotheses, but learned-model predictions are not additional environment observations. Limit their number and report their computation. A rationale may help an experimentalist inspect the choice, but fluent text is not evidence that the choice was useful.

### How the LLM learns

Begin with an open-weight model whose license, weights, tokenizer and training implementation can be pinned. Choose a small viable model after profiling, rather than assuming a larger model is necessary. Use three separately identified policy versions:

1. **Frozen pretrained policy.** The model receives the same action schema, state and tools but no ACE-specific weight updates. This tests whether pretraining and prompting alone suffice.
2. **Supervised policy.** Fine-tune adapters on training-world trajectories from several numerical teachers, including uncertainty and information-based design. This teaches the action contract and useful experimental behavior. Include teacher failures and difficult cases. Matching the teacher alone is not an advance over it.
3. **Feedback-trained policy.** Starting from the supervised model, train on preferences or returns measured in training environments. Compare DPO-style preference learning with a simpler supervised action ranker before introducing a larger reinforcement-learning system.

Preferences must reflect measured improvement in a predeclared task loss, adjusted for intervention cost. On training worlds only, fork the same pre-action learner state for a small fixed set of candidate interventions, execute the branches and evaluate the resulting predictors on an independent training-world scoring set. Charge every branch and scoring response. Use uncertainty-aware ties rather than turning negligible noisy differences into strong preferences. Do not reward a candidate using the live learner state after another candidate already changed it.

First test deployment with policy weights frozen throughout a new episode. This measures transfer and separates policy training from ordinary SCM updates. Online LLM fine-tuning within a deployment episode is a later, separately costed ablation. It is not necessary to claim that the policy was fine-tuned before deployment.

Define frozen weights by behavior and hashes, not by `policy_update=none`. Runner's cloud profile can perform initial and periodic teacher SFT while DPO is disabled. Disable every update path for the frozen arm. Train action likelihoods conditional on the actual experimental state, with prompt tokens masked from the action loss. Runner's historical DPO uses command-only likelihoods, while the research implementation uses prompt-plus-command sequences. Choose and test one explicit state-conditioned objective rather than treating those implementations as equivalent. [Cloud configuration](/Users/pat/code/ACE-Runner/ace/config.py:278), [command-only likelihood](/Users/pat/code/ACE-Runner/ace/dpo.py:287).

All policy-training worlds, scoring responses and checkpoints are separate from final evaluation worlds. The LLM never sees the private evaluation outputs, hidden mechanism parameters or final stopping score. SFT and DPO are implementation choices, not proofs of optimal intervention selection.

Route every deployment measurement through one charged executor. The research critic can query true mechanisms directly, so selecting `lookahead_on_student` alone does not remove privileged feedback. Replace such deployment critic access with learned-model uncertainty or an explicitly paid validation set. Training-only oracle access is allowed only as labeled, charged simulation supervision. [Critic implementation](/Users/pat/code/ACE/ace_experiments.py:865).

## Experiments that distinguish the explanations

### First comparison: change the policy, hold the learner fixed

Use one fixed SCM learner, fitting schedule, initial data set and final refit rule for every acquisition policy. Compare:

- Uniform random selection from the legal intervention menu.
- Coverage-based selection with no language model.
- A numerical uncertainty or expected-information policy using the same SCM and observations.
- A learned nonlanguage action ranker with the same numerical state, training trajectories and candidate menu.
- The frozen LLM, supervised LLM and feedback-trained LLM.

Choose the strongest numerical comparator on development worlds, then freeze it. Retain all comparator results. Random is a useful reference, but beating random alone is insufficient for the headline. A deterministic coverage policy must not be labeled random.

For a prediction objective, information-based acquisition is the appropriate main numerical competitor. For an optimization objective, include a competitive Bayesian optimization policy such as expected improvement, Thompson sampling or a causal model-based optimizer. Do not use a deliberately weak exhaustive grid as the only noncausal baseline.

On tractable linear-Gaussian fixtures, use exact posterior updates as a reference. On nonlinear tasks, compare expected reduction in held-out intervention-prediction risk as well as information gain, since information about nuisance mechanisms need not help the declared objective. Provide the numerical student with the same preference pairs or rollout returns used to train the LLM. Give numerical comparators equivalent semantic features where such features are supplied to the LLM.

### Second comparison: identify what the LLM contributes

The essential contrasts are trained versus frozen LLM, trained LLM versus the learned numerical ranker, and the full system versus its version without access to causal structure. Use the same environment interface and data budget.

Test meaningful domain descriptions against anonymized variable names. Also test incorrect or irrelevant descriptions as a robustness condition. Any extra structured facts must be available to numerical policies in a usable encoding. A win from supplying only the LLM with the true mechanism family is not a language-reasoning effect.

A graph ablation must be trained and evaluated with the corresponding graph access removed. Simply deleting graph tokens from a model trained to depend on them is a distribution shift test, not a clean component ablation. Preserve target/constraint information in every arm.

Separate proposal quality from selection quality. For the historical K-candidate route, cross LLM, random and numerical proposals with the same learned-model scorer and with random selection among the K candidates. Hold K and the intervention budget fixed. This reveals whether the LLM supplies useful candidates or whether the scorer and hand-coded bonuses account for the result. A direct learned-action policy is a distinct treatment. At deployment, unselected candidates do not receive free simulator outcomes.

### Third comparison: separate acquisition from final prediction

After the policy comparison, cross a small frozen set of policies with structured and unstructured predictors. Give both predictors the same measured outputs where feasible. An SCM-free predictor must receive intervention masks and values so that it can represent the permitted queries. A roots-only predictor is not a valid control for arbitrary internal interventions.

Score both the rolling model and the same predeclared final refit. If a benefit appears only after extra fitting, report it as a policy-plus-refit result. The existing delivery study supplies the reason for this control, not the outcome of the new comparison. Differences in supervision, parameter count and compute remain explicit.

## Benchmarks and a concrete experimentalist use case

Use a ladder of increasing difficulty rather than one hand-picked success case.

**Controlled intervention tasks.** Start with fresh chains, forks and colliders whose graph is supplied but mechanisms are unknown. Include a correctly specified prior, missing mechanism families, coefficient changes and no-change controls. Vary observation noise and intervention costs. Keep the old unfavorable regimes. Split by graph templates and mechanism families as well as random seeds, so transfer is more than interpolation between coefficients.

**Independent interactive environments.** Adapt suitable environments from BoxingGym or another primary experimental-design benchmark. Check that their actions, losses and observation budgets match our question. A benchmark may support experimental design without supporting arbitrary causal interventions. Do not label every experiment as a do-operation unless the environment implements that semantics.

**Engineering transfer.** Use an actual executable compressor model or a manipulable physical testbed. OpenModelica's ThermoPower compressor is a candidate, not yet an integrated ACE environment. It needs performance maps, boundary conditions, numerical validation and legal actuator definitions. A steady-state causal surrogate must be distinguished from the simulator's coupled or dynamic equations. Time-dependent experiments require an explicitly time-unrolled model or a validated settling procedure.

The illustrative objective is to reduce electrical or shaft power while delivering a required pressure and flow without exceeding a temperature limit. Inputs include permissible speed and valve settings, current sensor readings, costs and the task objective. Outputs include the next intervention and, eventually, a recommended operating setting with measured performance. A possible causal sketch links speed and valve setting to flow and pressure, then to power and temperature. This sketch must be verified against the chosen simulator before being treated as its graph. ACE might choose an intervention that distinguishes an efficiency change from a restriction, rather than sweeping all settings. That is the hypothesis the demonstration should make visible.

An engineering result should report interventions used, time and money where meaningful, energy or power regret at the recommended setting, and constraint violations. A visual compressor demonstration alone is not such a result. Recorded Causal Chambers data can support prediction and restricted logged-data checks. It cannot answer arbitrary new adaptive interventions. Live chamber access requires a separate arrangement and is not supplied by CURC access.

Primary candidate sources: [BoxingGym](https://arxiv.org/abs/2501.01540), [ThermoPower compressor documentation](https://build.openmodelica.org/Documentation/ThermoPower.Gas.BaseClasses.CompressorBase.html), [Causal Chambers paper](https://www.nature.com/articles/s42256-024-00964-x), [Causal Chambers remote-lab access](https://www.causalchamber.ai/).

## Measurement and the submission threshold

The proposed development budget is 16 shared initial responses followed by at most 64 additional unit-cost interventions per episode. These are design values to qualify, not a frozen scientific protocol. A variable-cost study uses a declared common cost budget instead. Fit normalizers come from the shared initial data or an independently fixed scale, not each policy's differently selected history. Score all policies on the same sealed evaluation actions.

The headline endpoint is **experimental cost to a prespecified useful accuracy**, with target attainment also reported. Nonattainment is retained as censoring at the common cap, never dropped. Report capped cost and attainment together, complete error-versus-cost curves, absolute errors, and the cost of training, scoring and inference. A rule that performs fewer experiments by failing more often is not more efficient.

An evaluator may retrospectively report when a learning curve first crosses a private threshold, but the policy cannot use that private score to stop. Demonstrating actual autonomous stopping requires a separate, paid validation procedure available equally to all arms. Do not present a retrospective crossing as a deployable stopping rule.

Use independent systems as the paired statistical units. Repeated policy and learner seeds are nested sensitivity measurements. Define family weighting, primary contrasts, multiplicity correction, practical effect thresholds and the treatment of failed episodes before confirmation. Determine confirmation sample size from development variance and a declared precision or power target. A preliminary planning range is 30–50 systems per family across at least three families, subject to that calculation and measured resources. It is not a guarantee of adequate power.

Submission requires all of the following:

1. A reproducible LLM action policy with an explicit training and deployment mechanism.
2. An independently evaluated benefit over a strong numerical acquisition policy on fresh systems. If the headline claims a particular percentage reduction, its uncertainty must support that wording, rather than only a favorable point estimate.
3. Ablations that distinguish language pretraining, ACE-specific training, causal structure and extra computation. If fine-tuning adds no benefit, do not claim that it does.
4. A concrete experimentalist task with an observable useful outcome, plus transfer beyond the synthetic training generator. The scope may be simulation-based if stated clearly.
5. Preserved failures, response accounting, operational limits and independent scientific review. The previous delivery outcomes remain visible where relevant.

If these conditions fail, investigate the identified failure before expanding compute. A narrow conditional gain can support a narrow paper, but the LLM cannot remain only a decorative element. Patrick and the coauthor retain final submission approval.

## Work sequence and CURC use

**Stage 0: recover the actual mechanism.** Audit the existing ACE-Runner LLM proposer, reward computation, preference updates and telemetry. Recover exact model/runtime provenance and identify what was executed. Convert the next-intervention interface into a small, deterministic contract. Produce artificial tests for action validity, intervention masks, budget charging, candidate isolation and private-data exclusion. No new scientific run is needed to finish this design stage.

**Stage 1: qualify one small working loop.** Prepare data, tokenize prompts, download any separately approved weights and verify inputs on CPU resources. Use a single small model and a short GPU smoke only after estimating its memory requirement. Establish that adapters receive nonzero gradients, parameters change during training, the frozen-policy control does not change, and the logged selected action is the one executed. Exercise the loop on artificial fixtures before scoring fresh systems.

**Stage 2: development screen.** A proposed 24-world screen covers four regimes with six fresh worlds each. It diagnoses policy behavior and informs sample size, not confirmatory claims. Fix candidate counts and model/learner budgets. Compare the seven policies above with paired initial data and all failures retained. Select at most one training recipe and one main numerical comparator using these development results, then freeze them.

Policy-training worlds, recipe-selection worlds and final confirmation worlds are three distinct sets. Reserve their seed and family namespaces before generating any of them. The 24-world screen is the recipe-selection set, not the source of policy-training preference labels. Audit benchmark contamination and reused public examples separately from seed independence.

**Stage 3: prospective acquisition study.** Register fresh families, worlds, models, policy training data, prompts, action menus, costs, fit schedules, evaluation actions, endpoints and stopping rules. Evaluate the frozen policies and the small set of decisive ablations. No post-outcome method changes or replacement worlds.

**Stage 4: engineering transfer and synthesis.** Validate the compressor interface or obtain physical testbed access. Assess decision quality and costs. Build the broader paper around the LLM-SCM-environment loop and the measured utility, with the delivery study as a supporting result. Revisit the title, abstract and figures only after the relevant evidence exists.

At 19:18:31 UTC on November 2, the shared CURC channel worked on `login-ci5`. The user queue returned no jobs. Account `ucb736_asc1` listed CPU and GPU QOS options. Advertised resources included A100 20 GB slices and RTX Pro 6000 24 GB slices, as well as larger GPUs. These are inventory observations, not guarantees of immediate availability or software compatibility. Scratch existed and `df` reported free capacity. A dedicated write/read/cleanup check remains necessary before a justified submission.

A proposed first execution envelope is **16 CPU-core-hours including GPU-host CPUs and 2 GPU-hours**, covering CPU preparation, a single-GPU smoke of at most 30 minutes, and a bounded development step only if the smoke justifies it. This is a planning ceiling, not an allocation or an extension of either prior budget. The full confirmation campaign needs a fresh measured projection. Exact GPU type, memory, CPU count and wall time must follow measured demand. Prefer the smallest compatible slice that fits. Do not use multiple GPUs without measured need.

Use `ucb736_asc1`, a new exclusive scratch output directory and explicit Slurm resource limits. Record CPU preparation, model loading, GPU work and output timestamps. During the smoke verify GPU memory/utilization plus a live compute process or framework counters and training/inference progress. Allocation and CUDA availability alone do not establish GPU use. A declared startup deadline triggers diagnosis if real GPU work has not begun. Preserve failed attempts and never retry an ambiguous submission or touch unrelated jobs.

Keep CPU simulation and SCM fitting off reserved GPUs where practical. Batch policy decisions across independent active episodes to sustain GPU inference, or release GPU resources between long CPU phases. Report the resulting decision latency. Do not keep a GPU server idle for hours while sequential CPU experiments run. Use the same SCM device and fitting implementation across acquisition arms so the LLM comparison is not also a learner-device comparison.

No new job, installation, model download, API call or scientific response was started while preparing this agenda. Old publication and foundation automations remain paused. The expired October exploration envelope and the delivery paper's resource reservation do not silently fund this new campaign.

## Immediate deliverables

The historical implementation audit and primary-literature comparison are complete for this agenda. They do not qualify a repaired implementation. The first implementation milestone is a CPU-qualified state/action/reward contract and a reproducible policy-training recipe, followed by one measured GPU qualification. The next decision is whether this concrete design and its measured resource projection justify a new execution cycle, not whether to submit the preserved delivery paper.

Archive provenance is in [Preserved delivery paper](delivery_paper_archive_2026-11-02.md). This agenda supersedes submission-only follow-ups for the current project direction.
