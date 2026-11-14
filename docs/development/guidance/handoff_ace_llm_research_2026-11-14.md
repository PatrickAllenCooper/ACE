# ACE research handoff and experiments before submission

November 14, 2026. For Patrick Cooper, Alvaro Velasquez and the next researcher working in `/Users/pat/code/ACE`.

**The 24-system comparison is complete. The priority now is to establish whether a repaired LLM contributes useful experimental decisions beyond strong numerical controls.** The current DPO recipe does not establish that benefit. A new saved-data diagnosis and experimental agenda were committed on November 12. They provide concrete hypotheses to test, but no successor scientific experiment has been executed or frozen in the inspected project record.

This handoff updates the [November 12 handoff](/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md), which remains preserved. The source snapshot for this update is `64832c36ba544823ce344655d0f2220e8ae278dc`. The latest accepted execution record remains November 10. No fresh SSH, scheduler or remote archive observation was made for this document.

## Current state and the change since the last handoff

The [accepted comparison](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md) contains all 24 systems and eight policies, with 192 evaluable runs and no failed or unresolved comparison cells. Reporting, selective collection and independent numerical review are complete. No job or collector needs to be restarted to finish it.

The newer [experimental agenda](/Users/pat/code/ACE/docs/development/guidance/ace_llm_experimental_agenda_2026-11-12.md) proposes experiments E1 through E10. Its [offline analysis directory](/Users/pat/code/ACE/results/ace_llm_intervention_offline_diagnosis_20261112/README.md) contains scripts, outputs, a manifest and three topics recomputed by separate agents. The proposal and analysis were committed as `64832c36b`. Statements inside those files that committing remains necessary are stale.

Keep three evidence levels separate:

| Evidence | What it supports | What remains open |
| --- | --- | --- |
| November 10 executed comparison and independent review | Recorded policy behavior, descriptive outcomes, accounting and custody | Added LLM utility was not demonstrated |
| November 12 offline diagnosis | Post hoc explanations and checks using saved records | Human review and prospective tests of causes |
| E1 to E10 experimental agenda | Candidate work sequence and proposed decision rules | Implementation, qualification, scientific freeze and measured resources |

The offline session did not load model weights, run LLM inference, collect new environment responses or use CURC. It did perform numerical analysis and model-based calculations on saved states, so it is more than a prose review. Results, training and deployment forensics had second-agent checks. Other topics rest on a single analyst. These checks are not a human independent review or a new confirmation study. The session reconstructed private target means for the 24 development systems. Retire those systems from every future evaluation role and retain them only for labeled historical diagnosis.

## The result that the next researcher must preserve

All policies used the same SCM learner on three-variable linear systems with a known graph and known noise. The screen crossed chain, fork and collider graphs with two coefficient regimes, two noise levels and two cost menus. There was no SCM-free arm.

| Policy | Mean capped total cost to target | Ever at target out of 24 | Finally at target out of 24 | Mean final NMSE |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 14.250 | 21 | 20 | 0.04113 |
| Menu coverage | 12.458 | 22 | 21 | 0.01974 |
| Local information | 11.958 | 23 | 18 | 0.01800 |
| Direct SCM target risk | 11.625 | 23 | 21 | 0.01722 |
| Numerical ranker | 13.583 | 21 | 20 | 0.03572 |
| Frozen LLM | 13.625 | 22 | 20 | 0.02008 |
| Supervised LLM | 14.625 | 21 | 19 | 0.02708 |
| DPO LLM | 15.167 | 19 | 18 | 0.03642 |

Lower values are better. Cost includes eight initialization units and the first privately evaluated crossing of NMSE 0.04. Nonattainment receives the common cap of 32. Ten systems were already at target after initialization. This retrospective endpoint is not a deployed stopping rule or actual spending saved. All eight policies, 28 paired contrasts and nine predeclared groups remain in the report.

DPO chose X = −1 on all 576 acquired actions. Its capped cost was lower than target risk on two systems, tied on thirteen and higher on nine. Against random, those counts were three, thirteen and eight. DPO had lower mean final error than random, so it did not lose on every metric. Actual comparison charges were 6,097 logical cost units and 5,884 accepted responses. The supervised model stopped in 23 systems, five above the final error target.

The mechanism is implemented, including fine-tuning. Qwen3-0.6B received 160 supervised LoRA updates and 119 DPO updates using 32 separate training systems. Training collection charged 10,240 units for 9,145 responses and retained 3,584 preference comparisons. During development episodes the LLM weights stayed fixed. The SCM posterior updated after each accepted response. Policy input contained the graph, task, legal actions, costs, budget, measured history and posterior. The policy returned one action or stop. The executor charged and performed it, then updated eligible nonintervened mechanisms.

## What the new diagnosis changes

Three findings make a small diagnostic program more valuable than another large benchmark run.

First, the repeated action was visible before deployment. Saved generations show the supervised adapter choosing X = −1 on 108 of 128 nonterminal training prompts and DPO on 123. The labels mixed numerical teachers, including random and coverage policies, without identifying the teacher in the prompt. The learned numerical ranker also concentrated on two actions. This points to supervision, representation and optimization as shared suspects, rather than proving that language models cannot learn intervention selection. See the [training verification](/Users/pat/code/ACE/results/ace_llm_intervention_offline_diagnosis_20261112/w1/training-forensics-verify.md).

Second, token-by-token greedy decoding is not equivalent to choosing the most likely complete legal command. Menu order, shared command prefixes, long numeric prompts and coarse budget cues are confounded in the recorded interface. Complete per-command deployment scores were not saved. Forward scoring and controlled input edits can separate these explanations without training a replacement policy. See the [deployment verification](/Users/pat/code/ACE/results/ace_llm_intervention_offline_diagnosis_20261112/w1/deployment-forensics-verify.md).

Third, the old endpoint provides limited discrimination because many systems already meet the target and the arms use distinct measurement-noise streams. Exploratory plan-once comparisons suggest limited benefit from sequential adaptation in this class. That is a hypothesis about this setting, not proof that no LLM benefit is possible. The implemented target-risk policy is an approximate Monte Carlo planner, not a Bayes-optimal oracle. See the [results verification](/Users/pat/code/ACE/results/ace_llm_intervention_offline_diagnosis_20261112/w1/results-analysis-verify.md).

The November 12 agenda's strongest phrases, including that the screen could not have shown a benefit or that its supervision was broken, should not become manuscript conclusions without the corresponding prospective evidence. Neither statistical insignificance nor a noisy planning reference establishes equivalence or impossibility.

## Ranked experiments to strengthen the paper

This ranking considers the importance of the claim gap, diagnostic value and cost. It is a recommendation for the next protocol, not a claim that these experiments have run. The first two can be prepared in parallel. The later stages depend on their results.

A separate [design review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_experiment_priorities_review_2026-11-14.md) identified six material issues in the earlier proposal. The requirements below incorporate its corrections on endpoint units, evaluator validation, informed-reference claims, untouched text families, joint power and the claims supported by each contrast. They still need implementation and qualification in the successor protocol.

| Priority | Experiment | Why it is worth doing | Cost and dependency |
| --- | --- | --- | --- |
| 1 | Saved-adapter causal input and decoding audit, narrowed E2 | Explains the observed failure before spending on retraining | Forward inference only initially, using pinned existing weights |
| 2 | Evaluator and baseline qualification, narrowed E1 | Prevents a more expensive study from measuring noise, a weak comparator or an invalid endpoint | Fresh CPU calibration, no LLM training |
| 3 | Numerical learnability followed by one repaired LLM recipe, E4 then E6 | Establishes whether public SCM state can support learned decisions and whether the LLM can use it | CPU first, then a measured small training allocation |
| 4 | Direct policy versus language-to-SCM compiler with equal-information controls, E5 and E7 | Most important test of the proposed LLM contribution | One predeclared task class admitted without LLM results |
| 5 | Registered independent confirmation, E8 | Turns development findings into evidence for the paper's headline | Sample size and allocation derived from measured variance and runtime |
| 6 | Frozen engineering transfer, E9 | Demonstrates useful experimental decisions beyond the training generator | Validate simulator early, evaluate frozen method later |

### 1 Explain the repeated action before changing the recipe

Score all legal complete commands for the frozen base, supervised adapter and DPO adapter on the exact saved prompt tokens. First reproduce recorded behavior under the original runtime or establish an explicit numerical equivalence tolerance for a different platform. Compare the original trie-greedy decoder with whole-command likelihood ranking. Treat length-normalized scores as a separate proposed rule, not an interchangeable implementation.

Then use declared artificial public states and saved training states to vary posterior uncertainty, target, costs and budget separately. Use feasible state edits, not arbitrary combinations that violate the SCM contract. Include meaning-preserving changes to menu order, variable identifiers and history presentation. Measure action rankings, command probabilities and response to the edits, not only whether output is legal. Cluster uncertainty by system because prompts from one system are dependent.

**Proceed when** the experiment distinguishes an implementation defect, decoding bias, weak state use or a training limitation. If the decoder explains the behavior, fix and qualify the interface before training. If command scores themselves ignore relevant state, changing the decoder alone does not repair the policy. Forward diagnostics on the old development prompts explain history only and cannot supply a new performance estimate.

This is the highest-value immediate experiment because it can change the next implementation decision with little compute. It should precede the November 12 proposal's 50-step training pilot. A separate artificial resource pilot can later measure training feasibility without conflating it with adapter forensics.

### 2 Qualify the measurement and the numerical reference

On a fresh calibration namespace, compare random, coverage, local information, plan-once and target-risk planning. Use a higher-precision planning reference and a limited lookahead control where feasible. Establish convergence of action values rather than assuming a larger fixed draw count is accurate.

Test an A/A pair, a deliberately worse policy, a leakage canary and the response-accounting rules. If common random numbers are introduced, key them so each arm retains its correct marginal environment law under adaptive decisions. Verify that coupling changes variance, not the expected treatment. Do not rewrite the original screen's noise streams.

Choose one scientifically meaningful primary endpoint before new LLM evaluation. Fixed-budget true target error is simpler than a new threshold-band statistic. Actual cost to reliable attainment is stronger for an experimental-efficiency headline, but it requires a public, charged stopping rule and separately reported success and violations. A private first crossing remains descriptive. Learner posterior risk can be a diagnostic, not a substitute for truth under misspecification.

**Proceed when** numerical convergence, calibration and leakage checks pass at predeclared tolerances with enough simulation replicates to estimate them. A nominal A/A acceptance interval is not credible without a Monte Carlo uncertainty calculation. Validate any conditional-expectation evaluator against realized outcomes in every relevant family, including paired policy effects and interval coverage. Average calibration across a regime can hide opposing policy biases. Synthetic action mixtures need independently established endpoint effects, because an oracle action fraction need not equal the fraction of final benefit. Report failures before scoring new policies.

Keep the endpoint and every admission threshold in the same units. Cost from an averaged curve, mean first-crossing cost and actual paid cost are different estimands. Do not convert a log-error improvement into intervention savings with the agenda's unvalidated universal formula. Use measured budget–error curves or reliable public stopping to support that conversion.

### 3 Establish learnability and repair one policy

Train a numerical student from the public state before expanding LLM training. Compare it with a coarse-feature or state-blind control. Supply state-conditioned high-precision teacher values, sufficient optimization and fresh validation systems. A numerical student that cannot improve on a constant action is a warning about the target or interface, not a reason to buy a larger LLM.

Once the numerical student succeeds, test one repaired small-model recipe with the same examples. Favor a compact posterior representation, shuffled minibatches, a reliable listwise or supervised target, and unambiguous action codes. Add DPO only as a separately evaluated stage if there is reliable preference information to exploit. Freeze a small diagnostic ablation set that separates interface changes from supervision changes. Do not credit a bundle of simultaneous fixes to one mechanism.

Keep the frozen LLM, supervised-only checkpoint, matched numerical student and original recorded recipe as references. An old-recipe execution on fresh data, if needed for attribution, is a new declared control cell and does not replace or rescore the accepted screen. Use multiple training seeds and validation-based stopping. Report all attempts.

**Proceed when** a policy uses target-relevant posterior changes, remains stable to irrelevant formatting changes, and performs comparably to a competent numerical student on fresh systems. Set thresholds before scoring. An action-share cap or entropy floor alone is inappropriate when the planner itself legitimately prefers one action. Compare behavior with the task's required distinctions. Imitating a teacher is a learning milestone, not evidence of experimental superiority.

### 4 Test the LLM contribution at equal information

The most promising decisive comparison is between a direct LLM intervention policy and an LLM that translates domain text into a checked SCM prior, constraint or objective for a numerical planner. The second role uses language where language may add value while retaining numerical planning. It deserves a primary comparison, not a decorative demonstration.

Select at most one initial class through an LLM-blind numerical audit. Structural knowledge in text is a reasonable first candidate. Include an uninformed planner, an informed numerical reference, a rule parser feeding the planner, a learned numerical text control, a matched numerical policy, the compiler and the direct policy. Give comparable usable information and data to all relevant arms. Never call beating an uninformed planner an isolated language-model advantage.

Cross correct text with omitted and misleading text. Reserve separate text families for class admission, tuning and sealed confirmation, with author or template separation fixed before admission. Fresh systems alone do not make previously inspected text families fresh. Test feedback shuffling only where a competent adaptive policy actually needs the feedback. A policy-input SCM ablation must be trained under that input condition. If the executor still uses the SCM, that ablation tests the chooser's dependence on SCM information, not whether an SCM learner is superior to an SCM-free learner.

**Proceed when** the task has demonstrable useful information beyond strong no-text planning, simple equal-information controls do not already exhaust it, and a candidate succeeds on sealed validation. Preserve every audited class. Class admission must not use LLM scores. An oracle-informed heuristic gives a reference gap, not a certified information ceiling unless its optimality is proved. Report the task-selection rule as part of the scientific scope.

My recommendation is to retain both direct and compiler roles through this test. A compiler win would support a narrower LLM-plus-SCM mechanism. Whether it satisfies Patrick's required headline contribution is a scientific framing decision to settle before confirmation, not after seeing which arm wins.

### 5 Confirm the claim rather than select another recipe

Register the chosen method, all model seeds, two primary contrasts, endpoint, practical effect size, sample size, failure treatment and analysis before generating confirmation systems. The contrasts should test benefit over the strongest no-text planner and over the strongest equal-information numerical control selected on separate development data. Add compiler hypotheses only if that role is part of the declared headline, with multiplicity adjusted accordingly.

Size the study from measured variance components, including systems, policy seeds, response replicates and text families. Specify the minimum meaningful effect for each primary contrast and power the complete joint decision rule. Half an informed-versus-uninformed planning gap does not establish the effect against an equal-information control. Do not adopt the previous agenda's 96 to 156 systems as a justified sample size without those measurements. Distinguish a recipe-level result across training seeds from performance of one chosen checkpoint. Keep private evaluation inaccessible to the policy and retain nonattainment, failures and all registered systems.

**Submission-quality success** requires confidence bounds supporting the prespecified useful improvement, reliable target attainment and successful leakage checks. A favorable point estimate or failure to reject a difference is insufficient. Equivalence requires its own justified margin and adequate power. Independently implement or review the evaluator and analysis, with human scientific review in addition to agent checks.

Passing the primary contrasts supports superiority over their named frozen comparators. It does not by itself establish that text caused the gain, that the policy adapts to measurements, or that it recovers from misleading text. Each behavioral claim needs its corresponding prespecified ablation. Transfer remains a separate result.

### 6 Show an experimentalist outcome in a validated environment

Use a validated wind-tunnel or compressor-related simulator to ask for an operating setting that meets a pressure requirement while reducing current or power. Validate feasible interventions, settling assumptions, measurements, costs and the reference optimum before comparing policies. A wind tunnel is related to the presentation's compressor story, not evidence about an actual compressor.

Measure the quality and constraint violations of the recommended operating point, the paid interventions, inference time and training bill. Apply a common final recommendation rule so the acquisition comparison remains interpretable. Separate simulation results from physical hardware claims. If the same family helped select the method, use another registered family for the transfer claim.

Prepare the environment specification early. Register the eventual transfer evaluation and reporting policy before confirmation outcomes. Do not quietly omit a scheduled failed transfer or describe a success-conditioned run as general transfer. Real hardware is a valuable later extension, but a paid remote laboratory is not necessary to diagnose the current failure and requires a separate scope and spending decision.

## Literature supporting the priorities

The following primary sources were checked for this handoff. They motivate controls and hypotheses, not predictions that ACE will succeed.

- [Foster et al., Deep Adaptive Design, ICML 2021](https://proceedings.mlr.press/v139/foster21a.html) learns numerical experimental policies before deployment. It makes a trained nonlanguage policy an important comparator and motivates reporting training cost separately from decision latency.
- [Sussex et al., Model-based Causal Bayesian Optimization](https://arxiv.org/abs/2211.10257) learns mechanisms and propagates uncertainty through a causal graph. It supports a strong causal planning comparator for an optimization task. Its guarantee does not automatically apply to our prediction-risk heuristic.
- [Gupta et al., Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.838/) uses shuffled outcomes to examine adaptation and reports a useful hybrid based on LLM priors. It motivates separating language knowledge from adaptation to measurements. Its result is not proof that all trained LLM intervention policies fail.
- [Mazzaccara et al., Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.291/) reports information-gain preference training for questions. It supports testing informative supervision but does not establish numerical intervention efficiency in physical systems.
- [Razin et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html) studies likelihood displacement in preference optimization. It motivates inspecting absolute command likelihoods and probability shifts. It does not identify the cause of ACE's collapse without our own diagnostic.
- [Causal Chamber documentation](https://docs.causalchamber.ai/) distinguishes public datasets, physical chambers and live remote experiments. Logged datasets alone cannot answer arbitrary new adaptive interventions. Any simulator or live API must be pinned and qualified before use.

## Work to defer and conditions before submission

Do not launch a larger model sweep, repeat the current DPO recipe, choose a favorable benchmark from LLM outcomes, or rerun accepted delivery studies. Avoid another underpowered single-realization screen. Do not assume more training or a higher GPU budget repairs absent state use. The old delivery paper remains archived as a component study, and the broader LLM utility requirement remains unmet.

Before submission, the paper needs a reproducible and consequential LLM mechanism, fresh evidence against strong equal-information controls, a practical experimentalist outcome, complete cost and failure reporting, independent scientific review, and approval from Patrick and Alvaro. A diagnostic or negative-result paper is a legitimate alternative, but it does not automatically satisfy Patrick's existing positive-utility requirement. No venue or acceptance outcome is guaranteed. Recheck official venue rules and dates when choosing a submission target.

The November 12 agenda also flags historical README, slide and preprint claims for audit. Verify identity, attribution and the original metric audit before proposing public corrections. Do not repeat those claims as current evidence, and do not submit, withdraw a preprint or circulate revised marketing material as part of this handoff.

Routine diagnosed implementation and operational corrections retain Patrick's blanket approval. Do not re-ask for each small fix. The proposed larger compute tranches, new weights or installations, changed custody model and paid laboratory use are not established by this document. Prepare concrete protocols and measured projections first. This request is for evaluation and handoff, so no model execution, experiment, allocation or manuscript change was performed.

## Evidence custody and operational restart

The [execution ledger](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md) remains the record of executed jobs. Read it together with the newer agenda and this handoff. Its November 10 completion supersedes historical pending states, while the November 12 analyses add research work without changing the original reports.

- [Final reporting receipt](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/receipt.json) binds the reader report, CSVs and display data. SHA256: `11e6c378598dcbc739403b7a9f53c6ea86e850fdc0e0d4ef898aab8b35e58583`.
- [Execution acceptance](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/acceptance.json), [retrieval handoff](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/handoff.json) and [numerical review](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/numerical-review01/review.md) cover different stages. Preserve their original flags and dates.
- [Interactive results](/Users/pat/.cursor/projects/Users-pat-code-ACE/canvases/ACE-24-system-comparison.canvas.tsx) passed TypeScript and static data checks. Rendered interaction review remains unclaimed.
- [Offline diagnosis manifest](/Users/pat/code/ACE/results/ace_llm_intervention_offline_diagnosis_20261112/sha256sums.txt) lists 412 members. Its README records omitted regenerable caches, source paths and known errata. Scripts still contain temporary-session paths and should not be blindly rerun.
- Exclusive local evidence is under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/`. The final report mirror is `ACE-LLM-intervention-24-system-reporting-20261110-01`.

Keep large artifacts on CURC. The verified finalization archive is `/projects/paco0228/ACE_evidence/ace-llm-finalization-20261110-01/raw.tar`, recorded at 3,390,535,680 bytes with SHA256 `413c38d990cd7f6a807d8280b8c44ba181da6b59c651faf614643d06f56ea3f9`. It preserves 107,284 original files. Separate archives retain upstream training, runtime and model artifacts. All 242 reports and selected verification records are local. This removes the old 85 GiB Mac free-space planning target. It does not mean that local verification is free of storage cost or that CURC supplies an institutional backup guarantee.

The accepted completed cycle records 49 jobs, 100 submission journals, 40,265 allocated CPU seconds and 4,458 A100 MIG slice seconds. Including the single 3,600-second CPU allowance gives 43,865 of 57,600 CPU admission seconds. GPU admission was 4,458 of 7,200 slice seconds. No reservation remained outstanding at closeout. Whole-cycle CPU is unknown, and subsequent unmetered analysis is not included. Do not add overlapping process, elapsed, allocation and reservation measures.

The final reporting job `35533604` completed `0:0`. Its 7,677.7 MiB peak against a 7,680 MiB request leaves no demonstrated memory headroom. All jobs, failed attempts and collection failures remain preserved. No old job needs resubmission.

For any future CURC operation, use `curc-access`, check transport and account `ucb736_asc2`, then reconcile both controller locks and all relevant journals before mutation. Never duplicate a pending or running submission or retry an ambiguous one. Keep CPU preparation off GPUs where practical. Measure the smallest viable GPU allocation, verify real use with independent signals and progress, and monitor startup and sustained utilization. Do not assume the agenda's full-GPU runtime estimates, quotas or local-model speed are measured.

The [previous handoff](/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md) links the world, learner, episode, policy, training and collector modules. Executed source was `a9346022f145cc29439325b6f291597b2e6ec037`, with scientific freeze `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`. Use pinned copies to interpret history and new source versions for new protocols.

Leave the dirty `.DS_Store`, presentation v9, old fixtures and untracked archive files untouched. Preserve the November 12 handoff and agenda as dated records. Keep the current LaTeX editor open. Use clear prose, consistent intervention terminology and explicit units. The next concrete output should be a reviewed, bounded protocol for experiments 1 and 2 above, followed by execution only within the applicable authorization and resource limits.
