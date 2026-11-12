export const meta = {
  name: 'ace-understand',
  description: 'Parallel deep-read of the ACE repo: manuscript, LLM mechanism, training recipe, numerics, saved-evidence forensics, ops, prior evidence, literature',
  phases: [
    { title: 'Read', detail: 'one reader per subsystem or evidence class' },
    { title: 'Verify', detail: 'independent re-derivation of quantitative forensic claims' },
  ],
}

const REPO = '/Users/pat/code/ACE'
const SCRATCH = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1'
const G = REPO + '/docs/development/guidance'
const SR = REPO + '/scripts/research'
const RES = REPO + '/results'

const CONTEXT = `
PROJECT CONTEXT (read carefully, you start with no other context).
The repository ${REPO} is a research codebase called ACE (Active Causal Experimentalist). The current research question: can a language model (LLM) that is trained to use an explicit structural causal model (SCM) choose interventions that give an experimentalist a reliable answer at lower experimental cost than strong numerical policies? The current orientation document is ${G}/handoff_ace_llm_research_2026-11-12.md. Read it first (it is short). Two more anchor documents: ${G}/ace_llm_intervention_agenda_2026-11-02.md (research design and submission conditions) and ${G}/ace_llm_intervention_24_system_results_2026-11-10.md (the completed 24-system comparison).
Headline state: a 24-system development screen compared 8 intervention policies (uniform random, menu coverage, local information, direct SCM target-risk planning, learned numerical ranker, frozen Qwen3-0.6B, supervised-finetuned LLM, DPO-finetuned LLM). Direct SCM target-risk planning was best (mean capped cost-to-target 11.625). The DPO LLM was worst (15.167) and chose the single action 'set_x_neg' on all 576 acquired actions. The frozen LLM chose 'observe' on 574 of 576. The supervised LLM chose set_x_neg on 481 of 523. The paper is not ready. I (the lead researcher's assistant) am preparing an experimental agenda to strengthen the eventual paper as much as possible, and your report is one input to it.

HARD RULES.
1. Read-only on all project data. Do NOT modify, create, delete, move, stage or commit anything under ${REPO}, /Users/pat/code/ACE-Runner or /Users/pat/ACE_Study_Results. No git commands that change state.
2. Do NOT launch experiments, training, model inference, model-weight loading, SLURM, SSH, CURC, downloads or installs. Do not run repository launch, collection, campaign or finalizer scripts. Pure offline analysis of already-saved files with /usr/bin/python3 (Python 3.9, numpy, pandas, scipy available) is allowed where your task asks for it. Importing a repository module only to call a pure deterministic helper (for example rendering a prompt string from a fixture) is allowed, but nothing that loads model weights or executes an episode on new systems.
3. If you need scratch files (analysis scripts, intermediate outputs), write them ONLY under ${SCRATCH}/<your-label>/ . Create that directory yourself.
4. Never scan or extract the large 'archive.tar' or 'archive.tar.part-*' files. The results tree is 18 GB, so target specific directories instead of grepping all of ${RES}.
5. Report only what you verified in files. Mark inferences as inferences. Cite file paths (with line numbers where practical) for every important fact. If documents disagree, the newest dated document wins, and you should report the disagreement.
6. Be skeptical and precise. Preserve evidence that contradicts the convenient story. Numbers must be exact where the files give them.
7. In addition to your structured return value, write your complete detailed notes as markdown to ${SCRATCH}/<your-label>.md (this is scratch space, not the repo). Later agents will read these notes, so make them self-contained, with file paths. Your structured return should be a compact distillation (roughly 600 to 1200 words in total across fields).
`

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'Compact prose distillation of what you established' },
    key_facts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          fact: { type: 'string' },
          evidence: { type: 'string', description: 'file path(s) and line numbers, or the computation performed' },
          confidence: { type: 'string', enum: ['verified', 'likely', 'inferred'] },
        },
        required: ['fact', 'evidence', 'confidence'],
      },
    },
    numbers: { type: 'string', description: 'Compact tables or exact numbers worth carrying forward (may be empty)' },
    surprises_or_contradictions: { type: 'array', items: { type: 'string' } },
    implications_for_agenda: { type: 'array', items: { type: 'string' }, description: 'Concrete consequences for what experiments should or should not be proposed next' },
    open_questions: { type: 'array', items: { type: 'string' } },
    notes_path: { type: 'string' },
  },
  required: ['summary', 'key_facts', 'implications_for_agenda', 'notes_path'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    checked: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partially_confirmed', 'could_not_check'] },
          my_value: { type: 'string', description: 'What you independently derived' },
          how: { type: 'string' },
        },
        required: ['claim', 'verdict', 'my_value', 'how'],
      },
    },
    additional_findings: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string' },
  },
  required: ['checked', 'overall'],
}

const READERS = [
  {
    label: 'manuscript',
    prompt: `YOUR LABEL: manuscript
TASK: Establish exactly what the current manuscript is and what it would need to become the broader paper.
READ: ${REPO}/paper/README.md ; ${REPO}/paper/aistats_ace_2027/paper.tex (4559 lines. The \\title appears near line 3022. Determine what lines 1 to 3019 contain) ; ${REPO}/paper/aistats_ace_2027/delivery_theory.tex, delivery_claims.tex, delivery_prospective_claims.tex, delivery_evaluation_design.tex, claim_index.json ; ${G}/delivery_paper_archive_2026-11-02.md ; ${G}/ace_submission_audit_2026-11-01.md ; ${G}/delivery_panel_revision_2026-11-02.md ; ${G}/delivery_submission_review_2026-10-24.md ; ${REPO}/docs/ACE_evidence_and_public_claims_2026-10-08.md.
ANSWER:
(a) Title, abstract and the complete list of contributions and quantitative claims in the current manuscript, each tagged confirmatory or exploratory, with its evidence source.
(b) The limitations the manuscript itself declares.
(c) Every weakness or reviewer-style criticism already recorded in the submission audit, panel revision and submission review documents. Quote the sharpest ones briefly.
(d) Does the current manuscript say anything about an LLM intervention policy? If so, what exactly.
(e) Which sections, results and theory could carry over as supporting material into a broader paper whose headline is an SCM-conditioned LLM intervention policy, and which contributions are entirely missing.
(f) The claims most attackable by a skeptical AISTATS, ICML or TMLR reviewer, and what experiment would defuse each.
(g) Any venue, deadline or formatting commitments recorded anywhere in these files.`,
  },
  {
    label: 'mechanism',
    prompt: `YOUR LABEL: mechanism
TASK: Reconstruct precisely how the LLM chooser sees the state and how its action is selected, and identify every feature of that path that could make its choice insensitive to the SCM state.
READ CODE: ${SR}/llm_intervention_contract.py, llm_intervention_task.py, llm_intervention_world.py, llm_intervention_scm.py, llm_intervention_episode.py, llm_intervention_grammar.py, llm_intervention_action_scoring.py, llm_intervention_canonical_scoring.py, llm_intervention_language_policy.py, llm_intervention_persistent_policy.py, llm_intervention_policy_cli.py, llm_intervention_precision.py, llm_intervention_screen_prompt_fixture.py, llm_intervention_screen_realization.py, llm_intervention_model_service.py, llm_intervention_screen_policy.py.
READ DOCS: ${G}/ace_llm_intervention_shared_task_2026-11-02.md, ace_llm_intervention_world_2026-11-04.md, ace_llm_intervention_grammar_2026-11-02.md, ace_llm_intervention_action_scoring_2026-11-02.md, ace_llm_intervention_canonical_scoring_2026-11-02.md, ace_llm_intervention_language_inference_2026-11-02.md, ace_llm_intervention_long_context_2026-11-02.md, ace_llm_intervention_precision_2026-11-02.md, ace_llm_intervention_model_service_2026-11-04.md.
LOOK FOR SAVED PROMPTS: ${RES}/ace_llm_intervention_screen_prompt_cpu_20261108 (prompt fixtures and CPU qualification), ${RES}/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu (deployment records of the LLM arms), ${RES}/ace_llm_intervention_scientific_tokenization_20261108/terminal02/remote/launch/preparation.
ANSWER:
(a) Reproduce one complete real chooser prompt exactly as the model received it in the scientific campaign (find a saved one. If none is saved verbatim, render one from a saved state with the repository serializer, a pure function). Report its length in characters, and in tokens if a token count is recorded anywhere. Mark which spans vary with the state (posterior means and covariances, history, remaining budget, costs, graph, target queries) and which are constant boilerplate. Estimate the fraction of the prompt that is state-dependent.
(b) How numbers are serialized: precision, notation, ordering. Does the measured history grow with each step. Is there any truncation and on which side. What is the maximum context used.
(c) Exactly how the action is chosen: generation or constrained scoring of each legal command. If scoring, is it a sum or mean of token log-likelihoods, is there length normalization, what are the exact command strings for the 8 acquisition actions and stop, how many tokens is each, which tokens do they share as prefixes, how are ties broken, is a chat template applied, is Qwen3 thinking mode disabled, is sampling involved at all.
(d) Whether the information needed to choose well is actually present in the prompt in usable form: target queries, per-action costs, remaining budget, which mechanisms are still uncertain. For a competent reader, could the best action be computed from the prompt alone.
(e) A ranked list of concrete hypotheses for why a trained adapter could output the same command for every state (for example: command prior learned as a bias on a late shared token, state tokens far from the answer position, numeric content tokenized into long digit strings that the 0.6B model ignores, supervised targets nearly constant, prompt boilerplate dominating). For each hypothesis name the saved artifact or cheap offline measurement that would confirm or refute it.
(f) Whether per-decision score vectors (a log-likelihood for every legal command at every decision) are saved for the frozen, supervised and DPO arms, and the exact path and schema if so.
(g) What the stop rule is for LLM arms and for numerical arms.`,
  },
  {
    label: 'training',
    prompt: `YOUR LABEL: training
TASK: Reconstruct the full training recipe for the supervised (SFT) and preference (DPO) adapters and for the learned numerical ranker, in exact numbers, from code and design documents. Identify recipe features that predict a constant-action policy.
READ CODE: ${SR}/llm_intervention_training_dataset.py, llm_intervention_training_collection.py, llm_intervention_training_exposure.py, llm_intervention_training_plan.py, llm_intervention_training_engine.py, llm_intervention_supervision.py, llm_intervention_replicated_branches.py, llm_intervention_replicated_feedback.py, llm_intervention_learning.py, llm_intervention_checkpointing.py, llm_intervention_gradient_probe.py, diagnose_llm_intervention_gradients.py, llm_intervention_scientific_training_gpu_worker.py, collect_llm_intervention_scientific_training_cpu.py, llm_intervention_ranker.py, llm_intervention_ranker_training.py, llm_intervention_numerical_policy.py, llm_intervention_cpu_owned_supervision.py.
READ DOCS: ${G}/ace_llm_intervention_training_dataset_2026-11-04.md, ace_llm_intervention_training_collection_2026-11-07.md, ace_llm_intervention_training_exposure_2026-11-07.md, ace_llm_intervention_training_engine_2026-11-07.md, ace_llm_intervention_training_plan_2026-11-08.md, ace_llm_intervention_training_cpu_2026-11-08.md, ace_llm_intervention_training_gpu_2026-11-08.md, ace_llm_intervention_screen_training_shape_2026-11-08.md, ace_llm_intervention_ranker_training_2026-11-07.md, ace_llm_intervention_gradient_diagnostics_2026-11-02.md, ace_llm_intervention_checkpointing_2026-11-02.md.
ANSWER, WITH EXACT NUMBERS AND FILE:LINE CITATIONS:
(a) Data: number of training systems, episodes and states per system, how states were generated (which behavior policy visited them), how many SFT examples exist, which numerical teachers label them and how teachers are assigned to examples, whether the teacher label depends on the state or is nearly constant by construction.
(b) SFT optimization: number of updates (160), batch size, epochs over the data, learning rate and schedule, optimizer, LoRA rank, alpha, dropout and target modules, maximum sequence length and truncation side, loss masking (command tokens only or prompt plus command), precision.
(c) Preference data: at each training state, which candidate actions were branched and executed, how many replicates, how the preference label is computed (replicated prediction-error reduction per acquisition-cost unit), the tie or uncertainty rule, why only 119 strict pairs out of 3,584 collected comparisons were used for DPO updates (state the exact exposure rule), and whether pairs are constructed within a state or across states.
(d) DPO optimization: loss form, beta, reference model, whether likelihoods are conditioned on the full state prompt, batch size, learning rate, number of updates (119), whether each pair is seen once, checkpoint selection rule.
(e) Numerical ranker: input features, architecture, loss, training exposure, and what 'matched example exposure' means.
(f) Recipe features that predict or permit collapse to one command regardless of state. Consider: marginal label imbalance, one pass over very few examples, DPO pairs dominated by one preferred action, preference labels that are approximately state-independent, the well-known DPO failure modes with few pairs, learning rate relative to the tiny update count, loss masked to a short command with shared prefixes. For each, say what in the code or documents supports or contradicts it, and which saved artifact would settle it.
(g) What the gradient diagnostics established (nonzero gradients, parameter change) and what they did not establish.
(h) What a successor recipe could change at low cost, and what the code already supports (for example a different loss, more updates, listwise ranking over the full menu, teacher distillation from target-risk scores).`,
  },
  {
    label: 'numerics-protocol',
    prompt: `YOUR LABEL: numerics-protocol
TASK: Characterize the world generator, the SCM learner, the numerical policies and the evaluation protocol, and determine analytically how much room exists in this world class for ANY adaptive policy to beat direct target-risk planning.
READ CODE: ${SR}/llm_intervention_world.py, llm_intervention_scm.py, llm_intervention_target_risk.py, llm_intervention_numerical_policy.py, llm_intervention_screen_policy.py, llm_intervention_screen_protocol.py, llm_intervention_screen_reporting.py, llm_intervention_cost_curves.py, llm_intervention_paired_episode.py, llm_intervention_episode.py, llm_intervention_episode_seal.py, llm_intervention_campaign.py, llm_intervention_scientific_campaign.py, llm_intervention_ranker.py, llm_intervention_task.py.
READ DOCS: ${G}/ace_llm_intervention_world_2026-11-04.md, ace_llm_intervention_target_risk_2026-11-02.md, ace_llm_intervention_screen_policy_2026-11-07.md, ace_llm_intervention_screen_protocol_2026-11-07.md, ace_llm_intervention_screen_reporting_2026-11-07.md, ace_llm_intervention_cost_curves_2026-11-07.md, ace_llm_intervention_paired_episode_2026-11-07.md, ace_llm_intervention_scientific_contract_2026-11-08.md, ace_llm_intervention_scientific_campaign_2026-11-08.md.
ANSWER, WITH FILE:LINE CITATIONS:
(a) World generator: graphs, coefficient regimes (what do the regime names such as matched_low mean exactly), noise levels, how the 24 systems were drawn, the initial 8-observation schedule, the 8 target queries (what they ask), the public scale, the 0.04 NMSE threshold and how it and the budget (8 plus 24 cost units) were chosen. Was it anticipated that 10 of 24 systems meet the target after initialization.
(b) SCM learner: the exact model (conjugate Bayesian linear regression per mechanism, known or unknown noise variance, priors, what 'matched' versus mismatched prior means). Does the posterior covariance depend on observed responses or only on the design (parent values).
(c) Target-risk policy: exact scoring formula, myopic or multi-step, how cost enters, tie rules. Local-information policy: formula. Coverage and random: rules. Is target risk the exact one-step Bayes-optimal action for the declared endpoint under the learner's own model.
(d) THE KEY ANALYSIS. In a linear-Gaussian model with known noise, posterior covariance is a function of the design only, so the value of adapting to observed outcomes is structurally small and a fixed open-loop design can be near optimal. Determine from the code whether this applies here, and where it does not (for example when an intervened or observed parent value is random, so the realized design depends on outcomes). State clearly whether the 24-system screen could in principle have distinguished a genuinely state-adaptive policy from a good static schedule that depends only on graph, cost menu and target queries. Note that menu coverage (a static rule) scored 12.458 against target risk 11.625.
(e) Enumerate the conditions under which a learned or language policy could legitimately beat target-risk planning: non-myopic budget effects with heterogeneous costs, learner misspecification (nonlinear truth, linear learner), unknown noise, model-family uncertainty, semantic prior knowledge not encoded in the numeric SCM, large or continuous action spaces where numerical scoring is expensive or candidate omission matters. For each, say whether the current code can express it and estimate the code change required (small, medium, large) with the specific modules affected.
(f) Which protocol components are reusable as-is for a successor (freeze, pairing, sealed evaluation, cost curves, reporting) and which are hardwired to 3 variables and 8 actions.
(g) Weaknesses in the endpoint definition (retrospective threshold crossing, cap at 32, ties from initialization) and what more sensitive endpoints the saved data already supports.`,
  },
  {
    label: 'prior-evidence-core',
    prompt: `YOUR LABEL: prior-evidence-core
TASK: Establish what earlier ACE evidence is solid, what was refuted, and what assets and lessons a new agenda must honor. This covers the original ACE pipeline, the metric audit, the PEV method, the delivery study, and the Mission-1 LLM proposer study in the sibling repository.
READ: ${G}/metric_audit_2026-09-10.md ; ${G}/paper_reframe_2026-09-17.md ; ${G}/pev_design.md ; ${G}/handoff_2026-09-25.md ; ${G}/research_execution_2026-09-25.md (3380 lines. Read the structure and the latest state of every thread, do not summarize every entry) ; ${G}/handoff_delivery_2026-10-07.md (1001 lines, read the summary sections) ; ${G}/delivery_paper_execution_2026-10-06.md ; ${G}/delivery_final_release_acceptance_2026-10-24.md ; ${G}/delivery_paper_archive_2026-11-02.md ; ${REPO}/docs/ACE_evidence_and_public_claims_2026-10-08.md ; ${REPO}/docs/FULL_ACE_COMPONENTS.md. In the sibling repository (read-only): /Users/pat/code/ACE-Runner/HANDOFF-2026-10-21.md, /Users/pat/code/ACE-Runner/HANDOFF-2026-10-peter-study.md, /Users/pat/code/ACE-Runner/studies/2026-10-peter-baseline/package/stats_appendix.md, /Users/pat/code/ACE-Runner/docs/decisions/ (especially 0002 and 0003), /Users/pat/code/ACE-Runner/docs/code_health.md.
ANSWER:
(a) A table of prior results that are solid enough to cite in a broader paper, with exact numbers, sample sizes, and caveats. Include the surviving 5-node result and its unresolved asymmetries, the delivery confirmation (geometric mean ratio 0.182719 and the prospective study), and anything else.
(b) What was refuted or retracted, so that the new agenda never leans on it.
(c) The fate of PEV (propagated expected variance acquisition): was the CURC ladder ever run, what did it show, and is PEV related to the current target-risk planner.
(d) Mission-1 (the earlier LLM proposer study): design, the 1.0412 ratio, why it detected no LLM advantage, and what its execution disclosures imply for the design of new LLM comparisons.
(e) How the original ACE LLM pipeline worked (which model, what the LLM proposed, how DPO was used, the hand-coded heuristics) and how it differs from the November design. Record every previous observation of LLM policy collapse or constant behavior. There may be a pattern.
(f) Controls and lessons that any new experiment must honor (matched students, one evaluator for all arms, no selection on favorable bundles, parser defects, privileged critic access).
(g) Reusable assets: environments, emulators, datasets, infrastructure, accepted protocols.`,
  },
  {
    label: 'prior-evidence-portfolio',
    prompt: `YOUR LABEL: prior-evidence-portfolio
TASK: Inventory every research direction, benchmark and theoretical idea that was previously proposed or tried in this project, with its status and outcome, so the new agenda neither repeats dead ends nor misses a prepared asset.
READ: ${G}/research_portfolio_2026-09-25.md ; ${G}/broad_experimental_agenda_2026-10-01.md ; ${G}/broad_experiment_registry_2026-10-01.json ; ${G}/research_recovery_agenda_2026-10-01.md ; ${G}/technical_path_2026-10-01.md ; ${G}/scm_foundation_viability_agenda_2026-09-29.md ; ${G}/ace_theoretical_ideation_2026-10-08.md ; ${G}/ace_foundation_cycle_closeout_2026-10-09.md ; ${G}/ace_foundation_component_results_2026-10-09.md ; ${G}/ace_foundation_retention_results_2026-10-09.md ; ${G}/ace_foundation_mismatch_results_2026-10-09.md ; ${G}/ace_foundation_mixture_theory_2026-10-09.md ; ${G}/external_benchmark_feasibility_2026-09-27.md ; ${G}/neuronbench_feasibility_2026-09-27.md ; ${G}/causal_chambers_gate_2026-10-01.md ; ${G}/chambers_calibration_protocol_2026-10-02.md ; ${G}/chambers_physics_correction_2026-10-02.md ; ${G}/chambers_prediction_protocol_2026-10-02.md ; ${G}/action_language_source_intake_2026-10-01.md ; ${G}/action_separation_design_2026-10-02.md ; ${G}/prior_abstention_diagnostic_2026-10-02.md ; and list the remaining files in ${G} whose names start with transfer_, protocol_, runner_delivery_ or repair_ and read the ones that look relevant. Skim the matching scripts in ${SR} (boxing_*.py, chambers_*.py, neuronbench_*.py, audit_causalman_*.py, partial_id*.py, action_language_*.py, foundation_*.py, connected_*.py) only as needed to confirm what was actually executed.
ANSWER:
(a) A table: direction, what was proposed, what was executed, outcome (positive, negative, mixed, abandoned, never started), the reason, and the key document.
(b) External benchmarks and physical or simulated testbeds that were assessed: BoxingGym, Causal Chambers, NeuronBench, CausalMan, the compressor (OpenModelica ThermoPower), anything else. For each: feasibility verdict, blockers, whether an adapter or data already exists locally, and whether it supports true adaptive interventions or only logged data.
(c) The theoretical ideas recorded in the ideation and mixture-theory documents. Which could underpin a real LLM role (for example semantic priors, model-family selection under misspecification, action language), and what evidence already exists for or against each.
(d) The action-language experiments of late September and early October: what they tested, what they found.
(e) The TabPFN or foundation-predictor cycle: results, conditional benefits and harms, and why the six language proposals failed the output contract.
(f) Directions that the documents explicitly rule out or warn against repeating.
(g) Your judgment, clearly labeled as judgment: which prepared assets give the cheapest route to a concrete experimentalist task with executable interventions and a useful outcome.`,
  },
  {
    label: 'ledger-ops',
    prompt: `YOUR LABEL: ledger-ops
TASK: Extract the operational reality of the completed November cycle so that a new experimental agenda can be costed and sequenced realistically.
READ: ${G}/ace_llm_intervention_execution_2026-11-02.md (1513 lines, newest entry first. Read the newest entries fully and skim the rest for stage boundaries, failures and durations) ; ${G}/ace_llm_intervention_account_renewal_2026-11-07.md ; ${G}/ace_llm_intervention_remote_custody_2026-11-09.md ; ${G}/ace_llm_intervention_scientific_campaign_2026-11-08.md ; ${G}/ace_llm_intervention_gpu_qualification_2026-11-02.md ; ${G}/ace_llm_intervention_staged_runtime_2026-11-07.md ; ${G}/ace_llm_intervention_screen_execution_design_2026-11-08.md ; ${G}/ace_llm_intervention_paired_campaign_integration_2026-11-08.md ; ${RES}/ace_llm_intervention_scientific_finalization_20261108/acceptance01/review.md ; the skill file at /Users/pat/.claude/skills/curc-access/SKILL.md if it exists (read only, do not connect).
ANSWER:
(a) A timeline of the cycle from November 2 to November 12: each stage (contract, realization, collection, tokenization, training, CPU campaign, GPU campaign, finalization, reporting), its wall-clock span, compute consumed, and number of failed attempts.
(b) The approved resource envelope (CPU seconds, GPU slice seconds), what was consumed, what remains, and exactly what scope the 'blanket approval for routine diagnosed operational recovery' covers. What would require a new approval from Patrick.
(c) Standing process rules that a new experiment must follow (freeze before scoring, namespaces, exclusive evidence directories, once-only submission, controller locks, custody and backup). Distinguish rules that apply to scientific runs from rules that would not bind a cheap local offline diagnostic.
(d) The dominant sources of overhead and failure in the cycle (for example transfer and custody, tokenization, GPU startup). Roughly what fraction of effort was scientific computation versus operations.
(e) Measured unit costs that allow projection: seconds per LLM decision, seconds per training update, CPU seconds per numerical episode, GPU memory, tokens per prompt.
(f) Any recorded statements of Patrick's priorities, deadlines, target venues, coauthor expectations, or constraints on paid APIs and model choices.
(g) A realistic projection: calendar days and compute for (1) an offline diagnosis on saved artifacts, (2) a short GPU retraining of a revised recipe plus a 24 to 48 system screen, (3) a 100 to 150 system prospective confirmation. State the assumptions.`,
  },
  {
    label: 'literature-policy',
    prompt: `YOUR LABEL: literature-policy
TASK: Web literature research (this is the one kind of network use you are permitted: literature search and reading public papers). Use WebSearch and WebFetch (load them through ToolSearch if they are deferred). Keep search queries generic and scientific. Never put any person's name from this project, the project name, repository paths or unpublished results into a query or URL.
First read the 'Where the literature leaves room for ACE' section of ${G}/ace_llm_intervention_agenda_2026-11-02.md so you know what is already cited.
RESEARCH QUESTIONS:
(1) Closest prior art from 2023 to late 2026 on language models as experimental-design, active-learning or intervention-selection policies, especially with fine-tuning from measured outcomes: amortized Bayesian experimental design (Deep Adaptive Design and successors, transformer or PFN based amortized BED, in-context experimental design), LLM Bayesian optimization (LLAMBO, LMABO and critiques such as the 2025 study that used shuffled feedback), BED with LLMs for question asking, preference or RL fine-tuning of LLMs for information gathering, LLMs in causal discovery with interventions, causal Bayesian optimization with LLM priors, amortized active causal structure learning, in-context reinforcement learning and decision-pretrained transformers. Verify the works already cited in the agenda exist and are described accurately, and find what the agenda missed. For each key paper give authors, year, venue, URL, the one-sentence claim, and why it matters for positioning.
(2) Known failure modes that could explain a fine-tuned small LLM emitting one constant action: DPO likelihood displacement and degenerate solutions, DPO with very few pairs, imbalanced supervised labels and majority-class collapse, behavior cloning from a planner, weak numeracy and number tokenization in small LLMs, insensitivity to long numeric context, in-context regression limits. Cite primary sources and extract concrete, testable diagnostics and remedies (for example listwise or regression-to-scores distillation, reward-weighted regression, numeric encoders, feature tokens, chain-of-thought over SCM summaries, tool calls to a numerical planner).
(3) Evidence about where language models add real value in experimental design, as opposed to where numerical planners dominate: semantic prior knowledge and warm starts, hypothesis and mechanism-family proposals, natural-language objectives and constraints, large combinatorial action spaces, model misspecification. What do the most careful ablation studies conclude.
(4) What a strong reviewer at a statistics-leaning ML venue would demand as comparators and controls for a paper claiming an SCM-conditioned LLM intervention policy reduces experimental cost.
DELIVER: a positioning map (what is taken, what is open), a must-have comparator and control list, a list of candidate method fixes with citations, and an honest statement of which claims would be novel if demonstrated. Flag anything you could not verify. Do not invent citations: if you cannot open a source, say so.`,
  },
  {
    label: 'literature-benchmarks',
    prompt: `YOUR LABEL: literature-benchmarks
TASK: Web research on benchmarks, environments and venues (this is the one kind of network use you are permitted: literature and documentation search). Use WebSearch and WebFetch (load them through ToolSearch if they are deferred). Keep search queries generic and scientific. Never put any person's name from this project, the project name, repository paths or unpublished results into a query or URL.
First read the 'Benchmarks and a concrete experimentalist use case' section of ${G}/ace_llm_intervention_agenda_2026-11-02.md, and ${G}/external_benchmark_feasibility_2026-09-27.md and ${G}/causal_chambers_gate_2026-10-01.md, so you know what was already assessed locally.
RESEARCH QUESTIONS:
(1) Executable environments suitable for evaluating an adaptive causal intervention policy with a known or supplied graph, real do-operations, measurable costs and a useful outcome. Assess at least: Causal Chambers (is there a public remote-lab API, and do the authors publish mechanistic simulators of the chambers in an installable package), BoxingGym, CausalBench style gene perturbation data, standard interventional causal discovery testbeds, process-control simulators such as the Tennessee Eastman process, OpenModelica ThermoPower compressor models and Python FMU tooling, any compressor or turbomachinery map simulators with open licenses, CausalAssembly, and newer 2025 to 2026 benchmarks for LLM-driven scientific experimentation or automated experiment design. For each: what it provides, license, install burden, whether interventions are executable on demand or only logged, whether there is a natural experimental cost, and whether semantic variable descriptions exist that a language model could exploit.
(2) Which of these would let a language model show something a numerical planner cannot easily do: use of textual domain knowledge, natural-language objectives, mechanism-family hypotheses, misspecified numeric models.
(3) Upcoming venue deadlines after November 12, 2026 that fit a paper on LLM-guided causal experimental design: ICML 2027, UAI 2027, CLeaR 2027, NeurIPS 2027, TMLR (rolling), AISTATS 2027 status, relevant workshops. Give dates only if you can find them on an official page, otherwise say they are not announced and give the usual month from prior years, labeled as an estimate.
(4) How comparable papers structure their evaluation: number of systems or tasks, statistical tests, cost endpoints, ablations.
DELIVER: a ranked shortlist of 3 to 5 environments for a 'concrete experimentalist task' with a justification and integration effort estimate for each, plus the venue timeline. Flag anything you could not verify. Do not invent facts: if you cannot open a source, say so.`,
  },
]

const FORENSIC = [
  {
    label: 'results-analysis',
    prompt: `YOUR LABEL: results-analysis
TASK: Quantitative re-analysis of the saved 24-system comparison data with /usr/bin/python3 (pandas, numpy, scipy). Write scripts under ${SCRATCH}/results-analysis/ and run them. Do not write anywhere else.
DATA: ${RES}/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/ contains policies.csv, cells.csv (192 rows: world_id, policy_id, cost_stratum, graph_stratum, regime, capped_total_cost, capped_acquisition_cost, final_error, ever_attained, final_attained, failed, unresolved, actual_total_cost, termination_reason, first_crossing_total), curves.csv (4540 rows: world_id, policy_id, total_cost_units, acquisition_cost_units, normalized_mse, action_id, accepted_receipts, prediction_0..7), pairs.csv, strata.csv, actions.csv, display-data.json, derivation.json. Also read ${RES}/ace_llm_intervention_scientific_finalization_20261108/numerical-review01/review.md for the scope of the previous arithmetic review.
COMPUTE AND REPORT:
(1) Sanity: reproduce the policy summary table from cells.csv.
(2) Per-system table: error after initialization, and capped cost-to-target for each of the 8 policies. Identify the 10 systems that already met the target after initialization and characterize them by graph, regime and cost stratum.
(3) Headroom. On all 24 systems and on the informative subset (those not already at target after initialization): the best-of-8 envelope (minimum cost over all policies per system) versus target_risk. How many cost units could any policy have saved relative to target_risk even with hindsight. Is there measurable room above target_risk in this screen at all.
(4) More sensitive endpoints from curves.csv: for each run, the area under the log error versus cost curve over the acquisition phase, the error at fixed costs (12, 16, 24, 32), and the final error. Rank policies on each, with paired differences against target_risk and against uniform_random (mean, SD, sign counts, Wilcoxon signed-rank and an exact sign-flip permutation p-value, clearly labeled as descriptive post hoc numbers on development data).
(5) Power planning. Using the observed paired SD of (policy minus target_risk) for capped cost and for the curve-area endpoint, estimate how many independent systems are needed for 80 and 90 percent power to detect improvements of 1 and 2 cost units (and of 10 and 20 percent in the curve-area endpoint), and for a noninferiority margin of 1 cost unit. Show the formulae and note the heavy ties.
(6) Action structure of the strong policies. From the action_id sequences in curves.csv: for target_risk and local_information, the action distribution by graph stratum, cost stratum and regime, and by step index. Are their choices close to a fixed sequence given graph and cost menu (compute, for each graph by cost cell, how often the modal action at each step is chosen across systems). This measures how state-dependent good behavior actually is in this screen.
(7) Where the constant set_x_neg policy (the DPO arm) loses most and least, by graph, regime and cost stratum, with an explanation in terms of which mechanisms that action can and cannot identify.
(8) Non-monotonicity: how often does error rise after an acquisition, how often does a run cross the target and later leave it, per policy.
(9) The supervised arm: on which systems and at what costs did it stop, and what was the error at stopping.
(10) Recommendations, labeled as recommendations: endpoint and sample-size choices for a successor experiment, and whether initialization size or the threshold should change so that fewer systems are uninformative.
Give compact numeric tables in your notes file and the most important numbers in your structured return.`,
    verify: `Independently re-derive, with your own fresh code under ${SCRATCH}/results-analysis-verify/, the most consequential numbers claimed below from the raw CSV files in ${RES}/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/. Do not read or reuse the earlier agent's scripts. Check at least: the count and identity of systems at target after initialization, the headroom of the best-of-8 envelope against target_risk, the paired SD used for power planning and the resulting sample sizes, the curve-area ranking of policies, and the claim about how fixed the target_risk action sequences are. Report each as confirmed, refuted or partially confirmed with your value.`,
  },
  {
    label: 'training-forensics',
    prompt: `YOUR LABEL: training-forensics
TASK: Offline forensic analysis of the SAVED training supervision and training outputs, to test candidate explanations for the constant-action behavior. Read-only, /usr/bin/python3 analysis only, no model or weight loading. Write scripts under ${SCRATCH}/training-forensics/.
WHERE THE DATA IS: the training collection is at ${RES}/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection (about 12,400 files). Training outputs are at ${RES}/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training (about 235 files) and the CPU preparation at ${RES}/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/cpu/preparation. Tokenization preparation is at ${RES}/ace_llm_intervention_scientific_tokenization_20261108/terminal02/remote/launch/preparation. Acceptance reviews sit in each stage directory (for example ${RES}/ace_llm_intervention_scientific_training_gpu_20261108/acceptance01). Design documents: ${G}/ace_llm_intervention_training_collection_2026-11-07.md, ace_llm_intervention_training_exposure_2026-11-07.md, ace_llm_intervention_training_plan_2026-11-08.md, ace_llm_intervention_training_gpu_2026-11-08.md, ace_llm_intervention_training_dataset_2026-11-04.md. Code that defines the schemas: ${SR}/llm_intervention_training_collection.py, llm_intervention_training_exposure.py, llm_intervention_training_plan.py, llm_intervention_supervision.py, llm_intervention_replicated_feedback.py. Start by mapping the directory layout and file schemas (list directories, open a few files) before computing anything. Never open archive.tar files.
COMPUTE AND REPORT, WITH EXACT COUNTS:
(1) The schema of the saved supervision: what one supervised example and one preference comparison contain.
(2) Supervised targets: total examples, and the distribution of target commands overall, per teacher, per graph, per cost menu and per step index. What fraction is set_x_neg. What fraction is stop. Which 160 examples were actually used for the supervised updates and what was their target distribution. How well does the majority command alone predict the supervised target (majority-class accuracy), overall and within graph by cost cells.
(3) Preference comparisons: total (3,584 expected), the distribution over ordered (preferred, rejected) action pairs, the margin distribution, how ties and strict preferences were declared, and the composition of the 119 strict pairs used for DPO updates: frequency of each action on the preferred side and on the rejected side. Is set_x_neg overrepresented on the preferred side.
(4) State dependence of the preference labels. For each unordered action pair, in what fraction of states does each side win. Fit the simplest state-independent ranking (for example a Bradley-Terry strength per action, or simply win rates) and report how accurately a constant ranking predicts the strict labels. If a state-independent ranking predicts nearly all labels, the supervision itself rewards a constant policy.
(5) Label noise: compare replicate-to-replicate variability of the measured error reductions with the size of the margins. What share of comparisons are distinguishable from noise.
(6) Training logs: loss trajectories for the supervised and preference stages, any gradient or parameter-change records, and any saved evaluation bundles of command scores or generations before and after training (test names in the repository mention bundles with 28 scores and 10 generations). If command scores on fixed prompts are saved for the base, supervised and DPO checkpoints, tabulate them: which command wins on each prompt and by what margin.
(7) Teacher behavior: how state-dependent were the teachers themselves on the training states (for target risk and local information, how concentrated are their choices given graph and cost menu).
(8) A verdict table for the candidate explanations: (i) marginal label imbalance in supervised targets, (ii) preference pairs dominated by one preferred action, (iii) preference labels approximately state-independent, (iv) label noise, (v) too few updates. For each: supported, contradicted or undetermined by the saved evidence, with the numbers. Preserve contradicting evidence explicitly.`,
    verify: `Independently re-derive, with your own fresh code under ${SCRATCH}/training-forensics-verify/, the most consequential quantitative claims below from the raw saved training collection at ${RES}/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection and the training outputs at ${RES}/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training. Do not read or reuse the earlier agent's scripts. Check at least: the supervised target-command distribution and the share of set_x_neg, the composition of the 119 strict DPO pairs by preferred and rejected action, and the accuracy of a state-independent ranking on the preference labels. Also actively look for evidence that contradicts the earlier agent's verdicts. Report each as confirmed, refuted or partially confirmed with your value.`,
  },
  {
    label: 'deployment-forensics',
    prompt: `YOUR LABEL: deployment-forensics
TASK: Offline forensic analysis of the SAVED deployment records of the 24-system comparison, to measure decision-level state sensitivity of every policy. Read-only, /usr/bin/python3 analysis only, no model or weight loading. Write scripts under ${SCRATCH}/deployment-forensics/.
WHERE THE DATA IS: numerical arms at ${RES}/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu (about 7,100 files). LLM arms at ${RES}/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu (about 2,100 files). The 242 original reports at ${RES}/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque. Display data at ${RES}/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/display-data.json and curves.csv. Schemas are defined by ${SR}/llm_intervention_episode.py, llm_intervention_paired_episode.py, llm_intervention_episode_seal.py, llm_intervention_language_policy.py, llm_intervention_model_service.py, llm_intervention_screen_policy.py, llm_intervention_target_risk.py. Start by mapping the layout and schemas (list directories, open a few files) before computing anything. Never open archive.tar files.
COMPUTE AND REPORT, WITH EXACT COUNTS:
(1) What is recorded per decision for each arm. In particular: are per-command scores (a log-likelihood or numerical score for every legal action) saved at every decision for the frozen, supervised and DPO LLM arms, for the numerical ranker, and for target_risk and local_information. Give exact paths and field names.
(2) If LLM score vectors are saved: for each LLM arm, the distribution of the margin between the best and second-best command, how each command's score varies across decisions (between-state variance) relative to the gaps between commands, whether the rank order of commands ever changes with the state, and how the score vector correlates with state features (graph, cost menu, remaining budget, step index, size of posterior uncertainty). Summarize this as a state-sensitivity measure for the base, supervised and DPO models. Did training increase or destroy sensitivity relative to the frozen base.
(3) The supervised arm: characterize the 42 non-set_x_neg acquisitions and the stop decisions (which graphs, cost menus, steps, remaining budget).
(4) The frozen arm: the 2 non-observe actions, and how close other commands were.
(5) The numerical ranker: its action distribution, and how state-dependent it is. It was trained on matched examples and also underperformed. Compare its choices with the LLM arms and with target_risk at matched decision indexes (note that states diverge across arms after the first differing action, so use agreement at the first post-initialization decision, where all arms share the same state, as the clean comparison).
(6) Decision-level regret, if target_risk's per-action scores are saved or can be recomputed from saved posterior states without running new episodes: at the shared first post-initialization decision in each system, the target-risk score of every action, and the regret of each policy's chosen action under that score. This gives a decision-level endpoint with far more resolution than cost-to-target.
(7) At the shared first decision: how often is set_x_neg the target-risk optimum, and how peaked is the decision (gap between best and second-best action, and between best and set_x_neg).
(8) Prompt length, tokens and latency per decision for the LLM arms, if recorded.
(9) Anything in the records that contradicts the simple story 'the DPO adapter ignores the state'.`,
    verify: `Independently re-derive, with your own fresh code under ${SCRATCH}/deployment-forensics-verify/, the most consequential quantitative claims below from the raw saved deployment records at ${RES}/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu and ${RES}/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu. Do not read or reuse the earlier agent's scripts. Check at least: whether per-command score vectors exist and the state-sensitivity conclusions drawn from them, the first-decision agreement between policies and target_risk, and how often set_x_neg is the target-risk optimum at the first decision. Also actively look for evidence that contradicts the earlier agent's conclusions. Report each as confirmed, refuted or partially confirmed with your value.`,
  },
]

phase('Read')

const readerRuns = parallel(READERS.map(r => () =>
  agent(CONTEXT + '\n' + r.prompt, { label: 'read:' + r.label, phase: 'Read', schema: FINDINGS })
    .then(out => ({ label: r.label, out }))
))

const forensicRuns = pipeline(
  FORENSIC,
  f => agent(CONTEXT + '\n' + f.prompt, { label: 'read:' + f.label, phase: 'Read', schema: FINDINGS }),
  (out, f) => {
    if (!out) return { label: f.label, out: null, verify: null }
    const claims = JSON.stringify({ summary: out.summary, key_facts: out.key_facts, numbers: out.numbers }, null, 1)
    return agent(
      CONTEXT + '\nYOUR LABEL: ' + f.label + '-verify\nTASK: You are an independent verifier. A previous agent analyzed saved data and made the claims listed at the end. Your default stance is skeptical: a claim is refuted unless your own independent computation reproduces it.\n' + f.verify + '\nWrite your notes to ' + SCRATCH + '/' + f.label + '-verify.md.\n\nCLAIMS TO VERIFY (JSON from the earlier agent):\n' + claims,
      { label: 'verify:' + f.label, phase: 'Verify', schema: VERDICTS }
    ).then(v => ({ label: f.label, out, verify: v }))
  }
)

const [readers, forensics] = await Promise.all([readerRuns, forensicRuns])

const result = {}
const dropped = []
for (const r of readers) {
  if (r && r.out) result[r.label] = r.out
  else dropped.push(r ? r.label : 'unknown-reader')
}
FORENSIC.forEach((f, i) => {
  const r = forensics[i]
  if (r && r.out) result[f.label] = { findings: r.out, verification: r.verify }
  else dropped.push(f.label)
})
if (dropped.length) log('DROPPED (no result): ' + dropped.join(', '))
return { result, dropped }
