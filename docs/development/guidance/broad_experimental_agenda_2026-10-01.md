# SCMs and foundation models: an experimental portfolio

**Designed 1 October 2026. Status: proposed studies, not new results.**

This document broadens the [recovery agenda](research_recovery_agenda_2026-10-01.md). It preserves that agenda's external-task and independent-language gates, and adds explicit studies of intervention-trained models, causal memory, representation, and hypothesis search. The [execution ledger](research_execution_2026-09-25.md) remains authoritative for completed work. The initial work is local analysis and CPU feasibility; a listed experiment is not a submitted job. No Azure or other closed-source model calls. No changes to another project's jobs.

## 1. Scientific position

The question worth pursuing is: **what reusable knowledge helps a learner predict and choose interventions in a new system, and what evidence tells it when that knowledge has stopped applying?** This includes numerical knowledge learned across systems, structured experiment memory, language about an apparatus, and representations of observations. It does not require an LLM to choose every action.

Our prior failures narrow the problem but do not settle it. Matched learners and evaluation removed the original ACE advantage; incorrect semantic priors sometimes hurt; local residual repair improved changed mechanisms but damaged unchanged ones; a tiny direct-prompt model failed an exact reasoning canary. These findings do not rule out intervention-trained inference, better experiment memory, or a different model with a verifiable interface. They do require strong controls and a clear source of useful information.

The most promising synthesis is **a reusable model of intervention responses that updates from a small experiment history, with explicit uncertainty about mechanism validity and support**. Language is optional: it can name an actuator, propose a constrained hypothesis, or help retrieve experience. Each role must earn its place through a separate comparison.

### What the evidence establishes

- The [25 September handoff](handoff_2026-09-25.md) records why the original superiority claims failed: evaluation, frozen-world, and learner confounds. The old headline cannot be reused.
- On the same-menu numerical pair-design task, the [binary-tree study](../../../results/local_connected_factorial_pair_confirmation_20260929/README.md) and [random-recursive study](../../../results/local_connected_random_recursive_pair_20261001/README.md) passed their gates; [fanout](../../../results/local_connected_fanout_pair_confirmation_20260929/README.md) did not. The final comparisons use mean feasible MSE .02694/.03972, .02270/.05918, and .01327/.01535 respectively for risk versus fixed factorial hub. These are sequential synthetic studies with related mechanism equations, not three independent replications of a general claim.
- Semantic proposals have not shown a distinct benefit over numerical selection. A wrong source-count cue was accepted on all three reused BoxingGym Signal worlds. Acceptance does not prove the cue true, especially when a misspecified smaller model predicts better at low sample size.
- [Action-block CV repair](../../../results/local_action_blocked_residual_dev_20261001/README.md) selected 66 unchanged-motif repairs and increased unchanged error by 8.8%–21.8%. The failure is more specific than “transfer fails”: we lack a reliable distinction between changed mechanisms, shifted parent support, and estimation noise.
- The [language stress test](../../../results/local_action_language_temporal_stress_eval_20261001/README.md) exposed parser brittleness, not model superiority. The [open-model partial-ID canary](../../../results/research_partial_id_open_canary_20261001_v2/README.md) exposed both schema and reasoning failures; its deterministic control already solves the finite suite.
- PEV and NeuronBench controls remain mixed. Forecaster choice changes some acquisition rankings, so policy and predictor effects must be separated.

### What is not yet known

We do not know whether pair gains are caused by interaction excitation, graph geometry, posterior calibration, or a weakness of the fixed comparator. We do not know whether a mechanism change is detectable within the allowed target budget, whether text adds information absent from numerical data, or whether amortized learning can reuse our experimental structure outside its training generators. These are the first objects of study.

## 2. A common experimental structure

Represent a task by public metadata M, acquired history H={(action, response, mask, cost)}, a legal action interface A, and a held-out intervention distribution Q. The system predicts response distributions and optionally ranks actions. A private evaluator owns latent parameters, gold constraints, and held-out outcomes. Individual counterfactuals are a different target and require explicit cross-world assumptions; interventional prediction alone does not identify them.

The architecture has replaceable components: observation encoder, context/memory encoder, mechanism predictor, uncertainty estimator, acquisition rule, and optional language interface. Start with known graphs and observed variables where possible; only later introduce graph uncertainty or hidden variables. Do not change those assumptions and claim the result is the same task.

For every candidate, perform **same-data prediction first**, then **same-learner acquisition**, then a factorial comparison if both improve. For language plus acquisition, the four essential cells are numerical/numerical, language/numerical, numerical/adaptive, language/adaptive. Keep data, model capacity, and fitting budget matched within the relevant contrast.

Every experiment gets a source pin, task-family split, proposed effect, primary outcome, exact query ledger, baseline implementation, compute allowance, and explicit stop condition before outcomes are examined. Include receipts for failures. Fresh random noise on one physical system is not an independent system.

## 3. Eleven work packages

### B1. Explain when joint interventions help

**Hypothesis.** Pair selection is useful when a single action cannot adequately excite the parent configurations that determine an important downstream mechanism. Graph size alone is a poor predictor of that benefit.

**First experiments.** B1a is a zero-query analysis of saved action trajectories: parent support coverage, feature-matrix conditioning, interaction exposure, posterior calibration, action frequency, and per-mechanism error. Relate these to paired performance with leave-one-topology-out analysis; call this exploratory. B1b freezes a small crossed experiment varying parent correlation, interaction strength, action cost, and observation noise while matching graph size and learner. Include additive systems and systems whose parents are already well covered as negative controls. Use independent equations and a fixed crossed design, not a sequence of favorable families.

**Controls and endpoints.** Best fixed factorial/space-filling design, randomized legal pairs, greedy D/A-optimal design where defined, and an ablation that retains the pair menu but removes posterior adaptation. Match realized cost and test on the same reachable intervention distribution. Score error-versus-cost and calibration; count simulator calls used in acquisition. Candidate enumeration is not an oracle query unless a response is obtained.

**Decision.** An explanatory feature that predicts gains on a held-out family motivates a narrow theory and external replication. If the best fixed design closes the gap, stop the adaptive superiority claim and retain the finding about experimental support. No FM is needed here. Local CPU is sufficient for the audit.

### B2. Transfer action design to an independently authored system

**Hypothesis.** The useful component transfers to a new intervention process after the learner is calibrated there.

**First experiments.** B2a audits CausalMan at the pinned revision in the recovery plan and a Causal Chambers recorded-action dataset. Check legal pairs, actuator semantics, available repetitions, private/public separation, world independence, per-query runtime, and licenses. B2b runs one bounded CPU custody smoke for the first viable task, then fixed-data predictor comparisons before policy comparisons. CausalMan is an external simulator; Causal Chambers is physical recorded data. Keep those evidence types distinct.

**Controls and endpoints.** Constant/linear, flexible nonparametric, and appropriate mechanistic predictors on identical observations; matched fixed/random/coverage policies with the selected learner. On a recorded pool, charge every label revealed and evaluate only supported actions. No invented response at an unobserved action; no claim of physical online experimentation from replay. Split by apparatus configuration or experimental session where available, not random rows.

**Decision.** Preserve the existing 20% practical improvement target for B2 confirmation over the strongest simple policy, with a paired interval excluding zero. If only one apparatus or model exists, limit the claim to that system; do not manufacture population replication by reseeding noise. Drop a candidate after one documented feasibility failure and use the predefined fallback. CPU first; no GPU for download or adapter construction.

### C1. Distinguish mechanism drift from shifted inputs

**Hypothesis.** A targeted diagnostic experiment can separate a changed conditional mechanism from an unchanged mechanism receiving unfamiliar inputs, more reliably than passive residual magnitude.

**First experiments.** C1a uses saved data to classify four situations: unchanged/in-support, unchanged/shifted support, changed/in-support, and changed/shifted support. Compare uncertainty and false-repair rates at a frozen coverage level. This is the one bounded redesign permitted by the recovery plan. C1b, only if the task is recoverable, uses paired probe actions that revisit comparable parent contexts to test conditional invariance. Count these probes against the adaptation budget. If matching parent contexts is impossible through available actuators, record that limitation explicitly.

**Architecture.** A frozen source predictor plus a sparse residual module and a three-way decision: reuse, repair, or request a diagnostic probe. Inputs include local support and intervention masks, not hidden change labels. A learned gate is a later option; start with conditional two-sample or likelihood-ratio controls and sequential evidence thresholds that account for adaptive sampling.

**Controls and endpoints.** Source-only, scratch, full fine-tuning, retrieval mixture, and an evaluator-only oracle that knows which mechanism changed. Report changed-node gain, unchanged-node harm, abstention/coverage, and total query cost separately. The oracle is a recoverability test, not an eligible method.

**Decision.** Require unchanged error within the existing 5% margin in every prespecified stratum, alongside changed-mechanism improvement. Fresh confirmation needs simultaneous uncertainty bounds for the protection claims. Failure with oracle localization suggests insufficient support/model capacity; failure only in the gate suggests a detection problem. Stop threshold tuning on the exposed 12 worlds.

### F1. Learn reusable inference from intervention context

**Hypothesis.** Pretraining on diverse experiment histories can teach a small model to infer intervention responses in new systems with fewer target observations than fitting from scratch.

**First experiments.** F1a builds a CPU task generator and split audit. Hold out equation families, graph generators, parameter ranges, and action combinations separately. Reserve external tasks completely. Verify that a per-task numerical/oracle method can solve the task and that a constant or retrieval baseline cannot exploit an artifact. F1b is a small model proof of concept with matched capacity and training data; it does not yet qualify as a foundation model.

**Architecture.** An intervention-conditioned set/graph transformer. Each observation token includes values, missingness, intervention targets/values, and cost. The context encoder aggregates rows; per-variable mechanism slots condition a shared response decoder. Train on held-out interventions using a proper predictive score. Use variable permutation augmentation and permutation-equivariant components; only treat experiment order as exchangeable for static, resettable systems. Dynamic histories need time and carryover information. Known graph, uncertain graph, and no-graph variants are separate ablations.

**Controls.** Ridge/kernel/GP or ensembles as appropriate, numerical source retrieval, a parameter-matched generic context transformer, the same architecture trained on observational data only, and shuffled intervention masks. Compare a pretrained initialization with equal-compute scratch training. Report both matched target-data performance and total pretraining/inference compute; they answer different questions.

**Decision.** Promote only if intervention pretraining improves an unseen mechanism-family split and uncertainty remains usable under shift. A seed split alone cannot demonstrate reusable causal inference. The falsifier is a gain that vanishes with a generic context model or only survives on training-family equations. A bounded one-GPU prototype is permissible after recoverability and data checks; it need not wait for a hand-written method to solve the very problem the learned model is intended to solve.

### F2. Learn fast experiment selection

**Hypothesis.** Once a numerical policy is useful, a learned policy can preserve its quality while reducing decision time, or generalize selection across task sizes.

**First experiments.** F2a profiles the current selector to establish whether action computation is a real bottleneck. Build an offline history/action-value dataset without test worlds. F2b distills a set/graph policy that sees only public history and action descriptors. Direct optimization of information or prediction value is a later ablation after imitation is stable.

**Controls and endpoints.** Teacher policy, cheap greedy design, random/fixed design, and nearest-history retrieval. Measure regret relative to the teacher, held-out prediction, decision latency, and total amortization break-even over repeated tasks. Include unseen graph size and illegal/unavailable action masks. Approximate oracle labels used for training must be declared and charged to pretraining compute.

**Decision.** Do not train if the existing selector is already cheap relative to experiments and inference. A speed claim can succeed without better accuracy; prespecify a practical latency target and acceptable loss before training. This direction follows established adaptive-design literature and needs a specific transfer or reliability contribution.

### A1. Treat semantic priors as hypotheses to challenge

**Hypothesis.** Text is useful when it narrows a mechanism class that the initial numerical data cannot distinguish, and its errors can be discovered through a small number of targeted experiments.

**First experiments.** A1a performs the bounded saved-data abstention diagnostic in the recovery plan. A1b constructs new tasks with two or more hypotheses that fit the initial observations but diverge under an affordable legal intervention. Cross correct, misleading, irrelevant, absent, and paraphrased metadata. Freeze family names and transformations so familiar-law retrieval is measured separately from compositional generalization.

**Architecture.** A typed candidate library with provenance and data-updated weights. The controller selects an action that challenges the leading candidate when its plausible alternatives predict different responses. Start with manually supplied candidate hypotheses, then replace only the proposal component with an offline open model if candidate quality can change the outcome. Do not interpret an LLM's verbal confidence as a calibrated probability.

**Controls and endpoints.** Broad/data-only selector, numerical grammar enumeration, equal-size random candidate library, retrieval, and privileged correct-family candidate. Equal fitting budgets and same acquired observations first. Measure wrong-cue regret, recovery cost, correct-cue benefit, and predictive log score. The target need not be true-family recovery when families are observationally indistinguishable; score the declared identifiable quantity.

**Decision.** Wrong-cue harm must remain within 5% of the broad control and correct cues must supply a practical benefit. If an inexpensive data-only selector ties, language has not earned an advantage. If no allowed action separates the candidates, the correct outcome is unresolved uncertainty, not forced model selection.

### D1. Compile experimental instructions without giving the answer to the compiler

**Hypothesis.** An open model can turn independently authored apparatus descriptions into a useful action schema better than strong non-model extraction.

**Critical boundary.** The new local scorer reads the gold schema to grade proposals. It is an offline evaluator, not evidence that a deployed compiler knows the rules. If a runtime validator has the complete formal schema and can enumerate the legal menu, that enumerator is an obligatory control and language cannot claim an action-availability advantage. Separate public type/unit checks, any genuinely available trusted safety kernel, and hidden semantic gold used only for evaluation. All policies get the same runtime kernel.

**First experiments.** D1a collects at least 20 independently authored source descriptions, with source documents held out as groups and gold adjudicated before inference. Record ambiguity and permit abstention. Include costs, joint exclusions, proxy actuators, units, temporal constraints, and no-action cases. The existing static schema cannot express all of these; extend the formal representation and audit evaluator correctness before admitting such tasks. Twenty descriptions are an intake minimum, not a powered study or twenty independent domains. D1b compares frozen rules, templates, lexical/retrieval controls, and one constrained-output open model. Later compare unconstrained output to distinguish serialization from semantic errors.

**Endpoints.** Correct bindings and constraints, legal-menu recall, invalid proposals, unsupported claims, abstention, and useful reachable experiments. For enormous menus evaluate query-based legality and sampled coverage rather than exhaustive enumeration. Safety rejection counts as a failed proposal and consumes its declared proposal budget. Zero invalid executed actions may be enforced by the environment; it cannot by itself establish model competence.

**Decision.** Proceed to downstream acquisition only if semantics improve over the strongest extractor on held-out source documents and the improvement survives symbol renaming and irrelevant-text controls. Do not present same-author paraphrases or twenty clauses from one manual as independent transfer. Public source acquisition can begin without waiting for a collaborator; ambiguous cases remain excluded pending adjudication.

### M1. Use causal experiment memory as foundation-model context

**Hypothesis.** A compact record of interventions, supported conclusions, and unresolved alternatives is more useful than a narrative transcript when context is limited or the environment changes.

**First experiments.** M1a constructs numerical context tasks with a fixed token/record budget. Compare raw chronological history, random/truncated history, nearest-neighbor retrieval, sufficient-statistic summaries when available, and mechanism-indexed records. Inject obsolete episodes, renamed variables, altered units, and one contradicted mechanism. Require every retained claim to link to the actual experiment that supports it. Start with deterministic retrieval and a numerical consumer; open-model context processing is a conditional follow-up.

**Architecture.** A memory graph of hypothesis–experiment–outcome records, with scope, uncertainty, intervention mask, and expiration/contradiction status. Retrieval follows the target intervention's relevant mechanisms, but must account for graph uncertainty. Updating one invalid claim removes or weakens dependent conclusions. Compare this bookkeeping to ordinary retrieval with the same memory budget.

**Endpoints.** Prediction and action quality at equal context budget, stale-claim use, recovery after contradiction, and preservation of unaffected predictions. The genuinely new target is selective revision under evidence and scope changes, not a longer chat history. Do not use synthetic gold causal summaries unless explicitly labeled an oracle upper bound.

**Decision.** If deterministic sufficient statistics dominate, retain them. Only run a model comparison where heterogeneous records or ambiguous mappings leave demonstrated headroom.

### R1. Test whether learned representations preserve interventions

**Hypothesis.** A representation that is good at observational prediction may discard distinctions needed for intervention prediction; intervention-aware training or alignment can restore them.

**First experiments.** R1a renders a small known SCM through a fixed nonlinear observation map into vectors and then optional images. Contrast invertible and deliberately information-losing maps. Compare latent-state oracle, raw observations, PCA/random encoder, predictive encoder, and intervention-trained encoder. Hold out interventions and renderer styles. R1b conditionally tests a frozen open representation with a small causal adapter; existing open weights only after license/memory audit.

**Architecture.** An encoder with variable or object slots, sparse mechanism decoder, and a consistency loss between intervening in latent space and encoding the resulting observation. Alignment can be evaluated with interchange interventions and off-target controls. Treat this as a falsifiable abstraction hypothesis; latent identifiability depends on intervention coverage, observation map, and model assumptions.

**Endpoints.** Held-out intervention prediction, intervention composition, preservation of unrelated factors, and transfer across renderers. Linear probe accuracy is insufficient. A model trained with true latent labels is privileged supervision; match or disclose that information.

**Decision.** Stop causal abstraction claims if observational encoders with matched capacity perform equally, if information loss makes the task impossible, or if gains require test-latent access. This is a higher-risk direction after the numerical and data gates.

### H1. Use models to propose hypotheses that exact tools can refute

**Hypothesis.** In a large candidate space, an open model or learned proposal network can reduce search cost while an exact checker protects conclusions within an explicitly declared hypothesis class.

**First experiments.** H1a expands the finite partial-ID suite into a restricted SCM program grammar with an exact small-instance oracle and controlled growth in candidates. Separate proposal recall, consistency checking, and identification. Include misspecification: the true program is absent from the candidate class. H1b compares a learned/open-model proposer with enumeration, branch-and-bound or SAT/SMT where applicable, and numerical heuristic search under equal compute and proposal counts.

**Architecture.** A counterexample-guided loop: propose candidate structure or distinguishing query, obtain an exact certificate/counterexample, update a typed hypothesis set. Proposal code is restricted to a safe grammar. Do not execute arbitrary generated programs. Soundness of a certificate is conditional on the checker and hypothesis-class assumptions; an incomplete candidate list cannot certify identification over all SCMs.

**Endpoints.** Time and proposals to useful hypotheses, candidate coverage on exhaustively solvable instances, false identification claims, interval calibration/width, and intervention value. The old 20-task suite stays a regression/refusal control, not a gain benchmark.

**Decision.** Continue only if the proposal mechanism offers a meaningful search benefit beyond exact/heuristic controls without increasing false certainty. Failure of the old 1.5B direct prompt does not justify repeated prompting or establish failure of all verified architectures.

### X1. Test combinatorial response prediction on real perturbation data

**Hypothesis.** Mechanism composition can predict a held-out combination of perturbations better than adding the observed single-perturbation effects.

**First experiments.** X1a audits one public combinatorial perturbation dataset for biological replicates, batch effects, intervention efficacy, pair coverage, and licenses. Start with a reduced feature set and CPU baselines: no-change, mean effect, additive singles, low-rank regression, and nearest perturbation. X1b conditionally compares modular composition with a generic interaction model and frozen open domain embeddings.

**Endpoints.** Held-out pair interaction residuals and reproducible response features; both pair-with-seen-components and unseen-component splits, kept separate. Group by perturbation and biological replicate, not random cells. Gene perturbations are imperfect interventions and the true cell SCM is unknown. Claim response prediction under the dataset protocol, not recovered molecular causality.

**Decision.** This is an optional external transfer lane, not the first GPU project. If additive/low-rank controls exhaust reproducible signal, stop. Audit pretrained-model overlap with the evaluation dataset. Domain expertise may be needed to adjudicate biological claims, but data feasibility and simple baselines can proceed locally.

## 4. Portfolio-wide discriminating experiments

The same small set of manipulations makes many packages more informative:

1. **Information removal:** anonymous labels, shuffled metadata, shuffled intervention masks, or absent source history. Establish which information supplies a gain.
2. **False information:** wrong mechanism description, stale episode, incorrect graph edge, or unit mismatch. Evaluate recovery and false certainty, not only average accuracy.
3. **Composition:** familiar modules in an unfamiliar graph or action combination. Keep genuinely unseen equations as a separate, harder split.
4. **Support shift:** distinguish new parameter values, new parent contexts, new intervention types, and new domains. Do not pool them into one OOD score.
5. **No possible gain:** additive/fully covered systems, uninformative action menus, exact sufficient statistics, or nonidentifiable targets. A useful method should abstain or tie here.
6. **Capacity and compute controls:** match learners and fitting effort; separately report the benefit of extra pretraining and the cost required to obtain it.

This yields an interpretable failure map rather than a single leaderboard. Factor interactions must be frozen before confirmatory analysis; explanatory correlations found after scoring remain exploratory.

## 5. Statistical and custody rules

- Development examples may guide implementation but cannot become confirmation by renaming their seeds. Lock whole families/source documents/physical configurations as appropriate. Search public corpora for provenance; holdout wording alone cannot exclude foundation-model pretraining exposure.
- Choose one primary contrast per promoted package. Use paired task-level comparisons and cluster uncertainty at the independently sampled system/document/biological unit. Training seeds are nested repeats, not independent environments.
- Use development variance to simulate power for a practical effect before freezing confirmation. Twenty to forty systems is a starting design range, not a guarantee of power. Expand once according to a prespecified rule or use a declared sequential method; do not peek and add seeds until significance.
- Preserve existing B2 20% effect and C1/A1 5% harm margins. New lanes need endpoint-specific tolerances before scoring; do not copy a percentage to incomparable metrics. A safety/noninferiority claim needs an appropriate upper bound, not a nonsignificant harm test.
- Broad screening is exploratory. For a common confirmatory family of selected claims, use a prespecified multiplicity procedure such as Holm; alternatively designate a single primary paper claim and clearly label all other analyses. Selection across tasks remains part of the research history even after a fresh test.
- Score prediction and decisions separately. Count all observed responses, failed/rejected proposals, test queries, source data, preprocessing, pretraining, and inference. Benchmark-only oracle labels and hidden graph truth never reach the policy.
- Check query equivalence, natural-label masks, stable worlds, metric parity, strict output parsing, and train/test leakage. Do not count direct interventions on a child as natural labels for that child's mechanism.
- Keep immutable raw outputs, task/source revisions, environment lockfiles, hashes, and independent score recomputation. Byte-identical replay is useful for deterministic pipelines; tolerance-based numerical replay with seed/hardware records is appropriate for nondeterministic GPU computation.

## 6. Execution waves and resource ceilings

These are planning caps, not resource requests or measured runtime estimates. Each job still needs an actual pilot and explicit account/output/time. Queue waiting is acceptable. Run at most four small ACE CPU jobs concurrently in the first wave and one GPU experiment at a time unless a later measurement justifies more.

**Wave 0 — locally, first 2–3 working days.** Produce B1a saved-result diagnosis; B2a source/data feasibility; D1a source/provenance and schema-gap inventory; and F1a intervention-context task/split design. In parallel, specify C1a and A1a as their single bounded redesigns. Initial allowance: up to two CPU-hours per diagnostic before profiling and deciding whether CURC is appropriate. This is effort ordering, not a promise that all research completes within three days. No model training or model API calls are required.

**Wave 1 — CPU discrimination, roughly the following week.** Run the first viable external custody smoke; compare strong fixed-data predictors; evaluate B1b and the C1/A1 recoverability controls; create deterministic M1/H1 prototypes and R1 information-loss controls. Prioritize B2 and F1 preparation, with D1 running independently as source material becomes ready. Aggregate planning ceiling: 80 CPU-core-hours on CURC, at most 16 per package before a stop/go review. Download volumes and host-memory requests come from data audits. F1 task generation should be streamed/bounded rather than producing an uncontrolled archive.

**Wave 2 — at most two learned candidates.** Select one numerical candidate (prefer F1; C1 only if protection is plausible) and one language/context candidate (D1, A1, M1, or H1). Each gets a small offline smoke with a cap of 15 GPU-minutes if feasible from measured load time, followed by at most four GPU-hours of prototype work only when the smoke verifies progress and learning/valid outputs. Combined initial ceiling: 8.5 GPU-hours, single-GPU jobs; actual allocation may be much smaller. Record memory, utilization, process/framework counters, tokens/steps, startup and compute timings. Do CPU preparation first. A 15-minute limit that cannot load the selected model is a reason to choose a smaller model or justify a different measured limit before submission.

**Wave 3 — fresh confirmation for at most two claims.** Freeze task families, primary endpoints, controls, and power calculation. Use CURC CPU for numerical matrices and the smallest measured GPU allocation for genuinely GPU-bound work. Report positive, negative, and stopped lanes together. External confirmation precedes broad generalization claims. R1 and X1 expand only if an earlier lane supplies reusable components and their own feasibility gates pass.

The hourly monitor should record the next discriminating deliverable for each active lane and its blocker. It should not keep rerunning checks without advancing a local artifact when cluster access is unavailable. Do not keep the queue busy for its own sake. This document does not change the scheduler configuration or claim a live cluster status.

## 7. Deliverables and possible papers

The immediate deliverable is a source-pinned experiment registry and a small evidence packet for each active lane: question, data boundary, strong control, recoverability, resource measurement, and stop decision. Publish a frozen fixture only after its independent authorship and adjudication have actually occurred; metadata alone does not establish independence.

Three potential contributions can emerge independently:

- **Conditional intervention design:** explain support/interaction conditions, then replicate a matched-cost gain externally. This can be a numerical causal-learning paper with no FM claim.
- **Reliable reuse under intervention and drift:** an intervention-trained context model or modular memory that generalizes across mechanisms and knows when to probe or abstain. A small proof of concept supports an architectural result; the term foundation model requires broader demonstrated reuse.
- **Verified scientific interfaces:** language compilation or hypothesis proposals that improve search/experiment access over strong deterministic controls, with explicit boundaries on safety and identification.

The corrected ACE audit and negative experiments are retained as a technical report and methodological evidence. Do not force all lanes into the old paper or let an unverified submission deadline set the evidentiary standard. No specific venue/deadline is assumed by this plan.

## 8. Prior art and novelty boundaries

Sources checked 1 October 2026. This is a targeted landscape check, not a systematic review or a claim of novelty.

- [AVICI: Amortized Inference for Causal Structure Learning](https://arxiv.org/abs/2205.12934) already learns causal inference from simulated observational/interventional data with architectural symmetries. F1 must distinguish intervention-response transfer and reliability from existing structure inference.
- [CausalFM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7f8cabaf2de70e1a9d3eb187f02bd58c-Abstract-Conference.html) explicitly trains prior-data fitted models for causal inference. “Train a transformer on SCMs” is therefore not itself a new contribution.
- [Deep Adaptive Design](https://arxiv.org/abs/2103.02438), [implicit DAD](https://arxiv.org/abs/2111.02329), and [Deep Adaptive Bayesian Screening](https://arxiv.org/abs/2607.16927) already learn experiment policies. F2 needs matched design quality, transfer, or a specific reliability result.
- [LLM-SR](https://arxiv.org/abs/2404.18400) combines equation proposals and numerical search. [Model Discovery Agent, v5](https://arxiv.org/abs/2608.09696v5) combines proposals, Bayesian inference, and value-of-information design. A1/H1 need wrong-prior recovery, verified search, or a materially different task.
- [BoxingGym](https://arxiv.org/abs/2501.01540) evaluates model discovery and experiment selection. Reproducing an agent loop on its tasks is a baseline activity.
- [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) studies causal abstractions of neural representations. R1 must use held-out intervention tests and matched representation controls.
- [Causal Chambers](https://arxiv.org/abs/2404.11341) supplies physical intervention systems/data; [CausalMan](https://github.com/boschresearch/CausalMan) supplies a manufacturing simulator. Their suitability for our exact acquisition task remains a feasibility question.
- [Ahlmann-Eltze, Huber, and Anders](https://www.nature.com/articles/s41592-025-02772-6) found simple linear controls competitive against the tested perturbation models; [Systema](https://www.nature.com/articles/s41587-025-02777-8) examines systematic variation in perturbation evaluation. X1 must include additive/linear controls and avoid metrics dominated by common expression structure.

Novelty, if established, should reside in a demonstrated capability and the conditions under which it works: learning useful intervention context, challenging wrong priors, distinguishing support shift from mechanism change, or preserving causal conclusions through selective memory revision.
