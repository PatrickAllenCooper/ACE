# Research portfolio: learning, testing, and reusing causal mechanisms

Date: 25 September 2026. Starting revision: `f01a158`.
Status: proposed investigations; no new experiment jobs or paid API calls launched by this planning pass.

## Recommendation

Pursue three scientific directions in parallel: **fallible semantic priors over mechanisms**, **interventions that expose otherwise hidden interactions**, and **reuse and local repair of mechanisms across systems**. Maintain a smaller control track that settles the existing PEV result and makes the evaluation trustworthy. Reserve a small exploratory track for using SCMs to test whether foundation models revise their beliefs when evidence contradicts familiar semantics.

The organizing hypothesis is that a foundation model may supply useful *structure before data*, while an SCM supplies a testable, modular account of what happens under interventions. The language model should make infrequent, inspectable proposals. Numerical inference, acquired evidence, and explicit experimental costs should govern acceptance and action selection. A model's verbal confidence is not a posterior probability.

My highest-priority longer-term architecture is a library of reusable local mechanisms with small task-specific adapters and explicit uncertainty. A new system should inherit useful modules; a changed system should require learning only the changed mechanisms when the assumptions permit it. Increasing graph size then need not mean relearning every function or repeatedly presenting the entire graph to an LM. This is a hypothesis about transfer and sparse change, not a proven complexity result.

The September 17 design note already proposed mechanism priors, an amortized critic, and an in-context estimator. Those are predecessors of this plan, not new discoveries here. The additions are falsifiable experiments on misleading semantics, intervention reachability, sparse repair, stronger controls, and explicit decision gates.

## 1. What we actually have

Evidence sources: [handoff](handoff_2026-09-25.md), [metric audit](metric_audit_2026-09-10.md), [PEV design](pev_design.md), [paper reframe](paper_reframe_2026-09-17.md), [local pilot README](../../../results/local_matched_prelim_20260917/README.md). Paths in this document are relative to this file unless marked as code paths.

### Established problems and reusable assets

- The original ACE superiority claims cannot be retained: root weighting, evaluation distribution, a changing baseline simulator, and learner capacity/training differed. Correcting a p-value cannot repair these comparisons.
- We have corrected simulator/evaluator code, a metric parity test, extensive baseline artifacts, intervention logs, two SCM families, CPU baseline runners, and PEV implementations. These are valuable infrastructure and diagnostics.
- The audit reruns fixed important problems but retained the smaller baseline learner. They are historical references, not the final matched baseline for every new method.
- Larger tested ACE/DPO settings did not establish an advantage over Random. The handoff reports harm from DPO in the examined regimes. This does not show that preference learning is universally harmful or that all active design is ineffective.
- The repository's historical “Bayesian OED” is oracle lookahead, not an adequate test of posterior-predictive Bayesian experimental design. Rename it in future reporting; include a genuine small-system Bayesian design control.
- GPU jobs repeatedly exhausted **host RAM**, including the ranking canary after the attempted cleanup fix. The source of the growth remains unresolved. More RAM and a repeated submission are not evidence of a fix.

### Promising but provisional

The 5-node local pilot reports end-of-campaign non-root broad-domain MSE of approximately 0.165 for ACE, 0.161 for round-robin, 0.223 for Random, 0.172 for Random with an ensemble, 0.045 for PEV, and 0.039 for PEV-var. Sample sizes and campaigns differ across historical ACE and pilot arms; these are not a newly certified common-protocol comparison. The PEV pilot has three seeds and 40 reset campaigns per seed. Repeated campaigns on one fixed SCM do not provide independent graph-level replication.

PEV is worth investigating. The naive variance ablation being numerically better means the pilot does **not** establish that the proposed integrated-variance score is the source of the gain. No statistically significant ACE/round-robin difference is not proof of equivalence. No large-system PEV success has yet been demonstrated.

### Additional issues visible in code

At the starting revision:

- `baselines.py:SCMLearner.train_step` trains on a masked replay buffer; the ACE path has different training machinery. Matching `(64,64)` and 100 epochs alone does not match the learner. Executed samples, observation refresh, root handling, stopping, initialization, and evaluation timing also need alignment.
- `run_baseline` creates a fresh student every episode. Its query budget is checked at the start of an episode, and counts printed in each row are final run totals. Extra episodes can consume more data without improving the same learner. This cannot answer the persistent-system sample-efficiency question.
- ACE records detailed mechanism loss before taking an action; the baseline records after updating. Some ACE proposal/scoring paths see ground-truth mechanism losses. A deployable method cannot receive hidden evaluator labels; retain such a controller only as an explicitly privileged diagnostic.
- Root membership in the aggregator is inferred from naming conventions, with a heuristic fallback. New datasets must serialize the graph and identify roots structurally, including isolated nodes.
- PEV removes leaf candidates. Its comparison must include Random and round-robin with that same restriction, plus a cheap coverage policy. Otherwise graph knowledge can be mistaken for uncertainty-directed acquisition.
- PEV simulates the ensemble mean and root noise, omits non-root process noise, and uses only five members. Its estimated residual variance includes model error. Its covariance score approximates a single-observation variance reduction and does not account exactly for correlation within an acquired batch.
- PEV currently evaluates O(N) candidate targets against O(N) mechanisms. General linear scaling in N is not established. Sparse graph scoring and cached local changes are research/engineering tasks.

Two prior interpretations should be retired: Random has not been proved optimal on the homogeneous family, and a 16-unit ReLU network is not categorically incapable of approximating sine on a bounded interval. The actionable issues are relative capacity, fit, data coverage, and uncertainty calibration.

## 2. Shared scientific contract

### Separate the scientific questions

1. **Within-system acquisition:** one persistent learner, one fixed SCM, a sequence of interventions, exact cumulative data budget.
2. **Across-system transfer:** a learner or library trained on separate SCMs, then evaluated on a new system. Report pretraining cost separately and its amortization across deployment counts.
3. **Historical reproduction:** reset campaigns retain their original meaning and are clearly labeled. Do not pool them into the first two questions.

Begin with known DAGs and fully observed variables, preserving ACE's mechanism-learning task. Unknown graphs, hidden confounding, and partial observations are later stress tests, each explicitly changing the problem.

### One harness, two controlled comparisons

For acquisition comparisons, use exactly the same simulator instance, learner, replay, optimizer, observation schedule, per-action samples, and update count. For architecture comparisons, fix the acquired dataset first, then allow each architecture the same declared tuning budget. Afterwards evaluate the strongest combinations online. Report both data efficiency and compute/error tradeoffs; a five-member ensemble has additional computation even at equal environment samples.

Each run records a system specification hash, all nodes/edges/coefficients/forms/noise distributions, graph seed, mechanism seed, learner seed, acquisition seed, source revision, environment version, model/prompt version, and parent context coverage. Check sample budget **before each oracle call**, including initialization, observations, candidate probes, and repair calls. Evaluation samples belong to a separate inaccessible test service. If validation labels drive actions or training, they become acquired data and must be counted.

A completed artifact requires the requested budget or documented stopping condition, finite valid metrics, complete action/query logs, matching system hash, and an atomic completion record. A Slurm COMPLETED state or a summary file alone is insufficient. Failed and timed-out cells stay in the ledger.

### Objectives and uncertainty

Use final post-update non-root mechanism error at a fixed budget as the primary metric. Preserve the historical broad product-domain error as a stress metric. Add error under a prespecified distribution of feasible held-out interventions, since independent parent coordinates can be poorly covered by the allowed actions. Report observational error separately. New heterogeneous domains need prespecified, training-derived or domain-specified scale normalization as well as raw error; a quadratic cascade must not dominate solely through units or exploding magnitudes.

For probabilistic models also report predictive log score and interval coverage. For transfer report adaptation curves, changed-node error, unchanged-node degradation, and detection delay. For every policy report samples, actuators, wall time, host memory, GPU memory, and API tokens/cost.

Pilot systems are development data. Freeze the primary comparison, budget, metric, and method settings before a fresh confirmation set. Proposed confirmation: at least 20 independent SCM instances per principal setting, with additional learner seeds nested within system; determine the final count from pilot variance and a prespecified power calculation. Bootstrap paired effects by SCM, not by step or reset episode. Three main scientific claims require multiplicity control (e.g. Holm); report intervals and effect sizes regardless of significance.

Default promotion target: at least 20% lower primary error than the strongest relevant control, with a paired confidence interval excluding no improvement on fresh systems. A “does not hurt” claim requires a prespecified 5% noninferiority margin and an interval supporting it. Inconclusive is an allowed outcome. Pilot thresholds guide resource allocation; they are not evidence of confirmed superiority.

## 3. Control track: resolve PEV and locate the bottleneck

**Question:** Is there acquisition headroom after matching the full learner, and can ensemble uncertainty find it?

Start with 12 short CPU canaries: the legacy 5-node system, homogeneous 30-node, heterogeneous 30-node; four policies each (Random ensemble, non-leaf round-robin ensemble, PEV, PEV-var). Run ten campaigns for historical reproduction and a separately labeled persistent campaign. First time one campaign; choose the remaining length to fit the cap. No LM is needed.

Then a development sweep of 72 cells: three settings × six policies × four system/learner seeds. The six policies are Random over all nodes, Random over non-leaves, non-leaf round-robin with balanced values, a graph/coverage heuristic, PEV, and PEV-var, all with the same ensemble. The fixed 5-node example supplies learner-seed replication only; the two 30-node families supply independent system instances. Add single-student controls to the final architecture study, rather than confusing ensemble benefits with acquisition in this first screen.

Run four inexpensive diagnoses before expanding:

1. Replay exactly the same datasets through the two former training paths. Quantify the remaining learner effect.
2. On small linear-Gaussian systems, compare the ensemble score with exact posterior variance reduction. This validates acquisition math without claiming a deep ensemble is an exact posterior.
3. Compare approximate acquisition with an explicitly privileged, ground-truth diagnostic selector. A large gap motivates inference/selection work. Little gap motivates a different task or objective. Privileged selectors are not deployable baselines.
4. Plot error against parent-context coverage, noise, indegree, and batch size. Compare ensemble-mean simulation with member-wise noisy simulation, and IVR with batch-aware covariance updates on small cases.

**Decision:** Replicate PEV if gains survive non-leaf/coverage controls; retain PEV-var if it is simpler and as good. If neither wins, the new research tracks can still proceed using Random/round-robin or a genuine small-system Bayesian controller. The portfolio does not depend on rescuing PEV.

## 4. Direction A: useful scientific knowledge that can be wrong

**Question:** Can foundation-model knowledge reduce the data needed to learn mechanisms while recovering when its scientific expectations are wrong?

An anonymous graph of randomly assigned functions offers little semantic information to exploit. Conversely, a benchmark that gives away the equation through its description proves little. Construct independently authored simulator families with realistic but incomplete metadata: units, qualitative roles, admissible ranges, and actuator descriptions. Candidate domains are saturation in reaction/transport systems, threshold responses, and thermal or electrical components. Randomize parameters and compositions and hold out entire mechanism families. Metadata must not contain the answer or be generated by revealing the hidden formula to the tested model.

### Minimal architecture

The closed model outputs a bounded, typed library of candidate local expressions, constraints, and uncertainty statements. Fit numerical coefficients using acquired data. Maintain a predictive mixture containing both proposed expressions and a flexible data-driven fallback. For a metadata-only proposal, a conceptual prior is

`p(f | text) = (1 - rho) p_LM(f | text) + rho p_base(f)`,

with nonzero fallback support and reliability selected on development tasks. This is an architecture prescription, not an assertion that LM outputs supply calibrated probabilities. Start with finite model weights from held-out/prequential predictive scores; call them predictive weights unless Bayesian evidence is actually computed. If a proposal is revised using observations, do not reuse those observations as if the revised proposal had been an independent prior; evaluate revision on future data or use a properly accounted proposal/inference procedure.

For initial closed-model usage: one proposal call per system, a capped number of local forms, and at most one revision call after a failed predictive check. Cache output and compile a restricted expression language. No LM inference per candidate or per data point. Candidate scoring uses fitted models. Data-dependent revision is an optional ablation, not necessary for the first result.

### Decisive experiments

- **A1, prior quality on fixed data:** 12 development systems across three domain families. Compare an MLP/ensemble, broad hand-authored grammar, standard symbolic search, closed-model proposals from anonymous data summaries, and proposals with metadata. All numerical fits see identical data and fitting budgets.
- **A2, semantic intervention:** independently vary correct metadata, anonymous identifiers, plausible incorrect metadata, and wrong units that a validator should reject. Measure recovery as evidence accumulates. A strong prior may initially hurt; the question is whether explicit revision or mixture support controls that harm.
- **A3, acquisition factorial:** candidate source (LM versus numerical grammar) × controller (Random versus the same uncertainty/design algorithm). This distinguishes better hypotheses from better action selection. Add a direct closed-model action selector as a small diagnostic, with the same information and API budget.
- **A4, transfer:** freeze on two families; test unseen compositions and a held-out family. Include semantics-compatible transformations and label permutations. Simple renaming does not remove memorized mathematical laws, so use fresh compositions and incorrect-prior controls as well.

Pilot settings: N=5 and 15, 200 initial observations, then 10 batches of 20 acquired samples (400 total); optionally continue to 20 batches (600 total) as a separately frozen endpoint. Hold the action menu fixed for this track. These are proposed budgets, not historical ACE settings.

**Promotion:** fresh-system advantage over the strongest numerical grammar/symbolic/ensemble control, and demonstrable recovery under incorrect descriptions. If a hand-authored library matches it, retain the useful numerical method but abandon the claim that a foundation model supplies unique value.

**Novelty constraint:** LM-proposed equations already appear in [LLM-SR](https://arxiv.org/abs/2404.18400); LLM model proposal plus Bayesian design is central to [Model Discovery Agent, v4](https://arxiv.org/abs/2608.09696v4). Our candidate contribution is measurable reliability and recovery of local semantic priors under controlled mismatch and transfer. That still needs a deeper related-work comparison before any novelty claim.

## 5. Direction B: expose the mechanism before estimating it

**Question:** Are we failing because the learner is weak, or because allowed interventions do not expose the input combinations required by the evaluation objective?

Consider a mechanism with an interaction `y = a*x1 + b*x2 + c*x1*x2`. In a deterministic example where unmanipulated parents remain zero, all single-parent probes fail to distinguish different c values; joint perturbations expose c. With Gaussian parents there is generally nonzero support, so the claim becomes poor finite-budget coverage/information, not absolute nonidentifiability. Sweep background variance to connect the exact toy obstruction with realistic weak excitation.

### Architecture and principle

Use the known graph to construct candidate *sets* of actuators, initially size one or two. A small factor-graph controller scores how each set changes input coverage or predictive uncertainty at nearby mechanisms. Start with exact Gaussian/linearized experimental design; then test ensemble approximations. A representative cost-aware objective is

`expected reduction in prediction risk under the task's reference distribution / experiment cost`.

Estimate this using a student posterior or fitted hypothesis mixture, not environment labels. Batch scoring must account for redundant observations. Restrict candidates using shared children, local motifs, and bounded indegree; retain random candidate sets to measure screening failures. Bound shortlist size and measure its missed-opportunity cost. Sparse structure can reduce work, but unrestricted pair enumeration and descendant propagation can remain quadratic.

The foundation model's role is an **action compiler**: translate an instrument description into permissible actuator combinations, constraints, and a few experiment templates. Give a numerical baseline the same compiled menu to isolate interpretation from numerical design. If the menu is already formal and simple, expect no unique LM benefit. The harder semantic task is understanding heterogeneous actuators, costs, and physical constraints, rather than choosing among anonymous node numbers.

### Decisive experiments

- **B1, analytic sanity check:** derive and simulate linear-Gaussian cases and the interaction example. Show when one-target designs can or cannot identify the relevant parameter under the specified support assumptions. This is a proposed theorem/diagnostic, not an existing result.
- **B2, matched action sets:** compare Random pairs, coverage pairs, uncertainty pairs, and single-target methods. Within-menu comparisons identify algorithm effects; single-versus-pair comparisons identify capability effects. Charge `cost = samples * (1 + lambda * number_of_actuators)` and report lambda in {0,1,4}, alongside raw samples and actuator use.
- **B3, scaling:** embed a fixed number of difficult motifs in N=15, 30, 100, 300 graphs, then increase difficult-motif count at fixed N. This separates total size from the amount of information that must actually be acquired. Match indegree and report density.
- **B4, language contribution:** correct formal menu; closed-model compiled menu; corrupted descriptions; direct closed-model controller. Verify invalid action rate and model-error consequence using an independent menu validator.

Development screen: 12 independent small systems, four numerical policies, fixed budgets of 100/400 acquired samples, no closed-model calls until B1/B2 show a meaningful bottleneck. Then at most two calls per system for menu compilation and correction. N=100/300 are gated scale tests, not first-wave full neural training jobs.

**Promotion:** a cost-adjusted advantage within the same action space, surviving joint-random and coverage controls. If only the richer action space helps, report that finding accurately. If language merely translates a menu, frame it as an interface contribution; do not call it superior causal reasoning.

## 6. Direction C: learn once, then repair locally

**Question:** Does mechanism modularity pay off when tasks share components and only a few components change?

The previous experiments reset students and largely studied isolated systems. That leaves the main foundation-model promise—reuse of knowledge across tasks—mostly untested. This track makes transfer and compositional generalization the primary outcomes.

### Proposed neural architecture

1. A small shared encoder ingests each node's acquired parent/output data, intervention masks, units, and local graph roles. Use a permutation-invariant dataset encoder; parent permutation equivariance must preserve each parent's identity/role, not collapse all parents into an undifferentiated sum.
2. A reusable expert library supplies local function families/features. A hypernetwork or gated adapter maps the node's context to a low-rank local update. Begin with numerical-only modules; a closed model can propose typed descriptors or a small operator vocabulary offline.
3. A Bayesian linear output layer over frozen features, or an independently evaluated ensemble, provides uncertainty. Exact linear-layer uncertainty is conditional on its features and is not full uncertainty over the mechanism representation.
4. Conditional predictive checks flag changed mechanisms using observed parents. Adapt local modules while protecting unchanged modules. Distinguish a changed mechanism from an upstream distribution shift; marginal output drift alone is not evidence of a mechanism change.
5. An experiment controller chooses actions near uncertain or suspected changed modules, accounting for downstream effects and actuator constraints.

This gives the foundation model a slow role—organizing a reusable mechanism vocabulary—and the small numerical model a fast role—fitting, updating, and acting. Shared weights are trained offline; task-specific adapters remain separate to avoid corrupting other systems.

### Experiments and controls

- **C1, fixed-data transfer first:** train on 5–15-node systems, test on unseen 30-node systems with known local families. Compare scratch local MLPs, warm-start local MLPs, a shared network without mechanism routing, numerical modular routing, and semantic modular routing. Equalize data, tuning opportunities, and report parameters/compute.
- **C2, sparse changes:** change k in {1,3,10} mechanisms, with separate coefficient-change and function-family-change cases. Test whether adaptation cost tracks k more strongly than N. Measure false alarms, detection delay, recovery samples, and degradation on untouched mechanisms.
- **C3, composition:** hold out graphs, module combinations, and entire mechanism families separately. Include related and unrelated source libraries to expose negative transfer. Compare a retrieval-only nearest-module baseline before crediting a learned hypernetwork.
- **C4, scale:** N=30/100/300 at bounded indegree and fixed k, then dense/long-range graphs as stress tests. One initial graph scan still costs O(N+E); dense descendants or correlated modules can destroy the expected locality advantage.

Build a 1–5M parameter prototype, not a large new foundation model. Initial synthetic corpus target: 5,000 small SCMs; profile generation/training throughput on a 500-system subset before committing the full corpus. Start with a bounded-degree additive/interaction family and a neural/spline family not expressible by the initial symbolic vocabulary. Use 12 held-out development systems for initial evaluation; confirmation systems remain untouched.

**Promotion:** at least 2× fewer adaptation samples to a frozen error target than the strongest warm-start/retrieval baseline, with no supported >5% deterioration on unchanged modules; confirm on fresh systems. If semantic descriptors add nothing over numerical modules, the numerical architecture may still be useful, but the paper's foundation-model claim narrows.

**Prior art:** modular transfer follows the [independent-mechanism literature](https://proceedings.mlr.press/v80/parascandolo18a.html) and [meta-transfer objectives](https://arxiv.org/abs/1901.10912). [AVICI](https://arxiv.org/abs/2205.12934) concerns amortized graph inference, not this exact known-graph mechanism task. [Causal Foundation Models](https://arxiv.org/abs/2609.03003) reviews related causal effect estimators. Neither “modularity” nor “pretraining on SCMs” is a novelty claim; the proposed test is active local repair and compositional mechanism transfer with measured semantic contribution.

## 7. Small exploratory track: SCMs as tests of foundation-model belief revision

Reverse the direction of the project: use our controllable SCMs to test when a closed model's familiar scientific story overrules experimental evidence.

Generate paired worlds with the same observational behavior but different responses to an available intervention. Give a model a fixed action budget and require numerical predictions, explicit uncertainty, and a choice of discriminating experiment. Compare familiar labels, anonymous labels, and deliberately misleading labels. Include null pairs that the permitted action menu truly cannot distinguish; score abstention/uncertainty instead of demanding an impossible correct label.

Compare direct prompting, tool-assisted statistical inference, and the mechanism-library architecture. Exact simulators grade predictions; another LM does not grade the scientific conclusion. Counterfactual claims require a specified shared-exogenous-noise model; agreement on interventional distributions alone does not identify individual counterfactuals.

Pilot: 20 paired tasks × two closed models × at most four calls = 160 calls. This is a small diagnostic and possible benchmark artifact. Continue only if it isolates a reproducible failure or an architectural improvement beyond existing scientific-agent benchmarks. [BoxingGym](https://arxiv.org/abs/2501.01540) already tests experiment design and model discovery; merely asking LMs to run experiments would duplicate that idea.

## 8. Literature constraints and external validation

This is a targeted literature check, not an exhaustive novelty review. Checked primary sources on 25 September 2026:

- [LLM-SR](https://arxiv.org/abs/2404.18400): equation proposal informed by language-model knowledge. Required reference for A.
- [Model Discovery Agent, v4](https://arxiv.org/abs/2608.09696v4): model proposal, Bayesian inference, and value-of-information design. Closest broad architectural overlap. Its version history reports a prompt-leak correction; use the corrected version and assess its released implementation before selecting a comparison.
- [LeGIT](https://arxiv.org/abs/2503.01139): language-guided intervention targeting for graph discovery. Adjacent task; do not compare graph accuracy directly with mechanism MSE.
- [Active Bayesian Causal Inference](https://proceedings.neurips.cc/paper_files/paper/2022/hash/675e371eeeea99551ce47797ed6ed33e-Abstract-Conference.html): a genuine Bayesian active causal framework. Use a tractable special case as a principled reference.
- [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html): amortization of sequential design. A learned fast acquisition network alone is not a new concept.
- [When and Why LLM Causal Priors Help](https://arxiv.org/abs/2609.06941): recent prior-selection work for amortized causal effect inference. A's wrong-prior recovery experiments must be distinguished from prior selection on validation domains.
- [LLM-SRBench](https://arxiv.org/abs/2504.10415) and [science-grounded causal benchmarks](https://arxiv.org/abs/2510.16530): relevant controls for memorization and semantic leakage.

After a synthetic gate passes, select one independent external environment rather than inventing more favorable synthetic families. Candidates: an appropriate [BoxingGym environment](https://github.com/kanishkg/boxing-gym), or [NeuronBench](https://github.com/murphyk/neuronbench) for intervention forecasting. These are simulated scientific domains, not evidence from a real laboratory. NeuronBench adds dynamics and partial observation, so adaptation is a separate milestone. LLM-SRBench is a useful fixed-data equation-recovery test; it does not automatically supply an intervention environment.

Use repository/version-pinned tasks, independently chosen before scoring our method. A failed external transfer is a result, not a reason to silently substitute a better-looking task.

## 9. Live CURC feasibility and resource plan

Read-only live check on 25 September 2026, around 18:15 MDT:

- Shared SSH master works; remote host `login-ci5`.
- Remote `/projects/paco0228/ACE` is at `bcd017d`, behind local `f01a158`. Remote untracked logs/backups exist; preserve them. Future runs should use an explicit immutable revision and isolated output directory.
- No PEV result directory was found under the expected ACE scratch root, and no PEV/ACE jobs matching the inspected experiment prefixes were in the current queue. This is scoped evidence, not a claim that every possible custom job name was checked.
- Accounting confirms the 128G node-importance OOMs and ranking canary OOM, and the 48-hour no-DPO timeout. No large GPU resubmission is justified by the existing cleanup change.
- `acpu` has 420 configured nodes and a **24-hour partition limit**. `cpu-long` permits seven days at the QoS level, but that does not establish permission to exceed this partition's limit. Use checkpointed jobs under 24 hours until CURC clarifies the combination.
- A100, H200, L40, and RTX Pro 6000 resources are configured; both full GPUs and some partitions with GPU slices are present. GPU normal QoS is 24 hours, GPU long is seven days. Availability, fair-share, and start times are not guaranteed by this inventory.
- The user's associations include `ucb736_asc1`. Future ACE submissions should explicitly select that authorized account, resource type, and QoS. Other projects' jobs are outside this plan.

The existing default PEV script expands to **200 jobs**, not approximately 180: 60 + 20 + 30 + 30 + 30 + 30. Its requests sum to 2,610 job-hours at four CPUs per job, or **10,440 requested CPU-core-hours**. These are requested upper bounds, not measured runtime. This makes staged screening preferable to launching the full matrix before validating its scientific and operational assumptions.

### Proposed first-wave caps, not spending already authorized or incurred

- Up to **4,000 CPU-core-hours** across the shared controls, numerical fits, reachability experiments, and simulation. Initial 12 canaries capped at four CPUs × four hours each = 192 core-hours. The 72-cell development ladder at four CPUs × eight hours is a further upper bound of 2,304 core-hours; reduce campaign lengths if measured cost demands it.
- Up to **96 GPU-hours** for the small modular prototype, initially one GPU at a time. Use six checkpointable jobs capped at 16 hours if throughput supports it; stop early if the pilot shows no learning. This is an experiment ceiling, not a promise the full model will fit it.
- Up to **1,000 closed-model calls**, with an additional proposed hard spend ceiling of **$300**, whichever is reached first. Choose two accessible model snapshots at execution time. Provider access and current per-token prices have not been checked in this planning pass; compute a token-based estimate before launching. No paid calls were made here.
- A small optional ACE memory diagnostic gets at most eight GPU-hours within the same GPU ceiling, reducing prototype time accordingly. Instrument host RSS, CUDA allocated/reserved memory, replay/object counts, and candidate-clone lifetimes. Run LM-free and no-lookahead controls. Do not let closing historical ablations consume the new research budget.

Cache semantic libraries on the local/API side and send versioned JSON artifacts to CURC. CURC handles numerical training/evaluation; it need not keep a paid API call inside an expensive GPU allocation. Parallelize independent small systems as capped Slurm arrays; checkpoint between campaigns and within long persistent campaigns. Record actual elapsed core/GPU hours, not only requests.

## 10. Execution order and deliverables

### First two working days, subject to queue access

The shared-harness owner freezes the protocol, serializes system definitions, isolates the test evaluator, and validates exact sample accounting. The numerical owner runs small analytic B1 cases and prepares the 12 CPU canaries. The semantic owner prepares independent domain descriptions and the restricted proposal schema for A1. The architecture owner implements a numerical modular baseline and C1 fixed-data experiment. These are parallel work packages; no owner or agent has been assigned yet.

Deliverables: `protocol_v1.json`, completed canary ledger, simulator/replay parity report, 12 development systems for A/C, and a measured cost forecast. These filenames describe proposed deliverables, not files created in this planning pass.

### Following three to five working days

Run the screened 72-cell control study, A1/A2, B1/B2, and C1 concurrently within the caps. Run the 160-call belief-revision screen only after the API recorder and grading harness work. Each lane produces an effect-size plot, failure-case examples, cost/memory profile, and explicit continue/stop recommendation. Do not wait for every historical GPU ablation before learning from these lanes.

### Weeks two to four

Choose at most two winners. Freeze their settings and fresh confirmation sets; run graph/size/semantic shifts and one external task. Only then consider combining A's priors, B's action design, and C's reusable modules. A combined architecture is harder to diagnose, so component evidence comes first.

### Stop rules and useful negative results

- A fails against a broad numerical grammar: stop model-specific proposal tuning; retain a clear test of the limits of semantic priors.
- B gains vanish against cost-matched random pairs: stop the acquisition claim; keep an identifiability/actuation analysis if it is substantive.
- C gains vanish against warm-start or nearest-module retrieval: stop the hypernetwork build; a simple reuse baseline may be the right system.
- PEV loses after leaf/coverage controls: report the result and use the strongest simpler controller.
- All fail: retain a reproducible evaluation package and explain the demonstrated boundaries without generalizing to all foundation models or all SCM learning.

## 11. What happens to the paper

Preserve the legacy writing and all experimental outcomes. Treat the current AISTATS draft as requiring a substantive rewrite, not as submission-ready. Finish the corrected comparison/audit as a self-contained technical report; distinguish measured results, code-level concerns, and hypotheses. Do not replace failed claims with favorable assumptions about unfinished jobs or simulated reviewer acceptances.

PEV is a candidate result within that report until independent matched replication supports more. A new direction should earn its own paper around one verified contribution. Do not force all three into an ACE salvage narrative or let the old deadline determine the scientific claim. Venue dates and rules need a fresh official check when there is a stable result to submit; this plan makes no deadline assurance.

## Provenance of the planning pass

Repository code and documents were inspected at `f01a158`; only the pre-existing `.DS_Store` change was present initially. Live cluster checks were read-only. The archive search independently found the September 17 predecessors:

- `ACE/docs/development/guidance/pev_design.md`, object `sha256:19a5c79b547ed622f632ef61bc5113808167f18bde105bffafdb32b2893b7d85`.
- `ACE/docs/development/guidance/paper_reframe_2026-09-17.md`, object `sha256:e8b07704cb0f127c7b01dc7d621afa26f8ec7a8b9dab72b6078ff698f46d4c82`.
- `ACE/jobs/curc_submit_pev_ladder.sh`, object `sha256:a1a23972aa8c31410c2f6d4e281625de71b1ad91609a618be8f4b789f990a855`.

The current local files, code, and handoff are the working source of truth. No new scientific results are claimed by this document.
