# SCM–foundation model viability agenda

Date: 29 September 2026. This is a research agenda with initial gates, not a claim that an FM advantage has been established. No Azure or other closed-model calls are authorized for this agenda.

## Common contract

Treat a foundation model as an infrequent proposer of a typed, falsifiable object: a mechanism family, module mapping, actuator schema, or hypothesis set. Keep inference, action costs, and grading in numerical code. Use the existing known-graph, fully observed task for comparable mechanism experiments; mark hidden-confounder and partial-identification tasks as different settings. Freeze systems, actions, metrics, and controls before fresh confirmation. Report improvement over the strongest matched *simple* control, not only over an old ACE result. Count every environment sample, intervention, and model call; record seed, source revision, protocol and artifact hashes. Confirm on independent systems and at least one externally selected environment before a broad generalization claim.

The prior portfolio and execution ledger document the failed original ACE comparisons and the results below. The corrected metric and exact query accounting are prerequisites. A positive numerical acquisition result alone is not evidence of a foundation-model contribution.

## Track 1: semantic priors that can be wrong

**Question.** Can an open model supply a compact mechanism-family proposal that improves small-data prediction or intervention choice, while a numerical gate prevents damage when the label is misleading?

**Current evidence.** The n16 wrong-proposal screen has MSE .02564, gated .02128, broad .01750: recovery is incomplete. On BoxingGym Lotka, the open-model proposal and a data-only family selector chose the same coupled family with heldout MAE .348. Thus syntax and plausibility have been demonstrated; semantic value has not.

**Next gate.** Freeze 24 independent systems in two physical-story families plus anonymous and misleading descriptions. Use the same observed data and learner for typed open-model prior, data-only selector, broad mixture, and oracle-family upper bound. Prespecify a holdout domain and wrong-prior rate. Require lower mean heldout error than the best nonsemantic arm with paired interval excluding zero, and wrong-label performance within 5% of the broad arm. Stop semantic expansion if the data-only selector ties or wrong priors remain harmful. Begin locally with a small proposal cache; use CURC only for the larger numerical matrix. Do not fit a confidence-to-posterior mapping on confirmation systems.

## Track 2: interventions that expose hidden interactions

**Question.** Does posterior-aware pair selection improve mechanism learning after exact action costs, natural-label masks, and a strong fixed schedule are matched?

**Current evidence.** Fresh binary-tree systems support the pair selector over fixed factorial (.02694 versus .03972 MSE). Fresh fanout systems do not establish a gain over factorial hub (.01327 versus .01535; paired interval includes zero). The graph-family effect is material; a generic active-design claim is premature.

**Next gate.** Compare pairs against fixed factorial, randomized pairs, and single actuators under the same *total cost*, plus an adaptive risk selector that knows only available data. Freeze a mixed topology distribution before running; report each topology separately and pooled with topology as a factor. Promote only if the pair policy beats the strongest same-menu fixed policy on two topologies and improves a heldout feasible-action loss, without a hidden-simulator-loss input. This is primarily a numerical SCM track; any FM role must be a separately tested action-menu proposal.

## Track 3: modular memory and sparse repair

**Question.** Can a library reuse unchanged local mechanisms while identifying and repairing the few changed ones under distribution shift?

**Current evidence.** On an easy one-change task, the changed node was almost always found, and numerical mixture often tied adaptive scratch. In connected three-change systems, a four-response assay found 23/36 changed motifs in its top four; unchanged descendants were false positives after upstream shift. In the six-system pair pilot, risk acquisition beat static allocation, yet source-warm was worse than scratch for changed motifs at 16 source trajectories. A new out-of-bank nonlinear target form is validated as a simulator gate, not an efficacy result.

**Next gate.** Freeze a connected-system development set with three changes, upstream covariate shift, and one out-of-bank form. Compare source warm start, numerical retrieval mixture, scratch, and local repair on identical acquired trajectories; separately compare acquisition policies. Primary outcomes: changed-motif feasible error, unchanged-motif degradation, and trajectories to a frozen target. Require at least 2× fewer target trajectories than the best scratch/retrieval control with no supported >5% unchanged-module degradation before a GPU architecture scale-up. Descriptor-guided retrieval is a separate semantic ablation. Use small CPU runs first; reserve a capped CURC GPU prototype only after the numerical gate passes.

## Track 4: action-language compiler

**Question.** Can an open model translate natural intervention constraints into a *validated* actuator menu that increases useful reachable experiments?

**Current evidence.** The numerical joint-action code and cost/mask ledger work, but no language-to-menu advantage has been measured. This is the largest untested systems idea. A model suggesting illegal or unrepeatable actions is not useful.

**First fixture.** Create paired natural-language and machine-readable descriptions for a small set of actuator regimes: joint target prohibition, cost cap, required safety exclusion, and an indirect proxy actuator. A deterministic validator rejects actions that violate the formal schema. Compare an open-model compiler with a template/rule parser and a supplied-menu oracle; grade valid action recall, invalid proposal rate, reachable information gain, and downstream heldout mechanism loss at matched cost. The language must contain the needed constraints; the answer must not be leaked through exemplars. A tiny hand-written suite is only a parser smoke test. Advance to fresh paraphrases and external environments only if the model improves reachable useful actions while maintaining zero executed invalid actions. No remote job is justified until the validator and fixtures exist.

## Track 5: partial identification and belief revision

**Question.** Can an open model distinguish what observations identify from what requires an intervention, and update a familiar prior locally after contradictory evidence?

**Current evidence.** No FM result. The exact finite gate in `scripts/research/partial_identification_gate.py` constructs two latent-variable SCMs with the same observed joint distribution but different do(X=1) outcomes. It also calculates a contradiction event that rules out one candidate. This establishes a gradable task, not model competence or a general identified set over all SCMs.

**Next gate.** Generate 20 independent paired-world tasks with renamed variables and balanced prior-favored mechanisms, plus indistinguishable-under-menu null tasks. Require the open model to return a typed candidate set, a numerical range for a prespecified interventional query, a legal discriminating action or abstention, and an updated prediction after evidence. Compare direct open-model answers, a tool-assisted exact-enumeration baseline, and a fixed rule baseline. Score coverage, calibration/log score, action value, false certainty, and repair locality. Freeze prompts and tasks before scoring. Promote only if open-model assistance improves a task metric over the exact-tool baseline or exposes a robust benchmark failure worth publishing; generic experiment-selection accuracy is already covered by scientific-agent benchmarks.

## Order and resource gates

First complete exact Track 5 and a Track 4 validator smoke test locally, while analyzing existing Tracks 1–3 artifacts. Then run one small, frozen independent development batch per track. Use CURC for CPU matrices that materially exceed local cost and for a modular GPU prototype only after its numerical gate. The user is willing to leave jobs queued; queue delay alone is not a reason to alter another project's jobs or submit duplicate ACE jobs. No Azure or other closed-source calls. Record proposed caps and actual usage in the execution ledger. Select an external environment *before* scoring; BoxingGym is one candidate, with repository revision and task pinned.

## Related work boundary

LLM-SR already proposes equations as programs. Model Discovery Agent already combines model proposals, Bayesian inference, and value-of-information design. BoxingGym already evaluates experiment design and model discovery. Independent causal mechanisms are established prior art. The distinctive tests here are robustly *wrong* semantic priors, validated action compilation under constraints, local repair under connected covariate shift, and calibrated refusal where interventional effects are not identified. These are hypotheses for comparative study, not novelty claims before a focused literature review.
