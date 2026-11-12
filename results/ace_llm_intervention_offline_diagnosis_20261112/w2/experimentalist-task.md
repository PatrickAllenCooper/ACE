# Experimentalist task agenda (applied-scientist lens)

Prepared 12 November 2026 for the ACE agenda panel. Reading and design only. Nothing under `/Users/pat/code/ACE` was changed. No model, simulator, SLURM job, SSH session, install or network call was run. Arithmetic in section 9 was done with a scratch calculator script, `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w2/calc/arith.py`, and reproduces the brief's own chamber estimate (4 policies x 30 episodes x 16 interventions = 19,200 s, 5.33 h, 269 EUR) as a check.

Sources: the state-of-evidence brief (`../w2/brief.md`) and the notes `../w1/literature-benchmarks.md`, `mechanism.md`, `numerics-protocol.md`, `prior-evidence-portfolio.md`, `literature-policy.md`, `training.md`, `ledger-ops.md`, plus the handoff `/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md` (lines 60 to 68). Statements marked **(assumption)** are not verified in any file I read and must be checked at integration.

## 0. Summary

- **Task.** An operating-point decision on the Causal Chambers wind tunnel: choose intake and exhaust fan loads that keep the downwind chamber pressure and the exhaust fan speed above stated specifications at the least total fan current, using chamber time billed per second. The same task runs on an independently authored simulator family (many independent systems) and on the real Mk2 tunnel through the Remote Lab (frozen policies).
- **Ladder.** Procedural rigs with semantic descriptions (training, 4,096 systems, test apparatus classes held out) → fresh and held-out procedural systems (gates) → wind-tunnel simulator family (128 fresh systems, confirmatory) → pyCycle compressor rig (64 fresh systems, zero-shot) → real wind tunnel (one apparatus, pool plus online episodes) → optional second unit.
- **Gates that respect the standing rules.** Every family gets a numerical-only headroom audit before any language model is run on it, so no family is kept or dropped because of LLM performance. The LLM ladder has three kill points (interface learnability, in-distribution state dependence, transfer). A negative result narrows the claim and the hardware stage becomes optional and numerical-only.
- **Why the LLM could legitimately win here, and the control for each reason.** Variable semantics (anonymized and misleading text, a numerical student with the same text embeddings), model-form misspecification (engineer-encoded physics planner), switching costs (non-myopic planner at matched compute), measurement use (shuffled feedback), argmax-of-prompt (a rule on the prompt's own numbers).
- **Spend.** 0 EUR before the hardware freeze (1,000 free credits cover calibration). About 280 to 310 EUR for the hardware run at academic rates, cap 400 EUR including contingency, plus an optional 120 EUR second unit. GPU about 20 slice-hours for training the 0.6B arms. Everything else runs on local CPU.

## 1. What the lens requires, and why the November world failed it

The November world (3-node linear-Gaussian, known graph and noise, 8 menu actions, one constant prose and task payload) gave a learned policy no room: sequential adaptation was worth -0.7% +/- 1.7% and target risk already sat at the state-conditional optimum. Its first-crossing endpoint needed about 300 systems to resolve one cost unit. The literature predicts no LLM advantage in exactly that regime. So the experimentalist task has to satisfy four requirements before anything else:

1. **Audited headroom.** The room any policy could exploit (adaptivity, mechanism knowledge, switching costs, residual model information) is measured with numerical policies only, before an LLM touches the family.
2. **A declared reason the numerical planner could be handicapped, each with an equal-information control.** Semantics the planner cannot read, a misspecified generic model, and history-dependent costs.
3. **A decision-level endpoint with common random numbers.** The brief found decision-level regret needs 3 to 17 systems where first crossing needs 67 to 340.
4. **One-to-one mapping onto legal actuators of a real apparatus with billed costs,** so the simulated protocol and the hardware protocol are the same object.

## 2. Candidate environments

| Candidate | Real apparatus | Executable interventions | Legal actuators and safety | Measured cost | Useful decision | Semantic content | Independent authorship | Independent systems | Integration (notes) | Role |
|---|---|---|---|---|---|---|---|---|---|---|
| Causal Chambers wind tunnel (Mk1 simulators in `causalchamber`, Mk2 hardware via Remote Lab) | Yes | Yes on actuators `load_in`, `load_out`, `hatch`; `chamber.set` returns when hardware has changed | PWM 0.1 to 1 (fans never below about 0.1), hatch 0 to +/-45 degrees, about 8 s settle | Yes: 1 credit = 1 s = 0.014 EUR academic | Yes: least-current fan setting meeting pressure and flow specifications | Prose per variable, textbook fan physics, documented mechanism ladder | Yes (chamber authors, MIT simulators, graph validated by randomized experiments) | Simulation: unlimited parameter draws. Hardware: one apparatus (example IDs `wt-h0pe-i4ug`, `wt-demo-ch4lu`, `wt-ptdm-73iz`, physical distinctness unknown) | Stage A about 1 week, Stage B 1 to 2 weeks plus human sign-up | **Primary** (rungs W and hardware) |
| Causal Chambers light tunnel | Yes | Yes (LED RGB, polarizers) | 0 to 255, angles | Billed | Target irradiance at least LED power | Malus law in prose | Yes | One apparatus | `lt.Deterministic` | Could (E10b). Headroom doubtful: a two-coefficient law already explained `lt_malus_v1` |
| pyCycle compressor | No | Yes in simulation (speed and throttle, single-rig suitability unverified) | Surge and choke give a natural legality constraint | Synthetic rig time | Least shaft power at a pressure-ratio specification with surge margin | Strong (maps, surge, speed lines) | Yes (NASA, Apache-2.0, shipped maps) | Maps x scalings x inlet conditions (about 10 map files) | 1 to 2 weeks, unprototyped, install | **Should** (rung C, zero-shot) |
| ThermoPower via OpenModelica | No | Yes, dynamic | Same | Synthetic | Same | Strong | Yes | Maps must be supplied (default tables empty) | 2 to 3 weeks, Docker, FMU export, FMPy needs Python 3.10+ | Not this cycle |
| ACE-Runner scenarios | No | Yes, including internal nodes (not available on hardware) | Grids | Synthetic charged calls | Decision-style outcomes | Named variables with units | No (team-authored). Mission-1 root design is a structural null, `missile_defense` reads a variable outside its DAG, `tlam_satellite` near-degenerate | Few scenarios, seeds vary noise only | Cheapest | Could, as a labeled robustness row only, and only if Patrick confirms the scenarios may appear in a public paper |
| BoxingGym | No | Designs, not do-operations on a supplied graph | None | None ("ignores resource constraints") | Prediction, EIG | Prior toggle | Yes | 10 environments | Adapters partial, Python 3.11+, PyMC | Not used. Prior local LLM results negative |
| Gaussian Bayesian networks (bnlearn ECOLI70, MAGIC-NIAB) | No | Do-sampling on a supplied graph | None | None | None natural | Gene and trait names, memorization risk | Yes (fitted to data) | One network each, subgraphs not independent | 3 to 5 days, more than 8 nodes needs limit changes | Not used. Reproduces the known-graph linear-Gaussian regime with no headroom |

The wind tunnel is the only candidate with real do-operations, billed cost, a validated graph, prose and a documented misspecification ladder, and it is the nearest physical analogue of the compressor story in the talk. Its weakness is one apparatus, which the hardware design addresses with a recorded pool, explicit single-apparatus scope and an optional second unit.

## 3. The task: wind-tunnel operating point (WT-OP)

### 3.1 Apparatus and legal actuators

Mk2 actuators: `load_in` and `load_out` (PWM duty, legal 0.10 to 1.00), `hatch` (0 closed, fully open at +/-45 degrees). Sensors used: `rpm_in`, `rpm_out`, `current_in`, `current_out`, `pressure_downwind`, `pressure_ambient`. The tunnel is a dynamical system. The task models steady state after the documented 8 s wait. Ambient pressure is a documented common cause of all barometers, so the response is the difference `dP = pressure_downwind - pressure_ambient`. Sensor drift is handled by blocked, randomized ordering.

### 3.2 Causal graph and the explicit SCM given to every policy

Eight nodes, within the generic layer's 8-node limit:

- Intervenable roots: `L_in`, `L_out`. `H` (hatch) is fixed per task configuration.
- Measured mediators (not settable on hardware): `w_in` (`rpm_in`), `w_out` (`rpm_out`).
- Responses: `C_in`, `C_out` (currents), `dP`.

Edges: `L_in -> w_in -> C_in`, `L_out -> w_out -> C_out`, `(w_in, w_out, H) -> dP`, plus the documented air-path speed coupling (`L_in -> w_out`, `L_out -> w_in`, weakened by the hatch) and supply coupling between currents if the published ground-truth graph contains them. **(assumption)** The exact edge list and the sign structure of `dP` in each fan are transcribed from the chamber's published graph and the simulator source in E1 step 1. I could not read either locally.

Two SCM conditions, each a Bayesian model linear in parameters with known (calibrated) noise:

- **Generic SCM** (the shared learner in all primary comparisons): cubic polynomial in each parent plus pairwise products for multi-parent nodes, isotropic shrinkage prior. Physics-agnostic but flexible enough to converge.
- **Physics SCM** (engineer-encoded reference): affinity-law features, speed squared for pressure, speed cubed for current, dead zone below the minimum load, priors with documented signs.

### 3.3 Action menu and cost model

- Grid: `L_in, L_out` in {0.10, 0.325, 0.55, 0.775, 1.00}, 25 settings at the configuration's hatch angle.
- Commands per state: "move to setting s" for the 24 settings other than the current one, and "replicate" (measure again at the current setting). 25 commands, at most 32 allowed.
- Cost in units of chamber seconds: move = 8 s settle + 2 s measurement (20 samples at the documented 10 Hz) = 10 units. Replicate = 2 units. E2 measures settle time; if longer, the move cost becomes measured settle + 2 s for all arms before any LLM run. Costs depend on the current setting (a switching cost), which needs the state-dependent executor change the numerics note sizes as medium.
- Start: one shared measurement at (0.55, 0.55), outside the budget. Budget B = 120 units, so between 12 moves and 60 replicates. History stays under the 80-request limit.
- No stop command in the base protocol (fixed budget). A public stop rule is applied to every trajectory post hoc as a secondary endpoint.

### 3.4 Objective, shared decision rule and loss

Objective kinds vary across systems so a state-blind policy is penalized:

- K1: minimize `C_in + C_out` subject to `dP >= D*` and `w_out >= r*` (required through-flow).
- K2: maximize `dP` subject to `C_in + C_out <= I_cap`.

Thresholds are drawn per system at the 0.3 to 0.7 quantiles of the truth's feasible range, so the optimum lies inside the grid and moves between systems.

Decision rule, fixed and shared by all arms so that arms differ only in the data they collect: chance-constrained plug-in under the shared SCM posterior, constraints met with posterior probability at least 0.9, otherwise the setting with the highest probability of meeting them. The LLM chooses interventions, not the final recommendation. This isolates the research question and keeps the solid post-collection refit result separate.

Loss per decision, in [0, 1]: normalized objective gap `(C(x_hat) - C(x*)) / (max over feasible C - C(x*))` if `x_hat` is feasible under truth, and 1 if it violates a specification.

### 3.5 Endpoints

- **Primary (simulation):** normalized area under the truth-conditional expected loss curve over cost 0 to 120, averaged over R replicate episodes with common random numbers keyed by (system, setting, visit index, replicate). Analyzed as the paired log ratio between arms, systems as units. A lower curve means every reliability level is reached with less chamber time.
- **Secondary:** chamber seconds (and EUR) to expected loss 0.05, censored at 120; same-state counterfactual one-step regret on each arm's visited states (decision-level, zero environment cost in simulation); cost and loss at the public stop rule (stop when posterior probability of meeting all specifications is at least 0.95 and posterior expected normalized gap is at most 0.05); specification-violation rate at 120; recovery delay under misleading text; latency and FLOPs per decision.
- **Hardware:** realized loss against the pool-estimated truth, pool-replay area, measured specification compliance and current at the recommended setting, billed credits and EUR.

### 3.6 Semantic content and its controls

Three text conditions with identical numbers: named (Causal Chambers variable names and freshly paraphrased documentation prose, written to limit verbatim recall of the published docs), anonymized (V1 to V7, no units, no prose), misleading (the roles of the two fans swapped in names and prose). Equal-information numerical controls: the physics SCM planner (knowledge encoded numerically), a numerical student given a frozen-LLM embedding of the same text (the Gupta et al. control), and a prompt-feature rule (argmax-of-prompt).

### 3.7 LLM interface (fixes the November failure modes)

- Compact state text of at most about 1,500 tokens (November: 1,990 to 4,160): apparatus description, graph, objective sentence with numeric thresholds, costs, budget, current setting, a table of visited settings (count and mean responses, 3 significant digits), and per-candidate predictions from the shared SCM (predicted responses with SD and probability of meeting the specifications, 2 to 3 significant digits). No per-action planner scores.
- Commands carry single-token letter identifiers permuted per episode **(assumption: single capital letters are single tokens in the pinned Qwen tokenizer, checked offline in E4)**. One forward pass, softmax restricted to legal identifiers, argmax at deployment, sampling at temperature for RL rollouts. This removes greedy trie branching, length bias and position-identity confounding.

## 4. Second family: compressor rig (CR-OP, for E7)

- Model: pyCycle single-spool rig, inlet at ambient, compressor with a shipped map (`axi5`, `axi3_2`, `HPC_map`, `LPC_map`, `Fan_map`, `ncp01`), duct, variable-area throttle nozzle. **(assumption)** A throttled single rig is expressible. The note says suitability is unverified.
- Nodes (7): speed command `N`, throttle area `A`, corrected flow `W`, pressure ratio `PR`, efficiency `eta`, shaft power `P`, surge margin `SM`. Edges `(N, A) -> W`, `(N, A) -> PR`, `(W, N) -> eta`, `(W, PR, eta) -> P`, `(W, PR, N) -> SM`, to be confirmed against the element graph.
- Menu: 5 speed levels (70 to 100% design) x 5 throttle levels, plus replicate. Costs (notional rig seconds, stated as synthetic): speed change 30, throttle-only change 8, replicate 2. A setting with true `SM < 5%` or a non-converged solve is a surge event: the response carries a surge flag and a 30-unit trip penalty, predeclared as environment semantics.
- Objective: least `P` subject to `PR >= PR*` and `SM >= 12%`. Thresholds at truth quantiles.
- Systems: maps x design pressure ratio, flow and efficiency scaling x inlet temperature. Truth tables (25 solves per system) are precomputed and noise is added at episode time.

## 5. Training generator (G)

- Device library of about 16 classes across electrical, thermal, non-polarization optics, hydraulic-static (valves, orifices, tanks) and mechanical (springs, dampers, gear trains, DC motors). Each class has canonical mechanism forms (affine, dead-zone affine, power law with exponent 0.5 to 3, saturating, quadratic loss, bilinear), parameter ranges, units and three independently written descriptions, frozen before any model sees them.
- **Held out from G entirely, by physical category chosen now:** turbomachinery (fans, pumps, compressors, turbines), pressure sensing, polarization optics. So the wind tunnel, the compressor and the light tunnel are all unseen apparatus classes. Three further library classes are held out for G-heldout.
- Rigs: 2 actuators, 2 to 4 mediators, constraint and cost responses, at most 8 nodes, 5 x 5 grid plus replicate, cost pairs (10, 2), (6, 1), (20, 2), objective kinds K1 and K2.

## 6. Ladder, namespaces, and what is frozen when

| Namespace | Family | Systems | Replicates | Use |
|---|---|---:|---:|---|
| G-train | procedural | 4,096 | n/a | training (privileged labels, RL) |
| G-dev | procedural | 32 | 8 | E3 numerical audit, E4 artificial states |
| G-fresh | procedural | 64 | 4 | E5 in-distribution gate |
| G-heldout | procedural, held-out classes | 64 | 4 | E5 semantic generalization |
| W-design | wind-tunnel simulator | 256 | n/a | building the truth-free static design only |
| W-dev, W-dev2, W-dev3 | wind-tunnel simulator | 32 each | 8 | E1 audit and at most two lever rounds |
| W-train | wind-tunnel simulator | 2,048 | n/a | in-family training (should) |
| W-conf | wind-tunnel simulator | 128 (96 to 192 by rule) | 4 | E6 confirmatory, generated only after the final family definition |
| C-dev / C-conf | compressor | 24 / 64 | 8 / 4 | E7 audit / confirmatory |
| HW-calib | real tunnel | 1 apparatus, 20 off-grid settings | n/a | E2 |
| HW-pool, HW-online | real tunnel | 1 apparatus, 2 hatch angles, 6 task configurations | see E9 | E9 |

**Developed on simulators (iteration allowed):** all code, generators, learners and planners, the LLM interface, training recipes and hyperparameters (selected on G-dev and G-fresh only), in-family training on W-train, the digital twin and the hardware protocol (on the twin only).

**Frozen before E6:** the W-conf generator hash and namespace, arm roster, model and adapter hashes, prompt templates, permutation seeds, scoring rule, learner priors, decision rule, endpoints, failure treatment and analysis code, with independent review.

**Frozen before any hardware credit is spent (E8 manifest):** everything above plus hatch angles, specification levels, block schedule and arm-order seeds, pool protocol, latency budget (2 s per decision on the hardware host), noise prior from E2, session plan, spend cap, abort rules and the hardware analysis script.

**Never touched before its own scoring:** W-conf, C-conf, the hardware pool and online episodes.

**Failure treatment (predeclared):** illegal LLM commands cannot occur under restricted scoring. A deterministic simulator exception is a code defect: fix and rerun all arms on that system, both attempts preserved. pyCycle non-convergence is a surge event. On hardware, a dropped session or API error aborts the episode, which is preserved and rerun once in a later block; the primary analysis uses completed episodes and a sensitivity analysis scores aborts as loss 1; more than 10% aborts pauses the run for review. Latency overruns execute but are flagged and billed.

## 7. Experiments (ordered)

### E1. Wind-tunnel task and numerical headroom audit (must)

**Question.** Does WT-OP on the independently authored simulator family leave measurable room beyond the strongest numerical planner on the same SCM, through adaptivity, mechanism knowledge, switching costs or residual model information, and can the endpoint resolve a 10% difference with at most 192 systems?

**Design.** Family W: `wt.SimA1C3` (pressure, rpm) plus `wt.ModelB1` (currents), with per-system parameters drawn around the published values (omega_max, L_min 0.08 to 0.25, C_min, C_max, S_max, Q_max, r0 0.6 to 0.8, beta 0.10 to 0.20, barometer error and precision), a fan-coupling term absent from the Mk1 simulators but documented on Mk2 (kappa in [0, 0.2]), AR(1) ambient drift, hatch per task in {0, 15, 30, 45} degrees, objective K1 or K2. Arms, all on the generic SCM unless noted, R = 8 with common random numbers, on W-dev (32 systems):

| Arm | Description |
|---|---|
| S0 | state-blind decision, no measurements |
| S1 | space-filling static order (maximin), no replicates |
| S2 | best truth-free static design, greedy on W-design prior draws |
| S3 | uniform random |
| P1 | target risk, 256 draws, common random numbers across candidates |
| P2 | constrained knowledge gradient (cKG), one step |
| P3 | cKG with two-step rollout and switching costs |
| P4 | cKG with the physics SCM (planner and learner) |
| P5 | hybrid: plans with the physics SCM, decides with the generic SCM |
| P6 | oracle-form planner: the true simulator form with family-prior posterior by importance sampling |

Headroom ratios (areas, generic condition): H1 = S2 / best of P1 to P3 (adaptivity). H2' = P2 / P5 (value of mechanism knowledge through action choice alone, the most an LLM can add in the generic condition by acting on semantics). H3 = P2 / P3 (non-myopia under switching costs). H4 = best of P1 to P5 / P6 (residual information gap). Triviality: mean loss of S0 and frequency of the modal optimal setting. Variance components sigma_s (between systems) and sigma_r (within system per episode) of the paired log ratios. 2,560 episodes.

**Decision rule.** Pass if (a) H2' >= 1.25 or H4 >= 1.25 or H3 >= 1.15, with 95% CI lower bound above 1.05, (b) S0 mean loss >= 0.30 and no single setting optimal in more than 25% of systems, and (c) the formula in section 9 gives n <= 192 for a 10% effect at 80% power with Holm over three contrasts. H1 >= 1.15 is required before any claim mentions adaptivity, otherwise claims are limited to prior or semantic effects. On failure, apply levers in fixed order on fresh dev batches, at most two rounds: (L-a) no shared initial measurement, (L-b) budget 80, (L-c) hatch as a third decision variable on a 3 x 3 x 3 grid (27 settings, identifiers extend to lowercase letters). **Positive:** W becomes the confirmatory family and the paper states the measured ceiling on any policy's gain. **Null or negative after two rounds:** reported as "a real-apparatus operating-point task of this kind is solved by numerical SCM planning", no LLM arm runs on W, and the hardware stage can only be a numerical demonstration (an ask). No LLM output exists yet, so the choice cannot depend on LLM performance. The C1 comparator for E6 (best of P1 to P3) is fixed here.

**Prerequisites.** Install approval for `causalchamber` 0.2.8 (MIT, numpy-only wind-tunnel file) in an isolated environment, or vendoring the pinned MIT file. Reading the simulator source and published graph to fix edges and signs. New generic-layer versions: basis-function learner, state-dependent costs, shared measurement namespaces for common random numbers, truth-conditional evaluator, more than 8 arms and 64 worlds per roster, new sealer and cost-curve versions.

**Cost.** Code 8 to 10 working days (module changes sized small to medium in the numerics note). Compute: 2,560 episodes x about 20 decisions, one-step arms about 0.5 s and rollout about 7 s per decision, about 15 CPU-hours locally (local CPU is not charged to the CURC envelope), about 2 hours wall. Spend 0.

**Main risk.** Monotone, smooth physics lets cKG solve the task in a few moves, so every ratio falls below threshold. Or the Mk1 simulators expose different inputs or signs than assumed.

### E2. Apparatus calibration and interface smoke on the real tunnel (must)

**Question.** Does family W cover the real Mk2 tunnel's steady-state responses, settle times, noise and drift, and does the remote interface support the protocol at the assumed costs?

**Design.** Fixed, non-adaptive protocol, no policy and no endpoint: four load steps (each fan 0.1 to 1.0 and back) logged at 10 Hz for 15 s each (60 s); two anchor settings x 10 approaches x 10 s (200 s); 20 off-grid settings, loads {0.2, 0.45, 0.7, 0.95} squared at hatch 0 plus 4 at hatch 30 degrees (200 s); hatch checks, 3 angles x 2 settings (60 s); session and latency overhead about 200 s. Total about 720 s, inside the 1,000 free credits. Off-grid settings keep the 25-point task grid unseen until E9.

**Decision rule.** Pass if for each modeled response at least 18 of 20 calibration settings fall inside W's 95% prior predictive band, at least 90% of load steps settle within 2% of steady state in 8 s or less, and the noise SD lies inside the family range. On failure, before W-train or W-conf is generated and before any LLM training on W: widen the failing ranges to cover the calibration estimates with 50% margin, set the move cost to measured settle plus 2 s, and rerun the E1 audit on a fresh dev batch. If E2 cannot run before W-conf must be frozen, W-conf is frozen on the documentation-based family with ranges widened by 50% and E2 then only informs the twin. **Positive:** the paper can state that the simulator family covers the apparatus. **Negative:** a documented simulator-to-hardware gap, corrected before training and reported.

**Prerequisites.** Patrick's own sign-up (the terms forbid accounts registered by automated methods) and acceptance of Terms v1.1. A separate Python 3.11+ environment with the pre-release client `causalchamber` 1.0.0rc1 (the local Python is 3.9). A read of what the package's dataset module sends (the changelog mentions a call-home of unverified content).

**Cost.** 0 EUR. 1 to 2 days of code. Calendar dominated by account approval.

**Main risk.** Sign-up delay, pre-release client changes, unknown shared-pool waits.

### E3. Procedural training generator and numerical audit (must)

**Question.** Does a generator of semantically described rigs, with the test apparatus classes held out, have headroom for a learned policy, and do its descriptions carry exploitable information?

**Design.** Section 5 generator. Audit on G-dev with the E1 arms (R = 8, 2,560 episodes). Semantic-informativeness check: a numerical student (MLP) trained on 2,048 G-train systems with privileged-truth targets, with and without device-class one-hot inputs, evaluated on G-dev.

**Decision rule.** Pass if the E1 headroom criteria hold on G-dev and the class-informed student's area is at least 10% lower (CI lower bound of the ratio above 1.03). If the descriptions carry no exploitable information, increase form diversity across classes once and re-audit. **Positive:** the semantic channel is trainable and testable. **Negative:** only non-semantic LLM claims (adaptivity, costs) remain possible, declared before any LLM training.

**Prerequisites.** E1 code. Descriptions written and frozen before any model sees them.

**Cost.** 5 to 7 days of code and writing. About 10 CPU-hours locally. Spend 0.

**Main risk.** Leakage of test physics into the library (for example DC motor laws that mimic fan laws), or a library so unlike real apparatus that nothing transfers.

### E4. LLM interface and learnability qualification (must)

**Question.** With the section 3.7 interface, is the model free of position and identifier bias, does the decision-relevant content survive tokenization, and can a small fit learn state-dependent choices better than a state-blind counter (the November supervised model never beat one)?

**Design.** 1,200 artificial states from G-dev (never W or C). Models: Qwen3-0.6B (pinned), Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (both cached locally, 7B for frozen probes only). Q1 bias: choice frequency by position and by letter under random permutations (chi-square). Q2 survival: linear probes on the final hidden state recover the candidate with the highest probability of meeting the specifications, the remaining budget and the current setting (fit 800, test 400). Q3 learnability: LoRA rank 16 on all attention and MLP projections, at most 500 steps, batch 16, on 2,000 G-train states with privileged-truth soft labels, tested on 400 held-out states against the best state-blind baseline (label frequency by objective kind and stage). Measures per-decision latency and token counts.

**Decision rule.** A model passes if Q1 shows no position or letter above twice uniform (p > 0.01), Q2 top-1 recovery is at least 60%, and Q3 top-1 agreement with the truth-best action is at least twice and cross-entropy below the state-blind baseline. The smallest passing model becomes the primary policy model. **If no model passes Q3**, the LLM ladder stops and the paper reports learnability failure under this interface. The numerical ladder continues.

**Prerequisites.** Approval for local forward-only inference with a written protocol and for the Q3 fit (local, or within the remaining 2,742 slice-seconds of the CURC envelope). Approval to use the cached 1.5B and 7B weights. The separate offline diagnosis of the saved November artifacts informs the interface only, never tuning to the 24 outcomes.

**Cost.** 3 to 5 days of code. About 3,600 forward passes plus probes, 1 to 3 hours on the Mac (7B dominates). Q3 about 0.3 slice-hours or a few hours locally.

**Main risk.** Q3 fails at 0.6B, forcing the 1.5B model and its compute ask.

### E5. Policy training and in-distribution gates (must)

**Question.** Does the trained LLM policy learn a state-dependent acquisition policy that beats its frozen base and approaches the strongest numerical planner on fresh procedural systems, and does it generalize to held-out device classes?

**Design.** Arms with matched states and exposure:

- L2-x: supervised listwise distillation of privileged-truth one-step values. For each state and legal action, value = expected reduction in truth decision loss per cost unit over 64 simulated outcomes. Soft targets softmax(value / tau), tau one quarter of the per-state SD of values. About 30,000 states, 2 epochs, then 2 DAgger rounds relabeling student-visited states. Targets come from the truth, not from the planner, so the policy is not trained to narrate a numerical optimizer.
- L3-x: L2-x then on-policy GRPO, 100 updates, 8 systems x 8 trajectories per update, return = minus truth area, group-normalized advantages, KL to L2-x 0.02, entropy bonus 0.01, rollout temperature 1.5, LoRA rank 16 on all projections, learning rate 1e-5 (scale and settings follow Kriz et al.'s Qwen3-0.6B result).
- N8-x: numerical student (MLP on the prompt's numeric features plus device one-hot, which is a zero vector for unseen classes) and N9-x (N8-x plus frozen Qwen embedding of the description), same targets and states. N8 is the successor of the November numerical ranker.
- Should: L3-w and N8-w trained in-family on W-train, and L2-x, L3-x at 1.5B.

Gates on G-fresh (64 systems) and G-heldout (64), R = 4, arms L1 (frozen), L2-x, L3-x, N3 (target risk), N4 (cKG), N8-x, N9-x. 3,584 episodes, about 31,000 LLM decisions.

**Decision rule.** T1: the primary candidate beats L1 on G-fresh with paired area ratio at most 0.85 and 95% CI upper bound below 1. T2 (state dependence): top-1 agreement with the truth-best one-step action at least 35%, no action above 40% of choices, action entropy across states at least 2 bits. Both are required to proceed to E6 and E7. Otherwise the LLM ladder stops and the paper reports that the recipe did not learn a state-dependent policy even in distribution. T3 (informative): ratio to the best generic numerical planner on G-fresh. Named versus anonymized on G-heldout is reported as the first test of the semantic channel. The primary LLM arm for E6 is fixed here by rule: L3-x if it beats L2-x on G-fresh, else L2-x.

**Prerequisites.** E3 and E4 pass. Reviewed new engine versions (listwise soft targets, batching, DAgger loop, GRPO). The current engine admits only two named recipes, SFT before DPO and up to 16,384 steps. A new compute ceiling.

**Cost.** Code 1.5 to 2 weeks. GPU (a100_3g.20gb slice, 0.6B): SFT 60,000 examples at 0.1 to 0.25 s = 1.7 to 4.2 slice-hours. GRPO 128,000 sampled decisions at about 0.05 s generation plus 0.1 s update = about 5.3 slice-hours. In-family arm about 5 more. With a 30% failure margin (26.4% of GPU time failed last cycle) the ask is about 20 slice-hours. 1.5B about 2.5 times (about 30 slice-hours, optional). CPU for truth labels and simulators about 20 core-hours. Calendar 2 to 3 weeks including custody. Throughputs are estimates until E4 measures step times.

**Main risk.** Collapse recurs or GRPO games the return. New training code is the critical path.

### E6. Frozen transfer to fresh wind-tunnel simulator systems (must, primary confirmatory)

**Question.** On fresh systems of an independently authored simulator, does the frozen trained LLM policy reach a reliable operating point with less chamber time than the strongest numerical planner on the same SCM, and is any gain due to semantics, measurement use and language pretraining rather than amortized training or extra computation?

**Design.** W-conf, 128 systems (96 to 192 by the section 9 formula from E1 variance components), R = 4 with common random numbers.

| Arm | Role |
|---|---|
| N1 static space-filling, N2 random | floors |
| N3 target risk | kept control |
| N4 cKG | strongest myopic planner |
| N5 cKG two-step rollout at compute matched to the LLM's per-decision FLOPs (about 1.2e12 for 0.6B at 1k tokens) | extra-computation control |
| N6 cKG with physics SCM | engineer-encoded reference |
| N7 straddle rule on the prompt's own probabilities, cost-weighted | argmax-of-prompt control |
| N8-x numerical student, N9-x with text embeddings | kept ranker control, semantic-feature control |
| L1 frozen LLM, L2-x supervised LLM | kept controls |
| L3-x | primary (or L2-x by the E5 rule) |
| L4 anonymized, L5 misleading, L6 shuffled feedback, L7 graph withheld from prompt | semantic, recovery, measurement-use, causal-structure ablations |
| should: L3-w, N8-w, L3-x at 1.5B | in-family and capacity |

L6 shows history and predictions fitted to measurements permuted across visited settings, while the shared final decision uses the true data. Primary contrasts, Holm over three: **C1** L3-x versus the best generic planner fixed in E1, **C2** L3-x versus N9-x, **C3** L3-x versus L4. Before freezing, an LLM legality smoke runs on 2 W-dev systems with endpoint computation disabled.

**Decision rule.** **Full positive:** C1 ratio at most 0.90 and Holm-significant, C2 and C3 Holm-significant in the LLM's favor. License: "a trained SCM-conditioned LLM uses variable semantics to reach reliable operating points with at least 10% less apparatus time than the strongest numerical planner on the same SCM, on fresh systems of an independently authored simulator, and a numerical policy given the same inputs and text embeddings does not match it". The N6 contrast says whether it also matches hand-encoded physics. **Non-semantic positive** (C1 passes, C3 fails): the trained policy beats the planner, LLM-specific only if C2 passes, otherwise "amortized policies beat this planner". **Null** (C1 not significant, ratio CI inside 0.90 to 1.10): parity, no benefit claim. **Negative** (ratio at least 1.10 or significantly above 1): the policy loses on transfer, claim narrowed to E5. If L6 is within 5% of L3-x, the policy is reported as not using measurements and any gain is attributed to priors only. E6 outcomes select the hardware design through the E8 rule only.

**Prerequisites.** E1 pass, E5 T1 and T2, W-conf generated after E2 (or under its fallback), frozen manifest with independent review.

**Cost.** 8,192 episodes (plus 1,536 for should arms). LLM forward passes about 72,000 (92,000 with should arms) at about 0.4 s, 8 to 10 hours on the Mac or about 1 slice-hour batched. Numerical about 33 CPU-hours locally, rollout arm dominant. 2 to 3 days with review. Spend 0.

**Main risk.** Many arms invite selective reading, handled by the fixed hierarchy and Holm. Recall of the published chamber documentation can mimic semantic reasoning, handled by fresh paraphrases and L5.

### E7. Second independent family: pyCycle compressor, zero-shot (should)

**Question.** Does the E6 result hold on a second independently authored engineering simulator with different physics, safety constraints and costs, never seen in training?

**Design.** CR-OP (section 4). Numerical audit on C-dev (24 systems, R = 8, six numerical arms, 1,152 episodes) with the E1 rule. Then C-conf (64 systems, R = 4), arms N3, N4, N6 (map-shaped physics SCM), N8-x, N9-x, L1, the primary LLM arm, L4. 2,048 episodes. Contrasts C1 to C3 as in E6.

**Decision rule.** No headroom: reported, no LLM arm runs. Confirmatory rules as in E6. Sign agreement with E6 on C1 and C3 licenses "generalizes across two independently authored engineering simulators". Disagreement narrows the claim to the wind-tunnel family. A negative here bounds a positive E6, it does not erase it.

**Prerequisites.** Install approval for `om-pycycle` 4.2.0 and OpenMDAO (Python 3.9 compatibility unverified). A check that a throttled single-spool rig is expressible. E5 pass.

**Cost.** Code 1 to 2 weeks (unprototyped). Truth tables 88 systems x 25 solves = 2,200 cycle solves (assumption: under an hour). About 15,000 LLM forward passes (about 2 hours locally). Spend 0.

**Main risk.** pyCycle cannot represent the rig cleanly or solver failures cluster near surge.

### E8. Hardware protocol freeze by digital-twin dress rehearsal (must)

**Question.** Given E2's calibration and E6's effect sizes, which hardware design yields an interpretable result within the spend cap, and does every frozen component meet latency and session limits?

**Design.** Twin ensemble of 32 W-family members fitted to the E2 data. Simulate the exact E9 protocol: pool collection, online blocks, latencies measured on the hardware host, AR(1) drift fitted from the E2 anchors, session overheads. Candidate designs: D1 powered (pool plus 3 arms: primary LLM, C1 comparator, N6, x 24 blocks), D2 descriptive (pool plus 5 arms: primary LLM, L3-w or L1, C1 comparator, N6, N1, x 12 blocks), D3 parity (pool plus 2 arms x 12 blocks). Predicted power for C1 on replay and online, predicted credits and EUR.

**Decision rule (declared now).** E6 C1 ratio at most 0.85 and D1 replay power at least 0.8 within 400 EUR: run D1. E6 positive with ratio in (0.85, 0.90]: run D2. E6 null: D3 only with Patrick's approval. E6 negative: no LLM arm on hardware, and a numerical-only demonstration (C1 comparator versus N1) only with Patrick's approval. Any arm that cannot decide within 2 s on the hardware host is excluded with the reason recorded (the rollout planner probably is). Independent review of the manifest before any E9 credit.

**Prerequisites.** E2 data, E6 result, hardware host environment.

**Cost.** 2 to 3 days. About 5 CPU-hours. Spend 0.

**Main risk.** Twin misfit from only 20 calibration settings makes predicted power optimistic.

### E9. Frozen-policy run on the real wind tunnel (must)

**Question.** Do the frozen policies execute legal interventions on real hardware with billed costs, deliver validated operating points, and preserve the simulated ordering?

**Design.** One Mk2 tunnel. Pool: 25 grid settings x 2 hatch angles x 12 approaches, each 8 s settle plus 6 s measurement (enough for the initial measurement and two replicates), randomized and interleaved with online blocks over at least 3 days (8,400 s). It doubles as the truth estimate and allows replay of any policy using recorded samples only, never interpolated settings. Online: the E8 design, 6 task configurations (2 hatch angles x 3 specification levels at the twin's 30/50/70% feasible quantiles), arm order from a frozen Latin schedule, reset to (0.55, 0.55), budget 120, recommendation validated by 2 approaches, about 180 s per episode with latency. Policies run from the Mac. Endpoints in section 3.5, with 200 replay episodes per arm per configuration and bootstrap over approaches for truth uncertainty.

**Decision rule.** Calibration: each arm's replay area inside the twin's 90% predictive interval ("calibrated transfer"). Ordering: hardware C1 sign equals E6. Agreement: online mean inside the replay 90% band per arm, which validates replay. Under D1, a Holm-significant C1 on replay licenses "on this apparatus the frozen LLM policy reached validated operating points with X% less chamber time (Y EUR) than the strongest planner, at Z% less fan current than the default setting". **Null:** parity on hardware, the simulation claim stands unconfirmed physically. **Negative** (reversed sign, CI excluding 0): the advantage does not survive the real apparatus, the claim is narrowed to simulation, and twin-versus-pool residuals name the mechanism (coupling, dead zone, drift). All inference is conditional on one apparatus.

**Prerequisites.** E8 manifest and review, Patrick's spend approval, credits purchased, approved local policy service.

**Cost.** D2 19,900 s (5.5 h, 279 EUR). D1 22,060 s (6.1 h, 309 EUR). With 30% contingency 362 to 402 EUR, so a 400 EUR cap (about 28,600 credits). At the commercial rate the same run is about 760 to 840 EUR before contingency. Calendar 1 to 2 weeks.

**Main risk.** Shared-pool waits, dropped sessions, drift larger than the twin's, a single apparatus.

### E10. Second physical unit or second apparatus type (could)

**Question.** Does the hardware result replicate on a second physical apparatus?

**Design.** (a) Preferred: a second Mk2 unit that the operators confirm is physically distinct. Pool 25 settings x 1 hatch x 12 approaches (4,200 s) plus 2 arms (primary LLM, C1 comparator) x 6 blocks (2,160 s). (b) Alternative: a Light Tunnel Mk2 operating-point task (target irradiance at least LED power with polarizer and LED settings, polarization optics held out from G), which first needs its own E1-style audit on `lt.Deterministic` and its own frozen namespace.

**Decision rule.** Runs only if E9 is positive or null. Sign replication licenses wording for two apparatus. Failure is reported and the claim stays single-apparatus.

**Prerequisites.** E9. Operator confirmation (a message Patrick sends). Spend approval up to 120 EUR.

**Cost.** (a) 6,660 s, 93 EUR, 121 EUR with contingency. (b) About 1 week of code plus similar spend.

**Main risk.** Only one physical unit exists in the pool. The light-tunnel task lacks headroom.

## 8. Sequence, gates and calendar

1. **Week 0 (now).** Send the asks in section 10. The separate offline diagnosis of the saved November artifacts runs in parallel and informs E4 only.
2. **Weeks 1 to 2, in parallel.** E1 and E3 (numerical only, local CPU). E2 as soon as credentials arrive. **Gate G1:** E1 and E3 pass, E2 done or its fallback declared. Then generate and freeze W-conf, G-fresh, G-heldout and C-conf.
3. **Week 3.** E4. **Gate G2:** smallest passing model, or stop the LLM ladder.
4. **Weeks 3 to 5.** E5 on CURC. E7 code and C-dev audit in parallel. **Gate G3:** T1 and T2, or stop the LLM ladder.
5. **Weeks 5 to 6.** E6 and E7 confirmatory in parallel (local). **Gate G4:** E6 rule.
6. **Weeks 6 to 7.** E8 freeze and review. **Gate G5:** spend approval confirmed.
7. **Weeks 7 to 9.** E9 sessions over at least 3 days. Then E10 if justified.

About 9 to 10 weeks from approvals puts the full ladder in late January 2027 at the earliest. ICML 2027 (paper January 22) is realistic only for a simulation-first paper without hardware. UAI 2027 (estimated February) fits only if nothing slips. NeurIPS 2027 (estimated May) or TMLR is the realistic target for the full ladder. Kill points: G1 (no headroom, numerical-only paper path), G2 (interface not learnable), G3 (no state dependence), G4 (negative transfer, no LLM hardware).

## 9. Supporting calculations

### 9.1 Sample size for paired log-area ratios

n = (z(1 - alpha/(2m)) + z(1 - beta))^2 x (sigma_s^2 + sigma_r^2 / R) / delta^2, with m = 3 (Holm, worst case for the first test), alpha = 0.05, beta = 0.2, so the leading factor is (2.394 + 0.842)^2 = 10.47. delta = ln(1/0.9) = 0.1054 for a 10% reduction.

Planning values. sigma_s from the November paired log final posterior-risk contrasts (SD = |diff/t| x sqrt(24)): local information 0.104, coverage 0.193, random 0.258, ranker 0.278. sigma_r = 0.5 is an assumption for one episode's paired log-area difference under common random numbers. E1 measures both.

| sigma_s | R | n for 10% | n for 15% | n for 20% |
|---:|---:|---:|---:|---:|
| 0.10 | 4 | 68 | 29 | 15 |
| 0.20 | 4 | 97 | 41 | 22 |
| 0.25 | 4 | 118 | 50 | 26 |
| 0.28 | 4 | 133 | 56 | 30 |
| 0.20 | 8 | 67 | 28 | 15 |
| 0.25 | 8 | 88 | 37 | 20 |
| 0.28 | 8 | 103 | 44 | 23 |

Rule: W-conf uses n = 128 and R = 4. If E1 gives sigma_s above 0.27, use R = 8 (n at most 104 for sigma_s up to 0.28), otherwise raise n, cap 192. E7 with 64 systems and R = 4 detects about a 13% reduction at the same error rates. E5 gate precision (64 systems, R = 4): SE 0.044, 95% half-width 0.087 on the log ratio, enough to decide a 0.85 threshold. E1 precision (32 systems, R = 8): SE 0.054, half-width 0.106, enough to classify headroom ratios of 1.15 to 1.25.

### 9.2 Hardware power

Online blocks without common random numbers, assumed per-block paired SD 0.65 on the log area (assumption, E8 replaces it). Minimum detectable effect at 80% power, two-sided alpha 0.05: 12 blocks 41%, 24 blocks 31%, 48 blocks 23%. A 15% effect needs about 126 blocks, about 600 EUR for two arms alone. Hence the recorded pool with replay for within-apparatus precision, online blocks as a validity check, and a design chosen by the E8 rule from simulation evidence only.

### 9.3 Chamber time and money (0.014 EUR per credit, 1 credit = 1 s)

| Item | Seconds | EUR | With 30% |
|---|---:|---:|---:|
| E2 calibration (free credits) | 720 | 0 | 0 |
| E9 pool, 25 x 2 x 12 x 14 s | 8,400 | 118 | |
| E9 online D2, 5 arms x 12 blocks x 180 s | 10,800 | 151 | |
| E9 online D1, 3 arms x 24 blocks x 180 s | 12,960 | 181 | |
| E9 overhead | 700 | 10 | |
| **E9 total, D2 / D1** | 19,900 / 22,060 | 279 / 309 | 362 / 402 |
| E10a, pool 4,200 + online 2,160 + 300 | 6,660 | 93 | 121 |

Check against the brief: 4 x 30 x 16 x 10 s = 19,200 s = 5.33 h = 269 EUR, matching the stated 5.3 h and 270 EUR.

### 9.4 Compute

- E1 and E3: about 2,560 episodes each, about 15 and 10 CPU-hours on local CPU.
- E4: about 3,600 forward passes plus probes, 1 to 3 hours locally. Q3 about 0.3 slice-hours.
- E5 (0.6B): SFT 1.7 to 4.2, GRPO about 5.3, in-family about 5, margin 30%, about 20 slice-hours. 1.5B about 30 slice-hours optional. About 20 CPU-hours for labels and simulators.
- E6: about 72,000 to 92,000 forward passes (8 to 10 h local at 0.4 s), about 33 CPU-hours numerical.
- E7: about 15,000 forward passes, 2,200 cycle solves.
- Basis: brief unit costs (LLM decision 0.4 to 0.7 s with 2 to 4k-token prompts and greedy generation, SFT update 0.62 s single example) scaled to about 1k-token prompts and one forward pass per decision. The remaining envelope (13,735 CPU-s, 2,742 slice-s) covers only E4 Q3, so E5 needs a new ceiling. The account's 450,000 SUs are not the binding limit.

## 10. Asks for the lead

1. **Remote Lab.** Your own sign-up and acceptance of Terms v1.1 (no automated account registration, fees non-refundable, data jointly owned with either party free to publish under an open licence, the company may use your name for marketing, Swiss law, possible rate limits on automated access). Use of the 1,000 free credits for E2. A later top-up of up to 400 EUR at the academic rate for E9, released only after the E8 gate, and up to 120 EUR for E10 (could). Confirm academic eligibility (commercial pricing is about 2.7 times). Whether hardware data may be released openly.
2. **Installs.** `causalchamber` 0.2.8 (MIT, numpy-only simulators) in an isolated environment or vendoring the pinned file. A separate Python 3.11+ environment with `causalchamber` 1.0.0rc1 for the Remote Lab client. `om-pycycle` 4.2.0 with OpenMDAO for E7.
3. **Local model execution.** Forward-only inference on the Mac (E4, E6, E7 and the hardware policy service) under a written protocol, and the small Q3 fit.
4. **Model weights.** Use of the cached Qwen2.5-1.5B-Instruct (capacity arm) and Qwen2.5-7B-Instruct (frozen probes only, unless a larger-GPU ceiling for training is granted).
5. **Compute ceiling.** About 20 GPU slice-hours and 60 CPU core-hours for E4 to E6 at 0.6B. Optional 30 slice-hours for the 1.5B arm. Agreement to the roster and custody redesign (more than 8 arms, more than 64 worlds).
6. **Operator contact.** A message from you to the Causal Chambers operators asking whether the listed wind-tunnel IDs are physically distinct units and about shared-session limits and latency.
7. **Venue and date.** ICML 2027 cannot carry the hardware stage. UAI, NeurIPS 2027 or TMLR for the full ladder.
8. **Negative-paper acceptability.** Whether a rigorous paper stating that numerical SCM planning suffices on a real-apparatus task, with a hardware demonstration, is acceptable if LLM gates fail. This decides whether E9 ever runs numerical-only.
9. **ACE-Runner scenarios.** Whether they may appear in a public paper, if used at all.
10. **Public preprint and README.** Reconcile the advertised 70% claim before releasing any new hardware data or results under the project's name.

## 11. Theory items

1. **Headroom bounds what any acquisition policy can add.** With a shared learner and decision rule, the gap between any policy and the generic planner splits into an information term (Bayes-optimal under public information, semantics included, versus the planner's prior) and a planning term (myopic versus optimal under the same information). H2', H4 and H3 estimate these terms. If all are near 1, no policy, including an LLM, can win by more than the residual, which justifies the numerical-only gate as a criterion independent of LLM performance.
2. **Pool replay is unbiased** for menu-restricted policies when actions map to recorded settings and approaches are exchangeable within a setting, with a drift-bias bound under interleaved blocks. This justifies the E9 replay endpoint and the rule against interpolating unobserved actions.
3. **Variance components with common random numbers** (sigma_s^2 + sigma_r^2/R) and the cost-optimal replicate count given the cost of a system versus a replicate. This justifies n and R.
4. **Calibration of the chance-constrained plug-in decision** under a correct model, and the direction of its bias under the generic model's misspecification, which explains specification-violation rates.
5. **Value of non-myopia under switching costs:** a two-setting example with move cost c_m and replicate cost c_r showing the myopic loss ratio as a function of c_m/c_r, which motivates H3 and the extra-computation control.

## 12. What not to do

- Do not run any LLM arm on a family before its numerical-only headroom audit passes and its confirmatory namespace is frozen. Do not drop or add a family after seeing LLM results.
- Do not touch the Remote Lab during development except the fixed E2 calibration, and never on the 25-point task grid before E9.
- Do not treat chamber IDs, hatch configurations, specification levels or replay episodes as independent systems.
- Do not interpolate unobserved settings in the pool or count them as responses.
- Do not put per-action planner scores in the primary LLM prompt. Report the argmax-of-prompt rule whenever derived numbers are in the prompt.
- Do not let the LLM make the final recommendation in the primary analysis.
- Do not train the primary cross-family policy on wind-tunnel or turbomachinery classes. Label any in-family policy as in-family.
- Do not count decision latency as experimental cost in the primary endpoint. Report billed latency separately.
- Do not use first-crossing NMSE or greedy trie decoding again.
- Do not build ThermoPower via OpenModelica this cycle. Do not use BoxingGym or bnlearn as the experimentalist task or as transfer evidence.
- Do not use ACE-Runner scenarios as headline transfer (team-authored, known defects).
- Do not spend beyond the free credits before the E8 gate, and do not rerun hardware episodes to improve a result.
- Do not scale the November DPO recipe, rerun completed studies or the old pipelines, or tune against the 24 development outcomes.

## 13. Biggest uncertainties and unverified assumptions

Uncertainties:

1. Whether WT-OP has headroom over cKG at all. Fan physics is monotone and smooth, so E1 may fail, which is why it comes first.
2. Whether a 0.6B or 1.5B model learns a state-dependent policy under the new interface after eleven months of collapse.
3. Whether small Qwen models know enough fan and compressor physics for zero-shot semantic transfer, and whether they recall the published chamber documentation.
4. Mk1 simulator fidelity to Mk2 hardware (coupling, dead zone, drift, sensor noise).
5. Remote Lab practicalities: account approval, pre-release client, queue waits, session limits, latency, academic pricing.
6. Engineering time and custody overhead against venue dates (half of last cycle's failed submissions were harness rejections).
7. One physical apparatus limits the hardware claim.
8. All GPU throughput figures are estimates until E4 measures them.

Assumptions to verify at integration: the exact wind-tunnel edge list and signs; single-token letter identifiers in the pinned tokenizer; that `wt.SimA1C3` and `wt.ModelB1` accept the per-system parameters listed; that hatch angles 0 and 30 degrees give distinct responses; a throttled single-spool rig in pyCycle and its Python 3.9 compatibility; cycle-solve time; sigma_r = 0.5 and the hardware per-block SD of 0.65; and that the listed chamber IDs are distinct units.
