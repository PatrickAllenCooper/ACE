# literature-benchmarks: environments, comparable evaluations and venue timeline

Prepared 2026-11-12 by the `literature-benchmarks` subagent. Read-only on the repository. Web sources were opened on 2026-11-12 unless stated. Confidence labels: **V** = verified on a primary page I opened, **S** = seen only in a search-engine summary or third-party page, **M** = from memory and not re-verified, **I** = my inference or arithmetic.

No experiments, installs, sign-ups or model runs were performed. No project name, person name, path or unpublished number was placed in any query or URL.

## 0. What the repository had already assessed (local files)

- Agenda benchmark ladder: controlled tasks, independent interactive environments (BoxingGym), engineering transfer (ThermoPower compressor "candidate, not yet integrated"), and the statement that "Live chamber access requires a separate arrangement". `/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md:135-149`.
- BoxingGym pinned to upstream `b43e38cb...`, MIT, Lotka-Volterra and Signal adapters exist. Outcome: a data-only numerical family selector matched the LLM proposal; a wrong source-count cue was accepted on all three reused Signal worlds. `/Users/pat/code/ACE/docs/development/guidance/external_benchmark_feasibility_2026-09-27.md:7-26`, `/Users/pat/code/ACE/docs/development/guidance/broad_experimental_agenda_2026-10-01.md:19`, `/Users/pat/code/ACE/docs/development/guidance/research_recovery_agenda_2026-10-01.md:13`.
- Causal Chambers: only the logged `lt_malus_v1` dataset was used (prediction, not acquisition). "pre-existing physical dataset, not a callable simulator". `/Users/pat/code/ACE/docs/development/guidance/causal_chambers_gate_2026-10-01.md:7,19`. The team already pinned `causal-chamber-package` commit `9fb5d82...` for the published Malus model. `/Users/pat/code/ACE/docs/development/guidance/chambers_physics_correction_2026-10-02.md`. Local numbers on `white_64`: published physics basis test NMSE 0.0953, tensor Fourier 0.0798 (`/Users/pat/code/ACE/results/chambers_published_physics_screen_20261002/result.json`, `/Users/pat/code/ACE/results/chambers_prediction_screen_verified_20261002/result.json`).
- CausalMan (Bosch, pinned `17529dad...`): engineering feasibility passed (9 actions, 144 rows, 157 nodes / 170 edges), scientific task suitability failed on the first probe menu. `/Users/pat/code/ACE/docs/development/guidance/technical_path_2026-10-01.md:11,63-65`.
- NeuronBench (MIT, Python >= 3.11): custody smokes only. `/Users/pat/code/ACE/docs/development/guidance/neuronbench_feasibility_2026-09-27.md`.
- AISTATS 2027 deadline recorded locally as October 6, 2026 and closed. `/Users/pat/code/ACE/docs/development/guidance/ace_submission_audit_2026-11-01.md:46`. Confirmed online (section 5).
- Local tooling state (read-only check with `/usr/bin/python3`, Python 3.9): `causalchamber`, `fmpy`, `OMPython`, `tespy`, `pycycle`, `openmdao`, `pgmpy`, `pymc`, `arviz`, `pcgym`, `causalbench`, `causalAssembly` are all absent. `torch` and `transformers` are present. `omc` is absent. `/usr/local/bin/docker` exists.
- Current world interface: `/Users/pat/code/ACE/scripts/research/llm_intervention_world.py` (393 lines; `Node(name, parents, unit, intervention_bounds)`, `LinearMechanism`, `ResponseAdapter`) and `/Users/pat/code/ACE/scripts/research/llm_intervention_scm.py` (149 lines; `LinearGaussianSCM` with known noise variances). Any external environment needs a new response adapter and, for nonlinear physics, a learner other than the plain linear head (I).

## 1. Environment assessments

### 1.1 Causal Chambers (largest change versus the local assessment)

**There is now a public, self-service Remote Lab with a Python API to real hardware, and the authors publish MIT-licensed mechanistic simulators in the pip package.** Both answers are yes.

Remote Lab (V):
- Website: "Connect to a Chamber in our Zurich lab with a single line of Python code", "run interactive experiments in a real, non-simulated physical environment". http://www.causalchamber.ai/
- Access: sign-up form https://forms.causalchamber.ai/signup; credentials arrive by email; stored in a file with `[api_keys] user/password` or env vars `CAUSALCHAMBER_USER`, `CAUSALCHAMBER_PASSWORD`. https://docs.causalchamber.ai/remote-lab/quickstart
- Pricing: "The base unit is one credit, which corresponds to one second of runtime." 1000 free credits on sign-up. Academic rate 0.014 EUR or 0.013 CHF per credit; commercial 0.038 EUR / 0.042 USD / 0.035 CHF. The USD academic cell reads "0.016 CHF", apparently a typo. Minimum card top-up 100 credits. Credits do not expire. Solidarity pricing on request. Maximum measurement rate about 10 per second. https://docs.causalchamber.ai/remote-lab/how-credits-work
- Interactive mode: `rlab = lab.login(...)`, `rlab.new_session(chamber_id, config, duration=...)` gives an exclusive connection; `chamber.set(target, value)` "returning when the hardware has made the change"; `chamber.measure(n, delay)`; `chamber.wait(ms)`; batches bundle instructions in one round trip. Minimum session 60 seconds. Shared-pool sessions are charged upfront for the requested duration and unused time is refunded. The docs explicitly name "active learning, experiment design, or control algorithms" as the use. https://docs.causalchamber.ai/remote-lab/running-a-real-time-experiment
- Queue mode: protocols of set/measure/wait, `experiment.cost()` returns the maximum charge before submission, refund if faster. https://docs.causalchamber.ai/remote-lab/using-the-experiment-queue
- Dedicated chambers can be rented at a flat rate (contact required).
- Terms v1.1, effective 17 March 2026 (PDF read in full, 10 pages, https://box.causalchamber.ai/gtc.pdf): "Accounts registered by bots or automated methods are not authorised and will be terminated." Automated algorithms are allowed for legitimate internal purposes if they do not impose unreasonable load. Company "reserves the right to impose reasonable rate limits or usage caps on automated access". Availability "at least 98%". "All fees are non-refundable unless required by law." Data generated through use are jointly owned and "Either party may ... release or publish such generated data under an open-source license". The company "may use Customer's name, logo, and service summary for marketing". Swiss law.

Hardware documented (V):
- Wind Tunnel Mk2: "time-series data from a dynamical system", up to 41 variables, one config `full`. Actuators `load_in`, `load_out` (PWM duty cycle, 0 to 1, running fans never below about 0.1), `hatch` (0 closed, fully open at plus or minus 45 degrees), motor and sensor parameters. Sensors `rpm_in`, `rpm_out`, `current_in`, `current_out`, `current_mot` (amperes), `pressure_intake`, `pressure_upwind`, `pressure_downwind`, `pressure_ambient` (pascals), `mic`, `hatch_angle`. Documented effects: load to speed and current with lag, cubic steady-state current, shared power supply coupling, speed coupling through the air path weakened by the hatch, atmospheric pressure as a common cause, sensor drift. Examples wait 8,000 ms for fans to stabilize. Chamber IDs in examples: `wt-h0pe-i4ug`, `wt-demo-ch4lu`, `wt-ptdm-73iz`. A ground-truth graph is provided. https://docs.causalchamber.ai/the-chambers/wind-tunnel-mk2
- Light Tunnel Mk2: i.i.d. data and images, up to 99 variables, configs `standard`, `linked_leds`, `linked_leds_sigmoid`, `camera_fast`, `led_matrix`. Actuators `red`, `green`, `blue` (0 to 255), `pol_1`, `pol_2`, sensor parameters. Malus law and other mechanisms are described in prose. https://docs.causalchamber.ai/the-chambers/light-tunnel-mk2
- Hexapod Mk1, Lens Array Mk1 and Circuit Stack Mk1 pages say only "still under development".
- Users can program feedback (set an input as a function of a measurement), remove or modify effects through mechanism variables, and add hidden confounders. https://docs.causalchamber.ai/case-studies/causal-inference/generating-real-data-with-a-known-causal-structure

Package and simulators (V):
- PyPI `causalchamber` 0.2.8 (PyPI shows Apr 20, 2026; `CHANGES.txt` shows 2026-04-26), MIT, "Python >=3.6", optional extra `torch`. Pre-release 1.0.0rc1 on 2026-10-29 moves the client to API version 1, adds `lab.login()` and `new_session()`, and requires Python 3.11 or newer. https://pypi.org/project/causalchamber/ , https://raw.githubusercontent.com/juangamella/causal-chamber-package/main/CHANGES.txt
- Simulator index (`causalchamber/simulators`): `wt.ModelA1`, `wt.ModelA2`, `wt.ModelB1` (deterministic fan speed and current from load), `wt.SimA1C2`, `wt.SimA1C3`, `wt.SimA2C3` (stochastic downwind pressure from `load_in`, `load_out`, `hatch`, `pressure_ambient`, with `barometer_error`, `barometer_precision`, `random_state`; returns `(pressure_downwind, rpm_in, rpm_out)`), `lt.Deterministic` (light-tunnel sensors), `lt.ModelF1-F3` (images), `lt.DecoderSimple` (needs torch and downloads a pretrained model). The wind-tunnel file imports only numpy.
- Physics in the Nature Machine Intelligence paper, Appendix IV (arXiv 2404.11341v2): A1 fan affinity laws (omega_max = 314.16 rad/s, L_min about 0.1 experimentally versus 0.2 on the datasheet); A2 torque-balance ODE; B1 current cubic in speed (C_min = 0.166 A, C_max = 0.26 A); C1 to C3 static pressure proportional to squared speed ratio, with impedance and hatch term r(H) = min(1, r0 + beta H / 45), S_max = 74.82 Pa, Q_max = 0.052 m^3/s, r0 = 0.7, beta = 0.15. Stated misspecification: fans modelled independently although coupled; impedance not computable. E1 Malus law fits with R^2 0.93, 0.97, 0.99 at brightness 64, 128, 255.
- Ground-truth edges validated by randomized experiments with two-sample Kolmogorov-Smirnov tests; wind tunnel all p below 8e-7 except L_out to intake pressure, p = 0.082 (rank-sum 0.029).
- The docs list 71 papers using chamber data, including a multi-fidelity causal Bayesian optimization workshop paper (UAI 2025 CAR workshop). I could not open that OpenReview page (browser verification), so its design is unverified.

Assessment against the brief:
- Interventions executable on demand: yes, on real hardware, and offline on the simulators.
- Natural cost: yes. Chamber seconds are billed. Changing fan load needs seconds of settling, repeat measurements are fast, so "move versus replicate" has a real cost asymmetry (I).
- Semantic descriptions: yes, per-variable prose plus mechanism notes.
- Useful outcome: can be defined as an operating-point task (target pressure at minimum fan current) or a prediction task (I).

Cost arithmetic (I, assumptions stated): 10 s per steady-state intervention (8 s settle plus measurement). A 24-intervention episode is about 240 s, 240 credits, about 3.4 EUR academic. 192 episodes (8 policies x 24) is about 46,080 s, 12.8 h, about 645 EUR academic (1,751 EUR commercial). 4 policies x 30 episodes x 16 interventions is about 19,200 s, 5.3 h, about 269 EUR. The 1000 free credits cover about 16.7 minutes, enough for an interface smoke only. Policy decision latency inside a session is billed too.

Limits to keep visible:
- One apparatus per chamber type. Configurations, targets and chamber IDs are not independent systems; the local rule against manufacturing replication applies (`/Users/pat/code/ACE/docs/development/guidance/causal_chambers_gate_2026-10-01.md:7`).
- No noise-free truth. The private evaluator must be a separately measured, charged validation grid with replicates.
- Drift and ambient pressure require randomized, blocked ordering of policies.
- Simulators implement the Mk1 models; Mk2 variable names differ (mapping page exists). Fidelity to Mk2 is not documented. The `SimA*C*` tutorial link points to an internal repository.
- Shared-pool waiting time, latency and maximum shared session length are not documented.
- Sign-up must be done by a human, and it is an external paid service. The handoff says no paid APIs or installs are part of the current scope (`/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md:109`). The changelog also mentions a dataset-module "call-home"; what it sends is unverified.

### 1.2 Compressor and turbomachinery simulators

- **ThermoPower (OpenModelica)**: version 4.0.0 on the OpenModelica documentation server, "licensed by Politecnico di Milano under the Modelica License 2", uses Modelica 4.0.0. The `Gas` package was renamed `IdealGas`, so **the agenda's link `ThermoPower.Gas.BaseClasses.CompressorBase.html` now returns HTTP 404**; the working page is https://build.openmodelica.org/Documentation/ThermoPower.IdealGas.Compressor.html (V). The compressor uses three 2D tables (`tablePhic`, `tablePR`, `tableEta`) over referred speed and beta lines, parameters `Ndesign`, `Tdes_in`, `eta_mech` 0.98, flanges inlet, outlet, shaft. Default tables are empty, so maps must be supplied. Test models `ThermoPower.Test.GasComponents.TestCompressorConstSpeed` ("Simulate for 2s"), `TestCompressorInertia`, and `ThermoPower.Examples.BraytonCycle.{Plant, OpenLoopSimulator, ClosedLoopSimulator}` exist. OpenModelica nightly library test, master, run 2026-11-12: 100 models, 97 simulate; the compressor tests and Brayton simulators show full execution (about 3.4 to 4.9 s each). An FMI-export branch run dated 2026-11-07 (OMSimulator, ThermoPower 3.1.0-master) shows 62 of 108 simulating, including the two compressor tests and both Brayton simulators (V, https://libraries.openmodelica.org/branches/master/ThermoPower/ThermoPower.html).
- **OpenModelica install**: official Mac binaries "were discontinued after version 1.16"; Docker or Linux recommended; source builds possible since 1.20.0-dev (V, https://openmodelica.org/download/download-mac/). Burden is high on the Mac and needs a container on a cluster (I).
- **FMPy**: 0.3.32 released Sep 9, 2026, Python >= 3.10, FMI 1.0/2.0/3.0, Co-Simulation and Model Exchange, pure-Python wheel (V, https://pypi.org/project/FMPy/). License not shown on the pages I fetched (BSD-2-Clause from memory, M).
- **pyCycle (NASA/OpenMDAO)**: Apache License 2.0 (LICENSE read), `pip install om-pycycle`, version 4.2.0 needs OpenMDAO 3.10.0 or greater. Map files `axi5.py`, `axi3_2.py`, `HPC_map.py`, `LPC_map.py`, `Fan_map.py`, `HPT_map.py`, `LPT_map.py`, `ncp01.py`, `hpt1269.py`, `lpt2269.py`. Example cycles include `simple_turbojet`, `high_bypass_turbofan`, `single_spool_turboshaft`, `electric_propulsor` (V). Steady-state by construction. Suitability for a single compressor rig with a throttle is unverified.
- **TESPy**: MIT, `pip install tespy`, docs v0.12.0.post1. `Compressor` (`eta_s_char`), `TurboCompressor` (`char_map_pr`, `char_map_eta_s`, `igva`). Rotational speed "is not a parameter" of the map formulation (constant-speed lines), so it does not give a speed actuator directly. A `PolynomialCompressor` for frequency-dependent maps was mentioned in a search summary (S).
- **ccp (centrifugal compressor performance)**: Apache-2.0, `pip install ccp-performance`, CoolProp or REFPROP. Interpolation at arbitrary speed and flow not confirmed.
- All of these are simulation-only with synthetic costs. The causal sketch must be verified against the chosen simulator, as the agenda already requires (agenda line 145).

### 1.3 BoxingGym

MIT; Python >= 3.11; PyMC; 10 environments (hyperbolic temporal discount, location finding, death process, IRT, survival, dugongs, peregrines, Lotka-Volterra, moral machines, emotion). `include_prior` toggles domain description versus abstract description. Metrics: standardized prediction error and EIG regret against the best of 100 random designs. Budgets 0, 1, 3, 5, 7, 10; 5 runs per environment; standard errors; no tests; no random baseline. The paper states it "ignores resource constraints". Finding: prior information does not reliably help and sometimes hurts. arXiv v2 dated 14 Oct 2025. The NeurIPS 2025 page lists it as a poster in the Workshop on Scaling Environments for Agents, not the Datasets and Benchmarks track (V, https://neurips.cc/virtual/2025/124548). Queries are designs, not do-operations on a supplied graph. Three of its environments (location finding, temporal discounting, death process) are the three Deep Adaptive Design tasks, which gives a route to the DAD comparator the agenda asks for (I).

### 1.4 Gene perturbation data

- **CausalBench**: Apache-2.0 code; Replogle Perturb-seq data CC-BY-4.0; K562 and RPE1; "over 200 000 training samples under interventions"; `pip install causalbench`, v1.1.1, Python 3.8. Logged data only; task is network inference, not adaptive selection (V).
- **GeneDisco**: Apache-2.0, `pip install genedisco` (v1.0.5, Python 3.8); active-learning benchmark for CRISPR experiments; pool-based status not stated on the README I read.
- **BioDiscoveryAgent** (arXiv 2405.17631 v3): 6 single-gene datasets plus one gene-pair dataset, 5 rounds x 128 genes, 10 runs, plus or minus 1 SD, no tests, baselines random, human pathway sampling and seven GeneDisco acquisition functions. Ablation separates prompt-only, observation-only and both.
- No supplied causal graph, very large action sets, and cost is a count of perturbations. Poor fit for an SCM-conditioned policy with a known graph (I).

### 1.5 Standard interventional causal-discovery testbeds

- **bnlearn repository** (page content CC BY-SA): discrete Asia, Sachs, Child, Insurance, Alarm and others; Gaussian ECOLI70 (46 nodes, 70 arcs), MAGIC-NIAB (44 nodes, 66 arcs), MAGIC-IRRI, ARTH150, distributed as RDA/RDS `bn.fit` objects (V). The Gaussian networks are linear-Gaussian and match the current learner class (I).
- **LeGIT** (LLM-guided intervention targeting, arXiv 2503.01139): Asia 8/8, Child 20/25, Insurance 27/52, Alarm 37/46; 33 rounds x 32 samples (or 16); 5 seeds; mean plus or minus SD; no tests; GPT-4o given variable names and descriptions; LLM used only for warm-up, then a numerical method. Human beats it on Child. No memorization analysis found in the part I read (V).
- **pgmpy** exposes `simulate(..., do=...)` (S; license not confirmed).
- **SERGIO**: GPL-3.0; gene IDs must be numeric indices, so no semantics (V). Used by CAASL (NeurIPS 2024), which reports returns over 100 random test environments against random, observational, DiffCBED and SS-Finite (S).
- DREAM4 and Sachs were not opened.

### 1.6 Process control

- **Tennessee Eastman**: `pyTEP` needs a licensed MATLAB engine and Python 3.7 (S). AIRI `iprl_benchmark` is MIT with a simplified TEP, an exothermic CSTR and a distillation column, but pins Python 3.9.13 and `ray==1.9.1`, has 2 commits and no variable documentation (V). CIPCaD-Bench has a `TenneseeEastman` data folder; license, simulator and graph details were not visible (a 2026 paper describes 33 variables and 32 edges, S). `tep-studio` (BSD-3, 2026) appeared in a search snippet; its PyPI page failed to load. **I found no verified pip-installable, license-clear TEP simulator with on-demand interventions.**
- **PC-Gym**: MIT, `pip install pcgym`; CSTR, first-order system, multistage extraction, nonsmooth control (README), crystallization and four-tank in the paper (S).
- **BOPTEST**: FMU-based building emulators with a REST API and a public web service; license and KPI names not verified.

### 1.7 Manufacturing SCMs

- **causalAssembly**: AGPL-3.0, `pip install causalAssembly`; semisynthetic sampler from distributional random forests fitted to assembly-line data; hard and soft interventions via `intervene_on` and `sample_from_interventional_drf`; names like `Station3_mp_41`; R needed to refit (V). Interventional samples are model extrapolations, not recorded interventions (I).
- **CausalMan**: AGPL-3.0, Python 3.9+, about 7.6 GB of dataset objects, variants micro (24 observable variables) to large (more than 400), hard interventions through `intervention_dict`, descriptive names (V). Already audited locally.

### 1.8 Newer LLM experimentation benchmarks (2025 to 2026)

- **PhysGym** (MIT; NeurIPS 2025 Datasets and Benchmarks per a search result): 97 problems, 100-experiment budget, four prior levels (L1 full context, L2 no context, L3 no descriptions, L4 anonymized names). Success by level: Gemini-2.5-flash 65.98, 49.48, 36.08, 30.93; o4-mini 62.89, 41.24, 31.96, 27.84; Claude 3.7 Sonnet 22.68, 20.62, 21.65, 12.37. Priors sometimes hurt. No error bars or tests. Deterministic, no noise or cost.
- **SciGym** (NeurIPS 2025 Datasets and Benchmarks): 350 BioModels SBML systems, 137 evaluated; action is changing initial concentrations (knockouts in an ablation); 20 iterations; species names kept; Tellurium/libRoadRunner; no noise.
- **NewtonBench** (arXiv 2510.07172): 324 tasks, 12 physics domains, shifted laws to resist memorization, noise sensitivity.
- **Auto-Discovery-Bench** (arXiv 2502.15224 v2, renamed from Auto-Bench): deterministic oracle-guided graph, relational and equation discovery.
- **CausaLab** (arXiv 2605.26029): interactive causal discovery on random SCMs; GPT-5.2-high 92% task accuracy with 0.471 all-edge F1 in the 6-node observational setting.
- **CausalGame** (arXiv 2607.04293): 14 scenarios, 30 LLM agents, best 68.0% survival versus analytical optima of 78 to 85%.
- **SimulCost** (arXiv 2603.20253 v4, ICML 2026): cost-aware simulation tuning; multi-round LLMs reach 66 to 81% success but are 1.5 to 2.7x slower than scanning.
- **SCOPE** (arXiv 2608.03501): scores experiment plans from 300 papers; nothing is executed.
- None of these supplies a known causal graph together with real do-operations and a monetary cost.

## 2. Where a language model could show something a numerical planner cannot easily do

| Capability | Best-supported environment | Evidence and caution |
| --- | --- | --- |
| Textual domain knowledge | Causal Chambers docs; PhysGym prior ladder; bnlearn networks with descriptions; compressor simulators | PhysGym shows large prior effects for strong models and occasional harm. BoxingGym shows no reliable benefit. Local BoxingGym results were negative. |
| Natural-language objectives | Wind tunnel (pressure, current, speed limits), compressor rig (power at required pressure under a temperature limit) | Not tested by any benchmark I found. A semantic-feature numerical policy is needed as control (agenda line 39). |
| Mechanism-family hypotheses | Causal Chambers model ladder A1/A2, B1, C1 to C3, E1 | A numerical family selector matched the LLM locally on Lotka-Volterra. |
| Misspecified numeric models | Causal Chambers (simulator versus hardware; linear SCM versus cubic current and quadratic pressure) | The chambers docs host an explicit model-misspecification benchmark for simulation-based inference. |

Literature on whether LLMs use feedback is contested: Gupta et al. (Findings of EMNLP 2025) found permuted feedback did not change open-model performance (for example Llama 39.4 versus 37 hits on IL2) and classical methods usually won; a March 2026 preprint (arXiv 2603.26177) reports 800 replicated experiments where feedback gives +53.4% discoveries (p = 0.003), vanishing under permuted feedback, and only for the stronger model version. Model capability appears to be a moderator (I), which matters for a 0.6B policy.

## 3. How comparable papers structure evaluation

| Paper | Systems or tasks | Repeats | Uncertainty | Tests | Endpoint |
| --- | --- | --- | --- | --- | --- |
| Deep Adaptive Design (ICML 2021) | 3 tasks (T = 30, 20, 4) | 2048 rollouts (256 for variational), 16 training seeds | plus or minus 1 s.e. | none | EIG bounds at fixed T, deployment time |
| CAASL (NeurIPS 2024) | synthetic linear plus SERGIO | 100 random test environments | not checked | not checked | return, SHD at fixed budget |
| LLAMBO (ICLR 2024) | 25 Bayesmark tasks, 24 HPOBench, private and synthetic sets | 5 seeds, 25 trials | unclear | none | normalized regret; ablation with names replaced by X_i |
| BoxingGym | 10 environments | 5 runs | s.e. | none | error after k experiments, EIG regret |
| BioDiscoveryAgent | 6 datasets (+1 pair) | 10 runs | 1 SD | none | hit ratio after 5 x 128 |
| Gupta et al. 2025 | 5 gene + 3 molecule sets | 5 runs | none in tables | none | cumulative hits; permuted-feedback control |
| LeGIT | 4 graphs | 5 seeds | SD | none | SHD, SID, BSF at fixed rounds |
| PhysGym | 97 problems x 4 prior levels | not stated | none | none | success within 100 experiments |
| arXiv 2603.26177 | one screening dataset | 800 replicates | p-values | yes, unnamed | discoveries per feature |

Observations (I): most comparable work reports performance at a fixed budget, 5 to 10 seeds, and no formal tests. Cost-to-target with censoring is rare. The project's paired-system design with retained censoring is stricter than the norm. Three controls recur and are expected by informed reviewers: permuted or shuffled feedback, anonymized variable names or removed descriptions, and a numerical learner given the same semantic embedding. Useful statistical tooling: interquartile mean, stratified bootstrap intervals, performance profiles and probability of improvement (Agarwal et al., NeurIPS 2021, `rliable`, S); data profiles (More and Wild 2009) and expected running time with unsuccessful runs (COCO), both M.

## 4. Ranked shortlist for the concrete experimentalist task

1. **Causal Chambers wind tunnel, in two stages.** Stage A: offline world built on `wt.SimA1C3` plus `wt.ModelB1` (pip, MIT, numpy only). Stage B: frozen policies on the real Wind Tunnel Mk2 through the Remote Lab. Only option with real do-operations, billed cost, validated graph, prose descriptions and a documented misspecification ladder; it is also the nearest physical analogue of the fan/compressor story. Effort (I): Stage A about 1 week; Stage B about 1 to 2 weeks plus human sign-up, terms acceptance and roughly 300 to 700 EUR at academic rates. Claim scope: one apparatus.
2. **Executable compressor rig: pyCycle first, ThermoPower FMU as the higher-fidelity variant.** Matches the agenda's illustrative objective. pyCycle is pure Python with shipped maps (about 1 to 2 weeks, I, unprototyped). ThermoPower needs OpenModelica in Docker or Linux, FMU export and FMPy (about 2 to 3 weeks, I). Simulation only; costs are synthetic.
3. **BoxingGym location finding, temporal discounting and death process.** Independent published benchmark with prior and no-prior variants and the DAD task families. Adapters partly exist locally (about 3 to 5 days per added environment, I). No cost model, not do-operations, prior local results negative.
4. **bnlearn networks with supplied graphs.** ECOLI70 and MAGIC-NIAB fit the existing linear-Gaussian learner (about 3 to 5 days, I); Asia, Child, Insurance, Alarm give comparability with LeGIT but need a categorical learner (about 1 to 2 weeks, I). No natural cost; memorization risk.
5. **PhysGym-style prior ladder as a control design** applied inside options 1 to 4, rather than a separate integration. CausalMan remains a fallback manufacturing SCM (AGPL, 7.6 GB, one failed menu locally).

Not recommended as primary: CausalBench and GeneDisco (logged pools, no graph), SERGIO (no semantics), Tennessee Eastman (no verified clean simulator), causalAssembly (model-based interventions, AGPL, R), SciGym, CausaLab, CausalGame (different task, no cost).

## 5. Venue timeline after 2026-11-12

| Venue | Status | Dates | Source |
| --- | --- | --- | --- |
| CLeaR 2027 | open, 4 days left | Paper deadline Nov 16, 2026 11:59pm AoE. Reviews Jan 19, 2027. Rebuttal Jan 26. Decisions Feb 22. Camera-ready Apr 5. Conference May 3 to 5, 2027. 12 PMLR pages plus unlimited appendix. Substantive AI-tool use must be disclosed. | Official page rendered in a browser: https://www.cclear.cc/2027/CallforPapers (V). City "Paris" only from third parties (S); one of them wrongly lists March 3 to 5. |
| ICML 2027 | announced | Submission opens Jan 8, 2027. Abstract Jan 14, 2027 AoE. Full paper Jan 22, 2027 AoE. First-stage rejection notification Feb 23. Final decisions May 3. Conference Jul 31 to Aug 6, 2027, Riocentro, Rio de Janeiro. Call for papers "will be posted soon". | https://icml.cc/Conferences/2027/Dates , https://icml.cc/Conferences/2027 (V) |
| UAI 2027 | not announced | Official `uai2027` page returns 404. Trackers give tentative Feb 5, 6 or 12, 2027. Prior years: Feb 25, 2026 (Amsterdam, Aug 17 to 21, 2026) and Feb 10, 2025. **Estimate: February 2027.** | https://www.auai.org/uai2026/call_for_papers (V for 2026) |
| NeurIPS 2027 | not announced | Official 2027 page returns 404. 2026: abstract May 4, paper May 6, 2026. **Estimate: early to mid May 2027.** | https://neurips.cc/Conferences/2026/Dates (V for 2026) |
| AISTATS 2027 | closed | Abstract Sep 29, 2026; paper Oct 6, 2026; reviews Nov 16, 2026; decisions Jan 20, 2027; journal-to-conference deadline Jan 18, 2027 (details "posted separately"); Montreal, May 3 to 6, 2027. | https://virtual.aistats.org/ (V) |
| ICLR 2027 | closed | Abstract Sep 18, paper Sep 25, 2026. Conference Apr 26 to 30, 2027. Workshops: acceptance of proposals Nov 29, 2026; suggested paper deadline Feb 1, 2027; notification Feb 26, 2027. Workshop list not yet published. | https://iclr.cc/Conferences/2027/CallForWorkshops (V) |
| ICML 2027 workshops | not announced | 2026 pattern: suggested submission Apr 24, 2026 for Jul 10 to 11 workshops. **Estimate: late May 2027.** | https://icml.cc/Conferences/2026/Dates (V for 2026) |
| TMLR | rolling | Action editor assigned within a week. Criteria: claims supported by evidence; of interest to some of the audience. J2C certification; TMLR joined the NeurIPS/ICML/ICLR journal-to-conference track (announced 2025-10-21). | https://jmlr.org/tmlr/ , https://jmlr.org/tmlr/editorial-policies.html (V) |
| KDD 2027 cycle 2, IJCAI 2027 | unverified | Third-party projections only (February 2027; January 2027). | S |

## 6. Could not verify

- Multi-fidelity causal Bayesian optimization workshop paper using the chambers (OpenReview blocked).
- Shared-pool wait time, latency, maximum shared session duration, and which chamber IDs are in the pool.
- Fidelity of the Mk1 simulators to Mk2 hardware.
- `tep-studio`, CIPCaD-Bench contents and license, Summit benchmark details, Olympus license, BOPTEST license, pgmpy license, FMPy license text.
- MCBO evaluation details beyond the abstract; CAASL uncertainty reporting.
- UAI 2027 and NeurIPS 2027 dates.
- I did not run any simulator, so all integration estimates are judgments.

## 7. Surprises

- A public arXiv preprint, 2602.02451 (v1 2 Feb 2026, v2 18 Jun 2026), carries the same title as `/Users/pat/code/ACE/README.md:1-3` and `/Users/pat/code/ACE/paper/neurips_ace_2026/paper.tex:46` and claims a "70-71% improvement over baselines at equal intervention budgets (p < 0.001, Cohen's d ~ 2)". It surfaced from a generic query. The local metric audit says "every ACE-vs-baseline number is invalid as printed" (`/Users/pat/code/ACE/docs/development/guidance/metric_audit_2026-09-10.md:1-7`), and `README.md:14,114` still advertises the claim. That it is the same project is an inference from the identical title; I did not extract the author list.
- The agenda's Causal Chambers statement is outdated: self-service sign-up, free credits and a real-time API now exist.
- The agenda's ThermoPower link is dead because of the 4.0.0 package rename.
- CLeaR 2027 and AISTATS 2027 overlap (May 3 to 5 and May 3 to 6, 2027).
