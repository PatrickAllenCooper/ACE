# Slide 1

ACE selects interventions using an assumed causal graph and uncertainty about local mechanisms. Compressor visual is a newly generated conceptual engineering illustration, not the apparatus used in the reported synthetic experiments. Slides1–4 introduce the theory; slides5–7 give precise illustrative inputs, scores and outputs; slide8 shows an archived graph; slides9–10 show historical evidence. Implementation: baselines.py, EnsembleStudentSCM / EnsembleLearner / PropagatedVariancePolicy, and scripts/research/persistent_scm.py. Historical PEV uses known graphs, independently initialized neural heads, model-only candidate contexts and covariance-based integrated variance reduction summed over descendant mechanisms. It does not identify the graph or optimize the proposed terminal-risk objective. Noise proxy is residual error, not a certified aleatoric estimate.

Sources
/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md
/Users/pat/code/ACE/docs/ACE_evidence_and_public_claims_2026-10-08.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Revision7 inserts the local-versus-joint explanation at4 and the proposed foundation-model/SCM overview at6. The worked example is7–9, archived graph10, empirical learning curves11 and closing comparison12.
Version8: technology name Active Causal Experimentalism and user-provided tagline Every experiment counts. New white compressor/SCM concept cover edited with built-in image generation from the prior concept asset. Illustrative machine, not an experimental photograph. Prompt saved in cover_asset_v8.json. All measured acquisition results remain historical synthetic neural-SCM comparisons.

# Slide 2

SCM definition: Xi=fi(Pa_i,Ui). Edges specify direct causes in the assumed graph, f describes the mechanism, U describes external disturbances. The toy is a fully observed, deterministic chain with dimensionless values. Its true rules are M=2.5X and Y=3M, with U=0. These truth equations explain the example to the audience; the policy receives only its candidate models and allowed actions, not oracle truth. Known graph does not imply known mechanisms. General SCMs may have dependent disturbances; identification and fitting require explicit assumptions.

Sources
/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Compressor analogy introduced in version8: X is normalized drive-command deviation, M normalized shaft-speed deviation, and Y normalized pressure-rise deviation. Other conditions are held fixed. The supplied graph is a simplified teaching chain with linear, zero-disturbance illustrative rules, not validated compressor dynamics. Negative teaching values mean below-reference deviations, not negative physical RPM or absolute pressure. Measured historical experiments later are synthetic systems. The real compressor can have additional direct paths, nonlinearities, operating limits and confounding, which this teaching model does not establish.

# Slide 3

do(M=m) fixes M independently of its usual parents and replaces M=fM(X,UM) with M=m. X→M is removed for this experiment, while M→Y remains. Row X=x,M=m,Y=y supplies a natural Y label with measured parent m, but is not a natural M=fM(X) label. Under do(X=x), both M and Y remain naturally generated and eligible under the stated retention/noise assumptions. The implementation excludes the intervened head and uses observed parents for eligible training. Candidate simulation propagates ensemble-mean parents. Historical acquisition-study evaluation instead predicts each mechanism from observed parents; the toy composed target forecast and delivery studies have different evaluation semantics.

Sources
/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Version8 uses the same simplified compressor chain as slide2. Internal intervention assumes an approved independent test-rig controller can hold shaft speed, bypassing the usual drive-command mechanism. It is a legal action of this illustrative simulator/test rig, not a claim that arbitrary internal compressor quantities can be clamped. Unit1 means a normalized above-reference speed. The incoming X-to-M edge is absent during the clamp. M-to-Y remains. Numbers and eligibility rules are unchanged.

# Slide 4

Analytic representation example, not measured sample efficiency. Same discrete setup as the next slide: ten independent five-valued inputs, nine two-parent deterministic five-valued mechanisms in a binary reduction tree, all relevant variables observed. Each local table maps the two direct parent values to its child. Nine local maps compose to predict the final Y. Each local map has5^2=25input rows,225total. Joint lookup has one row per complete ten-input vector,5^10=9,765,625rows. Displayed outputs m1... and y1... are placeholders, not experimental observations. Learning all local entries requires every parent setting accessible with the child mechanism intact. Intermediate variables must be observed. A noncausal regressor need not enumerate this grid. Actual ACE learners fit functions (neural mechanisms in the historical results), not literal local lookup tables.
Sources
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/SCM_free_grid_comparison.json
/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md

# Slide 5

Analytic representation-size illustration, not an ACE benchmark. Define a deterministic discrete SCM with10independently controllable five-valued input nodes and9two-parent mechanism nodes arranged as a binary reduction tree. Every endogenous output also has5values. A naive associative lookup table for the final response over the full joint input grid requires5^10=9,765,625entries. With the graph supplied, nine local mechanism tables require9*5^2=225entries, a43,402.78fold representation-count difference. This is not a measured intervention-count reduction. Learning local tables assumes each required parent configuration is accessible with the child mechanism intact and allvariablesobserved; without internal interventions, reachability can fail. Sharedparentsettings may reveal several labels. Noncausal regressors can exploit smoothness/sparsity and need not enumerate a grid; the comparator is specifically exhaustive lookup, not allassociativelearning. No empirical superiority over this new baseline has been measured. Current PEV scores candidates by descendant ensemble covariance reduction; details retained in the worked calculation.

Sources
/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Uniform random sampling baseline: independently draw one of N=5^10 joint configurations, with replacement and no SCM. Expected distinct fraction after n draws is 1−(1−1/N)^n. The smallest n for95% expected coverage is29,255,197. This is expected grid coverage, not a95% probability of complete coverage or a prediction-error target. The exhaustive/local entry ratio is43,402.78; do not present the random/local ratio as experimental intervention savings. Local access and determinism assumptions stated above remain required.

# Slide 6

Implemented paths are distinguished. The lower SCM uncertainty/intervention/environment loop summarizes the measured PEV acquisition procedure. The upper LLM policy is a separate implemented historical path, not the source of the plotted PEV gains and not demonstrated superior by this slide. ace_experiments.py:HuggingFacePolicy loads a pretrained causal language model; supervised_pretrain_llm teaches teacher-generated legal intervention commands using graph/node losses; dpo_loss_llm increases preference for the winner over loser relative to a reference policy. Candidate scores include lookahead improvement and configured bonuses/scaffolding. Historical default lookahead can query the environment for candidate scoring; all such queries must be charged. --lookahead_on_student instead simulates from current learned mechanisms. This conceptual connection does not assert that the historical DPO learner used PEV scoring, that the two paths have been experimentally integrated, or that LLM training guarantees good intervention selection. The diagram proposes the interface between the LLM candidate channel and an SCM scorer while showing the numerical loop separately. The recent language mechanism-proposal screen is a different interface and had six invalid proposals; it does not qualify the LLM policy. Inputs to the policy include the SCM state, errors/history and legal command syntax. No new training was run.
Sources
/Users/pat/code/ACE/ace_experiments.py (HuggingFacePolicy, supervised_pretrain_llm, dpo_loss_llm and winner/loser updates)
/Users/pat/code/ACE/scripts/research/persistent_scm.py:campaign
/Users/pat/code/ACE/baselines.py:PropagatedVariancePolicy
/Users/pat/code/ACE/docs/development/guidance/ace_foundation_cycle_closeout_2026-10-09.md

# Slide 7

Illustrative configuration only. Three linear ensemble members: M slopes1,2,3, Y slopes2,3,4; ensemble means2X and3M. Candidates ordered X−1,X0,X1,M−1,M0,M1. Reference parent values−1,0,1. Population covariance denominator3, residual noise proxy1/20, deterministic mean-propagated parent contexts, no jitter or epsilon exploration. Preintervention observation history empty for this hand-constructed initial state. One response remaining. Oracle truth hidden from score: M=2.5X,Y=3M. Production historical confirmation uses neural heads and different grids/batches. All numbers are generated by explain_ace_mechanics.py with exact rational arithmetic.

Sources
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
/Users/pat/code/ACE/scripts/research/explain_ace_mechanics.py
Version8 physical reading: X drive-command deviation, M shaft-speed deviation, Y pressure-rise deviation. Normalized toy values, zero disturbances. These exact arithmetic results are a teaching example, not measured compressor data.

# Slide 8

Exact score for M context±1 is160/387≈0.413436693. Y context±2 is640/1467≈0.436264485. Their sum is≈0.849701178. Under do(M=±1) the clamped M mechanism contributes zero, leaving only Y≈0.413436693. Zero contexts give zero covariance and zero score. X+1 tiesX−1; first-maximum tie rule choosesX−1. At that action the ensemble mean predictsM−2,Y−6. No oracle outcome is queried during scoring. The Y slope ensemble is evaluated at the simulated mean parent; it is not a member-paired composed posterior.

Sources
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Version8 physical reading: X drive-command deviation, M shaft-speed deviation, Y pressure-rise deviation. Normalized toy values, zero disturbances. These exact arithmetic results are a teaching example, not measured compressor data.

# Slide 9

Illustrative oracle row isX−1,M−2.5,Y−7.5. Eligible pairs M:(−1,−2.5),Y:(−2.5,−7.5); clamped X supplies no natural root-distribution update. For transparent arithmetic use one gradient step on half squared error, with etaM1/2 and etaY2/25. M slopes1,2,3 become1.75,2.25,2.75. Y slopes2,3,4 become2.5,3,3.5. At X−1, updated mean predictsM−2.25 andY−6.75. Signed final error drops from1.5 to.75 on this same illustrative row. This is not held-out improvement or production training: production uses neural heads, Adam and member-specific masks. Response budget becomes0. Do not associate this made-up row with the real archived trace.

Sources
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json
Version8 physical reading: X drive-command deviation, M shaft-speed deviation, Y pressure-rise deviation. Normalized toy values, zero disturbances. These exact arithmetic results are a teaching example, not measured compressor data.

# Slide 10

Exact archived DAG for smallest seed5000, not chosen for favorable performance. Edges point parent to child. HighlightX7 first chosen intervention and its descendant edges. FirstactiondoX7=3.857086181640625 buys50responses. Eachrow supplieseligible natural-mechanism labels using observedparents. Fullcampaign32interventionbatches1600responses plus400observationalresponses. All20DAGs in thecomparison shareonefive-layergenerator, not20independentgraphfamilies. Rawrows/candidatescores notretained for thisaction.

Sources
results/research_pev_shift30_mean_confirmation/shift30/pev/seed_5000/system.json
results/research_pev_shift30_mean_confirmation/shift30/pev/seed_5000/trajectory.csv
/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/recorded_curves_2026-10-09.json

# Slide 11

All20shift30systemsseeds5000–5019. Arithmeticmean noise-free feasible nonroot mechanism MSE, measured-parent prediction. Full32checkpoints shown atleft; samecurves4–32zoomatright. FinalACE.0154302049,random.0438980071,64.85%lowerratioofmeans,19/20pairedwins. Secondarycontrastp.00750. Prespecifiedcoveragecontrastfails p.05108 andnaivevariancescoringunresolved p.62176. Earlyrandomadvantage remains visible. No Bresults,confidenceband,newfit ornewhypothesis.

Sources
/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/recorded_curves_2026-10-09.json

# Slide 12

Three bars use counts of individual queries: measured PEV500 intervention responses at first mean-curve crossing versus random1400, and analytic29255197uniform random joint-grid draws for95%expected coverage. The third bar has a DIFFERENT endpoint/system/distribution, is not an empirical third arm, and cannot support a ratio of ACE savings relative to grid search. The shared linear axis deliberately leaves the measured bars nearly invisible; exact counts remain in data labels and the measured comparison is restated below. Measured values are retrospective first crossings of group mean curves across20synthetic30-node systems, MSEthreshold0.043898007078491626,50interventionresponses perbatch. Including observations, prefixes620and1760; both full campaigns ran2000totalresponses. The random arm is NonLeafRandomPolicy: uniform choice among graph-eligible nonleaf targets and random.uniform values, with the same SCM ensemble learner as PEV. It is not a deterministic direct policy; PEV is the direct SCM-scored policy. Third bar: ten five-valued inputs,9765625jointentries,uniform independent joint draws withreplacement. One queried configuration yields one deterministic lookup response in this analytic illustration. Other associative learners need not enumerate a grid. No new experiment has been run.
Sources
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/recorded_curves_2026-10-09.json
/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/SCM_free_grid_comparison.json
/Users/pat/code/ACE/scripts/research/persistent_scm.py:campaign
/Users/pat/code/ACE/baselines.py:RandomPolicy,NonLeafRandomPolicy