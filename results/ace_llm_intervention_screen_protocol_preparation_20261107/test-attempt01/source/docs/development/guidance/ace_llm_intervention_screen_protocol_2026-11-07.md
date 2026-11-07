# Prospective first-screen identities and schedule

November 7, 2026. Pure proposal preparation, **not a scientific protocol freeze or
execution authorization**. No world descriptor, private coefficient, response,
model or training update is generated. Settings below are prospective and
unqualified; no outcomes were used to select them. Parent owns GPU10 closure and
later scientific admission, execution, optimization and resource decisions.

`llm_intervention_screen_protocol.py` provides `build_proposal(source_pins=...)`,
`validate_proposal(proposal, expected_source_pins=...)`, `public_design(...)` and
`assembly_recipe(...)`. They perform no I/O, random draws or execution. The source
map must contain the exact twelve protocol/assembly dependency paths and lowercase
SHA256 values. This admits declared dependencies, not actual executable bytes;
the parent must authenticate files/runtime independently. Validation rejects any
extra, missing or modified schedule member against this explicit proposal version.
An artifact hash preserves review bytes; it does not grant scientific freeze status.

There are 32 proposed `policy_train` IDs 202611070000–202611070031 and 24
`development` IDs 202611070100–202611070123. Confirmation reserves the proposed
scope/range `confirmation:[202611071000,202611072000)` without enumerating worlds,
choosing a sample size or assigning families/laws. Registry collision/admission
checks remain pending; neither declaration authorizes generation.

Four regimes cross matched/shifted edge coefficients and low/high **known noise
variance**. Learner intercept/edge means start at zero with identity covariance in
each independent head, exactly matching the current constructor. True intercepts
would be independent N(0,1); true edges would be N(0,1) for matched or N(1.5,1) for
shifted regimes. Each node's fixed noise variance is .25 (low) or 1 (high), also
supplied publicly to the learner. These are laws, not realized coefficients. No
posterior is initialized from private truth. Structural coefficients are stationary.

Each development regime contains chain, fork and collider graphs crossed with
uniform/heterogeneous costs: six identities, eight policy arms each. Graphs have
three dimensionless nodes: X→Y→Z, X→Y and X→Z, or X→Z and Y→Z. Target is Z. Cost
stratum pairs share graph/law settings but have distinct world identities/draws;
the actual paired policy comparison uses the same world across all eight arms.
This is not a matched-realization contrast between cost strata.

Each training regime has four graph blocks, each paired across both costs. The
base three graphs occur once; the fourth is chain/fork/collider/chain respectively
for matched-low/matched-high/shifted-low/shifted-high. Thus training graph counts
are 12/10/10, the closest possible paired allocation for 32 identities. All four
regimes have eight identities; costs have sixteen each. Do not describe the three
training graphs as exactly balanced. Development graphs have eight identities each.

Actions: observe all; set X to −1, 0 or 1; set Y to −1, 0 or 1; jointly set X=−1,
Y=1. Every acquisition measures X,Y,Z, so all nonclamped mechanisms have available
parents. X/Y intervention bounds are [−2,2]. Uniform actions cost one; heterogeneous
Y interventions and the joint action cost two, the other four cost one. Eight task
queries use these intervention settings, measure Z, cost one per training score,
and have exactly equal weights 1/8. Public scale is fixed at 2 and normalized-MSE
tolerance at .04. Public prose is constant across regimes and contains no realized
coefficient, regime label, law shift, world ID, seed, descriptor pin or private score.

Every arm begins with eight separately charged `observe_all` responses, cost eight,
then has 24 additional cost units. Identical world/initial keys pair the initial
values across arms; each arm still reserves them in its own journal. All arms use
the same task, menu, learner assumptions and initial schedule within a world. The
eight development arms are uniform random, coverage, local information, target
risk, numerical ranker, frozen LLM, SFT LLM and DPO LLM. Their acquisition/policy
namespaces are arm-specific; their initial namespace is shared. Weights must remain
fixed within each future episode. Private attainment cannot control stopping.

Training behavior uses the four numerical methods, two per method per regime and
eight globally. Let r be regime index, p graph-block index and c cost index. Behavior
index is (p+c+r) mod4; teacher base is (p+2c+2r) mod4. Snapshot i uses teacher
(base+i) mod4. All assignments precede outcomes. Each world sees all four teachers
at its nonterminal snapshots; every snapshot position has eight labels per teacher
and two worlds per behavior/teacher combination. Each behavior has four worlds of
each cost. Planning draws are explicitly proposed as 32.

Acquisition-cost milestones after initialization are 0/6/12/18, with all eight
candidates, and 24 with an empty candidate set for budget-terminal stop. First
crossings can occur at 0, 6–7, 12–13, 18–19 and exactly24. Remaining budget is at
least 24/17/11/5 before the four nonterminal snapshots, so every cost≤2 candidate
fits despite overshoot. A unit-cost action remains available to exhaust exactly24.
No snapshot is selected from outcomes; negative gains never become stop labels.

Each nonterminal snapshot has four independent keyed branch replicates per action
and eight separately charged structural training scores per replicate, shared
across candidates. Scores never update learners. Ties use the existing fixed
heuristic multiplier2 and minimum gap0; all pairs, negative gains and ties remain.
Per world: 128 branch responses, 128 scores, at most32 trajectory responses, five
supervised records and112 pairs. Across training: 160 records, 3,584 retained pairs,
up to1,024 trajectory cost, 5,120 branch cost and4,096 scoring cost: ceiling10,240
response-cost units. Development trajectory ceiling is24×8×32=6,144 units. These
are unallocated response costs, not CPU/GPU time or optimizer exposure.

Keys use the existing `ace-linear-gaussian-world-v1` canonical SHA256 rule. The
proposal lists parameter key addresses and response bindings. Initial/measurement/
policy/train_score/private_eval streams are distinct. Training counterfactuals use
measurement with the exact assembly snapshot namespace
`digest({namespace:request_namespace,snapshot_index:i})`, then replicate/request/
action/intervention/node keys. Scoring uses `train_score`; development has no
training feedback recipe. All training namespaces are scope/world-separated from
development/private evaluation. Private structural means require no random draw;
a private_eval key is reserved for any separately specified noisy evaluator.

`assembly_recipe` returns typed graph/menu/task/Snapshot objects and learner
constructor parameters, without constructing a learner. World, independent world
pin, learner and exclusive output remain absent runtime bindings. Representation
bounds are satisfied: 3≤8 nodes, 8≤32 actions, 8≤64 queries, 32≤80 history rows,
5≤16 snapshots, 4≤16 replicates and256≤4,096 feedback responses. The full proposal
contains private law and seed information; only `public_design` is a public export.

Required parent decisions remain: independent scientific adoption of these choices
(including graph imbalance and cost-pair semantics); namespace registry admission
and separately authorized realization; authenticated source/runtime/model service;
matched numerical/language optimization architecture/exposure/SFT/DPO/reference
schedule; charged paired-initial/public-only runner integration; private evaluator,
failure caps, contrasts/multiplicity/selection and later powered confirmation;
measured resource limits and separate scientific execution authorization.
`optimization_schedule` is explicitly pending, unallocated and has no settings.

Only new pure tests run, covering balances, identity/scope separation, all cost-state
crossings, typed interface closure, guarded absence of draws/responses, public law
separation and malformed inputs. Exact commands, CPU/wall receipt, source copies,
proposal and rejected fixture bytes are retained under
`results/ace_llm_intervention_screen_protocol_preparation_20261107` and exclusive
external `ACE-LLM-intervention-screen-protocol-preparation-20261107-01`.
