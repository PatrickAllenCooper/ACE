# Propagated Epistemic Variance (PEV): a query-free acquisition for active mechanism estimation

*Design note, 17 Sept 2026. Status: implemented (`baselines.py`,
`experiments/heterogeneous_scm.py`, both baseline runners), smoke-tested,
preliminary local runs in progress; CURC ladders in
`jobs/curc_submit_pev_ladder.sh` and `jobs/curc_submit_proposer_ladder.sh`.*

## 1. What the audit established, and what a scale-surviving mechanism must do

The corrected results (`metric_audit_2026-09-10.md`, "Outcome") say three
things about the *only* mechanism that has ever won in this repo:

1. **Where ACE wins (5 nodes), it wins by probing a nonlinear mechanism's input
   domain.** 3–4× lower error on the collider X3 = 0.5X1 − X2 + sin(X2); a tie on
   the quadratic and the linear node. The value is in *which mechanism's inputs
   get covered where*, not in the LM per se.
2. **At N ≥ 15 nothing beats Random — not ACE, not ACE-w/o-DPO, not max-variance,
   not "Bayesian OED".** Two reasons. The LargeScaleSCM family is homogeneous
   (coefficients U(0.3, 0.7), a 0.2·sin on every fifth node), so uniform
   interventions at U(−5, 5) are already near-optimal coverage of the U(−4, 4)
   evaluation domain. And every adaptive method paid for its acquisition in
   *oracle queries*: ACE's lookahead trains a cloned learner on real samples
   for every candidate (4–5× query multiplier); the repo's "Bayesian OED" is
   the same one-step oracle lookahead with random candidates, scored on the
   observational loss. Under matched total queries those are 4–5× data deficits.
3. **An intervention on an upstream node generates a training sample for every
   descendant mechanism at once**, and the 25-step campaign at 30 nodes cannot
   even intervene on each node once. No existing policy used this: Random and
   round-robin spend 1/6 of their interventions on leaves, which teach nothing
   about any mechanism.

A mechanism that survives at scale therefore has to (a) cost no oracle queries
to score candidates, (b) target epistemic uncertainty *on the evaluation
domain*, not observational fit, (c) account for propagation to descendants,
and (d) not need an LM to read an O(N+E) prompt per step.

## 2. The acquisition

**Student.** K = 5 independently initialised `StudentSCM`s (`EnsembleStudentSCM`),
each trained on an independent Bernoulli(0.8) bootstrap of every batch
(`EnsembleLearner`). The oracle is queried once per observational refresh and
the batch is shared, so environment-sample accounting is identical to the
single-student baselines. Predictions are the member mean; the members'
disagreement is the epistemic variance s²_c(x) of mechanism f_c at input x.
Roots keep the learnable mean plus an empirical spread used only for simulation.

**Score.** For a candidate do(X_j = v),

    score(j, v) = Σ_{c ∈ desc(j)}  E_{pa_c ~ q_j,v}[ s²_c(pa_c) ]

where q_j,v is the distribution of c's parent context *induced by the
intervention, simulated with the student itself*: roots from N(μ̂, σ̂), every
other node from the ensemble-mean mechanism, X_j clamped to v
(`EnsembleStudentSCM.simulate`). Candidates are every node with at least one
descendant × an 11-point value grid over [−5, 5], jittered within the bin.
Zero oracle queries; O(C·S·N·K) small-MLP forwards, fully batched (5 nodes:
0.1 s/step; 30 nodes: ≈1 s/step on a laptop CPU). ε-greedy with ε = 0.05.

**Why this is principled.** The learner's objective on the evaluation domain
D = U(−4, 4)^{|pa_c|} is the integrated epistemic variance
IEV = Σ_c ∫_D s²_c(x) dx (for an ensemble read as a posterior over
mechanisms, the expected squared error at x is s²_c(x) up to the aleatoric
floor). Observing f_c at inputs X reduces IEV by an amount that, for a GP-like
model, is Σ_{x∈X} s²_c(x)² / (s²_c(x) + σ²) plus the reduction that spreads to
neighbouring inputs through the kernel — monotone in s²_c(x). The greedy rule
that maximises the first-order term is: send the observations to where s² is
largest. An intervention do(X_j = v) can only place observations at the
contexts it induces, for the descendants it reaches; the score above is
exactly that first-order expected reduction, with the induced contexts
estimated by the student. It is the model-based, query-free analogue of the
oracle lookahead ACE and "BOED" ran, targeted at the evaluation domain instead
of the observational one. Assumptions stated in the paper: (i) ensemble
disagreement is a usable epistemic proxy (deep-ensemble literature); (ii)
student-simulated contexts are accurate enough once the student is past the
first few steps — early on s² is large everywhere and the choice matters
little; (iii) greedy one-step.

**What it deliberately does not do.** No per-step LM call and no prompt: the
graph enters only through descendant sets and parent contexts, so cost is
linear in N. No preference optimisation. The LM, if it returns, returns as a
*prior over mechanism forms* (Section 5), not as a target proposer.

## 2b. What the first local runs taught, and the scoring that replaced v1

The first implementation scored a candidate by the epistemic variance its
induced contexts *visit* (naive uncertainty sampling). On the 5-node SCM it
spent 63% of its interventions on X2 — the collider's parent, the "right"
target — and X3's error did not move (0.61 vs 0.62 for Random with the same
student), while X5 was starved (0.23 vs 0.09) because X4 got 12% of the
probes. The disagreement on X3's slices is not reducible by data for a
16-unit student, and a rule that goes where variance is highest chases it
forever. This is the textbook failure of uncertainty sampling under
misspecification.

The default scoring is now the expected *reduction* over the evaluation
domain (Cohn 1996; the ALC / integrated-variance-reduction criterion), which
the ensemble provides for free through its across-member covariance:

    score(j, v) = Σ_c mean_x [ mean_r C_c(x, r)² / (s_c²(x) + σ_c²) ],

x the induced contexts, r a fresh sample of c's evaluation domain, C_c the
ensemble covariance of predictions, σ_c² the ensemble mean's running residual
variance. Disagreement that does not co-vary with the domain scores low. The
naive rule is kept as the `pev_var` ablation.

The same runs exposed the learner confound recorded in the audit addendum:
the baseline framework's student was a (16,) MLP at 50 epochs; ACE's is
(64, 64) at 100. Every arm of the ladder now runs ACE's student.

## 2c. First matched-student result (local, 17 Sept; 3 seeds x 40 episodes)

Every arm on ACE's (64,64)/100-epoch student. End-of-campaign non-root
broad-range MSE per node: ACE 0.165 (0.129 on the paired seeds), round-robin
0.161, Random 0.223, Random + ensemble student 0.172, **PEV 0.045**, PEV
naive-variance ablation 0.039. PEV vs Random with the same ensemble student
p = 0.012; vs ACE p <= 0.001. Per node, PEV gets the collider (0.079 vs ACE
0.155 vs round-robin 0.401) *and* the quadratic (0.046 vs ACE 0.329 vs
round-robin 0.072); ACE had traded one for the other. Allocation: 44% of
interventions on X1 (two descendants), 47% on X2, 10% on X4 -- and X5 is
still the best of any arm, because the X4 values are chosen where the
quadratic is uncertain over the evaluation domain rather than near the
observational mean. Zero candidate-probe queries.

The naive-variance ablation matches PEV here: with an adequate student the
collider's disagreement is reducible, so uncertainty sampling works; its
failure in 2b was under the 16-unit student. Whether the reduction-based
score matters is a question for the heterogeneous family and N = 30.

Data: results/local_matched_prelim_20260917/. Superseded by the CURC ladder.

## 3. The ladder — what "beating the new baseline" means

The corrected baselines are the bar (`results/audit_reruns/`, end-of-campaign
non-root broad-range MSE per node): Random 0.36 / round-robin 0.34 at 5 nodes;
round-robin 0.078, Random 0.092 at 30 nodes; Random 0.059 / 0.090 / 0.118 at
N = 15 / 30 / 50. ACE's 5-node 0.14 is the second bar.

Every arm below runs in the same corrected framework
(`jobs/curc_submit_pev_ladder.sh`):

| Arm | Isolates |
|---|---|
| `random_ens` | the ensemble student alone (K=5 vs single MLP) |
| `round_robin_ens` | same, with the audit's best 30-node policy |
| `pev` | the acquisition, given the ensemble student |
| `random`, `round_robin` on the hetero family | the single-student reference there |

Suites: 5-node (171 ep, 10 seeds), LargeScaleSCM-30 (150 ep, 5 seeds),
HeterogeneousSCM-30 (150 ep, 5 seeds), scaling N=15/50 on both families
(40 ep). Success criteria, stated in advance:

- PEV < random_ens with p < 0.05 at 30 nodes on the hetero family (the setting
  where selection can matter) — the primary claim.
- PEV ≤ Random on LargeScaleSCM at 15/30/50 (does not lose where nothing can win).
- At 5 nodes, PEV within noise of ACE's 0.14 or better, at ≤ 1/4 of ACE's
  total queries — the "same win without the LM or the lookahead" claim.
- random_ens vs random tells us how much of any gain is the ensemble student.

And, in ACE's own pipeline (`jobs/curc_submit_proposer_ladder.sh`, GPU):
`--proposer random` and `--proposer heuristic` with `--no_dpo`, plus the
calibration suite's `none` row (`--proposer lm --no_dpo`). This decides whether
the LM contributed anything at 5 nodes beyond the loss-guided heuristic that
already serves as its teacher fallback.

## 4. The heterogeneous family

`experiments/heterogeneous_scm.py`: same hierarchical graph generator; each
non-root mechanism drawn from {linear, 0.25z², 2sin(1.2z), 3tanh(z), |z|,
z + 0.3·x₁x₂} with z = Σ c_p x_p. Form assignment and coefficients are fixed
from `coeff_seed`, so an ACE-side adapter reproduces the system exactly (as
`LargeScaleSCM` now does). Quadratic and product nodes extrapolate poorly for a
16-unit ReLU student trained mostly on observational data — the heterogeneity
that gives an acquisition rule something to buy.

## 5. Where an LM still fits — and the architecture beyond PEV

PEV is the principled floor: it uses nothing but the student. Three
extensions, in the order I would build them, each attacking a failure the
audit exposed:

1. **Amortised critic** (kills the remaining compute, not queries): train a
   network offline over synthetic SCM families to predict the PEV score (or
   realised loss reduction) from cheap state features — per-node residuals,
   per-parent coverage histograms, descendant counts. Per step it replaces
   the simulation. DAD / RL-BOED applied to mechanism estimation.
2. **LM as prior over mechanism forms, hypothesis-discriminating design**: the
   LM proposes a menu of forms per node from names and structure (exactly the
   menu of Section 4 on a semantically named family); each is fitted by least
   squares; the intervention chosen maximises disagreement among the top
   hypotheses (Box–Hill). This uses the LM for what it knows — plausible
   functional forms — and yields symbolic mechanisms. It is the natural
   successor to ACE's target prior, and the audit's per-node result (the win
   was on the one nonlinear collider) is its motivation.
3. **In-context mechanism estimator** (TabPFN/AVICI-style transformer over the
   interventional dataset): removes the per-episode reset and per-step gradient
   training, and provides calibrated uncertainty for the PEV score directly.
   Largest build; the right long-term learner.

The paper that this supports: "Query-free uncertainty-directed intervention
design for mechanism estimation", with the audit's negative results (LM
target prior does not scale; DPO hurts; oracle-lookahead BOED loses to
Random; observational validation hides interventional failure) as the
motivation and PEV plus the hetero family as the contribution — *if* the
ladder clears the criteria above. If it does not, the honest paper is the
evaluation study alone.
