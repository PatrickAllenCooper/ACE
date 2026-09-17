# Reframed paper plan (17 Sept 2026)

The AISTATS draft's claims are gone (`metric_audit_2026-09-10.md`). This is
the plan for the paper that the corrected data and the post-audit ladder can
support. Two candidate framings; the ladder decides between them. Either way,
the manuscript is a rewrite of `paper/aistats_ace_2027/`, not an edit, and it
does not go out on 6 Oct unless Framing A clears its bar with days to spare.

## The two framings

**A. Method paper** — *"Query-free uncertainty-directed intervention design for
active mechanism estimation"*. Contribution: PEV (ensemble student + expected
variance reduction over the evaluation domain, propagated through the
student to descendants, zero oracle cost); the heterogeneous family; a
controlled negative result on LM target priors and DPO calibration. Requires
the ladder to clear the pre-registered criteria in `pev_design.md` §3
(primary: PEV beats Random with the same ensemble student at 30 nodes on the
hetero family, p < 0.05; does not lose on LargeScaleSCM; matches or beats
ACE at 5 nodes at ≤ 1/4 the queries).

**B. Evaluation paper** — *"What does an LM prior buy in active mechanism
estimation? A matched-learner, matched-query study"*. Contribution: the
audit's four confounds as a methodology finding (evaluator, validation
distribution, non-stationary simulator, unmatched learner), the corrected
comparisons, the observational-vs-interventional validation gap, "BOED" as
oracle lookahead, DPO as harm. Honest, useful, and a harder sell at AISTATS;
natural home TMLR / CLeaR / a workshop. This is the fallback if A fails.

Both share Sections 2–4 below; A adds Section 5.

## Section plan

1. **Introduction.** The question: does an LM prior over intervention targets
   help estimate mechanisms? Prior versions of this work said yes by 3×; a
   metric audit showed the comparison was invalid four ways. What is true.
2. **Problem.** SCM, mechanism estimation objective *on the interventional
   (broad-range) domain*; campaigns with per-episode reset; total-query
   accounting (executed + candidate probes + observational). State the four
   things that must be matched and were not: evaluator, validation
   distribution, simulator stationarity, learner.
3. **Methods compared.** ACE (LM + oracle lookahead + DPO) and its ladder
   (w/o DPO; random / heuristic proposer with the same scaffolding); passive
   (Random, round-robin); oracle-lookahead "BOED" and max-variance (with
   their probe costs); PEV and its ablations (naive variance; ensemble +
   Random). All on one student.
4. **Results.**
   4.1 5-node: matched-student table. ACE vs Random/round-robin with the same
       student and the same queries; the proposer ladder; per-node (X3) story.
   4.2 30-node and scaling on LargeScaleSCM: nothing beats Random; DPO hurts
       monotonically in N; BOED loses; PEV must not lose.
   4.3 Heterogeneous family: where selection can matter; PEV vs random_ens vs
       round-robin; per-form breakdown (quadratic/product vs linear).
   4.4 Query accounting: per-method multipliers; PEV at 1×.
5. **(A only) PEV.** Derivation (Cohn/ALC with ensemble covariance), the
   propagation term, why naive variance fails (the X3 diagnosis), cost.
6. **Analysis.** Observational validation hides interventional failure (the
   4–8× gap, the drift). Homogeneous families defeat every acquisition
   including BOED — a no-free-lunch note for this benchmark genre. What the LM
   actually contributed (from the proposer ladder).
7. **Related work / limitations / conclusion.** Synthetic families only; the
   student class; greedy one-step; K = 5 covariance rank.

## Tables and figures (all produced by `scripts/analysis/aggregate_metric_audit.py`)

- T1: 5-node matched-student, end-of-campaign non-root MSE (mean ± sd, N
  seeds), best-over-steps, total queries. Rows: ACE, ACE-w/o-DPO, ACE(random
  proposer), ACE(heuristic proposer), Random, round-robin, random_ens, PEV.
- T2: 30-node LargeScaleSCM, same columns. Rows: ACE, ACE-w/o-DPO, Random,
  round-robin, BOED (audit, small student, marked), random_ens, PEV, pev_var.
- T3: HeterogeneousSCM-30 and N=15/50 on both families.
- F1: learning curves within a campaign (step 0→24), ACE / Random / PEV.
- F2: per-node error at 5 nodes (X2/X3/X5) by method — the collider story.
- F3: observational vs broad-range error over training for a passive
  baseline — the validation-gap figure.
- F4: query multiplier vs error, both scales.
- Supplement: the audit itself (the four confounds, with the before/after
  numbers), the pre-audit tables restated, DPO calibration suite, node-
  importance ablation, per-seed tables, action distributions.

## What is still needed before writing

| Item | Where | Status |
|---|---|---|
| Matched-student passive baselines, all suites | `curc_submit_pev_ladder.sh` | local 5-node running; CURC not yet submitted |
| PEV / pev_var / random_ens, all suites | same | same |
| ACE proposer ladder (random, heuristic), 5-node | `curc_submit_proposer_ladder.sh` | not yet submitted |
| ACE-w/o-DPO 5-node (`MODES=none`) | `curc_submit_dpo_alternatives.sh` | resubmit with gpu-long |
| ACE on HeterogeneousSCM-30 (LM arm, `--family hetero`) | needs a job script; 5 seeds x 24h GPU | not started |
| BOED 5-node per-seed | `curc_submit_metric_audit_reruns.sh SUITES=boed5` | resubmit |
| Ranking s1011, node-importance x6 | gpu-long resubmits | resubmit |

## Decision rule

After the CURC ladder lands (≈ 2 days of queue time): run
`aggregate_metric_audit.py` over `pev_ladder/` with the ACE arms; apply the
§3 criteria. Clear → write Framing A (needs ~10 days; AISTATS is out of reach
unless the ladder lands by ~25 Sept). Not clear → write Framing B for TMLR /
CLeaR; keep PEV as a negative-result section or drop it.
