# Metric audit, 10 Sept 2026 — every ACE-vs-baseline number is invalid as printed

**Status: BLOCKING.** No ACE-vs-baseline comparison in any paper version
(NeurIPS 2026, ICLR 2027, AISTATS 2027 draft) may be cited until the re-runs
described at the bottom have landed and the tables have been restated on one
metric. ACE-vs-ACE comparisons (calibration rules, node-importance, ACE-w/o-DPO,
seed expansion vs. original seeds) are unaffected.

Found while preparing the AISTATS reviewer rounds. Three independent confounds.

## 1. Two evaluators, two root weightings (both scales)

| | ACE (`ace_experiments.ScientificCritic.evaluate_mechanisms_detailed`) | Baselines (`baselines.ScientificCritic.evaluate`) |
|---|---|---|
| Root nodes | MSE of student mean vs observational sample, **weight 0.2** | same, **weight 1.0** |
| Non-roots | parents ~ U(-4,4), truth from `oracle.mechanisms` | teacher-forced on an observational held-out sample |
| `total_loss` | 0.2·Σroots + Σnon-roots | Σ all nodes |

Root MSE is irreducible (~1.0 per root; both students predict the mean) and is
identical in both files node-by-node (e.g. `results/curc_30node_baselines/random/seed_42/results.csv`
vs the ACE seed-42 `node_losses.csv`: X1 0.948/0.942, X2 0.938/0.939, X3 1.116/1.113).
ACE's `total_loss` is reproduced exactly by 0.2·(X1+X4)+X2+X3+X5 at 5 nodes
(roots are X1 and X4; X2 has parent X1) and by 0.2·(X1..X5)+rest at 30 nodes.
The 0.2 weight is reward shaping (`w = 0.2 if not student_scm.get_parents(node)`)
that leaked into the reported metric. The per-node `loss_*` columns are
unweighted in both files, so this confound alone is correctable from saved data.

## 2. Different non-root validation distributions (both scales)

ACE scores non-root mechanisms on broad-range parent contexts; the baselines on
an observational sample. This is **not** correctable from saved data: no student
checkpoints exist anywhere under `results/`. ACE also never logs the
observational score (`evaluate_model_detailed` exists but is never called).

## 3. The ≥15-node baselines ran on a non-stationary system

`experiments/large_scale_scm.py::LargeScaleSCM.generate` drew
`coef = np.random.uniform(0.3, 0.7)` **inside `generate()`, per parent, per call**.
Empirically, E[X10 | do(X1=4)] over five calls with identical torch noise:
2.758, 2.716, 1.988, 2.667, 2.182. ACE's `--large_scale` adapter
(`_LargeGroundTruthSCM`, introduced in a7422c3, 2026-04-08, "Fix three broken
experiments", Cursor-made) draws the coefficients once from
`np.random.seed(args.seed)`. So ACE learned a fixed SCM while Random, Round-Robin,
Max-Variance and Bayesian OED chased a moving target. The graphs *are* shared
(isolated-node fingerprint matches on all ten seeds: 42→∅, 123→X3, 456→X2,X4,
789→X1, 1011→X2,X5, 2024→X1, 2025→X5, 2026→X2); only the mechanisms differ.
Affects Table 2, every 30-node budget-fairness row, the scaling sweep's Random
arm at N=15/30/50, and the 30-node Bayesian-OED row. Invalid in both directions.

## What the tables look like with confound 1 corrected (2 and 3 still open)

Baselines re-scored from their own per-node columns on ACE's weighting.
These are NOT the final numbers — they still carry confounds 2 and 3 — but they
show the printed margins are artifacts.

| Comparison | As printed | One definition |
|---|---|---|
| Table 1, 5-node, final | ACE 0.61 med. vs Random 2.09–2.21 (“69%”) | ACE 0.61 vs Random 0.50, RR 0.48, MaxVar 0.46, PPO 0.51 |
| Table 2, 30-node, best | ACE 1.95 vs plateau 5.86 (“3×”) | ACE 1.95 vs Random 1.76 (final 4.13 vs 1.83) |
| Scaling, per-node best | ACE 0.05–0.14 vs Random 0.15–0.19 | N=15 0.057 vs 0.049; N=30 0.097 vs 0.059; N=50 0.112 vs 0.057 |
| Budget-fairness, per-node best | ACE wins 76% / 43–50% | 5-node 0.094 vs 0.086; 30-node 0.090 vs 0.059 |
| Seed expansion 2022–2026 | “fall to the plateau” (5.69 vs 5.78) | 2.21±0.47 vs original 1.95±0.77, Welch p=0.64 — they replicate |

The expansion row was the one place the paper compared like with like
(unweighted vs unweighted), and it showed ACE at the plateau.
Non-root loss alone: 5-node final ACE 0.22 vs baselines 0.035–0.10; 30-node
best ACE 1.15 vs Random 0.76.

## Fix (code, done 10 Sept)

- `experiments/large_scale_scm.py`: coefficients frozen at construction;
  `coeff_seed=<seed>` reproduces ACE's adapter draw bit-for-bit; new
  `mechanisms()` is the single code path for `generate()`.
- `baselines.py`: `GroundTruthSCM.mechanisms()`; `evaluate_mechanisms_broadrange()`
  (line-for-line port of ACE's evaluator); `ScientificCritic.evaluate_broadrange()`;
  `run_baseline` logs `ace_total_loss` / `ace_loss_*` beside the existing columns.
- `scripts/runners/run_30node_baseline_seed.py`, `run_5node_baseline_seed.py`:
  frozen coefficients; dual per-step logging; `ace_min/final_total_loss` in summary.
- `ace_experiments.py`: logs the observational unweighted score
  (`obs_total_loss`, `obs_loss_*`) per step in `node_losses.csv`.

Equivalence tests (frozen `LargeScaleSCM` ≡ ACE adapter; ported evaluator ≡
ACE's) live in `scripts/analysis/test_metric_parity.py`.

## Re-runs required (all CPU except where noted)

Every baseline cell, re-run on the fixed-coefficient system with both scores logged:

1. 5-node Table 1: random / round_robin / max_variance / ppo × seeds 141,271,314,577,618 × 171 ep; plus seeds 42,123,456,789,1011 to pair with ACE's own seeds.
2. 5-node budget-fairness (query-matched) baselines, same methods, 5 seeds.
3. 30-node Table 2: random / round_robin / max_variance / bayesian_oed × seeds 42..1011 × 150 ep (BOED ≈ 8 h each), plus seeds 2022–2026 for the expansion controls.
4. 30-node budget-fairness baselines.
5. Scaling sweep Random at N=15/30/50 × 5 seeds.
6. 5-node Bayesian OED (N=3, `run_reviewer_experiments.py`) — still scores with `SCMLearner.evaluate`; needs the same dual logging before re-running.
7. Duffing baselines — separate system; not yet audited.

Then restate every table on ONE stated metric. Recommended primary: mean
non-root mechanism MSE on the broad-range set (what the paper is about; roots
are a constant for every method), with the observational score in the supplement.

## Do not

- Do not commit the Table 1 Welch-p edit in `paper/aistats_ace_2027/paper.tex`
  as a "fix": it tests cross-definition numbers.
- Do not run reviewer rounds on the current draft.
- Do not describe the seed expansion as a non-replication.
