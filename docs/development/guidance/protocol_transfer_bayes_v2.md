# Bayesian source-library transfer confirmation v2

Frozen 26 September 2026 before fresh target seeds 4000–4019. This numerical mechanism screen tests whether a full-data Bayesian mixture can preserve family-switch transfer while avoiding the coefficient-change harm in the prior retrieval study. It does not use a language model, neural encoder, or active intervention policy.

Keep the v1 learned source library fixed: four families, ten independent source tasks per family, 64 samples per task, 2,560 source examples total. Each target system has 30 local mechanisms. Evaluate both family changes and within-family coefficient shifts, each at changed-node counts k=1,3,10. All methods see identical target examples, with budgets 120, 200, and 400. The compared methods are scratch, old-mechanism warm start, passive-assay nearest source retrieval, passive-assay predictive mixture, and a new full-data Bayesian source mixture. The new arm integrates a Gaussian linear likelihood under each old/source-centered prior and weights posterior fits by marginal evidence over **all acquired target data**. Prior model weights are 0.5 for the old mechanism and 0.125 for each of four learned source prototypes. No method uses changed-node labels or test outcomes for selection.

Development seeds 100–111 were scored once. At 200 target examples, changed-node MSE for warm/retrieval/Bayesian mixture was:

| Change | k=1 | k=3 | k=10 |
| --- | --- | --- | --- |
| Family | .2497/.0441/.0503 | .1800/.0216/.0236 | .2139/.0430/.0397 |
| Coefficient | .0539/.0570/.0543 | .0992/.1160/.0950 | .0622/.0889/.0616 |

The new arm substantially reduces retrieval's coefficient-change penalty in development while retaining a large family-change benefit. At 200 examples, its untouched-node MSE versus warm is .0256/.0250 (family k1), .0296/.0290 (family k3), .0274/.0271 (family k10), .0282/.0280 (coefficient k1), .0265/.0260 (coefficient k3), and .0263/.0254 (coefficient k10). Those are small mean changes; a fresh-system uncertainty interval is needed for any noninferiority claim.

Fresh confirmation: 20 independent target systems, seeds 4000–4019, all six change settings per seed, on CURC account `ucb736_asc1`. One CPU job per seed computes all six settings and all five arms; output root `/scratch/alpine/paco0228/ACE/results/research_transfer_bayes_v2`. The source library is reused across systems and its 2,560-example cost is reported separately. Primary comparison at budget 200: family changed-node MSE of Bayesian mixture versus warm and retrieval; coefficient changed-node MSE versus warm and retrieval. Also report all-node and untouched-node MSE at 120/200/400. Promotion requires a meaningful family benefit over warm, no supported >5% untouched-node deterioration, and no supported coefficient-change harm relative to warm on fresh systems. Use paired system-level intervals; inconclusive is acceptable. Do not select favorable k or budget after seeing results.

Every setting must have a schema-v3 receipt with 15 finite rows, a consistent source-library hash, and verified system/metrics SHA-256. Preserve all per-seed outcomes, logs, submission manifest, job ids, source revision, and CURC/local checksum parity. No Azure or other closed-source model API calls.
