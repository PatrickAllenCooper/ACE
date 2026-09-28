# Balanced posterior-risk pair control (development)

This post hoc control reuses the three corrected connected-chain systems at each of k=3 and k=10 (seeds 200–202, N=30, root SD 0.15, actuator penalty 4, cost budget 400). The frozen [protocol](../../docs/development/guidance/protocol_connected_balanced_dev_2026-09-28.json) gives the learner's posterior-risk selector the same pair action menu and cost as the original `risk_pair` arm, but restricts every decision to a motif with the fewest previous acquired batches. Its initial acquisition and padding RNG seeds are coupled to `risk_pair`; the two streams may diverge after their policies choose different actions. The policy sees student-predictive contexts and posterior covariance, never the true coefficients or sealed evaluation panel.

| Motifs | Balanced pair | Unconstrained risk pair | Fixed-order coverage pair | Coverage single |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.02508 | 0.04720 | **0.02210** | 0.06975 |
| 10 | **0.03799** | 0.04211 | 0.13774 | 0.05240 |

Entries are mean feasible-motif MSE over **three reused systems**; lower is better. At k=3, balanced pair improves risk on two systems and loses badly on one; pair coverage still has lower mean. At k=10, balanced and unconstrained risk are identical on two systems and balanced improves one. Coverage single beats balanced pair on one of the three k=10 systems. These are small, selected development comparisons, not independent confirmation or evidence of a foundation-model contribution.

`complete.json` hashes 36 metric rows and 240 action rows and records the protocol hash. The runner checked exact archived action parity for every old arm, metric parity within 1e-10, equal paired action costs, and that balanced motif visit counts differ by at most one. Run `python3 scripts/research/connected_balanced_dev.py --output results/local_connected_balanced_dev_20260928` to reproduce. No CURC job or closed-model API was used.

**Decision:** do not launch a fresh seed sweep on the same connected-chain generator. A distinct graph structure, a frozen adaptive coverage comparator, and confirmation systems remain necessary before B can be promoted.
