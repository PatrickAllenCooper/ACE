# Fanout connected-motif development screen

The frozen [protocol](../../docs/development/guidance/protocol_connected_fanout_dev_2026-09-28.json) replaces the chain with a shared-parent fanout: motif 0's child is one parent of every later interaction motif. The other parent is a fresh root. This is a different connected DAG, not a new seed of the chain. The exact learner, action menu, costs, posterior-risk scoring, natural-child mask, and sealed feasible-motif evaluator remain the same. No language model is involved.

Three development systems each at k=3 and k=10 used N=30, root SD 0.15, actuator penalty 4, and cost budget 400. Mean **final feasible-motif MSE** (lower is better):

| Motifs | Random single | Coverage single | Random pair | Coverage pair | Risk pair | Balanced risk pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.05593 | 0.06291 | 0.27119 | 0.21590 | **0.01370** | 0.02882 |
| 10 | 0.02011 | 0.22509 | 0.10169 | 0.15392 | **0.01034** | 0.04015 |

The risk pair policy beats coverage pair in all six development cells. It does **not** beat every pair control: random pair is better for seed 301 at k=3 (0.00140 versus risk 0.01773). At k=10, random single beats the balanced pair on two of three systems. Full system-level values are in `summary.csv`.

All pair arms spent 360 cost units (five batches of eight samples at cost nine per sample); single arms spend according to their lower actuator cost. In every risk-pair cell, three of five actions targeted motif 0, the shared upstream mechanism, and two of five targeted downstream motifs. The balanced policy cannot revisit motif 0 this often and has higher mean error, especially at k=10. This is a mechanistic clue, not proof that upstream concentration generalizes; it also shows why the earlier chain's balanced-coverage comparison alone was insufficient.

The six cell receipts include complete metrics, actions, system graph, exact source revision `4cc17875ab11e40c334cabe75ad41c5d2d1717e3`, and SHA-256 hashes. The suite receipt includes the frozen protocol hash and summary hash. Validation checked graph topology, topological edge order, downstream natural-label masking when motif 0's child is intervened on, exact action and budget accounting, and every file hash. All results were generated locally; there were no CURC jobs and no closed-model API calls.

**Next gate:** expand the *development* screen to additional independently generated fanout systems with the same frozen settings and all six arms. Do not call this a confirmation or promote B until a prespecified fresh-system comparison survives random pairs, strong coverage, and a distinct graph family.
