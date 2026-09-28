# Bounded-degree connected-SCM development gate

The [frozen protocol](../../docs/development/guidance/protocol_connected_binary_tree_dev_2026-09-28.json) tests a distinct connected graph structure after the fanout result. Motif j>0 takes one parent from motif floor((j-1)/2) and one fresh root. No motif child is a direct parent of more than two later motifs; the fanout graph used the first child as a direct parent of every later motif. The learner, action menu, cost rule, sealed evaluator, and eight numerical controls are otherwise the same. Implementation revision: `9c5679571e0bbcd2e8fcc5f5c268ff3f4d92692e`.

Three **development** systems (seeds 400–402, N=30, k=10, root SD 0.15, actuator penalty 4, budget 400) give the following final feasible-motif MSE (lower is better):

| Method | Seed 400 | Seed 401 | Seed 402 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Random single | 0.16483 | 0.06389 | 0.32143 | 0.18338 |
| Coverage single | 0.09704 | 0.07617 | 0.05651 | 0.07658 |
| Random pair | 0.09949 | 0.17403 | 0.04090 | 0.10481 |
| **Coverage pair** | **0.02997** | **0.06430** | **0.00826** | **0.03418** |
| Posterior-risk pair | 0.03559 | 0.09161 | 0.01740 | 0.04820 |
| Balanced-risk pair | 0.05168 | 0.03075 | 0.05880 | 0.04708 |
| Hub-coverage pair | 0.09462 | 0.04853 | 0.03528 | 0.05948 |
| Hub-random pair | 0.08227 | 0.02486 | 0.04085 | 0.04933 |

Coverage pairs beat posterior-risk pairs in **all three** systems and have lower mean error. Coverage pairs also beat coverage singles in all three under the same total cost budget. Pair policies spent 360 cost units on five batches (40 environment samples, 80 actuator uses); the coverage-single arm spent 400 on ten batches (80 samples, 80 actuator uses). This suggests that joint interventions may be useful in this generator, while the current posterior-risk controller has not shown a robust within-menu advantage. Three systems cannot establish a general pair-action benefit.

Each cell contains `metrics.csv`, `actions.csv`, `system.json`, and a SHA-256 receipt. The suite receipt hashes the frozen protocol and summary. Independent validation checked the pinned revision, graph topology and bounded outdegree, topological edges, natural-child masking, eight arms, exact per-action query/sample/actuator costs, finite scores, and every file hash. All runs used the local numerical simulator, with no CURC job or closed-model API. The shared CURC connection remained healthy and no ACE jobs were queued at the check.

**Development decision:** the frozen within-menu gate failed. Stop expanding this numerical posterior-risk selector on the current simulator. Preserve the pair-action finding as a hypothesis only; any future B work needs a stronger controller or a separate question, a prespecified graph family, and new confirmation systems. There is no foundation-model result here.
