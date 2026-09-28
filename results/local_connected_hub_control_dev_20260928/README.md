# Hub-first pair controls on fanout development systems

The [frozen control protocol](../../docs/development/guidance/protocol_connected_hub_control_dev_2026-09-28.json) tests whether the fanout risk policy's apparent advantage is explained by its repeated selection of the shared upstream motif. Implementation revision: `b207bbf5a8c54bf73cf04f8bac96aed538d6ee6d`. It adds two **nonadaptive** pair policies to the 12 already scored fanout cells (seeds 300–305, k=3/10, N=30, root SD 0.15, actuator penalty 4, budget 400):

- `hub_coverage_pair`: three motif-0 actions using distinct intervention corners, then one action each on downstream motifs 1 and k-1.
- `hub_random_pair`: the same three motif-0 actions, then two distinct downstream motifs from a public, seed-fixed permutation.

Both use the same last-two intervention corners, execute exactly five batches of eight samples, use 80 actuator operations, and spend exactly 360 cost units. They see no acquired outcomes when choosing actions. Their initial acquisition and padding RNG seeds match the risk arm; the streams can diverge because risk scores simulated candidates and chooses different actions.

Final feasible-motif MSE over **six reused systems per k** (lower is better):

| Motifs | Risk pair | Hub coverage pair | Hub random pair | Risk wins vs coverage | Risk wins vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | **0.00995** | 0.02677 | 0.02390 | 5/6 | 5/6 |
| 10 | **0.01137** | 0.02203 | 0.02401 | 5/6 | 4/6 |

The risk policy has lower mean error than either fixed schedule, but on **two systems at each k**, at least one predeclared static schedule is better. Choosing the better static schedule separately after seeing each system's score would be an oracle comparison and is not a deployable policy. The full per-system results and all action choices are in `metrics.csv` and `actions.csv`.

The runner reproduced all **six archived arms' actions exactly** and their metrics within 1e-10 on all 12 cells before adding the two controls. The receipt hashes the frozen protocol, 96 metric rows, and 600 action rows. An independent check verified those hashes and every hub arm's five actions, 40 simulated samples, 80 actuator uses, cumulative costs, and finite scores. The hub arms acquired new samples from the local numerical simulator. No CURC job or closed-model API was used.

**Decision:** the simple hub-first explanation does not fully account for the mean result, but the static controls remain competitive on individual systems. This is post hoc development evidence on the same fanout generator. No adaptive-design or foundation-model claim follows. Before promotion, test a graph family without one dominant shared parent and keep these hub baselines in the comparison.
