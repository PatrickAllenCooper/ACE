# Independent fanout development continuation

The frozen [extension protocol](../../docs/development/guidance/protocol_connected_fanout_extension_2026-09-28.json) adds independent seeds 303–305 at k=3 and k=10 under the same N=30 fanout graph, root SD 0.15, cost penalty 4, batch size 8, budget 400, six arms, and final feasible-motif MSE metric as the [initial fanout development screen](../local_connected_fanout_dev_20260928/README.md). It is a development continuation, not confirmation.

CURC job **33093296** ran on `acpu` under account `ucb736_asc1`, source revision `ec820bb94e9bb9753b1a14d223e437a6096a4031`, code checkout `/scratch/alpine/paco0228/ACE/code_connected_fanout_ec820bb`, and output root `/scratch/alpine/paco0228/ACE/results/research_connected_fanout_extension_dev_v1`. `submitted.tsv` records the job, account, revision, path, and time. Slurm reports `COMPLETED 0:0`; the local custody check independently passed the six cell receipts, suite and protocol hashes, source revisions, graph topology, complete action ledgers, exact query/sample/actuator costs, natural-child masks, and finite scores. A checksum dry run found no remote/local difference. The job produced seven receipts (six cells plus suite) and an empty error log.

Mean final feasible-motif MSE over **all six independent fanout systems per k** (initial three plus this extension; lower is better):

| Motifs | Random single | Coverage single | Random pair | Coverage pair | Risk pair | Balanced risk pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.36923 | 0.06828 | 0.54652 | 0.12396 | **0.00995** | 0.08138 |
| 10 | 0.06935 | 0.20609 | 0.11338 | 0.12376 | **0.01137** | 0.04188 |

The risk pair beats coverage pair in 6/6 systems at each k, random pair in 5/6 at k=3 and 6/6 at k=10, and coverage single in 5/6 at k=3 and 6/6 at k=10. Each pair arm spent exactly 360 cost units and executed five batches of eight samples. Full per-system values are in this directory's `summary.csv` and in the initial screen's `summary.csv`.

**Interpretation limit:** In all 12 fanout cells, the risk policy chose the shared upstream motif for exactly three of its five actions. Its advantage may therefore come from a simple graph-hub schedule rather than adaptive posterior design. The next control is a predeclared hub-first pair policy with matched action count, cost, and RNG handling. Do not promote the numerical risk selector or run a larger confirmation grid until it survives that control and a graph family without one dominant shared parent. These experiments contain no foundation-model contribution and used no closed-model API.
