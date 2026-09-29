# Connected transfer pair-policy development pilot

This six-system **development** pilot (seeds 1800–1805) ran at source revision `71d6dd31`. Each N=30 fanout graph has ten fitted interaction motifs and three changed motifs in the target (0, 5, 7); all forms remain inside the fitted three-feature family. Source posterior fits use 16 or 64 fully observed source trajectories. Each target arm receives the same four-trajectory observational assay, then five eight-trajectory pair batches. The primary target cost contract is 44 acquired trajectories, 80 actuator uses, and cost 364 per arm. There is no closed-model component.

Two pair policies share the source fits, assay, query cost, and intervention menu: a fixed varied-value factorial hub schedule and a posterior-risk selector. On each acquired dataset, both source-warm and broad-prior scratch posteriors are fitted; the evaluator uses a separate sealed feasible-intervention panel. Risk selects motif 0 for its first three actions in all six systems, then other motifs. It masks 16 natural mechanism labels (424 remain) per arm versus eight masked (432 remain) under the fixed schedule, because some later actions intervene on a motif child. Every mask and action is archived.

Mean final feasible-motif MSE (changed and unchanged motifs scored separately):

| Source trajectories | Policy / estimator | Changed | Unchanged |
| --- | --- | ---: | ---: |
| 16 | Fixed / source-warm | 0.01141 | 0.01276 |
| 16 | Risk / source-warm | 0.00528 | 0.00965 |
| 16 | Fixed / scratch | 0.01111 | 0.01540 |
| 16 | Risk / scratch | 0.00425 | 0.01250 |
| 64 | Fixed / source-warm | 0.00668 | 0.00872 |
| 64 | Risk / source-warm | 0.00474 | 0.00790 |
| 64 | Fixed / scratch | 0.01111 | 0.01540 |
| 64 | Risk / scratch | 0.00905 | 0.01377 |

The risk policy has lower source-warm changed-node error in 6/6 systems at source_n=16 and 5/6 at source_n=64. The source-warm estimator does **not** beat scratch on changed motifs at source_n=16 under either policy; it does at source_n=64. Thus this pilot suggests an acquisition effect and an uneven transfer effect. Six systems, one topology, in-bank changes, and a static comparator are too limited for a promotion claim. The risk and fixed policies see different acquired trajectories after the shared assay, so cross-policy comparisons mix action choice with resulting data; within-policy source-warm versus scratch uses identical data.

All 24 campaign receipts, graph/source/assay inputs, action and acquired arrays, source revisions, 480 node-metric rows, exact costs, natural/masked label counts, SHA-256 hashes, and byte-identical replay validate. There are 1,056 target arm-trajectory counts (24 × 44), with assay trajectories reused across policies and source-size analyses; this is not a count of globally unique target trajectories. Six source streams of 64 trajectories were generated and reused by prefixes. No CURC job was needed for this short CPU pilot, and no closed-model API or non-ACE job was used.
