# Protected transfer allocation-floor development check

[Frozen protocol](../../docs/development/guidance/protocol_transfer_floor_dev_2026-09-28.json); ACE source revision `eb444eebb708e8c7a91a2ef24c2b89b878b243b1`; CURC checkout `/scratch/alpine/paco0228/ACE/code_transfer_floor_eb444ee`; account `ucb736_asc1`; output `/scratch/alpine/paco0228/ACE/results/research_transfer_floor_dev_20260928`. Jobs `33088724` (floor 5) and `33088725` (floor 6) COMPLETED 0:0 in 21 and 24 seconds. Empty stderr. Remote/local checksum parity and independent local validation pass: each has 72 settings, 2,160 node rows, exact 200 target examples per setting, 2,560 separate source-library examples, finite metrics, and receipt/file hashes. No closed-model API call.

A four-example assay at each of 30 nodes precedes the adaptive allocation. Floor 5 guarantees at least five examples per node; floor 6 guarantees six. Both keep the same 200 total examples and are compared to archived uniform warm-start errors by exact seed, change type, changed count, and node.

At 200 examples, adaptive **warm** error ratios versus uniform warm on family changes (k=1,3,10):

| Floor | Changed-node ratios | Untouched-node ratios |
|---|---|---|
| Earlier floor 4 | 0.220, 0.155, 0.634 | 1.006, 1.040, 1.099 |
| Floor 5 | 0.220, 0.155, 0.563 | 1.006, 1.040, 1.098 |
| Floor 6 | 0.220, 0.188, 0.684 | 1.006, 1.033, **1.063** |

The frozen acquisition gate requires all family changed-node ratios ≤0.8 and all untouched-node ratios ≤1.05; coefficient-change changed-node ratios must also be ≤1.05. **Neither floor passes**: floor 5 breaches the untouched limit at family k=10 (1.098) and coefficient k=10 (1.052); floor 6 breaches family k=10 (1.063). The source-switch ratios against adaptive warm also fail its separate gate, most clearly at k=1 (0.985). The script's `passes_development_gate` field refers to that source-switch gate; this note computes the separate acquisition gate from `summary.csv`.

These are the same 12 favorable development systems used before. The failed simple floor adjustment does not warrant fresh CURC confirmation. The next acquisition idea needs a different way to protect untouched nodes under a fixed budget, or a revised tradeoff objective frozen before new systems. No result here establishes a foundation-model or transfer benefit.
