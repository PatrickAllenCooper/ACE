# Factorial hub development control (29 September 2026)

The [protocol](../../docs/development/guidance/protocol_connected_factorial_hub_dev_2026-09-29.json) and code were committed at `2b8e112f` before this run. It reused binary-tree development systems 400–402, with 30 nodes, 10 motifs, root SD 0.15, penalty 4, cost cap 400, and batch size 8. The new fixed policy used all four pair-value combinations on motif 0 and one pair action on motif 1. It spent 360 units and acquired 40 responses per system, matching the risk-pair arm. It made 120 new synthetic simulator responses across the three systems and no closed-model calls.

Final feasible-motif MSEs for systems 400, 401, 402 were:

| Arm | 400 | 401 | 402 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Factorial hub pair | 0.04683 | 0.05180 | 0.03092 | 0.04318 |
| Risk pair | 0.03559 | 0.09161 | 0.01740 | 0.04820 |
| Hub coverage pair | 0.09462 | 0.04853 | 0.03528 | 0.05948 |
| Coverage pair | 0.02997 | 0.06430 | 0.00826 | 0.03418 |

The fixed factorial control beats risk pair in one of three systems and has a slightly lower mean because of system 401. These reused systems provide no confirmation of policy superiority. The comparison reinforces the need for a fresh, frozen within-pair test that matches actuator-value diversity and exact cost. It does not establish an LM or a joint-action advantage.

The runner verified archived hashes and exact parity of every pre-existing metric/action row, and checked the five new actions, sample counts, and costs. `complete.json` records the source revision and file hashes. A second run produced byte-identical metrics, action logs, and receipt.
