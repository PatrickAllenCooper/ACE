# Fixed-value motif-selection development screen (29 September 2026)

The [protocol](../../docs/development/guidance/protocol_connected_fixed_value_dev_2026-09-29.json) was frozen before this run on reused binary-tree development systems 400–402. The selector scores the same pair-action menu as `risk_pair` and chooses the motif with the highest best-action score. It then uses a predetermined four-value cycle for that motif, instead of selecting the maximizing actuator values. Each arm still spends 360 units and acquires 40 responses per system. The policies can visit different motifs after their acquired data diverge.

Final feasible-motif MSE for the fixed-value control was 0.05840, 0.02066, and 0.03104 (mean 0.03670). The corresponding risk-pair values were 0.03559, 0.09161, and 0.01740 (mean 0.04820). The fixed-value arm won one of three individual systems, while the lower mean is driven by system 401. The factorial-hub mean was 0.04318 and balanced-risk mean was 0.04708. These are reused systems and a very small screen; no value-selection superiority or equivalence claim follows.

The runner verified both archived file hashes, exact parity of all prior metric and action rows, five new actions and their visit-specific value cycle, sample counts, and costs. A receipt records the source revision `1f2d2455c0003498c030a1f3f09a71a67ae98acc` and output hashes. A second run was byte-identical. The new arm used 120 synthetic simulator responses across three systems; no CURC job or closed-model call was made.

The next experiment should freeze a fresh-system comparison of the two adaptive motif policies. It should report motif paths and distinguish a failed value-optimization claim from the already confirmed advantage over the fixed factorial-hub control.
