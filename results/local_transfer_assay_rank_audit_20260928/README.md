# Four-response change-ranking audit

This evaluation-only audit replays the first four acquired target responses per node from the archived [20-system soft-mixture run](../local_transfer_soft_mixture_fresh_20260928/README.md). It ranks nodes by scratch-versus-source Gaussian log evidence. The evaluator uses changed-node labels only to measure rank; the ranking score itself uses only source posteriors and four target observations. All 40 source posterior hashes and the parent suite identity were checked, and a replay produced byte-identical outputs. No new observation or policy outcome was generated.

For k=1, the changed node ranks within the top eight in **80/80** source-size × family/coefficient × system cases. It ranks first in 79/80; the exception is family change, 16 source examples per node, rank seven. This is a strong feasibility signal for spending the 80 post-assay responses on eight nominated nodes in a future run. It is post hoc and comes from a favorable matched six-feature synthetic generator. It does not show that the resulting adaptive policy improves MSE, protects untouched nodes, or generalizes to misspecified mechanisms.

The next allocation test must freeze its rule and use untouched systems. The archive seeds 700–719 are now development evidence for this design and cannot serve as its confirmation.
