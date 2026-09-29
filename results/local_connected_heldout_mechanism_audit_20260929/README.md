# Connected held-out mechanism gate

At pinned revision `e9b347dc`, the connected simulator gained an optional target-only local term `0.85*tanh(1.7*x1 + 0.8*x2)` at motif 7. Twelve new N=30 fanout systems (seeds 1900–1911) were audited with common exogenous noise. The two parents of motif 7 stay identical; its child changes by exactly the added term (maximum identity error 1.11e-16). All other motif outputs and natural-label masks stay identical in this isolated change audit.

On independent 2,048-context uniform parent panels, the minimum three-feature linear projection residual MSE is 0.05840–0.06125. Thus the term is meaningfully outside the current `(x1,x2,x1*x2)` fitted bank under that reference distribution. This is a simulator and evaluation gate, not a transfer or acquisition result. The audit generated 6,144 paired source and 6,144 paired target diagnostic trajectories, acquired zero training trajectories, and made zero closed-model calls.

The suite receipt hashes its 12-row metric table, and a second run was byte identical. An old-versus-new simulator comparison covered 20 no-heldout cells (ten seeds, observational and intervention action each) and found byte-identical values, features, and masks. The old connected-policy results remain reproducible under the default simulator path. No CURC job was needed for this short local audit.
