# First CPU service preflight packet review

Date: November 7, 2026. Reviewer: Newton, an independent read-only agent separate from the implementation parent and the previous source reviewer.

Scope: actual launch01 packet, all worker inputs, the two artificial service manifests, Slurm script, once-only launcher, observer and terminal collector. The reviewer did not repeat closed implementation reviews or tests. No source execution, remote calls, archive hashing, model loading or scientific decoding was performed.

The reviewer verified the outer packet SHA `5372ded2302104f51650663c37db4aaa5f7678f4513e5469f19d7256096d5d61` and inner preflight SHA `d5ac0518ff222193b279fb9268aba1158d1ae5247b3919684cdebbccc70b2c46`. All 28 transfer objects match both custody copies. Seventeen prospective source files match source commit `376ccb348d8bdec6bb5351117884318f0ccb803e`. Original model inventory, artificial adapter, training receipt and three public fixtures retain their GPU10 lineage. Manifest changes are limited to schema, runtime and source bindings.

CPU-only CLI, staging, child and supervisor deadlines, renewed account, resource request and new output root agree. Dual controller locks, exclusive submission journals, remote readback and ambiguous-submission handling are present. Proposed admission is 10,566 of 57,600 CPU-seconds, including the 600-second reservation and unchanged preparation allowance. GPU accounting remains 836 of 7,200 GPU-slice-seconds.

One required finding concerned the collector. It asserted the requested resources before completing disposition and repository custody. An unexpected allocation would therefore prevent complete failure custody. The correction records observed allocation and actual costs, marks a resource mismatch as failed qualification, and still completes custody. Generic and typed GPU allocation counts are reconciled without double counting. Unknown counts remain unknown. Six isolated, hand-computed cases passed without executing the collector's top-level code.

A related correction records source-input pin mismatches as failed input qualification while preserving complete remotely observed custody. Remote-versus-transferred membership, size or byte-hash mismatches still prevent a verified-custody claim. The original collector and initial finding are preserved separately. Final collector SHA is `f0cba841750f22cdedc8def019ad8a5cb78e577ece1c642623c3dddf89bfc83a`.

The combined final read-only disposition closes R1 and accepts the input-pin delta, with zero required fixes remaining. The reviewer confirmed that unknown or conflicting accounting remains null with errors and that both custody copies match the final collector. Actual terminal collection remains pending.

Live reconciliation of the expected eighteen prior jobs and thirty-eight journals, renewed association and scratch readiness, remote byte authentication, a single submission and independent result acceptance remain required. This review does not establish runtime or model-service qualification.
