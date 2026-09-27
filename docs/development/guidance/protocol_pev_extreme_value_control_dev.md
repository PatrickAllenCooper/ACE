# PEV intervention-value control: three-system development screen

Frozen before CURC submission, 27 September 2026. This is a **development diagnostic**, not a fresh confirmation. It reuses shift30 systems 42, 123, and 456 from the archived noise-free replay. The completed PEV, PEV naive-variance, graph-matched coverage, and graph-matched random arms remain the reference outcomes. The new arms use the same persistent ensemble learner, known graph, evaluator, 2,000 total environment samples, 50 samples per executed action, 40 observational samples every three steps, three ensemble members, 20 training epochs, and zero candidate-probe queries.

Two new graph-matched, uncertainty-free controls separate intervention **value** from target scheduling:

- `nonleaf_extreme_coverage_ens`: cycle all eligible non-leaf targets as in coverage, alternating values −5 and +5 across target visits.
- `nonleaf_extreme_random_ens`: draw uniformly among eligible non-leaf targets and choose −5 or +5 with equal probability.

The existing coverage and random controls use values across [−5,+5]. The completed PEV and naive-variance arms used mean absolute values about 4.58 and 4.80, versus 2.61 and 2.44 for coverage and random, respectively, on the 20-system confirmation. The new controls test whether endpoint-valued experiments explain much of the apparent uncertainty-policy advantage. This was chosen after inspecting those action logs, so all findings remain exploratory.

Use only the three old development seeds at first: six CPU cells. Analyze **final** `feasible_mean_nonroot_loss`, not a selected checkpoint, and report all six arms per seed, executed query counts, target/value logs, graph and coefficient parity, and runtime. If extreme coverage approaches PEV or PEV-var, prioritize value-design controls over more uncertainty-scoring work. If both extreme controls remain materially worse, the target schedule may matter; this does not isolate IVR from naive variance. Do not promote either inference from three systems. Freeze any future independent-system protocol only after this diagnostic is validated.

The runner's one-step local smoke validates both new arms and exact 50-sample accounting. Its local re-instantiation of seed 42 reproduces the archived graph and forms exactly; coefficient JSON differs by at most 1.1e−16 across the local and CURC NumPy runtimes. CURC will run all new arms from one pinned revision. Never call Azure or another closed-source model API.
