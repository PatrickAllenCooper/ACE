# NeuronBench intervention-task feasibility audit

Read-only code/API audit on 27 September 2026. Pinned upstream [NeuronBench](https://github.com/murphyk/neuronbench) revision `c354622458c460b419cab821d482c879f0578377` (MIT). No solver was evaluated, no benchmark job was submitted, and no model API was called. The local checkout is an audit copy outside the ACE repository, not a committed dependency.

## Relevance and limits

NeuronBench is a partially observed, dynamic single-neuron intervention-forecasting benchmark. It has a shared nine-protocol action pool, noisy stochastic and deterministic modes, a hard experiment budget, and held-out spike-count/voltage forecasts. Five novel worlds are constructed so standard probes do not reveal the extra current; one world is a recallable control. This is closer to agenda B's *intervention reachability* question than the two-population BoxingGym time-query task. It is still not the original ACE stationary 30-node SCM benchmark, and its neuron-specific model class substantially changes the inference problem.

The repository is tied to the [Model Discovery Agent](https://arxiv.org/abs/2608.09696) work already identified as close architectural prior art. A NeuronBench result would be external evaluation, not a standalone novelty claim. We should not use this benchmark to tune a method after seeing its world-specific truth and revealing protocols.

## Reproducibility and information boundary

The pinned `pyproject.toml` specifies Python >=3.11 and NumPy/SciPy. CURC's current ACE environment is Python 3.10; the base Python 3.13 lacks NumPy. A future run needs a small isolated environment in ACE scratch storage. The local audit imported the package with Python 3.11 and confirmed `world.problem()` exposes five public fields and nine protocol choices, but performed no intervention.

The `World` object itself is **not safe to hand to an agent**. Its `discriminator()` returns the world-specific revealing protocol, and its internal truth and held-out evaluator are reachable from the same process. The upstream intervention API also reports cost but leaves enforcement to the runner. A future custody adapter must:

1. Pin the upstream revision and source hashes, instantiate worlds only in an oracle process, and serialize only `problem()` public fields to the proposer.
2. Accept only a legal protocol from the shared public pool, enforce the exact budget and distinctness/repeat rules, and write each observation and cost to a receipt. Never expose the Python `World`, `WORLDS`, `discriminator()`, `test_protocols` segments, or evaluator to a solver process.
3. Keep test targets in a separate scorer process. Forecasts must be frozen before scoring. Forward simulation of a solver's *own* hypotheses may be allowed under the benchmark contract, but the hidden true mechanism must not leak through that interface.
4. Start with deterministic one-world custody and budget accounting only. Add matched random and coverage baselines before any PEV-like design, and test stochastic mode only after noise/repeat accounting is validated. Do not infer independence from repeated trajectories of one fixed world.

The nine public labels and the author's code are inspectable by the researcher; even with process isolation, a hand-coded solver informed by the full source can accidentally encode the answer. Freeze an independently specified hypothesis library and action policy **without using world-specific discriminator mappings**, or use independent implementation review, before claiming blinded method performance. An initial smoke can establish engineering feasibility but no scientific advantage.

## Decision

Keep NeuronBench as an external B/PEV adaptation candidate. Do not submit a score-seeking run yet: the random-DAG PEV shift did not pass its development gate, and no leak-safe NeuronBench oracle/solver boundary exists. The next useful work is a small custody adapter and cost-only smoke, followed by a predeclared numerical-control comparison if its information boundary passes review. No non-ACE CURC job was touched by this audit.
