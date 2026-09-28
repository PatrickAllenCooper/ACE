# Trace-informed NeuronBench forecaster: development plan

The six designed NeuronBench worlds have four counted public actions per acquisition arm. Their archived observations include voltage samples and a public test-start offset. A [readiness audit](../../../results/local_neuronbench_trace_readiness_20260928/README.md) verifies 48 traces and exact spike-count parity. This makes a dynamics-based predictor technically possible, but the current ridge, mean, and nearest controls have shown that apparent acquisition gains depend on predictor choice.

## Model to implement

Use one shared, world-agnostic forced integrate-and-fire state-space family. Its state is membrane potential `v`, a spike-triggered adaptation state `a`, and a recovery state `h` driven by negative input. Between spikes, a candidate discrete step follows

`v_next = v + dt [-(v-v_rest)/tau_v + g I - a + h]`,

`a_next = a + dt [-a/tau_a]`, and

`h_next = h + dt [-h/tau_h + c max(-I,0)]`.

When `v` crosses a fitted threshold, record one spike, reset `v`, increment `a`, and enforce a fitted refractory duration. The recovery gain may fit to zero; it is not conditioned on a world name, textual description, or a known channel label. This is an approximate scientific model, not an implementation of the benchmark's hidden simulator. Treat voltage samples around spike peaks as events and fit subthreshold samples separately, because the spike waveform itself is outside this simple state model.

Fit a common parameterization and regularization rule using only the four public protocols and their voltage traces for each cell. Constrain positive time constants and gains and use a bounded, documented optimization budget. Any choice among model variants or regularization strengths must use public leave-one-protocol-out prediction only and must be frozen before private scoring. Do not use the upstream `World` object, hidden discriminator mapping, held-out target values, or source-world names as features.

## Gates before outcome scoring

1. Independently reconstruct each observed input waveform and match the archived trace length, recorded-index stride, public test-start offset, and spike count. Do not infer a physical time step from the index stride without checking the public upstream protocol specification.
2. On the observed protocols, compare leave-one-protocol-out spike-count error with the archived ridge and nearest controls using only public labels. Inspect subthreshold voltage fit separately; a count fit alone does not show a useful state model.
3. Apply the **same frozen fitter and compute cap** to the random and coverage arms. Produce all six forecast-label predictions per cell from public files only; hash and commit predictions and model settings before loading any private target.
4. Score separately with the existing private scorer. Report each designed world and predictor; do not pool six selected worlds into an inferential claim. A fresh benchmark/domain is required before promotion.

If the model cannot reproduce public held-out protocols better than the simpler controls, stop this implementation instead of increasing complexity after seeing private scores. Trace access changes the forecaster information set compared with scalar-count baselines, so any comparison must state that difference. No closed-source model API is needed.
