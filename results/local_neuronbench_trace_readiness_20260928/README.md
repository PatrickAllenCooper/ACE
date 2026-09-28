# NeuronBench public trace readiness

This read-only audit checks the archived public observations from the `z_rebound` canary and five later designed NeuronBench worlds. The [script](../../scripts/research/audit_neuronbench_public_traces.py) reads no held-out target or upstream world object. The shared CURC connection was healthy and no ACE `acer_` job was active; the archived files were already local, so no new allocation or transfer was needed.

All **48 public voltage traces** (six worlds × two acquisition arms × four actions) have finite values, a uniform recorded-index stride of 10, valid test-start offsets, and file hashes in [audit.json](audit.json). Counting upward zero-voltage crossings after each public `test_start` exactly reproduces all 48 archived spike counts. This verifies that the traces contain event timing as well as the scalar count; it does not validate a forecasting model or establish the physical time unit of a recorded-index step.

The `h_sag` and `ca_rebound` random arms acquired the same four protocol labels and returned the same four spike counts. Their corresponding public voltage traces nevertheless differ: waveform RMSEs range from **1.02 to 27.03** in the archived voltage units. Thus a scalar-count-only forecaster discards information that can distinguish these two designed worlds. This is an information-availability check, not proof that the extra information predicts their private forecast labels.

Next, the [trace forecaster plan](../../docs/development/guidance/neuronbench_trace_forecaster_plan_2026-09-28.md) specifies a common dynamics model and public-only fitting gate. Predictions must be committed before any private score. No model/API call or new environment query occurred here.
