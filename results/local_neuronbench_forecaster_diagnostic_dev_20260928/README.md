# NeuronBench retrospective forecaster sensitivity

This zero-query development diagnostic uses the already-counted four-action public observations from the six designed NeuronBench worlds. Its [frozen protocol](../../docs/development/guidance/protocol_neuronbench_forecaster_diagnostic_2026-09-28.json) and public-only predictor code were committed at `07a529c3567fc800bf476bb07e760075aeba418a`. All 24 mean/nearest predictions were then written and committed at `d8e6e55e92fddab02a0444f38d9be20ed97252e9` **before** the separate private scorer ran. Zero new oracle actions and zero closed-model calls. Independent checks validate all 24 public hashes, private target hashes, six-label sets, finite predictions, and score parity.

Official floored spike MSE, lower better. Ridge is the already archived fixed-penalty forecaster; mean and nearest are this diagnostic's frozen controls:

| World | Acquisition | Ridge | Mean | Nearest |
|---|---|---:|---:|---:|
| z_rebound | random | 51.8401 | 81.5833 | 176.9167 |
| z_rebound | coverage | 3.4100 | 31.0625 | 5.5833 |
| h_sag | random | 45.4633 | 133.6667 | 67.2500 |
| h_sag | coverage | 2.2961 | 64.3958 | 27.2500 |
| na_fatigue | random | 493.8772 | 51.3333 | 154.7500 |
| na_fatigue | coverage | 88.8386 | 53.3125 | 48.7500 |
| ca_rebound | random | 22.1660 | 155.0000 | 81.0833 |
| ca_rebound | coverage | 28.2513 | 45.7500 | 77.7500 |
| d_type | random | 4.0092 | 127.3125 | 24.5833 |
| d_type | coverage | 61.2919 | 55.6458 | 74.5833 |
| textbook_M | random | 1.0463 | 1.0000 | 1.2500 |
| textbook_M | coverage | 0.6943 | 1.5625 | 0.4167 |

Coverage beats random on 4/6 worlds with ridge, 4/6 with the mean forecaster, and 5/6 with nearest. The *identity* of a loss changes: `ca_rebound` favors random with ridge but coverage with mean/nearest; `d_type` favors random with ridge/nearest but coverage with mean. `na_fatigue` random ridge error is far larger than its constant-mean error. Hence acquisition and forecaster are coupled at four observations; neither the earlier coverage gains nor reversals isolate a mechanism-aware action advantage. These same six scored designed worlds were reused, so counts of wins have no inferential meaning. Do not tune a forecaster to this table and call it confirmation.

A next architecture needs an independently specified mechanism model and acquisition rule, with common fixed-data evaluation to assess the predictor before another online comparison. The mechanism library must not encode the upstream world-specific truth or discriminator mapping.
