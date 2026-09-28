# Signal-localization source-count reliability diagnostic

[Frozen protocol](../../docs/development/guidance/protocol_boxing_signal_prior_gate_dev_2026-09-28.json); ACE fit revision `f28464be057d7843a4955efb5040c973c37ccae0`; CURC public-only sparse checkout `/scratch/alpine/paco0228/ACE/code_boxing_signal_gate_f28464b`; account `ucb736_asc1`; output `/scratch/alpine/paco0228/ACE/results/research_boxing_signal_prior_gate_dev_v1`. Jobs `33091738`–`33091740` for seeds 42/123/456 COMPLETED 0:0 in 18/26/26 seconds with empty stderr. Each used the same 16 already acquired public observations and 32 public forecast coordinates; no new environment query or model API call. The private files were absent from the fit checkout. Predictions, validation choices, hashes, and CURC/local checksum parity were validated and committed at `47dcef524711ce174dd59d003139945a656d9ed5` before separate held-out scoring.

The first eight observations choose model widths and penalty; the next eight compare proposal versus fallback by public validation MAE. Both candidates are then refitted on all 16 with frozen hyperparameters. The “correct” proposal uses three Gaussian sources, matching the public source count but not the upstream equation. The deliberately wrong proposal uses one Gaussian source. The fallback is flexible RBF kernel ridge. These are numerical proxy priors, not language-model outputs.

Held-out MAE across 32 private responses (lower is better):

| Seed | Fallback RBF | Correct three-source | Wrong one-source | Gate with correct | Gate with wrong |
|---|---:|---:|---:|---:|---:|
| 42 | 4.5937 | 4.3564 | 4.7212 | 4.3564 | 4.7212 |
| 123 | 3.5428 | 2.1913 | **2.0872** | 2.1913 | **2.0872** |
| 456 | 3.0513 | 2.6923 | **2.5784** | 2.6923 | **2.5784** |

The public validation gate **accepted both proposals in all three worlds**. It failed to reject the false source count. The wrong one-source model harms versus fallback on seed 42, but does better on seeds 123 and 456; the correct-count model loses to the wrong model on those two. Thus neither exact source count nor this eight-sample gate supplies a reliable advantage on the reused development worlds. A one-source radial bump can approximate a multi-source field over sparse, noisy queried locations. The fallback here differs from the earlier RBF baseline because its hyperparameters were chosen on only the first eight observations, as frozen for the gate comparison.

All three scored receipts match frozen public predictions, private target hashes, 32 coordinate-aligned responses, and independent MAE recalculation. This is a negative development result, not a fresh-system statistical conclusion. Do not extend this same source-count gate to a large CURC sweep. A new A-track prior should encode a more discriminating structural prediction, with misleading-metadata recovery tested before a fresh confirmation protocol.
