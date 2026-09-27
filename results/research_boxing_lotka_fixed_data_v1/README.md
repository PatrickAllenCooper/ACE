# Fixed-data numerical controls on three BoxingGym development worlds

Protocol: `docs/development/guidance/protocol_boxing_lotka_fixed_data_2026-09-27.json`. These are the same three worlds and eight acquired observations per world from the oracle custody smoke; the sixteen held-out responses per world were read only by a separate scoring process. This is a development feasibility check, not fresh confirmation or a foundation-model result.

| Seed | RBF MAE | Fourier MAE | Privileged equation MAE |
| --- | ---: | ---: | ---: |
| 42 | 4.176 | 2.733 | 0.253 |
| 123 | 4.327 | 4.898 | 0.351 |
| 456 | 0.601 | 0.546 | 0.449 |

The RBF and Fourier arms fit only time-response pairs, with width and ridge penalty chosen by leave-one-out error on the eight public observations. The mechanistic arm has the **true Lotka–Volterra equation form and upstream initial state supplied to it**; only four rates are fitted. Its performance is therefore a deliberately privileged ceiling for a later typed-proposal test, not evidence that an LM or ACE discovered the mechanism. The high-value question is whether a locally run open model can propose a valid typed mechanism from a descriptive message without reading benchmark source, and whether it beats a broad numerical control on fresh worlds. The familiar predator–prey description can prompt recall of a textbook law, so a positive result here alone would remain narrow.

ACE revision `d1c061c907168a9489eeb1ec161be335574fbcd3`, pinned CURC checkout `/scratch/alpine/paco0228/ACE/code_boxing_fixed_d1c061c`, account `ucb736_asc1`, jobs `33037474`–`33037476`, output `/scratch/alpine/paco0228/ACE/results/research_boxing_lotka_fixed_data_v1`. All three completed with `0:0` in 15 seconds each, one CPU per job. Stderr was empty. Each receipt records the source revision, exact 8/16 counts, zero closed-model calls, and SHA-256 for fitted models and score files. Local receipts, finite scores, and CURC/local checksum equality were validated. No non-ACE job was changed.
