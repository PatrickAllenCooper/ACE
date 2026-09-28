# Fixed-data numerical controls for BoxingGym signal localization

[Frozen protocol](../../docs/development/guidance/protocol_boxing_signal_fixed_data_2026-09-28.json); ACE fit source revision `473bc820f4e46aba4323a02dc4b6b3cb2196029f`; CURC public-only sparse checkout `/scratch/alpine/paco0228/ACE/code_boxing_signal_fit_473bc82`; account `ucb736_asc1`; output `/scratch/alpine/paco0228/ACE/results/research_boxing_signal_fixed_data_dev_v1`. Jobs `33091055`–`33091057` for seeds 42, 123, 456 all COMPLETED 0:0 in 11–14 seconds with empty stderr. Each fit used exactly 16 public observations and 32 public forecast coordinates. The CURC checkout omitted private responses and source locations. Three predictions per coordinate and model settings were committed at `1bf280aeede1a18e91917a7635d9f8737115a75e` **before** separate private scoring. Receipt hashes, finite predictions, and CURC/local checksum parity pass; no model API call.

Held-out mean absolute error across 32 private responses per world (lower is better):

| World seed | Flexible RBF kernel | Three Gaussian sources | Privileged inverse-quadratic fit |
|---|---:|---:|---:|
| 42 | 4.2762 | 4.3564 | 2.2963 |
| 123 | 3.2738 | **2.1913** | 3.2117 |
| 456 | 3.7370 | 2.6923 | **1.1352** |

The generic three-source fit uses the public descriptive cue that three equal-strength sources contribute to the signal, but not the simulator's inverse-quadratic response law or true source locations. It beats flexible RBF on two of three development worlds. This shows a hand-designed structural prior can sometimes help; it does not establish any unique foundation-model contribution. The inverse-quadratic fit is privileged with the exact upstream equation and constants, but still estimates six source coordinates from only 16 noisy observations by five fixed multistarts. It loses to the Gaussian model on seed 123. Although the frozen protocol called it a “ceiling,” the **fitted result is not an achieved upper bound on performance**; only its model family has privileged knowledge. Do not use it as a numeric ceiling in comparisons.

These are three reused development worlds, with heavy-tailed responses near sources. No family selection, statistical inference, active acquisition, LM proposal, or fresh confirmation is established. A next A-track test should freeze a fallible structural proposal and a numerical fallback before new worlds, compare descriptive, anonymous, and misleading metadata, and test whether evidence can reject a wrong proposal. Public-data numerical controls must remain in every comparison.
