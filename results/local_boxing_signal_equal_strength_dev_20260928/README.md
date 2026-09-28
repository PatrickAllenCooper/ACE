# Signal equal-strength public diagnostic

This zero-query, zero-model-call development check uses the same 16 public BoxingGym Signal observations in each of the three previously scored worlds. The [protocol](../../docs/development/guidance/protocol_boxing_signal_equal_strength_dev_2026-09-28.json) was written before running this diagnostic. Only the first eight observations select width and fit three Gaussian centers; the next eight provide a public validation MAE. Both models use the same center-selection rule and a nonnegative offset. The descriptive model constrains all three source amplitudes to be equal; the numerical comparator fits separate nonnegative amplitudes.

| World seed | Equal amplitude MAE | Free amplitude MAE |
|---|---:|---:|
| 42 | 0.9900 | 0.8454 |
| 123 | 0.4887 | 1.2823 |
| 456 | 1.9400 | 1.8443 |

The public equal-strength cue helps in only one of three reused worlds. Thus this simple Gaussian encoding is not a robust discriminating semantic prior. There is no justification to schedule an open-model GPU run that can only pick this same restricted model family. This does not rule out richer semantic proposals or a different independently authored benchmark. No private responses, source locations, or API were used in fitting or selection. Each cell has a public-input hash and a hash of its diagnostic output. These worlds have already been used for development, so no inferential claim follows.
