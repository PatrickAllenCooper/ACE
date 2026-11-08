# Source inventory finding: fixed

Parent fix now requires a dictionary containing exactly the 18 preparation source names before digest comparison. The affected method passed all three rejection subcases (empty, missing gradient probe, unrelated extra helper). A new explicit valid exact-18 fabricated control also passed. Two selected checks ran in 0.042 s unittest; child CPU 0.131252 s. The other nine methods were not rerun. No production source or test changes were made.

The full failed attempt-01 was retained and verified against its original manifest. The original handoff remains historical; this delta closes its source-membership finding. This is stdlib toy-control evidence, not model/runtime/scientific qualification.

qualify_llm_intervention_screen_training_gpu.py: `4665d03e2b910921e5b7e9bc2d74e24d07278852d9d7d5b89ec1cc56932d2b56`

test_qualify_llm_intervention_screen_training_gpu.py: `087059008a06a606fedd2b846176175efb0cc780363646794aeb2765caacd46d`

Exact command, copied source, explicit positive runner, stdout/stderr and all fabricated inputs are retained in `delta-02`; complete inventory is `delta-02-manifest.json`.
