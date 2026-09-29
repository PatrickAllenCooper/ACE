# Action-menu validator smoke gate

At source revision `377b72c5`, 11 deterministic proposals tested typed targets and values, joint-target limits, a cost cap, an excluded actuator, an indirect proxy actuator, duplicate/unknown targets, malformed target type, and extra fields. All checks passed. Invalid proposals were rejected before simulation: zero executed invalid actions, zero simulator queries, and zero closed-model calls.

This is a deterministic safety-boundary smoke test. It does **not** test whether a foundation model can translate natural language into the schema, whether the proxy's effect is correctly modeled, or whether the resulting menu improves experimental design. Those remain the next Track 4 experiments.

`result.json` and `complete.json` record source revision and SHA-256 receipt. An independent rerun at the same revision reproduced the result byte for byte (SHA-256 `74267be624a3b498944fb534f06a4c6334343a2cb57079200ad60aa5555a0f60`). No CURC allocation was needed.
