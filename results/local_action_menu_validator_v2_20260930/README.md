# Action-validator hardening smoke

At revision `0910df0e`, the validator passed 13 deterministic cases, including non-object proposals (`null` and a list), which are rejected cleanly before simulation. No invalid action was executed. The result hash is in `complete.json`. This is a safety-boundary check, not a language-model assessment. No simulator query, CURC job, or closed-model call was made.
