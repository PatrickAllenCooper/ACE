# Exact partial-identification fixture

Source revision `ff1b9ba8300f7252653dec6082601d9e1854bc07`; run locally with `ACE_SOURCE_REVISION` set to that revision. No CURC allocation, environment acquisition, or closed-model call.

Two candidate SCMs both generate `P(X=0,Y=0)=P(X=1,Y=0)=1/2`. Their candidate-set predictions for `E[Y|do(X=1)]` are 0 and 1/2. Observations alone therefore do not select between these candidates. If a `do(X=1)` query returns `Y=1`, the constant model has zero likelihood and the XOR-latent model has positive likelihood. Under equal prior weights, the posterior then places all mass on the latter.

This is an exact finite enumeration and a validation of the proposed grading task. The `{0, 1/2}` range is relative to the **two specified candidate SCMs**, not the identified set over all possible latent-variable SCMs. It does not evaluate an open model's predictions, calibration, or action choice.

`result.json` and `complete.json` contain the exact probabilities, source revision, and SHA-256 receipt. A second run at the same revision produced an identical result hash (`2b3de2941699b89e715e3f7d6694daf252de67548569b66b516924bfe1519d9b`).
