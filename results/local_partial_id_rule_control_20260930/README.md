# Public-text arithmetic control

Frozen source revision `8cdad90a31ef3d433eca3b4d576aff179f745af3`, using only the twenty public descriptions and the separate stage-2 reveals from `results/local_partial_id_task_suite_20260930`. The control parses the displayed Bernoulli parameter and candidate truth tables, enumerates action response laws, and calculates the exact posterior after each reveal. It never reads the answer key. The separate scorer reads the key and grades the response file.

All 20 intervals, identification decisions, actions, posteriors, and supported-candidate sets are exact. Response SHA-256: `e75bb41c2f7c49ba71f5383ec852a0da04fa0cfca5e02c9003aa1402aede6d68`; score SHA-256: `575fac0e33b269fb8759139a69b44bafcf5eefd9540b2de62f9387cf0953702f`. Invalid JSON, illegal actions, false certainty, and invalid posterior smoke cases were checked separately. Zero simulator queries or model calls.

This in-family rule control is strong and simple. Matching it on this synthetic family would show that an open model can execute the task, not that the model adds scientific value.
