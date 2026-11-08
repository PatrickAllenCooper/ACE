# Collection01 failure consumer interface

Original acceptance/accounting are final, published once and unchanged. This sidecar makes the independently established no-response boundary explicit; it is not a second acceptance, a successful collection, or authority for collection02.

- Repository acceptance: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/failure-acceptance01/acceptance.json`
- Acceptance SHA256: `00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee`
- Repository accounting: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/failure-acceptance01/accounting.json`
- Accounting SHA256: `eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b`
- No-response proof: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/no-response-proof01/proof.json`
- Proof SHA256: `59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998`
- Original acceptance mirror: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-failure-acceptance-20261108-01`
- Proof mirror: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-no-response-proof-20261108-01`

Acceptance exact fields: `status=accepted_failed_before_response`, `failure_accepted=true`, `collection_accepted=false`, `complete_custody_verified=true`, `job_id="35407428"`, `required_findings=0`. The original acceptance's `job_id` is the failed job; this proof uses explicit `failed_job_id` with the same string. `packet_sha256`, `disposition_sha256`, and `freeze_sha256` bind the original attempt. The proof additionally binds `realization_acceptance_sha256=5e28e1aa16512eca3230154526dab65df65330f6920fcc094548c320736a65c1` to the original accepted input, not a new realization.

Proof has explicit integer zeros for `environment_response_attempts`, `environment_responses`, `unknown_return_attempts`, `scientific_worlds_generated`, `model_loads`, `tokenizer_calls`, `training_updates`, `policy_updates`, `records_created`, `pairs_created`, and `scientific_response_cost_units`. These follow from the verified synchronous staging exception before worker/session creation; the exact traceback, source pins and both-inventory corroboration are in proof.json. Node-local partial runtime deletion remains unverified, and this execution did not qualify a runtime.

Baseline uses existing `ace-scientific-screen-accounting-baseline-v1`. Exact41 terminal rows and84 journal hashes are included, with11386CPU/1601GPU,3600 preparation and14986CPU admission, zero outstanding reservation. Its exact joins are:

```
input_pins.predecessor = 69cc97784637b0ec522da6e00e00a1f08bdfabe9712e2104a43010ee19f2035f
input_pins.acceptance  = 00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee
input_pins.disposition = fd3a52be3054c1dd8d0d2b324dea3e64bdf8bc3676c28e7367cc0bbf06a3c685
```

No success status is introduced. The original collector remains `terminal_failed_outputs_preserved` with acceptance=false. Failure acceptance cannot satisfy the scientific token preparer's accepted-collection contract. Recovery source/contract/resource/once-only authority and fresh live baseline checks remain separate requirements. Both audit attempts, including the first auditor pin-count assumption error, remain under failure-acceptance01/audit; no original failure evidence was removed or repaired.
