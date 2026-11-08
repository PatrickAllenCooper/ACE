# Recovery02 authority interface — authoring only

This is a schema template, not an authorization record. Parent supplies the exact object below after independent adoption, with the two angle-bracket values replaced by independently supplied CLI pins. No extra keys. Its bytes are independently pinned with `--recovery-authority-sha256`.

```json
{
  "schema": "ace-scientific-realization-recovery-authority-v1",
  "status": "authorized",
  "recovery_id": "realization02",
  "failed_job_id": "35402725",
  "failure_acceptance_sha256": "5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d",
  "no_draw_proof_sha256": "5fb94f9d07da3acd25c979250471037e8981066e0b201716f8c32d0b681fb9ef",
  "predecessor_freeze_sha256": "57a3e6dd27ded5cec007be235b799a2bec711fc2c5630ff16fef87d74273cd28",
  "successor_freeze_sha256": "<independently adopted successor SHA256 CLI input>",
  "source_commit": "<full source commit CLI input>",
  "original_namespace_claim_sha256": "58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29",
  "accounting_sha256": "63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727",
  "original_root": "/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01",
  "recovery_root": "/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02",
  "local_recovery_registry": "/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json",
  "remote_recovery_registry": "/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json",
  "one_submission_only": true,
  "automatic_retry_allowed": false,
  "replacement_worlds_allowed": false
}
```

The namespace descriptor retains the original exact schema `ace-scientific-screen-namespace-descriptor-v1`, status `predeclared`, successor `freeze_sha256`, original `proposal_sha256` and `proposal_semantic_sha256`, `world_ids` (policy_train 202611070000..031; development 202611070100..123), `namespaces_from: "exact original proposal bindings"`, new local/remote registry paths, recovery realization root, `retry_allowed: false`, and `replacement_worlds: false`. Add exactly `recovery_authority_sha256`, `original_namespace_claim_sha256`, and string `failed_job_id: "35402725"`. See builder `expected_namespace` for the literal object.

Adoption remains the exact six-key object: schema `ace-scientific-contract-adoption-v1`, status `adopted`, `freeze_sha256`, `review_sha256`, `required_findings: 0`, `adopted_before_first_draw: true`. Independent pins for both review bytes and adoption bytes are required. Failure eligibility does not confer adoption or recovery authority.
