**Independent static review of new campaign training lineage — 2026-11-08**

**Disposition: no required defects found in the bounded lineage-admission change.** This reviewer did not author the lineage module. This is a distinct static review of the new module and its four fabricated tests, with local JSON reads and hashes to resolve the fixed GPU03 identities. It does not re-audit accepted training or qualify a later campaign/service execution.

Reviewed source bytes:

- [llm_intervention_campaign_training_lineage.py](/Users/pat/code/ACE/scripts/research/llm_intervention_campaign_training_lineage.py): SHA256 `3f3796576a6d2ddac7465dc73a918b3001dd0830fc548352baa6a77a50358f06`.
- [test_llm_intervention_campaign_training_lineage.py](/Users/pat/code/ACE/scripts/research/test_llm_intervention_campaign_training_lineage.py): SHA256 `9a4588a33d00cd7fbbde054dad830a1dbed17c5bd48ef279f42901da6ee7ac15`.

The review did not import or execute either module, call `admit_training`, repeat tests, read adapter/model tensors, deserialize models, run policies, submit jobs, or edit either source file. Only the existing service receipt-field contract was consulted outside these two source files.

### Fixed accepted inputs

The following hashes were recomputed from local saved JSON bytes and agree with the new module's fixed pins or the engine pin inside the fixed acceptance object. The engine pin is transitively fixed by the acceptance byte hash, even though it is not a separate module constant.

| Saved JSON | SHA256 of file bytes |
| --- | --- |
| [GPU03 acceptance](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/acceptance03/acceptance.json) | `4f49ed2e09da27f87b02ca3e20fc835ae5a9b5eeeeb26ae1039421375ed0a3d7` |
| [Prepared batch](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/launch03/prepared.json) | `ceb5847336cf664b215a97d0c8f56f86f2f696e02e44f684ec5c9277e364653e` |
| [Engine terminal](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal03/remote/qualification/session/worker/artifacts/fit/training/terminal.json) | `b1ca5dd545de84b84a2dcda2dfd2505ed4e7bdf9c469f688913e989fc73e78d3` |
| [SFT checkpoint receipt](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal03/remote/qualification/session/worker/artifacts/fit/training/sft.checkpoint.json) | `feebd8800d9405a1cf0ccbb58a38fe301a3f58fc5b6741e777824008ce067680` |
| [Final checkpoint receipt](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal03/remote/qualification/session/worker/artifacts/fit/training/final.checkpoint.json) | `d3727084d31ff0ee785283d7432acca0f2fac89f66dd8288c28493737180e56c` |

Admission checks acceptance bytes before interpreting the qualification fields, requires job `35379434`, accepted artificial training and reload status, zero remaining requirements, and false scientific qualification. It checks the prepared bytes independently, then authenticates the engine terminal using the acceptance-pinned engine hash. The original inner false provenance flags remain unchanged.

### Role and batch joins

The saved SFT receipt has `completed_steps=4`, attempt index `3`, stage `sft`, and both optimizer-started/returned flags true. The final receipt has `completed_steps=6`, attempt index `5`, stage `dpo`, with both flags true. These are the exact counts and roles required at source lines 79–96. The complete engine terminal declares four SFT plus two DPO updates and zero environment responses.

The adapter hashes recorded by both acceptance and the corresponding checkpoint receipts agree:

- SFT: `711c1782845217a13764b54c9ba2df3786345fb211ac05811aeab109ee77cd02`.
- Final DPO: `3a03d902690a61f737af0362e85e1e3be2f141f3e337a038ab6778742f682011`.

The admission code hashes the supplied opaque adapter bytes and compares them with both declarations. This review checked that code and the JSON declarations; it did not reread or hash the actual adapter files.

The prepared training plan's recomputed semantic hash is `fc6c1f51e436c91125d70ffb1462732d32ef7ecd67b567e07ae9360f030eff9b`, matching the prepared declaration, acceptance and engine terminal. The exposure's recomputed semantic hash is `d9d0fa602649a7ab2e12d8936cee9261a0826f7438cc5c10f2ff8c20bd99edc4`, matching the prepared declaration and acceptance. These semantic digests are distinct from the JSON file-byte hashes above.

The pinned batch contains the four records `artificial/17/snapshot-000`, `artificial/17/snapshot-001`, `artificial/4/snapshot-000`, and `artificial/4/snapshot-001`. Tokenized-record original-record pins agree with exposure record pins; plan and exposure schedules agree; engine exposures agree with plan exposures. The retained batch has four SFT exposures, two selected DPO pairs, two unused strict pairs and two ties. The SFT checkpoint's last record is `artificial/4/snapshot-001`; the final checkpoint's last record is `artificial/4/snapshot-000`, pair index `1`, matching schedule entries 3 and 5. Prepared collection-plan and collection-terminal pins also agree with acceptance. This checks the new admission joins against already accepted records, without repeating their training validation.

### Downstream service boundary

Each returned role has treatment `artificial_adapter`, training scope `artificial`, and a role-specific `checkpoint_sha256` computed from the exact checkpoint receipt bytes. The outer `acceptance_sha256` is returned separately. The existing service contract at [llm_intervention_model_service.py:118](/Users/pat/code/ACE/scripts/research/llm_intervention_model_service.py:118) accepts `adapter.training_receipt_sha256` and authenticates those receipt bytes at line 175.

The later outer driver must set that service field to `roles[policy_id]["checkpoint_sha256"]` and supply the corresponding checkpoint receipt file. It must set the service adapter hash from `roles[policy_id]["adapter_sha256"]`. Substituting the outer acceptance pin would bind different bytes. The new module returns the correct distinct role material; no future service-freeze construction or execution was available for this bounded review, so this is a preserved integration obligation, not a claim that a service has already used it.

### Test evidence and limits

The four new tests were read, not run. They cover distinct receipt/acceptance pins and 4/6-update role outputs; altered input bytes and swapped adapter rejection; fabricated repinning with wrong scope, job, qualification, remaining requirements or checkpoint count; and non-byte/duplicate-object rejection. Their pin constants are deliberately patched to fabricated values, so their successful execution alone cannot authenticate the production GPU03 constants. The local JSON hash checks above address that specific gap without executing the code.

The coordinating agent supplied the result **four tests passed on the initial attempt, 0.049418 child CPU-seconds, pure fabricated acceptance pins**. This review did not reproduce that timing or test result. Review CPU usage was not measured. No source changes or additional tests are required by this review; whole-campaign acceptance and future service execution remain outside its scope.
