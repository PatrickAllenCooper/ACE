# Scientific training lineage source review — 2026-11-08

**Required findings: 0 for this scoped, prospective role-admission helper.** This is an independent read-only source/interface review, not acceptance of a scientific training run or development campaign. Neither reviewed module nor its tests was executed during this review.

## Reviewed bytes and evidence

- `scripts/research/llm_intervention_scientific_training_lineage.py`: SHA256 `f3b328203d5758418d0afd677513822e8d7f134acca10545aab2d3fda1d3fc35`.
- `scripts/research/test_llm_intervention_scientific_training_lineage.py`: SHA256 `bf30600e963a758bea44b915f3ea64a5cd1119085c1684aa48acba6b9871c10f`.

Both working files match the independently supplied pins and the retained source copies under `/Users/pat/.codex/tmp/ace-scientific-lineage-wx66pr4_/source`. The existing `run.json` and `stderr.log` report eight new stdlib tests passing, return code zero, 0.131353834 seconds wall, 0.106725 seconds child CPU and 19,611,648 bytes peak RSS. These are inspected prior results, not a new test run.

## Findings and interface checks

The audit acceptance byte pin and the exact independently supplied `expected` map are distinct inputs. The helper checks the frozen scientific, collection-acceptance, CPU-preparation-acceptance, model, runtime, prepared-byte, plan-semantic and exposure-semantic pins, plus the reconciled job ID and full source commit. Replacing an acceptance and its hash alone cannot change those expected identities. The acceptance must explicitly attest training, checkpoint reload inference, custody, runtime, owned cleanup and scheduler verification, with integer zero required findings. Worker completion, an artificial acceptance or an exit code does not meet this contract.

The prepared schema and field names agree with `llm_intervention_training_plan.prepare_complete_collection`: `ace-complete-collection-training-plan-v1`, nested `training_plan`/`exposure`, and their separate semantic hashes. The helper binds original prepared bytes without reserialization and checks both exact ordered 160-record rosters, `policy_train`, `qwen_screen_singletons`, 160 SFT, D in 1–128 and checkpoint indices `[159, 159+D]`. The engine terminal fields it requires exist in `engine.fit` output; integer counts and zero new environment responses are checked against both the terminal and independent acceptance.

The checkpoint checks match actual engine timing. The engine writes `sft.checkpoint.json` before the first DPO update, while `attempt` still denotes completed SFT index 159 and `completed_steps` is 160. It writes `final.checkpoint.json` after all updates, with completed DPO attempt index `159+D` and count `160+D`. Both returned optimizer-step flags are therefore correctly required. Each opaque adapter hash must match both its independently accepted adapter pin and the original checkpoint's `files.adapter`; each complete checkpoint receipt is separately byte-pinned by the acceptance. No tensor loading, optimizer restoration or model initialization occurs here.

Returned roles correctly map `sft_llm` to treatment `sft` and `dpo_llm` to treatment `dpo`, both with training scope `policy_train`. These satisfy the treatment/scope rules in `llm_intervention_model_service.validate_freeze`, including use in a separately admitted development freeze. The future coordinator must map each role's `checkpoint_sha256` to the service adapter's `training_receipt_sha256`, supplying that original receipt and the separately pinned adapter file. The outer acceptance hash is not a replacement checkpoint-receipt hash. `restore_adapter` explicitly treats the receipt pin as lineage metadata, so this upstream admission remains necessary. The scientific campaign assembler likewise leaves actual checkpoint-role/byte authentication to its caller.

The new source imports only stdlib, performs no file writes, hashes supplied adapter bytes opaquely, and reports zero new training updates/responses and false development-execution/scientific-utility qualification. The historical artificial lineage file has no working-tree diff and retains SHA256 `3f3796576a6d2ddac7465dc73a918b3001dd0830fc548352baa6a77a50358f06`.

## Scope and remaining acceptance

This is a consumer of a genuine independently pinned audit, not a replacement full-plan or training-results verifier. Correctness of all 3,584 retained comparisons, exact exposure selection/numerical recipe, tokenizer/model provenance, optimizer/reference/recomputation evidence, source capture, runtime, resources and custody must already have been established by the referenced CPU and GPU audits. The compact fabricated tests deliberately omit many full-plan fields; they establish the local binding/rejection behavior, not scientific input validity or actual training. Supplying caller-invented expected pins and an invented acceptance cannot create independent authority.

The future audit must emit the documented scientific acceptance schema and independently bind its checkpoint, terminal, prepared and expected lineage fields. Service freeze/identity construction, development descriptors, inference runtime/resource admission and campaign execution remain separate. No actual scientific training or utility is accepted by this review. Optional suggestions: none.

Interface source pins inspected: engine `1e25ceb9b2de9b84666d99cdb6c02087000b9c2bfe6ea493d13dcff1a8fe5a49`; complete CPU-plan bridge `a62662b3ee323ea0ae30685391fab94b00c4e6bb057e710bd799a38317a66aee`; model service `a7735dd35f56d7653a5ff42520db7fd67b98395e62333cccce31583317f7134e`; language policy `4544c3b47b0d92bf18fa8f09d508c1aeb29cd0fa8293940d8453dff701ab466d`; scientific campaign `e3072b8a4c18ef459154fc71ebec8874992d4cee7c829e55b4203cf5111565d8`.
