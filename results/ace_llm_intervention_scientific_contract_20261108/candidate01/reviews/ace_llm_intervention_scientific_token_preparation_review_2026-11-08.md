# Scientific token preparation source review

2026-11-08. **Required findings: 0.** Independent static review of the new scientific CPU preparation entrypoint and its fabricated tests. This is source acceptance, conditional on independently accepted complete collection inputs and a separately reviewed concrete execution packet; it is not execution or scientific-result acceptance.

Reviewed bytes:

- `scripts/research/prepare_llm_intervention_scientific_training_cpu.py`: SHA256 `225c89d322d7c312da5d2e242db068129cb83f6e0b728a0c6c211dcee0804838`.
- `scripts/research/test_prepare_llm_intervention_scientific_training_cpu.py`: SHA256 `6d029232508d10410ff1bf34218957c87063655a9b7ec3c9c7938c771ca6d800`.

Both were captured before review under `/tmp/ace-scientific-token-static-review-pvsc0jcx/`, with `capture.json`. Final reconciliation found the working files byte-identical to those snapshots and Godel's final handoff pins. No author delta remained to review. The integration contract was `/tmp/ace-scientific-cpu-integration-map.md`. Closed helpers were consulted only for the new call signatures, serialized fields, captured-import behavior and supervision boundary; their component reviews/tests were not repeated.

## Input and exposure contract

`packet_shape`, `authenticate` and `collection_binding` require the original pinned proposal, its separate twelve-source protocol map, the collection's thirteen-source map, and an independently pinned accepted collection receipt. The receipt must bind the exact collection object, including freeze and realization-acceptance pins, world/configuration pins, namespaces and the complete original file inventory. This worker relies on that independent acceptance for execution provenance and original journal accounting; it does not manufacture acceptance or treat a complete terminal as sufficient authority.

The ordered 32 training worlds expand to exactly 160 world-major record/receipt pairs. Plan, inventory, terminal, raw bytes, receipt associations and every ordered candidate pair are checked, retaining 3,584 comparisons and 32 terminal-stop records. Per-world accounting must be complete with no unknown returns, and actual totals must reconcile within the fixed caps. Original complete custody is hashed before and after successful preparation, including journals. Reads reject symlinks/nonregular files, escaping relative paths, pin changes and tree changes. Output, input, tokenizer and collection trees must be disjoint. No collection/realization call, development-payload processing or response-generation path is introduced.

`prepare_pipeline` saves the existing fixed exposure before tokenizer construction. D=0 fails without rescue, recollection or a changed recipe. Otherwise the schedule is 160 SFT steps followed by D strict DPO steps, D=1..128, with checkpoints `(159, 159+D)`. The complete-collection bridge recomputes the same deterministic exposure and must match it exactly. Unselected and tied pairs remain in the raw records; `one_pass_dpo=True` is correctly not required.

## Prepared artifacts and execution boundary

The bridge receives explicit `qwen_screen_singletons`, using the existing screen recipe and `canonical_selected_unpadded` path. The worker checks the 112 adapter shapes/1,146,880 parameters, full raw-record and tokenizer bindings, token metadata, nonempty complete command IDs with EOS, prompt+32 <=8192 and the 4,194,304 stored-token cap. Existing bridge/engine validation supplies exact legal-command membership and label alignment. No truncation or equal-length-pair substitution is added.

The numerical preparer receives identical records, receipt pins, roster, exposure/digest, checkpoint indices and outer packet provenance, with `numeric_recipe(artificial=False)`. Its existing API verifies the fixed numerical schedule and derives feature pins using hidden128/seed109/CPU FP32/continuous Adam metadata. It does not initialize or fit a ranker. The two outputs retain their independent plan pins and all qualification flags remain false.

Only six authenticated nonweight tokenizer files are permitted in the tokenizer directory. The seven-file accepted model map supplies tokenizer identity without staging its weight file. Loading is local-only with `trust_remote_code=False`. The caller must supply the independently accepted model-map pins; map shape alone is not model acceptance.

The entry authenticates the flat packet/source closure before project imports, uses captured bytes for the expanded project namespace, and rechecks inputs after work. The existing archive staging, runtime provenance/cache/import checks and CPU environment are reused. The new caller-supplied resource contract requires explicit CPU-only Slurm identity, requested/allocated CPU and memory, account/partition/QOS, wall time and no requeue/restart. Absolute startup/work/cleanup/hard deadlines share the parent's monotonic origin. The reused supervisor owns one child process group and confirms natural exit and cleanup before parent artifact validation and publication. Runtime receipts and actual captured project pins are checked again by the parent.

Exclusive directories/files, retained candidates, failure receipts and interruption receipts prevent an interrupted output from becoming an implicit retry. The original collection is never rewritten. Full-batch readers replace the old artificial small-fixture assumptions: 64 MiB record/receipt batch, bounded 2 GiB original tree and 256 MiB output reads. These bounds do not demonstrate that the full scientific stage fits any proposed time or memory allocation.

## Retained test evidence and remaining gates

Read `/Users/pat/.codex/tmp/ace-scientific-preparation-tests-vwnpkj34/HANDOFF.md` and the existing `attempt02` transcript: 16 fabricated methods passed, return code 0, no source changes; 0.746272583 seconds wall time and 40,943,616 bytes peak RSS on macOS. The retained source-pin map matches current bytes. Pipeline APIs, tokenizer, staging and supervisor are doubles where stated: this evidence covers glue and negative cases, not actual tokenization or a staged end-to-end run. The earlier passing attempt and failed AST constant-resolution inspection remain preserved; neither was rerun.

Evidence pins: handoff `a13cb8a1ded0e30d642bbad79a6c4e5ca34b1907d2fbdfa918678975188a0ba5`; `attempt02/run.json` `bfcc80e08be26af068d16a5c2be69891c81ccf032d48f314d7a78c113cc0dadb`; `attempt02/stderr.log` `688343ecefca9b994e6d7d1363c996bd173e5a7022795729dcb1a87870120905`; `attempt02/source-pins.json` `feb88c87958fa07acdec904270be1785a389f227a0ae543ce42df364dbc7262d`.

Before execution, the parent still owns committed entry/source closure and bootstrap pins, independent collection acceptance, exact tokenizer pins, bounded resource/deadline admission, once-only launch coordination and complete custody. Actual preparation must subsequently receive independent result acceptance. No tests, project entrypoints, models, tokenizers, numerical imports, tensors, replay, remote operations or allocation were executed during this review. Author files and original acceptance artifacts were not edited. Optional suggestions: none.
