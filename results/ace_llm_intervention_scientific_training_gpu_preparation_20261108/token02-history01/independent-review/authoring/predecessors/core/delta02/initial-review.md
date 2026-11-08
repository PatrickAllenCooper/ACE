# Scientific training GPU recovery coordinator review — 2026-11-08

**1 required finding in the initial supplied core. Not ready for a concrete build until R1 is corrected and rechecked.** Independent static source/interface review of `/tmp/ace-scientific-training-gpu-recovery-lineage-6wvo33h9`:

- Builder: `f73835f83c2cabc22162a4f8093b84cc02d19d2d0a78962137fdbfdd025e76f1`.
- Launcher: `6971ac253180c0164591dcf01b6a46381e246bda02a798854168c07a06310153`.
- Monitor: `3fc1fee1a4ce36399bcb5eac762e67c5c6de56256347e51331c6f421eb38ef61`.

Scope is new coordinator admission, source/producer interfaces and recovery history. Closed training algorithms, independent failure audit and old tests were not repeated. GPU collector is outside this review. Actual collection02 and CPU token acceptances and43/88 accounting remain future authority.

## R1 — Wrong frozen token closure key prevents building

`build_packet.py:420` uses `f['stage_closure']['stages']['tokenize']`. The exact pinned scientific freeze15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d has closure-stage keys `CPUcampaign`, `GPUcampaign`, `collect`, `finalizer`, `realize`, **`token`**, `train`. The separate top-level stage roster calls the token stage `tokenize`, but that is not its closure-map key.

Consequently `make_packet` raises `KeyError('tokenize')` on the real frozen input after successful future acceptance checks/source capture and before destination publication. Launcher reconstruction calls the same function, so the same mismatch blocks live admission. Correct only the coordinator lookup to `token`; do not change the immutable freeze or production source. A focused actual-frozen-metadata lookup/closure check is sufficient for this correction; no training/tokenizer execution, old tests or scientific packet build is needed. The supplied three recovery-lineage tests exercise metadata transitions, not this actual closure lookup.

## Other reviewed boundaries

The recovery delta explicitly pins collection amendmentfa94858d…, its exact reviewdb2a672a…, failed collection acceptance00c4557a…, no-response proof59c21d63… and failed41/84 baselineeeec362d…. It preserves science freeze15602614… and sourcea9346022…. Failure runtime and partial node-local cleanup uncertainty remain false/unverified; unknown or nonzero failed scientific counts reject. The separately pinned actual future collection acceptance/result/disposition/inventory must extend41/84 to42/86 with all prior rows/journal hashes and actual2CPU×elapsed costs retained. Collection02 roots, resources and deadlines must match the separate amendment, without rewriting original realization or namespace authority.

The future token outer packet must capture those same operational/failure/collection bindings and the accepted42/86 baseline. Its inner accepted preparation packet and exact18-field CPU acceptance remain compatible with the frozen GPU supervisor: scopepolicy_train, zero findings, complete custody, runtime qualified for that CPU execution, owned cleanup, explicit prepared/plan/exposure/numerical-plan pins, and false training/scientific qualification. Collection and token jobs are distinct. Original collection packet inputs/journals and the complete collection output map are joined to the independently supplied collection transport inventory. All CPU outer and inner input pins, original prepared/terminal/journal paths and saved terminal summaries are joined to the CPU inventory before packet creation. Neither inventory is manufactured by the coordinator.

Prepared bytes must match the independent CPU acceptance before decoding; full CPU transport/inventory and terminal checks follow before construction. Saved complete160-record singleton plans, D1..128, fresh seed109, unchanged exposure/plan hashes, all3584 retained pairs and prompts at most4744 are required. Longer prompts reject rather than truncate. No tokenizer, base model, response generator, world generator or scientific algorithm is called by these scripts. The actual CPU acceptance remains the independent semantic authority; this authoring review supplies none.

The new baseline must extend actual42/86 to43/88 while retaining every predecessor allocation/journal and actual token CPU charge. Remaining frozen stage caps plus the single3600CPU allowance must fit57600CPU/7200GPU. Source capture pins the unchanged29-source GPU roster plus archive helper, full67-source metadata map and shared CPU/GPU files; R1 is the concrete closure-key incompatibility found in that path. No other closure or packet-key conflict was found by this read-only inspection.

GPU profile remains4 requested/maximum4CPU,15360MiB,one `a100_3g.20gb`,2700seconds,ucb736_asc2/aa100/gpu-testing,no-requeue. Deadlines240/2620/2630/2645/2670,progress180 and telemetry90 match the frozen stage and supervisor. The packet makes no new scaling or throughput guarantee. CPU preparation precedes this allocation. Actual staged runtime/model admission, independently evidenced GPU work and retained owned cleanup remain the unchanged supervisor's responsibilities, not facts established by authoring or exit status.

Both launcher locks, committed concrete packet/review and mirror checks, exclusive roots/intents, full live43/88 accounting/journal reconciliation, source/model/runtime metadata checks and one sbatch are retained. Both live passes recheck original collection/CPU input pins. Timeout/ambiguity retains evidence and prohibits blind retry. The bounded monitor reads the exact training marker and `gpu-sample-*`/`progress-*` paths emitted by the supervisor; file presence/hashes do not themselves certify utilization or success. It issues no job mutation. The local builder remains dependent on the parent's existing lock discipline.

## Evidence and limits

Initial source snapshot, handoff/diffs and fabricated-test sources are preserved at `/Users/pat/.codex/tmp/ace-scientific-gpu-core-review-t13smsyx`. The actual freeze key was inspected as JSON; target code was not imported or executed. The author's retained three-method lineage evidence and syntax evidence remain original and were not rerun. No old test, model/tokenizer/runtime, scientific draw/response, remote action, allocation, packet build, source edit or commit occurred. This review does not assess the unfinished separate collector or create any future acceptance.
