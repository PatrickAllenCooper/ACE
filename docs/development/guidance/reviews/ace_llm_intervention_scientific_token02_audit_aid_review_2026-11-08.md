# TOKEN02 saved-artifact audit aid — independent static review

**2 required P2 corrections before relying on the aid for the affected shape/deadline conclusions.** No actual TOKEN02 terminal evidence was opened or audited. These findings concern the new checker, not the scientific job or its accepted predecessors.

Reviewed `/private/tmp/ace-token02-acceptance-auditor-preparation-q2cnnygo`:

- `audit_saved.py`: `88ed9bae77d6fb198d4cac0f3de780ea89509389400211d1620c003a850fc320`.
- `HANDOFF.md`: `daac09ac197ac4d09f8c496753ad7c68811b94f0cf8a0c9b6abd99ec4f673106`.
- `reference-pins.json`: `6a9d2fbe59901b2b9e292bdc72a54200e89eac326c0f53673ac3805f28d2b91c`.

## R1 — Derive the fixed numerical width instead of trusting its reported value

`audit_saved.py:206–214` allows any integer `feature_width` from1 through20000, then computes parameter count and every feature shape using that same reported width. This checks internal agreement but does not establish compatibility with the pinned encoder. A different width with correspondingly changed parameter count and shape metadata satisfies these predicates; `validate_plans` does not supply the missing width check.

Pinned `llm_intervention_task.py` (`a90031ad…`), constants at21 and `numerical_features` at192–262, fixes N=8 nodes, A=32 actions, Q=64 queries, H=80 history rows. Its scalar, target, node, menu, history, query and candidate blocks give:

`width = 7 + N + N*(6+N+(N+1)+(N+1)^2) + A*(3*N+3) + H*(4*N+A+3) + Q*(3*N+4) + (3*N+4) = 8891`.

The pinned ranker preparation (`0909428c…`,116–136) then fixes the hidden128 parameter count to `8891*128 + 128 + 128*128 + 128 + 128 + 1 = 1,154,817`. Require this source-derived width, parameter count and corresponding per-record action-count×8891 shape. This is small integer arithmetic over pinned source constants; it needs no numerical package, tensor construction, tokenizer or scientific replay. Preserve the explicit limitation that a valid feature signature is metadata and is not an independent recomputation of tensor contents.

## R2 — Do not compare worker-journal durations as parent-origin elapsed time

`audit_saved.py:276–278` compares every phase `elapsed_seconds` with550 and describes rejection as being outside the original work deadline. Pinned `llm_intervention_model_service.py` (`a7735dd3…`,216–233) creates `Journal.started = time.monotonic()` in the child and writes phase elapsed time relative to that new journal origin. Scientific worker47bbf574… constructs that journal after parent staging and child startup (`child`,448 onward). It is not `started['started_monotonic']`, the parent origin used by owned supervision1c08423b… and the absolute300/550/590 deadlines.

A worker journal starting200seconds after the parent can report a370second phase at parent-relative570seconds; `<550` on that phase and `<590` on parent/resource elapsed values do not detect the crossed work deadline. This is a static consistency example, not a claim about any saved TOKEN02 timing. The actual supervisor source enforces its absolute work deadline; the aid must not represent a check on a different clock as independent confirmation of that enforcement.

Keep phase ordering, finite/nonnegative durations and CPU-only event checks, but label their time origin correctly. Establish original-origin work timing through an explicit authenticated clock/source-control-flow join, or leave that timing conclusion to the handoff's separate source/command/trace audit and report the phase check as worker-relative only. Do not invent a journal origin, infer it from an unaligned elapsed field, or claim exact cleanup timing when the saved receipts do not measure it. The existing startup-marker and parent hard-bound checks remain useful and distinct.

## Other checked boundaries

All supplied reference file hashes/sizes match the manifest; all14 production-source copies present in the supplied inner packet match its captured file pins. Reviewed selected-function global dependencies statically: recipe functions receive the appropriate literals; runtime callbacks receive the expected namespace attributes; collection/plan/phase/accounting functions receive their required bindings. No selected function or source module was imported, extracted for execution, or invoked by this reviewer.

The exposure derivation agrees with the pinned source's feasible action ordering, stop command, strict/tied indexing, stable unordered-command-pair hash and lexical tie-break, one SFT per record, at most one selected strict pair per record, unexposed strict/tied accounting, and complete SFT-before-DPO schedules. Complete record/receipt identity and3584 retained comparisons remain gated by the pinned accepted collection binding. D=0 rejects through the retained plan validator. Prompt formatting, token/command association, recipes,112 adapter shapes/1,146,880 parameters, checkpoint indices159 and159+D, and the source's `one_pass_dpo` meaning align. No token identity is independently re-established without retokenization; the handoff correctly assigns that provenance to the authenticated source/runtime execution.

Runtime reuse matches the saved stage, package/module/native-prefix, duplicate NumPy metadata, cache, generated-provenance and captured-source receipt interfaces. Independent authentication of parent generated-source provenance remains a caller prerequisite, as documented. Lifecycle receipt names/flags, worker-terminal hash, stage hash, process-group closure and zero-work fields match the producer; R2 limits the timing conclusion. The accounting adapter reuses the exact6227 transition preserving43 predecessor rows/88 journals, adding the actual TOKEN02 job/pair, retaining failed costs and requiring44/90. Full scheduler validation, whole-custody/readback, original failure retention, publication/failure-file checks, and final acceptance authorization remain outside this library and explicitly required by its handoff. It has no acceptance-writing CLI.

Optional suggestions: none beyond the two corrections. Do not expand this into a new framework or repeat closed component tests/reviews. A focused static recheck of corrected source is sufficient for this review request; no actual audit, tokenizer, model, tensor or scientific execution is authorized here.

Original supplied tree and independent static pin/shape metadata are preserved in `/tmp/ace-token02-audit-aid-static-review-r5lnfgkx`. Receipt SHA256: `4478b09649434476ee3a5e9da0a8e0cad17ac6bff9bfdecab3224243b31acebb`. Only hashes, AST/source text, supplied reference JSON and integer dimension arithmetic were inspected. No tests, target callables, actual terminal artifacts, remote operations or commits were performed. Author files remain unchanged.

## Focused R1/R2 closure — 2026-11-08

**R1 and R2 resolved; 0 remaining required findings for the corrected audit aid.** Reviewed new source `/private/tmp/ace-token02-audit-aid-r1-r2-z69bc7os/audit_saved.py`, SHA256 `b3804801e460b9aff44465683dc0fb5cb627af299f883f666ec773c94ab0cf8c`. Original source88ed9bae… and initial review67cc6061… are retained unchanged in the author's new tree; the initial findings above remain historical. Reference manifest remains `6a9d2fbe59901b2b9e292bdc72a54200e89eac326c0f53673ac3805f28d2b91c`.

R1 now reads the hash-authenticated encoder's unique AST tuple assignment, requires integer constants8/32/64/80, derives width8891 and parameter count1,154,817, and requires exact integer width/count plus every action-count×8891 shape. `audit_artifacts` invokes this check. Tensor signatures remain metadata only, with no claim that tensor contents were regenerated.

R2 removes the invalid parent550 comparison from worker-journal elapsed values while retaining finite/nonnegative durations, phase ordering and CPU-only events. It explicitly returns `phase_time_origin='worker_journal_started'`, `original_origin_work_deadline_verified=False`, `exact_cleanup_duration_verified=False`, and the requirement for a separate authenticated source/command/trace audit. Startup-marker/original-offset and parent hard590 checks remain. The revised handoff carries the same limits; no clock origin or cleanup duration is invented.

AST delta comparison found only `audit_artifacts` and `audit_lifecycle` changed, plus the two new dimension/shape helpers. Other aid functions are unchanged. Read, but did not run, the two new fabricated methods: exact shape positive and self-consistent wrong-width/count/shape negatives; worker-clock counterexample with explicitly unverified timing, invalid durations/order/events, and startup/offset/hard-bound negatives. Retained `attempt01` reports both passed once,0.140292 childCPU seconds and0.149143291 wall seconds. Receipt SHA256 `4c55acba55c146cdc418a46008feed7384cdda115482c21e4d860d77f592de4b`; test source `2491e3562021bbc55a26fc401332f6533408c75abbefea2ec311465cda76c428`. No old test was repeated by this reviewer.

Independent delta snapshots and comparison receipt are at `/tmp/ace-token02-audit-aid-delta-review-erwwtm_3`; receipt SHA256 `40b6747ec0a9dece4a8ab0ba5c69388eda9add4e29567c1319fb02b5787f0c45`. This closure qualifies the aid's corrected scope only. Full custody, actual runtime/source/command/trace/resource evidence and independent acceptance remain separate. No actual terminal output, target callable, test, model/tokenizer, remote action or scientific execution was performed in this recheck.
