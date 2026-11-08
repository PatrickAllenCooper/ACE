# Scientific GPU training collector stream integration — review handoff

Stable authoring in `/private/tmp/ace-scientific-training-gpu-stream-89i4t7aj`; independent review required before use. The original collector9346be8d, entire original authoring/test tree (including retained fixture bytes), its closed review, and accepted TOKEN integration source/test/handoff/review are copied under `predecessors`. `predecessor-inventory.json` compares every original byte and documents original fixture hardlink counts; snapshots are independent regular files. Original tree, GPU core, packet roster and production code were not changed.

Only transport changes: one bounded1800-second read-only helper stream replaces the bulky per-file remote SCP and second raw-tree local copy. The stream includes whole `launch` (GPU training), `cpu` (TOKEN), `common` (collection02), `failed01`, both separate remote namespace files and all empty directories. The accepted helper performs two independent exclusive materializations. Exact complete before/after file/directory maps, aliases, upstream pins, accounting and journal observations are compared before materialization. The original post-transfer authority/error/resource/GPU-evidence/44-job90-journal/accounting/cleanup/failure disposition code is byte-identical. Both controller locks, terminal-job gate, input/intent/result/source/commit joins and final two-copy capsule readback remain.

The complete independently pinned accepted TOKEN capsule is still separately authenticated and copied to both `upstream-cpu-custody` trees, including its captured archives, failures, receipts and upstream raw evidence. Those local disk copies and small coordinator copies remain; they do not require another remote transfer. No model/adapter tensor or scored outcome is decoded. Opaque checkpoints and all failed attempts stay in whole custody. Memory/CUDA/progress signals remain metadata, not utilization percentages or acceptance.

## Standalone interface additions

Keep all existing collector CLI arguments, including independently pinned full packet commit, collector/self/review, packet/job/result/local-intent and accepted CPU-capsule inventory. Add exactly two required arguments:

- `--stream-helper-repository-path`: repository-relative committed path of accepted helper bytes.
- `--stream-review-repository-path`: repository-relative committed path of accepted helper review bytes.

Place `stream_custody.py` and `stream_custody_review.md` adjacent to this `pull_terminal.py`. Their exact independent pins are constants: helper `fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a`, review `5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371`. Both adjacent files must also match bytes at the supplied repository paths in the same full `--packet-commit` via `git --no-replace-objects show`. The new collector and its new independent review must likewise be committed under the existing separate collector interface. This is outer-only binding: no core, child, scientific input roster, packet JSON, launcher or monitor change is needed. Parent supplies the actual reviewed repository paths/commit; no future acceptance is synthesized.

Helper loading occurs only after existing self/review/packet/Git/copy/CPU-capsule admission and the first conclusive terminal observation. Both terminal capsule directories are claimed exclusively; their `remote` children remain absent for helper materialization. Helper source/review, request, exact emitter argv, before/after observations, archive, producer stderr, capture and materialization/failure receipts are retained in both capsules. Transport errors retain original exceptions and available partial bytes, without retry, replacement or acceptance. A second-copy failure remains explicit; original evidence is never deleted. Conservative free-space checks include raw materializations, both full retained archives and coordinator/CPU-capsule bytes. The helper's unchanged member/byte/path bounds can reject an oversized real capsule; they are not silently enlarged.

## New checks and limitations

Six NEW fabricated integration methods passed once: complete four-root/two-claim two-copy transport and independent archives; CPU-root/journal/accounting drift rejection; partial capture preservation without retry; materialization failure with retained archive/partial checkpoint; standalone full-commit/helper/review binding negatives; and unchanged original gate/remote-observer/post-transfer authority checks. Capture and Git were mocked. The exact accepted emitter/materializer ran only on tiny hand-built local bytes. No former helper, TOKEN or GPU collector test method was rerun. No SSH, remote read, build, submission, tokenizer/model, actual scientific response, source freeze edit or commit occurred.

`test01` retains source snapshots, all fixture outputs and injected partial-failure evidence, stdout/stderr, runner and costs. All six methods passed (unittest0.165s); process wall `0.262893834` seconds, child CPU `0.20395499999999997` seconds. No failed test attempt occurred; deliberately injected transport failures remain in fixture custody. No network-throughput or actual runtime qualification is claimed. Actual accepted TOKEN inputs, complete44/90 terminal accounting and independently reviewed eventual invocation remain required.

Collector SHA256: `1178743d3c8c32e341d4621071fd7a7235aba38e10c18d1809e7cdb8f4cfe821`

Test SHA256: `9fd03b0479b08bf662dc7f068f622c6c6aa512f3ceb3f0e33d7e84046f7d7991`

Test receipt SHA256: `b67325d480fb13a40f0e6746d7afa66743a6032976fdae8cdabb71e77c6b2c55`

Diff: `pull_terminal.py.diff`. Machine handoff: `handoff.json`. Preserve this tree unchanged after review handoff unless a specific finding is returned.
