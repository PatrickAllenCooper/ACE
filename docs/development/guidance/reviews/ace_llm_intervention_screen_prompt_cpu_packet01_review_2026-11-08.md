# Screen prompt CPU01 concrete packet/coordinator review — 2026-11-08

**Required findings: 0 for this concrete unsubmitted packet and captured coordinator.** The packet is consistent with the reviewed worker and its bounded CPU-only request. The launcher still requires a commit containing the exact packet and this independently pinned concrete review, followed by fresh guarded readiness checks. This review does not establish current remote readiness, actual tokenization, execution success or scientific acceptance.

Reviewed packet SHA-256: **`0979cb607643b4b5d01fbf56c6e701a3e474d4d2bbda1e1c9f4e78e91251ccfa`**.

Captured anchors:

| Artifact | SHA-256 |
|---|---|
| `launch_once.py` | `913b0d3945718644b0436af380ef118403c40c49f265bcd5a2d6b45217430284` |
| `build_packet.py` | `62cdc4d789c2ce4b8285bc0e29640a0090137c1c3c9a50b15c90c84b951206df` |
| `job.sbatch` | `1becd0c11a19de4bfc0618c40a9c9a02e562b8f1c243d85c79967af4a17ae444` |
| `packet-metadata.json` | `9b617cfd59d316f69fbe69b17b20fff47f4853dad76b0c01d88bdfecd527f5e7` |

Scope was the new `results/ace_llm_intervention_screen_prompt_cpu_20261108/launch01` packet, its external mirror, captured builder/job/coordinator, preserved build attempts, and narrow predecessor/source bindings. Inspection used local text/JSON, hashes, AST parsing, shell argument parsing and read-only Git objects. No builder, coordinator, worker, saved validator, tokenizer, model, numerical library, test or remote command was run. Existing component reviews were not repeated. Only this review was written; no commit was made.

## Concrete closure

Both packet copies contain exactly **277 files / 3,357,074 bytes**, with identical membership, sizes and SHA-256 values, no symlinks and no metadata-pin mismatches. The external root is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-prompt-cpu-20261108-01`.

The worker packet has exactly its reviewed keys and **264 named inputs plus `packet.json`**. All 23 source/helper origins match Git commit **`21b187887ecb62bdc4b3515f3d27b79b22769d2b`**. The entry matches reviewed SHA `c85747e3958165582fa8dcce6a60ea91c4c050acdad5f61fcccad3aec43c2533`; captured source review SHA `87de267ef29a0b088fae212c9ecd54fbfd8346bd5d6b4a58f4f1d85aa020da62` matches that commit's review bytes. The cleanup helper retains its accepted committed origin and pin.

All **240 envelope/prompt files** match their independently pinned frozen manifest and committed fixture originals. Coverage is exactly 3 graphs × 2 cost menus × 2 public noise variances × 2 numeric profiles × 5 history shapes = **120**. Manifest raw/canonical hashes remain distinct and correctly bound. The largest input is 144,464 bytes and the packet is 40,039 bytes, within worker read bounds. Maximum prompt size is 9,879 bytes, which is not a token-fit claim. Scope/request remain `artificial` / `all_120_frozen_fixtures`; no fixture is replaced, omitted or truncated.

The job's decoded `-I -B -c` bootstrap is byte-identical to the captured diagnostic bootstrap. Its entry path/pin, packet path/pin, full source commit and output agree with the concrete packet. The destination is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01`, with flat `inputs`, worker output `cpu01`, and explicit `logs/%j.out` / `logs/%j.err`.

## Accounting and once-only coordination

The accepted finalization receipts are unchanged copies at their provenance pins. Deriving predecessors from the retained finalizer intent plus its actual **61-second, 2-CPU, zero-GPU** terminal row gives exactly **36 unique terminal jobs / 74 intent-result journals**. The finalizer result authenticates its remote intent. Independently summed closed totals are **10,368 allocated CPU-seconds / 1,483 GPU-slice-seconds**. Admission is **10,368 + 3,600 cumulative preparation allowance + 600 new CPU reservation = 14,568 / 57,600 CPU-seconds**; GPU admission stays **1,483 / 7,200**, with no new GPU reservation.

The job, worker packet and coordinator agree on **2 CPUs, 7680 MiB, 300 seconds, `ucb736_asc2/acpu/cpu-normal`, zero requested GPUs and no requeue**, with the unchanged 120/260/280/290 deadlines. This remains a bounded measurement request, not a full-screen throughput projection.

Before remote mutation the coordinator requires both existing nonblocking controller locks, rechecks exclusive local attempt markers, and records its controller claim. It uses the existing shared SSH master without reconnecting. Fresh readiness checks require no related active job; exact prior scheduler rows and journal pins; account/QOS/partition and memory-ratio availability; unchanged complete predecessor launch/common inventories; runtime/tokenizer bindings; scratch ownership, space and write/read/cleanup readiness; and an unclaimed new root/output. Bulk archive and weight-byte authentication remains in the allocated CPU stage, without weight deserialization.

After transfer it verifies every captured pin and exact input-directory membership, then calls only the worker's stdlib admission path. Immediately before submission it repeats packet/readiness checks, rereads pins and rechecks the queue. Durable exclusive local and remote intent journals precede the **single `sbatch --parsable` call**; `SBATCH_*` environment overrides are removed. Failure, timeout or ambiguous response preserves evidence and attempt markers and provides no retry loop. A returned job ID does not confer capacity/scientific qualification. No launch/controller/submission markers were present in the reviewed packet copies.

## Preserved build correction and limits

Build-preparation01 retains the original builder and a candid failure summary: `ast.literal_eval` rejected symbolic dictionary keys before destination creation. Its stderr is summarized, not retained byte-for-byte. The only builder delta resolves dictionary-name keys from unique earlier uppercase string-literal assignments before `ast.literal_eval`; it does not execute source expressions. Build-preparation02 retains the exact captured corrected builder, full stdout, empty stderr and execution receipt: exit 0, **2.744232083 seconds wall**. Both preparation directories match their repository mirrors. These are build observations, not tokenizer or allocation evidence.

No optional source change is requested. Monitor/puller work is outside this captured packet review. Actual CPU tokenization, live resource/runtime/cleanup results, complete terminal custody and independent result acceptance remain pending. The separately identified scientific DPO pair-length gate is not established by prompt-capacity qualification.
