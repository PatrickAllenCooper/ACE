# Actual GPU02 transfer-continuation review

Required findings: **0**. The completed saved records support one successful, unambiguous submission of **35516125**, recorded at **2026-11-10T02:42:22.626671Z**, following the separately reviewed transfer continuation. This is an operational lineage review, **not scientific execution acceptance, GPU utilization qualification, terminal custody acceptance, or permission to retry**.

## Exact lineage

| Object | Binding |
|---|---|
| Actual driver | `a6996da4b7b4998fa0f576097a52cd72e25312d05697fc45ca83e25fccfab8c5` |
| Driver/source-review commit | `58df1d1b05d979e1f80e95c86cca0be530382174` |
| Reused independent source review | `5934324136123756b5233bf77945d9578ab223be5834b7751734ef7a3ce4baf2` |
| Original packet commit | `463e5b1140e8aa957e07636c603c3057da81e7d6` |
| Original outer packet | `25fb597c6f475fa0bbfe924062bc439bb197ca0b646cecdae2be1a080010d684` |
| Original failure/probe commit | `b476bc40bda7abb789416b37df6910a2196334aa` |
| Actual local canonical intent | `da8a38edafa5945c30b6c3feaa5c4bd86c578b4247b1fd8669b65b0d56ebd6fa` |
| Returned remote intent pin | `dd7148afb1fcad65834c717b39c1d4728860277e9e0ef75278c65378e6de0d1b` |
| Actual canonical result | `ab4719a46cb68901829bc5617d1321a94021ffca7be89046554c69e9518f51fc` |

Local Git reads bind the captured driver, independent source review and design review to the driver commit; all 200 original packet/input objects and original packet review to the original packet commit; and all five original failure/quiescence proofs to their separate proof commit. Both original packet copies still match those pins. The original failed SCP records, controller-release record and readiness bytes remain unchanged; the continuation does not reinterpret their failed/ambiguous status.

## Actual preservation, completion and submission

Both operation copies match for every common completed record, including source, independent review, four requests/transports/stdout/stderr, preservation acknowledgment, partial bytes, completion, pre-submission, storage references and terminal record. The invocation records also match their separate external mirror. The external operation additionally contains its explicitly referenced `storage-operations` tree; it is not falsely described as a second identical local tree.

The preserved partial is **2,392,064 bytes**, SHA `7b8939e8509c6849203e0eb3daad0817aba743dc163fab1580de2505269cf66e`. Its returned bytes, both independent local files, both acknowledgment paths, the original complete file prefix and the projects preservation record agree. The projects location is `/projects/paco0228/ACE_evidence/gpu02-transfer-continuation-20261110-01/partial.original`. Preservation record SHA is `4534dec735162aacaf219ec7653b72fcc4217bdd4d14111f9a98e80d54a7abaf`; local acknowledgment SHA is `c819ebd675d750c6f716d9d1a5ad472539ee639ea468a52f5d568d3dc0236d7b`.

The completion request is exactly the original **167 missing members plus the one truncated member**, 168 members / **414,954,743 bytes**. Every size/hash matches the original packet. The completion result preserves the 33 previously matching files, including separately authenticated readiness. Completion SHA is `b235ed6e7412ec88c656483a2246f8dbfbc0c188ae0e703f387b73278e2ab2b6`. Its hash is consumed by preflight and submission, and the complete local preservation acknowledgment precedes completion.

Pre-submission SHA `a84785fa6e9e9eee08983a77da6c234a6ed2d3c57cc72d56f1f6cdd6b890ced6` binds a fresh exact **47-row accounting / 96-path journal** reconciliation and verification of all 200 packet/input hashes. Independently summed closed costs are **30,241 CPU seconds / 2,750 GPU-slice seconds**. The unchanged candidate reservation is 11,520 CPU / 2,880 GPU seconds; the single 3,600 CPU preparation allowance yields **45,361 CPU / 5,630 GPU**, or 47,761 CPU including the prospective 2,400 CPU finalizer allowance. These are reservations, not GPU02 actual costs.

The captured request preserves the full 72,372-path historical input map, 35,751-file CPU launch roster, directory rosters, original failed-GPU inventory, namespace pins, accepted runtime/model metadata and recovery authority. Fresh readiness records the frozen output still absent and original failed GPU bytes verified. Resource, limit, source, storage, accepted-CPU and recovery fields match the original packet and canonical intent: 4 CPU, 15,360 MiB, one A100 3g.20gb, 2,880 seconds, no requeue; work/final 2,820/2,860 and session cap/reserve 1,200/600. No namespace, worlds or training updates were introduced.

All four transports returned zero, unambiguous, with empty stream errors, joined threads and owned-process-group ESRCH observations. Preserve/preflight/submit took 753.373 / 253.431 / 257.304 seconds within their 1,230-second transport bounds; completion took 67.005 seconds within 630 seconds. Commands bind the exact driver through `python3 -I -B` and its independently supplied SHA. The reviewed driver places both local controller locks around the operation and releases them in `finally`; the saved zero exit and completed gated phases support that control flow. This reviewer did not acquire or query locks.

Canonical local intent exactly equals the submit request, with its four continuation pins intact. Submit stdout, canonical result in both copies, invocation stdout and terminal record all identify the same single job and remote-intent hash. The terminal record remains `acceptance: false`; no automatic retry or second submission record appears in this completed operation.

## Storage, costs and limits

All 98 current storage-operation references resolve to pinned completed records. The final inventory `4eaa1586be8da784ddc312fa05ca7d8acd08f161e840a82b4b35d3b681658d3f` matches its actual current-operation membership/bytes. Both archive-verification receipts bind the previously accepted full CPU archive and backend `4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f`; both opaque adapter fetch receipts bind the existing packet pins. No tensor/model/private-result decoding or ancestor scientific audit was repeated.

The invocation reports 121.452504 child CPU seconds, 1,567.047469 elapsed seconds and 2,762,096,640 peak child RSS bytes. This covers the recorded local preparation/transport children, not whole-study or remote CPU accounting. No GPU02 elapsed cost is inferred here.

The returned remote-intent hash and projects preservation record are authenticated saved execution outputs. This review did not independently read their live remote paths. Full original remote intent (including final fresh-readiness augmentation), `fresh-before-preservation.json`, renamed partial, staged/completed members and both control/output roots remain mandatory terminal collector/audit custody. No current scientific result, telemetry, terminal state or outcome was inspected. Existing terminal collector and scientific acceptance gates are unchanged.

The evidence includes the first reviewer-inspector attempt: it omitted the separate readiness member from its roster predicate and requested a nonexistent archive-size field. Both inspector errors were corrected against the documented metadata contract, with completed checks reused. No production correction or test rerun occurred. Final saved-metadata checks and remaining joins have zero failures. Optional suggestions: none.
