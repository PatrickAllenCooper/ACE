# Collection02 coordinator source review — 2026-11-08

**0 required findings in the final core.** Independent static delta review against the accepted collection01 coordinator. Final stable pins:

- `build_packet.py`: `a4e58643fee5feef49d52d332cf7fd01ded7e02da9cadc37b5a928b4a4adbb6e`.
- `launch_once.py`: `0286fbf3f53b8806f0a54d297d4d92bf24b9d8bdb03e5fc045b3681bd0b4a61f`.
- `monitor_once.py`: `b621addc3df3af1013543189dac03941f5cf3a131c8a688351e7c66cf48658d6` (unchanged).

Source directory: `/tmp/ace-scientific-collection02-authoring-y5fc_jnn`. Author final handoff SHA256 `1125fada16afc7cb85501af37eaa159f6d10b04a999c9572b1111dd27406196a`. This review covers source authoring, not an actual built packet or execution. The parent may construct the local metadata packet under the existing controller discipline; exact packet inspection remains required before launch.

## Authority, failure and consumer joins

The new consumer retains original freeze15602614… and sourcea9346022… authentication and checks the original collection stage against its original120/850/875/890 tuple before applying a separate operational layer. It does not rewrite the frozen stage. `validate_collection_recovery` reconstructs the exact amendment object with independently supplied pins, fixed collection02 root, failed job35407428/root01, unchanged resources, new300/850/875/890 tuple, original claims, accepted realization and mandatory one-submission/no-replacement/no-automatic-retry flags. Unknown amendment fields reject. These semantics match exact authorityfa94858d… reviewed separately atdb2a672a…; the authority's original design-review pin2d472… remains intact.

The consumer matches the actual published failure interfaces: acceptance00c4557a… uses its existing `evidence_pins`, and proof59c21d63… has no invented schema requirement. Both must show the positive source/traceback pre-worker boundary and integer zero scientific attempts/returns/costs, records, pairs and updates; additional proof model/tokenizer/policy counters must be zero. Unknown counts reject. Failed runtime qualification and unverified partial node-local cleanup remain false; neither is silently upgraded. The independent proof is consumed without rerunning the failure audit.

The failure baselineeeec362d… must retain all40 predecessor rows and82 journals plus failed35407428, yielding41/84, actual252 additionalCPU, closed11,386CPU/1,601GPU and no outstanding reservation. Both original inventory transports must agree on the complete146 raw objects/2,336,700bytes, directories, accounting and journals. The supplied complete local failed raw tree must equal those pins. No failure-only subset or missing authority is repaired. Full unchanged remaining-stage caps and the new1,800CPU reservation remain within the global ceilings.

## Unchanged production and scientific inputs

Git capture and source authentication remain the accepted collection01 mechanism: fixed source commit,23 production files, original12 scientific pins and the unchanged supervisor/collection entries. The new builder additionally requires every generated supervisor input except the path-bearing collection admission to equal the original failed attempt's input byte hash. This includes all32 training descriptors, proposal, realization terminal/acceptance, sources and inner collection packet. The inner packet is regenerated only from the same pinned inputs and must remain byte-identical. Development payloads are not opened or staged; full56 identity metadata is retained separately.

Only the collection admission output, supervisor output/deadline/path and associated admission/outer attempt metadata change. The supervisor still receives the exact60-file input roster. Original32 training systems,160records/all3584pairs,9216attempt ceiling and10240response-cost ceiling remain unchanged. Production source, runtime archive/manifest, numerical/scientific mechanisms and original namespace authority are not modified. The900second batch still requests2CPU/7680MiB/zeroGPU, asc2/acpu/cpu-normal, one node/task and no requeue, using the authenticated source bootstrap. Startup300 and unchanged850/875/890 offsets retain one absolute origin.

## Once-only launch and read-only monitoring

After authenticating metadata, `admit` switches only operational locations to remote collection02, repository launch02 and external collection-02. Existing destinations/attempt markers reject reuse. Exclusive two-copy publication and readback preserve any build failure. As in collection01, the parent must hold both controller locks for the local build invocation; the builder itself is not a lock wrapper.

The launcher holds both locks for committed packet/review authentication, both local copies, two live reconciliations and the single sbatch. The active baseline is explicitly the failure41/84 baseline, not the prior realization40/82 record. Before creating the exclusive new root, only the accepted failed collection01 root may preexist. Both readiness passes verify all prior rows/journals, no active ACE job, site/account/resource feasibility, runtime manifest/archive metadata, both original claims, accepted realization terminal and32 training descriptors, plus every failed01 raw object's byte hash and exact directory membership. The failed root and claims are read-only.

The intent explicitly carries amendment, failure acceptance, no-response proof and failed-job pins. A fresh reconciliation and full input readback precede an exclusive remote intent and one sbatch. Existing attempts and ambiguous transport results stop without retry, preserving evidence. Timeouts remain bounded and partial bytes/text are normalized for retention. Full archive byte authentication remains the unchanged allocated stager's duty; login metadata checks do not claim otherwise.

The unchanged monitor obtains ROOT from the pinned current packet, requires the original unambiguous result/intent association, and observes scheduler status plus collection/worker/task marker hashes in an exclusive local evidence directory. It neither decodes private response contents nor controls jobs or manufactures acceptance.

## Collector boundary and retained evidence

An initial scope observation noted that the builder now includes `pull_terminal.py` in the pinned/transferred root-file roster. At that time final collector stability was unconfirmed. The author's subsequent final handoff and parent confirmation establish its stable pin `5d2bdfdfb47d1d728c5d3d5823f322cd378804979de1dd084a77f7e5ddf0974e`. It is an inert root file, not a child input or invoked by batch/bootstrap. This closes the unfinished-source dependency for building; it does not review or authorize collector execution. Collector behavior receives a separate review, and the concrete packet must bind the exact declared bytes. No core source correction was required.

Initial snapshot, final handoff and existing new metadata-only evidence are retained under `/Users/pat/.codex/tmp/ace-collection02-coordinator-review-zw2ci23v`. Final core bytes match the reviewed snapshot. Author metadata receipt32fa9a5f… records only the new recovery-metadata join and syntax check, with returncode0,0.1099095wall seconds and0.106259childCPU seconds; it was read, not rerun. It is not independent execution evidence. No project callable, old test, descriptor re-audit, model/tokenizer, runtime, scientific response, remote command, packet build, submission or commit was executed by this reviewer. The closed failure audit, production sources and original packets remain unchanged.
