# Downstream compact custody D1/D2 — focused independent recheck

**1 required finding remains (D3, P2). Original D1 and D2 are resolved in the reviewed correction.** No author file was edited and no target callable, test, build, backend/remote operation, model, tokenizer or scientific work was executed. This is a static delta review, not actual custody or scientific acceptance.

**D3 — create or require the missing stage parent before claiming the exclusive build preparations.** GPU build_packet.py:627–630 and finalizer build_packet.py:873–876 construct the repository preparation as LOCAL.parent / storage-build-preparation01, then call path.mkdir(mode=0o700, exist_ok=False) without creating its parent. The first path in the loop is the external preparation. Both actual repository stage parents are absent: results/ace_llm_intervention_scientific_campaign_gpu_20261108 and results/ace_llm_intervention_scientific_finalization_20261108. Thus a first build reaches FileNotFoundError after claiming the external preparation but before registering its operation intent; the retained external directory then correctly prevents an implicit retry. The later packet root mkdir(parents=True) is unreachable. Support the absent ordinary stage parent (or explicitly establish it before any preparation claim), while retaining canonical-path checks and exclusive preparation leaves. Cover this new first-build path and existing/aliased leaf rejection with a narrow fabricated case. No actual build is needed. Godel was notified directly with these exact locations.

D1 closure: GPU collector now acquires both named nonblocking locks, checks both destination capsules and the operation root, and exclusively claims its capsules before start_storage_operations/configure_packet_storage/admit_cpu_bytes. Finalizer does the equivalent inside coordinator_locks. Local source/packet authentication remains before transport. The diff moves the transport-capable work without weakening original source, terminal, intent, accounting or resource checks. Existing-operation and lock failures cannot reach a backend method.

D2 closure: OperationEvidence exclusively records intent.json and publishes its pinned identity to both caller evidence roots before transport. Every verification/input read and the whole initial binding records a numbered inventory/reference, including partial fetch or initial-verification failure. Manifests hash the complete operation directory, independently preserve original records, and distinguish completed/failed status with false acceptance/retry flags. The whole-binding wrapper covers failure before Inputs enters ctx. Publication failure retains the original manifest/reference; a secondary recording failure preserves the primary exception and attempts an explicit diagnostic. Builders use durable external/repository preparation destinations, launchers register before configuration and include the intent as a once-only marker, and collectors publish references into both newly claimed terminal capsules. Launch/collector registration and failure paths preserve primary errors and do not imply successful publication. D3 concerns creation of the new builder destinations, not the resolved reference protocol.

All operation directories must remain retained originals. Pinned references are not copies of the referenced bytes; future custody reviews must follow and verify them. No recursive predecessor raw copy or scientific-child input was added. This review makes no new authority-schema requirement.

## Exact reviewed source

Author tree: /tmp/ace-downstream-compact-d1d2-1d_57cvu/implementation. Snapshots are retained in source/ beside this review.

| Source | SHA256 |
| --- | --- |
| finalizer/build_packet.py | `2f6318cbbf6f025e2c9d05f29fd55f153121ca05d7ddf8c2f97e849abc0dba88` |
| finalizer/compact_inputs.py | `6dc4fe9b95cb0452c801a9be3f0a2102de9bc6ea76e777725b823501e82dda46` |
| finalizer/launch_once.py | `5ae9ba4c4a9a69d49b7b3772e5160fae3e45857717b56f1758a2daedd97de6dc` |
| finalizer/monitor_once.py | `8acbf213cba75c22c5ea83fdb997c198f80feaf071b6aff91b0eff6646e681c1` |
| finalizer/pull_terminal.py | `c699a98dbb566b331a3d2d072e1f0ac8f9b57a9ef2b8c5f68c38f27a533070bc` |
| gpu/build_packet.py | `4cc21179b634a4d84cdfb54834825cb096671dea7c1aa0299f6b2b3dd915215a` |
| gpu/compact_inputs.py | `6dc4fe9b95cb0452c801a9be3f0a2102de9bc6ea76e777725b823501e82dda46` |
| gpu/launch_once.py | `a773f1b660b0b6ae7a9731b70ea202c62e1259713783d297f360d3dec6574974` |
| gpu/monitor_once.py | `220ba06cf4d3bb2ef40d09e7d41b59300a3ba93163d5ab73bdbf45f2adc24b32` |
| gpu/pull_terminal.py | `4e392c0e40d71ebfe269eec6049cf8eb8be5cd500f677337ef0a06996ef8ca92` |

Shared adapter copies are identical. The new test source is 973f6249f97830f16b86023a12c726ea76588f124b9e950d24895d6893661bd8. Independent AST comparison finds only the intended builder registration/configuration/packet-evidence/main changes, launcher main changes, and collector main/source-pin changes. Other definitions/constants—including frozen packet construction, authentication, lock implementation, resource/accounting/intent observation functions and remote protocols—are unchanged. Both monitors are byte-identical. The protected scientific source/resources remain those already reviewed; no component review was repeated.

The lightweight actual-authority join remains closed: storage-authority.json 02a7a56bcd563433b356cef708d88841f5af30b4f91ddae4d3820394f9b7bbe1 matches the consumed fields of adopted evidence_storage_only authority, freeze156026/sourcea934, backend527d225b and independent backend reviewd59ffe. Those gates are unchanged in this delta. Actual backup or stage success is not inferred.

## Retained evidence

Attempt01 receipt6535dbd97c0bd0f7e842ba7a462384f291c08aa303e11131e3a6596feec05a0e records four passed methods and one failed fixture (0.510513 childCPU / 0.537534250 elapsed / 31,866,880 RSS bytes). The fixture expected intent.json one parent too high. Attempt02 receiptc4625c602f2ce9bdf2cc865f4f28ed44c72bd6d1f41499c928aef6c8e4bc0744 records only that corrected method passing (0.062938 childCPU / 0.066543709 elapsed / 22,806,528 RSS bytes). Its diff changes only the fake backend's intent-path assertions. All retained source bytes match each receipt's source_pins. The reviewer read the new tests, transcripts and receipts; no test was rerun. These five methods cover collector lock/destination ordering, initial/later transport failures, partial bytes and retry rejection, publication failure, and protected AST equality. They do not cover the newly introduced absent stage parent.

focused-static-checks.json retains independent changed-definition/constant lists, source hashes, receipt joins, fixture diff, and the observed absent parents. Original preparations, failures and prior review remain untouched. Review overhead is unmetered. No optional changes are required by this focused pass. A narrow D3 correction and pin propagation recheck is needed before these downstream builders are ready.

Predecessor review: /var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T/ace-downstream-compact-independent-review-xfrmbshh/review.md; SHA256 `bfa876aa740c0e10411bbd89e85bae7d1172767c697fdb24cbf921e5ab55f886`.
