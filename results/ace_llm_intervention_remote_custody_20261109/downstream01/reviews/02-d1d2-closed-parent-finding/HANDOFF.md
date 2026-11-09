# D1/D2 corrections — narrow recheck requested

Both required findings are addressed in new, disjoint source. Original source/review/test preparations remain unchanged. `predecessor/` and `independent-review/` capture exact reviewed bytes; initial test failure remains in attempt01.

D1: GPU collector retains its original explicit two-lock try/finally. Finalizer retains its original coordinator_locks scope. All transport-capable configure/admit work follows both locks, the existing terminal-destination guard and exclusive claims. A pre-existing new operation directory also rejects before backend invocation. Source/packet authentication remains local and before transport. No scheduler/resource/history/private-output gate changed.

D2: no random /tmp predecessor-use directory remains. The caller registers `storage-use.intent.json` with an exact intent hash and durable directory identity before the first backend call. Every archive verification and input read/fetch, successful or failed, writes an exclusive numbered inventory and reference. Inventories cover the complete operation directory including partial output bytes, source/argv/transport logs and receipts. References are published to both caller evidence destinations. Binding failures are recorded even when Inputs construction failed before the gate could enter ctx. Secondary evidence/publication errors retain the primary exception plus an explicit local diagnostic; they do not claim complete publication or permit retry.

Durable paths (all under the existing retained BASE or stage repository results; no new CLI arguments):
- Build: external sibling `<external-launch-name>-storage-build-preparation/records`; references in that preparation parent and repository stage `storage-build-preparation01`. Both preparation directories are exclusive. Packet outer files add `storage-preparation-intent.json` and `storage-preparation-reference.json`; build stdout includes the final operation reference after post-write verification. Later operation manifests/references remain in both preparation destinations.
- Launch: external sibling `<external-launch-name>-storage-launch-operations`; references in both original packet roots. Once-only markers include storage-use.intent.json; source/Git/review and existing-attempt checks precede registration. Failures before controller.json still produce launcher failure evidence.
- Collection: external sibling `<external-terminal-name>-storage-collector-operations`; references in both newly claimed terminal capsules and consequently their final/failure inventories. Original current backup-operations remains unchanged and independently retained.

Retain those whole original operation directories. Their pinned references replace copying; they are not disposable caches. Each manifest declares its exact operation status and false acceptance/retry flags. Future audits must follow these references and verify the inventory pins. No predecessor tree recursion/copy, child input, scientific source, parameter, namespace or resource change was introduced. Existing backend527d and authority02a7 schemas are unchanged.

Five new methods passed across retained attempts: collector-prefix fabricated guards (both stages, first/second lock conflicts, either existing capsule, existing operation root and positive control); initial verification failure with pre-registered caller links; later partial fetch failure and retry rejection; primary-error preservation under publication failure; static protected-function/constant equality. The first run passed4 and failed1 because the new fake backend checked the wrong parent for intent.json. The fixture assertion alone was corrected; only that failed method was rerun, successfully. No earlier method was repeated. attempt01:0.510513 childCPU/0.537534250 elapsed/31,866,880 RSS bytes; attempt02:0.062938 childCPU/0.066543709 elapsed/22,806,528 RSS bytes. No real backend/main/model/science/remote action occurred.

All source pins and receipt hashes are in handoff.json. Diffs are alongside each derivative. Both monitor files and GPU cpu_history_contract3893 remain byte-identical. Review only D1/D2 and their source-pin propagation; prior scientific/backend/authority reviews remain closed. No actual new packet, allocation, acceptance or commit is claimed.

## Exact changed-source pins

- `compact_inputs.py`: `6dc4fe9b95cb0452c801a9be3f0a2102de9bc6ea76e777725b823501e82dda46`
- `finalizer/build_packet.py`: `2f6318cbbf6f025e2c9d05f29fd55f153121ca05d7ddf8c2f97e849abc0dba88`
- `finalizer/launch_once.py`: `5ae9ba4c4a9a69d49b7b3772e5160fae3e45857717b56f1758a2daedd97de6dc`
- `finalizer/pull_terminal.py`: `c699a98dbb566b331a3d2d072e1f0ac8f9b57a9ef2b8c5f68c38f27a533070bc`
- `gpu/build_packet.py`: `4cc21179b634a4d84cdfb54834825cb096671dea7c1aa0299f6b2b3dd915215a`
- `gpu/launch_once.py`: `a773f1b660b0b6ae7a9731b70ea202c62e1259713783d297f360d3dec6574974`
- `gpu/pull_terminal.py`: `4e392c0e40d71ebfe269eec6049cf8eb8be5cd500f677337ef0a06996ef8ca92`
- `test_d1d2.py`: `973f6249f97830f16b86023a12c726ea76588f124b9e950d24895d6893661bd8`
