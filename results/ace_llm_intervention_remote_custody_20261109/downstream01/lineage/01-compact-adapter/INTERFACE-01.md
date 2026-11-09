# Downstream compact custody interface — proposal01

Source-only proposal. No remote operation, packet construction, scientific payload decode, runtime/model execution or new qualification. Accepted science/source/children, resources, namespace and all historical failures remain unchanged. Initial source snapshots retained separately at /var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T/ace-downstream-compact-custody-w48246u8. This directory is the exclusive downstream workspace.

## Exact unchanged payload requirements

GPU builder140374c5 `admit_cpu_bytes(ctx,raw_root)` (lines417–459) needs only every name in accepted cpu-packet.files, at `remote/launch/inputs/<name>`; each expected hash is independently in the accepted CPU inner packet and full original inventory. It still compares the *full* CPU output/campaign inventory,120CPUcells/192declaredcells, five attempts, 160+D numerical updates and all nested history. No campaign outcome is decoded in the builder. Its final main recheck line592 currently compares a full local recursive inventory.

Finalizer builder63ec2ae6 `admit_gpu_bytes(ctx,capsule,mirror)` (lines682–725) needs only every name in accepted gpu-packet.files, at `remote/launch/inputs/<name>`. It still runs `accepted_rosters` (lines642–665): all seven logical raw roots, both claims, exact CPU/GPU/campaign inventories, eight policy directories, no prior private finalization. Data reads have existing256MiB/member bounds and final child768MiB aggregate/256file limits. No remote omission changes those limits. The final main recheck lines834–836 currently recursively hashes both full capsules.

Launchers call the same builder admission functions at line50; they must bind the new manifest/helper/amendment in their pinned outer members and intent, then perform the same live scratch ancestor/root checks. Preserve both-lock reconciliation, independent job/source/acceptance pins and one-sbatch/no-blind-retry. These are operational coordinator edits only; child source/data semantics and scheduler commands remain frozen.

## Explicit custody authority inputs

Add independent CLI path/hash pairs for the predecessor remote-custody.json and storage operational amendment/adoption review (or the main agent's equivalent explicitly reviewed receipt), plus exact captured helper source/review pins. Capture them as *outer* packet members, never child inputs. Bind predecessor actual job, full outer packet, original before/after inventory transport byte pins, disposition, scientific freeze, namespace and actual accounting baseline. CPU/GPU stage acceptance schemas consumed by frozen children remain unchanged; no storage fields are silently injected into exact-key child acceptance schemas.

A compact receipt must distinctly identify:

- authoritative complete logical raw inventory `files[path]={sha256,bytes}` and complete directories with the existing raw labels;
- scratch original roots and independent terminal before/after observations;
- exclusive `/projects/paco0228/ACE_evidence/...` full raw backup root, complete independent readback and matching inventory hashes; no actual backup is claimed by this proposal;
- selected local file/directory roster and exact complement of omitted paths; local bytes are explicitly partial;
- independent two-local-copy selected capsule seals if required for review metadata, clearly labeled selected, never complete raw mirrors;
- nonrecursive predecessor capsule references, including original failure archive/source/coordinator/partial-attempt evidence absent from raw roots. A reference binds original full capsule seal/inventory, retention location and verification evidence. It must not erase the original failure or merely assert raw-root survival;
- a distinct actual custody verification/acceptance record. Exit0, manifest existence or path on /projects alone does not establish backup integrity or institutional backup behavior.

`disposition.remote_files` and original inventory `files/directories` continue to describe *all* remote bytes. New explicit storage-mode evidence must prevent interpreting that map as local materialization. Preserve actual scheduler/accounting and failure semantics; the collector never issues scientific acceptance.

## Proposed helper behavior; names await core backend handoff

1. Open/authenticate the independently pinned remote-custody receipt and its exact full before/after/project readback evidence. Require expected complete files/directories, roots/job/packet/authority pins; verify full project contents using the backend's bounded read-only observation interface.
2. Verify the exact selected local inventory is the receipt's subset and every selected hash/size matches the authoritative full map. Reject omitted/extra selected files, unsafe paths, symlinks, changed bytes, ambiguous root labels and receipts detached from the predecessor acceptance.
3. Return opaque bytes for an explicitly demanded relative path, independent expected SHA and unchanged byte bound. If absent locally, fetch only that member into an exclusive new staging directory, verify exact bytes/hash and preserve transport argv/partial stdout/stderr/errors. No automatic overwrite, broad recursion, truncation, deserialization or silent retry. Fetched cache must remain outside the sealed selected tree.
4. Recheck source receipt pins, selected local bytes and remote project/original inventory after capture and before publication. Return the same full logical inventory to unchanged history/roster code. Never fabricate files in the local selected view.
5. Expose an explicit ancestor-reference check replacing only finalizer nested_failure_capsule's local recursive location requirement. It must verify original complete failure capsule custody via authenticated reference, not waive it.

GPU builder can preserve `admit_cpu_bytes(ctx,raw_root)` signature by storing explicit verified custody context in ctx; finalizer can preserve `admit_gpu_bytes(ctx,capsule,mirror)` while switching only the full-vs-selected custody branch. The backend must not be imported from mutable ambient source; capture and authenticate helper bytes first. New meaningful tests will use a fabricated backend and tiny opaque bytes only.

## Collector changes required (separate from core implementation)

GPU collector70ae081: full local CPU capsule and hardcoded `upstream-training-custody/upstream-token01-custody` checks at593–602, recursive upstream copy688–689 and716–720, whole raw stream and full-copy seal843–844 need an explicit compact variant. Keep seven-root full remote observations, independent before/after, failure-preservation and47/96accounting, actual resource/utilization evidence and original claims.

Finalizer collectora59e885: builder admission359 and427 must use the same explicit context; recursive upstream-gpu-custody copying411–425 becomes a pinned reference; complete eight-root remote verification and48/98accounting remain. Raw transfer only changes storage placement/selected transfer. Full local-copy seals must not be relabeled without a new schema.

Core backup/transport/helper belongs to main; this sidecar owns GPU/finalizer adapter integration after the backend API is fixed. No parallel duplicate core or broad coding before that coordination point.
