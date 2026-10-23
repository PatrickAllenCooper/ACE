# One corrected saved-artifact replay — October 23, 2026

## Decision and scope

The proposed next step is **one additional1CPU/3GiB/15minute replay** on existing CURC CPU resources, account`ucb736_asc1`, partition`acpu`, QoS`cpu-normal`, no requeue and zero accelerators. Runtime qualification and the complete numerical replay share the850second child deadline. No environment installation, model fitting, optimizer update or new environment response is included. An ambiguous submission or failed attempt must be preserved and reconciled; it does not authorize another attempt.

The earlier explicit approval permitted one attempt, job33640933. It failed before checkpoint inference at the fileless`torch._VF` provenance check. That approval is consumed. The October23 publication goal authorizes preparation of this concrete correction; the prior single-attempt restriction still requires a new decision before submission. The proposal is not an executable freeze: output07 is unclaimed, and final authorization/output inode/claim/freeze/launcher authentication remain mandatory before a single submission.

## Concrete corrected inputs

The new qualifier and first-checkpoint guard implement the separately reviewed VF handling from commit`76783c24d9d02151955b27c2fbe4a57d5d42b9b6`. They check exact Python creator code/effective bindings and instance metadata, fresh-import reference continuity and independently pinned backing files. Unknown fileless modules remain rejected. This is not native execution attestation or arbitrary same-process tamper protection. Supervisor, sealed-descriptor transport, bootstrap, scientific package, original workers and inferential rules remain unchanged.

The captured source/configuration proposal is `results/delivery_vf_replay_proposal_20261023/proposal.json`, SHA`78a59c074ceb374b90a1829d62335da15e4919ef2b6b5bb8a34a070b8cc3cf2f`. Five captured source files accompany it. Intended output is `/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_07`. The input remains the preserved3893file candidate at `delivery_qualified_replay_20261008_05/package`, manifest`b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9`.

## Fresh remote evidence

Patrick restored the shared connection. The remote channel reached`login-ci3`. Installed`torch/_VF.py` hashes to`5c87e1f29223cef1b4c9259c392733e04d3aec31684fef1391c9801b9a189fde`; both native backing hashes match their independently preserved pins. This is an actual installed-file observation, distinct from the prior upstream reference. No dependency/model import was performed on the login node.

At16:23:57UTC, a bounded byte/metadata check authenticated all3893package files/1042525718bytes and exact3895file membership including manifest and sidecar. The original sampler hash matches. Scratch stat/df and dedicated write/read/cleanup pass; proposed output07 is absent. Both shared local controller locks were held and released. No ACE queue entry was present; remote submission/terminal journals identify the preserved failed03/05/06attempts. Fresh scheduler accounting confirms33640933FAILED1:0 with1CPU/3GiB and69allocatedCPU seconds. These observations must be refreshed as appropriate immediately before submission.

The first preflight stopped before package authentication or scratch writes because an overly broad text filter matched`interface` in an unrelated job name. Its failure is preserved in`remote01`. The corrected`remote02` matches ACE job-name prefixes or ACE working directories. No unrelated job was changed, canceled or duplicated. Readiness/source-file evidence is not successful target runtime or numerical qualification.

## Resource justification and required acceptance

The accepted original audit measured20.427793elapsed seconds and766464000peak process-tree RSS bytes. The previous attempt used69allocatedCPU seconds before the provenance rejection. These support the same small inference allocation; full runtime-inventory duration is still unmeasured and bounded by the shared deadline. No completion ETA is asserted. Existing reservations88.618333core-hours would become88.868333within the150ceiling if approved. Known historical allocatedCPU119413seconds and reservations/processCPU/elapsed scopes are separate; whole-sprintCPU remains unknown.

After a new approval: hold the controller lock; refresh queue/journal/readiness and source/input checks; claim exclusive output07; freeze exact authorization/inode/claim/source/runtime/resource lineage; derive the launcher from preserved06 with new independent pins; authenticate the transferred packet and check it independently; write exclusive local and remote submission journals; submit once. Preserve original06 and all predecessors. Success requires exact runtime, first-checkpoint inventory,640checkpoint/240primary-cell/48000cached-response replay, zero new updates/responses, successful descendant cleanup, scheduler resource/terminal reconciliation and complete exclusive local custody. Only then run both registered and descriptive reporters and integrate the same manuscript.

No allocation, actual new freeze, source transfer, replay, B reporting or public submission has occurred in this proposal work. Source/configuration review and its disposition are retained separately. A later larger runtime/resource requirement would need a new concrete decision.
