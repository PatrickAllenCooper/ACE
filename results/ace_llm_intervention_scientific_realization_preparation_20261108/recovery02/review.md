# Scientific realization02 recovery packet review — 2026-11-08

**Required findings: 0.** Independent review of the recovery02 builder/launcher/monitor/collector delta and exact built launch02 packet. No optional suggestions. This is prospective packet/coordinator review, not actual realization acceptance. The new packet is unsubmitted and not yet committed at this check; the launcher's existing committed-packet/review and fresh live gates remain required.

## Exact packet and authority

Repository: `results/ace_llm_intervention_scientific_realization_20261108/launch02`. External mirror: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-realization-20261108-02`.

| Object | SHA256 |
|---|---|
| Outer `packet.json` | `ae077ef888671701eaa29415248dc7442656eebe2e5693df14ea6a22d661d8d7` |
| Inner `inputs/packet.json` | `887d6d879a279b1d6291bad3b30425e3a0eae135f2279e1a0794b219769e929a` |
| `inputs/admission.json` | `5c4cb07b82c6558904fd4a40b093eba0a76ea6c72ed22e80d8082a8fd42aea09` |
| `job.sbatch` | `323ce63b62eddd3810abd9d3dd01ed4baa63b55c9d8c35c3ef2d886e562f10d9` |
| Adopted successor freeze | `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d` |
| Recovery authority | `ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50` |
| Adoption | `ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07` |
| Namespace descriptor | `5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36` |
| Freeze review | `088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416` |
| Accepted accounting | `63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727` |

Both mirrors contain exactly 45 independent regular files, 2,285,564 bytes, with identical hashes/directories, no symlinks or hardlink aliases, and no extra files beyond the outer's 44 pinned members plus itself. All inner pins match: 22 captured source files plus the unchanged proposal, with inner packet and admission separately pinned. Each captured source matches its Git blob at `a9346022f145cc29439325b6f291597b2e6ec037` and the adopted realize-stage map. The original probe remains pinned at its CPU03 artifact origin. The shell-parsed bootstrap argv exactly matches the captured diagnostic BOOTSTRAP constant, entry hash, inner packet/admission hashes, source commit and ROOT/realization destination. No bootstrap or source was executed for this check.

The actual recovery authority has exactly 18 keys and matches the authored interface: failed job35402725; independently accepted no-draw failure; predecessor/successor freezes; source commit; preserved original claim; accepted accounting; fixed original/new roots and new local/remote registry paths; one submission only; no automatic retry or replacement worlds. Adoption retains its six-key form and independently pinned review. The namespace descriptor adds explicit recovery linkage while preserving original 32/24 IDs and proposal bindings. The builder does not invent eligibility, adoption or authority.

The successor preserves the original science/design and stage resources. Exactly four infrastructure source hashes change within the unchanged 67-file production roster; the realize packet captures only its 22-source closure. Immediate resources are 2CPU,7680MiB,300s,zeroGPU,ucb736_asc2/acpu/cpu-normal,one node/task,no requeue; offsets120/260/280/290. The failed attempt's actual8CPU seconds remain charged: closed11048CPU/1601GPU,39jobs/80journals. With3600 preparation and600CPU reservation, admission is15248CPU/1601GPU. Whole-cycle projection is55568CPU/7181GPU; this is a cap projection, not measured completion evidence.

## Reviewed coordinator delta and once-only behavior

Authoring root: `/tmp/ace-scientific-realization02-authoring-pu6puc_p`; all four packet scripts equal the reviewed authoring bytes.

| Script | SHA256 |
|---|---|
| `build_packet.py` | `b67e2ac29bd4d06a540d8358179a2dcf63b12bd297141f990750b88b674371cb` |
| `launch_once.py` | `135b21ed12676a10b8d28ef01c7a8da754af2a0a33a4db15b64e072e25a10543` |
| `monitor_once.py` | `1ac08e52ecbb8dd85c6a28b3ad53fa0f44d952d89f78e6dfac22924a2af8f3c2` |
| `pull_terminal.py` | `deebd55fb60d7d3b426fdd5919c9f7d15484c817cbad7fbf5f38804c8e4d86ee` |

Reviewed the complete recovery02-vs01 diff (`a8020cfb1bc98bb4aded0788d0cd94dfe8ccdade84e469c9040bebb0a4cde370`) and new surrounding control flow. The monitor is byte-identical to its retained timeout-corrected predecessor and obtains recovery02 constants from the colocated builder. Closed worker/component checks were not repeated.

The builder preserves failed01 and captures both original inventory transports, all41 original raw objects/directories, and original claim. Its build acquires both controller locks and performs a bounded **read-only remote registry probe**; it is not metadata-only. Exclusive packet destinations preserve failed build attempts without automatic reuse. The saved probe has returncode0, empty stderr, original claim `58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29`,2058bytes,mutationfalse, at2026-11-08T12:55:45.306386+00:00. Transcript pin: `61fa15caee72f90784e16b6116d58518c8835878744c42a30f4979f3db796786`. This is retained login-side registry evidence, not a fresh compute mount-alias observation.

The launcher holds both existing locks through committed-byte validation, readiness, transfer/readback, new claims and submission. Both readiness passes reconcile exact39 accepted accounting rows and80 journal path/hash pairs, no active ACE jobs, resources/site admission, full failed41-object inventory and original remote claim. First readiness permits only the retained failed01 realization root before creating recovery02. Original local claim is rechecked before the new claim. Neither original registry is rewritten or adopted as a new claim. New registry names are append-only exclusive claims linked to the original; local claim/intent precede remote claim/intent and exactly one `sbatch`. Controller, build, intent/result, root or registry residue prevents reuse; timeout/ambiguity preserves state and never authorizes backoff, retry or replacement worlds.

The collector preserves whole recovery02 ROOT/realization, whole failed01 ROOT, both remote claims, both local claims and both new coordinator histories. It compares full second remote readback, original41-object roster, exact prior39 rows plus current terminal row, and80 prior journals plus current pair (40jobs/82journals expected). Authenticated remote-intent/local-intent semantic joining is retained and includes recovery bindings. Failed terminal states or missing output receipts do not exclude raw custody. Private scientific output JSON remains opaque; acceptance/scientific qualification/retry flags remain false. Actual CPU/GPU costs are added to11048/1601 regardless of success, with unknown/disagreeing evidence explicitly retained rather than treated as acceptance.

## Retained evidence and final submission bindings

The independently pinned failure acceptance (`5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d`) and no-draw audit (`5fb94f9d07da3acd25c979250471037e8981066e0b201716f8c32d0b681fb9ef`) are consumed as prior authority, not rerun. Both supplied inventory transports match their accepted pins. Their complete41-file/1,940,531-byte raw map matches both original local custody copies and the new packet projection. No private output or traceback was replayed to rederive the closed diagnosis.

Author `static01/receipt.json` pin `7945f777f2c64fb17e7c9f12ba33c82dfad0ba2e595ac93d46f4f9701fbdfaa4` records AST/compile-only authoring verification, including embedded scripts, no tests or target execution. `build01/receipt.json` pin `0abd4e58a1e7e66a635854c62ead23eebdb5f25fda252dae06a3b6dc3b571984` records successful parent build: childCPU1.306082s,elapsed3.901767334s,peakchildRSS32260096bytes. Its saved command pins the exact adopted and accepted inputs; stdout reports the reviewed outer/inner pins with namespace_claimed=false/submitted=false. These build costs do not establish worker runtime cost.

Independent metadata audit: `/tmp/ace-scientific-realization02-packet-audit-20261108.json`, SHA256 `e892838349b7a1aa6840cad7afb807b75876f6dc33a70bbc3e63de4060464daa`. It records exact two-mirror inventory, captured-source Git bindings, inner/outer/admission/bootstrap agreement, accepted failure-map agreement and saved registry readback. No new local recovery registry or submission journals existed at this check.

The eventual launch must supply the full packet commit containing these exact packet members and this review, outer pin `ae077ef8…`, and this review's final SHA256. That commit has not yet been supplied or independently checked; the reviewed launcher enforces its exact byte binding before remote mutation. Future intent/result/job/claim hashes are not invented here. The collector requires the eventual job ID plus original outer and submission-result pins. Current source/packet disposition is **0 required findings**, subject to those existing execution gates and independent actual-result acceptance.

Reviewer performed only local source/AST/metadata/hash/Git reads and wrote this review plus the local audit sidecar. No tests, builder/launcher/collector invocation, remote commands, models, scientific draws, source edits, staging or commits were performed by this reviewer.
