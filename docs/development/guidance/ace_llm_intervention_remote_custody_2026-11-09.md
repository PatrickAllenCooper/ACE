# CURC backups and compact local evidence

Patrick authorized this operational change on November 9: keep large artifacts backed up on CURC, retain reports and verification records locally, and continue the frozen 24-system comparison. This supersedes the proposed 85 GiB Mac capacity target. It does not change the scientific design, training, policies, source inputs, endpoints, resources or failure rules.

## Storage contract

The computation continues to read and write its original scratch paths. After each stage is terminal, the collector preserves every original raw file, including failed and partial outputs, in an exclusive archive under `/projects/paco0228/ACE_evidence`. CURC documents `/projects` as regularly backed up. Scratch remains a working copy, with no backup guarantee and a 90-day purge policy. No attempt is made to extend that purge period. The archive is a retention copy, not a location for active model or campaign I/O.

The collector must verify the complete original inventory before copying, the archive's exact members and byte hashes, the copy in `/projects`, and the original inventory again after copying. A file being present in `/projects` or a transfer exiting successfully is insufficient. The record distinguishes our verified copy from CURC's own backup schedule, which we cannot independently attest.

Local capsules contain a declared subset of source, reports, logs and verification records. Their full remote inventory and exact omitted-file list remain local. They must never claim that all raw bytes were downloaded. Required small opaque inputs may be fetched separately with independent hashes. Tensor deserialization and outcome decoding remain subject to the original scientific gates.

Existing complete local capsules and failed transfers are preserved in place. New capsules reference their independently pinned inventories rather than recursively copying them. This removes the repeated growth that motivated the earlier capacity target. The two existing Mac destinations reside on the same filesystem and are not separate storage devices.

Consumers must authenticate the predecessor receipt, verify the remote archive again and verify the exact selected local subset before using it. The original source, input, namespace, submission, accounting and scientific acceptance checks remain required. A storage receipt never grants runtime or scientific acceptance. Interrupted or ambiguous transfers retain their evidence and require reconciliation before a new exclusive attempt.

## Admission and current preparation

The November 9 live check found about 89 GB available in the user's `/projects` quota and 6.3 TB in scratch. A dedicated scratch write/read/cleanup succeeded. The two shared controller locks were acquired for that check. No ACE campaign was queued or running. Unrelated queued jobs were left untouched.

The CPU packet was built once from the accepted scientific training output. Its outer SHA256 is `b98873cae0e04221abcce02856ea7c610319f27f4c07e8275cc9bcfce1c2cd40`, and its inner SHA256 is `ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344`. It contains 152 files totaling 140,205,730 bytes in each existing packet copy. Build overhead was 2.803806 child CPU seconds and 3.166409 elapsed seconds. This is preparation, not an allocation or scientific result.

The requested CPU stage remains two CPUs, 7,680 MiB, zero GPUs and a 6,000-second limit on `ucb736_asc2`, `acpu/cpu-normal`. It covers five policies across 24 systems. The later three LLM policies complete the frozen 192-cell comparison. Closed allocation is 17,805 CPU seconds and 2,505 GPU-slice seconds. Adding the CPU stage's 12,000-second reservation and the existing single 3,600-second preparation allowance gives 33,405 CPU admission seconds within the 57,600-second ceiling. No new reservation has been charged by building the packet.

Submission still requires the reviewed storage implementation, an actual verified backup, the concrete packet review and committed source, followed by fresh lock, journal, queue and input reconciliation. The initial prepared packet has not yet been submitted. Later validated entries in the execution ledger supersede this preparation snapshot.

Source: [CURC filesystem guidance](https://curc.readthedocs.io/en/latest/compute/filesystems.html).

## Source review disposition

Independent final review `d59ffe14881d2921804bdd19ba2dfa17df7899013e7aa4fca6bfcbf28e9d28ef` closes all required findings in helper `527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b`, CPU collector `622ead7fa8e9b659aece8337f437675cd7d666a19af7a6beda3d607e06e3293a` and the two backup invocation wrappers. The source corrections preserve primary failures during cleanup, reject missing or insufficient capacity, authenticate captured remote helper bytes under an isolated interpreter, and leave preexisting output directories untouched on failed admission. Current-launch UTF8 command records are retained locally for the numerical-call audit. All initial failures and focused correction evidence are preserved. These fabricated checks establish implementation behavior, not an actual CURC backup or scientific acceptance.

The adopted storage-only amendment is `results/ace_llm_intervention_remote_custody_20261109/preparation01/storage-authority.json`, SHA256 `02a7a56bcd563433b356cef708d88841f5af30b4f91ddae4d3820394f9b7bbe1`. Its scope is the user-authorized storage change. Scientific runtime, source, inputs, budgets and acceptance rules are unchanged.

## Actual November 9 follow-through

Training and static-input backups both passed complete source, archive and projects readback checks. Their receipts are `1bb3020156bf69d7e9476e7745dfc044c35be7d7560aedb79e78a344059a24e9` and `5ac647f3b20bf1e027f8cda3c89c613e9ef65226ea5a3c701036a9cc3af88c55`. The static archive is9,970,851,840bytes with no static payload downloaded locally. Training archive565,248,000bytes preserves36,623files including failures. Local selected training evidence is136,764,246bytes, with omitted files explicitly listed. These are verified CURC copies, not scientific acceptance.

The original48-second full-input preflight timed out before any submission. A separately reviewed metadata-only coordinator allows1200seconds for verification and preserves the original failed records. It submitted the unchanged CPU packet once as35482135 at17:50:47UTC, using packet commit434b83e7a7b0bbe34ecef1caf939b591aad53494. The first bounded observation is PENDING/Priority. Job resources and all scientific settings are unchanged. Full terminal backup and actual acceptance remain pending.
