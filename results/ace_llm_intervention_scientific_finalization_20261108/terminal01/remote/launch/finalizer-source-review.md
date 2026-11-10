# Finalizer clock03: narrow numeric-profile review

**0 new required source findings. Prospective only; launch readiness remains open.** This review covers only clock03's numeric profile, dependent entry pin, scheduler/accounting propagation and its one new saved test method. It reuses clock02 review `887050a312db58c3964874233a8f0b854dc36b0a8188bc4151647df7ea7297f1`; the closed phase, lifecycle, marker, scientific and custody logic was not reviewed again or executed.

The six immediate predecessor files exactly match the reviewed clock02 sources. Applying only the declared replacements reproduces every clock03 implementation byte. Monitor and compact adapter are unchanged. All 43 preparation files / 1,057,187 bytes match the independent external mirror, including immediate predecessors and tested copies. Exact diffs and snapshot are preserved here; 75 new byte/metadata/profile checks passed without invoking project functions or tests.

The entry, builder and collector consistently declare **2 CPU, 7680 MiB, 4200 seconds / 70 minutes, zero GPU**, acpu/cpu-normal/ucb736_asc2, no requeue. Builder scheduler text and host allocation authentication use `01:10:00`; launcher live partition/QoS feasibility requires 4200 seconds; collector request authentication uses 70 minutes and observed elapsed<=4200. Actual accounting still charges actual elapsed*2 rather than the reservation.

The unchanged single-origin logic now receives preparation1500, reporting-startup120 capped at absolute1620, work3320, cleanup3350 and hard4190. These are ordered within the 4200-second scheduler cap. Reporting startup remains relative to authenticated worker completion, with no new origin. The latest-start work span is1700 seconds, cleanup allowance30 seconds, post-cleanup hard-stop span840 seconds and scheduler tail10 seconds. These are nominal phase allocations, not measured usable margins.

Candidate reservation is exactly **8400 CPU seconds**. The separate existing preparation allowance remains3600; local budget and both remote accounting expressions consistently use **closedCPU+12000<=57600**, with GPU<=7200 unchanged. The corresponding closed-CPU ceiling is45600. The supplied provisional example37073+12000=49073 leaves8527, but this review does not adopt37073 as an actual accepted baseline. No additional job or history row is invented: the required actual48/98 history still extends to49/100 after finalization.

Only the entry digest changes in source binding. The scientific frozen entry/profile1200 and its original offsets, upstream GPU limits, metadata observation1200/transport1230 bounds, source closure, acceptance contracts and full historical/post-custody obligations remain untouched. The existing clock-adoption schema and independent timing-evidence/review gate are unchanged. Its `predecessor_entry_sha256=b4d18f4b…` continues to name the original pre-clock recovery entry; the immediate clock02 predecessor `ec812c…` is separately preserved in this preparation. This distinction is explicit in the handoff.

The one retained new fabricated method passed once: returncode0, 0.105100 child CPU seconds, 0.150061375 elapsed seconds and26,230,784-byte peak RSS. It covers the changed preparation boundary, 70-minute scheduler profile, exact budget edge, source-pin/profile propagation and allowed delta. Run SHA is `8e475ec060c62efefd04442ef0ed8fcd0238b5dd73e16dda77cc010394912d8a`. No old method or new method was rerun by this reviewer. There was no source import, runtime/model/scientific execution, private-result read, remote action, lock, probe, allocation, packet build or commit.

**The original R1 full-path timing obligation remains open.** This review neither evaluates the current BEFORE/AFTER worksheet nor treats unassigned worksheet quantities as established safety margins. Complete custody/AFTER evidence, actual independent GPU acceptance, final bounded full-path sizing and its independent review, explicit operational adoption, committed source and concrete packet review remain prerequisites. The additional600 seconds is an implemented proposal, not an adopted resource change or a guarantee that the finalizer will fit. Optional suggestions: none.

Exact changed source pins:

| Source | SHA256 |
|---|---|
| entry | `8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f` |
| builder | `5d41e6b1c7b883eab73e6faea1e595ba3e66d75b595d556bdeb47f4642b0232f` |
| launcher | `11fae52bfcd13171f33d566202339e309fe0c2a9d2955ce326e3a4e601a2909b` |
| collector | `c58738e9fc85ff640b96eb6686f92893ab360056013dedcd4c1317baf0462112` |
