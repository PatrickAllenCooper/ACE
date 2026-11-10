# Independent concrete finalizer packet02 delta review

Required findings: **0**. This is a distinct review of the actually published successor packet, following the closed named-outer-reader source review df74fcfa. It is not a relabeling of packet01 review 46987400 and does not authorize uncommitted source or establish execution success.

Repository packet: `results/ace_llm_intervention_scientific_finalization_20261108/launch02`. Independent external packet: `ACE-LLM-intervention-scientific-finalization-20261108-02` under the established BASE.

- New outer SHA256: `dcd34d5c040b7a9bf2769736a8ee5fa103090c4ef1af3f7c663cd961eabbc433`.
- Preserved inner SHA256: `c71bafdf9f5a791c540c6a5a1a12ba0f2b84b9b221c2b018cbdebf9684728571`.
- Original outer SHA256: `1a30c9bdfef0b81f98089c774efa5ff781ddc2b5bc8411c9d08e3b198adb87d3`.

Both new copies have exactly 223 regular, single-link files totaling 782,762,459 bytes. Independent full hashes, stable file reads, exact membership, absence of symlinks and distinct copy identities matched. Only `build_packet.py`, `launch_once.py`, `monitor_once.py` and outer `packet.json` differ from the original authenticated member pins. The other 219 members are unchanged. In the decoded outer object, only the three corresponding `new_input_files` hashes change; every other field is equal. Original outer pins in both packet01 copies remain intact. The prior complete byte/source review was reused without repeating its entire inspection or tests.

The three replacement hashes match the exact independently reviewed final source:

- Builder `e46cd29633c9dcd83db174fb77ff5c7acf751553bfb46963129cdb1129a255cb`.
- Launcher `4508ff44bd2a7c0d8de678bfc18d54b73cf6c2829dfe62835285e1a344e2991c`.
- Monitor `4d22fb95146f6e308fab021f35023a491909d03af49b9b5ea4d4bc5c618cf373`.

The separately retained collector is `b1994191042e910923ffe23e504f2ecd9af6d5242ad6743a5537a2aa4e73e5c3`; it is not a child packet member and requires its own supplied source/review/commit pins when used. The successor routes local/external inputs to packet02 while preserving the unused remote/output root01. Entry 8a26ada2, inner c71bafdf, job bootstrap, adoption 4ad691f0, actual acceptance/recovery/baseline, namespace claims, all scientific inputs, source closure, runtime/storage configuration and resources/deadlines are unchanged. Fresh remote absence remains a launcher gate; this review made no remote observation.

The historical 42,123,690-byte outer failed the old default 32 MiB reader before any remote operation or submission, as retained in failure disposition f4cff65c. My prior review missed that size/reader incompatibility. Its zero-finding conclusion did not establish packet01 launchability. The original review and failure are preserved; source review df74fcfa independently covers the named 128 MiB readers, unchanged ordinary 32 MiB limits, focused new boundary evidence and new local routing. This concrete review binds those corrected sources to the new outer hash. It does not claim the old review covered new source.

Publication evidence is retained in `finalizer-operational-publication01` and its independent mirror. Its exact original/source/failure/review anchors, evidence inventory, successful run and full new packet inventory agree. The parent run records 2.514955334 seconds wall, 2.212053 child CPU seconds and 626,376,704 bytes peak RSS; the inner publication run records 2.440374959 seconds wall and 2.149764 process CPU seconds. The two timing scopes are not summed. This was one local metadata publication, with no scientific builder, runtime/archive verification, remote operation or submission. Original storage-build proofs remain historical records; they do not assert a new storage/build run.

The root transfer is now 738,753,046 bytes across 105 files and retains the 825-second bound. Inputs remain 44,009,413 bytes across 118 files and retain the 162-second bound. Both are below the reviewed 900-second per-group cap. The unchanged scheduler requests 2 CPU, 7680 MiB, zero GPU, 4200 seconds; absolute phase limits and the adopted bounded timing profile are preserved. Budget remains 49,073 CPU/4,458 GPU seconds including the preparation allowance and proposed finalizer reservation; completed history remains 48 jobs/98 journals, with finalizer completion prospectively 49/100. No job or accounting row is invented for either local publication or the prior local reader failure.

The new independent byte/metadata inspection passed 1,526 checks with no failed attempt. No component tests, fixture methods, source execution/imports, private payload decoding, models, storage/archive verification, locks, remote actions, Git operations or launcher were run by this reviewer. Exact review source, results, logs and publication evidence are retained. Remaining gates are parent commit/push and independent revision match, then the corrected launcher's fresh two-lock/readiness/full-history/storage/input/absent-root checks and once-only intent. Actual finalizer execution and later terminal acceptance remain open; execution/scientific qualification for this new finalizer is false.
