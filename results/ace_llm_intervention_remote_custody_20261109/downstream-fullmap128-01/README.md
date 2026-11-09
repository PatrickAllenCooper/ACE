# Named full-map metadata reader successor — source ready for independent review

Actual pinned observation: terminal02/remote-custody.json is33,829,045 bytes (f0f8a52c7f823b5a07041faf531382afa4d7c0364452413bbb17bd916dfc38e6),274,613 bytes over32MiB. Before inventory is32,644,997 bytes (722782e08d15108bf1b0b9b9b4d7111d3432a19ead6e8cadb2a8f9dc8bff3b46). `actual-size-observation.json` is a byte copy of the parent's published observation; this preparation did not read active capsules, outcomes, or archive. Neither object is treated as accepted execution. No archive access/reverification was performed.

Use `implementation/gpu/`, `implementation/finalizer/`, and `implementation/cli/`. Every predecessor source is preserved under `predecessor/`; diffs and exact before/after pins are in `diffs/` and `source-pins.json`. Backend4161 and final-report-batch02 are unchanged and outside this edit. The prior CLI preparation remains intact.

## Exact reader bounds

`role-bound-map.json` is the complete explicit map. Both builder/launcher/collector supplied-metadata routes and CLI use128MiB only for:

- cpu-inventory, cpu-disposition, cpu-acceptance
- gpu-inventory, gpu-inventory-after, gpu-custody-inventory, gpu-disposition, gpu-acceptance

These roles contain full remote/campaign/custody file maps. Ordinary sources, reviews, readiness, historical collection/TOKEN/training metadata and all unlisted metadata retain32MiB. No content-based or suffix-based automatic expansion exists.

Both compact adapters use128MiB only for storage receipt, seal, after. Seal was already128MiB. Authority, authority_review, backend, backend_review, reference remain32MiB; config capture remains128KiB. The GPU collector's pre-import packet/Git/mirror loop explicitly recognizes those storage filenames, the eight named metadata JSON filenames, and inputs/cpu-acceptance.json plus inputs/gpu-acceptance.json. Unlisted packet members retain its32MiB bound. Later supplied-metadata rereads use the same builder role function.

The finalizer initially used a broad256MiB metadata read; its initial build/launch reads now apply the requested named128MiB/ordinary32MiB policy. Its collector's previously128MiB-only gpu-custody-inventory exception becomes the same explicit role map. Its generic all-input copy loop already uses hashing/readback without a32MiB read cap, so no change was needed there. Existing256MiB exact-byte mirror comparisons and opaque flat input fetches remain unchanged. These comparisons follow initial metadata admission; they do not broaden that admission. Original archive/backend capacities, source/config APIs, stage resources/deadlines, selection rules, file-count/aggregate bounds and attempt gates are untouched.

`reader-callsite-map.json` traces initial ingestion, stored config, launch reauthentication and collector rereads. `definition-delta.json` and `static-scope-proof.json` show changes confined to named reader arguments/helper functions and dependent outer digests. Frozen GPU entry26940935 and finalizer28c5 already read captured inputs with256MiB defaults (lines93/82 respectively); no frozen-child change is indicated by the observed receipt. Their per-packet256KiB and aggregate512MiB gates, where applicable, remain unchanged. A future artifact exceeding any remaining explicit bound still rejects; no unbounded guarantee is made.

## Dependent source pins and CLI

Both builders pin the new compact adapter a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245. GPU collector pins its new builder. The successor CLI's known-input map points to this preparation's GPU sources and exact new builder/launcher/compact hashes; all49 accepted known metadata entries and116 builder flags remain unchanged. The CLI source pins that map, uses matching full-map bounds for known/pending inputs, and grants128MiB to only the three storage roles above.

Future usage of implementation/cli/assemble_gpu_cli.py is unchanged: explicit seven pending metadata path/pin pairs, four custody path/pin pairs, terminal02 remote+mirror roots, interpreter, and new exclusive output directory. It is not run here. Parent must supply accepted actual CPU acceptance/baseline/review and complete successful terminal02 custody plus fresh readiness. A later successful assembly writes literal command.json and eight-member storage-config.json, and still never runs a builder. Use the original CLI invocation-template field names, with this successor utility path; no actual future command was assembled.

## New bounded evidence

Four NEW reader-only methods passed once in1.013433208s wall /1.002760s child CPU /862,879,744 bytes child maxRSS. They use only AST-captured read/limit functions and supplied-metadata comprehensions from the new files, never coordinator main, locks, transport, project/model/scientific functions or actual data. Cases cover observed33,829,045-byte-size artificial records, exact128MiB acceptance,128MiB+1 rejection, ordinary32MiB rejection, byte/hash capture and tampered pins, storage capture+stored config, collector rereads and exact dependent digests. No old test/review was repeated.

`evidence/attempt01` retains executed source snapshots, source pins, original stdout/stderr and run.json. Sparse artificial input bytes are preserved in fixtures.tar.gz after exact full lossless readback (fixture-inventory.json);437,588,353 logical artificial bytes occupy427,527 compressed bytes. Originals of prior sources/tests are untouched; no failed new test attempt occurred. Packaging was1.097799s wall/1.090920s CPU and is separate from test measurements.

No repository source edits, current collector changes, actual capsule reads, private decoding, remote operations, builds, launches, allocations, locks or Git commits. Parent owns independent delta review, integration and all actual gates. This preparation is not a scientific or execution acceptance.
