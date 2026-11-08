# Remaining terminal custody storage projection

Additional capacity is required for the unchanged reviewed custody design. Current available space is **23.89 GiB**; the already-known-byte lower bound is **32.93 GiB additional**, a deficit of **9.04 GiB** before any new training/campaign outputs, new seals, acceptance archives, Git growth or filesystem overhead. Even omitting the existing training packet gives **28.08 GiB**. Repo, BASE (named external), and /tmp are on the same device; two destinations do not mean two disks.

The four terminal01 destinations were absent at observation. Existing TOKEN capsules are already on disk and are not recharged as originals; the calculation charges only their future copies. No deduplication, compression, reflink, hardlink or purge credit is assumed. Independent copy_file uses exclusive byte writes, and stream USTAR stores each regular name uncompressed.

## Exact known inputs and recurrence

TOKEN01 one complete capsule including seal = 637,169,940 bytes; TOKEN02 = 1,660,022,843. Both match pinned immutable custody seals, including both saved copy maps. S=2,297,192,783. TOKEN02 already nests failed TOKEN01; training deliberately adds a separate TOKEN01 capsule too. This is counted exactly as implemented, not flattened.

Each later raw capture repeats at least TOKEN02 raw R=336,832,982 bytes (31,608 files/477 directory headers). Uncompressed tar floor A=359,618,560 bytes includes512-byte headers/padding and10,240-byte final records. Known immutable training launch members K=104,268,992 bytes across86 names also recur in each raw capture; training embeds both coordinator copies, adding2K to each training capsule. Only named packet members and original intent/result were statted; no payload scan.

With one-copy capsule sizes T,C,G,F and zero unknown outputs/metadata:

T = S + 2K + (R+K) + (A+K)
C = T + (R+K)
G = C + (R+K) + (A+K)
F = G + (R+K) + (A+K)
Additional retained bytes = 2(T+C+G+F).

CPU collector is currently SCP: it creates no new tar. Its inherited training archives remain. Training/GPU/finalizer retain a tar and raw materialization per copy; each successor copies its complete predecessor capsule once into each of two new capsules. The nested size grows additively per generation, not by doubling the whole predecessor pair.

| New terminal stage | Both copies, known-byte floor GiB |
|---|---:|
| training | 6.35 |
| CPU | 7.17 |
| GPU | 8.86 |
| finalizer | 10.55 |
| Total additional retained | **32.93** |

## Estimates, explicitly not guarantees

A modest-growth scenario yields **47.0 GiB** additional; a larger-output scenario yields **76.4 GiB**. They add raw data increments training/CPU/GPU/finalizer of0.10/0.25/0.25/0.05GiB versus0.25/1.00/0.75/0.25GiB, propagated through every successor raw/archive/capsule, plus0.15 versus0.35GiB new metadata/coordinator/seal overhead per one-copy stage. These are capacity allowances, not measured future outputs. Existing training input footprint is about0.10GiB; existing TOKEN02 seal alone is32.29MiB, and future inventories repeat growing nested rosters. Exact future output/file counts are unknown.

For planning, provision roughly **81 GiB free before the remaining sequence** (larger scenario plus5GiB margin), and re-evaluate from actual terminal inventories. At current free space this means about **58 GiB more available capacity**; this is not a hard upper bound. New packet copies, audit capsules, Git/lossless archives, concurrent Mac writes and failures can add more. Collector stream preflight also requires2GiB slack and uses total raw/archive bounds after upstream copying; retained-byte floor alone is not sufficient operational headroom. No deletions/moves or path changes are authorized by this calculation.

## Source evidence (existing reviews reused)

- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_preparation_20261108/collector-seal64/authoring/pull_terminal.py` SHA `41603ce1bc2f64db019c7ff66bafdc3df3feeee994bf78445bda1b8f91bb1c34`: 800–803 copy both TOKEN capsules;808 both coordinator histories;825–840 stream and second complete copy;456–459 capacity guard.
- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_preparation_20261108/token02-history01/authoring/pull_terminal.py` SHA `8904b03d694f10b587103cbd0d7ed463551bb90c428140da0b96da69b87c093d`: 615–621 prior training capsule/coordinators;638–648 SCP full roots;663–669 second copy.
- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_preparation_20261108/token02-history01/authoring/pull_terminal.py` SHA `70ae08119d99dae473d3f03e63387aa783dca3a89f289a7c66d3ab8164b73a2d`: 46–47 all upstream roots;688 predecessor CPU capsule;712–722 stream and second capsule copy.
- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalizer_preparation_20261108/token02-history01/authoring/collector/pull_terminal.py` SHA `a59e8851ea56165b432d001f24da02165233c4da89d019c9889c447b8d817707`: 381 complete root chain;411–425 GPU capsule, stream and independent mirror.
- `/private/tmp/ace-stream-custody-r1-652c4msw/stream_custody.py` SHA `fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a`: 202–229 USTAR regular-file emission;241–246 exact archive bound.
- terminal01 seal `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/terminal01/custody-inventory.json` SHA `b02ecbbd83a3d9f48a3e7248fbb2512be66ffadac723a05bb2fb038465035cc9`.
- terminal02 seal `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/terminal02/custody-inventory.json` SHA `f0dbfeb471ec2afc4132596519376ba4abade6b9d17772922f7a315720019d88`.

Only reviewed source text, two pinned seals, named packet-file sizes and df/statvfs were read. No extensive scan, remote access, deletion, tests, private outcome decoding, model work or collector execution. Arithmetic/details and exact source hashes are in projection.json.
