**Prospective reporting launch01 packet consistency review — 2026-10-24**

Required findings remaining: 0

The reviewed local packet is consistent with the independently supplied freeze hash, the previously reviewed runner, source commit, accepted replay12 metadata, historical input lineage, output claim, and stated resource reservation. This review covers packet consistency and the two new submission wrappers only. It does not repeat the implementation review or establish successful submission, a live allocation, reporting completion, or publication acceptance.

The packet directory is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-reporting-launch-20261024-01`. Only this review file was written. No frozen packet file was edited; no launch script, submission wrapper, reporter, implementation test, model, or remote command was executed.

**Reviewed identities**

- Source commit: `c59e322725f88972592ce44b98b5a0467336acb4`.
- `report_freeze.json`: `256887bcbcbd989e0bab93655b68f371c0615d648a83d816ffeddac1f0a65686`.
- `run_delivery_prospective_reporting.py`: `00b4c8c05a6b146b2213efd4b5f098163fa9620a8f30fb9b8ebdb94e4526f2f4`, identical to the runner cleared in the prior implementation review.
- `run.sbatch`: `d2374eca226cf7bd7ad452af06f87d9287d4a7e58d1a7628fd4c92007eecf3c7`.
- `input_pins.json`: `e67b94fe73661e8d29b14f18bcea2e9960e14fdd4fd53cb1b3d6ada518935a74`.
- `remote_input_pins.json`: `eb3582ffc29c383d128818af44a71eb767870f8af9ead05744d13e89b1834986`.
- `perform_once_locally.py`: `f7a1d8470ead6c53fb1cc9c16073bf3df29113a243bf495f1a85dcd235ff3123`.
- `submit_remote_template.py`: `53080b514e3942a23e8f948a486d99cebb2eb8c8550db50ef6dc22eef5d50f1e`.

All seventeen files enumerated by `input_pins.json` were independently hashed locally and matched: ten source modules, three metadata files, the freeze, authorization, output claim, and Slurm script. The source and metadata subsets exactly match the freeze. Every source module also matches its `scripts/research/` Git blob at the full source commit above. The committed implementation review, current review copy, authorization review pin, and packet receipt review pin agree at `7da9879166e068929f60af1bd306cbeb9965a4c25c367e16f35a0807224412b5`; that review contains the explicit zero-required-findings disposition. The packet receipt's input-manifest, readiness, and reconciliation hashes also match their local evidence files.

**Metadata and original-input lineage**

The packet's replay acceptance, replay receipt, and runtime inventory are byte-identical to the separately retained accepted replay12 custody copies under `results/delivery_diagnostic_replay_20261024/terminal12`. Acceptance is for job `34863006`; its hash is `57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232`. The authorization and packet receipt bind that same acceptance. Its inventory and replay-receipt hashes agree with the report freeze and actual packet bytes. Its record explicitly leaves full reporting/publication acceptance false.

The package manifest hash is `b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9`; the accepted replay-contract hash is `7375f7fe0837197eee8c43decb1c73ffcb74bd6c956367c10323558ffd98c35d`. Both match the retained local package metadata in `delivery-anonymous-B-candidate-20261008-03`, including the manifest's contract entry and 3893 unique file entries. The freeze's exact six historical input pins match the contract/upstream fields for registration, acceptance, scores, complete, fit seal, and audit execution. Historical acceptance also matches the independently accepted replay's original-acceptance pin. These are hash/metadata comparisons; no scientific result calculation or model replay occurred.

The interpreter hash and environment/package selections agree with the accepted replay freeze and runtime qualification. The inventory has 28577 objects. Its `/gpfs/alpine1/scratch/...` prefix is the canonical spelling recorded in accepted dependency origins, while the report and accepted replay freezes use `/scratch/alpine/...`. A first literal string-equality assertion in the local review probe encountered that existing alias difference; comparison against the accepted qualification and unchanged replay environment resolved it. No packet correction was needed. This reviewer did not resolve or inspect remote paths live.

**Launch, output, and resources**

The Slurm script embeds the exact freeze and runner hashes above and launches the frozen environment's Python with `-I -B`. The bootstrap hashes the freeze before decoding it, checks output inode/ownership/private permissions and the output-claim hash, then authenticates and executes the captured runner bytes. The already reviewed runner authenticates the remaining frozen source and metadata bytes before their use. Its existing actual-allocation check and owned-process supervisor remain the runtime gates; this review did not rerun them.

The output path is consistently `/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_01`. The freeze, claim, transfer readback, and remote submission template agree on inode `685172495`. The claim records UID `348288` and mode `0o700`, and matches the recorded claim stdout. Both Slurm stdout and stderr target this output directory. The locally retained transfer stdout agrees with `remote_input_pins.json`, which reports all seventeen expected pins, inode `685172495`, and seventeen files; the transport receipt records return code zero. These are inspected recorded observations, not a new remote verification by this reviewer.

The exact Slurm directives request account `ucb736_asc1`, partition `acpu`, QoS `cpu-normal`, one node, one task, one CPU per task, `3G` memory, `00:30:00`, and `--no-requeue`. There are no GPU/GRES request directives; CUDA visibility is empty. No submission-wrapper argument overrides those resource directives. The unchanged reviewed runner supplies the 1750-second child deadline and rejects an actual accelerator allocation. The freeze and authorization agree on one CPU, 3221225472 bytes, 1800 seconds, and zero new fits/responses; the freeze also records zero optimizer updates.

The declared reservation arithmetic matches authorization and reconciliation exactly: `92.868333 + 0.5 = 93.368333 < 150` CPU core-hours. This is a reservation check, not a new measurement of actual resource consumption.

**Once-only submission wrappers**

The local wrapper checks this review's zero-required-findings disposition, all local input hashes, and equality with the recorded remote pin map before exclusively creating `submission_started.json`. It then exclusively creates the derived remote submission script, invokes the transport once, and retains transport/results or an exception record with retries disabled. No automatic resubmission path is present.

The remote template's literal output path, inode, and complete seventeen-file pin dictionary match the frozen packet. It opens the claimed directory without following its final component, checks inode/ownership/private permissions, requires exact pre-submission directory membership, and rehashes regular input files through the directory descriptor. It then checks the ACE queue is empty, available scratch space, and exclusive write/read/cleanup readiness. It exclusively writes and syncs the remote submission journal before its single `sbatch --parsable --chdir <output> <output>/run.sbatch` call. Existing journals or unexpected directory members prevent entry to that call. Timeout or ambiguous outcomes are retained with retry disabled for reconciliation.

Shell syntax validation (`bash -n`) and Python AST parsing of the bootstrap and both wrappers passed. The template's one intended intent substitution was also parsed without execution. Static inspection confirms the sole submission call targets the already hashed Slurm script and does not add resource overrides. No implementation tests were repeated. The wrapper hashes recorded here bind this review to their inspected versions; these local submission helpers are separate from the seventeen transferred job-input files.

The parent's actual once-only submission, scheduler-identity reconciliation, and later independent report-output acceptance remain separate operations. The existing same-UID/root exclusion from the stated custody boundary is unchanged.
