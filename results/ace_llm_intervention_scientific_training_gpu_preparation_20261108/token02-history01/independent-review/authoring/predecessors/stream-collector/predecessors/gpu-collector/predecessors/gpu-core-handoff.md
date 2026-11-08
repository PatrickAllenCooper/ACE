# Scientific training GPU coordinator authoring handoff

Status: stable for independent authoring review; NOT built, submitted, runtime-qualified, or scientifically accepted. Only this new temp copy changed. Original partial and every original test attempt remain in predecessor-partial; collection02 authoring remains untouched.

## Consumer interface

build_packet.py requires --NAME PATH and --NAME-sha256 SHA for every METADATA name listed below. Also --prepared PATH --prepared-sha256 SHA --plan-sha256 SHA and these three original inventory-relative paths:

- --cpu-packet-relative launch/inputs/packet.json
- --cpu-prepared-relative launch/preparation/session/worker/artifacts/llm-prepared.json
- --cpu-terminal-relative launch/preparation/session/worker/artifacts/terminal.json

These paths match the tokenization recovery draft (88320878518532419ad3450823de72ba67df9935764fd7fd5cf0c2ba9055bb34) at /tmp/ace-scientific-tokenization-recovery-authoring-qa7ojemp/build_packet.py. Reading that local interface sufficed; no dependency on executing its code or modifying its files.

Fixed independently pinned metadata: freeze, freeze-review, adoption, namespace-admission, recovery-authority (original realization authority); operational-amendment fa94858d..., operational-adoption-review db2a672a..., collection-failure-acceptance 00c4557a..., no-response-proof 59c21d63..., collection-failure-baseline eeec362d.... These are separately captured, not additions to or replacements of freeze156026/sourcea934.

Additional explicit pairs: namespace-claim; cpu-packet; cpu-acceptance; cpu-inventory; cpu-disposition; cpu-launch-packet; cpu-result; cpu-terminal; cpu-report; collection-acceptance; collection-packet; collection-result; collection-disposition; collection-inventory; baseline; previous-baseline; site-readiness.

previous-baseline means independently accepted collection02 accounting42jobs/86journals. baseline means independently accepted TOKEN01 accounting43jobs/88journals. collection-failure-baseline is actual closed41/84. No future job ID, elapsed time, CPU/GPU cost or approval defaults exist. All prior rows and journal hashes must survive both transitions; cost deltas use accepted actual CPU allocation times elapsed seconds. The failed collection's252CPU charge remains in actual11386/1601 baseline. Remaining frozen stage reservations must also fit57600CPU/7200GPU including3600prep.

Future collection acceptance uses the existing token-builder interface: status accepted, required_findings0, complete_custody_verified/accounting_complete true, scope policy_train, job_id, collection_packet_sha256, disposition_sha256, and original collection binding. Its root must be collection02/collection/session/worker/task/collection. collection-inventory is the original successful inventory_transport.json (returncode0, stdout JSON files map, captured literal ROOT/JOB). It joins all collection input/packet/journal hashes and complete accepted collection output map, without opening those scientific payloads here. No fabricated remote_files field is added to any original disposition.

The exact frozen CPU acceptance schema remains ace-scientific-training-preparation-acceptance-v1 with the existing required18fields, including job_id, exposure_sha256, numerical_plan_file_sha256 and numerical_plan_sha256. No amendment/inventory/review fields are added. Independent CLI pins bind the inventory and review sidecar. CPU outer packet must carry the explicit amendment, review, failure/proof and collection02 bindings emitted by the referenced token builder. Complete original CPU outer and inner input pins are joined to its transport inventory. These joins precede prepared JSON decoding; only saved CPU-accepted training token IDs are used.

## Packet and launch

Future repository packet: results/ace_llm_intervention_scientific_training_gpu_20261108/launch01.
External packet: /Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-training-gpu-20261108-01.
Remote root: /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01.

Outer coordinator packet is ROOT/packet.json. Supervisor packet is ROOT/inputs/training.json; independent captured conditional admission is ROOT/inputs/admission.json. Output is ROOT/training; worker artifacts ROOT/training/session/worker/artifacts, fitted outputs under artifacts/fit. A future terminal collector must use the outer packet and current submission result/intent pins, then retain the whole root; none is executed or supplied here.

Train remains4requested/max4CPU15360MiB,1a100_3g.20gb,2700sec,ucb736_asc2/aa100/gpu-testing. Exact source55be5ba7 / workerab2c999a, frozen29source roster plus runtime archive helper, complete67-source metadata capture. Deadlines startup240/worker2620/soft2630/cleanup2645/hard2670; progress180/telemetry90. Fixed fresh seed109,160SFT and independently prepared D1..128; singleton scoring and prompts<=4744, no truncation, adapter reuse, tokenizer, downloads or scientific responses. The existing supervisor enforces actual staged runtime/model/source admission, independent GPU work signals and owned cleanup. This authoring check does not requalify those implementations or live resources.

launch_once.py requires --packet-directory --packet-commit --packet-sha256 --packet-review --packet-review-sha256. It reauthenticates metadata, committed exact source/packet copies, both read-only namespace registries and both coordinator locks. Twice reconcile all prior job rows/journal pins, no active ACE jobs, exact original collection/CPU inputs, model/manifest and site limits before one exclusive sbatch. Remote archive bytes are authenticated by the allocated stage before runtime execution. Timeout/ambiguous transports retain partial streams and intents, never retry. The live account/QoS/partition check is future work, not established by this handoff. No scientific freeze or namespace registry is changed.

monitor_once.py requires --packet-directory --packet-sha256 --submission-result-sha256 --evidence-directory. One read-only Slurm and output/telemetry metadata observation, exclusive local evidence, no cancellation or acceptance. Parent owns repeated monitoring and terminal collector. Monitor bytes are unchanged from the preserved partial.

## Verification and limits

Only three NEW metadata tests ran in lineage03: valid failed41→collection42→TOKEN43 chain, amendment/captured-lineage negatives, and lost rows/journals/cost/inventory negatives. Passed3 in0.040sec unittest; wall0.085026375sec; controller CPU0.001479sec. Child CPU was not separately measured; no resource reservation/allocation occurred. Future successful jobs and prepared token IDs are fabricated; no model/scientific data or test suite executed. Prior4methods and prior failed-baseline method were not rerun. static04 records syntax checks of current files and embedded remote source, without executing them.

Historical evidence remains in predecessor-partial/tests01, predecessor-partial/delta02-failed-baseline and predecessor-partial/before-collection-failure-update. Their original receipts remain authoritative for historical costs. Per-file diffs against the preserved partial are alongside current files. New edit_lineage.py is an authoring-only transformation record, not a packet input.

Independent authoring review and later concrete packet review remain necessary, as do actual accepted collection02 and TOKEN01 full custody with independently supplied future accounting. No acceptance or permission is inferred from passing fabricated checks.

## Exact current file pins

- build_packet.py: `f73835f83c2cabc22162a4f8093b84cc02d19d2d0a78962137fdbfdd025e76f1`
- launch_once.py: `6971ac253180c0164591dcf01b6a46381e246bda02a798854168c07a06310153`
- monitor_once.py: `3fc1fee1a4ce36399bcb5eac762e67c5c6de56256347e51331c6f421eb38ef61`
- test_recovery_lineage.py: `742432a2465896527792e529094a5c3cba79ea235311e0e08c282aa711dec6e2`
- test_authoring_metadata.py: `e094013f39bc59b51c8c5e92485e35a840d79ca1b88c061b8946403a83a2dde8`
