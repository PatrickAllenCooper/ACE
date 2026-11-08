# ACE LLM intervention training GPU packet03 review — 2026-11-08

**Required findings: 0. Disposition: accept the concrete GPU03 packet for the bounded fresh-adapter qualification.** This closes the packet-review gate for the exact bytes pinned below. The parent retains responsibility for committing and pushing the packet and review; the launcher's committed-byte and live readiness gates remain applicable. This review establishes no successful GPU execution, reload, training/runtime qualification, or scientific result.

Scope: the local [launch03 packet](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/launch03), its control/path/accounting delta against the closed [packet02 review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_training_gpu_packet02_review_2026-11-08.md), and [validation03](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/validation03/check_saved.py). Verification used local text/JSON reads, byte hashes, static AST/shell argument parsing, and read-only Git object comparisons. No project scripts, saved validators, tests, models, numerical libraries, GPU work, remote commands, commits, or pushes were executed. Only this review was written. Closed worker, source/API, component, and failure-custody reviews were not repeated; their conclusions are incorporated through authenticated evidence pins.

1. **Actual inputs and committed source authenticate.** The inner manifest contains exactly **45 inputs** and **28 declared sources**; the outer manifest has **47 input entries**, giving **48 transfer pins** with `packet.json`. Every actual input hash matches. Outer membership is exactly the inner inputs plus `training.json` and `job.sbatch`. All 28 declared source files, plus the separately listed runtime-archive helper, match both working research files and Git blobs at **`a855243c37330d117fb6c87bc79314c38a7b610e`**, which is the packet's `source_commit` and HEAD at inspection. No research-source changes were pending.

   The worker is the **only changed inner input** against launch02; the other **44 inputs are byte-identical**, with unchanged membership, source roster, and resources. Worker SHA-256 **`faa3111ccfdbfb8004eb539a41dcc57df98183037a5ed25afd294fa2cec4f1bd`** matches the closed [workspace review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_training_gpu_workspace_review_2026-11-08.md), correction receipt, retained final source, and committed source. All **46 correction evidence pins** authenticate; this is evidence binding, not another implementation or test review.

   This incorporates the reviewed `torch._C._cuda_clearCublasWorkspaces` operation at the existing model-disposal boundaries, synchronized before release, exactly one successful call, and strict **before / immediately after workspace clear / after cache release** allocated/reserved measurements. All six counters must be nonnegative integers; missing/noncallable API, API failure, cleanup error, or nonzero final allocated/reserved memory remains rejection. No presumed byte reduction replaces the exact-zero gate. Actual allocator behavior remains a post-run measurement requirement.

2. **CPU acceptance, plan, and fresh-adapter scope remain fixed.** All **38 inherited CPU inputs** match the original CPU packet and GPU03 bytes. CPU preparation, acceptance, and prepared copies match their actual CPU originals and committed blobs, as well as the unchanged supervisor's independent constants. The accepted token plan, training-engine source, backend, singleton final scoring, and paired SFT reference scoring are unchanged. Seed **109**, empty new worker artifacts, initial fresh adapters with zero B, and later reloads only from checkpoints created by this attempt remain required. Neither failed attempt's adapters are inputs or resume targets. The packet retains four SFT and two DPO updates, zero new environment responses, no installs, and false training/runtime/scientific acceptance flags.

3. **Resource and time bounds are unchanged.** `job.sbatch` requests `ucb736_asc2/aa100/gpu-testing`, one node/task, **one `a100_3g.20gb`**, **4 CPUs**, **12 GiB**, **600 seconds**, and `--no-requeue`; the authenticated resource contract permits at most **8 allocated CPUs**. Existing sizing evidence remains the basis: GPU09's six updates in 126 seconds with 4.017/4.844 GB allocated/reserved peaks, GPU14's 82-second runtime integration, and GPU02's six updates within its failed 71-second allocation. These support the unchanged bounded slice, not a successful full-run prediction. CPU preparation is already accepted.

   The unchanged supervisor retains **240 seconds to first verified GPU work, 540 seconds for the worker, 550 seconds for the parent loop, and 580 seconds for final completion**, inside 600 seconds. Its independent owned-process, framework, and progress checks are preserved. Static shell parsing confirms the exact isolated `python -I -B` bootstrap, supervisor hash, and new inner-packet hash. The batch script differs from launch02 only in job name, output/input paths, and the inner-packet pin; offline, single-thread, and cuBLAS settings are unchanged.

4. **GPU02 closure and admission reconcile.** The predecessor is failed job **`35378508`**, and `predecessor_outer_sha256` authenticates the actual launch02 packet. [failure_acceptance02](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/failure_acceptance02/acceptance.json) authenticates the disposition, inventory transport, and closed failure review; its copied review equals the guidance review. These anchors and both failure-acceptance records match committed bytes. The prior closed review established 308 raw files / 66,951,929 bytes and 312 custody files; that full custody audit was not repeated. Both historical failures remain preserved failures, with no completed fresh checkpoint reload or training/runtime success accepted.

   Saved GPU02 readiness and remote intent agree on **29 prior terminal jobs, 8,494 CPU-seconds, 1,168 GPU-slice-seconds, and 60 journals**. The saved result authenticates its remote intent and job ID. Accepted GPU02 usage is **4 CPUs × 71 seconds = 284 CPU-seconds**, and **one slice × 71 seconds = 71 GPU-slice-seconds**. Adding that parent once and its intent/result pair gives **30 terminal jobs / 62 journals**, **8,778 closed CPU-seconds / 1,239 closed GPU-slice-seconds**. These are saved observations; fresh remote reconciliation remains a launcher gate.

   Admission is exactly **`8,778 + 3,600 + 4,800 = 17,178 / 57,600 CPU-seconds`** and **`1,239 + 600 = 1,839 / 7,200 GPU-slice-seconds`**. The candidate reservation remains eight allocated CPUs for 600 seconds and one GPU slice for 600 seconds. The previous candidate reservation is replaced by its actual usage, not charged twice. Packet, preparation script, launcher reconciliation, and collector baseline agree. The unchanged **3,600 CPU-second preparation allowance** remains an allowance; incompletely metered preparation overhead is not converted into measured usage.

5. **Once-only launch safeguards are preserved.** Both existing `ace-delivery-controller.lock` and `ace-foundation-controller.lock` are addressed with nonblocking exclusive locks held across readiness, transfer, verification, and submission. Their files exist; this review did not acquire them. The launcher requires committed equality for every transfer input, the outer packet, itself, and this review. It rejects an existing local intent, related active work, changed historical totals/job/journal counts, unsuitable account/partition/slice conditions, and an existing dedicated remote root.

   Local/external receipts and local/remote submission intents use exclusive creation. All **48 transferred hashes** are verified before local intent, then checked again remotely before exclusive remote intent. Exactly one `sbatch --parsable` follows that intent; failures and ambiguous outcomes retain evidence and do not retry. No cancellation or retry path was added. The review does not assert current scheduler/account state or remote-root absence.

6. **Every new destination is 03, and opaque hashes survived literal replacements.** New local routing uses `launch03`, `terminal03`, and `validation03`; remote logs, inputs, and qualification output use `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_gpu_20261108_03`. External launch/terminal/validation destinations use their corresponding `20261108-03` names. The launcher references this packet03 review. Legitimate inherited CPU/runtime paths remain unchanged.

   The monitor retains saved-job routing, `*-generation-*.json`, GPU samples, phases, and terminal/failure evidence. The collector retains complete custody, observed-resource accounting, all 48 input checks, and uncertainty for missing/conflicting allocation evidence, now adding actual usage to **8,778 CPU / 1,239 GPU**. Validation03 differs from validation02 **only in routing** and retains successful-terminal, resource, input, custody, first-work, cleanup, and saved-worker gates. Neither collector nor validator marks training/science accepted automatically. Neither was executed.

   Control-file diffs contain only the intended routing, accounting, predecessor, and correction-description changes. Embedded opaque digest comparison found **no unintended changes**: the only changed control-script hash literal is the deliberate predecessor outer pin, which matches launch02's actual bytes. Outer input-pin changes are exactly worker, `training.json`, and `job.sbatch`; their new hashes all match actual files. Runtime archive/manifest pins are unchanged. All **52 local launch files** equal their external local copies byte for byte, with exact membership and no symlinks or launch receipts. New local/external terminal03 and external validation03 outputs are absent; local validation03 contains only `check_saved.py`.

Concrete SHA-256 anchors; the authenticated inner and outer manifests enumerate every individual source/input pin:

| Artifact | SHA-256 |
| --- | --- |
| launch03/packet.json | `50e71a2ccc98e23994234ae0a01e9c00155132e2a67a62d02f5b6510ede49d50` |
| launch03/training.json | `566bbca00939cd339ebf14e58c95bd40eaffb76e467c2847df503317bea1e231` |
| launch03/job.sbatch | `0cc8c445c536e19e97f3f1373f36b09bc0cd72b497f251fb28881088c2f8ef1d` |
| Reviewed worker | `faa3111ccfdbfb8004eb539a41dcc57df98183037a5ed25afd294fa2cec4f1bd` |
| Unchanged supervisor | `5fe610684e077469444fd8b6db4fcd6b9b717ebe24b22deaf3425ece475dc6b7` |
| CPU preparation | `b4394a7b0160b2d8f23df709334dbbc6cace2bac3fb1cd053602c7260d569d73` |
| CPU acceptance | `1b22a1588fdb846649a01b597e0ec705342a6328cf81929d361e639db0512f81` |
| Prepared file | `ceb5847336cf664b215a97d0c8f56f86f2f696e02e44f684ec5c9277e364653e` |
| Training-plan source | `92c207e0a5fb6bd27faad77b05e0d407fd5fd2b9604386839ae35552e38f4924` |
| Training-engine source | `dccff3ada3e3c269368ffea5d6f23981575287528082e4fffdfc81f5af778651` |
| launch03/prepare_packet.py | `838b40e384211804f7e23b097dc3a02a68e29ef760b2ea852a229d860f29403e` |
| launch03/launch_once.py | `fcb6961d1cd2962706c77ca041bcd55f831f2ebd351948ee05630e644529b944` |
| launch03/monitor_once.py | `29f62b3d8091233c2373bfad8c0c63a8f064639f2ac584f58569fbf6cdd8055b` |
| launch03/pull_terminal.py | `ba1730ecbace2ecdfd33ef4237d4ff11440ea7ed3d8fa373bf207fec96fbb043` |
| validation03/check_saved.py | `e146a0c6662a84fdd43a1ffb6c89f87d1e08d642728c9b1bfbabdc3c357842e8` |
| workspace_correction01/receipt.json | `3e5d3620fcfe88dcd703d3fb33ee1938e79460e6e900b2287dd06e4c22c90106` |
| workspace_correction01/evidence-sha256.json | `6f00895fda5a33679c695c7e5181134900d355c0f37ca2f2664352fe8f17217c` |
| Closed workspace review | `f2901b1ca4297b58e243e066506aaf84a317b6d05155e387a0644487bcac9086` |
| Closed packet02 review | `32de421ba5b42d33ee3c1a2cf1638f6b0a02fc9c39c49388f62c798afa5d3081` |
| launch02/packet.json | `793c5b5588865a196354849972799d892bf2dc74855245177c1598fe3c7ddd5d` |
| failure_acceptance01/acceptance.json | `bd6a91f38c49eda4d6ea585d2606a45958654c1347d06538cfd4fbfb908c96cc` |
| failure_acceptance02/acceptance.json | `d2b2b7588d5c15f0a302f79f29f3272ac4155fe500a19a78d94967f857ff4ed2` |
| Closed failure02 review | `d541389ed6f4eb645667c4378c829a708ece3e8bbc2f80357886b0501b2f0c77` |
| terminal02/disposition.json | `5776313d3f8a30c06ec5787f57d816e4c7b4c4f31097bd1345b33bec199a9925` |
| terminal02/inventory_transport.json | `0582ae6f9ab950e1799f4e1b5ef02e00ecca9ccf173df33aea62c821e975bdc3` |
| GPU02 remote intent | `af6ca07b68bb8b54c9070bd3904a1a1f6cf227543ef2cf24ff20b5c2a7ad44a2` |
| GPU02 remote result | `106cbe32589af139410699d40dde75c706e9fa73bc0aa16945f0b48cdf411e3a` |

The inherited runtime archive pin remains `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98`; its manifest pin remains `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a`. This review authenticates their unchanged bindings, not remote bulk bytes. Successful measured workspace release, final reload/inference, observed resources/utilization, cleanup, and complete result acceptance remain post-run evidence requirements.
