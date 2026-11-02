# Execution of the renewed ACE intervention research agenda

## Objective and authority

Patrick has activated the goal: follow [the research agenda](ace_llm_intervention_agenda_2026-11-02.md) until there is a concrete answer to whether an LLM trained to use an SCM can give experimentalists a useful answer at lower experimental cost. This authorizes development and bounded qualification toward that question. It does not turn artificial tests or an isolated favorable pilot into the answer.

The current delivery paper remains preserved at `archive/delivery-scm-2026-11-02`. It is not on a submission path while this scientific prerequisite remains open. Accepted studies, generated reports and failed attempts are immutable. The old publication and October exploration automations remain paused. The active goal is the continuation mechanism for this new research.

Use the agenda's initial ceiling of 16 CPU-core-hours including GPU-host CPU and 2 GPU-hours for this first execution cycle. Reserve a conservative 3,600 CPU-seconds for local preparation, separate from measured child-process CPU. No allocation of the full remaining budget is implied. A new campaign beyond this envelope requires a measured projection and a resource decision. Do not charge this work to the expired October exploration or the delivery replay budget.

## First implementation milestone

New isolated modules are in `scripts/research/llm_intervention_contract.py` and `llm_intervention_learning.py`. They do not modify ACE-Runner or any accepted worker.

The action interface uses an immutable supplied graph and a shared legal menu. It validates intervention bounds and measurement names, records generated completions, binds the parsed action to the executed action, and reserves positive response cost before calling the environment. Repeating an intervention costs another response. Rejected proposals and uncertain or invalid responses terminate the attempt and remain in its journal. There is no automatic fallback or retry. The public state is a detached JSON object containing only declared fields. This is an API boundary, not a malicious-code sandbox. A separate trusted runner still has to keep true mechanisms and private evaluation out of the policy process.

Training scores action tokens conditional on the complete experimental-state prompt. Prompt and padding tokens do not enter the action loss. Supervised action training and state-conditioned DPO are separate operations. Torch-only low-rank adapters freeze the base model and keep a separate fixed reference adapter state. Evaluation mode disables dropout during likelihood scoring while retaining policy gradients. No new training package is installed.

These primitives are not a trained scientific intervention policy. The remaining implementation includes eligible mechanism fitting, branch-isolated training feedback, numerical and learned-policy controls, tokenization/source/model/runtime gates, process separation, a full episode runner and independent scientific reporting.

## New artificial evidence

Exclusive local attempts `ACE-LLM-intervention-preparation-20261102-01` and `-02` are under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline`. Each retains the exact four sources, command outputs and receipt. Attempt01 passes nine contract and six arithmetic/torch methods. Review-driven changes add generated-rejection and dropout/dtype checks. Attempt02 passes ten contract and eight learning methods. These use tiny artificial tensors and callback responses, not pretrained models or scientific worlds.

Attempt01 records 1.052461 child CPU seconds and 3.6667675 elapsed seconds, with maximum child RSS 313,589,760 bytes. Attempt02 records 0.941024 child CPU seconds and 1.0193475 elapsed seconds, with maximum child RSS 314,097,664 bytes. These are local macOS command measurements. Preparation, source inspection, review and transport are not fully metered. They are not a whole-project CPU total.

The current local preparation environment reports torch 2.14.1 and transformers 4.51.3. The existing CURC environment reports Python 3.10.19, torch 2.9.1, transformers 4.57.3, NumPy 2.2.6, SciPy 1.15.3, safetensors 0.7.0 and huggingface-hub 0.36.0. Neither environment has peft or accelerate. Historical environment labels are not treated as proof of current versions.

The cached Qwen3-0.6B model revision is `c1899de289a04d12100db370d81485cdf75e47ca`. Seven existing local files total 1,519,182,365 bytes. Header inspection finds 311 BF16 tensors with 751,632,384 stored elements. This is metadata inspection, not model loading or memory profiling. The [official model card](https://huggingface.co/Qwen/Qwen3-0.6B) lists Apache-2.0. Low-rank adaptation and preference training follow the general mechanisms of [LoRA](https://arxiv.org/abs/2106.09685) and [DPO](https://arxiv.org/abs/2305.18290). Neither implies optimal intervention choice.

## CURC qualification plan

The shared master and remote channel worked on November 2. The fresh user queue was empty. Both local shared controller locks were acquired before preparation. Scratch stat/df and a dedicated exclusive write/read/cleanup probe passed. No matching new-line remote submission journal existed at that check.

The next bounded job is CPU-only: one CPU, 4 GiB, ten minutes, account `ucb736_asc1`, partition `acpu`, QOS `cpu-normal`, no GPU or requeue. It verifies the five new source files, seven copied cached model files and declared runtime versions. It tokenizes an artificial public state, inspects the weight header and runs the 18 artificial methods. It does not load a pretrained model or generate scientific responses. Source packet and submission journals must be pinned and exclusive before the single `sbatch` call.

The CPU job requested at most 600 CPU-seconds. Its live scheduler allocation reports two CPUs despite the one-CPU request, so reserve 1,200 scheduler CPU-seconds conservatively. With the 3,600-second preparation allowance, current reserved CPU time is 4,800 of 57,600 seconds. The scheduler also reports billing=1. Do not infer physical-core count or replace the recorded allocation with that billing value. GPU work remains unallocated. A later smoke must justify the smallest compatible slice from model/header bounds and then measure actual memory, utilization, compute process and progress. Downloads, verification, tokenization and CPU simulation should not occupy a reserved GPU.

### CPU qualification attempt 01

Submitted once as job `35200690` at `2026-11-02T20:03:27Z`. Source commit `2752daf8ef005e9926657dfff4f05d9207780ab6`, packet commit `96d6f741f9f82ba6859dfb73fc476d253fcf60ea`, freeze SHA256 `9c114bc8d557f677dbd71bd36c6123daae9b3fff3d3bce96322c73b94ced6364`. All eight remote input files matched the local pins before submission. The existing cached model's seven files transferred successfully, with their full byte authentication delegated to this CPU job before tokenizer loading.

Remote root: `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01`. Exclusive local preparation: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-cpu-20261102-01`. Repository records: `results/ace_llm_intervention_runtime_20261102/cpu01`.

At 20:03:58 UTC the specific job was RUNNING, elapsed 18 seconds, account `ucb736_asc1`, partition `acpu`, AllocCPUS=2, memory 4G, no GPU. The source/packet pre-execution checksum checks passed. This is live progress, not terminal qualification. Do not duplicate or retry the job. Require terminal worker evidence, actual scheduler completion/accounting and exclusive local output custody before calling the CPU gate passed.

## Completion criteria still open

The active goal requires a tested trained LLM policy, fair numerical and learned controls, fresh independent evaluation, an acquisition-versus-learner distinction, complete costs and failures, and an experimentalist task with useful measured outcomes. Artificial interface tests establish none of the comparative scientific claims. The goal remains active.


### Completed CPU qualification and exact local custody

Job 35200690 finished COMPLETED/0:0 on November 2. All five source pins, seven cached-model objects, declared package versions and Python 3.10.19 matched. The worker passed all 18 artificial methods and tokenized an artificial prompt to 311 tokens with an eight-token action including EOS. No pretrained model was instantiated. Full exclusive local custody contains all 17 non-model remote objects, 61,965 bytes, with exact membership and byte hashes checked. `results/ace_llm_intervention_runtime_20261102/cpu01/cpu_acceptance.json` binds this narrow acceptance. Post-completion scontrol was unavailable. Identity is supported by the submitted normal job ID and matching sacct raw/display parent.

The scheduler allocated two CPUs for 285 seconds, or 570 allocated CPU-seconds. Worker process CPU was 5.096893548 seconds, elapsed time 251.824212783 seconds and peak RSS 814,465,024 bytes. Scheduler total CPU and batch RSS have different scopes. Child steps are not added to parent allocation. The roughly four-minute startup/verification interval needs phase-level diagnosis before GPU work. It does not establish measured GPU memory demand. The reservation remains 4,800/57,600 CPU-seconds including the preparation allowance and conservative CPU-job cap, with no GPU allocation.

### Tractable mechanism learner

`llm_intervention_scm.py` implements a linear-Gaussian reference learner using paid receipts. Clamped mechanisms are excluded, known hard-clamp values remain usable as parents even if not measured separately, and genuinely missing-parent rows are omitted. This is exact conjugate updating for the retained local factors with known noise and error-free covariates. It is not full inference over latent missing parents. Candidate forks are independent, composition uses predicted parents, and posterior planning draws do not call the environment.

A distinct review found one required correction to unmeasured hard-clamp parent handling. It is fixed. Independent nonorthogonal posterior and controlled-RNG covariance/noise/parent-flow fixtures strengthen the check. Eight artificial methods pass in exclusive preparation02 after six initial methods in preparation01. Final SCM source SHA256 is `296aadb60347da5c948028b7de10440e361f54a4427f2a0655e39961ca7e2a7b`. The [SCM review](reviews/ace_llm_intervention_scm_review_2026-11-02.md) has zero required findings remaining. These checks do not involve trained language models or fresh scientific worlds.

The SCM attempts record 0.095066 and 0.088943 child CPU seconds. Together with the two earlier local fixture attempts, measured local test CPU is 2.177494 seconds. This excludes other preparation and review. Exact sources, outputs and receipts remain in `results/ace_llm_intervention_scm_preparation_20261102` and exclusive external originals. The new SCM is outside the completed CPU job's source freeze.


### Charged episode and training supervision

The isolated episode runner now sends detached public JSON to a separate policy process, executes only its returned action through the charged executor, and updates the SCM from the resulting paid receipt. Initialization is charged explicitly. A budget-cap stop is labeled as a driver action, not an LLM decision. Invalid completions and worker failures terminate the episode without fallback.

Training-only preference construction forks the same pre-action learner for every candidate. Every branch and shared independent scoring response is charged. Scoring outcomes cannot update branch predictors. An explicit journal-to-learner receipt mapping prevents accidental row replacement. Gain is normalized by declared intervention cost and a fixed paired-standard-error tie rule retains ambiguous pairs. This tie heuristic is not inferential assurance. Future frozen callbacks must establish stream and world independence.

Uniform random, least-used menu coverage and expected local-parameter-information-per-cost are separately named numerical controls. The information control uses learned-model planning draws, not environment calls. It is a tractable reference teacher. A target-risk acquisition policy and matched learned numerical ranker are still required before the scientific comparison.

Eight initial artificial episode methods passed. A distinct review found and closed two required issues, then two new focused rejection methods plus the affected process episode passed. The fixes reject uncharged preloaded SCM state and contradictory scoring clamps. See [episode review](reviews/ace_llm_intervention_episode_review_2026-11-02.md). All attempts are preserved. The contract and SCM now have new source digests, so the earlier CPU receipt continues to bind its historical bytes only.

A separate actual-Qwen GPU smoke is being prepared. It is not submitted or qualified. Proposed resources are one A100 20 GB slice, four requested CPUs, 12 GiB host memory and fifteen minutes. Conservative allowance of eight allocated CPU threads would reserve 7,200 additional CPU-seconds and 0.25 GPU-hours if submitted. Do not count this proposed stage as an actual allocation. Its fixed artificial training labels test adapters and the action interface, not scientific usefulness. Full phase telemetry, independent owned-process GPU evidence, bounded cleanup, separate source/resource review and exclusive journals are required before execution.


### Once-only CPU prerequisite for the actual LLM smoke

After independent source and actual-packet review closed all required code issues, CPU job35202401 was submitted once at2026-11-02T20:46:09Z. Source commit `fb6fe997a89e2dc24f0efd26d24aaaa6c4ae248a`, packet commit `0d63b008`, freeze SHA256 `62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6`. Remote root `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01`, exclusive local preparation `ACE-LLM-intervention-gpu-cpu-20261102-01`, repository records `results/ace_llm_intervention_gpu_runtime_20261102/cpu01`.

All nine remote inputs were authenticated before the single submission. Both local controller locks remained held, prior new-line journals were reconciled, matching queue/accounting were empty, and a fresh dedicated scratch write/read/cleanup plus owner/df check passed. The request is oneCPU/4GiB/tenminutes on acpu/cpu-normal, accountucb736_asc1, noGPU and no-requeue. Conservatively reserving two allocated CPU threads adds1200seconds. Current reservation is6000/57600CPU-seconds with zeroGPU allocation. Actual accounting of this attempt remains open.

This new CPU stage times individual model-file hashing, torch import and tokenizer import/load. It prepares the exact artificial token fixture and runs nine new GPU-packet/cleanup checks. The old18-method CPU family is not repeated. It does not instantiate a pretrained model. The earlier macOS process-group PermissionError remains retained, with target-Linux qualification pending. Require actual scheduler success, complete output custody and both cleanup scenarios before qualifying this stage. Never duplicate a queued/running job or blind-retry an ambiguous submission.

The GPU worker/supervisor are implemented and reviewed but remain unsubmitted. Any later GPU freeze must bind the accepted CPU fixture and qualification receipt, declare package-version scope accurately and preserve the fixed smoke recipe. See [GPU qualification plan](ace_llm_intervention_gpu_qualification_2026-11-02.md) and [prelaunch review](reviews/ace_llm_intervention_gpu_prelaunch_review_2026-11-02.md). The broad goal remains unanswered. No fresh scientific worlds, trained scientific policy, or acquisition-efficiency result exists yet.


### CPU prerequisite accepted

Job35202401 completed0:0 with two allocated CPUs for20seconds, or40allocatedCPU-seconds. All43remote objects,95222bytes, have complete exclusive local custody and exact byte/membership verification. Both real Linux cleanup scenarios pass and all nine new methods pass. `cpu_acceptance.json` binds terminalSHA256 `06d4c4096137a8111d81af61db411aaf8397fe23f516a0a7c1801ca9fdfba127` and fixtureSHA256 `0c0d24ec20af68597aa79f439538c55a76395702246475d1b3466f496f4c1e52`. The earlier macOS failure remains preserved. It is resolved for this target-runtime cleanup gate, not erased.

Worker elapsed14.066608282seconds, processCPU3.982085642seconds, reapedchildCPU0.038981seconds, peakworkerRSS793161728bytes. Weight hashing took about2.545seconds, torch import2.682seconds and tokenizer import7.120seconds. The original252second delay did not recur. These timings do not identify its cause or guarantee future loading speed. Current known allocatedCPU is610seconds across the two new CPU jobs. Reservation remains6000/57600seconds. These are overlapping measurement categories, not an additive whole-project total. No GPU/model/scientific response was used.

Fresh scheduler evidence again shows requestedCPU1 becoming allocatedCPU2 and CPUs/Task2. A narrowly reviewed allocation delta now requires requestedCPU4/memory12GiB from ReqTRES while allowing the already-budgeted actualCPU4–8 with NumCPUs/CPUsPerTask/AllocTRES agreement. GPU/account/time limits are unchanged. Three focused pure resource methods pass, with zero required review findings. The CPU-qualified cleanup/deadline/run function ASTs remain unchanged. Preparation04 preserves this lineage without repeating Linux cleanup. The later GPU packet must bind this delta and the accepted CPU fixture. The actual GPU smoke remains unsubmitted at this entry.


### GPU01 definite submission rejection and diagnosed QOS correction

The first GPU packet was reviewed with zero required packet issues and submitted once at2026-11-02T20:54:04Z. Slurm rejected it with Error7 before assigning a job ID: the chosen `gpu-normal` QOS does not permit `a100_3g.20gb`. No matching queue/accounting record exists. All14remote objects are preserved with exact local custody under `ACE-LLM-intervention-gpu-20261102-01/rejected_remote`, with repository copies in `gpu01/rejected_remote`. No GPU allocation, model load or training occurred. The rejected packet remains unchanged.

The error was an incorrect QOS choice in our request. The [official CURC GRES table](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html#available-gres-on-alpine) and fresh scheduler QOS inventory show that the20GB A100 slice is available under `gpu-testing`, specifically for GPU workflow qualification. This artificial adapter/interface smoke fits that purpose. Production scientific studies will use a supported production QOS. The correction keeps the same partition, GPU slice/count, model, artificial recipe, CPU/memory/wall limits and utilization checks. It does not bypass a workload-efficiency failure or request a larger GPU.

A focused pure check accepts gpu-testing and rejects gpu-normal in both the frozen request and live allocation. The proposed GPU02 packet must independently freeze the corrected QOS gate and preserve GPU01's definite rejection. No blind retry or existing-job replacement is permitted. GPU01's unallocated proposal is not counted as actual cost or a second GPU allocation. Current actual allocatedCPU remains610seconds, current CPU-only reservation6000seconds. A successful GPU02 submission would reserve7200additionalCPU-seconds and900GPU-seconds, for13200/57600CPU-seconds and0.25/2GPU-hours in this cycle.


### Corrected artificial GPU02 submitted once

Job35202931 was submitted once at2026-11-02T21:00:53Z after zero-required independent actual packet review. Source commit3a8a1f21, packet250d79dc, freezeSHA256 `fd041a2401d93a5044ef4c01e37da3a83d7b6fe6fa97f83fde7999fe012a8dd8`. All13 remote input objects and exact membership matched independent local pins. Both shared controller locks remained held and all six previous new-line submission journals were reconciled. Queue and matching accounting were empty, with fresh scratch owner/df/write/read/cleanup before the one submission.

Remote root `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_02`. Exclusive local `ACE-LLM-intervention-gpu-20261102-02`, repository `results/ace_llm_intervention_gpu_runtime_20261102/gpu02`. Initial scontrol reports RUNNING, accountucb736_asc1, aa100/gpu-testing, oneA10020GBslice, fourCPUs/12GiB/fifteenminutes, no-requeue. This allocation is for artificial adapter/interface qualification only. Scientific worlds and utility comparisons remain ungenerated. GPU01 has no allocation and is not counted twice. Conservative cycle reservations are13200/57600CPU-seconds and900/7200GPU-seconds. Actual usage and qualification remain open until supervised terminal evidence, scheduler accounting and complete custody.
