# Execution of the renewed ACE intervention research agenda

## Current state: November 7, 2026

The broader LLM intervention question remains unanswered. No fresh scientific training or development worlds have been generated for this cycle. The earlier delivery paper remains preserved and off the submission path.

The paid training-data builder, prospective screen schedule, private cost-to-accuracy reporter, supplied-model SFT/DPO engine, matched-exposure proposal, numerical-policy trainer, complete-roster descriptive reporter, paired episode runner and response-sealing path are implemented and independently reviewed. Both trainers save the supervised policy and use it as a fixed reference during preference training. These components need complete runner integration and actual model/runtime qualification before scientific use.

GPU09 qualified a bounded artificial training/reload experiment. GPU10 failed during imports before loading a model or generating a response. Patrick supplied the approved renewal INC1508105. ACE now uses **ucb736_asc2**, with the same research ceiling. The pending diagnostic 35341862 was moved to that account, then failed before imports because its pinned allocation guard still required the old account. That oversight and all 10 remote artifacts are preserved. The corrected account-only launch02 was submitted once as **35343088** at 18:02 UTC, with one CPU, 3 GiB and an eight-minute limit. Both probes have ended. The original import order timed out and the second SciPy-first order completed all imports. Sequential cache effects prevent attributing this difference to order alone. Complete terminal custody and independent diagnostic review are accepted with zero required corrections. This accepts the diagnostic evidence, not model-service startup. A new reviewed CPU-only staged-runtime qualification is now queued as **35343957**. SSH works, and both controller locks are released.

Current closed allocation is **4,763 CPU-seconds** and **836 GPU-slice-seconds**. With the cumulative 3,600-second preparation allowance, admission is **8,363 / 57,600 CPU-seconds**. The new staged-runtime job reserves 1,800 additional CPU-seconds, giving admission **10,163 / 57,600**. The18:38:32UTC check confirms RUNNING with1CPU/3GiB/noGPU, and runtime inventory is in progress. The prior diagnostic's unused reservation is released. GPU admission remains 836 / 7,200. These quantities are not whole-cycle process CPU. The dated entries below preserve the evidence and earlier states. Do not repeat closed tests, reviews or artificial model runs merely to reconfirm them.

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


### GPU02 failed before child creation, typed-GRES parser diagnosis

Job35202931 failed1:0 after8scheduler seconds with4CPUs andoneA10020GBslice allocated. The supervisor rejected the actual allocation record before creating its worker. Slurm's valid record contains typed `gres/gpu:a100_3g.20gb=1` without a redundant generic `gres/gpu=1`. Our parser incorrectly required both. No model was loaded, no CUDA work was attempted and no training or scientific response occurred. This is an allocation-parser defect, not a model or utility result.

All21remote objects totaling102689bytes have complete exclusive local custody. `gpu02/failure_disposition.json` binds the failure and accounting. The attempt used32allocatedCPU-seconds and8allocatedGPU-seconds. Known new-cycle allocatedCPU is642seconds. GPU reservations remain900seconds and CPU reservations13200seconds, distinct from actuals and unmetered preparation.

The bounded correction requires the exact typed GPU count1, accepts an omitted generic count and requires agreement when the generic count is present. Other GRES types and duplicate/conflicting counts remain rejected. One new regression method uses the saved real scheduler record, includes both positive representations and six negative subcases. Preparation06 records0.065075childCPU seconds and0.120582042elapsed seconds. No completed old tests repeated. A new reviewed exclusive GPU03 packet would keep model, recipe, CPU/memory/GPU/time caps unchanged. Its cap would increase conservative cumulative reservations to20400/57600CPU-seconds and1800/7200GPU-seconds. It is not submitted at this entry.


### GPU03 submitted once

Job35203022 submitted once at2026-11-02T21:05:33Z. Source0c87bce4, packet131a5d8d, freezeSHA256 `2ef3056c6a26d9bb7ce89dd2577eaa890a8e5a5610aab9e796aec778b20281db`. All15input objects matched remotely. Both controller locks held, eight prior journals reconciled, queue empty and dedicated scratch readiness passed. The independent actual packet review has zero required issues. GPU02's terminal failure and exact custody were established before this separate submission.

Same artificial workload and caps: oneA10020GBslice, fourrequestedCPUs,12GiB,fifteenminutes,accountucb736_asc1,aa100/gpu-testing,no-requeue. Remote `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_03`. Exclusive local `ACE-LLM-intervention-gpu-20261102-03`. Conservative cumulative reservation20400/57600CPU-seconds and1800/7200GPU-seconds. Actual GPU03 usage and model/interface qualification remain pending. Scientific data, trained intervention policy and utility comparison remain ungenerated.


### First actual Qwen training reached GPU computation, then failed the action interface

GPU03 job35203022 finished FAILED/1:0 after252scheduler seconds. All54remote objects,146515bytes, are retained in complete exclusive local custody. The repository copy is `results/ace_llm_intervention_gpu_runtime_20261102/gpu03/terminal`. `terminal_inventory.json` and `failure_disposition.json` bind the exact files and scoped result. No success receipt exists.

The model completed8SFT and4DPO adapter updates. All12 recorded gradients were finite and positive. Training CUDA events total3929.577133milliseconds and maximum framework allocation was3301825024bytes. Twenty-four independent NVIDIA samples observed the owned compute process. These jointly establish real GPU computation. They do not establish a successful intervention policy. The recorded suffix was `{"action_id":"a0}` with a missing closing quote. The strict parser rejected it before any response reservation. Both scientific and artificial response counts arezero. No adapter checkpoint was persisted because the old worker saved it only after successful generation. Preserve that limitation.

The combined imports/CUDA-ready interval was162.811203392seconds after input verification, model loading35.487865564seconds, and first forward completed213.909811386seconds after worker start. These events identify slow phases, not the exact cause. The next worker should split import and CUDA initialization timing, qualify CPU-only imports before allocation, and save trained adapters before generation. The allocation was dominated by startup. Do not infer efficiency from the short successful training steps or expand resources without diagnosis.

Worker failure records11.762056279processCPU seconds and221.770262659elapsed seconds. Supervisor records0.121680565processCPU seconds and246.408147276elapsed seconds, with no remaining owned group and no cleanup errors. Slurm allocated4CPUs for252seconds, or1008CPU-seconds, plus252GPU-seconds. Cumulative known new-cycle allocation is1650CPU-seconds and260GPU-seconds. Conservative reservations stay20400/57600CPU-seconds and1800/7200GPU-seconds. These scopes overlap and exclude unmetered local preparation.

An isolated finite legal-command grammar is now implemented with six new artificial methods passing. It preserves every affordable action andstop, binds exact prompt/token round trips, and does not contain rewards or repair old output. It is a new constrained-generation interface to apply equally to frozen and trained LLMs. Actual-tokenizer and worker integration remain unqualified. See `ace_llm_intervention_grammar_2026-11-02.md`. No new GPU attempt is prepared or submitted after GPU03 at this entry. The next work is CPU qualification and bounded reviewed integration, then strong target-risk numerical acquisition and matched learned controls. Fresh scientific training/development/confirmation worlds remain ungenerated. The active goal remains unanswered.


### Independent GPU03 disposition and grammar preparation

Independent saved-evidence review found no custody/source/count discrepancy and confirmed failed interface qualification. It requires pre-generation verification/adapter persistence, precise cleanup wording and finer startup/shutdown timing before scaling. Current reports disclose these limits, with worker implementation changes still open. The first six generated tokens matched the valid target, followed by wrong token92 instead of9207 and EOS. Eight generated tokens were below the32token limit. No cause of the omission is identified.

Pure single-prompt grammar review found zero required defects. One optional independent exact-set oracle was added and passed as one new focused test. Initial six methods were not repeated. Actual tokenizer/callback/model qualification remains open. `results/ace_llm_intervention_grammar_preparation_20261102/review01/review_receipt.json` pins the reports and source/preparation lineage.

The shared CURC master disappeared during the final queue check after complete GPU03 terminal custody was secured. A private reconnect request is pending. Do not claim fresh queue access from old evidence. No new job was submitted after35203022. Continue local worker diagnostics and CPU qualification preparation, without altering any original packet. Fresh access and both shared controller locks must be reconciled before another submission.


### CPU qualification of constrained Qwen decoding submitted once

The restored CURC channel worked and the user queue was empty. Both shared local controller locks were held. Ten preceding new-line journals were reconciled, and fresh dedicated scratch ownership/df/write/read/cleanup passed. Twelve input objects and exact membership matched independent pins. Implementation review fixed one lazy architecture-import issue, then its narrow recheck and distinct actual-packet review had zero required issues.

Job35204167 submitted once at2026-11-02T21:36:07Z. Source334700845d19c8e6406a7c01865a5a95d74faa6e, packet45de9aae, freezeSHA256 `cf73af49666a76079c44cedbffcd93e63f19a561ae4da745438ba449019cf89c`. Remote `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_decoding_cpu_20261102_01`. Exclusive local `ACE-LLM-intervention-decoding-cpu-20261102-01`, repository `results/ace_llm_intervention_decoding_runtime_20261102/cpu01`.

Request1CPU/4GiB/tenminutes/acpu/cpu-normal/accountucb736_asc1,noGPU,no-requeue. Conservatively reserving two allocated CPU threads adds1200CPU-seconds, for21600/57600CPU-seconds and unchanged1800/7200GPU-seconds. No new GPU attempt is submitted. The CPU command authenticates source/model/runtime/oldfixture, imports the actual Qwen3 class without weights, and checks the real tokenizer and Transformers prefix mask. Full target qualification and actual allocation/custody remain pending. Fresh scientific worlds and a trained scientific intervention policy remain ungenerated.


### Target CPU decoder and Qwen implementation accepted

Job35204167 completed0:0 with two allocated CPUs for85seconds, or170allocatedCPU-seconds. All36remote objects totaling142195bytes have exact exclusive local custody. `cpu_acceptance.json` SHA256 `9e5c27346d1ee0ad251fa582128e97ce9de6d61ba66a8e3fa8c875745a02d864` binds source/runtime/oldfixture, newdecodefixture, processorchecks and Slurm identity/resources. New decoding fixtureSHA256 `30aeca96be45748d3bcc1e0233c02eafec91605d9a76cb0667d6cb369f3d6ec7`. The old fixture remains0c0d24ec unchanged.

Allthree commands and14distinct prefixes pass the actual Transformers4.57.3 CPU mask with allowed logits unchanged and all other tokens excluded. Two negative callback binding checks pass. The actual Qwen3 class imports without instantiation. Worker5.196020884processCPU seconds/65.846629784elapsed seconds/840159232peakRSS bytes. Torch import25.624700010seconds and Qwen3 implementation import5.881041544seconds were recorded. At60seconds the stack sampler observed Python reading import code through SciPy/Transformers. This identifies one sampled import operation, not the cause or full duration of every delay. Startup and process teardown remain materially slower than kernel work, so these short qualification attempts do not establish efficient production use.

No pretrained model, gradient update, GPU allocation or environment response occurred in this CPU stage. Known cumulative new-cycle allocation is1820CPU-seconds and260GPU-seconds. Reservations remain21600/57600CPU-seconds and1800/7200GPU-seconds. A subsequent unchanged-cap GPU04 would bring reservations to28800CPU-seconds and2700GPU-seconds. It must bind the accepted decoder, reviewed newworker and CPU evidence in a new packet, then pass actual generation and cleanup before qualification. It is not submitted at this entry.


### Constrained-generation artificial GPU04 submitted once

Job35204547 submitted once at2026-11-02T21:44:37Z after the reviewed CPU bridge and all20remote input pins matched. Source334700845d19, packet912422a1, freezeSHA256 `ae98aa3627d8738eef31c2307a5f9731c2de643ec5825a697e15215ba70d077b`. Both shared controller locks remain held, twelve previous journals were reconciled, userqueue and matchingaccounting were empty, and freshscratch ownership/df/write/read/cleanup passed.

Remote `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_04`, exclusive local `ACE-LLM-intervention-gpu-20261102-04`, repository `results/ace_llm_intervention_gpu_runtime_20261102/gpu04`. Samecaps:oneA10020GBtesting slice,4requested/≤8allocatedCPUs,12GiB,15minutes,accountucb736_asc1,aa100/gpu-testing,no-requeue. The new finite-public-menu decoder admits allaffordableactions andstop, preserves the fixedtrainingrecipe, saves adapter/checks beforegeneration and gives imports/load/cleanup separate timing. This is artificial training/interface qualification only.

Conservative cycle reservations are28800/57600CPU-seconds and2700/7200GPU-seconds. Known actual allocation beforethisattempt remains1820CPU-seconds and260GPU-seconds. ActualGPU04cost/supervision/custody/qualification remains open. No scientificworlds orutilitycomparison has been generated.


### Actual constrained Qwen training/action qualification accepted

GPU04job35204547 completed0:0 after36schedulerseconds. All80remoteobjects4804060bytes match completeexclusive localcustody. Independentpost-runreview foundzero requiredfindings. `gpu04/qualification_acceptance.json` bindsfreezeae98aa36, terminal184ca5b5, supervisor6bc070b3, adaptere7eea9dc, pre-generationchecke5679a6b andcompleteinventory4cc55223. Source33470084 andpacket912422a1 remainunchanged.

The modelcompleted8SFT+4DPO updates withfinitepositivegradients, unchangedbase/reference andchangedadapter. Itgeneratedthevalidcommand `{"action_id":"a0"}` andexecutedonechargedartificialinterventionX=-1,responseY=-1. Scientificresponseszero. TrainingCUDAevents1.185772835seconds, frameworkpeak3301825024bytes andthreeownedNVIDIAsamplespeak3718MiB independentlyestablishGPUwork. Bothownedprocess-group andfinalNVIDIAcleanup pass. Thisqualifiesoneartificialtraining-to-actionpath, notutility,generalization, numericalrepeatability orcausaleffectoftrainingonvalidgeneration.

Firstforward25.414seconds, worker28.979073166elapsed/10.519730091processCPU/2404438016peakRSS, supervisor30.836143406elapsed/.037279901CPU. Verification12.159seconds andnamedimports9.724seconds dominatedstartup. CPULoadcall.066seconds isnotallphysicalweightIO. Schedulerallocated4CPU×36seconds=144CPU-seconds and36GPU-slice-seconds. Knowncumulativeallocation1964CPU-seconds and296GPU-seconds. Conservative reservations28800/57600CPU-seconds,2700/7200GPU-seconds. Theseoverlap andexcludepartlyunmeteredpreparation. No efficiency/speedupclaim.

GPU03andallpredecessorfailures remainpreserved. No freshscientificworldsaregenerated. Nextwork: target-relevantnumericalacquisition, sharedpublictaskdescriptor, matchedlearnednumericalranker, chargedpolicy-trainingsupervision andseparatefreshdevelopment/confirmation. Adapterreload/multiepisodeinference andproductionresourcequalification remainopen. Acceptedpaperstudiesandarchivedmanuscript remainunchanged.


### Target-aware numerical acquisition reference

The new isolated `llm_intervention_target_risk.py` estimates target prediction risk reduction per intervention cost. It retains shared-ancestor uncertainty through an exact centered moment recursion for the current independent-head linear SCM. Candidate lookahead uses noisy model-generated fantasy receipts and the same local update eligibility. It never queries the environment while planning. Negative estimated gains remain visible. Its Monte Carlo errors describe planning precision, not scientific confidence.

Eight new distinct analytic fixture methods pass across preparations01–04. The initial six exposed no issue, but a separate large-offset probe and independent review identified cancellation in the raw-moment implementation. The centered correction passes the new regression and three affected checks. One optional fixed-fantasy downstream-risk oracle then passed alone. All source versions and the failed probe remain preserved. Independent review has zero required findings remaining. Final source SHA256 is `910d2a7046a0f403835befb2db3c625b0919e4be20d48bc09baa58423df9afee`.

Design and limits are in `ace_llm_intervention_target_risk_2026-11-02.md`. The final preparation05 receipt binds sources, tests, review and all preceding attempts. No pretrained model, scientific world or new CURC allocation was used. The three metered fixture commands used0.117156,0.085137 and0.085011childCPU seconds. Probe, implementation and review overhead are not fully metered. Cycle allocation remains1964CPU-seconds and296GPU-slice-seconds, with conservative reservations28800/57600CPU-seconds and2700/7200GPU-seconds.

Next implementation must expose the same structured prediction task to every LLM and numerical policy. The matched learned numerical ranker, charged training-world supervision, adapter reload and multi-episode inference, then separately frozen development and confirmation remain required. No fresh scientific namespaces have been generated. The strongest present conclusion is a successful artificial Qwen training/action path plus reviewed numerical primitives. Lower experimental cost remains an open scientific question. Both shared controller locks were released after all GPU04 custody was secured, and the final live user queue was empty.


### Shared task, matched learned control and reloadable language policy

New source commit19e966aa adds a common prediction-task envelope and a learned numerical intervention control. Each policy receives the target, weighted prediction queries, accuracy goal, complete paid history and SCM summaries. The numerical encoding contains the graph, intervention/measurement masks and full parameter covariance, with 8891 features per candidate. Its default two-layer ranker has1154817 trainable parameters. This is a learned numerical comparison, not a matched-architecture or equal-compute control.

Thirteen distinct artificial methods pass. Independent review found two required defects before scientific use. Episodes that exceeded the representation limits could spend on initialization before being rejected, and ranker initialization could alter accelerator randomness. Both are fixed and their narrow regressions pass. The final task/ranker receiptSHA256 is `517bc586ebd21f8b17e55d57c48cf4a4e117548941135dc02bec7ae513dcb8a5`. Two tiny artificial neural fits were executed, including an affected recheck. No pretrained weights or scientific worlds were used.

A separate converter now binds numerical indices and language JSON labels to one pinned public-state record. Strict stop preferences and ties remain distinct. Four fabricated-record methods pass. Independent review found no required defects. Its final receiptSHA256 is `eacb7bc585dbea20d2297cda196ef79a66ceadbe89beadf1ac27640294941c26`. This checks alignment, not the execution or correctness of a scientific supervision campaign.

The new language-policy primitives authenticate saved adapter bytes before deserialization, validate the complete tensor layout before installation, preserve CPU randomness, freeze inference parameters and share the exact training/deployment chat prefix. Generated commands must satisfy the prompt-bound legal grammar. Eight distinct artificial methods pass. One required review finding was corrected: Transformers must be told explicitly not to merge stored model defaults into the declared decoding configuration. The new regression exercises the actual local configuration helper. Canonical label text is also checked before tokenization. Independent recheck has zero remaining required defects. Final language receiptSHA256 is `cb3f19f0cf1265f05fdc6b6c364202395fba7f3efa7933b0e9546dcf3dc14f07`.

The local cached-Qwen tokenizer profile found613,877,2526,4341 and16010 prompt tokens across five artificial configurations. The five-node,32-row case exceeds4096tokens. The old512-token GPU qualification therefore cannot justify the new task's context or resources. All129profile responses are artificial zero-measurement fixture responses, with charge journals preserved. No model weights were loaded. The profile uses localTransformers4.51.3, not target4.57.3. It measures token counts only, not GPU memory or throughput. Preserve its original source pins after later decoding fixes.

All predecessor sources, tests, logs, profiles and reviews are preserved under the new task/ranker, supervision and language-policy preparation directories, with exclusive external originals. Metered local childCPU seconds are1.129219/1.010986/1.078357 for task/ranker commands,0.086115/0.077777 for converter commands,0.959031/1.312065 for language fixtures, and1.767928 for token profiling. These exclude partly unmetered implementation, inspection, review and Git work. They are not whole-cycle costs.

No new CURC allocation or scientific response occurred. Conservative reservations remain28800/57600CPU-seconds and2700/7200GPU-seconds. Known allocated totals remain1964CPU-seconds and296GPU-slice-seconds. A fresh read-only helper check found masterPID34953 and a successful live user queue query returned no jobs. Both controller locks remain released. This observation does not replace readiness and journal reconciliation before the next submission.

Next work is target-runtime CPU qualification of the new prompt/batch/adapter path, followed by a measured long-context artificial GPU stage only when justified. The all-position vocabulary projection in the current loss primitive deserves memory analysis before longer training batches. Any action-position optimization needs score and gradient equivalence checks. Then freeze charged policy-training supervision, matched learned arms and separate fresh development/confirmation. No scientific training worlds have been generated. Lower experimental cost remains unanswered. The delivery paper archive and accepted original studies are unchanged.


### Shared-task CPU integration submitted once

Job35207064 was submitted once at2026-11-02T22:42:46Z. Source commitd89d2f7e79f1d676b258d4b9ae72c211820342a5 and packet commite23876aaf57d49d997e0b477f97f7727e6a6ddc8 are frozen. FreezeSHA256 is `f71998c5b73893149b5a7cc9cbb556922ef9ec9667a14b25a0b84b0bbfead622`. All23remote input objects matched independent pins. Both controller locks were held, fourteen prior submission journals were reconciled, six prior job IDs were terminal, the user queue was empty and dedicated scratch write/read/cleanup passed. Independent implementation and actual-packet reviews found zero required defects.

The new action-position scorer retains the full transformer context and gradients while excluding vocabulary rows unused by the action loss. Five new artificial score/gradient methods pass, including fixed-reference preference gradients and malformed-input rejection. A separate CPU allocation-gate check has one historical positive and six fabricated negatives. No old tests or accepted scientific studies were repeated. Local fixture costs0.827029 and0.037904childCPU seconds exclude implementation and review overhead.

The target job requests1CPU/4GiB/10minutes onacpu/cpu-normal/accountucb736_asc1, at most2allocatedCPU threads, noGPU andno-requeue. Worker deadline560seconds. It will authenticate the actual runtime, saved task envelopes and accepted adapter inventory, then check the actual Qwen implementation using tiny random models. It does not load pretrained weights, execute optimizer updates or generate environment responses. Actual full-size inference and long-context GPU demand remain open.

Remote output `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_cpu_20261102_01`, exclusive local `ACE-LLM-intervention-task-cpu-20261102-01`, repository `results/ace_llm_intervention_task_runtime_20261102/cpu01`. Conservative reservations now30000/57600CPU-seconds and2700/7200GPU-seconds. Known prior allocation1964CPU-seconds and296GPU-slice-seconds excludes this job. Runtime qualification and actual new cost remain pending at this entry. Do not resubmit a queued/running or ambiguous attempt.

A distinct independent design recommendation for the first24-world/eight-arm screen is preserved in `reviews/ace_llm_intervention_first_screen_design_2026-11-02.md`. It proposes separate32training worlds, charged replicated branch feedback and fixed useful-accuracy endpoints. It is not a scientific freeze. The runner, generator/evaluator separation and adequate training exposure still need implementation. Fresh scientific namespaces remain ungenerated, and later ablations, confirmation and engineering utility remain part of the active goal.


### Accepted target CPU scoring, reload and generation integration

Job35207064 completed0:0 with two allocated CPU threads for163seconds, or326allocatedCPU-seconds. Complete exclusive local custody contains43remote files totaling4908699bytes. Every byte and relative path matches the remote terminal inventory, and all23frozen input pins still match. Worker terminalSHA256 is `3641f036680256a5c5a22a54ae9e4f0562f717dc2ffcd975a238555a438c2926`. The source and packet remain unchanged.

In the actual Py3.10.19/torch2.9.1/Transformers4.57.3 runtime, tiny random Qwen models produced identical full/selected supervised and preference losses. The maximum difference among eight adapter parameter gradients waszero for both objectives. Reloading the fabricated tiny adapter into the unchanged tiny base also gavezero score difference. These are scoped FP32 integration checks, not a universal numerical proof or full-size pretrained result.

All five authenticated task fixtures produced the same613/877/2526/4341/16010prompt lengths observed locally. Two actual tiny-Qwen generations returned the legal action `a00` on different public histories. Neither was sent to the environment. The accepted full-size GPU04 adapter's112tensor inventory and1146880parameters were checked against its independent digest and the cached architecture, but that full model was not loaded or restored in this job. Optimizer updates, pretrained-weight loads, environment responses and scientific worlds wereallzero.

The worker used5.348364032processCPU seconds,129.90897520584986elapsed seconds and851607552peakRSS bytes. Slurm batchMaxRSS was1206120K andTotalCPU6.282seconds across the job. These are different measurement scopes. Authentication ended at10.609seconds, imports at125.354seconds and the tiny execution ended at129.907seconds. An owned stack sample observed a Transformers import operation. That sample does not identify the cause of every delay. Startup dominates this CPU qualification, and no GPU efficiency claim follows.

Known cycle allocation is now2290CPU-seconds and296GPU-slice-seconds. Conservative reservations remain30000/57600CPU-seconds and2700/7200GPU-seconds. They overlap with process and elapsed measurements and exclude partly unmetered local work. No new GPU allocation was made.

Next work is actual full-size adapter reload, long-context memory and gradient qualification with measured GPU demand, then the separately frozen charged training and paired development study. The first-screen design recommendation remains unexecuted. The active scientific question remains unanswered. Accepted original studies and the archived manuscript are unchanged.

Independent saved-artifact review found zero required corrections. Final acceptance SHA256 is `90ef0d4ab4a74589735726bab5ca22bbd8ff1720c67044e1d526d9f63974a31d`. It binds the complete inventory, parent checks, original terminal and separate review.

After acceptance, both controller locks were released. The final live user queue was empty. The complete43-object custody is explicitly tracked, including logs and adapter bytes. No task remains running from this attempt.


### Full-size shared-task GPU qualification submitted once

Job35208932 was submitted once at2026-11-02T23:07:20Z. Source384c2d817d68ff27e8cb3ff99d2c673749966a4c, packet6d4d2478e148d7718d1311b89230bda55e71f7a7 and freezeSHA256b17599b6ed864c603270c097d86ada40eaebc696dfd9815dac80f7769be599c9 are fixed. All26remote input pins matched. Both controller locks were held, sixteen prior journals and seven terminal job identities reconciled, the live ACE queue was empty and scratch write/read/cleanup passed. Independent source review found one discrepancy-preservation issue, fixed before freezing. Its narrow recheck and the distinct frozen-packet review found zero required corrections.

This is new full-size artificial qualification, not a repeat of the prior short prompt or tinyCPU checks. It restores the accepted GPU04 adapter into the pinnedQwen0.6B, compares full/selected losses and112adapter gradients at613prompttokens, and measures threeSFT/threeDPO updates at4341/7343tokens. It freezes FlashAttention-only SDPA, BF16base/FP32adapters and no activation checkpointing. Full prompt/grammar/label-mask tokens were prepared onCPU. It saves the original preference reference separately, reloads the new artificial policy into a fresh base and generates four commands without executing them. All scientific worlds and new environment responses remainzero.

RequestoneA10020GBtesting slice,4CPUthreads with at most8allocated,12GiB and15minutes onaa100/gpu-testing/accountucb736_asc1. Startup420/worker840/supervisor870seconds and owned-group/GPU cleanup remain enforced. No scientific use of testingQOS is authorized by this packet. The bounded test may fail and every failure is retained. No blind retry.

Remote output `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05`, exclusive local `ACE-LLM-intervention-task-gpu-20261102-05`, repository `results/ace_llm_intervention_task_runtime_20261102/gpu05`. Conservative reservations now37200/57600CPU-seconds and3600/7200GPU-seconds. Known allocation2290CPU-seconds/296GPU-slice-seconds excludes this new job. Local preparation is not fully metered. Five pure contract methods and one new discrepancy-persistence method passed, with no repeated old numerical qualification. Actual runtime, resource measurements and complete custody remain pending.


### Full-size GPU05 failed and preserved before optimization

Job35208932 is terminalFAILED1:0. It allocated4CPUthreads andoneA10020GBslice for48seconds, or192CPU-seconds and48GPU-slice-seconds. Full60-object terminal custody totals5325554bytes and matches the complete remote inventory. All26frozen input pins remain exact. Its original source, freeze, logs and diagnostics are unchanged.

The actual full-size adapter restored and firstGPUforward completed at40.149389386642724seconds. Full and selected supervised losses were identical at4.774877548217773. However,100of112adapter tensors failed the fixed elementwise gradient tolerance, with largest absolute difference0.023575246334075928. The saved comparison contains every tensor's diagnostic. The worker correctly stopped before DPO comparison, all optimizer updates, longer-context training, new-checkpoint reload and repeated generation. No environment responses or scientific worlds weregenerated. This is failed numerical qualification, not an LLM intervention-utility result.

Workerfailure measured41.53631062572822elapsed/8.222815741processCPU seconds. Supervisor43.0998718040064elapsed/.026351455processCPU seconds. Two independent ownedNVIDIA samples peaked5970MiB. Ownedprocessgroup and GPU cleanup passed. These measurements overlap with scheduler accounting and are not additive. Known currentcycle allocation2482CPU-seconds/344GPU-slice-seconds, conservative reservations37200/57600CPU-seconds and3600/7200GPU-seconds. Local work is not fullymetered.

A versioned PyTorch2.9.1 source inspection identified a missing control: evaluation mode does not activate strict deterministic FlashAttention backward. This is a plausible explanation that remains unproven. Separate independent design review found0required findings for a fresh eight-backward, zero-update diagnostic crossing default/strictdeterminism with repeatedfull/selectedpaths and captured projection-boundary gradients. Keep CUBLASconfiguration,weights,inputs,RNG andFlashbackend fixed, preserveallgradients and originaltolerances. StrictmodeaffectsmorethanFlash, so it cannotbyitselfuniquelyidentifythecause. Proposedone-slice5min allocation would need anew reviewedsource/resourcefreeze, not an automaticretry. No newallocation hasbeenmade.

Bothcontrollerlocks were released after completecustody. A final liveuserqueuequery wasempty. Scientifictraining/development/confirmation namespaces remainungenerated. Theactivegoal remainsunanswered, but thisturn made concrete full-size runtime/numerical-diagnosticprogress. Next implement the bounded discriminating diagnostic, then qualify the chosen computation recipe and proceed to the independentlyfrozen scientificstudy. Do not loosen thresholds, relabelGPU05, repeatacceptedstudies or spendnewGPUtime without usingthisnewdiagnostic evidence.

Independent saved-artifact review found0required corrections and confirmed all60files/26inputpins,112tensor diagnostics/100failures, zero updates, cleanup and exact parent accounting. Final failed-attempt dispositionSHA256 `bfc56119af8834d36fadc4f2fda3ce0048af41ee12cc819cb6672cd15e71a5bb`. This accepts custody and reconciliation only, not the failed qualification. Next scientific initialization must exclude all artificial qualification adapters.


### Discriminating full-size gradient diagnostic submitted once

Job35209536 was submitted once at2026-11-02T23:28:35Z. Sourceff1a3ab79fd9541d5c9ce4a17f0a7e989b4435b0, packetb9f1052cb0936ff740aa988af6570a88e5d2d860, freeze0ff56c88449d9ae053b97bb83a6bc7799369294358d69f2700cfb70294daa6c3. All25remoteinput objects matched. Bothcontrollerlocks held,18priorjournals/eightterminaljobIDs reconciled, emptylivequeue, scratchstat/df/dedicatedwrite-read-cleanup passed. Distinct implementation and frozenpacket reviews0required. Five new fabricated methods passed. No old qualification repeated.

The new diagnostic records all112adaptergradients and outputprojection inputs/inputgradients for eight backwards. It crosses ordinary/strict deterministic algorithms with full/selected actionprojection, two repeats each, fixedweights/RNG/cuBLASconfiguration/Flashbackend and zero optimizerupdates/responses. Captures persist before comparisons, using original tolerances. Numerical discrepancies are diagnostic results, not concealed failures. Completeness is not scientific qualification.

RequestoneA10020GBslice,4CPU with at most8allocated,12GiB and5minutes, accountucb736_asc1/aa100/gpu-testing/no-requeue. Startup180/child250/supervisor280seconds and ownedcleanup remain enforced. Conservative reservations39600/57600CPU-seconds and3900/7200GPU-seconds. Known priorallocation2482CPU/344GPU-slice seconds excludes this job. Localpreparation remains partlyunmetered. Scientific namespaces remainungenerated.

External ACE-LLM-intervention-gradient-gpu-20261102-06 and repository results/ace_llm_intervention_task_runtime_20261102/gpu06 preserve the packet and once-onlyjournals. Do not duplicate or blind-retry.


### GPU06 completed and isolated two numerical effects

Job35209536completed0:0. Complete77-file63386825-byte custody and25inputpins independentlyreconstructed. Defaultsamepathrepeats failed88/112full and82/112selectedadaptergradients. Strictsamepathrepeats arebitwiseidentical. Strictfull/selected stillfails101/112 withmaxabsolute.026554226875305176. Alllossesmatch, alignedhiddeninputs areidentical, dHiddenmax3.814697265625e-6. Theseobservationsseparaterepeatvariationfromstablecrosspathdifferences, withoutuniquekernelattribution. Zerooptimizerupdates/responses. Fullscientificruntime remainsunqualified.

Actualallocation260CPU-seconds/65GPU-slice-seconds. Knowncycletotals2742CPU/409GPU-slice seconds. Conservative reservations39600/57600CPU and3900/7200GPU-seconds. Worker12.510996682CPU/59.81611355487257elapsedseconds, supervisor.070712429CPU/61.563073116354644elapsed. TwelveownedNVIDIA samplespeak6770MiB,8backwards/positiveCUDAevents andframeworkmemorysupportGPUwork. Ownedcleanupcomplete. Scopesoverlap,localpreparationpartlyunmetered.

Independentreview0required. Results andfutureprecisioncorrectiondispositions preserved in reviews/ace_llm_intervention_gpu06_results_review_2026-11-02.md. NextcandidateusesstrictdeterminismandseparateFP32outputprojectionpreservingtheBF16tiedbase. Itneedsfreshreview/freeze, unchangedSFT/DPOgradientgates,longcontext/reload/resourcequalification. GPU05remainsfailed. Scientifictraining/development/confirmation remainungenerated.


### Replicate-aware training feedback implemented

The new pureaggregator llm_intervention_replicated_feedback.py binds thepublictask/queryorder/weights/scale, averagesquerylossesinsideeachreplicate, then estimatespairwiseSEacrossreplicateblocks. Itkeepsallcandidatepairs,negativegains,ties andrawlosses. Nine newartificialmethods passfrom/tmp, parentintegrationreview0required. IndependenthandoracleSE.5differsfromincorrectpooledquerySE.6708203932499369. No scientificworld,response,fit,model orallocationoccurred. Provenanceflagsremainfalse. Chargedexecution,independentstreams andsharedstate stillneedfuture runnerintegration. Source/tests/review/receipt preserved under results/ace_llm_intervention_replicated_feedback_preparation_20261102/preparation01 withexclusiveexternal originals. Exacttestinterpreterversion/CPUunrecorded, elapsed.005s, preparatoryread-onlydiscoverymissesdisclosed. Resourceallocationtotalsunchanged.


### Strict FP32 projection qualification submitted once

Job35209701 submittedonce at2026-11-02T23:42:37.056532+00:00. Source08311835b45d520143c071c14ab4b68fa1563f9b, packetd00e7b6aa9bdcc84ad4486594de689ecd3a78642, freeze61e4b5c48f8d6dbbb06abcdfa1645cbbac0c4ffa20e50854aa635006265daac4. Bothsharedcontrollerlocksheld,20priorjournals/nineterminaljobIDsreconciled, queueempty, scratchstat/df/write-read-cleanuppassed. All30remoteinputpinsverified. Independentimplementationreviewclosedoneevidence-persistenceissue, finalrecheckanddistinctpacketreview0requiredremaining.

Newrecipeusesstrictdeterminism,FP32outputprojectionindependentoforiginalBF16tiedbase, highestFP32/noTF32/CUBLASconfiguration. All8SFT/DPOcapturecomparisonsmustpersistbeforethenumericalgate. Withinpathrepeatsmustbeexact, full/selectedloss/gradientthresholdsunchanged. Onlyafterpassing,3SFT/3DPOartificialupdatesat4341/7343tokens, frozenreferenceverification,freshbasereload andfourlegalgenerations. Noenvironmentcommandexecuted,noscientificworlds. Failurepreservedwithoutthresholdrelaxation.

RequestoneA10020GBslice,4CPU≤8allocated12GiB10minutes/accountucb736_asc1/aa100/gpu-testing/no-requeue. Startup180/child540/supervisor570seconds. Conservative reservation44400/57600CPU and4500/7200GPU-seconds. Knownpriorallocation2742CPU/409GPU-slicesecondsexcludesnewjob. Localpreparationpartlyunmetered. Sevennewartificialpreparationmethodsacrossfourcommands passed, nooldtestsrepeated. Fullruntime/sciencequalificationpending. External ACE-LLM-intervention-precision-gpu-20261102-07 and repo results/ace_llm_intervention_task_runtime_20261102/gpu07 retainexclusivepacketandjournals.


### GPU07 precision candidate failed before optimization

Job35209701FAILED1:0,37seconds4CPU=148CPU-seconds and37GPU-slice-seconds. Complete98-file63624463-byte custody, all30inputpinsunchanged. Independentreviewreconstructedall8captures/boundarycomparisons,0requiredcustody/reportingcorrections. Samepathrepeatsbitwiseidentical. CrossSFT97/112gradientfailures,max.05474281311035156; crossDPO105/112,max.014989033341407776. SmallnonzerodHidden differencesremainwithidenticalhiddeninputs. All8comparisonsretained. Nooptimizerupdates,longcontexttraining,reload,generation,environmentresponses orscientificworlds. FP32projectionalonefailed,notreclassified.

Knownallocation2890CPU/446GPU-sliceseconds, conservative44400/57600CPU and4500/7200GPUreservations. Worker11.2206079CPU/28.366372967604548elapsed; supervisor.029181548CPU/30.869004738982767elapsed. SevenownedNVIDIAobservationspeak6976MiB, actualGPUforwards/backwards, ownedcleanupcomplete. Scopeoverlap/localunmeteredpreparationremain.

Nextcandidateusesfixedsingletonprojectionrowsandidenticalaction-onlyreductioninfull/selectedpaths, withnewindependentarithmeticoracles andpreservedcomparisontelemetry. Thisisnewnumericalrecipe, notproofGPU07hadoneidentifiedcause. Noautomaticretry/allocationfollows. Futureproductionrequiresqualificationandfreshscientificsource/resourcefreeze.


### Charged replicated branch integration complete

Newllm_intervention_replicated_branches.py connectsstrictpaidbranch/scoringjournals tothereplicateaggregator. Samepreactionforkforeachcandidate/replicate, exactrequestmappings,taskweights/scale, sharedpaidscoringwithoutlearnerupdate andbaselineimmutabilityareenforced. Reservationsprecederesponsesandstaychargedthroughunknown/failedreturns. Allplannedcosts/responsesmustcompleteforaterminalreceipt. No fallback,filteredcase orautomaticretry. Callbacknoise/honesty/world/streamindependence stillneedfuturefrozenrunner.

Nineartificialintegrationmethodspassoncefrom/tmp,Python3.9.6,.059unittestseconds,nonefailed. Parentintegrationreview0required. Completefixture/callbackfailurecustodyretainedin results/ace_llm_intervention_replicated_branches_preparation_20261102/preparation01 andexclusiveexternaloriginals. Noscientificgeneration,model,GPUorallocation. PreparationCPUunmetered. Nochange tocycleallocationtotals.


### Scientific runner staging requirement before the first screen

The current ProcessPolicy createsachildperdecision. LoadingtheLLMforeveryinterventionwouldrepeatedlypaytheobservedstartupcost andleaveGPUresourcesidle. Beforeproduction, implementapersistentpolicyprocess thatloadsonepinnedpolicycheckpoint andservesboundedpublic-taskrequests. ThetrustedCPUepisodeprocessownstheenvironmentandchargedledger. TheLLMprocessreceivesonlythevalidatedpublicenvelope, monotonicallyidentifiedrequests andlegalmenu, thenreturnstheexactcompletionandmatchingrequestidentity. Bindmodel/source/adapter/recipepinsatthehandshake, enforceper-requesttimeoutsandoutputbounds, preserveallfailureswithoutfallbackorretry. Thisisprocessseparation,notamalicioussame-userOSsandbox.

Performtraininghistoryconstruction,paidbranch/scoringaggregation,numericalrankertrainingandtokenizationonCPUbeforeGPUallocation. Runthenumericaldevelopmentarms onCPU. LightweightCPUenvironment/SCMupdatesneededbetweenLLMdecisionsmayrunwiththeGPUserver, butmustberecordedseparatelyandmustnotcausesustainedGPUidleness. Loadthefrozen,SFTandDPOcheckpointsasdeclaredtreatments, preservingpristinebase/freshscientificadapters. Confirmreloadandrequestisolationonartificialfixturesbeforefreshscientificdata. No productionruntime orscientificgenerationisqualifiedbythisdesignnote.


### Canonical arithmetic GPU08 submitted once

Job35209904submittedonceat2026-11-02T23:56:21.198808+00:00. Source226c24f89ac971d389ef27d8b4fa6b86c8521125, packet3e6b3f75afdce23ae043349bdff66b9161ac8151, freeze64ed97bb55fe7f97bd6ac36fc868c000673b6448cc78933f15dcb8abb26247d2. Bothsharedlocksheld,22priorjournals/tenterminaljobIDsreconciled, emptylivequeue, scratchstat/df/write-read-cleanup passed. All32inputpinsmatchedremote. Independentnewrecipe/design/implementationanddistinctpacketreviewsclosed,0requiredremaining. Sixnewartificialmethodsultimatelypass, initialtwofixtureerrorsretainedandcorrectedwithoutoldtestrepeats.

FixedsingletonFP32projectionrowsandsharedaction-onlylogsoftmax/reductionpreservethemathematicalobjectivebutdefineanewnumericrecipe. Originalthresholds, all8pre-updatecaptures/comparisons andexactrepeatsrequired. Per-captureCUDAinterval/memoryandpre-gatebase/adapter/projectionhashesnowpreserved. Anyfailurestopsbeforeoptimization. Onlyafterpass sixartificialSFT/DPOupdates,longcontext,frozenreference/reload/repeatedgenerationmayqualify. Noenvironmentcommandorscientificworld. Noarbitrarybatchcompositioninvarianceclaim.

OneA10020GBslice4CPU≤8allocated12GiB10minutes, accountucb736_asc1/aa100/gpu-testing/no-requeue,180/540/570limits. Conservative reservation49200/57600CPU and5100/7200GPU seconds. Knownpriorallocation2890CPU/446GPU-slicesecondsexcludesnewjob. Localpreparationpartlyunmetered. No newscientificfreezeorproductionbudgetfollowsfromthisruntimejob. ExternalACE-LLM-intervention-canonical-gpu-20261102-08andrepogpu08retainpacket/journals.

### GPU08 passed the numerical gate and failed on long-context memory

Job35209904 ended FAILED1:0 after60seconds. Complete111-file63,953,429-byte custody and32input pins match. Independent saved-artifact review found0required corrections. All eight comparison pairs have identical losses and all112adapter gradients, aligned hidden inputs and dHidden. This qualifies the narrow canonical arithmetic comparison at613tokens, not the whole runtime or a scientific result.

Exactly one SFT update completed on4,341tokens. Its peak allocated GPU memory was16,565,905,920bytes. The next7,343-token forward exhausted the19.62GiBdevice before backward. No DPO update, trained checkpoint, reload or generation completed. No environment responses or scientific worlds weregenerated. The attempt remainsfailed. Its comparison gate's zero-update count applies only to the earlier comparison stage.

Actualallocation240CPU-seconds/60GPU-slice-seconds brings knowncycle allocation to3130CPU/506GPU-slice seconds. Conservative reservations remain49200/57600CPU and5100/7200GPU-seconds. Worker24.38985518processCPU/54.9537197467871elapsed and supervisor.063278358CPU/57.44533712696284elapsed overlap with scheduler costs. Twenty-one ownedGPUprocess samples peak19952MiB, eightbackwards andoneoptimizerstep establish realGPUwork. Ownedgroup/GPUcleanupcomplete. Localpreparationpartlyunmetered,whole-sprintCPUunknown.

The next bounded candidate uses explicit non-reentrant decoder recomputation under eval-mode scoring, preserving full inputs and objectives on the same20GBslice. Fresh off/on gradient comparisons, unchanged tolerances, per-step recomputation counts, reference/reload/generation and complete custody remain required. No allocation follows from this diagnosis alone. Scientific namespaces and accepted original studies remain unchanged.

### Explicit decoder recomputation GPU09 submitted once

Job35210147 was submitted once at2026-11-03T00:14:36.992891Z. Source25f3d89e8bc7a2e114f356b75e08429fa329efca, packete09a30fba1e3f1d82a2a6668f8301527322fa115, freeze5a4289812cbd3181dee1b36e372cd1f3fbed4456eceaffa4adbaa9de1f0b603f. All34inputpins match remote. Bothcontrollerlocks held,24priorjournals/11terminaljobIDs reconciled, queueempty, scratchstat/df/dedicatedwrite-read-cleanuppassed. Implementationreview caught one source-membership omission, fixed beforefreeze. Narrowrecheck anddistinctpacketreview0requiredremaining. Three newartificialmethods passed once, including finite-differencegradientoracle, eval/frozeninputrecomputation andstate/cacheguards. Preparationcommand.873461childCPU/1.183505083elapsedseconds/262750208peakRSS excludesotherdevelopment/review. Initialsource/test andpre-fixreviewsnapshots retained.

This candidate preserves fullinputs andobjectives while explicitly recomputing28decoderlayers duringbackward. Eight new canonicalselected checkpointoff/onSFT/DPO comparisons precedeoptimization, atunchangedtolerances. Rawcaptures include actualmode/recomputationcounts. Eachofthesixoriginal4341/7343-tokenupdates must show28or morelayerrecomputationsbeforeoptimizerstep. Frozenreference, freshbasereload, fourlegal/repeatedgenerations, cleanup andexternalacceptance remainrequired. GPU08 staysfailed despiteitsnarrowcanonicalgate. No scientificworld orenvironmentresponse occurs.

OneA10020GBslice,4CPU≤8allocated,12GiB,600seconds/accountucb736_asc1/aa100/gpu-testing/no-requeue. Limits180startup/540worker/570supervisor. Conservative reservation54000/57600CPU-seconds and5700/7200GPU-seconds. Knownpriorallocation3130CPU/506GPU-slice seconds excludes09. Actualmemorysavings remainunqualified. No automaticproductionallocation. ExternalACE-LLM-intervention-checkpoint-gpu-20261102-09 and repogpu09 preserveexclusivepacket/journals. Neverduplicate orblind-retry.

### GPU09 accepted as a bounded artificial runtime qualification

Job 35210147 completed with exit 0:0. All 140 files, 53,487,613 bytes and 34 input pins have complete exclusive local custody. The independent review found no required corrections. All eight checkpoint-off/on comparisons have identical losses and all 112 adapter gradients. The earlier GPU08 full/selected comparison remains separate evidence.

The original six updates completed, three SFT and three DPO, including the 7,343-token stress contexts. Each backward recorded 28 decoder recomputations. Frozen base, projection and reference checks passed. A trained adapter reloaded into a fresh base with identical scores. Four legal commands were generated, with identical results for the repeated state. All four chose a00. This verifies execution and repeatability, not useful action selection. No command was executed against an environment, and no scientific world was generated. Artificial adapters must never initialize scientific training.

Peak GPU allocation was 4,016,784,384 bytes and peak reservation was 4,844,421,120 bytes. Twenty-eight owned-process GPU samples, eight comparison backwards and six update intervals support real GPU computation. The scheduler allocated four CPUs and one 20 GB slice for 126 seconds, or 504 CPU-seconds and 126 GPU-slice-seconds. Known cycle allocation is now 3,634 CPU-seconds and 632 GPU-slice-seconds. Historical cumulative requested caps remain 54,000 CPU-seconds and 5,700 GPU-seconds, including preparation allowance where previously recorded. These categories overlap and are not a whole-sprint process-CPU total.

Input authentication took 41.221 seconds inside the GPU allocation and the first forward occurred at 54.779 seconds. This is material preparation overhead. Before production, keep avoidable dataset construction, tokenization, verification and reporting on CPU, retain authenticated loading, and use a persistent policy process to amortize loading across decisions. The short qualification does not establish production throughput or utilization percentage.

Complete cleanup passed. A fresh queue check was empty and both controller locks were released after custody. The exact pinned artificial training/reload recipe is now accepted. Original-reference restoration into a reloaded model, resumed optimization, a persistent real-model server, scientific training and intervention-efficiency results remain unqualified. The next scientific stage still requires its own reviewed data/source/model/runtime/resource freeze. No new allocation is implied.

### Closed-attempt accounting and remaining execution envelope

A fresh final scheduler query reconciles all 12 allocated jobs in the 13 recorded submission attempts. Seven completed and five failed. One earlier submission was definitively rejected before allocation. All failures remain included. There were no outstanding jobs in the final snapshot.

Actual scheduler allocation totals 3,634 CPU-seconds, including GPU-host CPUs, and 632 GPU-slice-seconds. Adding the cumulative 3,600 CPU-second preparation allowance once gives 7,234 accounted CPU-seconds. This is allocation plus an allowance, not measured whole-sprint process CPU. The historical conservative reservation ledger of 54,000 CPU-seconds and 5,700 GPU-seconds remains preserved. Its CPU value includes the preparation allowance. The rejected attempt's reservation was released before its successor and is not an extra charge in that ledger.

Future admissions use closed actual allocations plus every outstanding maximum reservation, including the candidate, plus the cumulative CPU preparation allowance. Outstanding includes queued, running, ambiguous and unreconciled submissions. The original ceilings remain 57,600 CPU-seconds and 7,200 GPU-seconds. At this closed snapshot, headroom is 50,366 CPU-seconds and 6,568 GPU-slice-seconds. Unused terminal reservations are released, not counted as consumed compute. This does not raise either ceiling or authorize allocating all remaining headroom.

The independent full-cycle review required one historical-label clarification, now corrected in a new receipt with zero required findings remaining. Both earlier and corrected receipts are preserved under results/ace_llm_intervention_cycle_accounting_20261102. Reacquire both controller locks and refresh journals, queue and accounting before any future admission. Keep the preparation allowance cumulative and enlarge its charge within the same ceiling if evidence requires. Measured demand, CPU-first preparation, source/resource review and a new freeze remain mandatory. No new scientific allocation has occurred.

### November 4 continuation: persistent transport review closed

The corrected persistent policy transport passed its independent delta review with no required findings remaining. It drains bounded diagnostic output during shutdown and assigns session request IDs independently of episode-local decision indices. Five new artificial checks passed in the prior source02 run. The original twelve methods and all source01 evidence remain unchanged and were not rerun. The final review is separately bound in results/ace_llm_intervention_persistent_policy_20261102/acceptance20261104.

This qualifies the transport implementation only. The authenticated Qwen initializer, repeated real-model requests and scientific runner still require integration. No scientific world, model run or new allocation occurred in this closure. A fresh shared CURC channel reached login-ci4 on this continuation. The visible queue contained unrelated held ECP jobs, which were left untouched. Previous ACE allocations remain closed. New submissions require both controller locks and current journal/accounting reconciliation.

### November 4 authenticated model service prepared

A new inference-only service authenticates thirteen captured project sources, seven model files, optional adapter/receipt bytes and the exact declared Python/package versions before loading one policy. It serves complete public SCM states with fresh request-local generation caches, legal-command validation and bounded lifecycle deadlines. It has no optimizer or environment callback. Artificial adapters cannot enter a development session. Byte authentication does not by itself qualify scientific training provenance.

The independent implementation review found two deadline/evidence-order issues and one follow-up noncontiguous-capture bound issue. All are corrected, with zero required findings remaining. Eleven new artificial methods passed once across three preserved source snapshots. The seven initial checks were not rerun after the three focused checks or the final arithmetic oracle. These fixtures used 1.490481 child CPU seconds, excluding unmetered implementation/review overhead. No model, scientific response or allocation occurred. Evidence is under results/ace_llm_intervention_model_service_preparation_20261104.

The next distinct runtime integration is prepared separately. It must check one persistent frozen policy and one persistent saved artificial adapter across different tasks and repeated states, complete raw/accepted token lineage, independent GPU telemetry and clean termination. The implementation review does not close that gate. A separately frozen first scientific screen and matched training-data assembly remain pending.

### November 4 paid world-response adapter reviewed

The new known-DAG linear Gaussian world module has passed twelve new artificial methods, including hand-solved chain, fork and collider cases. Hard interventions replace the targeted mechanism and its noise. Returned observations contain only the requested variables. Explicit keyed namespaces separate mechanism generation, shared initialization, measurement noise, policy randomization, training scores and private evaluation. The public export contains no true mechanism parameters.

The adapter verifies a durable reservation in the existing charged executor before preparing each response. It does not create a second charge. Its training callback interface connects the same pre-action learner to replicated action branches and separately charged scores. Structural truth remains private, and it cannot serve as a free deployment measurement or stopping oracle. No scientific descriptor was generated.

Independent review found no required implementation correction. One custody description was corrected: an injected malformed test suffix was overwritten by the executor's failure record. The exact rejected input was not separately captured. The valid final journal, original prefix, injection source and rejection records remain preserved, without presenting reconstructed bytes as original evidence. The final narrow review has zero required findings. All 133 original files and the twelve-method run remain unchanged. The new test command used 0.158244 child CPU seconds and 0.235240167 elapsed seconds, excluding other preparation. No model or allocation was used. Source, reviews and successor receipts are under results/ace_llm_intervention_world_preparation_20261104.

### Persistent service GPU10 packet frozen, not yet submitted

The two-session artificial integration packet has passed a distinct prelaunch review with zero required findings. Source revision209f701dd7e32337367aa5b4def22aedc6474ab7, packet SHA e1a911fcf4d087742e1e0cc2ed87b351ea13ee966113c9b40245704a1afe5c68 and final script SHA de099f6f52eb7982394bf7d9008e8420859837b99c1d2755760c8f356fee9538 are pinned. Earlier bootstrap preparations remain preserved and unsubmitted. Four initial synthetic evidence methods and three focused supervision checks passed once. They are not real model execution.

Both controller locks are held by local PID7883. Fresh reconciliation finds the same twelve terminal jobs, 26 paired submission journals, 3,634 allocated CPU-seconds and 632 GPU-slice-seconds. The only visible queue entries are unrelated held ECP jobs. Scratch ownership, free space and a dedicated write/read/cleanup passed in a new exclusive output. The exact declared Python/package versions still match.

The proposed request is one A10020GB slice, four requested CPUs with an eight-CPU guard ceiling, 12GiB and ten minutes. This is the smallest GPU type in the live aa100 listing and accommodates the prior measured 4.844GB GPU reservation/4,900MiB process peak. The worst-case reservation adds4,800CPU-seconds and600GPU-seconds. Closed allocations plus that reservation and the cumulative3,600CPU-second preparation allowance total12,034/57,600CPU and1,232/7,200GPU seconds. These are admission bounds, not actual new consumption. Full source/input remote readback and exclusive once-only journals precede submission.

This job uses only saved public fixtures and a saved artificial adapter plus a pristine frozen base. It adds zero training updates, environment responses or scientific worlds. It must retain unaccepted tokens, confirm independent GPU activity and cleanup for each session, and reconcile scheduler completion plus complete custody. Runtime success will not establish intervention-efficiency or scientific-training benefit.

### GPU10 submitted once

Job35252306 was submitted once at2026-11-04T16:32:29.687832Z. Packet commit58a0e539592085a1b2822cefa0fa06b0a21366fb contains the independently reviewed preparation. All24 transferred input/script objects matched their pins before submission. Source209f701dd7e32337367aa5b4def22aedc6474ab7, packet e1a911fcf4d087742e1e0cc2ed87b351ea13ee966113c9b40245704a1afe5c68 and final script de099f6f52eb7982394bf7d9008e8420859837b99c1d2755760c8f356fee9538 remain unchanged. Both shared locks are held bylocal7883. The fresh submission guard again reconciled all12terminaljobs and rejected any related active job, changed allocation totals, priornewjournal or input drift.

Remoteoutput /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10 and exclusive local ACE-LLM-intervention-model-service-gpu-20261104-10 retain the once-only intent/result. No retry or duplicate is allowed while pending/running or ambiguous. The reservation remains4,800CPU/600GPU seconds, not actual consumption. Complete two-session evidence, independent activity/cleanup, scheduler completion and terminal custody are still required. This submission does not generate scientific worlds, train a policy or provide an intervention-efficiency result.


### November 4 training-data assembly accepted

The single-world CPU builder now connects paid trajectories to matched SFT choices and replicated intervention preferences. Behavior and teacher assignments, cost milestones and candidate sets are recorded before responses. Every learned family receives the same saved public SCM state and labels. The numerical representation intentionally omits the language representation's names, units and prose. All strict comparisons, ties, negative gains and costs remain available. Only paid trajectory responses update the working SCM. Training score responses stay separate and charged.

Six new artificial test methods passed once. Independent review found zero required corrections after checking the 576-file custody, fixed assignments, cost crossings, matched views, learner isolation and failure accounting. The seven saved assemblies include three completed and four deliberately failed attempts. Together they retain 105 reservations, 181 cost units, 104 known returns, 103 accepted responses and one unknown return. A failed assembly never authorizes training on its completed subset. These are artificial fixture counts, not scientific data. The sole test invocation used 0.355728 child CPU seconds and 0.484445 elapsed seconds, excluding other preparation. No previous suite was rerun.

The new acceptance receipt is 6848d4e9a0fad69524bf2b307aa1d168dd443775e1d99167cd8c6f33b2c3c208 in results/ace_llm_intervention_training_dataset_acceptance_20261104. Its independent review is f9e2a0d4cb94c71a7f950d77d7e4d1ae0f58664daf5c3c7fc7c045e7553fdd8b. The original preparation manifest d303a938ad08ec687990227f65a4077cd1671c225453f27a3e972c7b42422928 and all original failures remain unchanged. Thirteen reviewed source pins still match. The separate exclusive acceptance directory preserves the review without changing the original preparation inventory.

This closes a training-data implementation gap, not the scientific comparison. The 32-training/24-development-world protocol remains ungenerated and unfrozen. Its complete executable/source/model/runtime admission, fixed exposure and optimization schedule, CPU collection supervisor and independent reporter still need integration and review. Existing blanket authority remains in force for routine work within the original limits. Qualification and a new scientific freeze must precede collection or training. Scientific worlds, model training and intervention-efficiency results remain zero for this preparation.

### GPU10 waiting for maintenance, shared locks released

The saved live check at2026-11-04T16:33:36.853490Z shows job35252306 PENDING with ReqNodeNotAvail,_Reserved_for_maintenance, zero elapsed time and zero allocated CPUs. Its scheduling estimate is not a reliable ETA. The valid queued job remains unchanged. There was no cancellation, duplicate, requeue or partition switch. The outstanding reservation remains4,800CPU-seconds and600GPU-seconds, giving admission totals12,034/57,600CPU and1,232/7,200GPU seconds including closed actual allocations and the cumulative preparation allowance. Pending status is not GPU utilization or successful runtime qualification.

Both shared controller locks held by localPID7883 were released after the once-only journals and pending-state evidence were saved. Reacquire both and reconcile the queued job and all journals before any mutation or additional admission. Keep the shared SSH master available. On the next continuation, inspect this exact job before any new GPU work. Terminal success still requires both complete persistent sessions, repeated-token checks, actual owned GPU activity, clean termination, scheduler resources and complete exclusive local custody. If maintenance continues, advance only independent CPU work and leave the queued allocation intact.


### November 7: GPU10 failed before model loading

The fresh CURC channel reached login-ci5. Job35252306 is terminal FAILED1:0 after204seconds. The former maintenance-pending observation is historical. Complete exclusive custody contains217remote files totaling4,909,550bytes, with all24input/script pins matching. Independent failure review found no additional required custody or cleanup correction. The worker failed while importing Qwen through Transformers and SciPy, ending with an eigsh partially initialized ImportError. No model-load marker, ready handshake or generation occurred. The saved-adapter session never started. No scientific world, training update or environment response was produced.

All173GPUprocess queries succeeded with no owned rows. They do not prove a numerical utilization percentage or complete MIG observation. No framework GPU progress was recorded. The startup alarm may have contributed to partial import state, but its delivery and the underlying cause are not established. The next action is a bounded CPU-only import diagnosis, not another GPU attempt.

The failed allocation consumed816CPU-seconds and204GPU-slice seconds. All13allocated ACE jobs now total4,450CPU-seconds and836GPU-slice seconds. The cumulative3,600CPU-second preparation allowance remains separate from measured process CPU. Including it gives8,050accountedCPU-seconds against57,600. The GPU ceiling remains7,200seconds. GPU10's unused reservation is released after terminal reconciliation. SchedulerCPUusage8.718seconds andbatchRSS468,984KiB overlap with allocation accounting and are not whole-cycle costs. The live queue was empty. Bothcontrollerlocks are held by localPID95527 for this continuation.

Failure disposition and its independent review are preserved under results/ace_llm_intervention_model_service_gpu_20261104/disposition10. Existing blanket approval covers a reviewed diagnosis and bounded correction within the original ceiling. No installation, environment modification, scientific fallback or favorable replacement is authorized by this failure.


### CPU-only import diagnosis submitted once

Job35341862 was submitted once at2026-11-07T16:25:19.132284Z, source49439d67c80aa6f30f9139eed65b8c3a58464b38. It requests1CPU,3GiB,eightminutes,acpu/cpu-normal/accountucb736_asc1,no-requeue andzeroGPU. Two fresh processes compare the original model-import order with explicit SciPy imports first. Each has a210second soft alarm,220second outer wait and bounded owned-group cleanup. Full stderr/import timing,15second stack samples, phase events and16selected installed source files are retained even if imports fail. It authenticates existing frozen inputs but never instantiates weights or calls an explicit CUDA API. Dependency capability checks are not instrumented.

Independent review required two failure-custody fixes, both closed with zero required findings remaining. Four new fake-only correction methods passed once in0.076895childCPU/0.085187209elapsedseconds. Prior source/review evidence is unchanged. The final probe is a885504774e34df3245d759a57075456d46970807393b8e022d325b2a54d9fec, scriptd40242b2cc9144fb0423f9df4b9d8480cc3348ae022782a866de038af11ba266 and packet94b490df5adbcbecd311fca2ff20e810351d9d7a1ce42228669b6d48dd974c57. Allthree transferred files matched independent readback. Scratchstat/df/write-read-cleanup and fresh13-job/28-journal reconciliation passed under bothcontrollerlocks.

The480CPU-second outstanding reservation brings admission to8,530/57,600CPU-seconds including the cumulative preparation allowance. GPUactual remains 836 / 7,200slice-seconds. This is an admission bound, not consumedCPU. A16:26UTCsnapshot showedPENDING withzeroallocation. Do notduplicate. Timing differences alone cannot establish import-order causation because filesystem caches are shared and CPUthread/deadline settings differ fromGPU10. A successful diagnostic wrapper does not qualify imports or GPUruntime. Inspect each mode's actual receipts and complete terminalcustody before choosing a remedy.


### Prospective screen schedule implemented and independently reviewed

The first-screen protocol is now executable as a pure schedule builder, with32training identities and24development identities. Development crosses three known3-node graphs, uniform/heterogeneous costs and matched/shifted coefficient priors at two known-noise levels. Each development world is shared by eight policy arms. Training graph counts are12/10/10, not exactly balanced. Cost strata have different private draws and are not a paired cost-effect experiment. No private mechanism has been realized.

Eight shared initial observations precede24acquisition cost units per arm. The four nonterminal training milestones0/6/12/18 retain all eight feasible intervention choices. The final24-unit milestone provides only a budget-exhaustion stop label. Both learned families receive the same160prospective supervised states and3,584retained pair records, including ties. Four branch replicates per candidate and charged structural scores keep training feedback separate from development evaluation. The proposed public scale2 and NMSEthreshold.04 define a synthetic diagnostic, not an engineering requirement.

Six new pure methods passed once, and independent review found zero required corrections. ChildCPU2.479261seconds andwall2.610693875seconds exclude other preparation. The49-file original capsule is unchanged. New review acceptance is under results/ace_llm_intervention_screen_protocol_acceptance_20261107. Scientific namespace admission, complete runner/runtime qualification, matched optimization exposure, independent cost reporting and measured execution bounds still precede a scientific freeze. No data, model or experiment has been generated by this schedule preparation. The broader transfer/engineering/confirmation goal remains unchanged.


### Offline cost-to-accuracy reporter reviewed

The new private reporter reconstructs the prescribed SCM learner from sealed accepted receipts and compares its predictions with fixed structural queries. It reports the complete cost/error curve, inclusive first threshold crossing, final accuracy, target attainment, actual phase spending and failure status separately. Nonattainment and operational or evaluation failures receive the common acquisition cap. An early stop is not private certification. Invalid evidence is retained for an unresolved capped roster row, never converted to a zero-cost success or omitted.

Seven new analytic methods passed once. Independent review reconstructed the hand-calculated curves, checked12saved reports and21retained rejected bundles, and found zero required defects. The421-file preparation capsule remains unchanged. New acceptance is under results/ace_llm_intervention_cost_curves_acceptance_20261107. TestchildCPU0.191142seconds andwall0.236475375seconds exclude implementation/review overhead and do not reset the preparation allowance. No generator, policy model, scientific response or prior study replay ran.

The interface accepts independently pinned world/task/learner/history/terminal bytes. Parent integration must still establish original-ledger correspondence, chronology, common paired inputs and roster completeness, and keep all reports inaccessible to policies. This is the required endpoint implementation, not an empirical intervention-efficiency result. The training engine and complete frozen runner remain in progress while CPUdiagnostic35341862 waits for a scheduler slot.

### CURC running-job allowance blocks the CPU diagnosis

The November 7 16:41 UTC read-only scheduler check changes the interpretation of that wait. Job35341862 is PENDING with AssocMaxJobsLimit. The Alpine association for paco0228 under ucb736_asc1 reports MaxJobs=0, and the account has no running jobs. Slurm defines MaxJobs as the number of jobs allowed to run concurrently. This observed limit prevents the diagnostic from starting. Its cause and intended restoration are unknown. SSH is working. Patrick has been asked to contact CURC about the allowance, with no new authentication request or message sent to others by the agent.

The one queued job remains unchanged. No account or partition change, cancellation, duplicate submission or limit bypass occurred. Both shared controller locks held by localPID95527 were released after saving the queue and association evidence. Fresh reconciliation is required before future mutations. The six new live-query files are preserved exclusively in external custody, with disposition association_blocker01.json under results/ace_llm_intervention_import_diagnostic_20261107.

Closed allocation remains4,450CPU-seconds and836GPU-slice-seconds. Including the cumulative3,600CPU-second preparation allowance and this job's480CPU-second outstanding cap gives8,530/57,600CPU-seconds and836/7,200GPU-slice-seconds for admission. The pending cap is not consumed time. Local training implementation can continue, but scientific data generation, real-model training and intervention-efficiency comparisons remain unexecuted.

### Supervised and fixed-reference preference training engine accepted

The reusable supplied-model engine now prepares authenticated matched records and follows an explicit ordered training plan. It saves the supervised adapter before preference training, copies that adapter once as DPO's fixed reference, and keeps the same optimizer state across both stages. Initial, supervised and final checkpoints are mandatory. Complete labels and ties remain in the record inventory. Every update records finite loss and gradients, clipping, decoder recomputation, optimizer intent and return. Failure saves partial parameters, optimizer, reference and history without retry, rollback or favorable checkpoint selection.

The engine reuses the GPU09 arithmetic and optimizer settings, but does not copy its artificial3+3 update schedule. Both training stages require explicit caller-supplied exposure. Numerical-family training, actual prepared scientific data, authenticated model initialization, adapter export/reload, runtime supervision and full source/resource admission remain separate integration work. No artificial adapter can initialize a scientific run.

Independent review found zero required corrections within this documented scope. All254original preparation files,911,118bytes and18source pins match exclusive external custody. Six unique new tiny arithmetic controls have passing evidence across two retained invocations. The first invocation had one test assertion failure because nonfinite scores were rejected earlier than that assertion expected. Only that corrected method reran. Engine source stayed unchanged. Test-childCPU2.521239seconds andelapsed3.050262458seconds exclude other development and do not reset the cumulative preparation allowance. Scoring and recomputation counters were mocked around tiny real AdamW updates. No Qwen model, GPU, scientific world or environment response ran.

Source SHA dccff3ada3e3c269368ffea5d6f23981575287528082e4fffdfc81f5af778651 and the independent report are bound in results/ace_llm_intervention_training_engine_acceptance_20261107. This closes the reusable training-loop implementation gap, not real-model integration or experimental utility. A bounded matched-exposure proposal is being prepared separately before any scientific training freeze.

### Matched training exposure implemented and reviewed

The new pure exposure planner proposes one supervised update per complete record, followed by at most one preference update per record with a strict comparison. A fixed hash of the predeclared record identity and unordered action commands selects that comparison. Preference direction, gain size and record hashes do not enter the key. Eligibility still depends on the training labels. Every tie and unselected comparison remains in the saved record and exposure accounting. This is deterministic selection, not random uniform sampling.

The numerical and language schedules identify the same selected examples in the same order. This matches example exposure, not model size, loss, tokens or compute. If all comparisons are tied, the planner returns an explicit untrainable two-stage status. It does not fabricate preferences or silently adopt a different training recipe. For the proposed complete 160-record screen, if 32 terminal records have no strict pairs, the rule would give 160 supervised updates and at most 128 preference updates. No such scientific records have been generated.

Seven new artificial methods passed once in 0.126284 test-child CPU seconds and 0.206635833 elapsed seconds. Ten proposals and 18 rejected fixtures remain in the 110-file, 341,561-byte preparation capsule and matching exclusive external custody. The independent implementation review found zero required fixes. A metadata clarification distinguishes engine commit d337ba3312ec021adc01e22d0446014e86a869c7 from its file SHA dccff3ada3e3c269368ffea5d6f23981575287528082e4fffdfc81f5af778651. Both were independently checked. The initial and corrected reviews are preserved without a test rerun.

Acceptance and source/review lineage are under results/ace_llm_intervention_training_exposure_acceptance_20261107. Scientific adoption remains pending complete integration and a resource freeze. Next local work is numerical training that consumes this exact schedule, then the complete paired runner and independent roster reporting. The older numerical trainer's default epoch loop does not implement this proposed matched schedule and must not be treated as equivalent.

A final read-only CURC check at 17:00:24 UTC confirmed job35341862 still PENDING with AssocMaxJobsLimit, zero allocated CPUs and zero elapsed time. The association still reports MaxJobs=0. Fresh evidence is queue_scope02.json, preserved in both custody locations. No job, account, partition, allocation or controller lock was changed by that check. The existing admission totals and external restoration request remain unchanged.


### Numerical control trained on the matched schedule

The new numerical trainer consumes the same complete records and selected ordered examples as the language-model exposure proposal. It reconstructs input pins, exposure and FP32 feature hashes before initializing a pristine CandidateRanker. The explicit prospective control uses hidden128, seed109, one-thread CPU FP32 and Adam at .001 with beta.1 for DPO. A separate artificial-only hidden4 setting supports small controls and rejects scientific training scope. One optimizer carries moments across both stages. The supervised ranker is copied once as a fixed reference. The historical default epoch loop is unchanged and is not called.

Matching examples does not make the losses or compute identical. The numerical control uses categorical action likelihood, while the language model uses command tokens. A stop-only numerical row has zero loss and gradient but remains a declared Adam step, including existing momentum. Initial/SFT/final checkpoints, update intent/return, partial failures and both exported policies are preserved. Fresh export loads must reproduce every training-state score row before completion.

Six new tiny controls passed once, including an independently written loss/update loop using the same PyTorch Adam, a legitimate zero-gradient stop row and four injected failed fits. These are new artificial models and optimizer updates, not pretrained models or scientific training. The 198-file original preparation retains all 8,847,468 bytes. Test-child cost was1.125138CPU seconds and1.521781541elapsed seconds, with266256384peakRSS bytes. It is not a whole-cycle cost.

Independent review found one P2 defect in final receipt publication after a disk fsync exception. The corrected helper fsyncs a diagnostic candidate before atomically linking the authoritative terminal. A post-link interruption requires outer reconciliation and never writes a competing final failure. Five new stdlib-only helper checks passed once, using0.05032childCPU seconds and0.090842125elapsed seconds. No training suite reran for this correction. The separate recheck has zero required findings. Original review/probe and the24-file correction capsule remain preserved. Acceptance, current source pins and both reviews are bound in results/ace_llm_intervention_ranker_training_acceptance_20261107.

This closes the numerical training implementation gap. It does not adopt the prospective recipe, qualify scientific data or the target runtime, or demonstrate lower intervention cost. Full runner integration must bind common initialization, complete paid histories, private evaluation, saved policy lineage and the declared world/policy roster before any fresh scientific freeze. In particular, existing episode journal formats are not automatically authenticated inputs to the newer cost-curve reporter. An explicit sealing adapter must preserve accepted receipts and unresolved returns without inferring a returned response from a response-adapter prepared-outcome event.

A fresh read-only query at17:06:36UTC again found CPUdiagnostic35341862 PENDING with AssocMaxJobsLimit. live04.json preserves that observation. No allocation, job mutation, duplicate submission or lock acquisition occurred. The existing restoration request and8,530CPU/836GPU-second admission totals remain unchanged. Preparation CPU remains within the cumulative allowance without resetting it.


### Complete-roster descriptive cost reporting accepted

The new pure screen reporter consumes saved cost-curve report bytes, independent expected pins and a complete predeclared world/policy roster. It checks pins before decoding, paired input consistency, initialization, curve endpoints and charges. Every planned cell remains present. Failed episodes and invalid or unresolved output receive the common within-world acquisition cap, with failure, uncertainty and attainment counts kept alongside cost summaries. Total capped endpoints add the declared initial cost. Actual initial/acquisition charges stay separate, and logical per-arm sums are not physical cache spending.

All unordered policy pairs and every predeclared stratum use their full roster. There are no hypothesis tests, ratio floors or favorable subset summaries. A missing final error prevents an all-world final-error mean. An attained prefix in a failed episode does not become a successful low-cost run. Conflicting paired initial histories mark the implicated arms unresolved rather than selecting one as authoritative.

Six new handcrafted-report methods passed once in0.122699childCPU seconds and0.131078083elapsed seconds. No upstream cost reporter, models, worlds or response generator ran. Independent review checked all38saved outputs,307fixture rows,17roster rejections and the complete531-file/3,787,132-byte custody inventory, with zero required findings. Fixture counts are not scientific observations. Source SHA4c324e9d11ce8fc77f796e22a1c4b81234219bf4d108d9922c2497e484b63c8c and review/source/receipt lineage are bound in results/ace_llm_intervention_screen_reporting_acceptance_20261107.

The full24-world/eight-policy screen remains ungenerated and unfrozen. This reporter does not authenticate upstream execution or recompute scientific errors. The next implementation is a complete paired episode and sealing path that connects the reviewed protocol, paid executor, shared initialization, policy exports, private cost curves and complete-roster reports. Scientific source/model/runtime/resource admission and real-model service qualification still precede any scientific launch. The numerical trainer and its review correction were committed and pushed as9e15e3d52c21e27d9629bf938458cada6bf793c6, independently matched to origin.

At17:28:26UTC a fresh shared-channel scheduler/account query again confirmed35341862 PENDING/AssocMaxJobsLimit, zero allocated CPUs and elapsed time, with accountMaxJobs=0. queue_scope03.json is preserved in both custody locations. The running-job allowance remains the external blocker. No duplicate, mutation or additional allocation occurred. Admission remains8,530/57,600CPU-seconds and836/7,200GPU-slice-seconds, with no preparation allowance reset.

The staged whitespace check passes excluding only two intentional raw-byte negative fixtures: the oversized all-space report and the one-byte-tampered report under test_byte_pins_before_decode_and_reordering. Their original bytes and pins are retained. No fixture or completed test was rewritten or rerun for whitespace cleanup.


### Account renewal and retained diagnostic guard failure

Patrick supplied CURC renewal INC1508105 for ucb736_asc2, valid through November 6, 2027. This is the selected account for new ACE work. The cycle ceilings remain 57,600 CPU-seconds and 7,200 GPU-slice-seconds. The other renewal accounts were not adopted. Live association evidence confirmed access and cleared the expired-account blocker. Details and the required eventual Alpine acknowledgement are in ace_llm_intervention_account_renewal_2026-11-07.md.

The parent migrated existing pending job 35341862 under both controller locks at 17:55 UTC. Its first account-transition preflight rejected a difference between the local submission intent and the enriched remote intent before mutation. The original committed submission result authenticated the enriched remote intent. Read-only reconciliation confirmed no transition had occurred. The corrected migration succeeded once, preserving the job ID, source, output and resources.

The parent missed the probe's fixed old-account runtime guard. Job 35341862 then failed 1:0 before imports, using 12 allocated CPU-seconds and no GPU. Complete custody contains 10 files and 40,872 bytes under results/ace_llm_intervention_import_diagnostic_20261107/terminal01, with matching exclusive external originals. Process CPU of 0.166 seconds and batch RSS of 24,056 KiB are separate overlapping measurements. No diagnostic worker, model load or scientific response ran. This failure remains part of the evidence.

The new launch02 changes exactly one account literal in the probe to ucb736_asc2. The script's account, name, output and source pin follow that change. Four new fabricated guard subcases passed once in 0.068828 child CPU seconds and 0.077352084 elapsed seconds. Renewed-account admission reaches only a mocked worker boundary. Expired-account, wrong-CPU and GPU cases reject. No old tests or actual diagnostic workers ran.

The prepared probe SHA is 8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951. The script SHA is 9d9f230db70214ef85283e634dbc5e99b4ac84a3a58bc866a984370c08b98ca3. The independent account/packet/launcher review found zero required corrections. Launch02 is not yet submitted. A source commit, fresh readiness and reconciliation, and exclusive once-only journals precede submission.


### Corrected CPU diagnostic submitted once

The corrected account/packet/launcher review has zero required findings. Source and packet commit 869e4cbfe4e9e3a37ed0098225723442e4faa293 was pushed and independently matched to origin before execution. Both controller locks, fourteen terminal prior jobs, thirty submission journals and renewed association access were reconciled. Scratch ownership, free space and write/read/cleanup passed. All three transferred input pins matched independent remote readback.

Launch02 was submitted once as **35343088** at 2026-11-07T18:02:20.204215Z. The packet SHA is 546ffe5d314e790339b38bc86521b92992d56e481057a70a00a07780210a9057. Remote output is /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_02. Exclusive external custody is ACE-LLM-intervention-import-diagnostic-20261107-02. No job was requeued or duplicated. Both locks were released after the submission receipt was saved.

At 18:02:51 UTC the job was RUNNING with the intended account ucb736_asc2, one CPU, one node/task, 3 GiB and no GPU. The first worker had authenticated its original frozen source/model-file hashes and declared runtime, then captured installed sources. That is progress inside a diagnostic, not model loading, successful imports or scientific qualification. Require both terminal worker results, scheduler disposition and full local custody before deciding on a correction or any new GPU attempt.


### Paired intervention execution and sealed cost reporting accepted

The new paired runner executes predeclared policies on the same supplied system, with a fresh learner for each arm. It separates arm execution from private evaluation so CPU policies do not need a GPU allocation and a language-model service can later be reused across worlds. The public policy receives its legal intervention menu, task and learner state. True mechanisms and private errors stay outside that API. This is process separation for trusted code, not a hostile-code sandbox.

Every arm pays for its initial and acquired responses. Shared initial keys produce paired values without an uncharged cache or an extra initial draw. The wrapper captures a returned response before executor acceptance. A prepared simulator outcome alone never proves return. The sealing adapter authenticates the original journal, captures, interventions, charges and disposition, then emits the exact history schema required by the private cost reporter. Missing or unsealable arms remain explicit at the common cap. Conflicting initial histories leave all arms unresolved instead of choosing a favorable witness. Persistent-service closure must precede final reporting.

The independent paired review found an admitted zero-request case that opened an invalid unused adapter. That correction passed in the seven new end-to-end methods. A hand-specified scalar system gives the expected normalized-error curve [1/4, 1/9, 1/16, 1/25] and first crossing at acquisition cost two. Other fixtures retain broken/missing policies, invalid and unknown returns, partial or conflicting initialization, altered pins and unclean service closure. These are artificial responses and SCM updates, not new scientific worlds or pretrained-model evaluation.

The independent sealer review found that terminal mean and covariance values were checked only for shape and finiteness. The correction independently reconstructs each local posterior from batch sufficient statistics and the pinned prior/noise configuration. All mean/covariance entries and eligible counts must match one permitted whole accepted prefix. It preserves atomic last-update failure alternatives and explicit unavailable failed states. The comparison uses absolute tolerance 1e-10 and relative tolerance 1e-8. It adds no numerical rescue, floor or private scoring.

Four new correction methods passed, including exact scalar/two-parameter posterior oracles and altered-parameter rejection. The original six sealer methods were not repeated. The affected integration recheck used only saved inputs: all eight finalizations and 17 roster rows remain byte-identical. All cost reports and rejection outputs are unchanged. Twelve provenance records now include the strengthened posterior check. No response, policy call, environment draw or model call occurred in that recheck.

Both independent required findings are closed with zero remaining corrections. Original and corrected evidence remain separately preserved. The four capsules contain 878, 271, 621 and 273 files respectively, with exact external custody. Their scoped test-child CPU measurements are 0.144719, 0.113045, 0.656927 and 0.231226 seconds. The recheck driver separately reports 0.330756 self CPU seconds. These are not whole-development costs and do not reset the preparation allowance.

Combined acceptance is bound by SHA38154e8f94363a6e25e909e70b54c90e1164b3361c1f5f376a665f67f3931356 in results/ace_llm_intervention_paired_episode_acceptance_20261107. Current paired source is f4061bf16743212e4ae95b459749205595116f4f60db9cadd446c532509d07c3 and sealer source is 91b7a40829157cc21d586fd22ae17eee6556d1b73da1dfbf2f4b104564f39811. Earlier review-pending dispositions remain historical evidence for their original bytes.

The scientific screen is still ungenerated and unfrozen. Next integration must provide the complete multiworld controller, policy/checkpoint lineage, training/tokenization closure and exact stream namespaces. The current numerical CLI uses a SeedSequence keyed by seed and request count, whereas the prospective protocol describes a decision-index key. Those rules must be reconciled explicitly before adoption. Actual model-service reuse, target runtime and scientific resource admission remain required. None of these component results answers the experimental-efficiency question yet.

The staged whitespace check excludes only the intentional all-space oversized executor journal in the original sealer negative fixture. Its bytes and hash remain unchanged. All other staged files pass. No fixture was rewritten to satisfy a formatting check.


### CPU import diagnostic completed with a retained failed arm

Job 35343088 completed 0:0 after 301 allocated CPU-seconds on one CPU and no GPU. This is wrapper completion, not acceptance of both import sequences. Exact custody retains all 75 remote objects, 1,958,758 bytes, under results/ace_llm_intervention_import_diagnostic_20261107/terminal02 and matching exclusive external originals. Probe, script and packet match the reviewed pins. Raw stderr, stack samples, phase records, selected installed source files, allocation capture and process dispositions are retained.

The matched original import order timed out. Its failure record reports 217.046833 elapsed seconds and 6.236775 process CPU seconds. The deadline signal arrived at 216.733115 seconds, later than the nominal 210-second alarm, while a native import was in progress. The outer 220-second deadline then killed the owned group. Cleanup records a reaped child, no surviving group and no errors. The trace is in Transformers generation imports, Torch Dynamo and Triton's native module. SciPy had not loaded in that process. This is a startup deadline failure, not a repeated demonstration of the earlier partially initialized SciPy error.

The second, SciPy-first process completed the model-class and project-helper imports. Its terminal reports 49.060399 elapsed seconds and 5.692939 process CPU seconds, with no weights loaded, no updates and no environment responses. Torch import alone took about 90.270 seconds in the first process and 2.664 seconds in the second, computed from saved phase differences. Both share the same node and filesystem cache, and their order was fixed. The successful sequence demonstrates that those imports can complete in the installed runtime. It does not prove that SciPy-first ordering caused the speed difference or qualifies the persistent GPU service.

Stack samples repeatedly show Python source reads, path metadata checks and native-library loading. The large elapsed/process-CPU difference is consistent with startup I/O or loading delays. Cache state, filesystem latency, native-loader latency and other waiting were not independently measured, so no unique cause is asserted. The next correction should address and measure startup before another GPU reservation, keeping original and failed outcomes. No install or environment mutation has occurred.

The next bounded CPU qualification should check the original import sequence after explicit preparation and measure startup phases under the intended launch layout. If relocation is needed, use a separately authenticated copy of the existing runtime and preserve package/binary hashes, original sources, failures and actual import paths. [CURC's filesystem documentation](https://curc.readthedocs.io/en/latest/compute/filesystems.html) identifies node-local SSD scratch as `$SLURM_SCRATCH`. That is a possible preparation route, not evidence of a qualified relocated runtime. Data construction and packing belong on CPU resources. Actual staging cost and both GPU activity signals must justify any later GPU attempt. A warm shared-cache success alone is insufficient.

Closed allocation now totals 4,763 CPU-seconds across fifteen jobs and 836 GPU-slice-seconds. With the unchanged cumulative preparation allowance, CPU admission is 8,363 / 57,600 and GPU admission is 836 / 7,200. Slurm TotalCPU of 14.374 seconds and batch maximum RSS of 1,073,460 KiB describe this diagnostic and overlap with other measures. They are not a whole-cycle total. No related ACE job remains in the captured terminal queue.

The separate saved-evidence review found zero required corrections. It independently verified all 75 raw objects, input pins, worker outcomes, cleanup, scheduler resources and accounting. Its SHA is 2140a4a1eee29d4fcb6fafda7036b4e45d1bb18ade4a96a07f749996c804a346. The original review-pending disposition stays unchanged. A new acceptance record under results/ace_llm_intervention_import_diagnostic_20261107/acceptance02 binds it to the completed review, with SHA8cbc09522ef91548cc39efe27deace8ade78956d7e9f94b6796098c4b8bf2dc2 and an exact exclusive external copy. The next gate is a reviewed CPU-only qualification of the original sequence with measured preparation and actual import-path evidence. No follow-up allocation or GPU retry is implied by this acceptance.

Two authenticated raw SciPy optimize source captures contain upstream trailing whitespace on line 243. The staged whitespace check excludes only those two exact captured files. Their original bytes remain pinned and unchanged. All other staged files pass. Raw evidence was not edited for formatting.


### Staged CPU runtime qualification prepared

The accepted diagnostic does not justify an import-order-only change or a GPU retry. A new CPU-only path copies the existing runtime into a byte-pinned archive, verifies and stages its contents on node-local scratch, then runs the unchanged original import probe in the copied interpreter. It records preparation separately from startup and checks actual loaded paths before accepting a return. Original environment files, packages and the model are not modified. Platform libraries remain external. No claim of an atomically frozen source tree or universal dynamic dependency closure is made.

The helper is bounded at 100,000 objects and 16 GiB of file contents. The planned job requests one CPU, 3 GiB, 30 minutes, zero GPUs and no requeue on ucb736_asc2/acpu/cpu-normal. The copied child must return within the original 180-second startup target, with a separate 185-second absolute outer deadline and bounded process-group cleanup. Full CPU packing/staging cost is unmeasured and belongs to this allocation. A later GPU node would need its own measured staging cost. The proposal would make CPU admission 10,163 / 57,600, including the unchanged cumulative preparation allowance. No GPU reservation is added.

The first six tiny archive methods had five passes and one path-alias error. The corrected roundtrip alone passed. Independent review found traversal-error suppression and a forward-link containment gap. Both are fixed, with two new focused regressions and zero required findings on recheck. Final helper SHA7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e. These three invocations used 0.084278, 0.060828 and 0.055399 child CPU seconds.

The caller's three initial fabricated methods passed in 0.058243 child CPU seconds. Review found that startup-receipt I/O could extend its wait deadline. The caller now arms an absolute phase timer before process creation and receipt I/O, subtracts elapsed time before wait and preserves the original overall deadline. Two new fabricated methods passed in 0.056215 child CPU seconds. Final caller SHA9e63e644aba5c65a5c8b61d24018a8dad2282538b2834132b36da5d7516e72c7. No real runtime, model, scientific seed or environment response was used by these local methods. Their costs exclude unmetered implementation/review and do not reset preparation admission.

Launch01 is prepared under results/ace_llm_intervention_staged_runtime_20261107. It retains the original probe SHA8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951 and GPU10 service/freeze pins. Current packet SHAe985bb75930ec7ada68a6762a0509e86f32d8797f5cafa02a79eeb7f0ef75a7c. Caller recheck and final packet review, source commit, current readiness/journal reconciliation and one exclusive submission remain required at this preparation entry. No new CPU or GPU job has been submitted by this entry.


### Prospective numerical behavior keys integrated

The new llm_intervention_screen_policy.py connects all four numerical controls to the existing public TaskProcessPolicy interface. It uses the exact canonical world-key rule with the declared trajectory namespace and policy decision index. The index subtracts the validated initial-history count from completed requests, not intervention cost. This relies on the normal episode loop stopping after failed/ambiguous calls. Actual process call indices must still be reconciled by the outer controller. The old numerical CLI and seed law remain unchanged and are not substituted for this new interface. Training behavior and teacher snapshot modes explicitly reject rather than infer a key.

Nine new artificial methods passed once, including eight successful actual public subprocess calls across four controls and two decision indices, plus five expected CLI rejections. The comparison of 96 proposed development bindings realizes no scientific world. Working-SCM planning over hand-written states is local computation, not an environment response. The command used 1.296824 child CPU seconds, 1.441983875 elapsed seconds and 47,759,360 peak child RSS bytes, with a separate 0.002283 supervisor CPU seconds. These do not include implementation/review or reset the preparation allowance.

The distinct review found zero required findings and independently verified all 96 retained files and external custody. Source SHA3d34247e476117e6521528a889780948598117462b7aec53b7179e8f515bf67f, preparation manifest953d51d6a50f36fab1c0e526a391191388450f187ebd1fcb5b292d758a11d5ba, review9cd6a8d11034dbeb45a86c96cb0dcc91355043cc0818e46a1ba798b60a05ac47. Current reporting is separately bound under results/ace_llm_intervention_screen_policy_acceptance_20261107. All original preparation bytes remain unchanged. This closes the development numerical-key mismatch, not the multiworld/controller/training/model/runtime/scientific admission gates.

The caller correction recheck and final launch01 packet review now have zero required findings. Exact sources, packet, monitor, once-only launcher and all five original/recheck review documents are preserved in the exclusive launch custody. No new tests were repeated. All seven initial packet files and review provenance must be committed before the one submission. The prelaunch_review.json retains this unsubmitted state. Fresh scheduler/journal/readiness and remote pin gates remain enforced by the launcher.

The staged whitespace check excludes only the exact intentionally oversized all-space stdin fixture under screen_policy_preparation/fixtures/test_cli_rejects_bad_payload_and_pin_without_fallback/oversize/stdin.txt. Its original bytes remain unchanged. All other staged files pass.


### Staged-runtime CPU job submitted once

Job **35343957** was submitted once at 2026-11-07T18:36:17.244109Z from pushed source/packet commit5967034358672db1eac9932352728e69a87ea045, independently matched to origin. All seven original packet files, the monitor and the review provenance are committed. Both controller locks, fifteen prior closed jobs, thirty-two submission journals and renewed association access were reconciled. Scratch ownership/stat/df and dedicated write/read/cleanup passed. All five transferred input pins matched independent remote readback before the one sbatch call. Remote submission intent SHAdfe1b1d5c9e01cfa9969eca261016a5053135e538bc3aa2606d289653cf9fcfa.

Remote root is /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01. Exclusive local custody is ACE-LLM-intervention-staged-runtime-20261107-01. At18:36:36UTC the scheduler reports PENDING/Priority with the intended renewed account, requested1CPU/3GiB/30minutes and no GPU. Zero elapsed allocation was recorded. This is a valid queued job, not a failed attempt. Do not duplicate, requeue, replace or infer an ETA. The two controller locks are released. Actual resource admission and startup qualification remain pending.

The new proposal reserves 1,800 CPU-seconds, so current CPU admission is10,163/57,600 including the unchanged3,600 preparation allowance. Known closed allocation remains4,763CPU-seconds and836GPU-slice-seconds. No new science/LLM outcome is available. Full source/output/phase/log/receipt custody and exact accounting must precede acceptance. If the runtime archive is large, retain its complete bytes in exclusive external local custody and bind its digest and location in the compact repository record. Do not put the runtime archive into Git or claim a compact mirror contains the binary.

At18:38:32UTC,35343957 is RUNNING with exact1CPU/3GiB/noGPU resources. The runtime inventory has recorded17,000objects and about7.95GB of declared file sizes in its latest retained phase. These are in-progress metadata, not a completed archive or final byte inventory. No copied interpreter or model has run at that snapshot. The existing job remains valid.
