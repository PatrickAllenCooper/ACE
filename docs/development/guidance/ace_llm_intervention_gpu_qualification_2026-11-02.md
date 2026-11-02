# First actual LLM adapter qualification

This is the next Stage 1 gate in the renewed [research agenda](ace_llm_intervention_agenda_2026-11-02.md). It tests that the existing Qwen3-0.6B checkpoint can receive state-conditioned adapter updates, preserve a frozen base and reference, and emit an action that is actually executed. It uses one fixed artificial state and arbitrary preferred action. Success would not establish experimental utility or a learned scientific strategy.

## Why these resources

The separately completed CPU job authenticates seven cached model objects, the target runtime and tokenizer. The stored BF16 tensors occupy about 1.50 GB. Rank8 q/v adapters add 1,146,880 FP32 trainable parameters. The smoke caps sequences at512tokens, with one SFT sequence or a two-sequence DPO pair. BF16 full logits for two512token sequences require about311MB, with additional FP32 conversion, log-softmax and backward storage. Attention/MLP activation memory, allocator overhead and model-loading peaks remain unmeasured. The intended20GB A100 slice leaves bounded headroom for this small fixed smoke. Actual peak memory will determine subsequent requests, and any OOM is a retained failure rather than a reason to scale blindly.

Live CURC inventory on November2 lists A10020GB slices, RTX Pro24GB slices and larger devices. The20GB slice is the smallest listed CUDA choice and has a compatible established architecture for this runtime. Its node was reserved at the bounded check. A valid pending job must remain pending without bypassing reservations or changing partitions to hide a workload problem. Availability and start time are not guaranteed.

The proposed request is one `gpu:a100_3g.20gb`, four CPUs,12GiB host memory, fifteen minutes, account `ucb736_asc1`, partition `aa100`, QOS `gpu-normal`, one task, one node and no requeue. Four CPUs satisfy the partition's3840MiB-per-CPU memory policy. The source accepts at most eight allocated CPU threads and conservatively reserves7200CPU-seconds. If submitted, total first-cycle reservation becomes12000/57600CPU-seconds including the preparation allowance, and0.25/2GPU-hours. These reservations overlap actual job use and are not added to it.

## CPU preparation and bounded GPU startup

Existing cached weights were copied and completely authenticated by the CPU job. No download, install, dataset construction or scientific simulation belongs to this GPU attempt. Tokenization of the tiny fixed artificial prompt and the necessary model load are recorded separately. A fresh byte check protects the copied model at use. This adds startup I/O but does not claim that the earlier CPU receipt authenticates future file changes.

The CPU qualification took251.824213seconds of worker elapsed time but only5.096894seconds of process CPU. It had no phase-level I/O/import timing, so the cause of that gap is unknown. The new worker records verification, imports, tokenization, loading, first forward, every training step, generation and final output. First completed GPU forward must occur within420seconds. Overall worker deadline is840seconds, with an external supervisor limit and cleanup inside the900second allocation. A startup failure triggers diagnosis, not a blind longer retry.

## Fixed artificial computation

The worker loads the pinned model offline without remote code, attaches rank8/alpha16 q/v adapters, performs eight SFT steps and four DPO steps at learning rate0.0005 and beta0.1, and greedily generates at most32newtokens. Prompt tokens are masked from the action loss. The DPO reference is frozen after SFT. Labels are fixed before execution and contain no scientific outcomes.

Required evidence includes nonzero finite adapter gradients, changed adapter weights, unchanged frozen base bytes, unchanged reference weights and behavior, lower loss on this artificial supervised example, and a valid generated intervention bound to exactly one artificial response. Failing this fixed smoke is retained. It is not repaired by selecting an easier observed result.

The supervisor authenticates live full Slurm allocation, captures independent NVIDIA compute-process telemetry for its own worker PID, and checks framework memory and completed CUDA-timed training steps. Shared whole-device utilization is not attributed to this process. Missing MIG telemetry remains uncertainty to diagnose. Source/phase/terminal receipts, stdout/stderr, scheduler accounting and exact local custody are all required. CUDA availability or exit code alone does not qualify the run.

## Gates and scope

This draft is not a submission record. Independent worker/supervisor review and pure fabricated parser/deadline/resource tests must close before a frozen packet is built. Reconcile both shared controller locks, all new-line submission journals and current scheduler work, then perform bounded scratch stat/df/write/read/cleanup and authenticate the full input packet before one submission.

The completed CPU acceptance covers its historical source bytes. The later episode/SCM corrections and this GPU worker have new source identities. Neither source review nor this artificial smoke will qualify fresh policy-training, development or confirmation data. The scientific comparison still needs nonlinear/transfer tasks, independent streams, fair learned numerical and target-risk controls, complete response costs and an experimentalist utility endpoint.


## Prelaunch review and CPU prerequisite

The initial independent review identified five required fixes: realistic Slurm key parsing, absolute deadline accounting, owned process-group cleanup and preserved failures, post-forward progress supervision/final scanning, and runtime claims broader than the checked versions. The revised implementation adds colon-bearing scheduler fields, an external absolute worker cutoff and whole-supervisor timer, group cleanup independent of leader status, signal handling and durable failure intent, a120second stall limit and final receipt scan. It now explicitly checks eight declared package versions including tokenizers and Jinja2. Imported origins and CUDA build are recorded. This is not whole-runtime byte authentication. Full base and reference parameter bytes must remain unchanged. Reference likelihoods use a predeclared absolute1e-4/relative1e-5 numerical tolerance.

The actual owned-process cleanup fixture failed on macOS with PermissionError. That failure is retained in GPU preparation02. Seven other methods passed. No GPU was submitted, and the failed local cleanup was not blindly retried. A new CPU-only target-runtime preparation will run this new nine-method family on Linux, prepare the exact shared artificial token fixture, and time source/model hashes plus torch/tokenizer import/load separately. The old18-method CPU qualification is not being repeated. GPU launch remains gated on these new results and closure of the current code review.

The first GPU preparation receipt's RSS is the cumulative child high-water mark in a preparer process that had already run episode checks. It is not a pure GPU-packet-test memory measurement. Preparation02 ran in a fresh preparer and retains its failed-test CPU/elapsed measurements. No such preparation measurement is whole-project compute.


Two narrow follow-up review findings concerned failure-receipt I/O: an intent-write error could skip cleanup, and cancelling the final alarm could leave persistence unbounded. `finish_failure` now arms the absolute final deadline before any failure I/O, treats persistence errors as evidence, guarantees group-cleanup invocation, and retains the timer through shutdown. A new injected-I/O-error fixture passes. Preparation03 preserves this delta and its source pins. The local real-group failure remains preserved and unqualified until the target CPU stage.

The shared exact fixture constructor is now used by CPU preparation and GPU use. The later GPU freeze must independently pin the accepted CPU fixture digest and its qualification receipt. CPU preparation must be complete with actual scheduler completion, all nine target methods passing and exclusive local custody. The GPU packet has not yet been created or submitted.
