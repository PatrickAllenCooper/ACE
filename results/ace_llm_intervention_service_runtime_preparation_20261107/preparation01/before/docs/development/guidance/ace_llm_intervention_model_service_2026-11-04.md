# Persistent LLM inference for intervention episodes

The new service loads one pinned Qwen policy and serves successive public SCM states. The CPU episode runner retains the simulator, hidden coefficients, paid-response ledger and private evaluator. The policy receives the objective, legal intervention menu, observed history and current SCM estimates. It returns one validated intervention or stop command. A new request constructs its own prompt and generation cache.

This is the next connection between the qualified model recipe and an intervention study. It is not a scientific result. GPU09 qualified artificial training, reload and four individual generations. It did not qualify this service, a full episode or lower experimental cost.

## Input and execution contract

`llm_intervention_model_service.py` accepts an independently pinned manifest. Before importing project modules it checks thirteen source files, the seven model files, any adapter and its associated receipt, Python 3.10.19 and the eight declared package versions. It executes captured project source bytes through a dedicated import loader. Cached project modules, unlisted project imports, duplicate JSON keys, nonfinite constants, missing files and changed hashes fail closed. The entrypoint must be the frozen copy beside the manifest.

The handshake binds source and model inventories, adapter identity and the exact numerical recipe. Authentication binds receipt bytes but does not establish that a training campaign succeeded. A separate launch review must validate training provenance. Artificial adapters are prohibited in development sessions. The frozen treatment loads no adapter. There is no optimizer or environment callback in this service.

Inference retains the GPU09 BF16 base, FP32 adapters and fixed-row FP32 projection. It uses strict deterministic arithmetic, FlashAttention-only SDPA, a fresh greedy generation configuration, complete prompts and an 8,192-token bound. Inputs are never truncated. All model parameters are frozen. Before and after each request, tensor identities, storage, versions, shapes, devices and dtypes must match. These checks detect ordinary in-place changes, not malicious writes that bypass PyTorch version counters.

Each request has an ordinal and envelope digest. Returned tokens are captured as unaccepted evidence before grammar or model-state checks. Oversized returns retain at most 16,384 logical elements, including for noncontiguous tensors. A call that fails before returning tokens records that fact. Accepted generations separately bind complete tokens, prompt identity, a CUDA event interval and peak memory before returning a command. The event interval includes host gaps and capture overhead. It is not a GPU utilization percentage. Client receipts bind the ordinal sequence to wire request IDs and episode indices.

Initialization, inference, idle waiting and total session duration have explicit deadlines. The authenticated startup/session limits take effect before the remaining input reads. Exceptions leave a failure receipt and end the process. External supervision must still confirm owned-process cleanup, scheduler resources and GPU utilization using independent signals.

## Qualification and limits

Seven initial local artificial methods check file authentication, corruption, scope restrictions, captured-source execution, cached imports, symlinks, version mismatches and exclusive journals. Review corrections add three focused deadline/raw-output methods and one bounded noncontiguous-tensor oracle. These new methods were run once each. They do not load a language model. Existing transport and GPU09 tests are not repeated. Every preceding source and review snapshot remains preserved.

Before scientific use, a bounded actual-runtime integration must check a repeated state separated by a different state, several episodes and task rebinding, complete token records, request isolation, startup and inference costs, and clean shutdown. The separate integration driver prepares two four-request sessions, one frozen base and one saved artificial adapter, using only existing public fixtures. Its four synthetic saved-record tests do not execute the service. Use the smallest measured viable allocation. Keep dataset generation, tokenization and numerical work on CPUs. Persistent loading should amortize startup, but no throughput benefit has yet been measured.

The trusted launcher and filesystem are assumed not to be changed by hostile same-user processes. Model files are authenticated before library loading, while captured project sources avoid a second source-file read. This is not a same-user sandbox or installed-runtime byte inventory. Scientific training, independent evaluation, full attempt accounting and experimental-efficiency reporting remain separate requirements.
