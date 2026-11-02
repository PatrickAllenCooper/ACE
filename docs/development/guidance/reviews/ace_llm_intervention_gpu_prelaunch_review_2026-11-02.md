# Actual LLM smoke prelaunch review

A distinct read-only reviewer inspected the new GPU worker and later supervisor/CPU-preparer integration. No model, GPU or scientific world was executed by the reviewer. The worker's causal action loss, post-SFT DPO reference, frozen-base hashing and generated-sequence binding were judged sound. No scientific utility conclusion follows.

Five initial required issues were corrected:

1. Normal Slurm keys include both slashes and colons. The parser and realistic positive fixture now include `ReqB:S:C:T`.
2. Absolute deadlines now include supervisor preparation, an external worker cutoff and cleanup allowance inside the overall cap.
3. Cleanup follows the owned process group even when its leader exits. Races/errors are retained. Termination signals enter failure handling.
4. Progress is monitored after the first forward, with a120second stall allowance, final phase reread, all12training steps, and successful NVIDIA command status required for independent process evidence.
5. Runtime acceptance is explicitly limited to eight declared package versions, now including tokenizers and Jinja2. Origins and CUDA build are recorded. Source/model byte authentication is separate. No whole-runtime byte-closure claim remains.

Two narrow follow-up failure-path issues were then fixed. Intent-receipt persistence is best-effort and cannot skip cleanup. An absolute870second final timer remains active through cleanup, persistence and interpreter shutdown, with owned-group SIGKILL attempted at expiry. The final static delta review finds zero required code issues remaining. Supervisor SHA256 is `17d1c344f3ae533bd467cffa20df4a166eef13edee55b8100334ceb4ddaf62ef`.

Artificial evidence is preserved in GPU preparations01–03. Initial four pure packet methods passed. The expanded second run passed seven methods but failed actual local process-group cleanup with PermissionError on macOS. That failure remains open as a platform-qualification issue, not silently discarded. A focused new I/O-failure/cleanup fixture then passed. Target-Linux preparation must pass all nine methods, including both leader-alive and leader-exited cleanup scenarios. The earlier18-method CPU suite is not being repeated.

The new CPU preparer authenticates sources/model inputs, records individual hash/import/tokenizer phases and prepares the exact shared artificial prompt without instantiating a pretrained model. The later GPU worker must compare an independently pinned CPU fixture digest/content before model loading. The reviewer finds no new required issue in that refactor. The code-review gate is clear for a CPU-only packet freeze. Actual CPU acceptance, a separate GPU freeze/resource review and measured GPU execution remain pending.

20GBGPU and12GiBhost demand are estimates. The bounded smoke is intended to measure them. Full reference/base byte equality is mandatory, while reference-score tolerance is fixed at absolute1e-4 and relative1e-5 before execution. CUDA event duration establishes completed GPU work, not utilization percentage. Missing MIG telemetry remains uncertainty. Final acceptance requires independent scheduler accounting, cleanup and complete local custody.


A final distinct actual-CPU-packet check finds zero required issues. All nine input objects and six sources match committed source `fb6fe997a89e2dc24f0efd26d24aaaa6c4ae248a`. Freeze SHA256 `62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6`, script pins, oneCPU/4GiB/10minutes/noGPU, account/QOS/output paths and no-requeue agree. Model pins/revision/path and Python/six antecedent versions match earlier CPU qualification. Added declarations are tokenizers0.22.1 and Jinja2 3.1.6. The entrypoint selects nine new tests, not the original eighteen, and instantiates no pretrained model. This is packet review, not execution acceptance.


After the targetCPU stage passed, an actual-allocation delta was reviewed separately. ReqTRES must show CPU4/12GiB while actualCPU4–8 is accepted only when NumCPUs and CPUs/Task agree with AllocTRES. This follows observed Slurm task-CPU rounding and stays within the original conservative cap. GPU/account/QOS/node/time limits remain fixed. SupervisorSHA256 `5ac036598af3d31e76357a238f449ecee589edfa93cc2d3eec57cca42d2bb332`. The reviewer finds zero required issues. Three focused pure methods pass, including changed request/inconsistent count/over-allocation negatives. Cleanup/deadline/run function ASTs match the CPU-qualified predecessor. Test relocation into a separate allocation-delta file leaves the original CPU family at nine methods.


GPU01 was rejected before allocation because the original packet paired a20GBMIG with gpu-normal. The earlier packet review checked internal agreement but missed site compatibility. The retained Slurm Error7, official CURC GRES table and live QOS metadata identify gpu-testing as the intended workflow-qualification queue for this slice. A distinct narrow correction review finds zero required issues in replacing the frozen/live QOS gate with gpu-testing. The same GPU/count/CPU/memory/time and artificial recipe remain fixed. The focused new test rejects gpu-normal in either the frozen or actual resource record. SupervisorSHA256 `b8675a12ad9924427d81eadb1a3b9ec47b8ad0395fa231e0f35e77536e0e9d6b`. A new GPU02 packet remains required. Established scientific workflows must use production resources.


## Corrected GPU02 actual packet

Descartes independently compared the corrected actual packet to rejected GPU01. Zero required defects. All13 manifest hashes and12 explicit checksum entries match, four sources match commit3a8a1f21, and only the supervisor QOS gate changes. Worker, model/runtime pins, fixture, caps and recipe are unchanged. The new exclusive02 paths and job name agree. The packet binds the QOS correction and definite submission rejection. The workload is artificial adapter/interface qualification only. Freeze SHA256 `fd041a2401d93a5044ef4c01e37da3a83d7b6fe6fa97f83fde7999fe012a8dd8`. No tests or model execution repeated in this check.
