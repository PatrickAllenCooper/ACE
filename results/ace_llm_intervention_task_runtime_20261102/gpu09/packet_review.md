**Disposition: zero required findings in the frozen GPU09 packet.**

Reviewed November 2, 2026. This is a distinct, bounded static packet review of `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-checkpoint-gpu-20261102-09/input`, its external `input_manifest.json`, and `build_packet.py`. It verifies packaging, source and asset provenance, recipe, launch command, and resource limits. It does not repeat the implementation or GPU08 numerical-results reviews. Only this review document was written; no tests, build-script execution, model loading, SSH, allocation, or submission occurred.

**Packet identity and source closure.**

- Source commit: `25f3d89e8bc7a2e114f356b75e08429fa329efca`.
- `freeze.json` SHA-256: `5a4289812cbd3181dee1b36e372cd1f3fbed4456eceaffa4adbaa9de1f0b603f`.
- `input_manifest.json` SHA-256: `48edd53cd11ad6e198bf793400d1ae69ef7ef22a97b135028a643c4e7ec23956`.
- `source-pins.sha256` SHA-256: `55c135a3f09865435a04d8f2337e0dbc31633a2bced2ff3e411770542bc69aae`.
- External `build_packet.py` SHA-256: `0f752f9163a2b88b3b927d1bcdad14dc25c2c9db4925b6e987edde744e3f27cc`.

The input directory contains exactly 34 regular, non-symlink files: 16 sources, 14 assets, and the freeze, sbatch script, implementation review, and checksum list. Every manifest and freeze hash matches the actual bytes. The checksum list contains exactly the other 33 files. Every source is byte-identical to its `scripts/research/` Git blob at the stated commit.

Static AST inspection confirms exact equality of the worker's `SOURCES`, the supervisor's `required`, and the frozen 16-source inventory, including `llm_intervention_checkpointing.py`. All referenced local research modules are included. The worker's asset, model-file, runtime-package, and recipe contracts match the freeze.

The packaged implementation review matches the committed and current final review, including the addendum resolving the source-membership finding. Its SHA-256 is `f82bb1ad6a59f89640786b9d6d5bc922f4a244a2231251dceb9e8dba8de9e250`. The packet preserves its final reviewed implementation pins:

- `llm_intervention_checkpointing.py`: `f8bd09b1c9ed296da0913ac81aa71d2ce57e89f0a7a448582f9c4298c28dccb5`.
- `qualify_llm_intervention_checkpoint_gpu.py`: `fca1daced91b4e2c1de5e7e453800140638b776324114776ddded985a5919475`.
- `supervise_llm_intervention_checkpoint_gpu.py`: `ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec`.

**Asset lineage and recipe.**

All 13 inherited assets match GPU08's frozen pins and bytes; the sole added asset is `gpu08_failure_disposition.json`, SHA-256 `adf0e70831621792302ff5b738f05a3413ca28aa385534451b27f1f9cdf68c26`. It matches the finalized local disposition and identifies GPU08 source `226c24f89ac971d389ef27d8b4fa6b86c8521125`, freeze `64ed97bb55fe7f97bd6ac36fc868c000673b6448cc78933f15dcb8abb26247d2`, and job `35209904`. Its status remains `accepted_failed_attempt`: the narrow canonical comparison gate passed, one optimizer update completed, and full runtime qualification failed. This review authenticates that receipt without re-evaluating its numerical evidence.

All 14 assets also match the builder's declared local origins. The accepted adapter hash agrees with the complete GPU04 terminal receipt; GPU06 diagnostic acceptance authenticates its terminal hash; GPU07 retains its accepted-failure status and zero updates. No partly trained GPU08 adapter replaces the accepted GPU04 asset.

Relative to GPU08, the only recipe-value change is `activation_checkpointing: false` to `explicit_nonreentrant_decoder_v1`. Fixed-row FP32 canonical scoring, strict determinism, highest FP32 precision, original loss/gradient thresholds, BF16 base, FP32 adapter, FlashAttention, token limits, and three SFT/three DPO updates remain pinned. The worker retains no-TF32 and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The seven model-file pins, model revision/path, Python version, and eight package versions are inherited unchanged; model bytes and the live runtime were not inspected.

**Launch and resources.**

`gpu09.sbatch` SHA-256 is `5fa19450d5491ff80048433612b6c2e71a7f2d18b1cf8244eb5a74960a2c8632`. The builder's substitutions reproduce the saved script exactly. Its job name is `ace_llm_check09`; account `ucb736_asc1`, partition `aa100`, QoS `gpu-testing`, one node/task, one `a100_3g.20gb` slice, four requested CPUs, 12 GiB host memory, and ten-minute wall time agree with the freeze and allocation guard. The guard permits four through eight allocated CPUs.

The distinct remote root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09`. Working directory is its `input`; stdout/stderr use `logs/%j.out` and `logs/%j.err`; supervisor output is `gpu09`, with child output under `gpu09/worker`. The script checks all checksum-list entries before executing the pinned checkpoint supervisor with Python `-I -B`, the exact freeze hash, and supported CLI arguments. The supervisor launches the checkpoint worker. Existing-output rejection and `--no-requeue` are retained.

Startup/worker/supervisor/allocation limits remain 180/540/570/600 seconds, with child deadlines bounded by the supervisor's cleanup margin. Conservative increments are 4,800 CPU seconds and 600 GPU-slice seconds, producing 54,000/57,600 CPU and 5,700/7,200 GPU reserved seconds, consistent with the predecessor disposition. The resource basis explicitly identifies GPU08's long-context OOM and leaves checkpoint memory savings unqualified.

**Scope limit.** Static packet review is complete for these exact bytes. Fresh remote readiness, root availability, live allocation authentication, checkpoint numerical equivalence, long-context memory fit, runtime completion, and reload/generation evidence remain for the parent and actual GPU qualification. This report records no submission or live-readiness approval.
