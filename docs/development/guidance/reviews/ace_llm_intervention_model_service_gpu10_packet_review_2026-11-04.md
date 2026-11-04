**Disposition: zero required findings for the current GPU10 packet and final root job script.**

Reviewed November 4, 2026. Scope is `results/ace_llm_intervention_model_service_gpu_20261104/launch10` and the external `ACE-LLM-intervention-model-service-gpu-20261104-10` copy: packet/manifests, committed source identity, predecessor assets, script, preparation/admission records, and saved readiness evidence. This does not repeat the closed service/integration implementation reviews. Only this review document was written. No tests, fixture commands, models, remote commands, transfer, allocation, or submission were performed by this reviewer.

**Exact packet and source identity.**

- Source commit: `209f701dd7e32337367aa5b4def22aedc6474ab7`.
- `input/packet.json`: `e1a911fcf4d087742e1e0cc2ed87b351ea13ee966113c9b40245704a1afe5c68`.
- `input/frozen.json`: `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37`.
- `input/adapter.json`: `4bd9afe1fe24cc9e83d413856dab0383fc3518c30090ee19f1a73059ad5225d2`.
- Final root `job.sbatch`: `de099f6f52eb7982394bf7d9008e8420859837b99c1d2755760c8f356fee9538`.
- Current `launch_preparation.json`: `5f5c665213cce4dd0ea7c0c5fa7b8c13d5760faecb66d590d71b4868c6836c95`.

The input directory contains exactly 23 regular, non-symlink files: 22 pinned members plus `packet.json`. Every pin matches the actual bytes. All 15 Python sources are byte-identical to their Git blobs at the stated commit: the 13-source service closure, integration driver, and existing allocation-helper module. Both service manifests contain exactly the authenticated service-source subset and match the committed recipe/runtime contracts. Local and external input bytes and current script/preparation records agree.

The service is the final reviewed `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1`; the transport is `8ccd3ae0bd28729af8ebb758c9bd474a926d05362799f45bf0133e1e4b5e4a5d`; the driver is the final integration-reviewed `fb3e95013cb46cd53f3b44ee72a141d75616c39b762b8e62ea90d89f3a9de717`. These identities preserve the closed implementation dispositions without reopening their reviews.

**Lineage and bounded inference plan.**

`gpu09_terminal.json`, SHA-256 `3ac6b07536761593331cbff93450e944268119619e12a26e1f2be3271554c589`, is byte-identical to the retained GPU09 worker terminal. It records complete status and binds the saved adapter hash `7ace8718e319e433c807326d2d938e1df6b11e9336837582ad0be3163e153631`, which matches the packet's opaque adapter bytes. Its predecessor freeze matches GPU09's frozen input. `case0.json`, `case3.json`, and `stress.json` are unchanged GPU09 input bytes. The seven model-file pins, model path, Python version, and package versions also match GPU09; no model file was loaded or remotely rehashed here.

The manifests differ only in treatment/adapter metadata: frozen base versus saved artificial adapter, both with artificial scope. The adapter manifest binds the exact adapter and terminal pins above and declares artificial training scope. Each session permits four requests, 180-second startup, 30-second request/idle limits, and 240 seconds total. The driver runs the sessions sequentially with IDs 40–43 and the sequence case0, case3, stress, case0. Case3 and stress have identical task payloads, so the retained task binding produces the exact expected stress envelope; case0 has a distinct task. The final case0 repetition retains the intended interleaved repeat check.

The pinned plan has zero new training updates, environment responses, or scientific worlds. It uses the reviewed deterministic BF16/FP32 canonical projection and greedy-generation recipe. It introduces no world-generator execution. Actual eight-request completion, numerical repeatability, GPU activity, and cleanup remain execution results to establish.

**Final bootstrap and resources.**

The final shell command parses as the intended multiline Python bootstrap, with an explicit `if`/`ValueError` digest check before `runpy.run_path`; no `assert` remains. The command pins the exact driver and packet digests and forwards supported CLI arguments. The saved optimized-Python checks use that identical bootstrap: the correct pin prints `42`, while the wrong pin produces no stdout and raises the expected digest error. Those checks were inspected, not rerun. Their receipt hash is `3603cc271ecb8ce4362f3f2e1944e4886253b1a843627bf9c9b2ce68a66048ad`.

Both predecessor script/preparation pairs remain preserved under `bootstrap_preparation01` and `bootstrap_preparation02`. Their script hashes are respectively `c2a16f97bebce6c9ce52ef29ae1198a6797dff49ae29e4740478b7a5a3ce8028` and `abb79a4f8c134dfb6b71502ab8a32cc775d03ee196880149ae44cce82c0e1388`. They are historical evidence, not launch targets; the second's malformed escaping is explicitly recorded as discovered before execution/transfer. Input packet bytes stayed unchanged through both corrections.

The final script requests account `ucb736_asc1`, partition `aa100`, QoS `gpu-testing`, one node/task, one `a100_3g.20gb`, four CPUs, 12 GiB host memory, ten minutes, and `--no-requeue`. The allocation contract allows at most eight allocated CPUs. Preparation, packet, script, and allocation-helper expectations agree. Service/driver/final-watchdog/Slurm envelopes are 240/560/580/600 seconds. The distinct remote root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10`; logs use `logs/%j.out` and `logs/%j.err`, and driver evidence goes to `integration10` beneath that root.

The saved GPU inventory at 16:23:09 UTC lists 40/20/80 GB types, making 20 GB the smallest listed option. GPU09's retained terminal records 4,016,784,384 allocated and 4,844,421,120 reserved peak bytes; its scheduler record shows 126 seconds. This supports a bounded two-load inference qualification on the same slice. Preparation explicitly counts authentication/CPU loading as startup overhead rather than GPU utilization.

**Reconciliation and readiness scope.**

The 16:19:25 UTC reconciliation contains 12 unique completed/failed jobs in the intended account, totaling 3,634 allocated CPU-seconds and 632 GPU-slice seconds. Its 26 journal files form 13 intent/result pairs: the same 12 job IDs plus one rejected submission without a job ID. Queue entries are unrelated pending ECP work. The snapshot hash is `c875af6697bbc70aa203be2a0c73529f9aaeee42a1b418a96246f76f0f53edfc`.

Admission arithmetic is consistent: `3634 + 3600 + 8×600 = 12034 < 57600` CPU-seconds and `632 + 600 = 1232 < 7200` GPU-seconds. This uses measured costs for closed allocations, a 3,600-second preparation allowance, and the candidate's worst permitted allocation. Controller metadata consistently identifies local PID 7883; active ownership of both locks is parent-reported, not independently re-probed here.

During review, the external readiness receipt arrived at 16:25:05 UTC, hash `e1242f9bcac59d3b1d2bf5f90574ac5dd85a1a553cca93a320a75a1954ec39b0`. It identifies the intended root, reports a successful write/read/cleanup probe, available storage, no same-name accounting entry, the same unrelated queue, controller PID 7883, and `submitted: false`. This is inspected local evidence of the parent's readiness step, not a new remote check. Transfer authentication, continued lock ownership, once-only submission, and actual execution acceptance remain parent responsibilities; this report establishes the static packet disposition for the hashes above.
