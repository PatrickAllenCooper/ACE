**Accept GPU01 job35377234 as preserved failure custody only. No accounting or custody defects found.** Training, checkpoint inference, runtime qualification and scientific success remain unaccepted.

1. **Complete custody authenticated.** Independently hashed all **307 raw files / 66,963,772 bytes**: exact membership, sizes and SHA-256 values match both the fresh remote inventory and [disposition](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal01/disposition.json). No symlinks or mismatches. All **311 terminal-package files** match the external local backup and commit **`89098acf210e6682840463ebb6b95426e61a7503`**. This establishes preservation against the saved remote snapshot; no current remote state was queried.

2. **Source/input bindings reconcile.** All **48 transfer pins**, **45 inner inputs**, **38 inherited CPU inputs**, four record/receipt pairs and **2,257 stored tokens** agree. The CPU preparation, acceptance and prepared artifact also match their committed predecessor originals. The 28 declared research sources plus archive helper match source commit **`74be8f57a5d4010f99296066e0ad6ab38b9671b3`**; 53 committed launch files match packet commit **`da4200b0ab2eb6107e37761b3e208b806ef439b1`**. Saved model/runtime receipts agree with their expected pins; the original bulk runtime and base-model bytes were not independently rehashed here.

3. **Actual resources and costs reconcile.** Saved `sacct` and terminal `scontrol` agree: **FAILED, exit `1:0`, 77 seconds**, four CPUs, 12 GiB, one `a100_3g.20gb`, account/partition/QOS `ucb736_asc2/aa100/gpu-testing`, 600-second limit, zero restarts. Thus **308 allocated CPU-seconds + 77 GPU-slice-seconds**, counting the parent allocation once. Reported process CPU is separately **55.739 seconds**; batch MaxRSS is **11,510,376 KiB ≈ 10.98 GiB**, about 91.5% of requested memory.

   The 28 predecessor rows independently total **8,186 CPU / 1,091 GPU seconds**. Updated closed totals are **8,494 CPU / 1,168 GPU seconds**. Including the unchanged 3,600-second CPU allowance gives admission **12,094/57,600 CPU** and **1,168/7,200 GPU**. These are resource-seconds, not monetary charges.

4. **Real GPU work is evidenced; utilization percentage is unknown.** All 32 process queries returned successfully; **19/32** contained owned PID `3830102`, with **1,112–3,406 MiB** reported memory. First verified work occurred at **52.016661833 seconds**, within the 240-second gate. Independent framework evidence includes a **298.786 ms** first-forward CUDA interval and positive intervals for all six updates. Retained memory peaks reach **2.279 GiB allocated / 3.053 GiB reserved**; these are observed receipt maxima, not a completed-run peak report.

   Runtime staging took **34.220 seconds**; initial model loading took **1.554 seconds**. Engine execution reports **13.281 seconds for six updates**, approximately **0.452 updates/second**. This short failed run supports neither sustained efficiency nor an idleness conclusion.

5. **Six updates are preserved partial progress.** Four SFT and two DPO steps have consistent start, loss, gradient, optimizer-attempt, returned-update and completion records. Each records 112 present/finite adapter gradients and 28 checkpoint recomputations. All **10 `.pt` files** match their byte pins, including checkpoint aliases and the fixed reference; none was deserialized. The engine’s `complete` receipt covers its six-step scope. Pre-reload scoring, fresh checkpoint reloads and generation never completed; corresponding success receipts are absent.

6. **Failure location is direct evidence; padding causality is a supported inference.** The worker records `final_scores_before_disposal` at **05:43:47 UTC**. The traceback reaches canonical scoring → Qwen3 SDPA and ends with `RuntimeError: No available kernel. Aborting execution.` [Stderr line 10](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal01/remote/qualification/session/stderr.log:10) explicitly says **“Flash Attention does not support non-null attn_mask.”** Saved arithmetic confirms Flash enabled and math, efficient and cuDNN alternatives disabled.

   Frozen inputs and source explain a likely trigger: the first all-command batch combines completion lengths **7,7,7,6** with a 517-token prompt. `pack_actions` therefore constructs a **4×524** batch with one padded position in the stop-command row, and canonical scoring passes its attention mask. That derivation is exact from saved inputs/source. Attribution of the internal rejected mask specifically to that padding remains an inference: the failing record and transformed mask tensor were not captured. No remedy or alternative kernel was tested.

7. **Cleanup supports bounded termination, with a precise limitation.** [Cleanup](/Users/pat/code/ACE/results/ace_llm_intervention_training_gpu_20261108/terminal01/remote/qualification/cleanup.json) records leader exit **1**, group absent, no TERM/KILL and no errors. The parent’s “natural exit and clean owned group required” message reflects rejection of the nonzero exit, not evidence of failed cleanup. There is **no post-exit GPU sample**: source ordering raises before that check. Therefore independent post-exit GPU clearance is unverified, although owned-group disappearance and scheduler termination are recorded.

Exact SHA-256 anchors:

| Artifact | SHA-256 |
|---|---|
| Outer packet | `6b54c6e72b6dc9902c840b60dc41701afea1ecd613ccf716b23630c4d535eec5` |
| Inner training packet | `ccf9633f5874f869f03fe2cfc7de7ec6ca8f104c400d96c74409ef572088dee7` |
| Disposition | `b0196e321ae860b7140e0354a7a9e2da2ec4cf06a456a4782f56c9655683013c` |
| Inventory transport | `fee044c468205077f5354e234a8dc174b465f27d2795c739ce4b16e54a34bf0e` |

Review used local text/JSON reads, byte hashing and Git-object comparisons only. No numerical libraries, models, tensors or saved validators were loaded or executed; no remote calls, reruns or file edits occurred.
