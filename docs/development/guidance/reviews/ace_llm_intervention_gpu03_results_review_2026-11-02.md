# GPU03 actual-model failure review

Independent reviewer: Descartes. Scope: saved GPU03 evidence only. No models, tests or scientific worlds rerun.

All54files totaling146515bytes match the complete custody inventory. Archived sources match0c87bce4 and the actual token fixture matches the CPU-qualified fixture. Eight SFT and four DPO steps have finite losses, positive gradients and positive CUDA timings. Their event times sum to3929.577133milliseconds. Peak recorded Torch allocation is3301825024bytes. Twenty-four successful NVIDIA samples identify the owned worker PID, reaching3716MiB. This establishes real GPU computation, not an allocation-wide utilization percentage.

The run failed. There was one proposal and zero reservations, responses or spent units. The exact suffix was `{"action_id":"a0}`. Its first six tokens match the valid target. Token seven is92 instead of9207, then EOS151645. Only eight tokens were generated, below the32token cap. The target fixture contains valid JSON. The evidence does not identify why the model omitted the quote.

First forward completed213.909811386seconds after worker start. Failure was recorded at221.770262659seconds and supervisor completion at246.408147276seconds. Slurm recorded252seconds, fourCPUs and oneGPU slice. Actual allocation is1008CPU-seconds and252GPU-seconds. Child steps are not added. No success receipt or adapter checkpoint exists.

## Required follow-through and disposition

1. Persist verification results and trained adapter before generation. Reaching generation supports execution of the pinned base/reference/update checks, but final values and digests were not saved. This is disclosed now. Implementation in the next worker remains open.
2. State cleanup precisely. The recorded owned process group disappeared without errors. No independent post-exit NVIDIA clearance sample exists. Current reporting is limited to that scope. A future worker can add a bounded final sample.
3. Diagnose startup and shutdown before scaling. The162.81second imports/CUDA interval,35.49second load/transfer/adapter interval and roughly24second failure-to-supervisor interval combine multiple operations. Finer timing and CPU-only import qualification remain open. No larger allocation is justified by these records.
4. Treat constrained generation as a new method. A public-menu grammar must include every affordable action andstop for all LLM arms. It must preserve model logits among allowed tokens, require complete commands beforeEOS, and reject failures without repair/fallback. Pure grammar implementation is separate. Actual tokenizer, callback and length-bound qualification remain open.

The result is retained as failed interface qualification with real adapter training. It is not a learned intervention policy, an experimental-efficiency result or a reason to hide earlier failures.
