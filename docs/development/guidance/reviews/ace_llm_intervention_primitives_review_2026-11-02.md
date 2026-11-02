# Independent review of the new ACE intervention primitives

Agent Huygens performed a bounded read-only review of the new action contract, training arithmetic and CPU qualification worker. No models, jobs or old component tests were run by the reviewer. Five required findings were corrected and narrowly rechecked. The final review reports no required fixes remaining in this scope.

1. **Rejected proposals were not durable.** The contract now records proposal identity and terminal rejection before raising. It prohibits a silent replacement action. Unhandled driver exceptions are failures rather than normal closure.
2. **Generated failures could lose the completion or escape before journaling.** The generated-token path now opens a proposal before validation, preserves token lists and decoded text, and records the failing stage. Malformed decoded text remains available even when parsing fails.
3. **Dropout could contaminate preference log ratios.** Both policy and reference likelihoods use evaluation mode. Policy gradients remain enabled. Individual module modes are restored afterwards. A new dropout fixture checks equal reference and policy scores at equal weights.
4. **Python was recorded but not enforced.** The CPU worker compares the actual interpreter version with its independently pinned freeze before workload imports.
5. **A pinned but incomplete manifest could omit required inputs.** The worker now requires the exact five source filenames, seven model filenames and six package keys, plus exact model-directory membership.

The reviewer also noted an adapter dtype limitation. Inputs now follow adapter dtype, and a new whole-model casting fixture exercises this behavior. This is not evidence for every device or precision combination.

The initial artificial run passed 15 methods. After the rejection and dropout corrections, the affected fixture family passed 18 methods from an unrelated working directory, using preserved source copies. Initial and corrected sources, outputs and receipts remain in `results/ace_llm_intervention_preparation_20261102` and exclusive external custody. The CPU worker has not yet executed at the time of this review entry.

The review confirms the causal token shift, prompt masking, chosen/rejected DPO sign, detached reference scores, zero initial adapter update and frozen base parameters by source inspection. It does not qualify a trained pretrained model, tokenizer template, GPU allocation or scientific result. Trusted runner/process separation, eligible mechanism updates, training preference provenance, post-SFT reference timing and independent evaluation remain separate obligations. Transport and scheduler submission are outside this code review.
