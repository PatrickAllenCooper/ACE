# A legal intervention grammar for the LLM policy

Status: isolated implementation and artificial tests. Target-tokenizer and actual-model integration remain open.

## Why this change is needed

GPU qualification job35203022 completed eight supervised adapter updates and four preference updates. The model then returned `{"action_id":"a0}`. The closing quote was missing. The strict executor rejected the completion before reserving or taking a response. The run failed and remains failed.

This establishes a concrete output-format defect in that artificial run. It does not identify a general cause of policy failure or show that the LLM can choose useful interventions. The frozen tokenizer fixture contains the correct closing quote. The missing quote must not be repaired retrospectively and counted as an executed LLM action.

## The new interface

`scripts/research/llm_intervention_grammar.py` constructs the token sequences for every affordable action in the public menu, together with `{"stop":true}`. It verifies exact text-to-token-to-text round trips and appends the declared EOS token. It binds those sequences to the complete input prompt.

At each generation step, the grammar permits only tokens that extend the current suffix toward one of those commands. Model scores choose among the permitted tokens. The grammar contains no reward, preferred action, environment response or private evaluation result. The action IDs and remaining budget are already public to every policy.

The same grammar must be used for frozen, supervised and preference-trained LLM comparisons. Numerical policies receive the same legal menu. A strict completion check and the existing executor still validate the final command. There is no fallback to a numerical action and no repair of an invalid generated string.

This changes the policy representation from unconstrained generation to constrained generation. It is not evidence that the old interface worked. Greedy token choices need not maximize the probability of the whole command. This is a syntax-and-menu constraint. It restricts action IDs and serialization as well as JSON syntax, without establishing useful experimental choice.

## Scoped deterministic property

Each accepted command is a finite token sequence ending in exactly one EOS token. EOS cannot occur inside a command. At every nonterminal prefix, the allowed set contains precisely the next tokens of compatible commands. Starting from the bound prompt and choosing only allowed tokens therefore preserves membership in the set of command prefixes. The sequence must reach a complete command within the maximum command length.

Actual generation must allow at least that many new tokens, maintain the exact prompt binding and stop at EOS. Hardware errors, unavailable logits or an incorrectly integrated callback can still fail execution. The current implementation binds one unpadded prompt. It does not yet qualify variable-length batched generation or finished-row padding behavior.

## New checks and remaining work

Six pure fixture methods check that all actions and stop remain reachable, that the observed missing-quote error cannot pass, that affordability preserves stop, that prompt drift and partial outputs fail, that token round trips and duplicate commands are rejected, and that caller mutation or menu order does not select an action. These use character tokens, not Qwen or a scientific environment. Preparation01 preserves the exact source and test receipt. Local preparation cost was not separately metered.

Before another GPU attempt, qualify the exact Qwen tokenizer, including the stop command and length bound, on CPU. Integrate the callback in a newly pinned worker and exercise it with an artificial generation driver. Keep the old source, failure, tokens and journals unchanged. Save any newly trained adapter before attempting generation so a later interface failure retains the trained state.

The next worker also needs more precise timing. The failed run spent about163seconds between input verification and the combined imports/CUDA-ready event, then about35.5seconds loading and transferring the model and installing adapters. Roughly24seconds passed between its failure receipt and supervisor completion. Those phases do not identify whether imports, CUDA initialization, shared storage, shutdown or another dependency caused the delays. Separate these events and qualify CPU-only imports before scaling. Twelve recorded training steps used3.929577seconds of CUDA event time, excluding other GPU work. Peak framework allocation was3,301,825,024bytes. These measurements do not justify claiming that the252second GPU allocation was efficient.

The old worker reached generation after its frozen-base, reference and adapter-update checks, but their final values were not persisted. The next worker must record those results before generation. Cleanup established process-group disappearance with no recorded errors. It did not include an independent post-exit NVIDIA sample, so GPU-process clearance was not directly observed by that telemetry.

No new scientific worlds, training preferences, private scores or utility conclusions arise from this change. The research agenda still requires a strong numerical acquisition control, a learned numerical policy with matched supervision, independent development and confirmation, and a useful experimentalist task.

## Prospective v2 integration

The working implementation now connects the finite grammar to the Transformers prefix callback for a single unpadded prompt. It requires a separately pinned CPU decoding fixture and a generation cap sufficient for every menu command. A new source schema binds the grammar module explicitly. The unconstrained GPU03 packet remains unchanged.

The revised worker records individual Torch, Transformers and AutoModel imports, CUDA initialization, CPU model loading, GPU transfer and adapter installation. It samples its own Python stack every60seconds for diagnosis without treating those samples as acceptance. It records pre-generation checks and an adapter checkpoint, then the generated token sequence. A function-exit event and supervisor process-exit observation help distinguish execution from interpreter teardown. Failure cleanup now retains the final NVIDIA query or its uncertainty.

Five new local integration methods passed using the existing local framework and character tokens. One affected freeze-schema method also passed. These do not qualify CURC's tokenizer or framework version. Preparation01 records0.690354childCPU seconds for the new methods and0.051594seconds for the affected method. Peak child RSS239845376bytes is the cumulative high-water mark across that preparation. Other development overhead remains unmetered.

The next CPU-only command imports the actual target model class without instantiating it, authenticates the cached model and source bytes, reconstructs the previously accepted artificial prompt, and checks the actual tokenizer and Transformers mask at every distinct legal-command prefix. It preserves allowed-token logits and excludes all other tokens. No training, CUDA query, pretrained model loading or environment response is planned in this CPU stage.

Proposed request: oneCPU,4GiB,tenminutes on acpu/cpu-normal underucb736_asc1, noGPU andno-requeue. The two prior CPU stages allocated twoCPU threads, so reserve1200CPU-seconds conservatively if submitted. This would make21600/57600reservedCPU-seconds while leaving1800/7200GPU-seconds unchanged. It is not an allocation until a separately reviewed frozen packet is submitted once. The restored transport worked on the new check. Shared controller ownership and submission journals must be reconciled again before launch.
