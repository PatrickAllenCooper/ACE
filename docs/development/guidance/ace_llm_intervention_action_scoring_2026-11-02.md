# Full-context training with action-position logits

## Why this change is necessary

The shared task and SCM history can be much longer than the original artificial GPU prompt. The cached tokenizer measured 4341 prompt tokens for one five-node, 32-row artificial state. The original scoring primitive projects every sequence position to the full vocabulary even though only action tokens enter the loss. The new `llm_intervention_action_scoring.py` removes that unused vocabulary projection. It does not shorten the public state or detach the prompt from training.

The intended model path is the pinned Qwen implementation. In [Transformers 4.57.3](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py), a tensor-valued `logits_to_keep` selects hidden-state positions before the output linear layer. This is the required interface. Returning full-position logits despite the request is rejected.

## Objective and gradient preservation

Let the full causal transformer produce hidden states at every input position. For each row, a label at position j is predicted by the vocabulary logits at position j minus one. The new implementation finds the union of those prediction positions across the batch. It runs the full input through the transformer, then requests the final vocabulary projection only at that union. Each row still masks positions whose labels are absent, so unequal action lengths remain valid.

The supervised objective remains the mean of each action's negative sequence log probability divided by its scored token count. The preference objective still compares chosen and rejected sequence log probabilities against the same fixed reference. EOS remains a scored action token. No numerical reward, threshold, training example or optimization schedule changes.

For a fixed deterministic forward function, the discarded vocabulary rows have zero coefficient in the original loss. Removing those rows leaves the mathematical objective and its parameter derivatives unchanged. This statement relies on the output projection acting independently at each position, as the pinned implementation does. It is not a guarantee of bitwise equality across devices, kernel choices or precisions. The entire transformer context and its gradients remain present.

The scorer rejects mismatched input/label identities, first-position labels, labels on padding, attention holes, empty scored rows, nonfinite output and an ignored selection request. It does not silently fall back to context truncation or approximate training.

## Local evidence

Five new artificial methods pass. They compare supervised losses and all parameter gradients, compare preference adapter gradients with a fixed reference, test unequal scored-position masks, verify that prompt-token embeddings still receive gradients, and reject malformed input or unqualified output. The toy causal model's output projection sees three positions instead of nine while its contextual computation still sees all nine. This is an exact small oracle, not a pretrained Qwen benchmark.

The command used 0.827029 child CPU seconds, 1.081857417 elapsed seconds and 277708800 peak child RSS bytes. No pretrained model, optimizer update or scientific response occurred. Code and logs are preserved in `results/ace_llm_intervention_action_scoring_preparation_20261102/preparation01`. Implementation and review overhead are not fully metered.

`projection_profile` reports vocabulary tensor element counts only. It does not estimate total model memory or promise a speedup. Attention, transformer activations, gradients, optimizer state and generation cache still require measured resource qualification.

## Target CPU qualification

`prepare_llm_intervention_task_cpu.py` is a separate bounded integration worker. Its proposed allocation is one requested CPU, at most two allocated CPU threads, 4 GiB and ten minutes on `acpu/cpu-normal`, account `ucb736_asc1`. It has a 560-second worker deadline, no GPU and no requeue. A new allocation would reserve at most 1200 CPU-seconds, bringing the current conservative cycle reservation to 30000 of 57600 CPU-seconds. GPU reservations remain 2700 of 7200 seconds.

Before imports, the worker authenticates its exact source set, five saved artificial task envelopes, the accepted GPU04 adapter and the cached seven-file model inventory. Package versions and Python must match the target freeze. It verifies actual Slurm identity, account, CPU, memory, duration and absence of accelerators. A new allocation-gate check accepts prior scheduler metadata and rejects six fabricated resource/state changes. This is not fresh scheduler evidence.

The worker will use tiny randomly initialized instances of the actual Qwen class. It compares full and selected scoring losses and adapter gradients, restores a fabricated tiny adapter into the unchanged tiny base and compares scores, then generates two legal commands from two saved public task states with the real tokenizer. It also validates the accepted full-size adapter's tensor inventory against the cached model configuration without constructing or loading that pretrained model. The five token profiles use existing artificial observations. No environment callback or optimizer step is requested.

The source and packet reviews, controller locks, scratch readiness and remote authentication passed before job35207064 was submitted once. The job completed0:0 and the separate saved-artifact review found no required corrections. The final CPU acceptance records the qualified tiny target-runtime integration. It would not qualify full-size restored inference, long-context GPU memory, production efficiency or intervention utility. Those remain required before the scientific comparison.

## Resource-planning constraint for the next GPU stage

The current scoring context sets the model to evaluation mode while retaining gradients. The pinned Transformers checkpointing layer activates its built-in checkpoint path only when both its checkpointing flag and training mode are true. Therefore, enabling that flag alone would not provide activation checkpointing in this scorer. A resource projection must not assume those savings. See the [versioned checkpointing implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/modeling_layers.py).

First measure the intended full-size, selected-position workload with an explicitly frozen attention implementation. If activation checkpointing is needed, qualify its execution and gradient equivalence separately while preserving deterministic scoring. No checkpointing behavior was added or claimed in the accepted CPU source.

Final CPU acceptance: `results/ace_llm_intervention_task_runtime_20261102/cpu01/cpu_acceptance.json`, SHA256 `90ef0d4ab4a74589735726bab5ca22bbd8ff1720c67044e1d526d9f63974a31d`. Complete43-file custody, source pins, exact package versions and163seconds at2allocatedCPU threads are retained. The five prompt sizes matched. Full/selected SFT and preference losses and all eight gradient tensors matched exactly in the tiny FP32 checks, as did tiny-adapter reload scores. This result does not extend to the full-size model or GPU precision.
