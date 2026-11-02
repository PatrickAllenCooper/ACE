# Action-position scoring and CPU qualification review

Independent reviewer: Mencius, separate read-only review on November 2, 2026.

Scope: the new action-position scorer, its five artificial test methods and the new target CPU worker. Existing accepted learning and language primitives were read for interfaces but their completed reviews were not repeated. The reviewer ran no tests, models or remote commands and edited no files.

Zero required correctness or scope defects were found. The review checked causal indexing, union-of-scored-position masking, supervised normalization, preference-gradient flow and the qualified Qwen `logits_to_keep` interface. It also checked authentication order, actual resource validation and the worker's restriction to tiny random models and previously saved artificial states. The primary implementation reference is [Transformers 4.57.3 Qwen3](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py).

The parent executed five new artificial full/selected score and gradient methods. All passed. A separate new allocation-gate check accepted historical metadata and rejected six fabricated changes. Those checks do not prove a live allocation or target-runtime execution.

Full-size restored Qwen inference, BF16 GPU gradient comparison, long-context GPU demand and scientific acquisition utility remain open. This review does not establish them. The actual frozen packet needs its own narrow consistency check before submission.
