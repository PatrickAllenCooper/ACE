# Shared-task language policy review

Independent reviewer: Descartes, a separate bounded read-only implementation review on November 2, 2026. The reviewer did not run tests, load models or edit files.

## Scope

The review covers the new adapter restoration and shared-task token/generation primitives and their artificial tests. It does not revisit GPU04 acceptance, the old learning primitive or scientific results. Initial source `f0a9f3d2` and test `94717f00` are retained in preparation01. Final source SHA256 is `4544c3b47b0d92bf18fa8f09d508c1aeb29cd0fa8293940d8453dff701ab466d`. Final test SHA256 is `932577eaf7dcf0a10768666dcb1e9364d116efd9a6c1112b94065f6b1803dc26`.

## Required finding and correction

One required defect was found. A fresh generation configuration does not by itself prevent Transformers from applying stored model defaults. In the target version, explicit values equal to global defaults can still be replaced by stored values. The code now passes `use_model_defaults=False`. The versioned primary source is [Transformers 4.57.3 generation utilities](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/generation/utils.py).

A new artificial test calls the actual locally installed configuration-preparation helper. Conflicting stored sampling, beam, forced-token, suppression and repetition settings are inherited in a positive control and excluded by the corrected deployment call. Local version 4.51.3 is recorded. This is not execution in the target version 4.57.3.

The narrow static recheck found zero remaining required defects. The reviewer confirmed that the changed call and its regression close this finding. The original six-method command remains preserved. Two new methods and two affected generation checks passed after the correction.

## Optional finding

A normalizing tokenizer could encode a noncanonical label as an otherwise legal command. The production check now requires exact canonical legal text before encoding. One new negative test demonstrates rejection. This closes the optional gap without changing the matched supervision converter, which already produces canonical commands.

## Other inspected behavior

Adapter bytes are pinned before deserialization. Every tensor key, shape, dtype and finite value is checked before installation. Restoration uses a CPU model, preserves caller CPU randomness and does not seed accelerators. The final parameters are frozen. Training and inference share the prompt builder, and oversized prompts are rejected rather than truncated. Complete generated commands must match the bound prefix and grammar before return.

The fake generation fixture is not a real Qwen forward pass. The local real-tokenizer profile is separate evidence with its own original source and pins. Actual base/training provenance, target runtime, restored Qwen inference, multi-episode execution, GPU resources and scientific utility remain open. No intervention-effectiveness claim follows from this review.
