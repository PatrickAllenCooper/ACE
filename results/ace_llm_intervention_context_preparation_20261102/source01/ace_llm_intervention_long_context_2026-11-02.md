# Full-size shared-task GPU qualification

The accepted target CPU check qualified tiny Qwen integration and exact tokenization. The next bounded artificial stage tests the actual cached Qwen3-0.6B, full-size adapter restoration and longer public SCM histories. It generates no scientific worlds or environment responses.

## Fixed computation

Use the pinned model revision c1899de289a04d12100db370d81485cdf75e47ca, the accepted GPU04 adapter and the exact target runtime. Keep the base in BF16, q/v adapters and reference buffers in FP32. Restore the accepted adapter into a fresh base on CPU before transfer. Enable gradients only for the112policy adapter tensors.

Use SDPA with only its FlashAttention backend allowed. Unsupported operations fail rather than switching to a more memory-intensive backend. This uses PyTorch's existing implementation, with no installation. [PyTorch2.9 documents the backend-selection context](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html). No activation checkpointing is enabled or assumed.

Four CPU-prepared artificial envelopes have613,2526,4341 and7343prompt tokens. The final envelope repeats a neutral public note to stress length. Its numerical state is identical to the4341-token fixture. It is not a realistic scientific history or a universal maximum. Freeze full prompt, legal grammar, action and label-mask tokens before allocation. Runtime tokenization must match exactly. Context cap8192 and32newtokens apply without truncation.

First compare original and action-position loss paths on the613-token fixture before updates. Compare SFT and DPO losses and every adapter gradient with fixed tolerances: loss absolute0.0001/relative0.00001 and gradient absolute0.0001/relative0.01. Require nonzero finite gradient signal. Failure is retained, not repaired by adjusting tolerances after observing it. This is finite BF16 integration evidence, not a universal equality proof.

Then run three supervised and three preference updates, each with two arbitrary legal intervention labels. In each stage the first step uses4341prompt tokens and the next two use7343. These labels have no claim to scientific value. The preference reference is captured after supervised updates. Save its original tensors and verify its scores and bytes remain fixed. Check frozen-base bytes and changed policy adapters. Measure complete backward/update work, optimizer memory and CUDA event time.

Save the trained artificial policy, release the first model and restore into another fresh pinned base. Require exact retained-score and tensor agreement. The restoration helper gives this final policy a new identical reference. It does not reconstruct the earlier DPO reference, which is saved separately. Generate four constrained decisions over three states, including a repeated state. Persist every command and tokens. No command is executed and no response count or history changes.

## Bounded resources and failure handling

One A10020GB testing slice is the smallest compatible slice supported by the current site configuration. Prior GPU04 used3718MiB independently observed process memory and3.30GB framework peak on a289-token prompt. Those measurements establish the base fits, not long-context demand. This is the measured length qualification needed before a production allocation.

Request4CPU threads, allow at most8allocated,12GiB host memory and15minutes onaa100/gpu-testing/accountucb736_asc1. Preserve420second startup,840second worker and870second whole-supervisor deadlines. Require an actual forward by the startup deadline and progress within120seconds afterwards. Owned NVIDIA process memory and CUDA step timings/progress are independent utilization evidence. Whole-device utilization is not attributed to this job. Verify owned process-group and GPU cleanup.

A new attempt reserves at most7200CPU-seconds and900GPU-seconds. Conservative cycle reservations would become37200of57600CPU-seconds and3600of7200GPU-seconds. This leaves production resources but does not authorize wasting them. Actual accounting follows Slurm allocation and elapsed time. It is not added to overlapping process or reservation costs.

This stage may fail due to memory or kernel compatibility. Preserve the exact failure and make the next resource or implementation choice from those observations. No automatic retry, context truncation, scientific-world generation or production-QOS substitution is part of this packet. Accepted studies, prior qualification packets and the archived paper remain unchanged.
