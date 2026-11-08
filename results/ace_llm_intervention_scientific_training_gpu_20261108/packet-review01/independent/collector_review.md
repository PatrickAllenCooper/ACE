# Independent GPU training collector custody-seal bound review

2026-11-08. Parent reviewer, distinct from implementation author. Zero required findings for corrected collector41603ce1bc2f64db019c7ff66bafdc3df3feeee994bf78445bda1b8f91bb1c34.

The independently accepted TOKEN02 custody inventory is33,854,831bytes, exceeding the predecessor collector's32MiB metadata limit of33,554,432bytes. This is a downstream reader limitation, not an experimental failure. The correction introduces a dedicated64MiB reader at exactly the two complete upstream custody-seal reads. It requires the custody-inventory.json basename, independent SHA256, regular sole-link file, streamed fingerprint, bounded read plus sentinel and complete byte/size/digest agreement. Ordinary metadata retains its original32MiB reader unchanged. There is no truncation, omitted history, rewritten seal or automatic retry.

The source diff was inspected against80757151c2c6848b67929bae7453f18e4856da0ce41138cba74a7f407dd789a1. All complete capsule, raw-root, transport, two-copy, accounting, scheduler, source, failure-preservation and false-acceptance checks remain unchanged. No core, frozen child, recipe, runtime, model, resource contract, launcher, monitor or CLI changes. This source and review must be independently committed and pinned for the future standalone collector.

Two new boundary methods and their retained results were inspected without rerunning them. The exact observed size class is accepted as fabricated JSON with pin/readback while the ordinary reader rejects it.64MiB+1 is rejected before whole-file buffering or decoding. Both passed once,0.176654childCPU/0.183282292elapsed seconds,123944960peakRSS. These are fixture costs, excluding implementation/review and scientific allocation. Actual TOKEN02 data was not used as fixture content. Original source and complete fabricated evidence remain preserved.

This is a bounded per-seal limit. It does not imply that larger future nested custody seals will fit, qualify any actual training or permit altering originals. No actual collector, model, tokenizer, GPU, remote command, scientific response or completed component test was executed by this review.
