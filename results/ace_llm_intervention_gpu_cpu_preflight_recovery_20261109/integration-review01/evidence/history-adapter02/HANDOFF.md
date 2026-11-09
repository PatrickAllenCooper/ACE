# History adapter02 — schema-only successor

Source SHA256: `997482f0bb9e86d3f3c0b721cf77ba55bb2570a0610445757a10e0836d3944d0`.

The sole source change is `SCHEMA = 'ace-cpu-history-readset-v1'`, matching the reviewed producer and actual readset. The original adapter01 and tests remain unchanged; their source/handoff/receipt/inventory are captured under predecessor/. `source.diff` preserves the exact one-line delta. Static normalized AST comparison establishes no other code change. The public capture/install API, bounds, semantic predicates and external pre/post gates are unchanged.

Actual metadata-only attempt01 passed once using CPU acceptance `3c5ce7375d0c64292553ded6ad3078e634745eb7996d83fd102e344eeab3f063`, bundle `0311030f64634cdbe6e565c8721a3dc5b59a8a69e25c47545d9524446208a899`, readset `9be8223034cadedb0a3ff885236f923519b81f8c98bd46a7ffbcd2dafd9812b0`, and original campaign source `9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe`. All 133 files / 5,548,472 raw bytes and five CPU attempts passed capture_history. Exact input bytes/pins, invocation, helper and harness source, result and transcripts are retained in attempt01.

Cost: 0.219068 child CPU-seconds, 0.225666292 wall-seconds, 86,032,384 bytes macOS child maximum RSS. No original tests repeated. No install call, original project callable, live historical path scan, model, tokenizer, world, response, remote operation, lock, allocation or commit. Captured scientific metadata was not printed. This qualifies saved metadata admission only, not live installation, full preflight, independent review, GPU execution or scientific acceptance.

The assembler task continues separately.
