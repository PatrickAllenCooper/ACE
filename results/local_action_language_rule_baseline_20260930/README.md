# Text-rule baseline on the paired action fixture

The baseline reads only `prompts.jsonl`, extracts cost/joint/safety constraints with fixed phrases, then enumerates legal actions through the deterministic validator. It produces exact schemas and legal menus for **8/8** hand-authored tasks. This is an in-sample result: one missed phrase (`never use danger`) was added after an initial 6/8 run on the same texts. It must be frozen before any independent paraphrase set and cannot establish generalization.

Source revision `36f239ef`; `predictions.jsonl` replayed byte for byte (SHA-256 `c8f1c9bdb95e5c6a08a74c4f9d3d9a23de7ca018d7646b1e8717f65ce708e2c3`). `complete.json` pins the fixture hash and counts. The result sets a strong trivial-control bar for a future open-model compiler: success on these eight examples would add no evidence. No simulator query, CURC job, or closed-model call was made.
