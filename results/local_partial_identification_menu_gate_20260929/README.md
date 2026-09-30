# Exact action-menu identification gate

Source revision `3c664855`; exact two-model prior with equal weights. `do(X=1)` has 0.311278 bits of expected information about which candidate SCM generated the world. An irrelevant `do(Z=1)` has zero information. With both actions available, the oracle selects `do(X=1)`; with only `do(Z=1)`, it abstains and leaves two candidates unresolved.

This validates a discriminating-action and null-menu grading rule. It does not test a language model, account for model misspecification, or establish an identified set over unrestricted SCMs. Both actions have unit cost; no simulator calls were made. The result and receipt contain the pinned revision and SHA-256 hash. An independent replay produced byte-identical `result.json` (SHA-256 `aa0b18d092b30775c498cb1acba1aafe45ae84c42f85c1456978988cdca99abc`). No CURC or closed-model resources were used.
