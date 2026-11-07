**Independent ranker publication delta recheck — November 7, 2026**

**Required findings: 0. The original P2 is closed for the corrected source and documented acceptance contract.** The [initial review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_ranker_training_review_2026-11-07.md) and its original reproducer remain unchanged. Their finding remains valid for the earlier source; this separate disposition records the correction.

Scope was limited to the source/guide delta, new publication tests and preserved publication capsule. Static comparison confirms two added helpers, nine existing top-level functions unchanged at AST level, and only the two intended publication/disposition call substitutions inside `fit`. Training arithmetic, exposures, optimizer/reference handling and exports were not re-reviewed or executed. No old or new tests, models, Torch imports, scientific data, remote operations or commits were run for this recheck. Only this report was written.

At [lines 63–71](/Users/pat/code/ACE/scripts/research/llm_intervention_ranker_training.py:63), `_publish_terminal` writes an exclusive diagnostic candidate through the existing flush/fsync helper before calling `os.link` to publish the authoritative terminal. Candidate-write/fsync failures therefore cannot create a new success terminal. Link failures retain the candidate, and an existing target is not overwritten. The hard link is the helper's final operation; the successful [fit path immediately returns](/Users/pat/code/ACE/scripts/research/llm_intervention_ranker_training.py:290).

At [lines 74–81](/Users/pat/code/ACE/scripts/research/llm_intervention_ranker_training.py:74), `_failure_receipt` uses `lexists` to distinguish a visible terminal from a pre-publication failure. A visible terminal leads to `publication-interruption.json` with `publication_interrupted_reconciliation_required`, rather than a competing final failure receipt. This also handles a preexisting target conservatively without accepting or overwriting it. The [exception path](/Users/pat/code/ACE/scripts/research/llm_intervention_ranker_training.py:292) retains its prior partial-state handling and delegates its disposition to this helper.

The [guide's acceptance contract](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_ranker_training_2026-11-07.md:66) now requires clean API/process completion, no publication interruption and the original source/export checks. A terminal, candidate or diagnostic alone cannot establish acceptance. Directory durability, hard termination and interrupted publication remain explicit outer-supervisor obligations. This distinguishes atomic path publication from crash durability and resolves the original ambiguous-disposition defect within the sequential exclusive-output scope. The oracle description is also correctly narrowed to a separate loss/update loop using the same PyTorch Adam.

The preserved [five focused tests](/Users/pat/code/ACE/scripts/research/test_llm_intervention_ranker_publication.py:15) extract only `canonical`, `_save`, `_publish_terminal` and `_failure_receipt` through AST. Their source and saved outputs cover candidate fsync failure, link failure, preexisting-target rejection, successful same-inode publication with the link as the final helper operation, and interruption immediately after a real link. Saved failure fixtures have no success terminal; interruption fixtures have a reconciliation diagnostic and no `failure.json`. The saved log reports **five passed in one invocation**. Its receipt records **0.05032 child CPU seconds / 0.090842125 elapsed seconds / 20,873,216 bytes peak child RSS**. These are the original focused-child measurements, not new executions or whole-cycle costs. The controls do not constitute a full-fit interruption or target-filesystem durability qualification.

Independently verified the new repository capsule `results/ace_llm_intervention_ranker_publication_preparation_20261107` and external `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-ranker-publication-preparation-20261107-01`: **24 files / 68,901 bytes each**, all 23 manifest-member sizes/hashes matching, identical manifests, no extra/missing files, and all **five current source/review pins** matching. The initial review and original probe source/result hashes still match the previously recorded values. No prior full-capsule or numerical review was repeated.

This closes the publication correction only. Scientific adoption, integrated ranker training and target-runtime qualification remain pending under the original scope.

SHA-256 bindings:

| Artifact | SHA-256 |
| --- | --- |
| Corrected trainer | `0909428c47288ebc201abd988046d9be2283f70344d3764dcc9f6bfb845c0587` |
| Earlier reviewed trainer | `5ba9d41c9df4c35e81d77131ae735f92b452eefce6bdcd27d45b5b1e7f1d63c5` |
| Corrected guide | `6395d4828fe34598e3db6d4418d79e9e2397c21f8fe61e388de0f6785c8255ff` |
| Focused publication tests | `b42a96073da984bfbe95ba729c63881b1b63dbe8b8a3a58783392c77ae9c81d8` |
| New capsule manifest | `845df5c250c8d6d529c2cb638a8a9766df6e23cc4905904334ae2192c0b51f9b` |
| Focused execution receipt | `40230c963b32cdcd0e7f7413f801a52066c3609d45d0747c0204fe39c5506dfc` |
| New source manifest | `81085d2565314e9d0c0ecabba2c13700cfe6b22014ef4c6213db4cf258600c02` |
| Preserved initial review | `ee80faf6f3d71a8d00e3f046db5f67d8640a6ecd252607854a88fcb35f56d55a` |
| Preserved original probe source | `39244c9f0f4c02c036e5eefeaed7af5382ecc2905df5fbed9dfa7b0b1de4b26f` |
| Preserved original probe result | `94d9b17a86775e0334667ecf8df170f787f291469c48ddd970466750cd9f2bbd` |
