# Two-candidate partial-identification task suite

Generated from frozen protocol `docs/development/guidance/protocol_partial_id_open_model_dev_2026-09-30.md` at source revision `85a164ec5237fef12bf6a577ec5ee45ef23a5873`.

Twenty exact binary-SCM tasks use ten discriminating and ten null action menus. `stage1_prompts.jsonl` is the public input; `stage2_reveals.jsonl` is revealed only after stage-1 answers; `answer_key.jsonl` is private to the evaluator. `complete.json` records SHA-256 hashes. An independent Fraction-based audit recomputed all twenty candidate-set intervals and posteriors; a fresh generation reproduced every file hash.

This is a narrow synthetic development family. It used zero simulator queries, zero open-model calls, zero closed-model calls, and no CURC allocation. It tests neither model competence nor a scientific discovery yet.

A pre-result audit of the 20 answer-key cells found 10 nonpoint and 10 point candidate-set intervals for `E[Y|do(X=1)]`; only three reference outcomes collapse the posterior support to one candidate. The frozen four-task canary (`pair_2100`–`pair_2103`) contains **zero nonpoint intervals** and one support-collapse event. It can test parsing, action/abstention, and updating, but does not exercise numerical interval uncertainty. Before any full-suite interpretation, run a separately labeled balanced follow-up including at least one informative and one null task with nonpoint intervals (for example `pair_2104` and `pair_2105`), with the same frozen model/scorer. This limitation was found before viewing any model output.
