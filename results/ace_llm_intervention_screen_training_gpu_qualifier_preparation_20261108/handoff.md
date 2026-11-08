# Screen training qualifier: focused stdlib test handoff

One new batch: 10 methods, 9 passed, 1 failed in three subcases (0.330 s unittest; 0.421407 child CPU seconds). Full failed attempt, fabricated inputs, source copies, stdout/stderr and execution metadata remain in `attempt-01`; no rerun. All pins are toy module-local substitutions, never actual acceptance qualification.

Required exact-membership gap (P2): `validate_prepared`, line 137, checks only entries supplied in `prepared.source_sha256`. It accepts `{}`, omission of `llm_intervention_gradient_probe.py`, and addition of an unrelated but correctly pinned packet helper. The saved preparation builder declares exactly 18 source names. Require that exact set before comparing digests. The independently pinned production prepared artifact still prevents arbitrary replacement; these tests demonstrate missing structural validation, not a bypass of its immutable pin. The parent-owned qualifier is unchanged.

The other checks cover packet source/data membership, independently pinned helper/input binding, capacity runtime/model binding, affirmative CPU acceptance, token/raw-envelope/source identity, independent plan pinning, capture-before-helper execution, and AST verification that model authentication precedes heavy imports and all worker calls receive both explicit pins. Existing progress/cleanup controls were compared statically only; resource/deadline bindings are recorded in `retained-controls.json`.

Test source SHA256: `087059008a06a606fedd2b846176175efb0cc780363646794aeb2765caacd46d`.
Tested qualifier SHA256: `059bc0f426efc81dbab7bf7328e7f82c2285fd572ea243a69922429b431078fb`.

No models, tensors, scientific draws, tokenizer, GPU, remote actions, original tests or runtime qualification. After the parent fixes the membership check, only the failing new method needs a fresh retained delta attempt.
