# Paired action-language fixture

At source revision `0910df0e`, eight short English actuator descriptions cover four formal schemas: no joint actions, a cost cap that rules out pairs, a hazard exclusion with one legal pair, and a proxy-only regime. The model-facing descriptions are in `prompts.jsonl`; formal schemas and exact legal action lists are in `answer_key.jsonl` and must be withheld during inference. Legal counts by schema are 6, 6, 10, and 2. The deterministic validator enumerates and checks every candidate action.

This is a hand-authored smoke fixture, with two related paraphrases per schema. It is too small and too templated to evaluate language generalization. A frozen rule baseline solves all eight after development on these same texts; independent descriptions and a held-out parser are required before testing a foundation-model contribution.

Prompt and answer files replayed byte for byte (SHA-256 `56a539b71bad53e7012fa9b506fad30adcbcf3bbde7708d0dfe6becc15f90be4` and `d90959bcf3fdf003e42a87199cc7ccadb8a6887d587e9e6d2f827990ec5240fd`). Source and receipt hashes are in `complete.json`. No simulator query, CURC job, or closed-model call was made.
