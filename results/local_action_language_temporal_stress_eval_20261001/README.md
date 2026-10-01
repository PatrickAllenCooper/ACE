# Frozen rule parser on later-authored descriptions

The unchanged `scripts/research/action_language_rule_baseline.py` parser, previously 8/8 on the original fixture, produced **0/8 exact schemas and 0/8 exact legal menus** on the temporal stress fixture. Six descriptions raised `ambiguous or absent action cost limit`; the remaining two parsed but produced 24 actions where the gold menus contained 6 and 2. Evaluation SHA-256: `5f81b55ee0fdf23006358e456f37ed2eac48b536566b9cd5aad56d42ded65866`. A fresh fixture/evaluation replay was byte-identical.

This diagnoses sensitivity to wording, not a measured model advantage: no open model was run, and the texts were authored after the rule grammar was visible. Do not tune the parser on these texts and report them as held-out performance. A fair language-model comparison needs externally or independently authored descriptions, a frozen prompt and strict output parser, and the same deterministic action validator.
