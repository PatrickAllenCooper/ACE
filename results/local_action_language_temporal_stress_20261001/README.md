# Later-authored action-language stress fixture

Eight new descriptions reuse four formal actuator schemas from the earlier eight-text smoke fixture. Frozen at source revision `d6067d6e662894909520eab351c73e5c60c0ccad`, after the earlier parser had already been committed. Exact legal-action counts are 6, 6, 10, and 2 by regime. The public descriptions are in `prompts.jsonl`; formal schemas and complete legal menus are in `answer_key.jsonl`. All actions passed the deterministic validator, a separately recomputed canonical menu matched each gold set, and a fresh run reproduced every output byte and SHA-256 hash.

These descriptions were authored by the same agent after seeing the rule parser. They are a **temporal stress set**, not independent human-authored validation. No simulator query, GPU job, or model call was made.
