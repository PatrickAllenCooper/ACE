# Matched supervision conversion review

Independent reviewer: Descartes, separate bounded read-only review on November 2, 2026.

Scope: `llm_intervention_supervision.py` and its new artificial tests. This review concerns conversion of one pinned record into aligned numerical and language training views. It does not qualify the execution of a supervision campaign.

## Findings and disposition

Zero required defects. The reviewer checked raw-record and receipt pins, default policy-training scope, explicit artificial scope, the shared public envelope, available action identities, numerical action indices, exact JSON completions, duplicate-pair rejection and preservation of ties. Labels and provenance metadata stay outside policy inputs. Strict stop preferences are distinguished from unlabeled ties.

The optional coverage request was accepted. One new focused method checks supervised stop and strict preferences with stop on either side. It passed without repeating the original three methods. Production source remained unchanged. This extra test used fabricated records and artificial states only.

Reviewed production source SHA256: `184d685974cea991adc82ba349810e80dd8ddef1e88191d59443df1d9f1ed626`.

The original three methods are preserved in preparation01. The new stop fixture is preserved in preparation02 under `results/ace_llm_intervention_supervision_preparation_20261102`, with separate exclusive external originals. The commands consumed 0.086115 and 0.077777 child CPU seconds. Preparation and review overhead are not fully metered.

## Remaining scientific obligations

A matching receipt hash does not establish that its experiment ran. The future dataset producer must authenticate receipt contents, source, runtime, charged branch feedback and world namespaces. This converter must not be described as evidence of useful training, correct scientific labels or independent confirmation. Tokenization and actual LLM training remain separate gates.
