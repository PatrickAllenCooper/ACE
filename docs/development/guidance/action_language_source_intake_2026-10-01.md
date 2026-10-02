# Independent action-language source intake

Status: source candidates identified; **zero adjudicated benchmark tasks**, no model inference. These are externally authored documents, but multiple clauses in the same manual are not independent domains. No passage has been admitted to a confirmation set. References were inspected on 1 October 2026; pin source versions/content hashes before creating prompts and gold labels.

## Candidate source group: Opentrons apparatus/API documentation

- [Heater-Shaker module](https://docs.opentrons.com/python-api/modules/heater-shaker/): state-dependent latch/shaking rules, deck adjacency exclusions, height restrictions, and an exception for particular pipette access to tip racks. This offers conjunctions, exceptions, spatial context, and temporal state. Fix robot generation and API version; hardware-specific clauses cannot be mixed.
- [Pipette characteristics](https://docs.opentrons.com/python-api/pipettes/characteristics/): tip/pipette compatibility, capacity, and rate units. This offers type compatibility and dimensional constraints. Distinguish recommendations from hard restrictions.
- [Complex command sources and destinations](https://docs.opentrons.com/python-api/complex-commands/sources-destinations/): command-specific source/destination semantics and differences from single-location operations. This offers action granularity and binding, rather than an explicit cost cap.

All three belong to **one vendor/source group**. Hold this group out together when measuring transfer to a new documentation source. Public text may already appear in model pretraining; source independence from the ACE parser does not establish absence of pretraining exposure. No physical hardware will be operated by this study.

## Schema and evaluation requirements

The present static actuator schema cannot represent latch state, spatial adjacency, ordered commands, or compatibility exceptions. Do not force these into the old two-value menu or label failures as model failures. First add a small typed state-transition representation, tested against the documented API constraints where a numerical simulator is available. Keep hard errors, recommended practice, and ambiguous statements distinct. Only then freeze finite candidate commands, task context, and gold valid-action sets.

The original author is the documentation publisher; the task-context author and formal-gold adjudicator must also be recorded separately. Agent paraphrases are not independently authored descriptions. Gold must be checked before model inference, with ambiguous cases excluded or assigned an abstention target. A simulator/checker with complete formal rules must be a deterministic comparator; private gold is never a runtime model input.

Next intake target: a second independently authored apparatus or experimental-protocol source with an auditable formal interface. Aim for at least 20 adjudicated descriptions across source groups, while retaining document/source as the statistical unit. Neither a three-link inventory nor twenty clauses from one manual meets the intended external-transfer gate.
