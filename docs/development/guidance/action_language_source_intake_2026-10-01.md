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

## Typed transition implementation (02 October 00:37 UTC / evening local)

Implemented `action_state_contract.py`: finite typed state domains, conjunctive command preconditions, deterministic effects, exact legal-command enumeration, and sequential execution with rejection at the first illegal command. Rejected commands leave state unchanged. Unknown commands, incomplete states and undeclared values fail. Boolean states do not accept integer0/1. No expression evaluation, hardware execution, network access or model calls occur.

Engineering fixture models a latch and shaker with four commands. Exhaustively checked341 sequences up to depth4 from the declared initial state; accepted sequences preserve the invariant that an open latch cannot coexist with shaking. Source/result hashes retained at results/action_state_contract_check_20261002. These are internally authored fixtures, not adjudicated translations of vendor text. Gold benchmark count remains zero.

The contract currently handles finite state equality and conjunction only. It does not yet encode spatial exclusions, unit conversions, recommendations, ambiguity, parameterized commands or hardware-version differences. These limits must be declared in the task schema; do not force unsupported clauses into the fixture. Next admit only independently pinned descriptions that fit this finite representation, record context author and separate gold adjudicator, and retain an abstention target for ambiguous clauses. If a runtime receives the full contract, exact legal-command enumeration is the required comparator. The transition engine is evaluator infrastructure, not a measured language capability.

## Temporal benchmark scorer implemented (03 October 01:37 UTC)

`temporal_action_benchmark.py` verifies a bounded candidate sequence set against evaluator-private formal transitions. It rejects incorrect/incomplete gold, duplicate candidates and unverified provenance status; scores false legal proposals, out-of-candidate sequences, duplicates, exact menus, recall and abstention. Ambiguous tasks require null definitive gold and are scored for abstention. `public_task` exports only description, declared initial context and candidate commands; formal contract and gold remain private. Complete formal runtime knowledge must still be compared against exact enumeration.

Internal checks cover correct menus, illegal ordering, duplicate proposals, bad gold, ambiguity and exclusion of private rules from public inputs. They pass, with custody in results/temporal_action_benchmark_check_20261002. Provenance fields cannot establish real independence; the fixture is explicitly internal, so independent benchmark membership remains zero. Source pinning and actual separate adjudication remain required before inference. A finite command subset is used here; this scorer cannot evaluate unseen parameterized commands or unsupported spatial/unit rules.

## Retained source and reviewer packet after d492f90b

`results/temporal_source_intake_20261002` retains the fetched Opentrons page hash, normalized-text hash, two exact source spans with character offsets and hashes, and two pending context cases. Full source custody is `/Users/pat/.cache/ace/instruction_sources`. The candidate menu contains39 ordered sequences of length1–3 from close-latch/start-at500rpm/stop-shaking, with explicit API bindings. These context variants share one source group; they are not independent descriptions or benchmark replication. No contract or definitive gold has been supplied.

Reviewer dependency: a separate adjudicator who did not author these contexts must verify API/robot scope and unsupported repeated-command semantics, then produce or approve the formal transitions and complete menus. No such reviewer is currently assigned. Do not replace that role by setting a different name string or by accepting the preparation agent's internal contract. The scorer should continue rejecting these pending records. The exact source supports only part of the internal fixture's transitions; reviewed exclusions or explicit ambiguity targets are required for the rest.

The packet is development intake. Freeze source-group splits once enough independently authored groups/tasks are admissible; no model scoring or inference before that gate. Existing Causal Chambers source is a second group but its numeric/parameterized protocol is not yet supported by the temporal scorer. A prepared packet is not an independent adjudication result.
