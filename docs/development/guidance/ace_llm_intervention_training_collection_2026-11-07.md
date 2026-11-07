# Supplied-world CPU training collection

`scripts/research/llm_intervention_training_collection.py` integrates the immutable
single-world dataset primitive. It consumes supplied descriptors, never generates
worlds, loads a model, trains a policy or computes private evaluation. Numerical
behavior and paid SCM updates are the existing primitive's collection work.

Call `collect_training_batch` with explicit `scope`, an ordered tuple
`world_roster` of integer IDs, its independent `expected_roster_sha256`, and four
exact-membership dictionaries keyed by those IDs: `worlds`, `world_pins`,
`configurations`, `configuration_pins`. Descriptor pins are semantic
`WorldDescriptor.sha256`; configuration pins are
`digest(configuration_payload(configuration))`. Each configuration has exactly
the typed result shape of `screen_protocol.assembly_recipe`: learner constructor
parameters, assembly parameters, `execution_authorized=False` and the four
pending runtime bindings. The collector supplies a fresh learner and exclusive
output per world. Pin the entire configuration, including the public prior, task,
behavior, teachers, candidates, milestones, replicates, namespace and tie rule.

Also supply `expected_source_pins` for exactly `SOURCE_PATHS` and a nonexistent
`output` directory whose parent exists. The collector compares current local file
bytes with those independent pins; observed files after imports do not authenticate
loaded executable state. Source/runtime/scheduler qualification, namespace
admission, honest public assumptions, scientific authorization and freeze remain
caller obligations. This trusted sequential helper is not a sandbox or a wall-time
supervisor. Hard kills may leave an unresolved intent; they never permit a retry.

For `policy_train`, supply the complete **current** proposal and its independent
semantic `expected_proposal_sha256`. The exact ordered 32-world roster and every
configuration must equal the frozen proposal's `assembly_recipe`, validated
against the same independent source map. A self-consistent tuned proposal, subset,
replacement recipe, source-map substitution or wrong known-noise binding rejects.
Descriptors must already have been independently supplied and admitted. No
scientific descriptors are created by this integration or its tests. The scientific
path has only pure recipe-admission coverage here; actual scientific execution
and external admission remain open. `artificial` accepts explicitly pinned bounded
recipes for hand-written worlds and rejects a scientific proposal argument.

All descriptors and configurations authenticate, and all static configurations
validate, before the first response. The exclusive intent retains roster/pins;
input files retain descriptors/configurations; the durable plan fixes every record
ID and every ordered candidate combination before collection. Execution follows
roster order, independent of dictionary insertion order. The first failure stops
the entire batch: completed, failed and unstarted worlds remain explicit. There
is no retry, resume, replacement world or complete-case filtering. Earlier complete
records remain evidence, never permission to train on a failed subset.

Each world's original response captures, reservation journals and failure/terminal
remain intact. Batch accounting reconciles those journals using the primitive's
existing accounting, including unknown returns and returned-but-rejected values.
Expected journal paths are retained in the batch plan before responses. An unreadable
world plan leaves the original error and failed/unstarted roster reportable;
readable charges remain counted, with unresolved totals explicitly marked as lower
bounds. Optional reconstruction failures cannot turn unknown costs into complete zeroes.
Per-world trajectory/branch/scoring detail accompanies additive batch totals.
Ceilings are 32 worlds, 256 records, 16,384 response attempts and 1,000,000 reserved
cost units. These are logical caps, not CPU or monetary measurements. Proposal
bindings impose the smaller fixed scientific recipe. No allocation is requested.

Only a complete batch publishes a usable `inventory.json` plus successful
`terminal.json` binding its SHA256. A fsynced `terminal-candidate.json` is retained
before exclusive hard-link publication. There is no fallible I/O after that commit;
directory durability remains caller qualification. A postcommit interruption writes
`publication-interruption.json`, requiring reconciliation instead of a contradictory
failure. A candidate or provisional inventory alone never admits exposure.
Authenticate the terminal's independently retained pin and its inventory binding,
and reject any failure/interruption disposition before exposure. Inventory
IDs are `scope/world_id/snapshot-NNN`, chosen before outcomes. Every record retains
all pairs, ties, negative gains and terminal stops. Record/receipt file hashes,
relative paths and the full candidate/pair roster accompany the exact ordered
record roster. To call the unchanged exposure helper:

```python
import hashlib, json

terminal_raw = (output / "terminal.json").read_bytes()
assert hashlib.sha256(terminal_raw).hexdigest() == expected_terminal_sha256
terminal = json.loads(terminal_raw)
assert terminal["status"] == "complete" and terminal["exposure_ready"] is True
assert not any((output / name).exists()
               for name in ("failure.json", "publication-interruption.json"))
raw = (output / "inventory.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == terminal["inventory_sha256"]
inventory = json.loads(raw)
proposal = propose_exposure(
    records={r["record_id"]: (output / r["path"] / "record.json").read_bytes()
             for r in inventory["records"]},
    record_pins=inventory["record_pins"],
    supervision_receipt_pins=inventory["supervision_receipt_pins"],
    record_roster=tuple(inventory["record_roster"]), scope=inventory["scope"])
```

Pins establish byte/configuration bindings, not independent scientific provenance.
Exposure remains a proposal; optimizer, tokenization, model and numerical-family
training integration are separate. Failed batches retain a partial inventory when
readable and the full planned roster; never pass their records to exposure.

Seven NEW test methods include an executable two-world fixture (IDs 17 and 4,
directly hand-written laws), complete exposure handoff, whole-roster preflight,
exclusive refusal, failed/unstarted entries, raw invalid returns and interrupted
returns. The success fixture expects four records, six pairs (two ties), four
SFT proposal steps and two DPO proposal steps; it performs no optimizer updates.
The only scientific fixture reads the already retained prospective proposal and
checks recipe adoption/rejections without constructing descriptors or responses.
Four additional correction methods test publication write/fsync/link failures,
exclusive/postcommit dispositions, unreadable plans and failed reconstruction,
and late artificial action/query name bounds. The full-path correction controls
copy saved successful world evidence with explicit replay provenance and forbid
response calls; copied charges are not new responses or new resource charges.

Run only the new test module from the preparation source capsule, supplying its
exclusive fixture directory as the positional argument. The capsule includes the
existing `prospective-proposal.json` beside the copied tests. Preparation under
`results/ace_llm_intervention_training_collection_preparation_20261107` and matching
external `ACE-LLM-intervention-training-collection-preparation-20261107-01` preserves
each invocation, exact source/input bytes, outputs, failures, CPU/wall receipt and
hash inventory. This initial capsule is immutable. Corrections use separate
`results/ace_llm_intervention_training_collection_correction_20261107` and external
`ACE-LLM-intervention-training-collection-correction-20261107-01`. Only the four new
`CollectionCorrectionTests` methods run there; the original successful seven and
all older suites are not repeated.
Resource receipts measure test-child CPU/wall only; implementation, inspection and
custody are not fully metered and no whole-cycle cost is claimed. Parent owns
remote work, independent review and scientific adoption.
