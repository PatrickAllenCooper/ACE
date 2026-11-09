# Downstream adapter gate handoff — intermediate, not packet-ready

New source `implementation/compact_inputs.py` provides opaque input admission against the now coordinated remote-custody API. It authenticates independent canonical receipt bytes, expected roots/helper/full inventory, exact selected/omitted partition and selection policy; invokes fresh full backup verification; checks both selected local mirrors without changing receipt.local_root; accepts only independently pinned flat launch/inputs members; retains fetch failures and forbids retry; staging is exclusive and disjoint. No actual backend or remote action was executed. Parent/consumer must authenticate helper source/Git/review and operational amendment before constructing this gate.

Tests:8new fabricated methods passed once in attempt01 (wall0.059709s, childCPU0.051773s,RSS16400384bytes); selection-policy delta adds1new method passed once in attempt02-selection-delta (wall0.037552s,childCPU0.034192s,RSS16154624bytes). Prior8were not repeated. All original sources/transcripts retained. These are transport-boundary fake tests, not actual archive/custody/runtime qualification.

Exact current source:

- `compact_inputs.py`: `7d140427a2e7626931ba1f9b1624c49332a677ee282fdff7192c764c865a4e5f`
- `test_compact_inputs.py`: `2a608ad6739d35e4674562349ed53361c5657e27cc84606e33c993c5edab3308`

CPU_ACCEPTANCE_KEYS is exactly16keys in both inherited GPU/finalizer builders; it remains unchanged. The general core fetch_exact will permit all independently pinned safe members needed by future saved-evidence auditors, while this builder adapter intentionally limits its demand to launch/inputs. Audit callbacks must use verified remote full inventory for omitted files or fetch actual bytes; never fabricate local presence.

Required lifecycle correction sent to Kuhn: helper verify_backup must preserve immutable archive exactness without unconditionally requiring current scratch roots to equal a predecessor snapshot after authorized policy05/06/07 or finalization appends. Producer collect still requires full source-before/after equality. Caller’s existing stage-specific live preservation rules remain necessary.

Builder/launcher/collector derivatives are not yet wired. Files under implementation/gpu and implementation/finalizer are unchanged predecessor copies, not new qualified coordinators. Pending final helper pin plus explicit outer storage-amendment/ancestor-reference consumer binding. No new future acceptance, actual packet, upload, submission or scientific output exists.
