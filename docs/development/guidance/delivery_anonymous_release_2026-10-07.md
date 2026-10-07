# Anonymous delivery release preparation

This addresses R7 in the delivery review register. It is private preparation,
not a public release or a declaration that the manuscript is submission-ready.
Stage B and its independent scientific acceptance remain pending.

## Provenance contract

Original registrations, seals, execution records and personal custody locations
stay unchanged in private custody. A public artifact cannot silently substitute
relative paths into those original JSON bytes and retain their historical digest.

The new manifest records a relative artifact location, the released byte digest,
the immutable original digest, the artifact role and the exact derivation.
Numerical arrays, checkpoints, paid input ledgers, path-free fit receipts and
archived learner source are copied unchanged. Selected protocol metadata is a
new JSON object produced by an explicit top-level field projection. Its released
digest differs from its original digest. Receipt bindings referring to the
original protocol are checked against the manifest's **original** protocol digest.
The private derivation record retains original locations and both digests.

The verifier checks bytes, inventory, relative locations, identity/projection
structure and 2,005 receipt-to-artifact bindings in the current A/C candidate.
An independently held manifest digest is necessary for authenticity; changing
both the adjacent checksum and manifest is not an independent proof. Public
readers can verify the released objects and their asserted original-digest
mapping. They cannot reconstruct removed private metadata or prove historical
freeze timing from the anonymous manifest alone.

## Tools and tested preparation

- `prepare_delivery_release_plan.py` makes an explicit private A/C artifact plan
  from complete frozen receipts. It retains all 480 fits, all twelve histories
  including `124753321`, original online weights, paid rows, the exposed grid,
  all eleven physical conditions and all controls. It omits B until acceptance.
- `build_delivery_release.py` creates an exclusive directory and only seals a
  private derivation record after successful verification. Failed/interrupted
  directories are preserved. Originals are never edited or reserialized.
- `verify_delivery_release.py` is self-contained Python using the standard
  library. Its default root is its own location, independent of the working
  directory. It rejects path traversal, symlinks, unexpected files, duplicate
  locations, changed bytes and failed receipt bindings. It never loads pickled
  models, imports the learner, fits models or calls a simulator/network service.
- `reconcile_delivery_fit_snapshots.py` reconciles explicitly named B snapshots,
  checks every available receipt/model pair and input binding, rejects conflicting
  duplicate copies, and retains missing cells. It does not read losses or declare
  scientific acceptance. The current two-snapshot inventory has 560 cells; the
  remaining indices 560–639 are explicitly missing, not filtered out.

The initial private candidate contained **1,151 files**, including all nineteen
archived numerical learner modules (`ace.oracle`, `ace.grid_eval`, etc.). The
manifest digest is
`6b4a6b6c6982c7f3356793ecdc9696c4721aa07cb410d993466382b513faa7ea`.
The verifier passed from `/private/tmp`, outside both repository and release.
It is preserved as the pre-review candidate. The current candidate contains
**1,152 files**, adding a read-only physical replay adapter; its digest is
`aaa236b1de7ca8e387020b3cac60236dd0dc26d831ef437a6452c21b234459fb`.
Eight focused tests now cover conflicting snapshots, traversal/symlinks,
relocation and byte tampering, decoded JSON identifiers, verifier exemption,
original-versus-derived digest bindings and changed acceptance receipt sets.
The planner rechecks the full original A custody audit and the accepted C receipt
bindings, exact condition set and artifact sets before preparation.

For an independently received copy, run:

```sh
python /path/to/release/verify_delivery_release.py \
  --expected-manifest-sha256 DIGEST_FROM_INDEPENDENT_CUSTODY
```

No installation is required for this checksum entry point. This is **not** an
executable refitting or checkpoint replay command. Archived dependencies remain
recorded in the derived protocols. Do not install or modify the shared research
environment for packaging.

The distinct `replay_delivery_physical_release.py` entry point requires C's
recorded dependency environment. From a genuinely renamed package and an
unrelated working directory it reconstructs all 22 neural checkpoint predictions,
all 22 linear coefficient predictions, continuous NMSE and conditional block
bootstrap statistics. Maximum cached prediction discrepancy was **zero**. It
performs no optimizer updates or response acquisition. It does not qualify A
checkpoint replay, model refitting, Stage B acceptance or final anonymity.

```sh
python /path/to/release/replay_delivery_physical_release.py \
  --expected-manifest-sha256 DIGEST_FROM_INDEPENDENT_CUSTODY
```

The result, code hashes and failed attempts are bound in
`results/delivery_release_preparation_20261007/verification.json`. The verifier
reports whether an expected digest was supplied, but explicitly does **not**
attest to its independent origin. The builder's own checksum comparison is a
self-check. Treat the recorded original digests and human provenance review as
separate from integrity verification.

Two adapter failures were preserved, not hidden. First, Python generated an
unlisted import cache inside the bundle: disabling bytecode writes fixed the
read-only adapter. Second, NumPy scalar mean/sd promoted float32 network outputs
to float64, causing a maximum prediction shift of 2.27266e-5 in the first condition.
The frozen worker uses Python float casts; matching that arithmetic yielded
exact saved predictions without loosening tolerances or changing models.

## Anonymity and unresolved submission gates

The automated screen checks research artifacts and decompressed ZIP members for
known personal filesystem/account identifiers. It passed this candidate. It
is a limited screen, not a guarantee of anonymity: author identity can be inferred
from source history, citations, unique hashes, dataset metadata or attribution.
The verifier's own blocked-pattern list is intentionally exempted only when its
bytes match the executing trusted verifier; its digest is in the manifest.
JSON and NDJSON fields are decoded before screening so slash/Unicode escapes
cannot hide the screened identifiers. No candidate is uploaded or publicly published.

Still required before final release:

1. Complete and independently accept the original B chain; include every required
   raw checkpoint, shared response journal, score and failed/interrupted attempt
   disposition using the final composite custody inventory.
2. Include the frozen study workers, generators and actual replay/analysis entry
   points through an explicit derived relocation adapter. Their existing private
   path defaults and hash guards must not be silently edited or bypassed. Explain
   which commands verify cached results, replay checkpoints or refit models and
   which exact dependency environment each needs. A digest-only verifier is not
   sufficient for the final executable scientific reproduction release. C's new
   adapter covers checkpoint and linear prediction reconstruction; A/B adapters
   and the complete study-worker provenance remain outstanding.
3. Bind the complete claim index and original confirmation statistics/accounting
   into the derived release and verify all accepted numerical claims. The present
   package is A/C raw preparation, not the complete manuscript evidence bundle.
4. Deliberately review text, metadata, ZIP member names, binary artifacts, source
   attributions and licenses. Preserve third-party notices and establish archive
   redistribution terms; the archived runner declares MIT in its project metadata,
   which alone does not resolve all notice/distribution questions.
5. Verify the final relocated bundle against independent pinned digests, without
   access to private roots; obtain human approval of anonymity and submission.

These are release/review tasks, not new fitted experiments. Neither the completed
scientific nor reproducibility review required additional fits for the scoped
recipe/accounting paper. Keep the existing study and resource ceilings unchanged.
