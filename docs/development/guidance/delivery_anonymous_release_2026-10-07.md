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

## October 7 attribution replay and frozen-worker provenance follow-up

A new `replay_delivery_attribution_release.py` adapter reproduces the stored A
checkpoints using the original archived learner and exposed grid. It requires
A's exact recorded dependencies, the exact twelve-history membership and all
forty configurations in each history, including classical and matched-CPU
controls. It rejects already cached learner imports: use a fresh process.
Classical control pickle deserialization requires explicit trust after manifest
authentication; the standard-library checksum verifier never deserializes them.

```sh
python /path/to/release/replay_delivery_attribution_release.py \
  --expected-manifest-sha256 DIGEST_FROM_INDEPENDENT_CUSTODY \
  --trust-original-classical-pickles
```

This reconstructs continuous/snapped scores, observed-parent residuals,
predicted-parent chain errors, propagation shifts, eligible support coverage and
quantization-margin diagnostics. It does not fit models, acquire responses,
reconstruct the original online comparator, prove historical freeze timing or
qualify B. Numeric comparisons use fixed rtol1e-10/atol1e-12; the result reports
maximum discrepancy across every numeric score and diagnostic. A fixed first
lexical-history smoke is explicitly partial and cannot qualify the full matrix.

The separate `archive_delivery_worker_provenance.py` utility resolved immutable
Git objects and checked every byte against original A/C protocols and B's frozen
registration before creating exclusive PRIVATE source custody. It preserved
24 bindings: two A worker/guard files, two C worker/guard files, eighteen B workers
and two B generators. Inventory SHA:
`38b70084bad189fafbf2879aa54c056a1bbd9b9ce9bef21d73304bb773e52ccc`.
Eleven files trigger the known identifier screen. Do not copy those files into
an anonymous package unchanged or silently edit their defaults/hash guards.
Thirteen screen-negative files still require deliberate anonymity/license review.
The source archive is not an executable relocation or permission to rerun fits.

The numerical replay adapter is a newly hashed reconstruction implementation;
the original worker and guard byte identities remain independently recorded.
The original protocol checks were not bypassed. Full confirmation/claim/attempt
packaging, complete B composite custody and analysis/replay, explicit anonymous
worker relocation, redistribution review and final human approval remain gates.
The bounded adapter review and dispositions are in
`reviews/delivery_attribution_release_review_2026-10-07.md`.

The latest candidate supersedes earlier candidate references above:1,153files,
2,005bindings, SHA63caf4b16757f6aa79c5cf9b1d4452c0f8dfd5b2c45c377af68fdef9605ed28c.
The final full480replay reports zero discrepancy across all numeric results.
Its compact receipt is `results/delivery_release_preparation_20261007/attribution_verification.json`.
This closes A checkpoint/diagnostic reconstruction, but not the original online
comparator or full confirmation reproduction. The candidate remains private.

## Confirmation and empirical claim follow-up (October 7)

`extend_delivery_confirmation_release_plan.py` adds the full original twelve-case
seal (240 files), all original online weights and three-init delivery checkpoints,
raw complete response journals, the distinct interrupted attempt, original/prior
terminal accounting projections, and manuscript claim receipts/companions.
Original source/protocol/case bytes are preserved. Approval fields naming humans,
threads or private custody are omitted by explicit metadata projection. Such
projections remain newly hashed assertions; they do not prove historical approval
or freeze timing to an anonymous reader.

`replay_delivery_confirmation_release.py` requires recorded torch/NumPy/SciPy
versions and a fresh process. It reconstructs 48 model sets (288 neural heads),
both chain and flat scores, the archived paired log-ratio statistics and the
independently recorded ratio/interval/sign-flip result. All comparisons pass at
fixed rtol1e-10/atol1e-12; maximum numeric difference is 5.55e-17. It runs from a
renamed bundle and unrelated directory, using one CPU thread, without optimization
or new responses. Inference/checksum/import child CPU was 40.519136 seconds,
elapsed 41.587096 seconds, peak child RSS 463,060,992 bytes. No accepted A/C
inference was repeated to produce this new confirmation verification.

```sh
python /path/to/release/replay_delivery_confirmation_release.py \
  --expected-manifest-sha256 DIGEST_FROM_INDEPENDENT_CUSTODY
```

The accounting review establishes 60,962 charged attempts, 57,636 persisted
responses across twelve complete histories, and 3,326 interrupted reservations
whose returned-response count is unknown. The adapter recomputes these from
thirteen distinct journals; retained copies are counted once, and startup/executed
summary counters and refit initializations do not add acquisition calls.
See `reviews/delivery_confirmation_accounting_review_2026-10-07.md`.

`verify_delivery_claims_release.py` is an independent standard-library metadata
entry point for the current empirical macros and their LaTeX bytes. It uses the
stored A history contrasts, worsening diagnostics, initialization sensitivity,
physical condition errors/acceptance, confirmation statistics and disjoint
attempt accounting. It does not replay models or read B losses. Full prose/table
review and final B claim integration remain distinct gates.

```sh
python /path/to/release/verify_delivery_claims_release.py \
  --expected-manifest-sha256 DIGEST_FROM_INDEPENDENT_CUSTODY
```

`results/delivery_release_preparation_20261007/confirmation_verification.json`
binds the first complete confirmation replay and unchanged private derivations.
The later claim package changes only the generated macro/index/generator objects
and adds the new accounting receipt and macro checker. All 280 confirmation
replay/input/source objects are unchanged, preserving the qualified inference
byte identities without another model replay. This is private preparation;
full B custody, all-sprint compute/attempt accounting, anonymous worker relocation,
redistribution/anonymity and human submission approval remain outstanding.


### Latest verified private candidate

Candidate09 has1,423files/2,533bindings, manifestSHA
f49733578564e3b8e2e302bc1e41c282dead7a630beaf8764bf1226cbdb5a4fa.
All280 confirmation inference/input/source/dependency objects from candidate08
remain unchanged. The unrelated-directory check reconstructs all35current
empirical macros and verifies generated LaTeX bytes; its compact receipt is
`results/delivery_release_preparation_20261007/confirmation_claims_preparation.json`.
The accounting/implementation reviews are complete, with no remaining required
implementation fixes. The same manuscript compiled after the scoped disclosure
and provenance update. All-sprint accounting, full B replay and claim integration,
anonymous frozen workers, redistribution/anonymity and human submission review
remain gates. The package is private and must not be uploaded as a final release.


## Prospective source/replay-core preparation

The separate B preparation procedure is now
`delivery_prospective_replay_preparation_2026-10-07.md`. It preserves twenty frozen
worker/generator bindings and derives an explicitly separate fourteen-segment
numerical core with source-text/AST provenance. A reviewed preflight checks
original successful full acceptance through a hash-bound metadata projection,
without parsing its scientific analyses, and requires exact registered runtime.
Thirteen focused synthetic checks pass; the actual partial custody and local
runtime are correctly rejected. Five review findings are fixed and disposed in
`reviews/delivery_prospective_release_core_review_2026-10-07.md`. Evidence is
`results/delivery_release_preparation_20261007/prospective_core_preparation.json`.
This does not yet implement the full B replay/relative-artifact plan or qualify
inference. Existing A/C/confirmation candidate09 is unchanged. Continue full
B adapter preparation while the original chain waits; final raw custody, source
relocation, accounting, anonymity/redistribution and human approval remain gates.

## Prospective full-adapter preparation follow-up

Complete composite reconciliation, relative plan assembly and scientific replay
are implemented as separately hashed supplemental tools.26focused checks pass;
no B package is assembled from the current560fits. Actual partial-custody probes
reject at original completion before source/outcome access or output. Positive
full assembly and all640checkpoint replay remain unqualified. Original45ebeb89
workers/guards and the qualified A/C/F candidate09 remain unchanged.

Read `delivery_prospective_replay_preparation_2026-10-07.md` and
`reviews/delivery_prospective_release_replay_review_2026-10-07.md`. New metadata
projections preserve original private bytes, expose distinct original/derived
hashes and keep absent per-attempt elapsed telemetry unknown. An authenticated
manifest binds these derivations; it does not authenticate historical freeze
order or erase archive/source license and final anonymity obligations.

## October 7 21:09 UTC — positive interface and redistribution review

A synthetic full2477-file B fixture now passes actual planning/build/verification
and relocation with independent membership/identity routing checks. This does not
qualify real B custody or numerical replay. See the prospective replay preparation
document and `prospective_assembly_preparation.json`; candidate09 is unchanged.

A distinct local redistribution/anonymity review establishes the pinned Chambers
README's CC BY4.0 dataset and MIT software declarations, including the generator's
Gamella notice. Notice packaging is still absent from candidate09. Runner metadata
declares MIT but an authoritative copyright notice remains unresolved. An extra
organizational text scan found identifying authors/repository metadata in archived
`source/runner/pyproject.toml`, beyond current screen patterns. Keep the F original
source binding intact; a future metadata projection must separately bind original
and derived bytes and retain required attribution. Do not silently delete it or
remove attribution to obtain anonymity. `redistribution_preparation.json` and
`reviews/delivery_redistribution_review_2026-10-07.md` record these release gates.
No public upload is authorized by byte verification or the synthetic test.
