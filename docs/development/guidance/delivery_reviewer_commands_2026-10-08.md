# Delivery artifact reproduction commands

Internal preparation for the anonymous review release. These instructions cover
saved-checkpoint inference and cached analysis, with no optimization or response
collection. The original prospective study is accepted; its separate target
runtime replay and final release approval are still pending. Do not distribute
the private candidate or execute these templates as another preparation replay.

## Authenticate the release and choose the environment

Obtain the final manifest digest and verifier digest from the approved release
record, independently of the extracted package. Authenticate the verifier before
executing it, then use its manifest check to authenticate the other entry points
and artifacts. A checksum from the same untrusted download does not establish
that independent origin. Keep the extracted package read-only, use a fresh Python
process for each command, and write receipts in a new directory outside it.

The examples use absolute paths chosen by the reader. No author directory,
cluster account, network connection or credentials are needed for saved-artifact
inference. `PACKAGE_ROOT` is the extracted package and `RECEIPT_ROOT` is an
already created, empty directory outside it. `MANIFEST_SHA256` comes from the
final approved release, rather than the current private preparation manifest.
`AC_PYTHON`, `F_PYTHON` and `B_PYTHON` name existing Python executables in the
corresponding environments. `CHECK_PYTHON`, `AC_PYTHON`, `F_PYTHON` and
`B_PYTHON` must each be Python 3.11 or newer:
the verifier uses the standard-library `tomllib` module, including when called
by a replay interface. The byte verifier uses `CHECK_PYTHON`, which needs
no third-party packages. The version requirements below are those checked by
the interfaces; the archived runtime records also preserve broader environment
metadata. No environment installation is authorized by this guide.

- A: torch 2.5.1, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6 and
  scikit-learn 1.9.1, as recorded in `A/protocol.json`.
- C: torch 2.5.1, NumPy 2.4.6 and scikit-learn 1.9.1, as recorded in
  `C/protocol.json`. The A environment satisfies these checked C versions.
- Confirmation F: torch 2.5.1, NumPy 2.4.6 and SciPy 1.17.1, as recorded in
  `F/runtime.json`. The A environment satisfies these checked F versions.
- B: torch 2.9.1, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, SymPy 1.14.0 and
  PyYAML 6.0.3. The A/C/F environment cannot qualify B. The B adapter checks
  actual loaded module versions and origins in addition to distribution metadata.

The full historical F environment inventory includes unused cloud SDK packages.
It is not a requirement to use those services or install every recorded package
for the F saved-checkpoint interface. The interface loads the archived learner
source shipped under `source/runner`; installing a different learner release is
not a substitute for those authenticated bytes.

## Verify bytes and current archived claim macros

After independently authenticating the verifier, run from any working directory:
use the same Bash session for the snippets below. Stop on the first error and
preserve stderr alongside the receipt; a failed attempt is never acceptance.

```bash
set -euo pipefail
set -o noclobber
"$CHECK_PYTHON" -B "$PACKAGE_ROOT/verify_delivery_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/integrity.json" 2> "$RECEIPT_ROOT/integrity.stderr"

"$CHECK_PYTHON" -B "$PACKAGE_ROOT/verify_delivery_claims_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/archived-macros.json" 2> "$RECEIPT_ROOT/archived-macros.stderr"
```

Check exit codes as well as the JSON results. Byte verification checks artifact
membership, digests and relocation bindings. The macro interface reconstructs
the current 35 A/C/F empirical macros and their generated LaTeX bytes, including
disjoint confirmation journal accounting. Neither command replays checkpoints,
refits models, proves historical freeze timing, reviews anonymity, verifies every
prose/table statement or reconstructs final B manuscript claims.

## Replay archived attribution, physical prediction and confirmation

Use separate fresh processes. Full attribution includes classical regressors
stored as authenticated original pickles; the explicit trust flag is required
for that deserialization. `--smoke-first-history` would test only one history and
must not be reported as full attribution reproduction.

```bash
"$AC_PYTHON" -B "$PACKAGE_ROOT/replay_delivery_attribution_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --trust-original-classical-pickles \
  > "$RECEIPT_ROOT/attribution.json" 2> "$RECEIPT_ROOT/attribution.stderr"

"$AC_PYTHON" -B "$PACKAGE_ROOT/replay_delivery_physical_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/physical.json" 2> "$RECEIPT_ROOT/physical.stderr"

"$F_PYTHON" -B "$PACKAGE_ROOT/replay_delivery_confirmation_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/confirmation.json" 2> "$RECEIPT_ROOT/confirmation.stderr"
```

Full A reconstruction covers all 480 saved configurations, metrics, mechanism
diagnostics and quantization margins on the exposed grid. Original online weights
are replayed by F, rather than the A interface. C reconstructs 22
neural and 22 linear coefficient predictions across eleven conditions, their
errors and conditional block-bootstrap summaries. F reconstructs original online
weights and three delivery initializations for all twelve histories, paired
statistics and distinct acquisition-attempt accounting. Floating-point differences
are compared with each adapter's declared tolerance; numerical equality need not
mean bitwise equality. Retain an error exit and partial stdout as a failed attempt,
not a successful receipt. No accepted scores are overwritten by these interfaces.

## Replay prospective checkpoints and prepare descriptive reporting

These are future independent-reader commands. Current project qualification must
use the frozen, supervised `ace_delB_replay` launch03 under the existing resource
authorization. Directly running the CLI or inspecting a pinned receipt does not
replace that supervisor, allocation and custody gate.

```bash
"$B_PYTHON" -B "$PACKAGE_ROOT/replay_delivery_prospective_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --receipt "$RECEIPT_ROOT/prospective-replay.json" \
  > "$RECEIPT_ROOT/prospective-replay.stdout" 2> "$RECEIPT_ROOT/prospective-replay.stderr"
```

The successful full receipt must bind the same manifest, exact runtime, all 640
checkpoints, 240 primary cells, 48,000 cached responses, and zero new updates and
responses. It reconstructs the frozen four-test Holm analysis and descriptive
results without selecting systems, histories or scored initializations. Fresh
checkpoint predictions are compared with cached predictions; endpoint scores
and primary statistics then use the original cached predictions. A new
exclusive receipt outside the package is required; existence of a JSON file is
not evidence that its execution qualified.

After qualifying and independently recording the replay receipt digest as
`REPLAY_SHA256`, create a new report directory through the reporter:

```bash
"$CHECK_PYTHON" -B "$PACKAGE_ROOT/prepare_delivery_prospective_supplement.py" \
  --root "$PACKAGE_ROOT" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --replay-receipt "$RECEIPT_ROOT/prospective-replay.json" \
  --replay-sha256 "$REPLAY_SHA256" \
  --destination "$RECEIPT_ROOT/prospective-descriptive" \
  > "$RECEIPT_ROOT/prospective-descriptive.stdout" 2> "$RECEIPT_ROOT/prospective-descriptive.stderr"
```

The reporter checks pinned upstream closure, replay receipt counters and package
byte integrity before decoding the package's scientific score and acceptance
artifacts. The pinned replay receipt, including its analyses, is parsed earlier.
It does no neural inference or runtime
qualification. Its JSON/CSV outputs retain all 640 cells, 320 fixed-init0 arm/history
errors across 80 histories, 80 system log ratios, ablation/history summaries and
floor counts. Two generated LaTeX files contain fifteen descriptive tables.
`output_index.json` binds the generated objects and their source/input lineage.
These outputs contain no new superiority tests. The original registered claim
exporter and its primary/initialization tables are still separately required for
manuscript integration; the macro checker above does not cover them.

## Evidence still required for submission

Actual target-runtime supplemental execution, new report/source bindings and
final integrated claim/proof/prose/table/visual review remain open. The release
must incorporate this guide as a newly hashed artifact and provide final approved
pins; this repository document is not authenticated by the existing candidate
manifest. Exact commands have been checked statically against the entry-point
schemas, not executed as new study replays.

Saved-artifact reproduction does not reproduce response collection, optimizer
trajectories or the original complete audit execution. Historical worker/source
custody is preserved privately with explicit original/derived relationships.
Anonymous executable collection/training interfaces, if claimed, require their
own location/import/fixture/resource contracts and verification; the commands
above do not supply that evidence. Preserve all failed and interrupted attempts,
unknown telemetry and required source/data attribution. Final anonymity,
redistribution and human submission approval remain separate gates.
