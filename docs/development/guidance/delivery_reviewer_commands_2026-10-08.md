# Delivery artifact reviewer commands

Updated October 24, 2026. The original B study is accepted. Supervised replay12
qualified saved-artifact inference against **candidate03**, whose manifest is
`b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9`.
Report02 then completed both the registered and descriptive reports; its accepted
record is `86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10`.
The 19-file assembly is integrated into the same manuscript with 16 companions;
native compilation passed. The final publication successor has a distinct
manifest and source contract. This accepted history does not claim numerical
replay of that final manifest, original training execution or public approval.
Final human release/submission approval remains pending.

The commands below operate on the final publication package. They cover
saved-checkpoint inference and cached analysis, without optimization or response
collection. Do not repeat completed project replays to update this guide.

## Authenticate the release and choose the environment

Obtain `MANIFEST_SHA256` (the **final publication** manifest) and `CHECKER_SHA256`
(the final `verify_delivery_publication_release.py`) from an authenticated release
record through a trusted channel independent of the package being checked.
Neither variable is supplied here; candidate03's digest above is historical and
must not be used as the final pin. An adjacent checksum, helper self-hash or
unverified download does not establish that origin.

Choose absolute reader-local paths: `PACKAGE_ROOT` is the extracted package;
`RECEIPT_ROOT` is an already created, empty directory outside it. Keep the package
read-only and unchanged after authentication. Use a fresh trusted Python process
for every command and the same Bash session for these snippets. No author path,
cluster account, network connection or credentials are needed.

`CHECK_PYTHON`, `AC_PYTHON`, `F_PYTHON` and `B_PYTHON` must each name an existing
**Python 3.11 or newer** executable: verification uses standard-library `tomllib`,
including through replay interfaces. `CHECK_PYTHON` needs only the standard
library. The other environments must have these exact interface-checked versions;
archived runtime records preserve broader metadata. This guide authorizes no
installation.

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

## Authenticate first, then verify the shipped publication

Before executing any package code, including `--help`, authenticate the checker
with trusted checksum code. This inline code uses only the trusted Python
standard library. Stop on any error and preserve stdout/stderr outside the
package; a failed attempt is never acceptance.

```bash
set -euo pipefail
set -o noclobber
: "${CHECK_PYTHON:?Set a trusted Python 3.11+ executable}"
: "${PACKAGE_ROOT:?Set the extracted final package path}"
: "${RECEIPT_ROOT:?Set an empty external receipt directory}"
: "${CHECKER_SHA256:?Obtain the final checker pin independently}"
: "${MANIFEST_SHA256:?Obtain the final manifest pin independently}"
"$CHECK_PYTHON" -I -S -B - \
  "$PACKAGE_ROOT/verify_delivery_publication_release.py" "$CHECKER_SHA256" \
  > "$RECEIPT_ROOT/checker-authentication.json" \
  2> "$RECEIPT_ROOT/checker-authentication.stderr" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required")
path, expected = Path(sys.argv[1]), sys.argv[2]
if not re.fullmatch(r"[0-9a-f]{64}", expected) or path.is_symlink():
    raise SystemExit("Invalid independent pin or checker symlink")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("Checker authentication failed")
print(json.dumps({"authenticated": True, "sha256": actual}))
PY
"$CHECK_PYTHON" -I -S -B "$PACKAGE_ROOT/verify_delivery_publication_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/publication.json" 2> "$RECEIPT_ROOT/publication.stderr"
```

The authenticated checker captures and authenticates **both** local dependencies
against embedded pins **before executing either**, including on `--help`:

- `delivery_prospective_publication.py`:
  `598e9071d5fdb5bebff81679b59578e1b5f16487f1c0584dc417ee6606998db9`.
- `verify_delivery_release.py`:
  `9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52`.

It executes those captured standard-library-only sources, bypassing cached local
imports. Only then does the generic verifier authenticate the final manifest
against `MANIFEST_SHA256` and check package membership/bytes/bindings. Changed
helper pins require a reviewed checker and a new independently trusted checker
pin. All later commands depend on this successful bootstrap and integrity check.

The offline publication check verifies preserved candidate03 lineage, the new
source contract, both accepted report indexes and all 13 outputs, manuscript/16
companion/style/PDF byte bindings, and the reviewed accounting artifact's scope.
It does not rerun models, regenerate statistics, revalidate private execution,
prove original freeze timing, inspect page layout or grant publication approval.
The shipped registered reports can be verified this way without another replay.
The original exporter is included for provenance only: no public original-exporter
command is supported for projected final-package inputs.

The separate archived-macro check remains limited to the 35 A/C/F empirical
macros and their generated LaTeX bytes, including confirmation journal accounting;
it does not reconstruct the new B manuscript claims:

```bash
"$CHECK_PYTHON" -B "$PACKAGE_ROOT/verify_delivery_claims_release.py" \
  --root "$PACKAGE_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  > "$RECEIPT_ROOT/archived-macros.json" 2> "$RECEIPT_ROOT/archived-macros.stderr"
```

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

Replay12 is accepted for candidate03; report02 and the integrated reports retain
that lineage. The final package preserves the old manifest at
`publication/replay_input_manifest.json` and adds its own root manifest and
`publication/source_contract.json`. Do not relabel the old replay receipt as final
package execution, or pass the old manifest pin to verification of the new root.

**Descriptive regeneration requires a fresh, qualified full B replay against the
final `MANIFEST_SHA256`.** Replay12's receipt cannot be reused. The following
reader replay is optional for verifying shipped reports, but mandatory before
running the descriptive reporter below. A pinned receipt or CLI exit alone does
not establish supervised runtime, resource, cleanup and custody qualification.

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

After independently qualifying that fresh execution, record its full receipt
digest as `REPLAY_SHA256` in the external execution/acceptance record. Do not take
this pin from the package or substitute a projected historical receipt. Create a
new report directory through the reporter:

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
These outputs contain no new superiority tests. Fresh outputs remain outside the
package; its manifest does not authenticate added files. The accepted registered
primary/initialization tables are already shipped and verified by the offline
publication checker, separately from descriptive regeneration.

## Qualification and release boundaries

Authenticate this guide through the final publication successor's manifest and
source contract, using the authenticated external release record for the final
manifest/checker pins. Historical candidate manifests do not authenticate this
version. Final integrated scientific/proof/prose/table/page review,
anonymity/redistribution and human release/submission approval remain separate
gates. Offline accounting verification preserves disclosed failures,
measurement scopes and unknowns; it does not establish a whole-sprint CPU total.

Saved-artifact reproduction does not reproduce response collection, optimizer
trajectories or the original complete audit execution. Original worker/source
custody and original/derived relationships remain explicit. No anonymous original
collection/training execution is qualified here; such interfaces need their own
location/import/fixture/resource contracts and verification. Preserve all failed
and interrupted attempts, unknown telemetry and required attribution/notices.

The seven saved-artifact interface component review remains closed. This update
uses only static parsing and source/CLI matching of new or changed command blocks;
no model, replay, reporter, package checker or old test suite was executed.
