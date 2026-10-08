# Publication source snapshot preparation — October 8, 2026

The writing, caption and analysis-interface updates since the frozen B replay
candidate are outside that candidate's manifest. The new
`prepare_delivery_publication_snapshot.py` captures those current publication
bytes in separate private custody without changing the replay input or creating
a purported final release. This is preparation for the final source/report
lineage gate, not another numerical experiment.

## Input and boundary

Supply an independently obtained full forty-character Git commit pin, the
explicit repository root and a new external destination. The command reads
immutable Git objects with replacement objects disabled and verifies commit/tree/blob
object identities. It does not read dirty working-tree publication files, import
the captured tools, regenerate claims, deserialize checkpoints or acquire
responses. The caller must select the intended committed revision explicitly;
using a historical pin intentionally captures that historical writing.

Twenty files are included: the same `paper.tex`, eleven authoritative companions,
claim index, style provenance, required official style license and archived
official template, plus the claim generator, sync utility, table-data checker
and reviewer command guide. A fixed independent file set prevents silently
omitting a companion or notice. The captured sync AST must declare the expected
eleven companions; the embedded manuscript bundle must match their exact bytes.
Recorded official style/notice hashes and claim-generator/index source hashes
must match. Claim values are not recomputed or scientifically certified.

All source files are copied verbatim. The snapshot is private and may contain
author identifiers, attribution, local paths and pending-result prose. No source
guard or notice is removed. It is not a complete runtime/experiment closure: the
generator's original scientific inputs and saved-artifact replay package remain
in their separate custody, and the captured guide is not executed here.

Directory-descriptor-relative operations reject symlink redirection throughout
creation and writing; output is exclusive and outside the repository. Private
custody ancestors must be owned by the current user or root and protected from
group/other writes; root-owned sticky temporary ancestors are permitted, but
the final parent cannot be shared-writable. Malicious same-user/root processes
are outside this ownership boundary; ordinary permissions cannot constrain them.
The macOS descriptor ACL check permits deny-only entries and rejects every
allow entry, including inherited grants, on ancestors and created files/dirs.
The interface is macOS-only; an unexamined ACL implementation fails. Signatures
and constants are from the local Darwin SDK and [Apple ACL documentation](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/acl_get_fd_np.3.html).
Private directories/files use700/600 while writing, then500/400 on completion. These
ordinary read-only permissions discourage accidental edits; they do not prevent
the owner from changing permissions. Inventory SHA256 is the byte-authentication
pin. Retain failed/interrupted preparations; do not overwrite them as successes.

## Future use

```bash
python3.11 -B scripts/research/prepare_delivery_publication_snapshot.py \
  --repo /Users/pat/code/ACE \
  --revision INDEPENDENT_FULL_COMMIT_SHA \
  --destination NEW_EXCLUSIVE_EXTERNAL_SNAPSHOT
```

Run the capture from any working directory by giving the script's absolute path.
This is an internal custody command, not an anonymous reviewer command. It uses
Git and the Python standard library only; no install or compute allocation.

Once supervised exact-runtime B replay and both reporting interfaces qualify,
freeze the integrated manuscript revision and capture a new snapshot. Build a
new final successor manifest with the integrated writing, verified reports and
independent original/replay/report pins. Preserve the replay input manifest and
this preparation inventory as predecessors; neither authenticates later report
bytes. Complete scientific/proof/prose/page-layout, full accounting, deliberate
anonymity/redistribution and human submission approval before public release.

## Verification scope

Nine new fabricated Git-fixture methods pass from an unrelated directory,
including changed HEAD/dirty checkout, independent20-path membership, wrong/missing
pins, absent files, committed symlinks, incoherent companion/style/generator
bindings, output exclusivity, Git commit/blob replacements, post-preflight
ancestor symlink swaps, shared-writable parent rejection, corrupted tree identity
and permissive-umask private/read-only permissions. One real macOS ACL fixture
checks allow/inherited-grant rejection and deny-only acceptance in its own
temporary directory; no user ACL or environment is changed.
These tests execute the new metadata snapshot interface only. No previous
scientific/adapter replay, test suite or completed review is repeated.

Actual pinned publication capture and its hashes/limits are recorded in
`publication_snapshot_preparation_20261008.json`. The distinct implementation
review and disposition are in `reviews/delivery_publication_snapshot_review_2026-10-08.md`.
This capture does not qualify B runtime/results, anonymous original training,
final page layout or public submission readiness.
