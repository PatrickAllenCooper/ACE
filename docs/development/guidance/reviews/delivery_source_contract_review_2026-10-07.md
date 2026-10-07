# DISTINCT bounded source-contract review — 2026-10-07

The latest contract has no custody, original-source membership or projected-digest defect identified for its pinned candidate12 input. Its A/C/F scope, future B supersession dependency and ACE notice binding are now explicit. One P2 issue remains in the new preparation-tool provenance field: it hashes the source file at the end of the run rather than binding the executing source snapshot. This review does not authorize a B extension or change its acceptance prerequisites.

## Captured evidence and boundary

Final implementation and test snapshots captured at `2026-10-07T23:09:19.326400+00:00`:
`scripts/research/extend_delivery_source_contract.py`, SHA256
`6d58a4133b4b2d87971fe4ed3b545fe41bcb3e689ffdc9121e2f0d0e33571f69`.
`scripts/research/test_delivery_source_contract.py`, SHA256
`69a83472f36e62f03ef6993d885a9416d80d987171bc448808c7e6cef652306d`.
Line references below describe these snapshots, not subsequent edits.

Authenticated candidate12 plan SHA256:
`f2217b20a73d7ba2e47df881a3bfa26c8f1e4c27181c0e1c79929ca4eb6c2551`.
Authenticated candidate12 relocated manifest SHA256:
`4d1ea0f96a08593c2c7350eb9fe6cef0e106c5f4cd3551c010b1e6177db9628f`.
Only metadata and source bytes were inspected. The prior frozen-worker review
was context, authenticated by its existing `cbbe51…d12ea` pin. The latest
owner-confirmation ledger records Runner MIT authority RESOLVED; no licensing
finding is reopened.

## Actionable finding

### P2 — Bind the executing preparation source rather than a late disk hash

At [line 173](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:173),
`preparation_tool_sha256: sha(__file__)` rereads the file after the contract and
output plan have already been written. Python continues executing loaded code if
that file is subsequently edited. An edit during a run therefore causes the
derivation to attribute its outputs to different source bytes. Unlike the input
and projection snapshots, this field has no captured-source/executing-code check.

Fix: capture the preparation source at module startup, confirm that compilation
matches the executing module, and record the captured digest in the derivation.
The existing verifier's import-time pattern at
[lines 15–20](/Users/pat/code/ACE/scripts/research/verify_delivery_release.py:15)
is a local example. Add a synthetic source-drift regression demonstrating that a
later disk edit either rejects execution or leaves the recorded executing digest
unchanged. This is a source-attribution issue; no mismatch in the owner's actual
builds was observed or established by this review.

## Latest changes verified

[Line 120](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:120)
now expressly scopes the contract to A/C/F preparation and historical B custody.
[Line 128](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:128)
requires a new contract digest before B inclusion and names the identity design
helper and derived core/interface dispositions. This resolves the misleading
earlier instruction to retain all inherited dispositions unchanged.

The future dependency is concrete: the existing B planner at
[lines 419–421](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:419)
includes the original `delivery_prospective_design.py` helper. Its inherited-plan
copy and final writes at lines 333–334 and 495–502 do not implement source-contract
supersession. Before that planner is used with this contract, author a successor
record with the helper's released path/digest and identity role, updated B scope,
and replacement active manifest bindings. Preserve original source identities
and predecessor evidence. A synthetic integration regression should reject stale
private-only/B-false assertions alongside this helper. No B planner change or
outcome read is required now.

ACE notice bytes are authenticated against the fixed existing source-notice pin
at lines 92–93. Every one of the 24 worker records references
`notices/ACE_APACHE_2_0.txt` at line 101; the top-level digest at line 124 is bound
to that same artifact's released SHA256 at lines 166–167. The notice is captured
as an identity object with the other inputs at lines 139–141. Actual read-only
construction confirmed all 24 references and the notice digest against
candidate12's independently pinned manifest. This records the existing ACE source
notice; it neither replaces Runner's separately bound MIT notice nor changes
original worker bytes.

## Projection, custody and interface checks

The original identity-only assumption was corrected during the review. Updated
[58–71](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:58)
reconcile every claimed original copy, capture and authenticate one original byte
snapshot, and parse that snapshot to reproduce only the four named protocol/runtime
JSON projection locations. Interface and notice projections are rejected.
The serializer matches the builder's `write`: indent 2, sorted keys,
`allow_nan=False`, trailing newline. Contract runtime hashes at line 117 and
manifest bindings at lines 161–163 use these released bytes. Historical original
digests remain in the inherited entries and receipts.

Actual metadata-only reconstruction matched candidate12's manifest for all four records:

- A protocol: original `a454e59…ef169`; released `a9a7022…49fc3`.
- C protocol: original `35ac5cd…16336`; released `ed5592b…0abe`.
- F protocol: identity `31f0ea7…04dd1`.
- F runtime: original `bd894a7…83b1e`; released `b18bed4…15260`.

An additional synthetic probe confirmed one capture call and rejected projections
of an adapter and notice. `reconcile` performs separate custody hash reads; “one
snapshot” accurately describes the bytes used for parsing and derivation, not a
claim that all sources are read only once.

Calling only `snapshot_entry` and `make_contract` on authenticated candidate12
metadata produced 24 private-original bindings and five interfaces. All 24 archived
source files matched their pinned inventory digests, with 22 distinct identities;
the A/C/B guard copies remain separate binding edges. All eleven interface/runtime/
Runner/ACE-notice contract digest values matched the inherited manifest. No original
worker digest was included among candidate12 plan artifacts.

The five advertised entrypoints exist and accept the stated root/manifest arguments.
A requires the separately recorded `--trust-original-classical-pickles` argument;
the recorded base command must be combined with it. A/C/F descriptions correctly
identify saved-checkpoint inference/statistics, rather than collection, fitting or
original-worker execution. The macro interface excludes B and full prose/tables.

## Actual membership and unsupported-role assumptions

Candidate12 metadata contains the exact twelve A histories and forty expected
configuration labels per history: 480 model artifacts with corresponding receipts.
C has exactly the eleven condition names from its captured protocol, with each
condition's model, coefficients and predictions. F has the same twelve histories,
36 delivery model sets and twelve online flat checkpoints; these are artifact
membership checks, not checkpoint deserialization or numerical qualification.
Candidate12 contains no `B/` artifacts.

The constructor itself checks the 24 original source edges and the roles of the
five named interfaces; it does not validate the empirical matrix or reject all
additional artifact/entrypoint roles. Synthetic in-memory probes accepted a
`B/scores.json` metadata row and an extra `original-fit-worker` row while retaining
`B_included: false`. Neither probe read or created a score file. The fixture at
[14–28](/Users/pat/code/ACE/scripts/research/test_delivery_source_contract.py:14)
also succeeds without empirical matrix artifacts. Thus current truthfulness
depends on the independently pinned, known candidate12 input, not on a general
scope validator in `make_contract`. If the tool is reused for another plan, pin
the supported inherited plan or explicitly validate its scope before stamping
these flags; reject unsupported B/helper/training roles until a successor contract
exists. Do not describe this function's tests as full empirical membership validation.

The inherited `source/release/generate_delivery_claims.py` has the explicit
`original-claim-generator` role and is not advertised among the five interfaces.
Its repository-layout assumptions and writes do not make it a supported relocated
entrypoint. Its presence is historical source provenance, not an additional
qualified command.

## Verification and limitations

All six captured tests passed. To obey the sole permitted-write boundary, the
new tempfile projection test ran with an in-memory filesystem and custody
reconciler; this is not a claim that its real-filesystem variant ran. Separate
read-only checks used the actual custody reconciler on the pinned originals.
The new projection/corruption test at lines 64–78 addresses the initial regression;
the other tests cover missing/repeated source edges, included-original digests,
notice status and wrong named-interface roles.

The owner reports candidate13 build/renamed verification passed with 1431 files,
2548 bindings and all 1430 inherited manifest entries unchanged. That is owner
execution evidence, not a build run by this reviewer. The final candidate14 build
remained pending at the review boundary.

The owner performs the exclusive new metadata build and relocation separately.
This review did not call `extend`, build/relocate a candidate, run any old A/C/F
replay/review/fit, execute an original worker or B planner, read B scores, submit
jobs, install packages, or mutate Git. It establishes neither new scientific
acceptance nor historical freeze timing, anonymous original-worker execution,
anonymity certification or public approval. The only authored file is this review.

## Final static recheck — appended, 2026-10-07T23:11:37Z

This appendix supersedes the earlier current-status assessment while retaining
the observations and limitations above. It reviews only the requested base-pin,
scope/role guards and corrected test fixture; no additional implementation scope
or candidate build was undertaken.

Final tool SHA256:
`ee43f36bceadc683b3d24bf0195c473caaa3c818f8dab308f693244e33594ed9`.
Final test SHA256:
`f31203ec6d083abc66bfd4a9b0fdf79ab26ef8b7c23bdbf1e4d1b96f0138b98a`.
The source tool hash was unchanged between the final guard review at
`2026-10-07T23:11:12.268959+00:00` and the closing test capture at
`2026-10-07T23:11:37.211749+00:00`.

### Earlier snapshot findings and their disposition

The initial tool snapshot at `2026-10-07T23:04:39.447401+00:00` had SHA256
`897cd01ea9dc6cc0ff16197f7e2d9fdb7c2f5cadef09a48231c3887f3a62cbdb`;
the original five-test file had SHA256
`9b96ff17a45134850227068a799246d98e6849aa3dccfe29566f46a006b04d22`.
Its identity-only `snapshot_entry` at then-lines 57–62 rejected the inherited
project-json protocol/runtime entries. The owner preserved the failed preflight
and reports no outputs were created by it. The `5c3bb93…fe1634` correction
introduced the captured-original projection path; its actual derived hashes and
manifest comparison are recorded above. **Resolved for the current contract.**

Earlier synthetic probes demonstrated that an arbitrary plan could add B or
unsupported execution metadata while the constructor still stamped A/C/F status.
The earlier future-B wording also invited retention of all-private dispositions
after original helper inclusion. The notice/scope revision `6d58a413…571f69`
made the successor dependency explicit; the latest guards now constrain current
reuse as described below. **Resolved for the current preparation tool.** Future
B source-contract supersession remains an explicit unimplemented prerequisite;
no B planner integration is claimed.

### Latest guard changes

[Line 18](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:18)
hard-pins candidate12. At
[lines 138–140](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:138),
`extend` rejects an unknown expected pin before reading any caller-supplied path,
then authenticates the captured plan against the supported digest. A supplemental
mocked-read probe recorded zero `captured` calls for an unknown pin. This removes
the earlier assumption that any independently hashed base plan was supported.
The contract records that base digest at line 126.

[Lines 79–82](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:79)
reject `B/` artifacts, the three named B helper/core/replay paths and the original
fit/collection/full-audit roles before contract construction. The new synthetic
cases in
[lines 64–72](/Users/pat/code/ACE/scripts/research/test_delivery_source_contract.py:64)
exercise stale B inclusion, the identity helper, unsupported fitting and unknown
pin rejection without opening any B outcome. Full empirical matrix verification
still belongs to the earlier authenticated candidate12 preparation and numerical
adapters; the hard base pin now preserves the specific membership inspected here.

The first eight-test snapshot, SHA256
`b7cc1ed81b3ce63d50c986f7e1caa6cf5d17af118fa7b2a662149bbb6f50ca5e`,
had one failure: `test_wrong_adapter_role_rejected` used `original-fit-worker`,
which the new earlier guard correctly rejected with `unsupported original
execution role`, before its expected `role mismatch` branch. The final fixture at
[line 60](/Users/pat/code/ACE/scripts/research/test_delivery_source_contract.py:60)
uses `wrong-adapter-role`, so it continues to exercise the named-interface guard.
**Resolved.** All eight final captured tests passed, with the projection tempfile
test still using the explicitly disclosed in-memory filesystem/custody substitute.
No test files were written.

ACE notice authentication, all 24 source-notice references, the identity notice
digest binding, A/C/F scope and mandatory future-B-successor wording remain
correct in the latest diff at tool lines 97–98, 106, 125–133 and 173–174. Their
earlier authenticated byte and membership checks remain applicable; no new
numeric inference or outcome access was necessary.

### Remaining current fix and final disposition

The sole remaining P2 from this review is the preparation-tool provenance
snapshot issue, now at
[line 180](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:180):
`preparation_tool_sha256` still uses a late `sha(__file__)` read. Apply the
captured-executing-source fix and source-drift regression described above before
treating this field as proof of which implementation produced a derivation.
The base pin and scope guards do not address that independent field. No actual
candidate provenance mismatch was observed; this finding does not invalidate
the independently checked interface/runtime/notice digests.

No required current custody, membership, role, ACE/Runner notice or projected-digest
fix remains identified. Candidate15 build/relocation is owner work in progress
at this boundary; its success, final manifest and inherited-entry comparison are
not asserted by this static review. Earlier candidate13/14 prototypes and failed
intermediate checks remain owner evidence. The sole authorized output remains
this report, with no jobs, installs, Git mutations, original-worker execution,
old numerical replay or B score access.

## Last exact-hash disposition — appended, 2026-10-07T23:13:22Z

This limited recheck closes the sole remaining P2 above. Captured tool SHA256:
`7cd870249e36dfa0ef71c7b2b41c0df74c02e038bf25f58188bc61695b55a766`.
Captured test SHA256:
`4a4a66434df342b1498ce047bbfb6b3334fb8d00e5e0b6da17daa068df669d3d`.
Both were captured at `2026-10-07T23:13:22.461146+00:00`.

At [lines 20–23](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:20),
one import-time source snapshot is compiled with `dont_inherit=True` and compared
with the executing module's code object. A mismatch rejects execution.
`PREPARATION_SHA256` hashes that same snapshot, and
[line 188](/Users/pat/code/ACE/scripts/research/extend_delivery_source_contract.py:188)
records this constant instead of rereading the source pathname at completion.
This implements the requested provenance fix. **P2 resolved.**

The new regression at
[lines 75–83](/Users/pat/code/ACE/scripts/research/test_delivery_source_contract.py:75)
executes old compiled code against an altered source snapshot and expects the
implementation-mismatch error. Its alteration changes a compiled docstring
constant, so it exercises the actual code-object comparison. A focused in-memory
probe independently confirmed both the final executing digest and rejection
when a source constant changes; it wrote no files. No full suite or real-tempfile
regression run is asserted in this last recheck.

**No required current fix remains identified within the authorized static
review.** Earlier observations remain preserved above as snapshot history.
Candidate15 is preserved and candidate16 implementation qualification is owner
work, not a result established here. Future B successor-contract and scientific
acceptance prerequisites remain unchanged. No numerical replay, B score access
or other review scope was added; this report remains the sole authored file.
