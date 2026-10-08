# Bounded review of the new prospective supplement — 2026-10-08

**Disposition: no remaining findings in the latest reviewed source; no required
code fixes.** Numeric reporting, the descriptive table additions, and the revised
output path/missing-contract guards pass bounded fabricated checks. No real
Stage B outcomes were accepted, opened, or qualified in this review.

## Scope and reviewed version

Reviewed only `scripts/research/prepare_delivery_prospective_supplement.py` and
`scripts/research/test_delivery_prospective_supplement.py`, against
`docs/development/guidance/delivery_final_integration_2026-10-08.md`, including
the subsequent descriptive LaTeX additions, typed primary initialization,
ancestor-symlink rejection, canonical destination containment check, and explicit
missing-B-contract rejection. The final six-test run and additional guard probes
ran from `/tmp` with the research directory added to the import path.
The reviewed source SHA256 values are:

- Supplement: `d7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24`.
- Tests: `d53b2ef8d636170f0f99712e4e8a4cf70ef8832349f53f6eaf272cde1469774e`.

The final recheck was limited to the updated guard/regression fixture and the
six existing unittest methods. All six passed from `/tmp` in 0.509 seconds;
both source hashes were unchanged across that run. The earlier scientific
checks below are retained, without repeating the whole scientific review.

Imported gate, byte-integrity, path, and fabricated-summary interfaces were read
only to establish how the new module calls them. This does not repeat source,
core, accounting, replay, or theory reviews, and does not adjudicate the parallel
custody work. All executions used temporary fabricated fixtures. There was no
CURC, network, installation, model loading, original worker execution, response
acquisition, actual fit, or actual numerical replay. Only this report was written
in the repository; no code was edited or committed.

## Findings and resolved issue

No required implementation fix remains in the latest version. An earlier
destination-path finding was reproduced against the earlier implementation:
an external parent symlink into the input package bypassed its lexical
outside-package check and added unmanifested output files. That finding is
**resolved** by the latest guard at lines 150–153, which rejects symlinks in
root/destination ancestors and compares resolved input/output locations before
reading metadata or writing output.

The initial reproduction used `base/alias -> base/root` and destination
`base/alias/supplement`, with only the original metadata gate substituted for
the fabricated contract. Export succeeded inside `base/root` and added nine
unmanifested files; the existing input-file hashes stayed unchanged. This is
retained evidence about the initial implementation, whose lexical guard lacked
ancestor-symlink rejection and resolved containment. It is **not** a reproduction
against source SHA256
`d7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24`.
That captured current source already contains both corrections and passes the
fresh rejection probes below. The initial report's still-vulnerable description
was stale following the concurrent source update and has been corrected here.

Fresh fabricated probes verified rejection of an external destination-parent
symlink into the input, traversal components resolving a destination into the
input, and a root with a symlinked ancestor. All three rejected before any
`snapshot()` call and created no supplement under the input. A normal external
destination still passed, retained the entire input inventory and hashes, and
refused overwrite. These are filesystem checks, with no scientific test or real
outcome access. The updated positive test at lines 113–124 now permanently
checks destination-parent symlink rejection, traversal into the input, a direct
input-descendant destination, and identical complete input inventory/hashes after
each rejection and after normal export/overwrite refusal. It also includes a
manifest-bound interrupted/failed attempt sentinel. These additions passed in
the final bounded recheck. The earlier separate root-ancestor and
missing-contract probes remain supporting evidence.

## Gate and provenance behavior

The production export path first verifies the independently supplied manifest
SHA256, snapshots the manifest-bound replay contract, and calls the original
metadata `gate()` at line 159. Only afterward does it snapshot/decode the
independently pinned supplemental receipt. It requires full replay, the exact
recorded dependency map, 640 replayed checkpoints, 240 recomputed primary cells,
48,000 checked cached responses, zero optimizer updates and zero new responses.
The receipt must identify the same manifest and state that an expected digest
was supplied. Byte integrity then precedes decoding `B/scores.json` and
`B/acceptance.json` at line 170.

The new membership check at line 158 explicitly rejects a manifest without
`B/replay_contract.json` before receipt parsing or any score/acceptance access.
A fabricated missing-contract probe instrumented snapshots and the gate: only
`manifest.json` was read, the gate was not called, and no output was created.
The main task separately reported the same expected limitation on actual
candidate16, opening only its manifest. That main-task observation was not
rerun here and is not full Stage B qualification.

The scientific analyses contained in the replay receipt are decoded with that
receipt after the original gate; validation of its full-replay claims follows
the receipt decode. The score/acceptance artifacts are decoded only after both
gate checks and byte integrity. The pure `describe()` helper accepts already
decoded in-memory data and explicitly assigns acceptance responsibility to its
caller; it is not itself an artifact-release gate.

Additional fabricated probes instrumented `snapshot()` calls. Rejection of the
original gate, wrong runtime, 639 checkpoints, 47,999 cached responses, a nonzero
optimizer-update count, a false independent-digest flag, a different manifest
binding, and a wrong receipt pin all occurred without reading score/acceptance
artifacts or creating output. The supplied negative fixtures also check upstream
and supplemental rejection. The independent origin of an actual supplied pin
is a custody obligation, not something these tests prove.

The output records manifest, supplemental-receipt, score, acceptance,
registration, and tool hashes. `byte_integrity` retains and checks the input
manifest's original-hash/derivation declarations. In the normal external-output
fabricated run, an extra before/after check confirmed identical input inventory
and hashes. All indexed output hashes matched. These observations establish
fabricated preservation under that path configuration; they do not certify any
actual original-data hash. The destination-path finding is closed by the latest
guard and the fresh path probes described above.

## Scientific reporting checks

- All 640 ordered cells are retained. The fixed-init0 subset has 320 arm/history
  records over 80 shared system/history combinations: delivery, online, flat
  (`simpler`), and the short-fit ablation. Both collection histories and all forty
  systems are represented. Initializations 1 and 2 each retain 160 additional
  delivery/flat records in `all_cells`; no initialization is selected by its
  score. Explicit integer typing now rejects `primary_init=False`.
- MSE and NMSE remain absolute recorded values, including zero. NMSE is checked
  against MSE divided by that score cell's positive training target variance;
  the normalizer must agree across the arms/initializations sharing a history.
  Floor flags are retained for every cell. The `1e-12` floor affects paired log
  calculations, not the displayed absolute errors.
- The four contrasts remain in registered order: five-node delivery/online,
  five-node delivery/flat, thirty-node delivery/online, thirty-node delivery/flat.
  Each has twenty system rows in fixed ID order, totaling eighty. The code uses
  `mean_h(log(max(delivery_h, floor)) - log(max(control_h, floor)))`, then
  exponentiates. An independent scalar check gave the same system log ratio
  `-1.9560115027140732` for fabricated system `5:01` versus online; the supplied
  unequal-history fixture also distinguishes this from a ratio of mean errors.
- Eight descriptive history ratios, two short-fit ablation summaries (with their
  system log ratios in JSON), and four primary floor-count records are retained.
  Their values are checked against the pinned analyses. The ablation also
  averages the two history log ratios within each system before aggregation.
  The floor-count caption correctly explains repeated delivery counts across
  contrasts; these counts do not create additional independent observations.
- The latest positive export produced eight absolute-error tables and seven
  descriptive tables. An additional inspection of the actual fabricated LaTeX
  table bodies found row counts `[20, 20, 20, 20, 8, 2, 4]`. Each system log/ratio
  row matched its JSON value formatted to six significant digits. All forty
  systems appear four times in the absolute-error tables and twice in the
  contrast tables. The newly added file is covered by the output hash index.
- A fabricated worsening-delivery variant produced four false registered
  superiority verdicts while retaining all eighty system rows, including 76
  ratios above one, and all twelve zero-valued cells. No favorable exclusion or
  added significance calculation appears in the new reporting module. Its
  captions describe paired system units and descriptive history/ablation
  displays; they assert no collection-strategy superiority or broader causal,
  architecture, graph-size, or equal-compute claim.

The tests call the existing registered statistics helper on fabricated inputs to
construct consistent expected summaries. Those calls are fixture construction,
not added inferential analyses of actual outcomes. Table-count, membership,
formatting, and output-hash assertions are also not scientific tests.

## Test substitutions and qualification limits

The six supplied unittest methods passed on the latest source from `/tmp`,
including the added path, complete-inventory, and sentinel-preservation assertions. Supplemental
probes checked gate ordering, the unfavorable/zero fixture, typed primary init,
table bodies, normal-path input preservation, the corrected destination guards,
and explicit missing-contract rejection.
One initial auxiliary arithmetic probe stopped with a fixture-key `KeyError`
after successfully reproducing the path defect; it was corrected and the
arithmetic check completed. No source change was made for any probe.

The positive export fixture substitutes **only `reporting.gate`**, because its
contract contains fabricated registration metadata rather than a real original
acceptance projection. `snapshot`, `byte_integrity`, supplemental receipt
validation, analysis matching, rendering, and output hashing run. Its four-file
package and self-created pins do not represent a real full release. The
receipt's 640/240/48,000 counters and exact-runtime string map are fabricated
claims; no checkpoints were replayed and those dependencies were not executed
or qualified. Negative fixtures deliberately mock the original gate to isolate
supplemental rejection, or make it raise to check ordering. None demonstrates
successful actual original acceptance.

Failed or interrupted *attempts* are distinct from unfavorable scientific
contrasts. The new module renders the complete accepted score matrix; it does
not emit an attempt-disposition table or a failure journal. The final positive
fixture retains a manifest-bound `B/retained_attempt_sentinel.json` containing
an interrupted attempt with unknown returned responses and a failed attempt.
The complete before/after inventory/hash equality proves preservation of both
sentinel entries and their unknown/failure fields; the test additionally checks
the interrupted status explicitly. This confirms fabricated artifact retention
without treating failure as zero responses or deleting evidence. Actual
preservation and final disclosure of attempts remain unverified here and must
retain the manifest-bound journals/accounting evidence. Missing scientific cells
are rejected, never silently removed to obtain a smaller favorable display.

Similarly, separate initialization1/2 values are retained in the machine-readable
matrix, but the new LaTeX files contain fixed-init0 displays. They do not replace
the original exporter's initialization-sensitivity tables or primary estimates,
marginal intervals, raw/Holm p-values, and registered verdicts. Final integration
must retain those accepted displays and the sampling-model qualification; this
review does not claim the new files alone complete the manuscript contract.

Actual original acceptance, exclusive complete custody, full supplemental replay
in the recorded runtime, source/runtime qualification, and anonymity qualification
remain **unexecuted/unqualified** in this review. No manuscript integration or
claim of scientific superiority is authorized by fabricated success. Native
LaTeX compilation and page-level manuscript layout were not run. There are no
remaining required code fixes from this bounded review. Retain the original
acceptance and full supplemental replay gates before integrating any actual
Stage B numbers.
