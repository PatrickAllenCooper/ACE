# Optional descriptive reporting source integration review — 2026-10-08

**Current result: R1 resolved; zero remaining required findings in this bounded review.** The corrected source validates captured reporting and replay import descriptions before score decoding. The original finding, reproduction, seven-test results, and earlier hashes remain below as prior evidence.

## R1 correction recheck — snapshot D

Reviewed the corrected bytes captured at **2026-10-08 04:17:23.119129 UTC**, and rechecked all three scoped hashes at **04:17:35.048893 UTC** with no changes.

- [Static import description](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:109) parses captured bytes and walks all static import statements, including imports inside functions. [Reporting validation](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:136) compares the complete supported statement multiset: module names, import/from form, relative level, symbol lists, aliases, and multiplicity. A matching SHA pin no longer permits the R1 changed import.
- [Captured replay validation](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:147) checks the supported transitive static import closure, including deferred numpy/torch, verifier symbols, and `ace.oracle.MLPSurrogate`. [Export validation](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:160) requires the reporter's six replay names in the captured replay source's top-level definitions/assignments. All this occurs through [the pre-score call](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:355), before the first score snapshot.
- [The successor consumes validated descriptions](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:251) for direct imports, standard-library modules, and captured replay imports. It describes numpy/torch versions from the authenticated B runtime contract and the oracle learner identity from its registered source hashes. [Manifest bindings](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:288) bind reporting/replay/verifier source digests, the B runtime record, and `/interfaces/6/transitive_learner_import/sha256` to `source/runner/ace/oracle.py`. The B runtime record already binds the corresponding learner and carries the complete registered dependency versions; numeric version strings are contextual runtime values, not file digest edges.
- Snapshot D leaves captured-byte packaging, resolved output separation, default six interfaces, and false execution/reporting/public/anonymity qualifications intact. The AST checks establish the supported static source description and declared names; they do not prove successful execution, object types, or general program semantics. Actual package and supplemental replay remain unqualified.

**Bounded independent proof:** ran only the new [changed-closure test method](/Users/pat/code/ACE/scripts/research/test_delivery_prospective_reporting_interface.py:241), not the full suite:

```sh
cd /Users/pat/code/ACE/scripts/research
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.11 -B -m unittest -v test_delivery_prospective_reporting_interface.ReportingInterfaceTests.test_changed_import_closures_reject_with_fresh_matching_source_pin
```

Result: **1 method / 6 changed-source subcases passed in 2.068 seconds**. These include the exact original R1 module substitution, a changed standard-library module, changed reporter import symbols, changed replay numpy/learner imports, and a missing replay export. Each uses a matching reporter SHA pin; replay variants are authenticated by a fresh captured replay digest. The score guard detects any attempted score snapshot, and every case asserts absent plan/private outputs.

**Main's final full focused result:** inspected [the final log](/tmp/ace_reporting_source_tests_after_fix.log): **8 tests passed**, unittest **37.771 seconds**, wall **37.83 seconds**, user **29.02 seconds**, system **8.72 seconds**, maximum RSS **77,053,952 bytes**. This full run was performed by main, not repeated by this reviewer. It includes the positive package/relocation, six-interface default, replacement race, custody/rejection tests, and the new six changed-source subcases. Inspection confirms the new learner binding is generated; the positive test iterates direct import/runtime digest bindings but does not separately assert that new learner pointer.

No actual B outcomes, scientific report, numerical replay, CURC, or installation was accessed for this recheck. Main separately reports original acceptance/custody readiness; that operational status does not qualify the actual package or supplemental replay and was not independently audited here. Commit and actual package freeze remain main's work.

**Snapshot D — current reviewed source hashes:**

```text
scripts/research/extend_delivery_prospective_release_plan.py
  38793 bytes
  d6880d4b0a7302fa8d561ad18535129ed09d06f1eb8870a1cca63006455850d7
scripts/research/delivery_prospective_source_contract.py
  20618 bytes
  7879b477c52eec7bad143264a4363f55deec44b73c4b9600d1c983d27f9daa10
scripts/research/test_delivery_prospective_reporting_interface.py
  17269 bytes
  44c9562f79d2597960c8d760a4151e420c408cca23444c5465f939926743d640
```

## Prior review record — snapshots A/B/C

The following preserves the original defect and verification evidence. Its code locations and acceptance reproduction refer to the bytes identified as snapshot C, before the fix. On snapshot D, the original reproduction now raises `ValueError: unsupported captured reporting import closure` at preparation rather than producing the previously misdescribed contract.

**Prior result at snapshot C: one required finding (P2).** Independently pinned reporting bytes can be accepted while the successor records imports that those bytes do not make. The final seven fabricated tests pass; this does not cover the changed-source case reproduced below. No additional required finding was established within this bounded review.

## Scope and limitations

Reviewed the current changes in `extend_delivery_prospective_release_plan.py`, `delivery_prospective_source_contract.py`, and the new `test_delivery_prospective_reporting_interface.py`. Inspected the existing reporter, replay adapter, verifier, builder, and fixture constructors only as dependencies of this integration. Did not repeat historical A/C/F/B component reviews or run their test suites.

Checks were limited to fabricated custody and packaging, source inspection, and a fabricated alternative reporting source. No actual B outcomes, checkpoint/model deserialization, fits, report execution, numerical replay, CURC access, installations, or resource changes were performed. The fixtures reuse registered structural metadata and replace scientific contents with fabricated values and placeholder bytes. Existing fixture constructors were imported; their older tests were not run. Main owns actual B acceptance and local/remote custody. Its separately reported original-incomplete-gate rejection was not independently rerun here.

This review addresses the correctness of the declared reporting interface and its source/import/runtime closure. It establishes neither successful anonymous imports nor actual runtime qualification. No implementation changes or commit were made by this reviewer; the only persistent review output is this document.

## Historical required finding — resolved in snapshot D

### R1 — P2: validate captured reporting imports before recording a fixed closure

**Historical locations in snapshot C:** [reporting preflight](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:108), [fixed import and standard-library records](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:199), [optional source capture and pre-score dispatch](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:350), and [positive-only AST comparison](/Users/pat/code/ACE/scripts/research/test_delivery_prospective_reporting_interface.py:81).

The caller may supply a different reporting source and its independently supplied SHA256. The planner authenticates and captures those bytes, but `validate_reporting_before_scores` checks package collision, predecessor copies, and verifier identity only. It does not parse the captured source or compare its import modules/symbols with the closure that `transition` subsequently declares. The latter always names the current implementation's replay imports, `verify_delivery_release.relative`, and eight standard-library modules.

On snapshot C before the fix, replacing only

```python
from verify_delivery_release import relative
```

with

```python
from unrecorded_reporting_dependency import relative
```

and supplying the replacement's SHA256 succeeds. The new source is preserved exactly, one fabricated score decode occurs, and interface 7 still declares `verify_delivery_release.py` / `relative`; it does not declare the actual new import. The alternative source SHA256 in this reproduction is `31324d95ae7cc56c5c9c31ebd6edf3ebf70949db697c607e205880b2b0351f58`. The missing dependency is never imported or executed by the probe.

**Impact:** the successor's source-disposition record misdescribes authenticated bytes and omits their dependency. Correctly binding the declared verifier and reporter digests does not authenticate the relationship between those artifacts. Keeping `import_qualified = false` accurately withholds execution qualification, but does not make the stated import provenance accurate. This is a preparation metadata defect; the probe makes no claim about actual B execution or scientific results.

**Required correction:** before the first score decode, validate the supported reporting import closure and imported symbol sets from the captured `_raw` bytes, including agreement with the standard-library declaration. Have the transition use that validated description. Reject a changed closure that this reporting interface does not support; if a new closure is intentionally supported, describe and digest-bind its actual packaged dependencies and supported runtime records. Check the declared transitive relationship to the explicitly captured replay source as part of this interface's closure. This requires source/metadata correctness, not executing the reporter or adding a general security policy.

Add a fabricated rejection case with the changed source and a matching fresh independent pin, asserting no score decode and no outputs. The current AST test examines only the repository reporter and filters to the two expected module names; it does not reject a caller-supplied unexpected module or establish closure completeness.

**Original reproduction against snapshot C (now rejects on D)** using the existing Python 3.11 and fixture constructors only:

```sh
cd /Users/pat/code/ACE/scripts/research
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.11 -B - <<'PY'
import ast
import json
import tempfile
from pathlib import Path
from test_delivery_prospective_reporting_interface import reporting_fixture
import extend_delivery_prospective_release_plan as planner
from verify_delivery_release import sha

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory).resolve() / 'inputs'
    root.mkdir()
    args, _, _ = reporting_fixture(root)
    source = args['reporting']
    raw = source.read_text().replace(
        'from verify_delivery_release import relative',
        'from unrecorded_reporting_dependency import relative',
    )
    assert 'from unrecorded_reporting_dependency import relative' in raw
    source.write_text(raw)
    args['expected_reporting_sha256'] = sha(source)
    result = planner.extend(**args)
    record = json.loads(
        (args['private_dir'] / 'source_contract_B.json').read_text()
    )['interfaces'][6]
    actual = {
        n.module + '.py': [a.name for a in n.names]
        for n in ast.walk(ast.parse(raw))
        if isinstance(n, ast.ImportFrom)
        and n.module not in ('datetime', 'pathlib')
    }
    declared = {r['path']: r['symbols'] for r in record['import_records']}
    assert result['descriptive_reporting_included']
    assert actual != declared
    assert (args['private_dir'] / source.name).read_bytes() == source.read_bytes()
    assert record['import_qualified'] is False
    print({'accepted': True, 'actual': actual, 'declared': declared})
PY
```

## Checks completed without additional required findings

- **Gate and rejection order:** [the original gate remains the first operation](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:158). Optional pin/source/prerequisite, declared dependency identity, collision, and output overlap checks precede [the first score decode](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:390). Fabricated cases cover absent/malformed/mismatched pins, missing replay/predecessor, pin without source, missing/symlink sources, parent symlinks, exact/child collisions, predecessor symlink, verifier transform, and output overlap including input aliases.
- **Captured bytes and replacement:** [module capture](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:76) bounds regular source bytes, rejects observed symlink paths, and hashes one read. [Output inclusion](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:472) writes captured bytes into private snapshots. The replacement test verifies one capture per optional module and no reopening of caller paths after replacement with symlinks. Derivation metadata binds the source label, captured digest, and private snapshot.
- **Collision and output/custody checks:** [reserved optional interface names](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:361) reject exact and ancestor/descendant conflicts. [Final output comparisons](/Users/pat/code/ACE/scripts/research/extend_delivery_prospective_release_plan.py:366) resolve output/input aliases, require exclusive separate outputs, include inherited and optional sources in overlap checks, and run before scores/writes. The worker added normalization during review; that initial concern is resolved in the final snapshot. No general concurrent filesystem mutation guarantee was established.
- **Declared digest edges:** [interface/import/runtime bindings](/Users/pat/code/ACE/scripts/research/delivery_prospective_source_contract.py:235) bind interface 7 to reporting source, replay source, identity verifier, and `B/replay_contract.json`. The runtime record transitively binds its core/core contract, helper, learners, and retained/derived artifact relationships. For the repository reporter, inspected imports and symbols agree with the description; the replay adapter's top-level import outside the standard library is the included verifier, with scientific dependency loading deferred to functions. This static correspondence does not cure R1 for a different pinned source.
- **Disposition and default behavior:** the fabricated package preserves the 24 original source identities, unchanged historical predecessor bytes, notice bindings, and unqualified original execution dispositions. Reporting adds a seventh interface; omitting reporting retains six and omits the new reporting qualification field. The new interface's anonymous execution, import, reporting execution, generated report, and numerical replay flags are false. Global training/public/anonymity and B replay/reporting qualifications remain false. No generated reporting artifacts are included.

## Prior verification evidence

Independent snapshot C focused command:

```sh
cd /Users/pat/code/ACE/scripts/research
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.11 -B -m unittest -v test_delivery_prospective_reporting_interface
```

Result: **7 tests passed**, unittest reported **35.611 seconds**. Included fabricated build/relocation, original-byte preservation, tamper rejection, default six interfaces, source replacement, and rejection order checks. No older suite was selected. The first attempt with the system Python 3.9 failed before tests because fixture import requires `tomllib`; the existing Python 3.11 resolved this without installation.

The first successful focused run used the initial snapshot and passed seven tests in 36.145 seconds. Because worker edits changed the planner and new test file during review, the final focused suite and R1 reproduction were run again against the final candidate. All three scoped source hashes were unchanged between the final suite's starting capture and the final verification capture below.

## Reviewed snapshots and hashes

Repository HEAD at review: `49f52b0b070ef80aef4c9930efaef74c89db309d`. Hashes identify working-tree bytes, including the new untracked test file, rather than that commit alone.

**Snapshot A — 2026-10-08 04:09:05.773540 UTC**, initial captured bytes:

```text
extend_delivery_prospective_release_plan.py
  38695 bytes
  c8c76f80d71d598bfef000b2cf5bff78ee76283ff9bb3e76026b206d54a6a219
delivery_prospective_source_contract.py
  16891 bytes
  6b69cf86673ad010f188c1469060831d4666b8d2066a36e7af7f73625dd4d4b4
test_delivery_prospective_reporting_interface.py
  15016 bytes
  9b08ed19d9f2aa9583d81436b59a0f8db9974b5ae70afdfb84f1a40033ce38a9
```

**Snapshot B — 2026-10-08 04:11:09.010661 UTC**, worker candidate used for final focused suite; **Snapshot C — 2026-10-08 04:12:05.477796 UTC**, final verified bytes. B and C match. A final source recheck at 2026-10-08 04:13:53.402238 UTC also matched all three scoped hashes. Historical code/line references in the prior review record and its original finding apply to C; the correction recheck references D.

```text
scripts/research/extend_delivery_prospective_release_plan.py
  38779 bytes
  e6ff902b998d1704cc21fb60aff52b8536d5c2c7a97bc34a7e671cec1ae1ea7d
scripts/research/delivery_prospective_source_contract.py
  16891 bytes
  6b69cf86673ad010f188c1469060831d4666b8d2066a36e7af7f73625dd4d4b4
scripts/research/test_delivery_prospective_reporting_interface.py
  15677 bytes
  af17e910ffedd8d970d800200e498ae0f4000a48b32b1d8dfacc8c9566162b05
```

Supporting implementation bytes inspected, captured with C (not edited):

```text
scripts/research/prepare_delivery_prospective_supplement.py
  d7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24
scripts/research/replay_delivery_prospective_release.py
  80dc80239b403cf2b443987fb92640ea0eec938876d14ac6958c040bc8c7c1c3
scripts/research/verify_delivery_release.py
  9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52
```

The reproduction's verifier import digest comes from the fabricated inherited verifier fixture, not the repository verifier hash above. This is expected fixture provenance and does not demonstrate a successful import of the packaged reporter.
