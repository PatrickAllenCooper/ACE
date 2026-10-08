# Prospective fixture location implementation review — 2026-10-08

Disposition: **No required implementation defects identified within this bounded source-derivation interface and its supporting tests/guidance.** Runtime import guards remain **UNQUALIFIED**, and runtime import authorization remains deliberately open. This review does not authorize fixture execution or qualify anonymous execution, original training reproduction, numerical results, or release.

## Scope and reviewed identities

One distinct, bounded review of the new interface, expanded at the owner's request to inspect its new tests and guidance. Reviewed source identities:

- `scripts/research/prepare_delivery_prospective_fixture_locations.py` — SHA256 `155b2754d7a7c523e4d0dac7c4bf3d239e01233747d98c25e832a99795e08244`.
- `scripts/research/test_delivery_prospective_fixture_locations.py` — SHA256 `7149c023bae9037997d495b2b8dbc8e50596ceac01641215b8490c604aa79c71`.
- `docs/development/guidance/delivery_fixture_locations_2026-10-08.md` — SHA256 `6b01075dffae754a4bd15bfc564c58ed960268687eceaba06c2800e48b63a111`.

The already reviewed publication helper was read only as a dependency for `plain_path`, `exclusive_directory`, `store`, and `seal_directory`: `scripts/research/prepare_delivery_publication_snapshot.py`, observed SHA256 `ba2b3958ddd7019566fbe35c21f9186df994d463657f0731bb009c4e43ac6795`. Existing review dispositions were read as context; their source/replay/publication reviews and tests were not repeated. The resolved owner MIT determination was not reopened.

Verification consisted of static inspection, SHA256 comparisons, and an independent in-memory AST/byte reconstruction using standard-library analysis of frozen Git source blobs. Neither the new implementation nor its test module was imported or executed by this reviewer. No original tests, workers, model/IO modules, or fixtures were imported or executed. No scientific inputs, B scores/outcomes, private closure, target runtime environment, remote resources, or Slurm jobs were accessed. The two original fixture files were read only as frozen source text. Only this report was written; no commit was made.

## Source derivation and preservation

Implementation lines 18–33 and 61–93 bind the two original fixture identities and the exact original/replacement expressions. The original model and IO Git blobs at revision `45ebeb89d2c76daa97a55f07728239245e0c4f60` independently match both hard-coded fixture SHA256 values. `derive_model` requires exactly one matching simple top-level assignment for each name, compares each original RHS AST, calculates spans in UTF-8 bytes, and replaces spans in descending order. Masking only the two location values before the final AST comparison checks preservation of the remaining semantics. Direct byte splicing preserves comments, whitespace, imports, functions, classes, and assertions outside those spans.

Independent static reconstruction on the pinned model source identified half-open byte spans `PROJECT [322, 357)` and `SOURCE [367, 511)`. The two derived RHS ASTs exactly match the intended expressions, the three intervening/outside original byte segments survive unchanged, and the normalized original/derived ASTs are identical. Reconstructed identities:

- Original model: `913dd23cf0aa34a22447f9c807753a1011e1c6684a70abbe1443eb9f26533f67`.
- Derived model: `ceed644e3f328f39717ef7d46bcdfa8712f701e2bba2743eb4d3f728bd2b7ceb`.
- Original and unchanged prepared IO: `9cd49e43934fce0f542ba38e51f992a85533899b6634bc274bdd7e6d6fc63a2c`.

These are source-analysis results, not hashes measured from an executed preparation. The initial reviewer analysis harness could not literal-evaluate the named keys in `PINS`; resolving those keys from already extracted string constants corrected the harness. No reviewed source was executed in either analysis attempt.

Implementation lines 119–139 preserve separate original/prepared paths and bind every emitted file's digest and byte count. Only the prepared model has a changed identity; the prepared IO intentionally retains its original identity and original model-module import. No receipt or original source hash is overwritten.

## Closure and output checks

Implementation lines 40–58 and 96–121 authenticate the closure's captured bytes against a caller-supplied independent full SHA256, reject duplicate JSON keys/nonfinite constants, require the frozen revision/registration and false outcome-access flag, and require the 24-project/19-learner inventory with both fixture pins. Every listed source path must be canonical and relative; every listed digest must be a full SHA256 and must match its one captured byte snapshot. The derivation and output bytes use those captured snapshots. All closure reads, hash checks, and derivation checks precede output creation at line 155.

Completeness and original membership are rooted in the independently trusted closure pin; the count check alone is not an independent inventory authority. This reviewer checked the validation logic and the two original Git fixture identities, not an actual private 43-file closure. Captured byte authentication does not certify a later runtime import layout or make the original filesystem permanently immutable.

Implementation lines 97–100 exclude an existing destination and destinations inside the closure root or implementation repository. Lines 155–165 use the reviewed helper for exclusive directory creation, descriptor-relative exclusive file writes, private permissions and sealing, followed by a final path/descriptor inode comparison. Output names are fixed internal constants. The helper's existing trusted current-user/root, ACL, and macOS boundary applies; this review does not extend or repeat its custody qualification.

## Tests, guidance, and remaining gates

The five new test methods were inspected, not run. Tests lines 28–58 cover a separate exact byte oracle, function/class source preservation, UTF-8 offsets on the declaration line, multiline grouping, and rejected declaration forms. Lines 60–121 construct a fabricated full 43-source closure, patch the two fixture pins explicitly, inspect emitted identities/false qualification flags, reject an existing output, and check several metadata/source/path failures before output creation. The learner corruption case targets the final listed learner object, covering full traversal before writes. These checks exercise preparation mechanics; patched fabricated identities do not authenticate real B sources or qualify scientific guards. The owner's report that all five methods pass is supplied evidence, not independently reproduced here.

Guidance lines 10–40 accurately distinguish preserved originals, the two location edits, unchanged IO, and private metadata. Lines 42–59 explicitly state that original imports precede location declarations and that authenticated import closure, cached-module rejection, an explicit absolute layout root, exact dependencies, and separate execution authorization remain required. The current replacements resolve the root but do not supply those guards; the guidance and contract correctly leave them as future gates. Lines 61–78 offer only the preparation utility command and describe the tests as source/AST checks.

The implementation contract at lines 130–152 truthfully records source derivation, false execution/reproduction/anonymity/public qualification, and zero new fits/responses. Original location expressions in `ORIGINAL`, copied originals, and the emitted contract are private metadata, not an anonymous candidate. Source preservation of charging, heldout, calibration, predicted-parent, and optimizer assertions establishes byte identity only; their relocated runtime behavior remains unqualified. No required fix was found within this review's scope, and no runtime authorization is implied.
