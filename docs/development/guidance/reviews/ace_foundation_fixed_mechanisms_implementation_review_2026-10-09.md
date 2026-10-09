# Fixed mechanisms: implementation review

Date: 2026-10-09. **Required findings: 1 — restore must preserve the accepted canonical table pin.**

## Scope and exact inputs

- `scripts/research/foundation_fixed_mechanisms.py` SHA256: `2c4f3a13415177e48b9c6aa4cdd92b1f2dbd68538cedca7f6b6bdc79a47bd806`.
- `scripts/research/test_foundation_fixed_mechanisms.py` SHA256: `0fc63c53bf64b878d74a0b269781082d7ddfea91b6c1b8c771fa00cbaaaa8701`.
- `docs/development/guidance/ace_foundation_fixed_mechanisms_2026-10-09.md` SHA256: `a8eb8df955b0bac8030a90929c08d941e4f9ed18dea711f3aae24814b92e33fb`.

Read-only source review against the supplied design. The nine existing arithmetic fixtures were inspected, not rerun. One novel restore-only arithmetic counterexample was executed with Python stdlib and this pure module, without a teacher call, models, fits, scientific inputs or responses. Only this review file was written; no implementation/test changes, model qualification or commit.

## Required finding

### Preserve canonical payload identity through restore

**Location:** `foundation_fixed_mechanisms.py:130–140`; relevant serialization at lines 108–112. Existing roundtrip coverage: `test_foundation_fixed_mechanisms.py`, `test_pinned_roundtrip_and_corruption`.

`restore` authenticates the incoming JSON representation, then accepts any spelling that `float.fromhex` can parse. It reconstructs floats and canonicalizes them through `payload()`, without requiring that the returned table still has the accepted digest. Consequently, an accepted independently pinned input can produce a table whose exported payload and digest disagree with that pin. This conflicts with the adapter's exact-table-byte binding and weakens the restore contract needed by a future frozen runner. This is not a SHA collision or a bypass of an independently pinned canonical producer payload; it is an accepted-input/canonical-output consistency defect.

**Concrete new counterexample:** construct a one-knot `FixedTable` with knot 1.0, value 7.0, retained linear coefficients (0.0,1.0), provider `grammar` and fit-parent SHA `'0'*64`. Change only its payload knot spelling from `0x1.0000000000000p+0` to the numerically equivalent `0x1p+0`. Independently pin that input using the module's sorted compact JSON hashing convention. `restore(payload,pin)` succeeds, but:

- accepted input SHA: `dab2429319a32d659ea2da5aa0b22f4befe07f964687f8a808386f81f5fdd451`;
- returned table SHA: `22fdf90486a6bdbf54702c99cc7ca656f5349abd41c4fb0b2cbb92514b509e48`.

**Required correction:** enforce the declared object/list/string field shapes and canonical finite `float.hex()` representations. After constructing the table, require that its exported payload is the accepted input payload and its digest equals `expected_sha256`; otherwise fail explicitly. Add a focused negative fixture for the equivalent noncanonical spelling under its own matching input pin, and retain the canonical positive roundtrip. This does not require a model or scientific run.

## Other scoped dispositions

- **Grid:** fit parents are copied to plain finite floats; nondegenerate grids use exactly 129 points with exact endpoints. Nonfinite spans, collapsed/nonincreasing points and empty or invalid inputs fail before inference. A degenerate interval has exactly one knot. Restored/direct tables recheck the same grid policy and output count.
- **Copied state and teacher lifetime:** retained coefficients, knots and outputs become tuples of plain floats in frozen dataclasses. The returned object retains fixed data and the immutable retained-parameter object, not a teacher callable or model. Mutation of original coefficient/output lists cannot change these values. Hostile `object.__setattr__` use is explicitly outside the stated trusted-runtime guarantee.
- **Pointwise evaluation:** scalar branches use only the input and stored data. Inside interpolation uses a convex weighted sum without subtracting extreme table values; exact knots return exact stored values. Outside inputs evaluate the copied retained family. Single-knot behavior and boundary jumps agree with the design. Retained-family expressions match the documented linear/quadratic/tanh forms; historical vectorized bit identity is not claimed.
- **Extreme floats and failures:** invalid/nonfinite scalars, nonfinite spans and retained arithmetic overflow fail. A finite-input subtraction overflow during grid construction is rejected. Finite-output validation also rejects interpolation overflow rather than replacing an inside result with retention. Teacher exceptions, invalid outputs and output-count mismatches propagate as failures; there is no silent fallback. The source does not claim total finite behavior across an unqualified deployment domain.
- **Provenance boundary:** the ordered fit-parent digest and table payload bind supplied values, but do not authenticate fit eligibility, teacher source/weights, phase timing or absence of private access. Those obligations are explicitly assigned to a future runner. No actual-model qualification follows from these arithmetic fixtures.

The nine existing fixtures cover linear and quadratic interpolation oracles, one teacher call and partition/permutation invariance, copied/frozen state, outside retention and boundary discontinuity, degeneracy, canonical pinned roundtrip/corruption, teacher/overflow failures and rejected grids. They do not cover the accepted noncanonical restore case above. No other required correction was found in this bounded review.

## Scoped restore correction recheck — 2026-10-09

**Disposition: the required restore finding is closed; 0 required remaining issues in this correction.** Earlier observations above remain the historical review of the original bytes.

Current hashes:

- `scripts/research/foundation_fixed_mechanisms.py`: `3f3796b5be11f582d76bf07a107fc385e619c3eec475f639a83d9391f62f3bae`.
- `scripts/research/test_foundation_fixed_mechanisms.py`: `b65a80598f25a180826e7b89c169c9a7091d4216bf5cdbc2e4f94e5c02a4eca9`.
- Design document at recheck, recorded without a renewed design review: `5fa2052bb48e7f2adaed258861bb202c148b22f599a97898fd2ecb378d5122c2`.

Static inspection of `restore` now finds both required guards before returning: `table.payload() == payload` and `table.digest() == digest`, with `digest` already authenticated against `expected_sha256`. This rejects alternative hex spellings and container representations that do not reconstruct to the canonical exported payload, while preserving the original independent pin. The prior abbreviated-hex counterexample cannot pass these guards.

The new `test_noncanonical_pinned_payload_rejected` source independently recomputes matching input pins for uppercase substitutions in knots, values and retained coefficients, requires rejection for each, and retains a canonical positive roundtrip. These are alternate spellings of finite values and therefore exercise canonical reconstruction rather than merely a stale-pin mismatch. The author reports this new method passed; this recheck did not execute it or the previous nine methods.

Only the restore/test delta was reviewed and this disposition appended. No models, scientific execution, source edits or commit; no actual-model qualification is asserted.
