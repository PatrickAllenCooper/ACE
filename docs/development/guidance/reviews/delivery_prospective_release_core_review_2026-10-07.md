# Prospective release-core review and disposition

Independent read-only reviewer Anscombe (`01a117c1-ed0f-7f20-b255-5c754333d37a`).
Scope: the new prospective source-extraction/preflight tool and its synthetic
tests. This distinct review did not repeat A/C/confirmation reviews. No actual
B outcomes, model inference, fits, Slurm operations or environment changes.

## Initial findings

1. **Outcome parsing before validation.** Loading original acceptance JSON also
   decoded its primary and secondary analyses before verifying acceptance, while
   the helper reported no outcome access. Fixed with a custody-bound metadata
   projection. A strict lexical JSON scanner skips both scientific objects,
   including numbers and strings; it does not construct their values. The failure
   flag and successful audit supervisor are checked before projection.
2. **Malformed telemetry.** NaN, negative and boolean measurements, or boolean
   exit codes, could pass comparisons. Fixed explicit type/finite/nonnegative
   checks and an integer exit-code requirement before resource ceilings.
3. **Incomplete acceptance structure.** Scalar completion counts did not require
   original evidence maps. Fixed exact required original field membership,
   eighty training and forty evaluation receipt maps, three phase hashes, all640
   replay discrepancy entries, tolerance, timing, score-source labels and digest
   validation. The synthetic successful fixture now includes this full structure.

## Follow-up findings

4. **Metadata/hash race.** Reading text then hashing a path could bind parsed
   metadata to replacement bytes. Fixed one captured byte snapshot per metadata
   read; its digest is verified against the supervisor binding before projection
   and returned without rereading. The same rule was applied to registration,
   complete/supervisor metadata and original auditor/utility source extraction.
5. **Non-JSON digits.** Unicode-aware numeric matching accepted digits standard
   JSON rejects. Fixed explicit ASCII digit grammar throughout the skipped-value
   scanner, with a non-ASCII rejection fixture.

## Final disposition

No required fixes remain within the bounded review. Thirteen focused tests pass:
failed supervisor and missing completion; wrong runtime; incomplete evidence;
malformed telemetry/digests; failure flags; malformed ignored JSON; a decoder
spy that rejects decoding scientific numeric/string sentinels; deliberate file
replacement after metadata/source capture; exact function AST bodies and analytic
240-cell statistical fixtures. These fixtures are not scientific B outcomes.

All twenty original worker/generator bindings were checked in private custody.
Fourteen exact source segments (six numerical functions, five constants and
three utilities) have original text/AST hashes and a distinct derived-core hash.
The numerical core is identical across preserved prototypes. The final contract
also pins the preparation/preflight implementation.

Limits: upstream metadata validation does not replace full raw-artifact replay,
the original unchanged audit, target-runtime qualification, authenticated freeze
proof, final anonymity or public release. The eventual replay must reverify its
complete derived manifest and raw bindings immediately before using outcomes.
No final B release or fitted experiment is authorized by these tests.
