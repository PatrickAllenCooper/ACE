# Notice and optional metadata preparation review

Meitner performed a distinct bounded static review of the new notice planner,
TOML projection, builder, verifier and B byte-only verifier support. No actual
model replay, study work, SSH access or outcome opening occurred.

## Required findings, fixed

1. **P2: path aliases and bidirectional overlaps.** Planner now resolves every
   input/output path, rejects overlaps in both directions (including symlink and
   parent aliases), and validates the exact F metadata binding before any write.
2. **P2: independent notice source authentication.** Hardcoded reviewed digests
   authenticate ACE's license and the Chambers source receipt, README, generator
   and archive audit. Parsing/notice derivation use those captured bytes; every
   exact original notice source is retained privately. A self-consistent altered
   declaration/receipt pair cannot substitute for the independently reviewed pins.
3. **P2: live verifier source replacement.** The verifier captures its source at
   import, compares compiled snapshot code to the executing module, and pins that
   digest for exemption/reporting. Later disk edits cannot change the trusted pin.
   The planner stores an authenticated private verifier snapshot. This is a
   scoped compiled-implementation/snapshot check, not dependency supply-chain or
   historical freeze certification.

Final static recheck closed all three findings with no remaining required fixes
within this scope. Reviewer inspected tests but did not build/review candidate11.
Five focused new checks and eight existing release regression checks pass; those
regressions were required because the verifier/build transform interfaces changed.
They check retained license/dependency semantics, aliases, source pins, source
replacement, privilege exemptions and original/derived digest distinctions.

## Scope

Only optional `tool.poetry.authors`, `homepage` and `repository` are omitted from
newly projected metadata. Every other parsed TOML value stays unchanged. The
original F source digest still binds original private metadata, with separate
manifest links to the newly derived bytes. No historical source or receipt is
rewritten. Actual A/C/F model, response, numerical source and score objects remain
unchanged; no new inference is needed for this metadata-only transformation.

Runner metadata declares MIT, but no authoritative copyright/permission notice
was found for exact revision `e9f811fbc68fb70681c3d89182d8ebe024882ce3` in the
bounded local history/source search. Archived code_health explicitly records
`Add LICENSE (pyproject declares MIT)`. Available public exact-revision lookup
was restricted/404, which is not absence or ownership evidence. Do not infer
ownership from team metadata. Ask the owner for the authoritative grant/notice.

Original candidate09 and the first new pre-review prototype10 are preserved.
Final candidate11 remains private and explicitly blocks public release pending
Runner authority, complete B evidence, full worker/accounting/anonymity and human
submission gates. This preparation does not authorize an upload or extra job.
