# Publication snapshot implementation review — October 8, 2026

One bounded independent reviewer examined only the new metadata snapshot
interface. No scientific artifact, original worker, replay, prior tests or remote
channel was used by this review. Main implemented new Git/custody fixtures in
parallel, then requested scoped rechecks of the concrete defects.

## Disposition

Six required findings were resolved:

1. Git replacement objects could override the requested commit/blob pin. All
   calls disable replacements; captured commit and blob identities are hashed.
2. Path preflight alone permitted symlink swaps. Creation and writes use
   directory descriptors with no-follow/exclusive operations and a final inode
   check, rather than reopening paths for writes.
3. Umask did not ensure private/read-only storage. Explicit700/600 creation and
   final500/400 permissions protect ordinary local custody.
4. Ordinary-directory substitution required a trusted-parent boundary. Ancestors
   and final parent now require trusted ownership and no shared write grants;
   malicious same-UID/root processes are explicitly outside that boundary.
5. Tree payloads were an unchecked provenance edge. All traversed tree objects
   are parsed from captured, independently hashed bytes rooted in the pinned
   commit; `ls-tree` output is no longer the path-identity authority.
6. macOS ACL grants can bypass mode bits or expose inherited entries. Descriptor
   ACL checks reject allow/unsupported entries, permit deny-only/no-ACL state,
   and run on custody ancestors and created files/directories before data writes.
   Unsupported platforms or retrieval failures reject.

Final static recheck: **zero remaining required defects under the stated
ownership/security boundary**. Ordinary read-only permissions are not protection
from a malicious owner who can chmod or replace their own repository/files.

## Evidence and limits

Nine new fixture methods pass from an unrelated working directory. They include
commit/blob replacement, corrupt-tree identity, dirty checkout/different HEAD,
file membership, coherence, exclusive outputs, symlink swaps, shared-writable
parent rejection and permissive umask. A real macOS ACL fixture changes only its
own temporary files; it verifies allow/inheritance rejection and deny acceptance.
Initial ACL absence handling and deny-delete fixture cleanup failures are
recorded; the latter required removing only the test parent's ACL before its
temporary cleanup. Original user ACLs and experimental environments are unchanged.

Actual final03 capture uses the independently pinned8347ae99 writing revision
and a separately hashed captured utility. It preserves20 publication/source
files and all11 companions in private custody. Preparations01/02 remain
superseded evidence. This checks source/metadata coherence, not claim values,
target B runtime, final reports, anonymity, historical training or submission.
See publication_snapshot_preparation_20261008.json for hashes and cost scope.
