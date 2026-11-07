**Independent bounded static runtime archive review, November 7, 2026.**

**Required findings: 2, both P2.** The helper needs the two corrections below
before its complete-copy and staged-link checks can be relied on. Neither
finding requires importing the real runtime or changing the original environment.

Reviewed the current [helper](/Users/pat/code/ACE/scripts/research/llm_intervention_runtime_archive.py),
the new [artificial tests](/Users/pat/code/ACE/scripts/research/test_llm_intervention_runtime_archive.py),
and the source copies, receipts and recorded stderr in preparations 01 and 02.
This is static source and saved-evidence inspection. No tests, helper functions,
runtime imports, remote commands, GPU work, models or commits were executed.
Only this review document was authored. Earlier component reviews were not
repeated. Caller deadlines, process supervision, actual imports and qualification
of a relocated Conda runtime are outside this review.

**P2 R1. Directory enumeration errors can silently produce an incomplete archive.**

At [line 91](/Users/pat/code/ACE/scripts/research/llm_intervention_runtime_archive.py:91),
`os.walk` has no `onerror` callback. Its default behavior skips a directory whose
`scandir` fails. The parent iteration has already recorded that directory with
`lstat`, so the later object and byte counts describe only the successfully
enumerated subset.

Concrete counterexample: an otherwise readable source contains a nested
`lib/blocked` directory owned by the caller, with mode `000` and a file inside.
The parent directory is searchable, so `lstat(lib/blocked)` succeeds and records
it. Traversing `lib/blocked` raises `PermissionError`, which `os.walk` ignores.
The archive then contains that directory but none of its contents. Its recorded
stat remains unchanged. `pack` writes a successful manifest, and `stage` can
return success for this incomplete copy. A filesystem enumeration error can
cause the same omission without any concurrent or malicious actor.

Required correction: propagate traversal errors, for example with an `onerror`
callback that raises the received exception. Retain the exclusive partial output
as on other failures and do not publish a successful manifest. A focused new
artificial regression should cause a nested scan to fail after its parent was
enumerated, then require failure and absence of a success manifest. Mocking the
scan error avoids depending on the test runner's privilege or filesystem ACLs.
The current unsupported-object and byte/count-cap tests do not cover this case.

**P2 R2. Check symlink containment after constructing the entire link graph.**

At [lines 242 onward](/Users/pat/code/ACE/scripts/research/llm_intervention_runtime_archive.py:242),
each symlink is created and resolved immediately. A later symlink can change the
resolution of an earlier link. The earlier link is never checked again.

Concrete counterexample: a manifest and matching archive contain two sorted
symlink records, `a -> b/..` and `b -> .`, both with mode and source mode `0777`.
There are no files, so `file_bytes` is zero. All current record, ancestor, lexical
target, archive size and digest checks can pass. When `a` is created, `b` does
not yet exist. Non-strict resolution of `b/..` yields the output root, which
passes containment. When `b` is created, it resolves to the output root and also
passes. Now resolving `a` follows `b` to the output root and then follows `..`
to its parent. `stage` nevertheless returns success with an escaping link.

This counterexample concerns the staging validator's semantic guarantee. It
does not assert that an unchanged source accepted by the current packer's full
source-link resolution already contains this graph. Independently pinning the
manifest proves which bytes were supplied, not that their link graph is safe.
It also does not require a concurrent writer or hostile same-UID process.

Required correction: create all deferred symlinks first, then resolve and check
every link against the canonical output root, before returning success. Keep
the existing lexical rejection and directory-ancestor checks. Retain partial
output on rejection. A focused new artificial regression should use the two
links above and valid archive/manifest pins, then require rejection. The current
simple escaping-link tests operate at packing time and do not cover this
forward-reference staging case.

**Other inspected behavior and limits.**

Regular-file digests are computed from the stream actually supplied to
`tar.addfile`. The declared byte count and device, inode, mode, size, modification
and change timestamps are compared at the open/capture boundaries. Final file
opens use `O_NOFOLLOW`. The archive is uncompressed, and its completed bytes are
hashed before writing the manifest. These checks support captured-byte custody.
They do not constitute an atomic snapshot of a concurrently changing environment.
No such adversarial snapshot guarantee is assumed for this trusted-owned-root
review.

Manifest validation requires exact record fields, canonical relative object
names, sorted unique membership, directory ancestors, file-byte totals and
per-file digests. It imposes the stated 100,000-object and 16-GiB file-byte caps,
with a separate archive-size allowance. Staging authenticates the independently
supplied manifest digest and the archive digest before creating output. It
compares members in order, streams regular-file contents, rejects a further
visible member and delays link creation until regular files are written. These
checks do not cure either finding above.

Directory SGID is intentionally permitted in `source_mode`, while the copied
directory's stored and applied mode contains only `0777` bits. Other special
permission bits are rejected as written. This is the declared copy behavior,
not a defect. Canonicalizing the existing output parent in both entry points
correctly addresses the macOS `/tmp` alias mismatch without following an
existing final output object. Exclusive creation still rejects an existing
destination. Exceptions do not delete the created archive or staged tree.
Successful staging explicitly reports `runtime_qualified=False` and
`model_loaded=False`.

The caller still owns the independent pin, exclusive locations, terminal
failure recording and process/resource supervision. This review does not
authenticate a real Conda tree, prove executable relocation or authorize a GPU
retry. No additional required defect was identified in this bounded inspection.

**Preparation evidence and exact reviewed bytes.**

Preparation 01 records six artificial methods with five passes and one error in
the roundtrip method. Its stderr identifies the macOS alias containment failure.
Its recorded child CPU is 0.084278 seconds and elapsed time is 0.090649042
seconds. Preparation 02 changes only the two existing-output-parent
canonicalizations in the helper. The test file is unchanged. Its receipt selects
only the affected roundtrip method and records success, 0.060828 child CPU
seconds and 0.066007708 elapsed seconds. Thus the retained evidence supports
five earlier passes plus the corrected roundtrip pass, not a new six-method run
of the corrected helper. Those are author-run measurements, not review costs or
whole-preparation costs. Neither preparation covers R1 or R2.

Verified SHA-256 identities by reading the local files:

- Current helper and preparation 02 helper:
  `bf692be7a6aa21a540093bad0a993e19e1110212163cbba2631ebf7ff8bb2145`.
- Current test file and both preparation test copies:
  `887ce4eaacfa0080cd74d592960aa6f1b1d8e502ff96cabc0699d310b363eba3`.
- Preparation 01 helper:
  `a14283e07c3cc9d1e0de7a6c45cb04000321d4a064babb4c90444fe815c4366b`.
- Preparation 01 receipt:
  `afadd0392866bd62d0404007e8f326d8be3c997620c61f9f384d2e847c611d04`.
- Preparation 02 receipt:
  `6d52a089cf8a058184df739b442a9335d94924125689c16e2391ebe79f6469a1`.

Preparation 02's predecessor pin matches preparation 01's receipt. The review
disposition applies to the current helper hash above and must not be relabeled
as acceptance of a later correction.
