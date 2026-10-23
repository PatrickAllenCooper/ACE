# VF mount-alias and resource proposal review — October 23, 2026

Reviewer: Galileo, a distinct bounded read-only agent. Scope: new alias
correction, immediately adjacent path comparisons and proposed 30-minute
resource literals. No model, scientific fixture, installation, remote execution
or repetition of completed suites by the reviewer.

## Initial finding and disposition

One required finding: strict resolution of a missing pin path raised
`FileNotFoundError` before the existing validation test's expected `ValueError`.
The behavior failed closed but broke the validation exception contract.
Correction: normalize `OSError`/`RuntimeError` from strict pin-path resolution
to a descriptive `ValueError`. A new two-case missing-pin method and the exact
affected existing rejection method pass. No other existing suite was repeated
for this correction.

The reviewer found the canonical source/native pins and live code filename
cover the observed compute-node alias while retaining code equality, hashes,
global/builtin/closure/reference checks. Duplicate canonical keys, external
backing, symlink leaves and wrong hashes still reject. No additional
raw-versus-resolved comparison was identified that predictably fails on this
specific alias along the inspected qualifier, bootstrap, checkpoint guard,
supervisor and immediate replay path.

The scheduler check, supervisor resource assertion and qualifier assertion all
require 30 minutes/1,800 seconds, one CPU and 3 GiB. The child deadline is1,750
seconds, preserving50seconds nominal headroom. This is consistent preparation;
completion within the cap remains unmeasured. Frozen07 input hashes match the
terminal custody inventory and those sources remain unchanged.

## Final static delta

The reviewer confirmed the exception-contract finding resolved and reported
**zero remaining required findings**. The two new missing-path cases cover
missing-only and valid-plus-missing maps. The recovery guidance correctly
distinguishes observed alias failure, unfinished inventory work and proposed
30-minute allowance; its resource arithmetic and deadlines are consistent.

The artificial checks and static review do not qualify the actual target
runtime or scientific replay. No new attempt is authorized by this review.
