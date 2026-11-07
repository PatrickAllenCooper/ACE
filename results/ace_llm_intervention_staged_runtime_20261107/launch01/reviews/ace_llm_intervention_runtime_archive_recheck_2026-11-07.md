**Independent delta-only runtime archive recheck, November 7, 2026.**

**Required findings remaining: 0. Original findings R1 and R2 are closed** for
helper SHA-256
`7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e`.
The [original review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_runtime_archive_review_2026-11-07.md)
is preserved unchanged, including its historical two required findings. This
addendum records the correction disposition rather than rewriting that history.

Compared only the helper and test deltas against preparation 02, read the
preparation 03 receipt and saved stdout/stderr, and checked their SHA-256
identities against current source. No tests or helper functions were executed.
There were no runtime imports, model operations, remote commands, GPU work or
commits. Only this recheck document was authored. Caller and packet development
remain outside this component review.

**R1 closure: traversal failure now propagates.** The only inventory change is
the new `walk_error` callback, which raises the supplied exception, and its
explicit binding as `os.walk(..., onerror=walk_error)`. A failed nested scan can
no longer be silently skipped. The exception leaves `pack` before archive and
manifest publication. The already-created exclusive output directory is not
removed. This closes the unreadable-directory counterexample without requiring
permissions or ACL changes to the original environment.

The new `test_unreadable_inventory_must_fail_and_keep_incomplete_output` supplies
one parent enumeration and then calls the supplied error callback with a
`PermissionError`. It requires propagation, retained output directory and no
manifest. This is an artificial callback-wiring regression. It does not claim
to reproduce an actual CURC filesystem error. The direct callback binding and
raised exception are sufficient to close the identified static defect.

**R2 closure: containment now checks the completed graph.** The only staging
change separates creation and validation into two loops. All deferred symlinks
exist before any is resolved for containment. The original lexical target,
directory-ancestor and canonical-root checks remain unchanged. For the original
counterexample `a -> b/..`, `b -> .`, resolving `a` now follows the completed
`b` link and reaches the output root's parent. The containment check rejects
before directory-mode finalization or a success return. Created artifacts are
retained on this exception.

The new `test_completed_symlink_graph_rejects_late_escape` constructs exactly
that two-link archive, with valid mode fields, byte/count metadata and matching
archive/manifest hashes. It requires the `staged link escapes` rejection. This
specifically covers the original forward-reference gap rather than relying on
the previously passed packing-time tests.

**Saved regression evidence.** Preparation 03's command selects only these two
new methods. Its saved stderr says `Ran 2 tests` and `OK`, stdout is empty, and
the receipt records exit zero. The test diff adds only these two methods, leaving
all six previous method bodies unchanged. No prior method is selected by the
recorded command, and none was rerun during this review.

The author-run receipt records UTC `2026-11-07T18:28:01.236960+00:00`, cwd `/tmp`,
0.055399 child CPU seconds, 0.059140208 elapsed seconds and 21,725,184 bytes peak
child RSS. These are the two-method invocation's measurements, not reviewer
execution, whole-preparation cost or real runtime qualification.

**Final bindings.** Current helper and test bytes match their preparation 03
copies and the receipt's source pins. Preparation 03's predecessor pin matches
the preserved preparation 02 receipt.

- Corrected helper:
  `7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e`.
- Updated test source:
  `4aebece109bddd3d8584c6fc2e1bd83561d4f401793c187a03e8c0c22dd36508`.
- Preparation 03 receipt:
  `ca69b0d5f569c9486cbe7f97c21d60ee923e1240412ce594abc22ec5869e2d6f`.
- Preparation 03 stderr:
  `cbe4b84d897ac017db0552ccb3a31855ff0fb172bc344c967c4c4c92eb19e624`.
- Preparation 03 empty stdout:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Predecessor preparation 02 receipt:
  `6d52a089cf8a058184df739b442a9335d94924125689c16e2391ebe79f6469a1`.
- Unchanged original review:
  `b912ee1249875fcd5756af1e8915bbc1bd438bab195b1d576b9c6eb6c94fd650`.

No other helper changes occur in the inspected delta. The original trusted-root
scope, declared SGID handling and other review limits remain in force. This
closes the two helper findings. It does not qualify the caller, authenticate a
real archived environment, demonstrate executable relocation or successful
imports, or establish any GPU or scientific result.
