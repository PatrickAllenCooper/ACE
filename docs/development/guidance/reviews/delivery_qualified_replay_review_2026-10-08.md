# Qualified replay prelaunch review — October 8, 2026

Reviewed by Volta; final residual static recheck completed October 9 UTC.
No scientific execution, target-library imports, edits or messages by reviewer.

## Required findings and disposition

1. External package namespace paths could admit extra runtime/base directories. Closed: authenticated external initializer and exact parent-only namespace rule.
2. Pinned inventory JSON did not establish live environment bytes. Closed: full file/link membership and byte revalidation before first checkpoint, with loaded-module hashes bound to that inventory.
3. Output custody and same-byte validation were incomplete. Closed: descriptor propagation, nonblocking no-follow regular-file reads, exclusive/fsynced output buffers, captured manifest, same-buffer validation/hashing, and separated terminal custody failures.
4. Executed-main filenames misidentified code. Closed: independently pinned actual bootstrap, captured source identities, actual canonical main namespace for CLI, separate guard module state.
5. Historical supervisor could leave descendants after leader exit. Closed: separately authored Linux subreaper supervisor, no preexisting children, owned process-group cleanup after every outcome, bounded kill/reap iterations, first-primary and secondary failure separation.

Residual recheck found three descriptor issues and two cleanup issues under findings 3/5; all corrected. Final static recheck: zero required remaining defects. Historical workers, failed preparation and frozen candidate remain unchanged.

## Verification scope

29 focused methods pass from unrelated cwd:14 provenance/inventory methods,5 fabricated bootstrap/first-loader ordering cases,10 cleanup/descriptor cases. Cleanup syscalls are mocked on Darwin; bootstrap /proc pathname projection is emulated. Real temporary descriptor retarget/symlink/FIFO and inventory corruption checks run. No actual Linux child custody, PyTorch native object, complete target runtime or numerical replay is qualified by these checks.

Official PyTorch2.9.1 sources corroborate the naming distinction: [native parent attaches compiled_autograd](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/dynamo/init.cpp) and [implementation module name is autograd_compiler](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/dynamo/python_compiled_autograd.cpp). Installed identity still requires the exact native parent/attribute object plus independently pinned _C and libtorch_python backing bytes. Unknown/substituted fileless modules remain rejected.

Qualification and replay share only the already authorized single 1CPU/3GiB/15minute allocation. No additional preparation, installation, fits, optimizer updates, new responses, rescue or retry. Actual supervised runtime and scientific replay remain untested at this prelaunch review.

## Launcher/freeze integration follow-up

A further exact symlink allowance canonicalizes the approved base interpreter
spelling on the execution node as well as the observed target; the frozen ELF
hash remains required. Narrow static review accepts this correction.

Final-01 is preserved/unsubmitted. Launcher review requests explicit post-trust
failure custody before worker handoff and authenticated remote claim presence.
Those are packet preparation gates, not permission to submit yet.
