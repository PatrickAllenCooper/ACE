# Six/Cython compatibility preparation review — October23,2026

Reviewer: Mencius, distinct from backend and packet reviewers. Scope only the new six.moves/cython_runtime handlers, first-checkpoint pin forwarding and diagnostic-before-capture ordering. Original workers, backend/VF checks and scientific tests were not re-reviewed. No remote command, allocation, native/model import or scientific execution occurred.

Initial review found two required defects within the declared Python provenance scope:

1. Matching methods by their own code names allowed swapped importer attribute bindings. Correction maps each actual attribute to pinned source code, with explicit Python private-name mangling and the source's `get_source = get_code` alias.
2. Instance-level importer method shadows and unapproved moves-module behavior hooks could bypass checked class methods. Correction requires the exact importer instance fields and permits only standard module metadata, declared lazy-cache names and registered child names, rejecting unexpected behavior hooks.

The new focused regression rejects swapped find methods, an instance find_spec shadow and a moves.__getattr__ override, while retaining legitimate lazy range caching. A separate new sentinel regression checks exact mapping/descriptor/ModuleSpec types before any potentially dynamic field access and retains the normal positive. Both pass. The initial seven new pure-six/artificial-Cython methods and affected pin-forwarding method also passed; the reviewer independently ran those tests before reporting the two findings. Native creation attestation remains explicitly outside the inert Cython record's scope, with no required finding in that contract.

Final bounded static delta review: **zero required findings remaining; local preparation review gate cleared**. No final-delta tests were rerun by the reviewer. Actual target runtime and full saved-model replay remain unqualified.
