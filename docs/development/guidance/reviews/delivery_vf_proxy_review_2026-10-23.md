# PyTorch VF proxy correction review — October 23, 2026

Reviewer: Gibbs, distinct bounded static implementation review. Scope: the new VF handling in `qualify_delivery_replay_runtime.py`, pin forwarding in `delivery_replay_checkpoint_guard.py`, and fabricated checks in `test_delivery_vf_proxy.py`. No reviewer executions, model imports, remote commands or edits. Original studies, completed scientific reviews and failed launch06 were not repeated or changed.

## Initial required findings and dispositions

1. **Unchecked globals could replace proxy behavior.** The original draft checked method code and selected globals but allowed a shadowed `getattr` or `super`, and could miss a function's captured builtin dictionary. The correction rejects global shadows, requires the captured builtin dictionary and creator mapping, and compares effective `getattr`/`super` bindings to originals captured before dependency imports. New regressions cover both global names and a function created with substituted builtins, even after its global mapping is restored.
2. **Instance attributes could bypass authenticated methods.** `proxy.relu = replacement` would avoid `__getattr__`. The correction requires exactly normal empty module metadata plus `vf` and rejects extra attributes or altered metadata. A new callable-shadow regression verifies rejection.
3. **Native provenance exceeded what reference equality proves.** Equal `proxy.vf` and `torch._C._VariableFunctions` references plus file hashes do not prove that the object was executed from those native files. The correction deliberately narrows this guarantee to fresh-import reference continuity and backing-file integrity. A one-time capture after fresh dependency imports is retained in the same qualifier through first-checkpoint revalidation. Joint replacement after capture and recapture are rejected. Each VF record explicitly says `native_execution_attested: false`. Arbitrary same-process tampering before initial capture, or modification of the checker itself, is not covered.

## Final bounded recheck

Gibbs found **zero remaining required issues**. All three initial findings are closed within the stated scope. The recheck confirmed capture before inventory and reuse of the same qualifier at the first checkpoint gate. It did not establish actual Torch compatibility or numerical replay qualification.

The coordinator ran four new review regressions and two affected integration checks from an unrelated directory; all six passed. One new regression executes the exact official PyTorch2.9.1 `_VF.py` Python source against fabricated modules, with fabricated native bytes never loaded. It does not import real Torch. The earlier nine artificial methods passed on the pre-review source; that historical evidence and both earlier failed prototypes remain distinguished. No claim is made that all13 methods were rerun on the final source.

## Remaining execution gate

Independently authenticate installed target `_VF.py` and both native pins; do not substitute the upstream reference digest for an installed-file observation. Freeze a new attempt with source, input, runtime and resources; reconcile journals and live readiness before any authorized submission. A real exact-runtime supervised full640checkpoint/240primary-cell/48000cached-response replay with zero new updates/responses remains required. The failed33640933 approval is consumed.

Primary source: [PyTorch v2.9.1 `_VF.py`](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/_VF.py). A local reference and the upstream license are retained under `results/delivery_vf_proxy_preparation_20261023/references`.
