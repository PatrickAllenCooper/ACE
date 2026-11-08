# Scientific realization terminal collector source review — 2026-11-08

**Current disposition: 0 required corrections; R1 closed by the collector05 delta below.** The initial disposition was 1 required correction and is preserved below. This is a static collector review, not terminal custody acceptance, execution authorization, or a repeat of the packet/component reviews.

## Required finding

**R1 — Join the authenticated remote intent to the packet and namespace lineage before asserting `current_lineage_verified`.** In `pull_terminal.py`, the local intent is read without an independent pin (lines 213–216). Its packet/root/input map is checked, and its claim pin subsequently supplies the expected registry identity. Lines 311–312 authenticate only the *bytes* of the remote intent against the independently pinned submission result; they never decode that authenticated intent or compare its contents with the local intent. The two independent anchors—current packet and original submission result—therefore remain unjoined. Matching current input bytes, journal hashes, and both copies does not establish that the intent bound by the actual submission result names that packet, those inputs, or those namespace claims. Nevertheless line 358 can report `current_lineage_verified=true`.

After preserving raw custody, decode only the copied remote **coordinator** intent at `result['intent_sha256']` and require its original fields to equal the local intent, including packet/root/input pins, source/packet commits and local namespace claim pin. The launcher adds exactly `remote_utc`, `fresh_readiness`, and `remote_namespace_claim_sha256`; permit those explicit additions and require the remote claim pin to match the captured registry. Report disagreement through the existing retained-custody error/disposition path. No private output decoding, new framework or acceptance schema is needed. This correction is required before relying on the collector's lineage disposition.

## Reviewed bytes and evidence

- Authoring root: `/tmp/ace-scientific-realization-packet-authoring-w8rcim55`.
- `pull_terminal.py` and retained `collector04/source/pull_terminal.py`: SHA256 `9658f8c8fc9aab7b67334f8c97a12e82402b7c058d2889449185e621c12df73b`; identical at review completion.
- `collector04/static.json`: SHA256 `5f5278687d14dcaba214711c561464b339c0a45e950cc464cdc6808fc1c6322e`. Retained author evidence says AST/compile only, syntax passed, collector not executed, zero remote actions, no tests; elapsed 0.004860667 seconds. This is syntax evidence only.
- Read the final collector04 handoff and the launcher's remote-intent augmentation solely to establish the collector interface. Launcher/monitor changes and concrete packet review remain separately owned.

## Remaining scoped checks

The collector captures the entire actual ROOT as `remote/launch`, including `realization/`, inputs, journals, logs, partial output and empty directories; the remote registry and local claim are separately retained. Both coordinator histories are copied. Both existing controller locks are held; terminal roots and file writes are exclusive, with no retry/resume or submission path.

The remote observer requires a conclusive terminal row and no current/related queued ACE job. Before/after remote inventories, terminal accounting, complete related rows and journal maps must agree. All input pins are checked after transfer. Prior accounting must equal the fixed 38-row baseline, and journals must equal the original 78 plus the current intent/result (80). Two independent local copies and their complete inventories are compared. Symlinks/special files are rejected; raw hardlink relationships are recorded and copied names become independent regular files.

Failed terminal states, missing success markers and partial outputs remain eligible for raw custody. Transport or binding failures retain available bytes and failure inventories without granting retry. Output JSON is not decoded; only coordinator metadata and scheduler observations are interpreted. Resource/profile disagreements are separately explicit in `allocation.errors`/`resource_contract_matched`; marker presence and exit status do not establish scientific success. Acceptance, realization acceptance and scientific qualification remain false on every disposition.

No optional suggestions. No collector, embedded observer, project source, tests, model, tokenizer, remote command or scientific work was executed during this review. Only this review document was written; author files were not changed.

## Focused collector05 disposition — R1 resolved

Reviewed only the delta from retained `collector05/before-pull_terminal.py` (SHA256 `9658f8c8fc9aab7b67334f8c97a12e82402b7c058d2889449185e621c12df73b`) to `collector05/pull_terminal.py` and the current authoring `pull_terminal.py`, both SHA256 **`7fdbeee13c12248b54b820591e5431f3dc119800a83a9e5d5bca7d37fd6494c6`**. The diff adds only `validate_remote_intent` and its guarded call after full raw custody and registry capture.

The new function authenticates the supplied bytes against the expected remote intent hash, rejects overlapping augmentation keys in the local intent, and requires exactly all local fields plus the three explicit remote additions. Canonical JSON byte equality checks every original local field, including nested pins and commits, without conflating booleans and integers. Both remote and local claim fields must equal the valid expected claim pin; `remote_utc` must be a nonempty string and `fresh_readiness` a dictionary. The caller additionally reads the copied intent through the existing bounded, independently pinned metadata reader. Existing registry checks bind that claim pin to both captured registry copies. Validation disagreements append to the existing errors list, preserving custody and preventing a true lineage disposition. Only coordinator metadata is decoded.

Retained evidence: `collector05/checks.json`, SHA256 **`3e0ccbf3cf10e856d316f876f6fa9ee94db78b15720d9f962dea43c15b83858f`**, names the canonical positive and six negative cases (packet, source, claim, extra, missing, type) and binds the new source hash. Parent reports these new pure checks passed; the record reports 0.000263 process CPU seconds and 0.000262583 elapsed seconds, scoped to the coordinator metadata join without collector main, network or scientific outputs. No tests were rerun for this independent review.

**Final scoped disposition: 0 required findings; R1 resolved.** No optional suggestions. No collector/observer execution, job monitoring, remote action, private-output decoding or terminal acceptance was performed. Actual job state and future custody remain outside this source-delta disposition.
