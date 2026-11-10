# GPU02 attempt03 episode/session mapping diagnosis

Required finding: **R1 — the audit aid compares the configured logical root to a producer-defined resolved path.** This is a checker compatibility defect in attempt03, not evidence of a cross-world or decision mismatch. A narrowly reviewed correction is supported by saved, independently pinned path-binding evidence. No scientific source or actual artifact should change. Full GPU acceptance is still pending.

The failed report is `27a512367ba45499ca7b6339c086682a84c0129a11eca4292217b5a897c45466`. Aid `b2e48a8b65cd56583f2673c6651c5adc9fffe3d086ed2f65935127efe213fa43`, line 596, constructs `episode_directory` from the logical packet campaign root. Its first mapping, `cpu/cpu/worker/artifacts/campaign/policy-05/service/request-0000.episode.json`, pin `d01e39f024324008dd5033e52e91cc9cca4553092fc05960724ca091c8ca6623`, differs only in that path: `/scratch/alpine/…` versus `/gpfs/alpine1/scratch/…`.

## Source and historical binding

The captured sources were independently hashed against packet pins and the source bytes inspected. `llm_intervention_campaign.py:132–142,191–255` uses the supplied campaign root, ordered policy directory and ordered world directory. `llm_intervention_paired_episode.py:149–212` binds that world's task and passes `output/"episode"`; `llm_intervention_episode.py:109` passes that directory and decision index to `choose`. `llm_intervention_persistent_policy.py:455–471` explicitly records `str(Path(directory).resolve())` with task, identity, session and decision identifiers. The frozen scientific evidence validator at `llm_intervention_scientific_service_evidence.py:400–415` requires an absolute path without parent traversal, the exact policy/world/episode suffix, one consistent recorded campaign root, and exact remaining mapping fields; line 505 publishes the recorded root. These semantics predate the actual run.

The independently accepted CPU parent and worker `start.json` records are identical, 657 bytes, SHA `02ff21af91dec84b82bc0e46411c479eeb27cf694c01dccbad676fb6e15527f1`. Their recorded binding is:

```
logical:  /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu
resolved: /gpfs/alpine1/scratch/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu
mount_alias: true
root_device: 63
root_inode: 8388611
```

Both original CPU local copies match. Each record's hash is present in CPU acceptance `3c5ce7375d0c64292553ded6ad3078e634745eb7996d83fd102e344eeab3f063`, prior full archive receipt `f0f8a52c7f823b5a07041faf531382afa4d7c0364452413bbb17bd916dfc38e6`, and current GPU full archive receipt `e59bca378f54b0225eb8c74b73bfeebdad830faf5d3fd62fedfc56593de8e27c`. The start records bind CPU packet `ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344`; this is also the current GPU packet's accepted CPU predecessor. Its captured CPU producer is `ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893`.

That CPU producer's `_output_path_binding` (lines 83–104) admits only identity or the exact known mount substitution, checks resolved root equality, physical-root self-resolution, directory existence and equal device/inode, and rejects descendant symlinks. Main records the result before authentication at lines 787–798. The current GPU packet's campaign path equals the accepted logical CPU output plus `/worker/artifacts/campaign`. Every actual recorded mapping root and all three pinned original validator roots equal the recorded resolved CPU output plus that same suffix.

This is saved historical proof joined to actual GPU mappings and custody; it is not a fresh filesystem observation or a general assertion that arbitrary aliases are interchangeable. No remote check was necessary or performed. Local `Path.resolve()` was not used to reinterpret remote path strings.

## Independent actual metadata joins

The bounded stdlib inspection checked all **1,698** mapping records: 576 frozen, 546 SFT and 576 DPO, across the original 24 ordered development worlds for each policy. It found no mapping, world, task, decision, session or RPC mismatch within this scope. Each mapping exactly matches the derived recorded root and every other field, including identity, task hash, task binding index, decision index, session request ID, status and scope.

The independent checks join each original arm and its pinned receipt to the namespace world roster and source-plan pin, continuous receipt session ranges, per-world task binding and previous-task chain, the accepted CPU service identity, and the actual frozen validator's per-request world/decision/task/ordinal fields. They also match the opaque request task and full envelope hashes, worker request identity/ordinal/envelope metadata, generation ordinal/task/envelope, observed response identity/request/envelope, accepted response hash, and opaque completion bytes across generation/wire/accepted records. Mapping membership and session ordinals are complete, with no extra mapping member. Original validator evidence pins are:

- frozen: `215ef71ec792274331e17161d2122f4e58a48370db5fe2929869d4833515ab56`
- SFT: `036be71cb8eeb68c6c35a3046286caee72421fa1d01870cce1c1538fa041a5ad`
- DPO: `1e0e66d344ff6ec78c0df00a19183ea59bb45e6611f04469413596aa647318c4`

These are also pinned by the actual terminal's service-validation records. The 546 SFT count is reported here solely as observed membership; this diagnosis does not classify stopping decisions or scientific utility.

## Required correction boundary and remaining gates

The aid may derive one exact recorded campaign root from independently pinned accepted CPU path-binding records, after checking their packet/acceptance/full-custody joins and logical-root equality. It must reject absent, conflicting, tampered or unrelated bindings, unknown path substitutions, changed suffixes, traversal, and every nonpath mapping mismatch. Do not accept an arbitrary consistent root, use suffix-only matching, or replace prefixes broadly. If supplemental proof is added, preserve the original config, batch receipt and failed attempt; capture the added proof with explicit independent pins and lineage.

Keep the original source/packet/namespace identities, all 72 arms, all service/RPC/decision checks, opaque state-feedback and executor joins, scientific response accounting, runtime/GPU/cleanup gates, full historical custody, and actual transfer-continuation control joins unchanged. This note diagnoses the failed path predicate; it does not replace the corrected aid or independently accept the complete GPU execution. No retry, scientific rescue, new response, model call, tokenizer, project callable, test rerun, remote operation, allocation or source edit occurred.

The original failed aid attempt remains unchanged. Two reviewer inspection setup errors are retained: an incorrect external CPU capsule date, then an assumed worker-request record shape. The final inspector uses the frozen worker-request metadata schema; neither setup error was an actual execution defect. The successful one-off inspection and every earlier script/log/pin file are retained. The earlier independent source-roster diagnosis remains separately preserved at `results/ace_llm_intervention_scientific_gpu_audit_preparation_20261109/actual-roster-consistency01`, note SHA `86d0823fc48fd3f58fb3c7ad8a58af5d260c77eeb40388289435b6485be4081c`.

Disposition: **one required aid compatibility correction; no scientific mapping defect found in this narrow saved-evidence scope. No GPU acceptance issued.**
