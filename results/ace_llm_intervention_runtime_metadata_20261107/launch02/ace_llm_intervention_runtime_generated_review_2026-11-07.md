# Runtime generated-module diagnostic v2 source review

Date: 2026-11-07

Disposition: **0 required source or fixture corrections found in this bounded delta review.** The reviewed source is suitable for the separate packet review. This is not an actual-runtime qualification or permission to reinterpret the failed predecessor.

## Scope and evidence

Reviewed the changes from diagnostic v1 to v2 in `scripts/research/qualify_llm_intervention_runtime_metadata.py`, the six new delta fixture methods, the updated guide, and their saved preparation evidence. The scope is the generated Torch module, its temporary directory and import provenance, the Transformers LazyModule version observation, and preservation of failure and resource controls. Earlier helper, source and packet reviews remain separate and unchanged.

No diagnostic, package source, test, model, remote command or launcher was executed for this review. Inspection used source reads, static AST comparisons, JSON parsing, byte hashes and custody membership checks. No implementation, ledger or historical evidence was changed. Only this review was authored.

Final reviewed pins:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/research/qualify_llm_intervention_runtime_metadata.py` | 45,002 | `31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329` |
| `scripts/research/test_llm_intervention_runtime_metadata.py` | 32,274 | `6470197dcfb0482eb0819ffa27f0c813260818393934303ccfdc5fcb4cccd634` |
| `docs/development/guidance/ace_llm_intervention_runtime_metadata_2026-11-07.md` | 14,633 | `2d5be30175275f0761d02defa5e532876ebf40d0f0d19144cb26b8772bd91393` |

The preserved `preparation02/before/` source has SHA-256 `9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49`. Its test source has SHA-256 `05b22530a8a2c28e08fc9a81df07fcb5b53eb4f95e54a1ea10f183ca85f49b5f`. These establish the delta boundary.

The completed preparation is at `results/ace_llm_intervention_runtime_metadata_20261107/preparation02` and `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-preparation-20261107-02`. Both copies have exact membership and matching bytes: **244 non-directory objects, comprising 243 regular files and one fixture symlink, with 622,920 regular-file bytes including the top-level manifest**. The manifest lists 243 members and excludes itself. The retained symlink was checked by link-target text, without treating its target as a regular-file member.

| Evidence | SHA-256 |
| --- | --- |
| `preparation02/manifest.json` | `41f61393c50dfc331376ad6a1d73429ca7fb7a3f256ece9ab4a580476f02141e` |
| `preparation02/closeout.json` | `142dbc09539f221bcf50f7116d23451165a293cc1980c23f3c80d5d8f3d6b501` |
| `preparation02/attempt01/manifest.json` | `a9f994b55a6c41fd48b4186e290262ed6f42cb1081160287e0b16097fa0cdbb7` |
| `preparation02/attempt01/receipt.json` | `83311b37d35b5ec37cecf4524631fa55d873f64b62b4ecace10c5c6a9c522ea4` |
| `preparation02/attempt01/command.json` | `95c01137b757cbbe72377c879e478f746a02eaf3342b17fa110dcc76e9bbd21c` |
| `preparation02/attempt01/source_manifest.json` | `81a97c12b9dfb046bad9dcdb60cedfd845fb77bac27b67914da709b3e745c7ce` |

## Generated-module contract

The exception is specific to `_remote_module_non_scriptable`. It does not allow arbitrary temporary modules or an arbitrary `/tmp` search path.

The expected bytes are reconstructed by static parsing of independently pinned Torch files. The reviewed non-scriptable generator contains seven fixed string substitutions. The selected template is the pinned prefix plus the pinned remote-forward template. Parsing obtains string literals and the exact substitution dictionary without importing or executing either source. Changed template content, changed generator content, unexpected substitutions or nonliteral values fail the contract. The CUDA-related branch is part of a generated function definition. This diagnostic does not call that function or establish GPU work.

The six provenance source pins match their saved captured bytes and the original archive-manifest entries:

| Captured source under `lib/python3.10/site-packages` | SHA-256 |
| --- | --- |
| `torch/distributed/nn/__init__.py` | `4dae25d09af680e2bb79047e7b7dacb5668328f3a557f2a28b4484c7275793f4` |
| `torch/distributed/nn/api/remote_module.py` | `d51779fdbafb1ca1d99d002e6f32cf17bc1fef730a6251e3ef0278d8078bd112` |
| `torch/distributed/nn/jit/instantiator.py` | `567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615` |
| `torch/distributed/nn/jit/templates/remote_module_template.py` | `0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc` |
| `transformers/__init__.py` | `7eb0743ed843f24c1d4e8b8daf4b18249cb81403b4571a70c00be0a4c0a67bd4` |
| `transformers/utils/import_utils.py` | `66b2ac40be2c538dbd9409370c1f6bf769934b8b232edac365e16299c5ec1108` |

The supervisor creates a new private `worker-tmp` beside the staged prefix. Its path and stat identity are added to `stage.json` before that receipt is pinned. The fresh child receives matching `TMPDIR`, `TMP` and `TEMP`. Before package imports it checks the pinned root identity, canonical path, ownership, mode, emptiness and `tempfile` choice. This prevents the source change from silently admitting an unrelated preexisting temporary directory.

After import, the proof requires the authenticated instantiator's retained `TemporaryDirectory` object to name the same direct child directory as its exported template path. The private root must contain exactly that directory, which must contain exactly the expected generated source file. Canonical-path and `lstat` checks reject symlinks. Bounded no-follow reads check file identity and unchanged stat information across capture. Directory mode and ownership, regular-file type, link count and write permissions are checked.

The generated source must match the reconstructed bytes exactly. The loaded module must have the expected name, file, `ModuleSpec` name and origin, non-package status, and exact `SourceFileLoader` name and path. The three generated functions must have the expected function type, module name and code filename. Pinned Torch source-module files and specs are checked independently. These are concrete origin and byte checks, not a general proof of arbitrary in-memory behavior.

The post-import `sys.path` must equal the baseline followed by exactly this one verified directory. The mixed-module inventory must contain exactly the expected module and path. Additional aliases, paths or modules fail. Native-library path rules remain unchanged. Evidence keeps the observed mixed paths and modules rather than hiding the exception from the record.

This is a post-import provenance check within the stated trusted-owned process model. It is not pre-execution authentication of the generated module, a sandbox, or protection from a malicious same-UID or root actor. The guide states that limit.

## Transformers version and failure handling

The new version observation authenticates the pinned Transformers package source, the pinned `import_utils` source, their module/spec origins, and the exact `_LazyModule` class. It reads `__version__` from the plain `_objects` dictionary using direct dictionary access. It does not trigger lazy `__getattr__`, use package metadata as a substitute for an imported value, or accept an arbitrary object exposing a version attribute. Missing, non-string, incorrect or conflicting values fail. A directly present version, if any, must agree.

The target remains actual Torch `2.9.1+cu128`, distinct from distribution metadata `2.9.1`. Actual NumPy remains explicitly `2.2.5`, while both existing NumPy metadata directories remain preserved and expected. No deletion, relocation or repair of original metadata or caches is introduced.

Contract, baseline and temporary-root state are initialized before fallible observation. Unsafe generated paths fail before content reads. Safely captured source and provenance are preserved in the bounded evidence objects, including the fabricated content-mismatch failure. A missing or rejected proof cannot satisfy worker validation. The v2 terminal requirement and version check prevent a v1 result from satisfying v2 success. Existing terminal/candidate publication and their digest relationships are retained.

The temporary directory is ephemeral. Persisted source bytes, stat observations and receipt pins document the copy that was observed. They do not claim that node-local files survive cleanup.

## Saved fixture evidence and unchanged limits

The saved command selected only the six new `MetadataDeltaTests` methods. The eight original methods were retained and were not rerun. The new cases cover canonical generation, exact LazyModule version access, additional or forged origins and paths, content/template/substitution mismatches, ownership and symlink failures, inventory and temporary-owner mismatches, invalid LazyModule version cases, and fabricated worker success/failure publication. The success and content-failure evidence maps and terminal/candidate equality match their saved objects.

The fixture bootstrap blocks imports of the real NumPy, SciPy, Torch and Transformers packages. Fabricated modules and arithmetic/stdlib function code supply the required observations. The saved result supports the new guard behavior under those fixtures. It does not establish actual Linux package generation, actual runtime imports, real worker cleanup or measured scheduler behavior.

The one saved test attempt records six passes, return code 0, **0.284557 child CPU seconds**, **0.330680375 elapsed seconds**, **27,426,816 peak child RSS bytes**, and **0.001369 parent CPU seconds**. These are preparation-test costs only. They exclude implementation, review and copying and are not whole-cycle costs. No reviewer rerun was performed.

Allocation, child command, cleanup and terminal publication logic retain their previous controls. The bounds remain one CPU, 3 GiB, 600 scheduler seconds, 580 supervisor seconds, 180 seconds for successful child completion, 185 seconds for the outer child deadline, and five seconds for cleanup. Stage and worker process RSS limits remain 512 MiB and 1.5 GiB. Those process limits are not a guarantee about total cgroup memory or file cache. The prior near-limit batch RSS remains unresolved evidence, not an OOM diagnosis.

The prior 144-file failed diagnostic is not reclassified. It did not save this generated-module proof or an authenticated LazyModule version value. New source reasoning can explain why the correction is appropriate, but cannot supply observations missing from that run. No actual runtime, model service, model weights, scientific result, historical training execution or new resource allocation is qualified by this source review.

Required corrections remaining: **0**. Exact launch02 packet assessment is separate.
