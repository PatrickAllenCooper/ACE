# Independent runtime metadata implementation and packet review

November 7, 2026. **Two required corrections, both in the parent tools. No required correction found in the pinned diagnostic implementation.** This is a bounded static review of the new diagnostic, its saved artificial evidence, and the generated launch01 packet. The packet is not cleared for launch until the two findings below are resolved. No actual runtime or model-service qualification is established.

Only this review document was authored. Review operations read, hashed and compared local saved files and parsed source text. No test, diagnostic, launcher, copied runtime, numerical package or model was executed or imported. No remote operation, allocation, environment change, ledger edit or commit occurred. Earlier helper and results reviews were not repeated. The unchanged helper/probe pins were checked, and the probe's cleanup function was read only to assess its new caller.

## Required corrections

### R1. Write a real newline after remote JSON documents

**P2.** `results/ace_llm_intervention_runtime_metadata_20261107/launch01/launch_once.py:56`, within the generated remote `common` source.

The outer Python string produces remote source whose `f.write` argument is the two-character string consisting of a backslash and `n`, rather than a newline. Static parsing of both source levels confirms this. Every call to the remote writer therefore appends bytes `5c 6e` after a complete JSON document. For example, an otherwise valid empty object would be stored as `{}` followed by those two bytes. A normal JSON reader rejects that suffix as extra data.

This affects `readiness.json`, `submission.intent.json` and `submission.result.json`. The submission path can still proceed because it hashes the intent and prints a separate valid result to stdout. Consequently, successful submission would leave malformed authoritative remote journals, even though local transport receipts can look valid. This is an actual evidence-format defect, not a proposed adversarial threat model.

Correct the escaping so the generated remote writer appends one newline character, or omit the optional trailing newline. Preserve the reviewed launcher bytes as a predecessor and bind the corrected launcher in the eventual committed launch record. The local `save` writer is not affected. No diagnostic source or scientific input change is needed for this correction.

### R2. Do not report verified archive reuse before it has occurred

**P2.** `results/ace_llm_intervention_runtime_metadata_20261107/launch01/pull_terminal.py:62`.

The collector sets `runtime_archive_reused_unchanged` to `True` for every collected terminal job. Its checks authenticate the new input files and scheduler identity, but it does not require a completed stage receipt before setting that field.

A concrete counterexample is a job that passes submission but fails the allocation guard, node-local free-space check or archive digest check. Such a job has a terminal scheduler state and collectible logs, so this collector can finish and assert unchanged archive reuse despite staging never completing. A failure before the supervisor creates `qualification` also takes this path if the other job files are present.

Separate the planned input choice from observed staging. Either label the unconditional field as planned reuse only, or derive an observed-completion field from the authenticated stage receipt and expected archive/manifest pins. Missing or incomplete evidence must remain unconfirmed. This does not require a successful import worker: a failed import can still have completed authenticated staging. Preserve that distinction. The existing `model_service_qualified=False` and `runtime_contract_adopted=False` should remain.

## Scope accepted in the diagnostic and fixtures

The source uses an explicit observation contract for NumPy `2.2.5`, while preserving both NumPy `2.2.5` and `2.2.6` distribution metadata. It records metadata defaults separately from imported module versions. Torch's imported version is exactly `2.9.1+cu128`, distinct from the distribution metadata value `2.9.1`. No build suffix is stripped, duplicate metadata hidden, cache deleted, old service gate changed or runtime adopted. The intended Python, SciPy and Transformers versions are `3.10.19`, `1.15.3` and `4.57.3`.

The packet, sources and archive/manifest relationship are authenticated before helper execution or numerical imports. Helper and probe execution uses the captured authenticated source bytes. The script also authenticates the captured diagnostic bytes before executing those same bytes. Its supervisor bootstrap reads the trusted pinned 31,095-byte file without the child's explicit 1-MiB limit. Within the declared trusted-owned-path scope, which excludes malicious same-UID/root mutation and makes no denial-of-service guarantee, this difference is not a required defect. The child uses the bounded literal bootstrap.

The existing archive is staged into a new owned node-local directory. The unchanged helper performs full archive and object authentication before the copied interpreter is invoked. The supervisor retains only the six relevant cache/source pins before the helper parses its manifest, avoiding two simultaneous parsed manifest trees. No repack or original environment mutation is introduced.

Cache reasoning uses actual staged source integer mtime and size, reduced to the low 32 bits, against the captured timestamp header. A stale timestamp cache is reported as expected to be rejected by CPython, without decoding it or claiming that import behavior was traced. An eligible cache must match compilation of its captured source across recursive code fields. Neither outcome explains the historical Conda cache hash difference. The three known captured headers are timestamp mode. `-B` prevents new bytecode writes, not reads. Existing runtime startup hooks and caches remain trusted executable input.

The worker records metadata before Torch, retains the original Torch/attention/Transformers/Qwen import ordering, checks loaded origins and actual versions, and runs only the fixed small NumPy/SciPy arithmetic controls. Qwen classes are imported, not instantiated. The failure path attempts to retain module origins and process maps. The source correctly distinguishes absence of explicit CUDA calls from uninstrumented dependency capability checks. Observed module/native path containment is not a complete dependency proof or a sandbox.

The allocation guard checks the actual job identity, account, partition, QoS, one node/task/CPU, 3-GiB memory, ten-minute limit, no restart/requeue and no GPU TRES. The 580-second supervisor deadline and 185-second outer child deadline are distinct. Child startup and receipt validation are included in the latter, and success requires completion within 180 seconds. At least 200 seconds must remain before spawning. The owned process group is cleaned even after the leader exits, with the five-second window including cleanup receipt I/O. Missing, failed or interrupted terminal evidence cannot qualify the diagnostic.

The 512-MiB staging and 1.5-GiB worker thresholds are observed process-RSS limits, not cgroup guarantees. Prior batch RSS was near the allocation cap, and the file-cache contribution remains unknown. Scheduler termination and uninterruptible work can prevent complete terminal evidence. The guide states these limits rather than treating them as successful resource qualification.

Saved `stderr.log` reports eight new passing methods in one invocation. The test source and retained artifacts cover exact bindings and scheduler rejection, a tiny archive, stale and eligible cache cases, duplicate metadata versus actual module values, fabricated worker success/import failure, child receipt/deadline accounting, two tiny bootstrap children, and terminal publication interruptions. They use local Python `3.9.6`, synthetic modules, mocked signals/processes where indicated and tiny files. They do not qualify actual Python `3.10.19`, package imports, real process-group cleanup, the bulk stage or numerical libraries. None was rerun during this review.

## Custody, generated packet and resource accounting

Independently checked all preparation member hashes and exact membership in both copies. Each contains **145 files / 3,603,226 bytes**, including its manifest. The manifest enumerates the other 144 members. All five current source/test/guide/helper/probe bindings agree with `closeout.json`. Both launch01 copies contain the same ten reviewed files, and every generated `packet.json` input binding matches its file. No submission journal is present in either reviewed launch copy.

Preparation roots:

- Repository: `/Users/pat/code/ACE/results/ace_llm_intervention_runtime_metadata_preparation_20261107`
- External: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-preparation-20261107-01`

Launch roots:

- Repository: `/Users/pat/code/ACE/results/ace_llm_intervention_runtime_metadata_20261107/launch01`
- External: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-20261107-01`

The launcher requires committed input/launcher/review bytes, both local controller locks, no related active job, reconciliation of 16 prior terminal jobs and 34 journal paths, the recorded prior cost totals, the renewed account association, scratch write/read/cleanup and transferred-input readback. It records exclusive local/remote intent before one `sbatch` and retains ambiguous outcomes without retry. These are reviewed prospective checks, not newly observed remote facts. The readiness stage hashes the approximately 13-MB manifest and stats the archive size. It deliberately leaves the full 8,438,650,880-byte archive hash to the allocated stage. This review did not reread that archive or repeat earlier custody verification.

The generated request is **1 CPU, 3 GiB, 600 seconds, zero GPUs**, account `ucb736_asc2`, partition `acpu`, QoS `cpu-normal`, no requeue. The proposed admission arithmetic is **6,215 closed allocated CPU seconds + 3,600 preparation allowance + 600 new reservation = 10,415 / 57,600 CPU seconds**. Prior GPU admission stays **836 / 7,200 seconds**, with zero added here. No new allocation is evidenced by preparation.

The saved artificial invocation records **0.184439 child CPU seconds**, **0.210716417 elapsed seconds**, **25,509,888 peak child RSS bytes** and approximately **0.001379 supervisor CPU seconds**. These figures exclude source preparation and review and do not measure the whole cycle. They must not be added to overlapping admission allowances as though those were actual execution costs. The previous 35.303541-second staging interval supports attempting this bounded diagnostic but does not establish that the new run will finish or fit memory.

## Exact reviewed pins

All values below are SHA-256. They identify this review's bytes, not later corrections or an executed job.

| Object | SHA-256 |
| --- | --- |
| Diagnostic source | `9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49` |
| New test source | `05b22530a8a2c28e08fc9a81df07fcb5b53eb4f95e54a1ea10f183ca85f49b5f` |
| Diagnostic guide | `e7c955962c6b8aa0f1e8bf9458fe8304e603693182bfbee407aced0447ddc0ff` |
| Unchanged archive helper | `7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e` |
| Unchanged probe | `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951` |
| Preparation closeout | `12bc723d6f2d317e5da74aad5a2995207794da790e48d7f927b58b24447f927a` |
| Preparation manifest | `8fc247eaf43efff640e7dce8187da11fcb182c94399e5d80a6b86cc8e1d23761` |
| Saved test receipt | `65d5922d72e2d12266ee771d3fa6d09b860f2fb7b8e0ac1228be408675f9163e` |
| Generated packet | `c32df16f0f0960f25d8871b40d2ecd970ef0d3b93ffd352dcb4075f3cb5f7c63` |
| Generated diagnostic JSON | `a6f1fda50fc238b8c4b19e2113fc5bbc6fafdcf315cf21cf45677a3a41b3e555` |
| Generated job script | `69d64504ca8fcf678a27684567f82ec77f777e06365a39d7f5375979973b2c6c` |
| Packet preparer | `5d77f420307db017f720217032cba28324252da91082bbb696f7040905f3c0e1` |
| Launcher with R1 outstanding | `7d537f86ad3ee5c6400d51dc24a87f6f4dc2f342c1384d838535d160f7e5ef5b` |
| Monitor | `b0b3854768a9ba849521b66ef0233a7c63e9947969fc7bc43201fc454e85e3c1` |
| Collector with R2 outstanding | `9be377007ce92a5eb51d8d96b84cdbc0ea635f4b65e825c4ec2be181f376148c` |
| Prior archive, inherited pin only | `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98` |
| Prior manifest, inherited pin only | `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a` |

## Optional clarity and remaining boundary

The guide's line 74, “Original/staged objects and failed attempts remain in place for custody,” can imply that the node-local runtime survives scheduler cleanup. Clarify that original source/archive and retained job evidence remain in custody, while the node-local copy is ephemeral and represented by its stage pins. This is optional text clarity, not an additional code or execution gate.

After R1 and R2, a delta review can close these parent-tool findings without repeating the eight methods or old source reviews. Actual execution still requires the launcher's fresh reconciliation and committed bindings. Only later scheduler, stage, import, arithmetic, cleanup and complete terminal custody evidence can establish successful completion of this diagnostic. Even success would not qualify a model service, certify a historical runtime, or authorize a scientific or GPU retry.
