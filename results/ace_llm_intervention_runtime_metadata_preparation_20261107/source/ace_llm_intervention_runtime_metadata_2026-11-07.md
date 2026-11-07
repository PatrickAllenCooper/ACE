# Captured-runtime metadata diagnostic

This new CPU-only diagnostic observes the retained archive without repairing,
repacking, installing, deleting, renaming or quarantining any runtime object.
The previous service freeze and its NumPy 2.2.6 gate remain unchanged. This worker
never calls that service's `authenticate`, original probe `worker`, or project
closure imports. Successful diagnostic execution does **not** adopt a runtime,
qualify the model service, or certify any historical run.

The retained staged-runtime job failed its metadata gate before Torch, after a
35.303541-second staging interval. The captured prefix contains both NumPy 2.2.5
and 2.2.6 distribution metadata. Source/Conda evidence points to 2.2.5-associated
code, but no imported NumPy value was observed in that failed worker. The new
worker records metadata selection separately from actual imported module values.
Its exact actual-version contract is Python `3.10.19`, NumPy `2.2.5`, SciPy
`1.15.3`, Torch `2.9.1+cu128`, and Transformers `4.57.3`. Torch distribution metadata
`2.9.1` is a distinct observation; the build suffix is never stripped.

## Frozen diagnostic packet and CLI

The parent supplies `diagnostic.json` and its independently retained byte pin.
A separate launcher packet may carry scheduler admission and lineage; those fields
are not accepted as additions to the exact diagnostic schema. Paths below are
absolute. This module does not submit jobs or generate either launch packet.

```python
packet = {
    "version": "ace-runtime-metadata-diagnostic-v1",
    "sources": {
        "self": {"path": caller_path, "sha256": independently_frozen_self_sha},
        "helper": {"path": helper_path, "sha256": "7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e"},
        "probe": {"path": probe_path, "sha256": "8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951"},
    },
    "archive": {"path": existing_archive, "sha256": "6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98"},
    "manifest": {"path": existing_manifest, "sha256": "b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a"},
    "runtime_source": "/projects/paco0228/miniconda3/envs/ace",
    "output": new_exclusive_output,
    "resources": {"account": "ucb736_asc2", "partition": "acpu", "qos": "cpu-normal",
                  "cpus": 1, "memory_gib": 3, "wall_seconds": 600, "gpus": 0, "requeue": False},
    "limits": {"supervisor_seconds": 580, "child_success_seconds": 180,
               "child_outer_seconds": 185, "cleanup_seconds": 5,
               "stage_peak_rss_bytes": 536870912, "worker_peak_rss_bytes": 1610612736},
    "actual_versions": {"python": "3.10.19", "numpy": "2.2.5", "scipy": "1.15.3",
                        "torch": "2.9.1+cu128", "transformers": "4.57.3"},
    "numpy_metadata_versions": ["2.2.5", "2.2.6"],
}
```

The supervisor accepts these arguments:

```text
--packet /absolute/diagnostic.json --packet-sha256 PACKET_SHA
--self-sha256 SELF_SHA --helper-sha256 HELPER_SHA --probe-sha256 PROBE_SHA
```

The job script must authenticate the source **before execution**, then compile
and execute those same captured bytes with `__name__="__main__"`, `__file__` set
to the pinned path, and `sys.argv` containing that path and the above arguments.
Use `python3 -I -B -c BOOTSTRAP SELF_PATH SELF_SHA ...`; the module's literal
`BOOTSTRAP` implements bounded read → independent SHA-256 check → compile/exec.
The child uses exactly that bootstrap with the staged interpreter, `-I -B -X
importtime`, and additional internal `--worker`, `--staged-prefix`, and
`--stage-sha256` arguments. No self-hash is embedded in the source, so there is
no source-hash cycle. Both roles recheck the packet and source pins before
executing helper/probe bytes or importing numerical packages.

The supervisor independently checks the intended one-node, one-task, one-CPU,
3-GiB, ten-minute `ucb736_asc2/acpu/cpu-normal` allocation, running state, no GPU,
no restart and no requeue. It creates an exclusive
`$SLURM_SCRATCH/ace-runtime-metadata-JOB/runtime`. The unchanged archive helper
checks the manifest pin, archive digest and each extracted object; it never
executes the staged runtime. The independently supplied archive pin must also
match the manifest. The archive is reused, not copied into Git or repacked.
Original/staged objects and failed attempts remain in place for custody.

## Observations and cache reconciliation

Before numerical imports, the worker saves interpreter path/prefix/search-path
and loaded-origin/maps evidence; the three cache/source observations; and all
discovered distribution names, versions, metadata paths and distribution roots.
It also saves each name's default metadata resolution. Both NumPy metadata
versions must remain visible. A selected default is not an imported version.

For the known NumPy/SciPy `__config__` and stdlib `_sysconfigdata` cache/source
pairs, it records the full file SHA-256, 16-byte header, actual copied source
mtime/mtime-ns/size and corresponding source hash. Pins must agree with the
staged manifest. Reads are bounded to 1 MiB per file; caches are never modified.
For flags-zero timestamp caches it applies the copied source's **actual** mtime
and size, using the low 32 bits as CPython does. A timestamp or size mismatch
predicts rejection of that cache and source compilation. This is a three-file
prediction, not a traced import decision or a general cache-safety claim.
CPython's validation and `SourceLoader.get_code` fallback are in the
[Python 3.10.19 import implementation](https://raw.githubusercontent.com/python/cpython/v3.10.19/Lib/importlib/_bootstrap_external.py).

If a header remains eligible, the diagnostic decodes the bounded code object
without executing it and compares recursive code fields, constants, nested code,
line and exception tables with `compile(source, captured_co_filename, "exec",
dont_inherit=True, optimize=0)`. No field is silently discarded. A mismatch or
invalid eligible cache stops before numerical imports; no deletion or alternate
run follows. Hash-cache validation is handled explicitly if encountered, but
the three captured headers are timestamp mode. Local equivalence or expected
staleness does not establish why their historical Conda-record hashes differ.
Those three byte differences remain unexplained. `-B` disables new `.pyc` writes,
**not reads**; this is not source-only execution. The pinned runtime remains
trusted executable input, including its existing caches and startup hooks.

The package sequence is the original Torch → attention imports → Transformers
Auto classes → Qwen3 class order. It then records exact NumPy/SciPy/Torch/
Transformers `__version__` and origins. No class is instantiated. After the import
sequence, fixed two-element dot-product and 2×2 NumPy/SciPy solve controls may run.
There are no model/weight loads, explicit CUDA API calls, tensor training,
scientific inputs or environment responses. Dependency capability checks during
imports are not instrumented and are not represented as absent.

Loaded module paths and `/proc/self/maps` are captured on success **and failure**.
The worker rejects the old prefix, mixed interpreter search/module paths and
native paths outside the staged prefix or explicit system library roots.
The pinned caller itself may live outside the runtime. This is observed path
containment, not universal dependency closure or an OS sandbox.

## Bounds, terminal authority and limits

The supervisor timer is 580 seconds. A child starts only with at least 200 seconds
remaining. Its absolute 185-second outer timer starts before process creation
and startup-receipt I/O; successful return, worker-receipt authentication and
return-receipt I/O must fit within 180 seconds. The unchanged pinned
`probe.kill_owned` always runs, including when the leader has returned. Its
five-second window includes cleanup-receipt I/O, with no disk write before the
owned group is terminated. Additional INT/TERM signals are ignored only during
that cleanup window. The original supervisor deadline is then restored.
Native uninterruptible work and hard scheduler termination can still prevent
final evidence; absence is unresolved, never permission to retry.

Every phase records UTC, elapsed time, process CPU and cumulative process peak
RSS. The 512-MiB staging and 1.5-GiB worker observation thresholds are conservative
**design limits**, not measured demand or guarantees of cgroup headroom. The
previous batch MaxRSS of 3,143,400 KiB was near the 3-GiB allocation; its process
and file-cache contributions were not separately measured. File cache is a
possible contribution, not an established cause. The observed job failure was
not an OOM finding. A later actual diagnostic must reconcile scheduler memory
and phase evidence before accepting resource sufficiency. No allocation occurs
as part of this implementation's fabricated tests.

Each worker/supervisor terminal has an fsynced candidate and exclusive link.
Workflow exceptions publish a failed status through that same single terminal
path. Publication exceptions preserve a separate diagnostic and reconcile exact
published bytes; an already published terminal remains authoritative. There is
no success-plus-failure terminal pair, retry, resume or fallback. Storage faults
may leave only intent/partial artifacts and require outer reconciliation.

Only the new stdlib/fabricated tests execute during preparation: source/packet
and allocation rejection, tiny archive integration, cache/header comparisons,
duplicate metadata and origin checks, fabricated import success/failure, deadline
I/O faults, exact-byte bootstrap, and terminal publication. Real runtime packages
and the retained bulk archive are not executed or replayed. The exclusive evidence
capsule binds tested source bytes, command, stdout/stderr, retained failures,
resource measurements and matching external copies. Parent review and a frozen
launcher are still required before any actual diagnostic launch.
