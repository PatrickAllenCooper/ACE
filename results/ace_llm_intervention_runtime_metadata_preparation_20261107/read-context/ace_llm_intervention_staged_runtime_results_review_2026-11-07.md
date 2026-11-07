**Independent saved-evidence staged-runtime results review, November 7, 2026.**

**Required corrections to retained custody and accounting: 0. Qualification
remains failed.** Job `35343957` completed archive capture and node-local staging,
then its fresh worker failed the Python/package metadata-version gate before
Torch. The saved evidence does not qualify the copied runtime or model service.
The NumPy and Conda findings below require an explicit successor runtime contract
before a later qualification can succeed. They do not justify disguising the
captured package as NumPy 2.2.6.

This new review examines terminal results, receipt links, accounting and the two
saved forensic collections. Earlier helper, caller and packet source reviews
were not repeated. Read only the relevant failure-boundary source to interpret
the traceback and collector to establish the custody hash scope. Used stdlib
reading, hashing and comparison of saved files. No copied package, model, test,
policy or diagnostic was executed or imported. No remote work, environment
change, ledger edit or commit occurred. Only this review document was authored.

**Custody and execution bindings.** The exclusive external `remote/` tree at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-staged-runtime-terminal-20261107-01`
has exactly the recorded **314 files / 8,451,802,779 bytes**. The repository's
terminal `remote/` tree has exactly the same membership except for
`qualification/packed/runtime.tar`. Independently hashed all **313 nonbulk files /
13,151,899 bytes**, matched them to the captured remote inventory and disposition,
and compared their repository and external copies. No discrepancies were found.
The archive's local size is **8,438,650,880 bytes**.

The bulk tar was not redundantly read or rehashed in this review. Its accepted
custody evidence is the completed pack manifest's SHA-256, the collector's
recorded full local streaming hash after transfer and exact size agreement.
The collector source computes that hash and checks it against the supervised
pack pin before writing the complete-custody disposition. The captured remote
inventory explicitly identifies its tar pin as coming from the pack manifest,
not a fresh remote rehash. This review verified that chain and the size, not an
additional independent full-tar digest.

The manifest itself was independently hashed. It records **49,966 objects**:
44,320 regular files, 4,256 directories and 1,390 symlinks, with
8,397,575,073 regular-file bytes. Counts, unique path membership and byte totals
agree. `stage.json` binds the same manifest and archive hashes, object count and
file-byte total, and names
`/scratch/local/35343957/ace-runtime-35343957/runtime`. Its runtime qualification
and model-loaded flags are explicitly false.

The submission result binds the exact retained intent. The intent binds the
packet, four input files, source commit
`5967034358672db1eac9932352728e69a87ea045` and packet review. These input bytes,
the packet and submission result match the retained launch copies. This checks
execution lineage without reopening those source reviews. The job was submitted
once in the retained chain at `2026-11-07T18:36:17.244109Z`. Captured terminal
Slurm state shows `Requeue=0`, `Restarts=0` and `Reason=NonZeroExitCode`.

**Scheduler, phases and cleanup.** The saved `sacct` parent row is `FAILED`, exit
`1:0`, one allocated CPU for **1,452 seconds**, account `ucb736_asc2`, with
`cpu=1,mem=3G,node=1` and no GPU. Captured `scontrol` agrees on one node/task/CPU,
`acpu/cpu-normal`, a 30-minute limit and node `c3cpu-e2-u3`. Its local start/end
times are 11:36:37 and 12:00:49, consistent with 18:36:37 and 19:00:49 UTC and
the measured 24:12 allocation. This is a nonzero worker failure, not a scheduler
time limit or recorded out-of-memory termination. The captured terminal queue
contains no active ACE job. This review makes no claim about the current queue.

All 292 parent phase records are present in sequence. UTC, elapsed and process
CPU values are monotone. Key milestones are:

| Event | UTC | Cumulative parent elapsed seconds |
| --- | --- | ---: |
| Pack start | 18:36:42.265232 | 0.118209 |
| Pack complete | 19:00:06.061444 | 1403.914427 |
| Stage complete | 19:00:41.364987 | 1439.217969 |
| Original sequence start | 19:00:41.368188 | 1439.221159 |
| Original sequence returned, exit 1 | 19:00:44.949822 | 1442.802799 |

The pack-start to pack-complete interval is **1403.796219 seconds**, with
32.259478 parent process CPU seconds. Pack-complete to stage-complete is
**35.303541 seconds**, with 28.448057 parent process CPU seconds. The latter
includes the intervening checks and stage work. The values 1403.914 and 1439.218
are cumulative timestamps, not two durations to add.

The supervisor measures the worker at **3.574551 seconds**. Its internal failure
receipt measures **3.320802 elapsed / 1.014446 process CPU seconds**, with peak
RSS 183,760 KiB. Only `authenticate_start` is present in its phase records.
The traceback ends in `service.authenticate` at `actual Python/package versions
differ`. There is no `authenticate_complete`, `torch_start`, success terminal or
runtime-origin report. The preserved outer failure message mentioning the
180-second qualification bound is a generic failed-condition message. It does
not mean this worker ran for 180 seconds or timed out.

Cleanup names the same PID and process group as `worker.started.json` and records
return code one, `already_gone=true`, no kill sent, no remaining group and no
errors. This establishes the owned worker's observed termination. It does not
prove later site deletion of the node-local directory. Model-file bytes can be
read for authentication before the version gate, but no Torch model loading,
training or environment responses occurred on this execution path.

Slurm reports **62.087 process CPU seconds** for the job and **3,143,400 KiB**
batch MaxRSS, close to the 3-GiB request. This memory observation warrants
headroom attention for a future diagnostic, but is not evidence of an OOM here.
The parent failure receipt's 60.716850 process CPU seconds, worker measurements,
Slurm TotalCPU, elapsed times and allocation must not be summed as independent
costs or interchangeable peaks.

The retained fifteen-job predecessor accounting sums to 4,763 allocated CPU
seconds and 836 GPU-slice-seconds. This job gives **6,215 allocated CPU seconds
across sixteen closed jobs** and leaves GPU allocation at **836**. Releasing its
1,800-second reservation and retaining the existing 3,600-second cumulative
preparation allowance gives **9,815 / 57,600 CPU admission seconds**, with
**836 / 7,200 GPU-slice-seconds**. The collector's 280.707309 elapsed and 2.987869
self CPU seconds are separate local collection measurements. They are not a
complete preparation or research-cycle cost.

**NumPy wheel records and the later Conda evidence.** Both NumPy 2.2.5 and
2.2.6 `dist-info` directories occur in the captured manifest. Independently
verified the five selected NumPy files against their archive-member pins and
their external copies. Recomputed the saved RECORD comparisons from those
bytes and the authenticated inventory:

- NumPy 2.2.5: 981 matching hashed rows, 24 nonmatching hashed rows and 492
  unhashed rows.
- NumPy 2.2.6: 973 matching hashed rows, 34 nonmatching hashed rows and 492
  unhashed rows.

These exact rows agree with the saved forensic report. Neither wheel RECORD
fully describes the captured installed prefix. Unhashed rows are not successful
hash comparisons. The archived `numpy/version.py` explicitly sets
`version = "2.2.5"`, with `__version__` and `full_version` assigned from it.
This is observed source content, not an observed imported NumPy version.

The new four Conda metadata files also match their archive-member pins and
external copies. Recomputed their comparisons using `sha256_in_prefix` where
present, otherwise their recorded `sha256`:

- `numpy-base-2.2.5-py310he1678cf_2`: **1495/1496** match. The sole regular-file
  mismatch is `numpy/__pycache__/__config__.cpython-310.pyc`.
- `scipy-1.15.3-py310h0a862cd_1`: **2333/2334** match. The sole mismatch is
  `scipy/__pycache__/__config__.cpython-310.pyc`.
- `python-3.10.19-h6fa692b_0`: **2048/2063** match in the parent's hash comparison.
  Fourteen of the fifteen nonmatches are recorded `softlink` objects that the
  archive represents with a link target and no regular-file digest. The remaining
  one is `_sysconfigdata__linux_x86_64-linux-gnu.cpython-310.pyc`.
- The separate `numpy-2.2.5` Conda record owns zero paths. It is not another
  independently checked NumPy implementation.

All 24 wheel-2.2.5 and all 34 wheel-2.2.6 nonmatching paths match the saved
NumPy-base Conda installed-prefix expectations. This is materially stronger
evidence of consistency with that 2.2.5 Conda build than the wheel counts alone.
It does not identify the precise historical installation/relocation operation.
The three `.pyc` differences remain unexplained byte differences. The fourteen
Python symlink cases are comparison-type limitations, not fourteen demonstrated
corrupt regular files. No imported package version, publisher authenticity,
binary compatibility or runtime behavior follows from these captured local
metadata comparisons.

Both forensic trees have exact repository/external membership, seven files for
`numpy-forensics01` and five for `conda-metadata01`. They are separately derived
evidence, not additions to the original 314-file remote inventory. NumPy
forensics records 0.763075 elapsed and 0.756936 self CPU seconds for its local
comparison. The Conda comparison has no equivalent execution-cost receipt, so
none is inferred.

**Required interpretation and narrow next diagnosis.** The unchanged service
and freeze declare NumPy **2.2.6**. The failed gate compares distribution metadata
and Python version. Its failure does not retain the complete observed version
mapping, so this review does not claim which comparison field first differed or
which duplicate metadata entry was selected. Independently, the archived source
and installed-prefix records establish that simply hiding the **2.2.5** metadata
would force a misleading **2.2.6** label onto captured 2.2.5-associated code.
That is not an acceptable correction.

The narrow next step is a separately recorded **CPU-only archive-reuse
diagnostic**, using the existing archive and pins, with no modification of the
source environment, installation or GPU request. First record all discovered
distribution metadata and its paths, the interpreter identity and actual
package-code origins. Reconcile the three cache-file differences without
assuming they are harmless or importing them to obtain a desired answer.
Then establish a consistent prospective numerical runtime contract with explicit
parent/derived lineage before claiming qualification. Any changed version
contract and copied layout must be deliberate and must leave the original
freeze, archive, failed attempt and accepted studies unchanged. Actual imported
versions and the original import sequence need new observed evidence under that
contract. A metadata-only pass will not qualify the service.

This review does not prescribe deleting or moving either metadata directory,
waiving the existing gate, enlarging GPU resources or repeating the full pack.
Archive reuse must still account for its own verification/staging, resource
headroom, child deadline and terminal custody. No runtime or scientific success
should be inferred from the fast pre-import rejection.

**Principal SHA-256 bindings.**

- Terminal disposition:
  `5e08ae3c74dde362517403a427c2649adabdc0d775cac9cf70bac6b9fc2a3ed3`.
- Captured inventory and scheduler transport:
  `c501811b2423b49d25931beb079b08eff44de3545ec9b32794c8211f0e1e6226`.
- Runtime manifest:
  `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a`.
- Bulk archive, collector-bound scope described above:
  `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98`.
- Stage receipt:
  `f1a1450e1ae6f8b3c89c8876919fbc2c4c1b96422b050fcf44ccd0abc6181b07`.
- Worker failure:
  `84422cba5564906773ca024d6c3d081987a597e0b14041f16e4172eaf9b4548a`.
- Worker cleanup:
  `8af7d4e5fb48678153ee07150735f6aad597b227aeed76981fc9a0c551482642`.
- NumPy forensic report:
  `a67c1fc0d0693251d4a5435e84d6a4ab4e1bcd8c3ef9172366945a971c2c5626`.
- Archived `numpy/version.py`:
  `9c42885787550a2c0e218216099f44f8701ba537a9bb2e754a9991165c7be75a`.
- Conda comparison:
  `8f2a41a84323f8679737f1e83efb0ad855aefbc0caf9e0ea552389aad208b5b1`.
- Archived NumPy-base Conda record:
  `c3b361e91a88d3a990055fc4010747849113d77902c124b1e429de8ce1895651`.
- Unchanged prior service and freeze:
  `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1`
  and `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37`.

The original pending-review disposition and all raw failures remain unchanged.
This separate review closes saved-evidence inspection, not the failed runtime
qualification or the broader research objective.
