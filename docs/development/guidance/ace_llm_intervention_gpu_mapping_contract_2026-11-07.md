# Distinguishing GPU device mappings from native libraries

GPU12 reached model loading, then stopped before its handshake. The generic CPU runtime check treated every absolute process mapping as a library. Its rejected set contains only `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` and deleted `/dev/zero` mappings. All37 associated rows are shared and non-executable. Complete failure custody is accepted. The service itself is not qualified.

Linux documents `/proc/PID/maps` as a list of mapped regions and their access permissions, including shared and executable flags. A mapped pathname alone does not establish that the object is an imported library. See the [kernel proc documentation](https://www.kernel.org/doc/html/v6.12/filesystems/proc.html). NVIDIA's own [device definitions](https://github.com/NVIDIA/nvidia-modprobe/blob/main/modprobe-utils/nvidia-modprobe-utils.h) and [device creation implementation](https://github.com/NVIDIA/nvidia-modprobe/blob/main/modprobe-utils/nvidia-modprobe-utils.c) describe GPU device nodes, control minor255, major195 and the dynamically registered UVM device. These sources explain the classification. They do not authenticate CURC's installed kernel or driver.

## New isolated helper

`llm_intervention_gpu_mappings.py` is now integrated in service v3 and runtime contract v2. Actual runtime qualification of this successor remains pending. It has no numerical imports, device opens, model calls or allocations. Its capture function reads metadata for four exact paths and the character-device registry. Its validator requires root-owned character devices with the expected identities. NVIDIA mapping rows must match the current device's filesystem device and inode, remain shared and non-executable, and not be deleted. It accepts only the observed GPU0 device. A different GPU device requires an explicit reviewed contract change.

Deleted `/dev/zero` rows are handled separately. The accepted shape is the observed shared read/write, non-executable mapping with zero offset, positive backing inode and filesystem device00:01. The current `/dev/zero` must be the root-owned1:5 character device. Its inode is not equated with the deleted backing inode. This classifies a kernel-reported anonymous shared-memory shape. It does not authenticate that backing object through its current pathname.

The validator trusts the host kernel, procfs and root-controlled device namespace. It is not a hostile-process sandbox, GPU-assignment proof or statement of universal driver compatibility. Scheduler and owned-process telemetry remain separate requirements. No prefix-wide `/dev` exception is introduced. Arbitrary non-executable files remain rejected.

Every absolute raw mapping is reconsidered under the original runtime policy using independently bound copied and original runtime roots. The recomputed rejected set must exactly match the saved rejection list before device rows can be classified. This closes the initial review finding about an omitted unknown mapping. The helper returns a separate check view. The original snapshot, all raw map rows and device inventory must remain saved without alteration.

Only the post-model-load and post-request phases can use this classification. CPU preflight and pre-import checks keep the original behavior. The helper does not weaken Python-module, imported-version, generated-module, source, model or archive checks.

## Preparation evidence and limits

Six initial artificial methods passed. They cover exact device types, raw-record immutability, executable and unknown mapping rejection, malformed rows, inode/device disagreements, phase limits and deleted-zero shape. Initial preparation cost was0.032201childCPU seconds,0.035084291elapsed seconds and14991360peakRSS bytes.

A distinct reviewer required complete reconciliation of raw map paths. That correction adds independently bound roots and exact recomputation. One new focused method passed, covering three omitted unknown paths, four invalid root bindings and two allowed library paths. Its cost was0.032003childCPU seconds,0.035351542elapsed seconds and15646720peakRSS bytes. The earlier six were not rerun. Both preparations are preserved in exclusive custody. These are fabricated records, not observed device inventory or GPU runtime qualification.

## Integration contract and remaining qualification

1. Keep the accepted generic CPU diagnostic unchanged. Add the new helper to the explicitly captured project closure and new service manifests. Existing source snapshots and manifests continue to bind their original bytes.
2. Save the unmodified post-load/post-request snapshot and freshly captured device inventory before checking either. Run the original prefix guard on the separately validated check view. The saved-evidence validator must independently reconstruct the same view from raw observations and independently bound roots.
3. Preserve the pre-import and CPU-only behavior. A changed service closure requires new source lineage and qualification. Do not claim that the old25-input CPU acceptance authenticates future bytes.
4. Review and test the integration using fabricated records, then qualify the revised CPU import closure with explicit resources. No model training, simulator responses or scientific-world generation is involved.
5. Only then prepare a new bounded GPU packet. Retain the single measured20GBslice, existing host-memory and time limits, first-work deadline, independent GPU signals, complete request checks and process cleanup. The next scientific gate remains full service qualification, not a model-loaded receipt.

Current closed allocation is7072CPU seconds and941GPU-slice seconds. Adding the cumulative3600CPU preparation allowance gives10672/57600CPU seconds for admission. GPU admission is941/7200seconds. No job or reservation is active in the last terminal snapshot. Preparation measurements are not whole-cycle CPU accounting.


The four-file integration has passed independent static review with zero required fixes. Nine targeted artificial methods passed, including raw-before-capture order, failed-observation preservation, CPU exclusion and independent reconstruction at both GPU phases. CPU03 is prepared with 26 inputs, 16 service sources and a two-CPU, six-GiB, ten-minute limit. Its new manifests retain the original runtime archive, model bytes, artificial fixtures and adapter. The historical CPU receipt is predecessor evidence only. The GPU caller will be rebound only after the new CPU result is independently accepted.


## Physical device indices after GPU13

GPU13 reached the same model-ready phase but mapped `/dev/nvidia1`. The v1 classifier had incorrectly fixed the physical path to `/dev/nvidia0`. It rejected the observation before handshake, so no inference request completed. The raw snapshot and independently captured v1 inventory remain failure evidence. That inventory describes node0, not node1. It cannot retroactively authenticate the mapped node1.

The prospective v2 classifier accepts one canonical indexed NVIDIA character node, observed in the rejected map roster, alongside the three fixed special paths. It does not equate physical device indices with a process-local CUDA index. NVIDIA's [device naming definitions](https://github.com/NVIDIA/nvidia-modprobe/blob/main/modprobe-utils/nvidia-modprobe-utils.h) distinguish indexed device paths from the control and modeset minors. Its [implementation](https://github.com/NVIDIA/nvidia-modprobe/blob/main/modprobe-utils/nvidia-modprobe-utils.c) defines major195. The contract conservatively excludes minors254/255, leading-zero aliases, multiple indexed nodes and all other path forms.

Capture reads metadata only for the observed indexed node and the three special nodes. Root ownership, character type, registered NVIDIA major195, basename/minor agreement and raw-map inode/device equality remain mandatory. Executable, private or deleted indexed mappings still reject. UVM registry and deleted-zero checks are unchanged. Full raw-map reconciliation and all other runtime guards remain. Scheduler allocation, GPU visibility and independent owned-process telemetry still establish separate facts. This classifier is not GPU-assignment proof.

Five new physical-index fixture methods plus two affected observer/receipt integration methods were attempted. Six passed initially. The capture mock failed because Darwin device-number encoding overflowed for an artificial Linux UVM major. The corrected mock explicitly supplies fake encoding and major/minor functions. That one method then passed. Production code did not change for the test correction. These checks use no real device metadata or models. Both attempts remain preserved. Independent correction review, new source bindings and actual qualification are still required.


The initial index correction review identified an unsupported registry name in its fabricated fixture. Actual GPU13 metadata records `195 nvidia`, not `nvidia-frontend`. Reading the complete saved registry also reveals repeated unrelated InfiniBand names under different majors. The corrected parser preserves the full raw registry while requiring unique `nvidia` and `nvidia-uvm` identities. It validates other entries without requiring their names to be globally unique. A focused fixture uses the independently pinned complete GPU13 registry with synthetic device metadata. It does not authenticate the missing node1 inventory from GPU13.
