# Staged-runtime CPU launch packet review

November 7, 2026. **Zero required packet corrections. Conditional acceptance for the specified CPU qualification only.** Both component delta reviews are now closed at the exact source hashes below. The complete source and packet commit, fresh remote reconciliation and successful remote verification remain prelaunch conditions. This is not evidence of submission or runtime qualification.

Inspected the seven launch01 files, current production source copies, predecessor diagnostic acceptance and retained accounting, original service/freeze, and the two component rechecks. Used only local reads, byte hashing and Git inspection. No tests, reviewed-source imports, models, remote commands, submissions or commits were performed. Only this review file was authored. Component internals and completed tests were not reviewed again.

## Bindings and caller agreement

All four `packet.json` input pins match current bytes. The batch bootstrap authenticates and executes the captured caller bytes. Its helper, probe, service and freeze arguments use the corresponding full hashes below. Paths consistently name staged-runtime launch01 and the original GPU10 input directory. The output is the new exclusive `qualification` directory beneath the declared remote root.

The helper and caller copies equal their current production files. The probe is byte-identical to corrected diagnostic launch02. The service and freeze equal the preserved GPU10 input copies. The predecessor acceptance pin matches the completed diagnostic acceptance. The caller invokes the original probe's worker in `matched_import_order`, rather than its old eight-minute diagnostic supervisor. Consequently that old supervisor's resource gate does not conflict with the new thirty-minute caller gate. The original service still authenticates frozen artificial sources, model-file bytes and declared runtime. Hashing model files is not loading model weights.

| Launch01 file | SHA-256 |
| --- | --- |
| `job.sbatch` | `ac79adcfd9b7aca3599745691bca8fcf559a77afe232651869283ec7c2d999a5` |
| `packet.json` | `e985bb75930ec7ada68a6762a0509e86f32d8797f5cafa02a79eeb7f0ef75a7c` |
| `launch_once.py` | `3c81927e6098cae05494aad3a83512b00f93a13df1ec86ed20fc6faf87f3203d` |
| `monitor_once.py` | `1c4db4f521ee14c44099948249e2a4027becb779efe126b3cb0b95e5a72a482c` |
| `qualify_llm_intervention_staged_import.py` | `9e63e644aba5c65a5c8b61d24018a8dad2282538b2834132b36da5d7516e72c7` |
| `llm_intervention_runtime_archive.py` | `7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e` |
| `probe.py` | `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951` |

Original service SHA is `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1`. Original freeze SHA is `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37`. Diagnostic acceptance SHA is `8cbc09522ef91548cc39efe27deace8ade78956d7e9f94b6796098c4b8bf2dc2`.

## Resources and prior-state assumptions

Batch directives, packet and caller agree on `ucb736_asc2`, `acpu`, `cpu-normal`, one CPU/task/node, 3 GiB, thirty minutes and no requeue. No GPU is requested. The caller rejects GPU AllocTRES before preparation. The batch disables CUDA visibility. The packet's 1,740-second supervisor, 180-second acceptance, 185-second outer-child and separate five-second cleanup bounds agree with the bound caller/probe. The component recheck explains their signal-delivery limits.

The retained diagnostic submission has fourteen closed jobs totaling 4,462 allocated CPU-seconds and thirty earlier journals. Adding diagnostic35343088's 301 allocated seconds and its two journals gives **fifteen closed jobs, 4,763 CPU-seconds and thirty-two journals**. The recorded GPU jobs each allocated one slice, totaling **836 GPU-slice-seconds**. Admission is exactly **4,763 + 3,600 + 1,800 = 10,163 / 57,600 CPU-seconds**, with **836 / 7,200 GPU-slice-seconds** unchanged. These are admission/accounting quantities, not whole-cycle process CPU.

The launcher checks the live queue, terminal job count, CPU/GPU totals, journal count and renewed association twice. It records current journal hashes and scheduler rows. Its code checks counts and totals rather than comparing every historical identity/hash to an embedded expected map. These assumptions match the inspected retained state. They are not a fresh live observation from this review. Any changed or unresolved history needs reconciliation before launch, not a count adjustment to force passage.

## Once-only submission and remaining conditions

The launcher acquires both shared controller locks nonblocking. It requires a new remote directory, performs scratch ownership/stat/df and dedicated write-read-cleanup checks, transfers five pinned inputs, verifies their remote bytes, then rechecks those bytes immediately before submission. Service/freeze authentication remains in the worker. Local and external records use exclusive writes. Remote intent precedes the single `sbatch` call. Timeouts and malformed or failed submission responses retain evidence and require reconciliation. No automatic retry, requeue or GPU follow-up is present. The monitor reads the saved job ID and writes exclusive local/external snapshots. It neither submits work nor declares scientific acceptance.

At inspection, HEAD was `9e49cda9a0d5be43c2ff762b1dbf9dd166dcf9c0` and launch01 was untracked. No launch01 submission journal existed locally. **Commit all seven reviewed files and this review before running the launcher.** The launcher enforces committed equality for its six execution/packet files and this review. `monitor_once.py` is not in that automatic equality gate, so its inclusion in the complete commit remains an explicit operator condition. Preserve both component rechecks in that committed provenance as well. Changed reviewed bytes require a scoped binding recheck.

Archive recheck SHA `4e63ee6be775d20fdf24c7b5729ed5441eedb5ed1de1714ed636e113dfd063a7` closes its two findings. Caller recheck SHA `392c5ff099892c57283984c81ad0a2cafc6a73939be44c06200452661582e22e` closes the deadline finding. Both files and their source identities were verified locally. There is no remaining component-review condition for these exact bytes.

Actual acceptance still requires scheduler resources and terminal accounting, complete retained outputs, the caller's successful terminal, original worker/origin evidence and clean owned-process termination. Missing, conflicting or interrupted evidence is not success. This packet tests CPU startup after measured staging. It does not qualify the persistent GPU service, identify a causal cache/import-order effect, or answer the scientific intervention-efficiency question.
