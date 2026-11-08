# Scientific GPU campaign coordinator — independent source review

**1 required P1 integration finding before concrete build.** Scope: Godel's new builder, once-only launcher and read-only monitor in `/tmp/ace-scientific-gpu-campaign-authoring-u9n4rwd1`. Reused closed scientific production/runtime/service checks; this review does not reopen them or qualify a future allocation, actual result or scientific utility.

Exact reviewed sources:

- `build_packet.py`: `e1ffecad2840a87706f4cc02715a615af1eb1d82c6ff468c28eefc0677e32b7a`.
- `launch_once.py`: `a9f7f203f9a47b54fe7cc0bfbc1ae9f7fadeabcae9801ad943feffbcae359856`.
- `monitor_once.py`: `220ba06cf4d3bb2ef40d09e7d41b59300a3ba93163d5ab73bdbf45f2adc24b32`.

## R1 — Token predecessor join requires a file its producer never captures

`build_packet.py:312–313` includes `inventory` in the token predecessor joins, producing a required `collection-inventory.json` entry. Line322 unconditionally indexes that entry in the original token launch packet's `new_input_files`. The reviewed token builder `/tmp/ace-scientific-tokenization-recovery-authoring-qa7ojemp/build_packet.py`, SHA256 `88320878518532419ad3450823de72ba67df9935764fd7fd5cf0c2ba9055bb34`, has no `collection-inventory` in its metadata interface and does not otherwise publish this filename. Its publication at lines522/536 derives the immutable roster from the actual metadata inputs. Thus otherwise valid future token evidence from that producer fails with KeyError before the new GPU packet can be built.

Fix the new consumer to match actual upstream custody: remove the nonexistent token inventory-member requirement and, if retaining a nested original-capture join for the separately supplied collection inventory, use the reviewed training packet's existing `collection-inventory.json` member. The training builder already accepts and publishes that metadata file. Keep the independently supplied inventory, original collection result/disposition/packet/baseline joins and whole-history verification. Do not alter the closed token packet or fabricate a new field. A single new focused producer/consumer roster regression is appropriate; existing test batches need not repeat.

## Other reviewed boundaries

The builder pins original freeze15602614… and sourcea9346022…, preserves the complete67-source map and exact CPUcampaign plus five GPU additions, and uses AST literals instead of project execution. It retains the unchanged strict CPU acceptance schema, original CPU packet, 24 descriptor pins, complete CPU output/campaign file maps and all five attempt pins before carrying opaque bytes. It binds prepared/plan/exposure/model/runtime/training acceptance and both SFT/final checkpoint/adapter roles without deserializing models or private bundles. Frozen/SFT/DPO contracts keep original CPU input endpoints and the frozen GPU_ROOT/gpu output; no rebind, new fit, selection or treatment alteration is introduced. The allocated frozen wrapper remains responsible for complete scientific payload admission and runtime evidence.

The failed41/84 baseline and original collection-only amendment remain separate from the scientific freeze. The new consumer requires independently supplied actual accepted42/86,43/88,44/90 and45/92 histories, exact prior rows/journals and actual resource charges; future outcomes/costs are not defaulted. Public existing collection02 disposition field names match the consumer's custody interface; this was an interface check only, not a repeat collection acceptance audit. R1 concerns the token packet's distinct captured-file interface.

Requested profile is exactly4CPU15360MiB/one20GB A100 MIG/2880seconds, ucb736_asc2/aa100/gpu-testing/no-requeue; work2820/final2860/service reserve600/cap1200. Budget closure includes single3600CPU preparation allowance,11520CPU/2880GPU reservation and remaining2400CPU finalizer cap. Both locks, complete fresh45-row/92-journal reconciliation twice, no active ACE job, unchanged namespaces, original CPU roster/pins, live account/partition/QoS/MIG checks, model hashes, runtime metadata, exclusive roots/intents, readback and one sbatch are retained. Timeout/ambiguity retains evidence and prohibits blind retry; no cancellation path exists. Actual allocated runtime/device/cleanup qualification remains required. Retained timing sensitivity exceeds the2880 cap, so source compatibility is not a completion/headroom claim.

The monitor's GPU controller/phase/service paths match the frozen producer interfaces read here. Public telemetry distinguishes owned process memory and positive-generation progress, leaves utilization percent unknown and never asserts acceptance. It does not read private episode/generation payloads. The CPU monitor's separately corrected bundle path and new pin do not change CPU builder/launcher/worker interfaces used here; this GPU authoring does not enforce the superseded CPU monitor pin as a constant.

## Evidence and limits

Read the retained static01 receipt, six metadata02 author checks and lineage-delta03 receipt without rerunning any. Their records report passes; they are static author checks, not end-to-end runtime evidence. The final lineage delta introduced the faulty inventory join; its retained check did not establish that the token producer publishes that member. Original authoring read-path errors and predecessor snapshots remain preserved.

Independent source, handoff, argument-map, predecessor, retained-check and relevant interface snapshots: `/tmp/ace-scientific-gpu-campaign-coordinator-review-p2uf60uk`. Exact reviewed source pins and captured bytes were independently compared. No coordinator/production callable, model/tokenizer, scientific draw/response, private-payload decode, test rerun, remote action, build, submission, collector integration or commit occurred. Only this review and independent evidence snapshots were written. Concrete packet review and actual source/runtime/custody/accounting acceptance remain future gates.
