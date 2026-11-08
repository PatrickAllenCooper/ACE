# Scientific GPU phase derivative review

2026-11-08. **Required findings: 0.** Independent static review of the new scientific phase helper and new tests, limited to the derivative delta and integration boundaries. No source/test execution, numerical import, campaign, model, tokenizer, response, replay or remote operation was performed.

Reviewed SHA256 pins:

- `llm_intervention_scientific_gpu_phase.py`: `c89e26d2393b97b6115dfd29969b2d9f7fbdf0ebd2dc1505ac604d15faf4d1d7`.
- `test_llm_intervention_scientific_gpu_phase.py`: `2ef4c8cf23d4b502724199d8877d9a064391fc71d07ebeb4cf0e0071d90b215b`.
- Immutable predecessor `llm_intervention_campaign_gpu_phase.py`: `c0783ad85ad49dd5a9bcb147420c66fa74fa86484a04bc6951443cb23f3552ad`.

The full textual diff contains only the intended scientific profile, explicit limits argument/validation, corresponding receipt fields and explanatory documentation. Original process factory/registration, owned-handle cleanup, signal controls, failure retention and telemetry algorithm remain unchanged except the generation ordinal ceiling. No closed supervision suite was repeated.

The phase requires `development`, exactly 24 ordered world IDs 202611070100 through 202611070123, eight arms and the three frozen/SFT/DPO GPU services. Existing campaign validation binds supplied source plans and service identities. Every service must declare integer capacity exactly 576; this is a maximum capacity, not a requirement to generate 576 responses. The watcher accepts contiguous ordinals 0..575 and continues advancing past the old eight-request limit. Ordinal 576 is retained as uncertainty and is not counted as positive progress. Complete request/trajectory accounting and rejection of invalid final artifacts remain the outer validator's obligation; the watcher is not a substitute for that validator.

`_scientific_limits` requires an exact integer map: 120 <= reserve <= cap <=1200, 240 < work <=3540, work+20 <=final <=3570, plus three remaining service reservations at entry. Supplied absolute work/final deadlines must equal the original outer start plus those offsets. Each later service must still leave reserves for the remaining services and receives the smaller of its cap and that remaining-budget boundary. Global first verified GPU work remains due at original-start+240; each service has 120 seconds, and both owned GPU memory evidence and advancing positive generation evidence must remain fresh within 90 seconds. No per-service reset of the global clock was added.

The unchanged cleanup adapter calls `arm_final_deadline(final_deadline-580, owned_process)`: with the admitted helper's start+580 contract, this targets exactly the supplied absolute final deadline, not another 580-second extension. Inherited alarm handling/restoration, registered-process authority, dedicated post-exit empty-GPU observation and unresolved-cleanup custody are preserved. The terminal retains false source/runtime/allocation/dynamic/scientific qualification flags.

These are local timing ceilings, not admission of a 3570-second job or of the complete remaining scientific cycle. The external supervisor/packet must authenticate the frozen limits, preserve the original epoch, fit staging/work/cleanup in its actual allocation, and reconcile all CPU/GPU costs and other stages against the remaining cycle budget before any submission. This source neither changes that budget nor allocates resources. Saved samples establish positive owned-process memory plus GPU generation evidence, not a measured SM-utilization percentage or proven full-workload throughput.

Retained evidence: `/Users/pat/.codex/tmp/ace-scientific-phase-qa5m47oh/`. The existing transcript reports six new stdlib AST-extracted tests passed, return code 0, 0.113302333 seconds wall, 0.110679 child CPU seconds and 22,478,848 bytes peak RSS. They exercise explicit limits, absolute deadline/reservation rejection, request ordinals beyond eight and at 575/576, and AST preservation of ownership/cleanup functions. They do not execute a campaign or establish actual GPU capacity. Current source/test/predecessor hashes match retained `source-pins.json`.

Evidence pins: `run.json` `a493fc151e476ee3162bbeba8335325771199521524ada68c2b5b06197007b73`; `source-pins.json` `d4f5c7ad258ea264e17e58aa3c5e15b48c32801dba1e20acd196bd2c246fa2f0`; `stderr.log` `d6c9ba23a7acf4686b0a67a299a045c8508c175d0b09f218726720dbcb7accf9`.

Optional suggestions: none. Concrete outer/packet review and independent actual-result acceptance remain pending. Author files and predecessor evidence were not edited.
