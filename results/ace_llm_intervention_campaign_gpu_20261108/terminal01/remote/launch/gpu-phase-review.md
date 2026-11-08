# Independent review of campaign GPU phase

November 8, 2026. The coordinating reviewer did not author this module. This review reads the implementation, its captured helper contracts and new fabricated test evidence. No model, GPU, simulator response or original test suite was run by the reviewer.

Reviewed module SHA-256: `c0783ad85ad49dd5a9bcb147420c66fa74fa86484a04bc6951443cb23f3552ad`.

The phase executes the frozen, SFT and DPO services sequentially through the existing campaign API. It registers the actual policy object before constructor execution, captures its actual process handle and uses only that handle for cleanup. It retains a separate supervision directory outside all sealed CPU and GPU attempt trees. No process authority is reconstructed from a saved PID.

The original outer monotonic epoch controls the 240-second first-work limit, 550-second work limit and 580-second final bound. Each service has a 120-second startup bound, a maximum 180-second session and a 90-second fresh-progress requirement. Both a positive CUDA generation record and an independent owned NVIDIA process observation are required. The observed host-work counters do not claim to measure every SCM update or its CPU time. Allocation and actual runtime qualification remain outer responsibilities.

The separate outer-wrapper review identified a required cleanup-contract defect. The predecessor could raise to its caller while the only live cleanup watchdog remained in the exiting controller. The corrected `_hold_owned_cleanup` retains the actual handle until process-group closure, a dedicated empty owned-GPU observation and sampler termination are confirmed. Otherwise the original hard deadline exits the controller after its final owned-process action. Failed observations and failed evidence writes cannot authorize return. This fixes the identified handoff defect without accepting failed campaigns or extending the original deadline.

The caller must transfer its inherited timer to the overall work deadline before entering this phase. An inherited 240-second startup timer would otherwise terminate healthy later services. The new outer wrapper implements that transfer. The phase continues to enforce first work independently.

No other required defect was found in the inspected phase. Failed or incomplete arms remain preserved, later arms are marked unstarted when time permits, and no retry or private finalization occurs here. Successful return still requires outer source, input, runtime, training-lineage and dynamic evidence validation, external scheduler reconciliation and independent acceptance.

The first ten mocked methods and four subsequent new closure/isolation methods passed before the cleanup handoff correction. The new direct cleanup-hold method then passed both delayed successful closure and a simulated hard-exit branch. Those are local fabricated tests, with no actual signals, model workers or GPUs. The materially changed phase-level cleanup-query method also passed in a separate focused rerun, using 0.121696 child CPU seconds. That invocation ran no other methods. Historical sources and receipts remain unchanged. This review does not qualify the unseen concrete packet or an actual campaign execution.
