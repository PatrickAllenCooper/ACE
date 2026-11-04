# Persistent model-service integration: bounded static review

**Disposition: three required P2 findings remain before packet/submission.**
The concurrent saved-session validator was included in the final inspection;
it does not resolve the supervision findings below. No tests, service workers,
models, GPU operations or remote commands were run for this review. Only this
review document was written. Singer's model-service review remains separate.

Reviewed November 4, 2026. The initial driver had 198 lines and SHA-256
`99c82d8223139a5101c0d371a4b6417b57ae694cadfaa2edf86addf00c04d235`.
The final inspected driver includes `validate_session`, has 247 lines and SHA-256
`70be5bc0c6783c61e0ec05d2f675d454e9c6f42e3da7243dde33784dac383487`.
References below use the latter version; original locations are supplied for
the findings that predate the concurrent addition.

## Required findings

1. **P2 — Retain an independent absolute failure/cleanup deadline and owned-group fallback.**
   The [560-second alarm](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:106)
   only raises a Python exception. After that one-shot alarm fires, the
   [failure path](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:230)
   does not arm a final watchdog and ultimately disables the timer. Cleanup and
   failure-record writes therefore have no remaining absolute deadline. The
   transport's cleanup intentionally catches `BaseException`; its failure path
   performs receipt I/O and marks the object closed even when cleanup remains
   uncertain. Calling `current.close()` again can consequently be a no-op. In
   addition, `current` is assigned only after the constructor's handshake, so a
   constructor failure leaves the driver without that object's process handle.
   This is a concrete gap on timeout/startup-failure paths, not a claim that the
   normal five-second transport cleanup is unbounded. The reused GPU09
   [hard failure watchdog](/Users/pat/code/ACE/scripts/research/supervise_llm_intervention_checkpoint_gpu.py:171)
   explicitly covers cleanup, stalled receipt I/O and shutdown with a final
   owned-group kill and hard exit; the new driver does not use an equivalent.
   Reserve cleanup time within the ten-minute allocation, retain ownership of
   the launched group's identity through constructor failure, and keep a final
   absolute watchdog active until termination. Scope any fallback strictly to
   that owned group. Original locations: lines 48–53, 142–144, 181–194.

2. **P2 — Restore successful independent GPU activity and post-close cleanup checks.**
   The [acceptance predicate](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:223)
   accepts any nonempty `owned_rows`, ignoring the sampler's `returncode`.
   `gpu_process_sample` retains parsed stdout rows even for a nonzero exit, so a
   failed/partial `nvidia-smi` invocation can satisfy the independent activity
   gate. Require a successful sample for the corresponding session/PID; retain
   unsuccessful samples as uncertainty. Separately, neither the per-session
   close path nor final acceptance requires a successful post-close sample with
   no owned rows. The sampler can stop before taking one, or keep reporting
   rows, and an earlier positive sample still allows `terminal.json` to say
   complete. The reviewed transport confirms zero leader exit and a vanished
   process group, which is useful ownership evidence, but it does not perform
   the independent GPU cleanup check used by
   [GPU09](/Users/pat/code/ACE/scripts/research/supervise_llm_intervention_checkpoint_gpu.py:302).
   Persist and validate that check after each session, before admitting the
   second session or declaring completion; preserve uncertainty on failure.
   Original locations: lines 125–127, 154–175. The new saved-session validator
   strengthens process-close evidence but adds no GPU cleanup observation.

3. **P2 — Preserve scheduler output before checking command or allocation success.**
   The [scheduler call](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:169)
   uses `check=True` and retains only `.stdout`. A nonzero exit therefore jumps
   to a failure receipt without saving its captured stdout/stderr. On exit zero,
   `authenticate_allocation(...)` is evaluated before `save(...)`; a rejected
   allocation similarly loses the raw scheduler response. The generic exception
   string/traceback does not preserve those outputs. Save job ID, command,
   return code, stdout and stderr first; preserve partial output on timeout;
   then apply the unchanged allocation guard. This retains the evidence needed
   to diagnose a rejected account/resource/ownership assertion without another
   scheduler query. Original locations: lines 111–114.

## Concurrent saved-session validator

The parent added `validate_session` during this review and calls it after each
session with the expected freeze, identity and four fixture-envelope pins. The
inspected implementation checks four generation filenames/ordinals, wire IDs
40–43, command and response-digest consistency, startup identity/freeze pins,
zero update/response declarations, exactly one occurrence of each model-loading
phase, unchanged-tensor/cache declarations, positive finite CUDA intervals,
positive allocated-memory values, clean unsignalled transport close, raw-token
sidecar/hash consistency, and identical first/fourth prompt and sequence tokens.
Existing `failure.json` evidence prevents successful qualification.

This addition closes the original silent acceptance gap around the third
fixture: `choose()` builds an envelope from the stress state and currently bound
case3 task, but the validator now demands the pinned complete stress-envelope
digest. The historical GPU09 case3/stress tasks inspected here are equal; if a
later supplied stress task differs, acceptance now fails rather than silently
qualifying a different request. Detecting that mismatch before model loading
would be cheaper, but it is not an additional required correctness finding.

The saved test receipt and stderr report four new synthetic methods passing in
one invocation, with 15 negative subcases, exit zero, 0.066462 child CPU seconds
and 0.072684708 elapsed seconds. The current driver and test source are
byte-identical to their `preparation01` copies. This review read the source and
saved receipt without rerunning tests. The fabricated records exercise evidence
rejection logic; they establish no actual model load, generation, GPU activity,
cleanup, determinism or real-session qualification. Receipt consistency checks
also do not independently remeasure the worker's tensor state or recompute its
tokenization. Those claims continue to depend on the reviewed implementation
and a later real execution with independent telemetry and complete custody.

## Source closure, resource scope and remaining acceptance

The driver authenticates the outer packet bytes, captures declared input bytes,
requires the service source inventory plus its own source, the GPU09 helper,
two manifests, three fixtures and the adapter/receipt pair, and cross-checks
both manifests' source pins against the outer packet. The bootstrap executes
captured service/helper bytes; project imports use the service's captured-source
loader. The inspected dependency imports fit the declared project-source set.
Both services must share model-file pins/location; the frozen manifest forbids
an adapter, while the second binds an artificial adapter to the supplied receipt.
Installed packages are version checked, not authenticated as a complete runtime
byte inventory. The receipt's `status`/adapter digest are a lineage consistency
check, not a replacement for prior GPU09 external acceptance or packet review.

The unchanged allocation helper demands account `ucb736_asc1`, `aa100`,
`gpu-testing`, one A100 20 GB slice, four requested and four–eight allocated CPUs,
12 GiB and ten minutes. Each service declares 180-second startup, 30-second
request/idle limits, 240-second total and four requests; client limits are
185/35 seconds with five-second cleanup. The driver starts sessions sequentially
and uses the reviewed transport's separate process groups, exact handshake and
request binding, bounded output and monotonic request IDs. It supplies no
environment callback or training loop. Source declarations alone establish no
actual resource use or throughput. A completed packet, actual session records,
successful scheduler terminal accounting and full exclusive custody remain
required. No packet or submission is approved by this review.

Final inspected supporting SHA-256 values:

- Service (separately reviewed by Singer): `499494076357dfe7286718c0bccff7b3e4c3f0d0e6b2e9a483ab09a7b0357910`.
- Persistent transport: `8ccd3ae0bd28729af8ebb758c9bd474a926d05362799f45bf0133e1e4b5e4a5d`.
- GPU09 supervisor/helper: `ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec`.
- New validator test source: `96719708691922d83e092025413ba6eab91fc73aea45810bf76270704589b23e`.
- [Saved test receipt](/Users/pat/code/ACE/results/ace_llm_intervention_model_integration_preparation_20261104/preparation01/test_receipt.json): `1c4af03f8317922298392e98baaa6d6df324bf7e0e2b57a2ef06cd6915e348cc`.

Review operations were local read-only source/JSON inspection and SHA-256
calculation, followed by this document write. Full synthetic fixture-custody
reconciliation was not repeated. The world implementation was not modified.

## Final disposition after the preparation02 delta recheck

**Zero required findings remain for this reviewed implementation delta.**
Completed November 4, 2026. This addendum supersedes the initial three-P2
disposition for the exact hashes below. The original review remains unchanged
above and in `preparation02/initial_review.md`. Scope was limited to the three
corrections, their new focused methods and saved evidence. No tests, workers,
models, GPU operations or remote commands were executed during this recheck;
only this document was appended.

The deadline/ownership finding is resolved. The
[interrupt handler](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:144)
arms the hard absolute watchdog **before** raising the ordinary timeout, so a
nested transport `BaseException` handler cannot consume the only timer. The
watchdog targets the retained owned group with SIGKILL and calls `os._exit(124)`
at `start + 580 seconds`, leaving twenty seconds before the fixed ten-minute
allocation cap. Failure handling arms the same absolute limit without extending
it, retains it through cleanup/final I/O and does not cancel it in `finally`.
`current` is allocated with `__new__` before `__init__`, preserving access to
the process handle if handshake initialization fails. The independent GPU09
`terminate` helper is invoked on failure even when transport `close()` has
become a no-op; its result is saved. Successful terminal writing is also covered
by the final watchdog, which is cancelled only after that write completes.
This is a static control-flow conclusion, not an executed signal/kill test.

The telemetry finding is resolved. Final activity acceptance now requires both
a matching PID and a successful sampler return code with nonempty owned rows.
Immediately after each transport close, the driver
[saves and validates](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:256)
an explicit independent GPU sample requiring return code zero and an empty
owned-row list. Failure or uncertainty rejects the session before the next
session can start or the run can be declared complete. The transport's separate
zero-exit/group-cleanup checks and the saved-session validator remain in force.

The scheduler-evidence finding is resolved.
[`scheduler_record`](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_model_service.py:52)
uses `check=False`, saves command/job ID/return code/stdout/stderr before any
nonzero-return rejection, and returns stdout only after that durable write.
Allocation validation therefore cannot discard the observed scheduler response.
The timeout path separately preserves available partial stdout/stderr and a
timeout status before rethrowing.

The saved receipt names only the three new scheduler, GPU-sample and watchdog
methods. Its stderr records all three passing once, exit zero; measured child
CPU is 0.057972 seconds and elapsed time 0.06981825 seconds. These methods mock
the scheduler/subprocess, sampler inputs, signal installation, group kill and
hard exit. They do not demonstrate a real process kill, Slurm observation or
GPU cleanup. Static AST comparison confirms that `validate_session` and the
original four synthetic test methods are unchanged. Both working source files
are byte-identical to their saved preparation02 copies. No old method was rerun
by this reviewer or named in the retained focused command.

Final reviewed SHA-256 values:

- Driver: `fb3e95013cb46cd53f3b44ee72a141d75616c39b762b8e62ea90d89f3a9de717`.
- Integration tests: `7c57063e99f4904dfb63083e9b5cc9b1e2aedc2e642a2efb9e534261ee8408e4`.
- [Focused receipt](/Users/pat/code/ACE/results/ace_llm_intervention_model_integration_preparation_20261104/preparation02/test_receipt.json): `ce4a5e7cd7d32fc7529899f64725192b63617d99cca6535c6cc62605298700b0`.
- Saved stderr: `58de0024f29fab90436843ced1b20c5df20d9d7c4aaddea4f71b1b84595821dd`.
- Preserved initial review: `d8d9563d5c32d0d3b5c83fa84a97938044a62392d51201a57019726ae0cea923`.
- Persistent transport remains `8ccd3ae0bd28729af8ebb758c9bd474a926d05362799f45bf0133e1e4b5e4a5d`.
- GPU09 helper remains `ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec`.

This closes the three implementation findings. It does not approve an unseen
packet or submission, establish actual GPU qualification, replace Singer's
service review, or establish scientific utility. Actual two-session evidence,
independent telemetry/cleanup, scheduler completion and full exclusive custody
remain execution acceptance requirements.
