# Scientific CPU campaign wrapper review

Independent parent review of the agent-authored wrapper and its focused fixtures. The review found two required lifecycle corrections in the initial source. Both are resolved in the current source. **Remaining required findings: 0.** This is source review, not execution acceptance.

Reviewed final pins:

- `prepare_llm_intervention_scientific_campaign_cpu.py`: `8938d0d3a450f75e8f333989b70c89206de88ebcec928f7fe8d6a0d68c6166df`.
- `llm_intervention_cpu_owned_supervision.py`: `1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8`.
- `test_prepare_llm_intervention_scientific_campaign_cpu.py`: `7fda9fd7bf6553cb6bf8f5bc1f930a1186e80239dd52d048b0d6491e31072f79`.

## Findings and corrections

The initial wrapper, SHA256 `6fb403616a58a11b15b33a8d10649280cc736e48a2c3ee07791dec9d13d5395d`, called the closed artificial supervisor and then unconditionally canceled its alarm in the outer finalizer. A cleanup exception or unresolved process group could therefore outlive the controller. The new captured helper keeps the actual process handle through parent validation, receipt publication and the top-level failure path. It retires authority only after observing both leader exit and group disappearance. Otherwise it retains the original absolute hard deadline and attempts cleanup of that owned group. A receipt claiming closure cannot replace the independent observation. No historical helper was modified.

The inherited wait also switched to the work deadline without independently observing the declared startup milestone. The new supervisor checks the worker's packet pin, PID and monotonic timestamp in `worker/work-start.json`, and enforces the original absolute startup deadline before switching to the work deadline. Neither clock resets after staging. Failure handling preserves the first error and records secondary cleanup or publication errors. Logging failure cannot cancel an unresolved child's watchdog.

The five new lifecycle methods passed once with fabricated process, clock and signal objects. Their recorded child CPU is 0.072506 seconds and elapsed time is 0.078428750 seconds, with 31,129,600 bytes peak RSS. The original ten methods were not repeated. Initial and corrected source, test output and the delta are retained in `/Users/pat/.codex/tmp/ace-scientific-campaign-cpu-lifecycle-_hxbb5q2`. These tests do not establish actual Slurm or process-tree behavior.

## Scientific and interface review

The wrapper requires independently accepted scientific LLM training, the original accepted numerical plan, complete realization metadata and all 24 development descriptors. It checks the shared exposure and checkpoint roles before execution. The numerical ranker fits once using the same 160 supervised records and D selected preference pairs, with D between 1 and 128. It deploys the final ranker export without selecting a fit using development outcomes.

All 192 cells and the 96 numerical policy bindings are constructed before campaign responses begin. The CPU stage executes only the five numerical policies across all 24 systems. A partial policy attempt prevents success and further policy execution. The later GPU phase receives the unchanged bundle and attempt pins. No world-generation or private-scoring call is added. Private descriptors remain trusted controller inputs and are not included in policy stdin. This is the existing trusted-code boundary, not an operating-system sandbox against malicious code running as the same user.

The source preserves the proposal's task, learner, graph, action menu, initial actions, noise declarations, world IDs, measurement namespaces, strata and policy identity bindings. Exact service capacity is 576 requests for each prospective LLM service. Saved validation checks original ranker receipts and reload declarations, all 120 CPU arms, complete numeric-call evidence, actual runtime paths and versions, source hashes, output inventories and unchanged input bytes. It does not independently reload numerical model tensors during validation.

Resources are explicitly supplied and checked against actual Slurm allocation: 2 CPUs, 7,680 MiB, zero GPUs, account `ucb736_asc2`, `acpu/cpu-normal`, no requeue and at most 6,000 seconds. Per-decision timeouts are explicit and bounded. These ceilings do not prove that the scientific workload will finish within them.

The original ten new methods passed across three preserved attempts at `/Users/pat/.codex/tmp/ace-scientific-campaign-cpu-tests-mrhwe7me`. They used fabricated input records and substituted execution APIs. Total recorded child CPU was 0.602550 seconds and elapsed time was 0.683512666 seconds. The review inspected the source, tests, retained results and lifecycle delta without executing them. No model, tokenizer, fit, response, scientific draw, runtime staging or remote action occurred during this review.

The exact CPU acceptance interface is described in `/tmp/ace-scientific-campaign-cpu-interface.md`. The forthcoming GPU wrapper and finalizer still need their own integrated packet checks. The combined scientific freeze, actual execution, full custody and independent result acceptance remain open.
