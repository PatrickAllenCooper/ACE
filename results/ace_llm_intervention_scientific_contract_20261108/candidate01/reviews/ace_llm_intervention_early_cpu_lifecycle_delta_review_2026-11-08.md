# Early scientific CPU lifecycle delta review

2026-11-08. **Required findings: 0.** Independent static review limited to the captured shared-owner integration in the realization/collection outer and token preparer. Existing scientific, protocol, collection and preparation reviews are preserved.

Reviewed SHA256 pins:

- `supervise_llm_intervention_scientific_cpu.py`: `794fd63af304ad7093c943f57dece7668ddd6bd5102a8c5db55cea19ee5054f9`, predecessor `f0342ebec6f6b237e7a3fddf24786b4af14891b36fce0669746a9a474109e6d8`.
- `prepare_llm_intervention_scientific_training_cpu.py`: `47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3`, predecessor `225c89d322d7c312da5d2e242db068129cb83f6e0b728a0c6c211dcee0804838`.
- Shared `llm_intervention_cpu_owned_supervision.py`: `1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8`.
- Focused test snapshot: `bef505e941036ac071cfef1686f619b6b87b5b6da909bf4b6684309e319c3f46`.

The complete predecessor diffs contain only the helper name/pin, parent supervision integration, owner initialization and top-level failure/finally handling. Scientific binding, exposure, tokenization, runtime observation and worker functions are unchanged. The helper is part of each exact authenticated input roster, executed from captured bytes; neither caller changes globals in the old closed supervisor. Both retain the owner on args before staging, provide startup/work/hard times from the original parent epoch, pass the same packet pin and owned cleanup callback, and use the returned cleanup result before existing saved-worker validation. Their startup markers match the helper's exact three-field contract.

The shared API independently observes leader/group closure, retires its handle only after closure, preserves primary versus cleanup/publication errors, and holds an uncertain live handle under the original absolute hard deadline. Its bounded cleanup is followed by hard-deadline holding when necessary. Both main functions invoke that holding path before canceling the alarm, including failure-receipt publication errors. The token preparer now protects the original exception against a failure-receipt write error; the stage outer retains its existing best-effort failure publisher with an additional outer guard. This delta creates no execution/resource authority or changed scientific acceptance.

Read both retained attempts under `/Users/pat/.codex/tmp/ace-scientific-early-cpu-lifecycle-0eg_vibo/`. Attempt01 ran three new methods: AST/integration inspection and token-preparer failure unwinding passed; the stage-wrapper fixture failed before its mocked parent because macOS `/var` versus resolved `/private/var` violated the existing resolved-output check, followed by the test's empty-event assertion. Attempt02 changes only `Path(tmp)` to `Path(tmp).resolve()` and reruns only that failed method, which passes for realization and collection. Execution logic did not change between attempts. Costs: attempt01 0.064927 child CPU / 0.068118458 wall seconds; attempt02 0.055401 child CPU / 0.058467750 wall seconds. The failure and both source copies remain preserved.

The new main-path fixtures use fabricated parent/owner objects to establish original-exception survival and hold-before-cancel ordering; AST checks establish use of the captured helper rather than old supervisor globals. They do not independently qualify staging, process termination or actual scientific execution. The shared helper's separate lifecycle tests/review remain separate evidence and were not rerun here.

Current source/helper pins match the retained attempt02 map. No tests, project source, numerical/model imports, tokenization, responses, replay or remote work were executed by this reviewer. No author files were edited. Optional suggestions: none. Concrete packet/source closure must include the new pinned helper; actual runtime/custody acceptance remains external.
