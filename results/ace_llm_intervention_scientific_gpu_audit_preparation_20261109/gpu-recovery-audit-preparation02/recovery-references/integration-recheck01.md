Focused independent integration recheck — R2/R3 — 2026-11-09

**0 required findings; R2 and R3 closed on the exact sources below.** This succeeds integration-review01 SHA256741b51b19dd57d3d7fa246e6768c72dd514f9b62e79eee831455bb39be82a9b0, which remains unchanged with its initial findings. R1 remains closed by the previously reviewed schema-only correction; it was not re-audited here.

R2: adapter line292 now assigns the callback inside the restoring try/finally. The original callable is restored when entry is interrupted; the outer adapter finally still retires the callback. Entry line611 uses a language-level `with historical.install(campaign)` spanning the original controller barrier, GPU phase, runtime/service validation, post-custody marker, final input readback and terminal publication. Manual __enter__/__exit__ and delayed context registration are removed. The independent outer loader-removal finally remains. Static normalization of only these scope changes confirms the entire substantive controller body and loader finally are otherwise identical to reviewed entry39ced.

R3: authenticate line357 passes its already authenticated admission object to recovery_contract. The new parameter and equality at455 require admission['accounting_sha256'] == context['baseline']['sha256']; the existing context-to-authority baseline equality remains. There is no new schema, inferred live baseline, or change to the original admission predicate. Static AST comparison confirms those are the only authenticate/recovery_contract changes.

Exact delta: the adapter changes only placement of its assignment inside the restoring try. The entry changes only that helper pin, restoration scope and admission join. All other entry AST is unchanged from39ced, including scientific/GPU validators, import closure, clocks, resources and source bindings. The generator was inspected as source: it emits the updated admission call/guard and wraps the existing controller span in the with scope; it was not executed. Exact source diffs, AST comparison records and captured predecessor/current bytes are included.

Saved new evidence, not rerun: delta-test01 ran three new methods. The actual-entry with-body exception fixture and baseline equal/unequal fixture passed; the assignment-boundary fixture errored because its ModuleType subclass was correctly rejected before installation. That failed original is retained. Delta-test02 ran ONLY the affected method with an exact ModuleType and a line-trace BaseException immediately after callback assignment. It passed, asserting the body was not entered, the same primary exception propagated and the original callable was restored. The synthetic fixture imports only the new stdlib adapter; the isolated with AST replaces its body with the artificial exception, and the admission fixture uses the existing checker stub. No production/scientific callable was run by these fixtures.

Recorded delta-test01: exit1,0.159263 child CPU-seconds,0.164053875 elapsed seconds,26,722,304 peak RSS bytes. Delta-test02: exit0,0.136840 child CPU-seconds,0.140505834 elapsed seconds,22,380,544 peak RSS bytes. Combined0.296103 child CPU-seconds and0.304559709 elapsed seconds include the retained failed fixture. Current and tested source copies match all nine checked source bindings. The reviewer executed no tests, generator/project callables, actual metadata admission, remote operations, models, tokenizer or scientific work, and made no Git changes.

Reviewed pins:

- Entry:696be95ff571d584f4600b5a87fe6a442eaa2da863d07c0d464ac22d4e1de1fa.
- History adapter:674d114f496c63cd9891f6c2d47b0d74a9a657090c0751215f91e2580cc941c9.
- Generator:30e9cc2b2cfe7c51ae54032b1f24a864f1cd9e613ee2692f1f3065879581cac8.
- Unchanged preflight helper:b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8.

This closes the remaining source/integration findings from the initial review. It is not an actual preflight, authority-adoption, coordinator/concrete-packet, submission or scientific acceptance. Those independent operational and full post-custody gates remain required. Optional findings: none.
