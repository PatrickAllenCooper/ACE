# Narrow independent review request: training-data assembly

New implementation only, November 4, 2026. No scientific generation, model work,
remote execution, installs, commits, or old test suites. Parent owns integration,
runtime admission and the frozen 32/24 study schedule.

Changed source: scripts/research/llm_intervention_training_dataset.py.
New tests: scripts/research/test_llm_intervention_training_dataset.py.
New guide: docs/development/guidance/ace_llm_intervention_training_dataset_2026-11-04.md.

Review the callable single-world assembly, especially pre-outcome plan/teacher
assignment, cost-threshold crossing and terminal rules, fixed candidate feasibility,
ResponseAdapter/ChargedExecutor linkage, isolated trajectory learner, complete
replicated feedback and matched record/view linkage, and partial-failure charge
accounting. Existing world/contract/SCM/task/policy/branch/supervision logic is a
dependency interface, not reopened review scope.

The sole new test invocation passed six methods: "Ran 6 tests in 0.345s", exit 0.
Python 3.9.6; child wall 0.484445 s, user CPU 0.252952 s and system CPU 0.102776 s.
The exact invocation/environment/source, full output and all artificial fixture
journals (including deliberately induced failures) are in test-attempt01.
No failed preliminary test execution. One initial read-only rg search returned
exit 1 for no AGENTS matches. No other nonzero shell execution occurred.

Main independent analytic oracle: trajectory 5 responses/7 units, branches
18/24, scoring 12/30; 35 attempts/61 total units, zero unknown returns.
Milestones [0,1,2,6] fire at actual acquisition costs [0,2,2,6].
The one-step gain oracle is 5/36 for unit-cost actions and 5/72 for cost two.
A separate analytic fixture retains -0.02 gain and its tied/strict pairs.
Invalid scoring return and interrupted trajectory fixtures remain failed,
including one charged unknown return and a prior completed record.

Limits: independent runtime/source authentication, honest public text, scientific
namespace admission and world separation are NOT qualified. Pins are supplied/
observed bindings. Structural scores are explicitly privileged training labels.
Terminal stop is budget-only. Infeasible candidates/unreached milestones fail
without dropping them. Fixed heuristic ties remain in the shared record; the
existing matched_views exposes strict pairs to training consumers. Partial
records from a failed assembly are not authorized for complete-case training.

Please return required defects only with lines, without model execution,
scientific generation, old test reruns or edits. This capsule is a handoff,
not a claim that independent review has already passed.
