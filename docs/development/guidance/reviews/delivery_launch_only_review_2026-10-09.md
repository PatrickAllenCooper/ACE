# Actual-manifest local launch-only review

Distinct reviewer: Volta. Scope: shared production constructor, environment budgets, descriptor transport and safe Darwin launch-only harness. Static review only; reviewer ran no code or remote command.

One required finding: the harness's owned `ps` probe could remain running/unreaped when `communicate(timeout=2)` raised. Fixed by killing only that probe, attempting bounded reaping and retaining the original exception with separate cleanup notes. The probe PID remains excluded from existing-child checks.

Final scoped recheck: **zero required remaining fixes**. It also checked the individual environment-entry bound and aggregate argv/environment budget.

This review does not establish actual Linux memfd seals, subreaping, Slurm allocation, target dependency/native origins or numerical replay. The qualifier and platform adaptations in the harness are explicit substitutions. The original job33625983 failure and all18 terminal objects remain unchanged.
