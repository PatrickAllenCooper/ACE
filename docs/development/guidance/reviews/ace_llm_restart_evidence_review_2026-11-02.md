# Evidence and literature audit for the renewed ACE agenda

Two separate read-only agents examined historical implementation/evidence and primary literature on November 2, 2026. Their findings informed the new agenda. Neither agent ran models, fixtures, jobs or experiments, and neither edited the manuscript. This record summarizes their findings and the parent integration. It is not a review of a completed new method or a guarantee of publication novelty.

## Historical code and evidence audit

Agent Darwin examined ACE and ACE-Runner. The principal result is that LLM benefit remains unestablished, with meaningful adverse evidence rather than only an unfinished experiment.

- Mission-1 accepted 44 runs across four arms with eleven paired acquisition seeds on one emulator. After common refits at equal charged calls, the reported LLM/random-proposer ratio is 1.0412, with reported 98.75% interval 0.8870–1.2221 and two improving seeds out of eleven. Device differences and imperfect trajectory reproduction limit attribution. The parent checked the numbers and execution caveats directly in [the statistics appendix](/Users/pat/code/ACE-Runner/studies/2026-10-peter-baseline/package/stats_appendix.md). These are reported historical statistics, not a newly audited confidence procedure or population equivalence claim.
- The historical [publication policy](/Users/pat/code/ACE-Runner/docs/decisions/0002-baseline-publication-policy.md) selected favorable bundles. Those selections cannot supply an unbiased success rate.
- `policy_update=none` does not necessarily freeze all weights. The [cloud configuration](/Users/pat/code/ACE-Runner/ace/config.py:278) can still enable teacher pretraining. New frozen controls must disable every update path and verify hashes.
- The [Runner likelihood](/Users/pat/code/ACE-Runner/ace/dpo.py:287) scores command tokens without the experimental-state prompt. The [research implementation](/Users/pat/code/ACE/ace_experiments.py:1067) uses state-plus-command training. These are not interchangeable policy objectives.
- [Code-health notes](/Users/pat/code/ACE-Runner/docs/code_health.md:164) document parsing a prompt example when the completion contained no command. New completion-only parsing must bind generated tokens, parsed action and executed action.
- The [scientific critic](/Users/pat/code/ACE/ace_experiments.py:865) can access true mechanisms outside the ordinary experiment executor. Switching one lookahead option does not establish oracle-free deployment. New deployment observations must all be charged through the executor.
- The historical candidate proposer, scorer and hand-built bonuses require separate treatment. Compare proposals under a common scorer and random selection among fixed-size candidate sets. Do not attribute scorer benefit to the language proposer.

Reusable components include append-only observation logs, common-refit evaluation, paired study analysis and connected acquisition code. Reuse requires explicit version and interface qualification. These audits do not qualify a corrected implementation.

## Independent literature assessment

Agent Hypatia identified the strongest novelty pressure as learned LLM information gathering, numerical amortized experimental policies and known-graph causal optimization. Primary sources and exact scope are integrated in [the agenda](../ace_llm_intervention_agenda_2026-11-02.md).

The review prioritizes DAD as a numerical learned-policy comparator, information-gain preference training and ASIG as prior learned LLM policies, MCBO as causal numerical design, LLAMBO and LMABO as frozen LLM acquisition, Model Discovery Agent as LLM hypothesis generation, and Gupta et al. as the motivation for shuffled-feedback and embedding-parity controls. The parent opened these primary sources during integration. No literature statement establishes ACE's novelty or future success.

The recommended hypothesis is transfer of intervention selection and recovery from misleading semantic priors. Compare correct, anonymous and misleading metadata with real and shuffled policy feedback. Give numerical policies the same training supervision and usable semantic features. A numerical student matching the LLM would weaken a language-specific claim. Gains unchanged by shuffled feedback would support a prior-based explanation rather than experimental adaptation.

## Disposition and limits

The agenda incorporates all material audit findings, preserves negative results, separates training from deployment, and requires an independent acquisition-versus-learner comparison. It adds a simple conditional one-step decision-error decomposition as a research starting point. That decomposition is parent-authored and was not independently proof-reviewed in this audit.

The interactive agenda passed a local TypeScript check. Opening was queued in Codex. No pixel-level review or runtime execution of the new scientific method is claimed. The archive is a committed-source snapshot, not a fresh PDF export. Old submission and exploration automations remain paused. New model execution, installation and resource allocations remain outside this completed design task.
