# Supplied development campaign assembler

`scripts/research/llm_intervention_scientific_campaign.py` adds pure preparation for the unchanged prospective 24-development-world × 8-policy design. It reuses `prepare_plan` and `prepare_campaign`; it performs no generation, responses, execution, fitting, scoring, model/tokenizer loading, filesystem work or scheduler action. Existing preparation constructs fresh zero-prior SCMs for validation only. Returned descriptors are private driver material, not policy-facing envelopes.

Call `numeric_policy_bindings(proposal=..., proposal_sha256=..., expected_source_pins=...)` first. It returns 96 world/policy bindings. Append each exact `["--binding-json", canonical(binding), "--binding-sha256", binding_sha256]` suffix to its caller-owned numerical command.

Then call `build_scientific_campaign` with these keyword arguments:

- `proposal`, `proposal_sha256`, `expected_source_pins`: unchanged existing proposal, independent semantic digest and exact independent source-pin map. Existing `validate_proposal` rejects schema/design/source disagreements. No source file is opened or authenticated by this pure helper.
- `worlds_by_id`, `expected_world_pins`: separate exact integer-keyed maps for all 24 declared development IDs. Values are typed supplied `WorldDescriptor`s and independent semantic descriptor pins. Scope, ID, graph and public known noises must match. Realized coefficients are preserved, never generated, selected, or certified as draws from the proposed law.
- `policy_specs_by_world`: exact 24-ID map of ordered existing six-field policy specifications. Execution IDs are `uniform_random`, `menu_coverage`, `local_information`, `target_risk`, `learned_ranker`, `frozen_llm`, `sft_llm`, `dpo_llm`. The last four correspond to proposal arms `numerical_ranker`, `llm_frozen`, `llm_sft`, `llm_dpo`; their original proposal namespaces are preserved, not rederived from aliases. Commands remain caller-owned. Existing campaign validation preserves cross-world identity/provenance/timeout consistency and exact persistent commands.
- `services`, `expected_service_identity_pins`: the existing exact three-service settings schema and separate three-entry identity digest map. Each identity binds declared model, adapter/checkpoint, service source and numerical recipe; each service has exactly 576 maximum requests. Actual checkpoint bytes/roles, launcher behavior and source/runtime qualification remain caller-owned.
- `outer_provenance_receipt_sha256`: existing caller-owned outer provenance pin passed unchanged to every plan.

The output includes the existing campaign, whole-campaign semantic digest, 24 ordered source-plan pins, supplied-world pins, proposal/source/service identity pins, explicit alias mapping and numerical bindings. All scientific/source/runtime/scheduler qualification and execution/freeze flags remain false. Pin maps authenticate declarations against independent expected inputs; they are not admission certificates.

The campaign retains all four regimes, three graph strata, two cost strata and 192 cells. Every arm separately pays eight shared-key initial units and at most 24 acquisition units: 1,536 logical initial units and at most 6,144 total trajectory units. Arm-specific measurement namespaces remain distinct; shared keys do not make initialization free. Five policies have CPU phase labels and three have GPU labels; these are not allocations. Each persistent service covers at most 24 × 24 = 576 decisions (1,728 across three services).

Focused validation: **14 new tests passed**, no old tests run. Fixtures use explicit fixed coefficients and fabricated pins/commands, never scientific draws or responses. Guards prohibit world generation/evaluation, policy execution, learner updates, subprocesses and I/O during assembly. Negative cases cover proposal/source mutations, roster omissions/extras/order, descriptor/pin/scope/graph/noise mismatches, numerical binding substitutions, service/checkpoint identity disagreements and cross-world drift. The complete campaign is also validated through the existing pure API.

Exact test attempt: `results/ace_llm_intervention_scientific_campaign_preparation_20261108/attempt-01/`, including source snapshots, dependency `source-pins.json`, `execution.json`, stdout and stderr. Command: `/Library/Developer/CommandLineTools/usr/bin/python3 -B -m unittest -v test_llm_intervention_scientific_campaign`, from the retained `source/` directory. Exit 0; unittest 4.316 seconds; wrapper elapsed 4.429138 seconds; child CPU 4.408839 seconds. No failed attempt occurred.

- Source SHA-256: `e3072b8a4c18ef459154fc71ebec8874992d4cee7c829e55b4203cf5111565d8`.
- Test SHA-256: `3ce579831eb92d98934fa7ce231b2cd5dfddab52a1c1517bbe96c5cc2bfb0309`.

This handoff is ready for independent review. It does not authorize realization, scientific collection, service launch or a new freeze.
