# Scientific service evidence validator review

2026-11-08. **Required findings: 0.** Independent static review of the new scientific derivative and its new tests only. The closed campaign-service join/custody review is reused; no old tests or saved scientific evidence were replayed.

Exact SHA256 pins:

- `scripts/research/llm_intervention_scientific_service_evidence.py`: `d136efa7814435757e2c9d5026784a891f51ecafd72ea611c772dd0481d0b2b6`.
- `scripts/research/test_llm_intervention_scientific_service_evidence.py`: `57a6e15d872e30d6831db55b3d92f470e3662e954556205c74ca39ca3b623c31`.
- Immutable predecessor `llm_intervention_campaign_service_evidence.py`: `26ec6d87ebc6aab2328e2873fec6746749b9162c103f2144d12a890a5c99f516`.

The textual diff is confined to the scientific profile, the new pure `_scientific_shape` gate, its invocation and reported limit metadata. `_arm`, `_generation`, `_inventory`, request-family/prefix helpers and the remaining request/trajectory/closure joins are unchanged. The retained AST-comparison test covers the named core functions; this review read the full derivative diff and its integration points.

The gate requires development scope, exactly the ordered IDs 202611070100–202611070123, all eight ordered policies, the exact five-CPU/three-GPU phase map and all three service identities/capacities. Each service has integer capacity 576; legal ordinals are 0..575. This is capacity, not a demand to generate 576 responses. Existing prefix and join checks still account for completed, pending, failed and unstarted work.

Every source has eight initial actions costing eight units, eight positive-integer-cost actions and 24 acquisition units. The all-eight-arm cost envelope is `8*24*(8+24)=6144`; response attempts are bounded by initial responses plus acquisition cap divided by minimum action cost. With the actual menu's cost-one actions, the upper response bound is also 6144. The three service decision capacities sum to 1728. These are ceilings, not observed expenditure. Exact frozen menu/action semantics remain bound by the independently admitted campaign and source plans; the pure gate alone is not scientific admission.

The shared inventory constants now admit at most 65,536 filesystem members (files plus directories), and 2 GiB of file bytes, separately for each checked tree. The unchanged implementation rejects symlinks/nonordinary members and rechecks both trees after validation. These per-service limits do not fix the separate finalizer's old aggregate inventory limit. No actual workload-size or validation-time sufficiency is claimed.

The result still covers one service rather than accepting all 192 cells. Partial accounting stays explicitly partial, operational failures remain failures, and scheduler GPU clearance, training acceptance, cross-policy shared initialization and scientific acceptance remain unqualified. Validation declares no new responses, updates, model loads or GPU allocations and does not replay token decoding.

Read retained evidence at `/Users/pat/.codex/tmp/ace-scientific-service-evidence-nwclnb9i/`: four new stdlib AST-extracted methods passed, return code 0, 0.051033625 seconds elapsed, 0.047589 child CPU seconds and 20,250,624 bytes peak RSS. Coverage includes roster/scope/cost/request mutations, 575/576 membership and preserved core ASTs. Tests use fabricated dictionaries/files; they do not validate actual scientific trajectories or runtime/model behavior. Current pins match the retained source-pin map.

No source/test execution, numerical/model imports, tokenizer, scientific response, campaign replay or remote work was performed by this reviewer. Author files and predecessor evidence remain unchanged. Optional suggestions: none. Concrete GPU outer integration and later independent actual-result acceptance remain required.
