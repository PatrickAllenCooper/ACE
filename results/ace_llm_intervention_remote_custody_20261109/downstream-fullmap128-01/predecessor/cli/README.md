# Local GPU builder CLI assembly — not executed

`assemble_gpu_cli.py` assembles literal arguments for the unchanged reviewed downstream01 GPU builder. It performs no imports/execution of builder, backend or project sources, no subprocess/network/Git/lock operations, and no scientific/private-payload decoding. Only audit/accounting/custody metadata is JSON-decoded. It does not grant acceptance or establish readiness freshness; the parent supplies independently accepted actual records and fresh readiness.

The 49 existing metadata path/SHA bindings are in `known-inputs.json`, including each original pin source. The utility authenticates that map by an embedded byte SHA, then authenticates each supplied file at future invocation. Current CPU launch packet anchor remains b98873cae0e04221abcce02856ea7c610319f27f4c07e8275cc9bcfce1c2cd40. The builder remains e41b787fe0ede1191408b0941e7f9d806b9d4d38fcd0cf5718aebc2bfa16f8e1 with its unchanged same-directory source dependencies.

## Required future actual inputs

Supply explicit PATH and SHA256 flags for all seven builder metadata records: `cpu-acceptance`, `cpu-baseline`, `cpu-review`, `cpu-phase-terminal`, `cpu-disposition`, `cpu-inventory`, `site-readiness`. No defaults or inferred future hashes exist. Acceptance must identify accepted development CPU phase job35482135 and its existing inner packet/actual phase-terminal pin; baseline must join that acceptance/disposition to the existing training baseline at46jobs/94journals, no outstanding reservation. Successful disposition must join the original CPU outer packet/result. The review and readiness bytes are independently pinned nonempty records; semantic independent review and freshness are the caller's responsibility.

Supply explicit PATH and SHA256 flags for all four storage records: `storage-receipt`, `storage-seal`, `storage-after`, `storage-reference`. Their named sources are respectively terminal02/remote-custody.json, custody-inventory.json, inventory_after_transport.json, predecessor-reference.json. Supply `--cpu-remote-custody` as repository terminal02/remote and `--cpu-terminal-mirror` as the actual disjoint external capsule in the seal. The intended repository terminal is:
`/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02`

The phase terminal must be its `remote/launch/cpu/worker/artifacts/cpu-phase/terminal.json`; disposition/inventory must be its disposition.json/inventory_transport.json. Actual custody completeness and full archive verification remain the independent audit/closed builder's work. This utility does not access a remote backup or recursively walk capsules. It never reads active collector sources.

The other four storage members are already pinned: authority011c2d3b..., authority_review8f858bef..., backend4161aa58..., backend_review8f858bef.... Paths and full hashes are in known-inputs.json. Backend committed source is b7a44809604822d979c1ed05c3c0ce0128a597c0 at results/ace_llm_intervention_remote_custody_20261109/traversal02/remote_custody.py; review is traversal02-independent-review/review.md; authority is traversal02-integration01/storage-authority.json. They are copied into the eight-member config as path/pin bindings, not imported, rewritten, or re-reviewed.

## Invocation and produced files

`invocation-template.json` lists every utility flag with unresolved placeholders. It is not an actual command or authorization. Parent fills the actual values after collection and audit; do not run it while the active collector owns the locks. Also provide explicit `--python` for the future builder and `--output-dir` for a new exclusive local directory whose canonical parent already exists.

After all files/pins/basic metadata joins pass, the utility creates that directory0700 and writes:

- `storage-config.json`: exact schema ace-compact-downstream-inputs-v1, predecessor cpu, eight members with backend commit/repository-path bindings.
- `command.json`: JSON argv array for `[python, -I, -B, unchanged_builder, ...116 required flags...]`, cwd, all56 metadata bindings, generated config path/hash and pinned builder-source hashes. Literal path/value strings stay array elements; no shell interpolation. executed/scientific_qualified/independent_acceptance_performed_here remain false.

It prints only output paths/pins and executed:false. It never runs the builder or writes to packet/collector/capsule locations. Existing output paths reject. If a write fails after exclusive creation, partial files and a best-effort assembly-failure.json are retained, without retry. Source records are referenced in place and reauthenticated by the closed builder later.

The utility preserves the unchanged bounds:32MiB per ordinary builder/storage metadata,128MiB seal,128KiB generated storage config. An oversized actual record rejects; this utility does not silently widen the reviewed consumer contract. The backend4161 receipt must be verified/full-local-copy:false and the explicit two capsule roots must match the seal.

## Preparation scope

Utility main was not invoked. No tests or closed reviews/checks were repeated. Only AST syntax and assembly-map consistency were checked during authoring; no actual CPU terminal02 or audit data was read. No source, packet, collector or capsule was changed, and no commit/build/launch/remote/lock action occurred. Known mappings were assembled from the existing pinned CPU launch map and the prior49-input handoff, not from private inputs. Parent owns any further review, actual invocation and packet integration.
