# Missing stage-parent correction

New preserved source directory: implementation/. Previous D1/D2 source and all initial test failures remain at /tmp/ace-downstream-compact-d1d2-1d_57cvu and captured predecessor/ here. This corrects only Leibniz's follow-up missing-stage-parent regression. No actual builder/remote/scientific execution.

Both builders reject aliased repository stage parents, create the fixed stage parent (one directory, existing parent allowed), and confirm it is an ordinary canonical directory before claiming either preparation leaf. Preparation leaves still require exist_ok=False. The already-existing preparation guard remains before this setup. No external preparation is claimed when a repository parent is a symlink or a file. GPU collector's exact builder SHA was propagated. Shared adapter, both launchers, finalizer collector, all protected scientific/resource/history gates and monitors are unchanged.

One NEW fabricated method passed once, covering both builders across missing parent, existing preparation leaf, symlink parent and file parent, plus once-only rejection after successful leaf creation. It executes only the captured setup AST on artificial local paths; no builder main, project code or transport. Receipt retains source snapshots/stdout/stderr/argv:0.057852childCPU/0.064106583elapsed/24,641,536RSSbytes. No earlier test repeated; no failed test in this delta.

## Exact changed-source pins

- `finalizer/build_packet.py`: `ada28e3ec5bf1591bd5c00a419f7b01e47ec7d7fc08abf9d6722645aef07081c`
- `gpu/build_packet.py`: `e41b787fe0ede1191408b0941e7f9d806b9d4d38fcd0cf5718aebc2bfa16f8e1`
- `gpu/pull_terminal.py`: `0f149f47f0ed10865a4dacc1abb52bc81f79b68f1c527e490424ee7b06f17a6e`
- `test_missing_stage_parent.py`: `d530048ec685866b69ee45dde44c98fefe1d5604cbc73344d9c885fadfe62d3f`

All final-source pins and the test receipt pin are in handoff.json. Narrow recheck requested; no acceptance manufactured.
