# Downstream compact custody — final D3 path correction recheck

**0 required findings. D3 is resolved.** This closes the sole remaining required finding from the D1/D2 recheck; D1/D2 retain their prior resolved disposition. The original reviews, drafts and failed fixture evidence remain unchanged.

GPU build_packet.py:629–632 and finalizer build_packet.py:875–878 now reject an aliased stage parent, create the one fixed repository stage directory if absent, and verify that it is an ordinary canonical directory before either preparation leaf is claimed. The pre-existing preparation guard still runs first; each preparation leaf still uses exclusive mkdir(exist_ok=False). A symlink/file parent rejects before the external preparation is created. The repository results parent already exists, so no recursive directory creation is needed. This supports the actual first-build layout without relaxing once-only claims or adding transport.

The only production differences from the reviewed D1/D2 sources are these identical four-line parent guards/setup in both builders and the GPU collector's propagated builder pin. Independent byte comparison confirms every other production file is unchanged, including the shared adapter, both launchers, finalizer collector, monitors and CPU history contract. Source/resource/scientific-child interfaces and authority02a7 remain closed and unchanged. No new authority fields are required.

| Changed source | SHA256 |
| --- | --- |
| finalizer/build_packet.py | `ada28e3ec5bf1591bd5c00a419f7b01e47ec7d7fc08abf9d6722645aef07081c` |
| gpu/build_packet.py | `e41b787fe0ede1191408b0941e7f9d806b9d4d38fcd0cf5718aebc2bfa16f8e1` |
| gpu/pull_terminal.py | `0f149f47f0ed10865a4dacc1abb52bc81f79b68f1c527e490424ee7b06f17a6e` |

Author source root: /tmp/ace-downstream-compact-parent-fix-j795b8og/implementation. Exact current source snapshots, full pin map and actual diffs are preserved beside this review.

The single new fabricated method, test_missing_stage_parent.py SHA d530048ec685866b69ee45dde44c98fefe1d5604cbc73344d9c885fadfe62d3f, exercises only the extracted setup AST for both builders: absent parent, existing preparation leaf, symlink parent, file parent, and retry rejection after initial preparation creation. Retained receipt58adf5fb539aff9f05d9128918f8acf76c0884cd3f5d5884c398cc3fb8d30659 records one pass, return0, 0.057852 childCPU / 0.064106583 elapsed / 24,641,536 RSS bytes. Its captured source pins match all current handoff pins. The reviewer read the fixture/transcript/receipt and did not run it. No old tests or backend/component review was repeated.

This is source readiness for the reviewed storage-only outer amendments, not a concrete packet, backup, scientific-result or submission acceptance. All later independent input pins, fresh remote readback, full custody references, exact actual accounting, packet review/Git binding and once-only gates remain necessary. No target callable, build, remote operation, project/model/tokenizer/scientific work or commit was executed. Review overhead is unmetered. There are no optional suggestions in this narrow pass.

Previous focused review: /var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T/ace-downstream-d1d2-independent-recheck-r5nry3ti/review.md; SHA256 `9d96be6d39786a30c8fdf99318b43dab7c542c72a6cc9444c5baf9211da7fdc1`.
