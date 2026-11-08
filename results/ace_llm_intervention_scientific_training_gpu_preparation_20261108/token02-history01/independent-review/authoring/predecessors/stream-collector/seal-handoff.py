from pathlib import Path
import hashlib,json,os,datetime,shutil
root=Path('/tmp/ace-scientific-training-gpu-stream-current.txt').read_text().strip();root=Path(root)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path('/tmp/ace-scientific-training-gpu-collector-hxb4yf1k')
predecessor={}
for p in sorted(old.rglob('*')):
    assert not p.is_symlink()
    if p.is_file():
        rel=p.relative_to(old).as_posix();q=root/'predecessors/gpu-collector'/rel
        assert sha(p)==sha(q) and p.stat().st_ino!=q.stat().st_ino and q.stat().st_nlink==1
        predecessor[rel]=dict(sha256=sha(p),bytes=p.stat().st_size,original_nlink=p.stat().st_nlink,independent_snapshot=True)
(root/'predecessor-inventory.json').write_text(json.dumps(predecessor,indent=2,sort_keys=True)+'\n')
receipt=json.loads((root/'test01/receipt.json').read_text())
assert receipt['returncode']==0
pins={n:sha(root/n) for n in ('pull_terminal.py','test_gpu_stream_integration.py','stream_custody.py','stream_custody_review.md','pull_terminal.py.diff','test01/receipt.json')}
text=f'''# Scientific GPU training collector stream integration — review handoff

Stable authoring in `{root}`; independent review required before use. The original collector9346be8d, entire original authoring/test tree (including retained fixture bytes), its closed review, and accepted TOKEN integration source/test/handoff/review are copied under `predecessors`. `predecessor-inventory.json` compares every original byte and documents original fixture hardlink counts; snapshots are independent regular files. Original tree, GPU core, packet roster and production code were not changed.

Only transport changes: one bounded1800-second read-only helper stream replaces the bulky per-file remote SCP and second raw-tree local copy. The stream includes whole `launch` (GPU training), `cpu` (TOKEN), `common` (collection02), `failed01`, both separate remote namespace files and all empty directories. The accepted helper performs two independent exclusive materializations. Exact complete before/after file/directory maps, aliases, upstream pins, accounting and journal observations are compared before materialization. The original post-transfer authority/error/resource/GPU-evidence/44-job90-journal/accounting/cleanup/failure disposition code is byte-identical. Both controller locks, terminal-job gate, input/intent/result/source/commit joins and final two-copy capsule readback remain.

The complete independently pinned accepted TOKEN capsule is still separately authenticated and copied to both `upstream-cpu-custody` trees, including its captured archives, failures, receipts and upstream raw evidence. Those local disk copies and small coordinator copies remain; they do not require another remote transfer. No model/adapter tensor or scored outcome is decoded. Opaque checkpoints and all failed attempts stay in whole custody. Memory/CUDA/progress signals remain metadata, not utilization percentages or acceptance.

## Standalone interface additions

Keep all existing collector CLI arguments, including independently pinned full packet commit, collector/self/review, packet/job/result/local-intent and accepted CPU-capsule inventory. Add exactly two required arguments:

- `--stream-helper-repository-path`: repository-relative committed path of accepted helper bytes.
- `--stream-review-repository-path`: repository-relative committed path of accepted helper review bytes.

Place `stream_custody.py` and `stream_custody_review.md` adjacent to this `pull_terminal.py`. Their exact independent pins are constants: helper `{pins['stream_custody.py']}`, review `{pins['stream_custody_review.md']}`. Both adjacent files must also match bytes at the supplied repository paths in the same full `--packet-commit` via `git --no-replace-objects show`. The new collector and its new independent review must likewise be committed under the existing separate collector interface. This is outer-only binding: no core, child, scientific input roster, packet JSON, launcher or monitor change is needed. Parent supplies the actual reviewed repository paths/commit; no future acceptance is synthesized.

Helper loading occurs only after existing self/review/packet/Git/copy/CPU-capsule admission and the first conclusive terminal observation. Both terminal capsule directories are claimed exclusively; their `remote` children remain absent for helper materialization. Helper source/review, request, exact emitter argv, before/after observations, archive, producer stderr, capture and materialization/failure receipts are retained in both capsules. Transport errors retain original exceptions and available partial bytes, without retry, replacement or acceptance. A second-copy failure remains explicit; original evidence is never deleted. Conservative free-space checks include raw materializations, both full retained archives and coordinator/CPU-capsule bytes. The helper's unchanged member/byte/path bounds can reject an oversized real capsule; they are not silently enlarged.

## New checks and limitations

Six NEW fabricated integration methods passed once: complete four-root/two-claim two-copy transport and independent archives; CPU-root/journal/accounting drift rejection; partial capture preservation without retry; materialization failure with retained archive/partial checkpoint; standalone full-commit/helper/review binding negatives; and unchanged original gate/remote-observer/post-transfer authority checks. Capture and Git were mocked. The exact accepted emitter/materializer ran only on tiny hand-built local bytes. No former helper, TOKEN or GPU collector test method was rerun. No SSH, remote read, build, submission, tokenizer/model, actual scientific response, source freeze edit or commit occurred.

`test01` retains source snapshots, all fixture outputs and injected partial-failure evidence, stdout/stderr, runner and costs. All six methods passed (unittest0.165s); process wall `{receipt['wall_seconds']}` seconds, child CPU `{receipt['child_cpu_seconds']}` seconds. No failed test attempt occurred; deliberately injected transport failures remain in fixture custody. No network-throughput or actual runtime qualification is claimed. Actual accepted TOKEN inputs, complete44/90 terminal accounting and independently reviewed eventual invocation remain required.

Collector SHA256: `{pins['pull_terminal.py']}`

Test SHA256: `{pins['test_gpu_stream_integration.py']}`

Test receipt SHA256: `{pins['test01/receipt.json']}`

Diff: `pull_terminal.py.diff`. Machine handoff: `handoff.json`. Preserve this tree unchanged after review handoff unless a specific finding is returned.
'''
(root/'handoff.md').write_text(text)
handoff=dict(schema='ace-scientific-training-gpu-stream-authoring-handoff-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),directory=str(root),status='ready_for_independent_review_not_execution',pins=pins,predecessor_source_sha256=sha(old/'pull_terminal.py'),predecessor_inventory_sha256=sha(root/'predecessor-inventory.json'),handoff_sha256=sha(root/'handoff.md'),required_new_cli=['stream-helper-repository-path','stream-review-repository-path'],core_packet_roster_changed=False,child_sources_changed=False,new_tests=6,failed_test_attempts=0,child_cpu_seconds=receipt['child_cpu_seconds'],wall_seconds=receipt['wall_seconds'],old_test_repeats=0,actual_remote_actions=0,scientific_responses=0,models_loaded=0,acceptance=False)
(root/'handoff.json').write_text(json.dumps(handoff,sort_keys=True,indent=2)+'\n')
shutil.copyfile(__file__,root/'seal-handoff.py')
files={}
for p in sorted(root.rglob('*')):
    assert not p.is_symlink()
    if p.is_file():
        assert p.stat().st_nlink==1
        files[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
(root/'authoring-inventory.json').write_text(json.dumps(dict(files=files,excludes='authoring-inventory.json itself',all_regular_independent_files=True),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(directory=str(root),pins=pins,handoff_json_sha256=sha(root/'handoff.json'),authoring_inventory_sha256=sha(root/'authoring-inventory.json'),files=len(files)),indent=2))
