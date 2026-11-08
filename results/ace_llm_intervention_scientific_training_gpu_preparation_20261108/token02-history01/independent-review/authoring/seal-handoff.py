from pathlib import Path
import ast,json,hashlib,shutil,difflib,datetime
r=Path(Path('/tmp/ace-scientific-training-gpu-token02-current.txt').read_text().strip())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prod=Path('/tmp/ace-scientific-token02-authoring-8tbddys1')
for n in ('input-contract.md','operational-amendment-fields.json'):
 shutil.copyfile(prod/n,r/'predecessors'/('token02-'+n))
for name,oldname in [('build_packet.py','predecessors/core/build_packet.py'),('launch_once.py','predecessors/core/launch_once.py'),('monitor_once.py','predecessors/core/monitor_once.py'),('collector/pull_terminal.py','predecessors/stream-collector/pull_terminal.py')]:
 new=r/name;old=r/oldname
 (r/(name.replace('/','-')+'.diff')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new))))
# Parse-only checks, including remote strings; no target entrypoint executes.
static={}
for name in ['build_packet.py','launch_once.py','monitor_once.py','collector/pull_terminal.py']:
 text=(r/name).read_text();tree=ast.parse(text);compile(tree,str(r/name),'exec');embedded=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
   if any(isinstance(t,ast.Name) and t.id in ('REMOTE_COMMON','REMOTE_OBSERVATION') for t in node.targets):
    compile(node.value.value,'<embedded-read-only-source>','exec');embedded.extend(t.id for t in node.targets)
 static[name]=dict(sha256=sha(r/name),parsed=True,embedded_parsed=embedded,executed=False)
(r/'static.json').write_text(json.dumps(static,indent=2,sort_keys=True)+'\n')
receipts={n:json.loads((r/n/'receipt.json').read_bytes()) for n in ['history-test01','collector-test01','collector-test02']}
assert [v['returncode'] for v in receipts.values()]==[0,1,0]
pins={n:sha(r/n) for n in ['build_packet.py','launch_once.py','monitor_once.py','collector/pull_terminal.py','collector/stream_custody.py','collector/stream_custody_review.md','test_token02_history.py','test_collector_token02.py','history-test01/receipt.json','collector-test01/receipt.json','collector-test02/receipt.json']}
(r/'interface.md').write_text('''# GPU TRAIN01 consumer of TOKEN02 recovery history

Core remains build_packet.py / launch_once.py / monitor_once.py. No build/submission occurred. TRAIN01root, frozen sourcea934, freeze156, entry55be and workerab2c,2700s4CPU15360MiB1MIG/progress180/startup240 remain unchanged. Keep all previous CLI pairs. `--previous-baseline` STILL means the original accepted collection02 42/86 accounting; `--baseline` NOW means future independently accepted TOKEN02 44/90 accounting. The failed TOKEN01 43/88 record is separate `--token-failure-baseline`.

Add required path + independentSHA pairs for:
failed-token-packet, failed-token-stage, failed-token-result;
token-failure-acceptance, token-no-learnedwork-proof, token-failure-baseline, token-failure-disposition, token-failure-inventory, token-failure-readback, token-failure-custody-inventory;
token-operational-amendment, token-operational-review, token-operational-adoption-review, token-operational-guide, token-diagnosis;
collection-failure-disposition (already closed original collection01 disposition fd3a52be…).

These match Godel's shared TOKEN02 input-contract.md / operational-amendment-fields.json. The three TOKEN review/guide members are Markdown; other new members JSON. GPU consumer aliases Godel's42/86 `baseline` to its existing `previous-baseline`. GPU source captures all metadata as outer files; child source/data roster and acceptance schema stay unchanged. The failed-stage bytepin is original101d88; successful TOKEN02 stage is independently supplied `cpu-packet`, not synthesized. Its only permitted semantic differences from TOKEN01 are deadlines.startup120→300 and model_directory moved to TOKEN02/tokenizer. Original work550/cleanup570/hard590,600s3CPU11520MiB0GPU remain. Caller must supply actual accepted TOKEN02 prepared/plan/numerical/exposure/job/custody/review pins; none default to success or cost.

TOKEN02 producer carries `token_recovery` as a nested object, not imagined flat fields. GPU outer and original submission intent carry that exact object plus failed_token_root/failed_token_job_id. The launcher reauthenticates the core metadata, preserved failure history and original raw TOKEN01 input pins before its unchanged twice-fresh both-lock reconciliation and one sbatch. All historical rows/journals and actual allocation charges stay present; accepted collection42/86→failed TOKEN43/88→accepted TOKEN44/90→GPU45/92. Original freeze's TOKEN01root/startup remain historical; explicit token-only authority supplies successorroot/startup. No scientific freeze rewrite.

Standalone collector remains outside core/child packet roster. Keep existing CLI source/self/review/full packet-commit, helper/review-repository-path and cpu-terminal-custody/cpu-custody-inventory-sha256 arguments. `cpu-terminal-custody` now means the complete independently accepted TOKEN02 terminal capsule. Add required `--failed-token-terminal-custody PATH`, pointing to the complete original independently accepted TOKEN01 terminal capsule. Its custody-inventory pin comes from independently captured `token-failure-custody-inventory.json`; no invented second acceptance schema. Collector source/review/helper/review must be in the eventual full packet commit as before.

Collector streams five whole roots (`launch` GPU, `cpu` TOKEN02, `failed-token01` TOKEN01, `common` collection02, `failed01` collection01) plus both namespace claims. It discovers TOKEN02's own failed-root label from the uniquely matching independently pinned TOKEN01 outerpacket, then requires the complete original file and directory maps; it does not guess Godel's collector label. Thus the future accepted TOKEN02 inventory must contain that whole failed root under one distinct top-level label. Both complete TOKEN02 and original TOKEN01 local capsules are authenticated and copied independently to `upstream-cpu-custody` and `upstream-token01-custody`, retaining archives/transports/failures as opaque bytes. Two remote observations, one1800sec stream, two exclusive raw materializations and final two-copy readbacks remain. No adapter tensors or scored outcomes decoded; no new response/fit or scientific acceptance.

Actual failure/operational acceptance fields remain independent inputs and must conform to the agreed Godel interface. Unknown/nonzero learnedwork or responses rejects this narrow recovery. Node-local runtime cleanup is not upgraded. Existing metadata/helper size bounds remain fail-closed and must fit the actual supplied custody. Concrete packet/collector invocation review is still required. No future cost or admission is manufactured.
''')
(r/'handoff.md').write_text(f'''# GPU training TOKEN02-history delta — ready for independent review

Complete authoring tree: `{r}`. Core and separate collector both implemented. Predecessor unexecuted43/88-success coreee9c/6971/3fc1 and stream collector1178743d, all original snapshots/attempts and the initial in-progress draft are retained under predecessors/ and draft01/. The predecessor trees were never edited. Godel's shared producer interface snapshots and exact field expressions are retained; see interface.md for required CLI mappings.

Core: exact collection41/84 failure→accepted42/86→TOKEN01 accepted failure43/88→future TOKEN02 accepted44/90; TRAIN01root/frozen scientific code and methods unchanged. First training submission would extend to45/92. Independent full failure inventories, two custody copy maps, proof, amendment/reviews, original packet/input/journals, complete accepted TOKEN02 and all actual costs must agree. The unchanged captured child and make_packet function are retained. Monitor is byte-identical. Launcher adds only explicit TOKEN recovery-history joins/intent fields; its remote resource/source/model/readiness/lock/one-sbatch mechanism is byte-identical.

Collector adds failed TOKEN01 as a fifth stream root plus complete independent original failure capsule, retaining full accepted TOKEN02 capsule and original collection failures. Historical charge/row/journal tests now use45/92. GPU usage interpretation and actual allocation/cleanup/reporting gates are unchanged; no utility acceptance is inferred. It needs one new path CLI `--failed-token-terminal-custody`; helperfab01/review505842 remain independently bound outside the child/packet roster. Collector needs its own independent review before use.

Four NEW core metadata tests passed once (history-test01). Three NEW collector tests: initialcollector-test01 had two passes and one fixture error (missing parent_accounting in fake observation), fully retained; only that affected method reran in collector-test02 and passed. Production collector did not change for the fixture fix. All seven distinct new methods now pass. No old test rerun, real packet build, scientific descriptor/response/model/tokenizer execution, remote operation, allocation or commit. Tests used fabricated public metadata, mocked reads and local opaque stream bytes. Static remote-source compilation did not execute remote strings.

New attempts combined child CPU{sum(v['child_cpu_seconds'] for v in receipts.values()):.6f}s and wall{sum(v['wall_seconds'] for v in receipts.values()):.6f}s; historical attempt costs remain in preserved original receipts and are not silently combined with these. All raw outputs, source pins, failure fixture and runner are retained. `handoff.json` gives final pins; adjacent diffs bound the scope. No actual future acceptance/baseline/cost exists in this authoring tree.

Final core pins: builder `{pins['build_packet.py']}`; launcher `{pins['launch_once.py']}`; unchanged monitor `{pins['monitor_once.py']}`.

Collector `{pins['collector/pull_terminal.py']}`.

Tests: core `{pins['test_token02_history.py']}`; collector `{pins['test_collector_token02.py']}`.

Ready for separate independent core/collector review; this is authoring, not runtime/source/scheduler qualification or permission to submit. Await actual independently accepted TOKEN02. Concrete TOKEN02 packet review is a separate forthcoming parent handoff, not performed here.
''')
shutil.move(r/'NOT_READY.md',r/'draft01/NOT_READY-initial.md')
handoff=dict(schema='ace-gpu-training-token02-history-authoring-v1',status='ready_for_independent_review',directory=str(r),pins=pins,interface_sha256=sha(r/'interface.md'),handoff_sha256=sha(r/'handoff.md'),static_sha256=sha(r/'static.json'),source_commit='a9346022f145cc29439325b6f291597b2e6ec037',freeze_unchanged=True,training_child_unchanged=True,training_root_unchanged=True,new_methods=7,new_attempt_returncodes=[0,1,0],new_attempt_child_cpu_seconds=sum(v['child_cpu_seconds'] for v in receipts.values()),new_attempt_wall_seconds=sum(v['wall_seconds'] for v in receipts.values()),old_methods_repeated=0,remote_actions=0,scientific_execution=False,build=False,submitted=False,acceptance=False)
(r/'handoff.json').write_text(json.dumps(handoff,sort_keys=True,indent=2)+'\n')
shutil.copyfile(__file__,r/'seal-handoff.py')
files={}
for p in sorted(r.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():
  assert p.stat().st_nlink==1
  files[p.relative_to(r).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
(r/'authoring-inventory.json').write_text(json.dumps(dict(files=files,excludes='authoring-inventory.json itself',independent_regular_files=True),sort_keys=True,indent=2)+'\n')
# Shared file coordination; does not mutate Godel's tree.
with Path('/tmp/ace-scientific-training-gpu-token02-interface.md').open('a') as f:
 f.write('\nFinal consumer core/collector handoff: '+str(r/'handoff.md')+'; exact CLI mappings '+str(r/'interface.md')+'. Matched producer nested token_recovery and original baseline42/86 metadata name. Collector infers producer failed-root label from exact original packet pin and requires complete bytes/directories; no fixed producer label imposed. All7 NEW methods passed across3 retained attempts (one fixture error corrected alone). No build/remote/science.\n')
print(json.dumps(dict(directory=str(r),pins=pins,handoff_json_sha256=sha(r/'handoff.json'),inventory_sha256=sha(r/'authoring-inventory.json'),file_count=len(files)),indent=2))
