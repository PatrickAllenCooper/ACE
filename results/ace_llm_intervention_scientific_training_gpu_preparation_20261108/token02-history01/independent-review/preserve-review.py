from pathlib import Path
import ast,hashlib,json,shutil,tempfile
src=Path('/private/tmp/ace-scientific-training-gpu-token02-mcu8h32f')
out=Path(tempfile.mkdtemp(prefix='ace-training-gpu-token02-independent-review-'))
sha=lambda b:hashlib.sha256(b).hexdigest()
def inv(root):
 result={}
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('unexpected symlink '+str(p))
  if p.is_file():
   b=p.read_bytes();result[str(p.relative_to(root))]={'bytes':len(b),'sha256':sha(b)}
 return result
before=inv(src);shutil.copytree(src,out/'authoring');assert inv(src)==inv(out/'authoring')==before
handoff=json.loads((src/'handoff.json').read_bytes())
assert all(before[n]['sha256']==h for n,h in handoff['pins'].items())
read=lambda n:(src/n).read_text()
def funcs(n):
 s=read(n);return {x.name:(ast.dump(x,include_attributes=False),ast.get_source_segment(s,x)) for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)}
def assignments(n):
 return {t.id:ast.dump(x.value,include_attributes=False) for x in ast.parse(read(n)).body if isinstance(x,ast.Assign) for t in x.targets if isinstance(t,ast.Name)}
comparisons={}
for new,old in [('build_packet.py','predecessors/core/build_packet.py'),('launch_once.py','predecessors/core/launch_once.py'),('collector/pull_terminal.py','predecessors/stream-collector/pull_terminal.py')]:
 a,b=funcs(new),funcs(old)
 comparisons[new]={'added':sorted(a.keys()-b.keys()),'removed':sorted(b.keys()-a.keys()),'changed':sorted(k for k in a.keys()&b.keys() if a[k][0]!=b[k][0])}
a,b=funcs('build_packet.py'),funcs('predecessors/core/build_packet.py')
assert a['make_packet'][1]==b['make_packet'][1]
a,b=assignments('build_packet.py'),assignments('predecessors/core/build_packet.py')
unchanged=['ROOT','SOURCE_COMMIT','ENTRY','ENTRY_SHA','WORKER','WORKER_SHA','RESOURCES','DEADLINES','ROSTER']
assert all(a[k]==b[k] for k in unchanged)
assert assignments('launch_once.py')['REMOTE_COMMON']==assignments('predecessors/core/launch_once.py')['REMOTE_COMMON']
assert read('monitor_once.py')==read('predecessors/core/monitor_once.py')
a,b=funcs('collector/pull_terminal.py'),funcs('predecessors/stream-collector/pull_terminal.py')
for k in ('resources','public_gpu_evidence','stream_transfer','load_stream_helper','authenticate_stream_files','validate_remote_intent'):
 assert a[k][1]==b[k][1],k
receipt={'scope':'independent static TOKEN02-history and fifth-root collector integration review only','source_root':str(src),'snapshot_root':str(out/'authoring'),'file_inventory':before,'function_delta':comparisons,'unchanged_training_assignments':unchanged,'make_packet_source_identical':True,'launcher_remote_common_identical':True,'monitor_identical':True,'collector_resource_gpu_evidence_stream_mechanism_unchanged':True,'required_findings':0,'reviewed_code_imported_or_executed':False,'tests_run_by_reviewer':0,'remote_actions':0,'builds':0,'scientific_execution':False,'future_token02_acceptance_created':False}
(out/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
shutil.copyfile('/tmp/preserve-training-token02-review.py',out/'preserve-review.py')
print(json.dumps({'directory':str(out),'receipt_sha256':sha((out/'receipt.json').read_bytes()),'files':len(before),'bytes':sum(v['bytes'] for v in before.values()),'source_pins':handoff['pins']},indent=2))
