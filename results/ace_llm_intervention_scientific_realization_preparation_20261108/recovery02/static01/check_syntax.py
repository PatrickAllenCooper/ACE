from pathlib import Path
import ast, hashlib, json, time
root=Path(__file__).resolve().parent.parent
started=time.monotonic();cpu=time.process_time();records=[]
for name in ('build_packet.py','launch_once.py','monitor_once.py','pull_terminal.py'):
 data=(root/name).read_bytes();tree=ast.parse(data,filename=name);compile(tree,name,'exec')
 embedded=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Constant) and isinstance(node.value,str):
   text=node.value.lstrip()
   if text.startswith(('from pathlib import Path\n','import pathlib,','import pathlib,hashlib','import pathlib,json','need(root.parent','need(not os.path.lexists(root/')):
    compile(node.value,name+':embedded','exec');embedded.append(hashlib.sha256(node.value.encode()).hexdigest())
 records.append(dict(file=name,sha256=hashlib.sha256(data).hexdigest(),syntax='pass',embedded_script_sha256=embedded))
print(json.dumps(dict(scope='AST parse and compile only; no project or target-script execution',source_files=records,tests_run=0,remote_actions=0,packet_builds=0,namespace_claims=0,scientific_draws=0,wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu),indent=2))
