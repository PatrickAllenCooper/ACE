from pathlib import Path
import ast,hashlib,json,time
root=Path(__file__).resolve().parent
start=time.monotonic();cpu=time.process_time();p=root/'pull_terminal.py';data=p.read_bytes()
tree=ast.parse(data,filename=str(p));compile(tree,str(p),'exec');embedded=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
  if any(isinstance(t,ast.Name) and t.id=='REMOTE_OBSERVATION' for t in node.targets):
   compile(node.value.value,'REMOTE_OBSERVATION','exec');embedded.append(hashlib.sha256(node.value.value.encode()).hexdigest())
print(json.dumps(dict(scope='AST parse/compile only; collector not imported or executed',collector_sha256=hashlib.sha256(data).hexdigest(),embedded_sha256=embedded,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,remote_actions=0,tests_run=0,output_claims=0),indent=2))
