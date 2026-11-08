"""Authoring check only: compile code objects without executing them."""
from pathlib import Path
import ast
import hashlib
import json
import time

root=Path(__file__).resolve().parent
start=time.monotonic();cpu=time.process_time();rows=[]
for name in ('build_packet.py','launch_once.py','monitor_once.py'):
    p=root/name;value=p.read_bytes();tree=ast.parse(value,filename=name);compile(tree,name,'exec')
    fragments=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            s=node.value
            prefixes=('\nimport pathlib','\nneed(','\nfor n,h','\nprint(json.dumps(verify')
            if '\n' in s and s.startswith(prefixes):
                compile(s,'<embedded remote fragment>','exec')
                fragments.append(hashlib.sha256(s.encode()).hexdigest())
    rows.append(dict(path=name,sha256=hashlib.sha256(value).hexdigest(),bytes=len(value),
                     compile='passed_without_execution',remote_fragment_pins=fragments))
review=dict(scope='new script syntax only; no import/build/draw/stage/model/tokenizer/tests/remote/namespace claim/submission',
    files=rows,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
    source_execution=False,tests_run=0)
with (root/'static02-passed/run.json').open('x') as f:
    json.dump(review,f,sort_keys=True,indent=2);f.write('\n')
for row in rows:
    print(f"{row['path']}: compile-only passed; {len(row['remote_fragment_pins'])} embedded remote fragment(s) compiled")
