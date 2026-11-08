"""Only new static metadata resolver exercise; never execute captured source."""
from pathlib import Path
import ast
import json
import time
import build_packet as b

root=Path(__file__).resolve().parent
evidence=root/'metadata03'
start=time.monotonic();cpu=time.process_time()
entry=b.read(evidence/'captured'/b.ENTRY,b.ENTRY_SHA)
closure=b.decode(b.read(evidence/'closure-map.json','a40ab19df6e01ba52b38d5a7d567ac7bd2a5d9d027d9b472108c98427916588a'))
helpers=b.literal(entry,'HELPER_PINS')
b.need(helpers==closure['stages']['realize']['helper_pins'] and len(helpers)==8,
       'referenced-key HELPER_PINS must resolve exactly')
checks=['pinned entry resolves all eight named/literal helper keys against independent closure']
diagnostic_name=b.literal(entry,'DIAGNOSTIC')
diagnostic=b.read(evidence/'captured'/diagnostic_name,helpers[diagnostic_name])
for name in ('ARCHIVE_SHA','MANIFEST_SHA','RUNTIME_SOURCE','BOOTSTRAP'):
    value=b.literal(diagnostic,name)
    b.need(type(value) is str and bool(value),'diagnostic string constant required')
    if name.endswith('_SHA'):b.need(b.pin(value),'diagnostic pin invalid')
    if name=='BOOTSTRAP':compile(ast.parse(value),'<bootstrap metadata syntax only>','exec')
checks.append('pinned diagnostic archive/manifest/runtime/bootstrap metadata parsed without execution')
probe=b.read(evidence/'captured/probe.py',helpers['probe.py'])
probe_tree=ast.parse(probe)
b.need(any(isinstance(n,ast.FunctionDef) and n.name=='kill_owned' for n in probe_tree.body),'pinned cleanup probe missing')
checks.append('original CPU03 probe bytes authenticated and function presence parsed, never invoked')
example="K='key'\nV=[1, True, None, ('s',)]\nP={K:V, 'set':{1,2}}\n"
b.need(b.literal(example,'P')=={'key':[1,True,None,('s',)],'set':{1,2}},'nested literal resolution differs')
checks.append('nested containers and named references resolve')
for code in ("P=f()", "P=obj.field", "P={**other}", "A=B\nB=A", "P=1\nP=2", "P=MISSING",
             "P={'a':1,'a':2}"):
    try:b.literal(code,'A' if code.startswith('A=') else 'P')
    except ValueError:pass
    else:raise AssertionError('unsupported/ambiguous AST accepted: '+code)
checks.append('calls/attributes/expansion/cycles/duplicate definitions/missing references/duplicate keys rejected')
# Compile changed authoring code without executing its entrypoints.
pins={}
for name in ('build_packet.py','launch_once.py','monitor_once.py','check_metadata.py'):
    data=(root/name).read_bytes();compile(ast.parse(data),name,'exec');pins[name]=b.sha(data)
checks.append('current authoring scripts compile, no main function called')
b.write(evidence/'checks.json',b.raw(dict(checks=checks,source_pins=pins,
    entry_sha256=b.ENTRY_SHA,diagnostic_sha256=helpers[diagnostic_name],probe_sha256=helpers['probe.py'],
    closure_sha256='a40ab19df6e01ba52b38d5a7d567ac7bd2a5d9d027d9b472108c98427916588a',
    wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
    packet_built=False,namespace_claimed=False,scientific_source_executed=False,remote_actions=0)))
for check in checks:print('PASS '+check)
