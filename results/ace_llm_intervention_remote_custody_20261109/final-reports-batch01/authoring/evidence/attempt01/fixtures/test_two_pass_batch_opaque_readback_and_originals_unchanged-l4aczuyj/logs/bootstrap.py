import sys,hashlib,json,base64
def need(ok,message):
 if not ok:raise ValueError(message)
line=sys.stdin.buffer.readline(32)
need(line.endswith(b'\n') and line[:-1].isdigit(),'bounded source bundle framing required')
n=int(line);need(0<n<=1048576,'source bundle size bound')
source=sys.stdin.buffer.read(n)
need(len(source)==n and hashlib.sha256(source).hexdigest()==sys.argv[1],'independent source bundle SHA differs')
def pairs(items):
 out={}
 for k,v in items:
  need(k not in out,'duplicate source bundle key');out[k]=v
 return out
bundle=json.loads(source,object_pairs_hook=pairs)
need(set(bundle)=={'extension','extension_sha256','backend','backend_sha256'},'source bundle keys differ')
extension=base64.b64decode(bundle['extension'],validate=True)
backend=base64.b64decode(bundle['backend'],validate=True)
for raw,pin in ((extension,bundle['extension_sha256']),(backend,bundle['backend_sha256'])):
 need(0<len(raw)<=262144 and hashlib.sha256(raw).hexdigest()==pin,'individual source SHA differs before exec')
need(bundle['backend_sha256']=='527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b','original backend SHA differs')
v={'__name__':'captured_batch_reports','__file__':'<captured-batch-reports>'}
exec(compile(extension,v['__file__'],'exec'),v)
v['bind_sources'](extension_source=extension,extension_sha256=bundle['extension_sha256'],
 backend_source=backend,backend_sha256=bundle['backend_sha256'])
b=v['_backend']();data=sys.stdin.buffer.read(b.MAX_METADATA+1)
need(len(data)<=b.MAX_METADATA,'bounded request metadata required')
v['_specialize'](b.remote_main,remote_emit=v['_remote_emit'])(b.decode(data),sys.stdout.buffer)
