import sys,json,base64,os,stat,hashlib
value=json.load(sys.stdin);root=value['root'];fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);info=os.fstat(fd)
if info.st_ino!=value['inode'] or info.st_uid!=os.geteuid() or info.st_mode&0o077:raise ValueError('output identity differs')
if set(os.listdir(fd))!={'output_claim.json'}:raise ValueError('unexpected existing output; reconcile')
for name,item in value['files'].items():
 if '/' in name or name in ('.','..'):raise ValueError('basename required')
 raw=base64.b64decode(item['data'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('transport payload changed')
 if name=='output_claim.json':
  h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  with os.fdopen(h,'rb') as f:
   if f.read()!=raw:raise ValueError('claim differs')
 else:
  h=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
  with os.fdopen(h,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
actual={}
for name in value['files']:
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 with os.fdopen(h,'rb') as f:actual[name]=hashlib.sha256(f.read()).hexdigest()
if actual!={n:v['sha256'] for n,v in value['files'].items()}:raise ValueError('remote readback differs')
if set(os.listdir(fd))!=set(actual):raise ValueError('remote membership differs')
print(json.dumps({'pins':actual,'inode':info.st_ino,'files':len(actual)}))
