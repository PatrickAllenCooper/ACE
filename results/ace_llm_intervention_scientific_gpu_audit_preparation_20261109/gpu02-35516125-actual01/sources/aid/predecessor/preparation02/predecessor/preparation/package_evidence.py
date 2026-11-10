from pathlib import Path
import hashlib,json,os,stat,tarfile
root=Path(__file__).resolve().parent;external=Path(json.loads((root/'locations.json').read_text())['external'])
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
fixture_roots=[root/'attempt01/fixtures',root/'delta02/fixtures'];entries={}
for folder in fixture_roots:
 for p in [folder]+sorted(folder.rglob('*')):
  s=p.lstat();n=p.relative_to(root).as_posix();v=dict(mode=stat.S_IMODE(s.st_mode),inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
  if stat.S_ISDIR(s.st_mode):v['kind']='directory'
  else:
   assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
   v.update(kind='file',bytes=s.st_size,sha256=hashfile(p))
  entries[n]=v
(root/'fixture-inventory.json').write_text(json.dumps(dict(artificial_only=True,original_roots=list(map(str,fixture_roots)),entries=entries),indent=2,sort_keys=True)+'\n')
archive=root/'fixtures.tar.gz'
with tarfile.open(archive,'x:gz',format=tarfile.PAX_FORMAT,dereference=False) as tf:
 for p in fixture_roots:tf.add(p,arcname=p.relative_to(root).as_posix())
with tarfile.open(archive,'r:gz') as tf:
 members=tf.getmembers();assert len(members)==len(entries) and set(m.name for m in members)==set(entries)
 for m in members:
  v=entries[m.name]
  if v['kind']=='directory':assert m.isdir();continue
  assert m.isfile();h=hashlib.sha256();size=0
  with tf.extractfile(m) as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
  assert (size,h.hexdigest())==(v['bytes'],v['sha256'])
files={}
for p in sorted(root.rglob('*')):
 if any(p==f or f in p.parents for f in fixture_roots) or p.is_dir():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 n=p.relative_to(root).as_posix();q=external/n;q.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as a,q.open('xb') as b:
  for block in iter(lambda:a.read(1024**2),b''):b.write(block)
 assert hashfile(p)==hashfile(q) and not os.path.samestat(p.stat(),q.stat()) and q.stat().st_nlink==1
 files[n]=dict(bytes=s.st_size,sha256=hashfile(p))
receipt=dict(schema='ace-source-artificial-preparation-copy-v1',temporary=str(root),external=str(external),files=files,file_count=len(files),bytes=sum(v['bytes'] for v in files.values()),
 fixture_members=len(entries),fixture_archive_sha256=hashfile(archive),original_fixtures_preserved=True,lossless_fixture_archive_verified=True,
 independent_regular_copies_verified=True,acceptance=False,runtime_qualified=False,remote_actions=0)
data=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
for folder in (root,external):
 with (folder/'copy-receipt.json').open('xb') as f:f.write(data)
print(json.dumps(dict(receipt_sha256=hashlib.sha256(data).hexdigest(),file_count=len(files),bytes=receipt['bytes'],fixture_members=len(entries),external=str(external)),indent=2))
