"""Package only NEW artificial evidence; preserve original fixtures and every attempt."""
from pathlib import Path
import hashlib,json,os,stat,tarfile
root=Path(__file__).resolve().parent
external=Path(json.loads((root/'locations.json').read_text())['external'])
fixtures=root/'attempt01/fixtures'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
entries={}
for path in [fixtures]+sorted(fixtures.rglob('*')):
 s=path.lstat();name='fixtures' if path==fixtures else 'fixtures/'+path.relative_to(fixtures).as_posix()
 item={'mode':stat.S_IMODE(s.st_mode),'uid':s.st_uid,'gid':s.st_gid,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
 if stat.S_ISLNK(s.st_mode):item.update(kind='symlink',target=os.readlink(path))
 elif stat.S_ISDIR(s.st_mode):item.update(kind='directory')
 elif stat.S_ISREG(s.st_mode):item.update(kind='file',bytes=s.st_size,sha256=digest(path))
 else:raise ValueError('unexpected fixture kind')
 entries[name]=item
inventory=root/'attempt01/fixture-inventory.json'
with inventory.open('x') as f:json.dump({'artificial_fixture_metadata_only':True,'original_root':str(fixtures),'entries':entries},f,sort_keys=True,indent=2);f.write('\n')
archive=root/'attempt01/fixtures.tar.gz'
with tarfile.open(archive,'x:gz',format=tarfile.PAX_FORMAT,dereference=False) as tf:tf.add(fixtures,arcname='fixtures',recursive=True)
with tarfile.open(archive,'r:gz') as tf:
 members=tf.getmembers();assert len(members)==len(entries) and set(m.name for m in members)==set(entries)
 for member in members:
  item=entries[member.name]
  assert member.mode==item['mode'] and member.uid==item['uid'] and member.gid==item['gid']
  if item['kind']=='symlink':assert member.issym() and member.linkname==item['target']
  elif item['kind']=='directory':assert member.isdir()
  else:
   assert member.isfile() or member.islnk()
   if member.islnk():assert member.linkname in entries
   h=hashlib.sha256();count=0
   with tf.extractfile(member) as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b);count+=len(b)
   assert count==item['bytes'] and h.hexdigest()==item['sha256']
# Mirror ordinary evidence. Intentional live fixture aliases are represented only
# by the verified lossless archive + exact original metadata in the external copy.
files={}
for path in sorted(root.rglob('*')):
 if path==fixtures or fixtures in path.parents:continue
 if path.is_dir():continue
 s=path.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 rel=path.relative_to(root).as_posix();target=external/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert rel in ('locations.json','predecessor527d/remote_custody.py') and digest(target)==digest(path)
 else:
  with path.open('rb') as src,target.open('xb') as dst:
   for b in iter(lambda:src.read(1024**2),b''):dst.write(b)
  assert digest(target)==digest(path)
 t=target.lstat();assert stat.S_ISREG(t.st_mode) and t.st_nlink==1 and (s.st_dev,s.st_ino)!=(t.st_dev,t.st_ino)
 files[rel]={'bytes':s.st_size,'sha256':digest(path)}
receipt={'schema':'ace-source-artificial-preparation-copy-v1','repository':str(root),'external':str(external),'files':files,'copied_regular_files':len(files),'copied_bytes':sum(v['bytes'] for v in files.values()),'fixture_archive':{'path':'attempt01/fixtures.tar.gz','sha256':digest(archive),'bytes':archive.stat().st_size,'members':len(entries)},'fixture_inventory_sha256':digest(inventory),'original_fixtures_preserved_at':str(fixtures),'live_aliases_only_in_original_artificial_fixtures':True,'independent_regular_copies_verified':True,'acceptance':False,'runtime_qualified':False,'remote_actions':0}
data=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
for parent in (root,external):
 with (parent/'copy-receipt.json').open('xb') as f:f.write(data)
print(json.dumps({'receipt_sha256':hashlib.sha256(data).hexdigest(),'copied_regular_files':len(files),'copied_bytes':receipt['copied_bytes'],'fixture_archive_members':len(entries),'external':str(external)},indent=2))
