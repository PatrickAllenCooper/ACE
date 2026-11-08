"""Exclusive byte custody only; never execute project source or tests."""
from pathlib import Path
from datetime import datetime,timezone
import os,stat,json,hashlib,tarfile,gzip,time,resource,io

REPO=Path('/Users/pat/code/ACE')
DEST=REPO/'results/ace_llm_intervention_scientific_training_supervisor_preparation_20261108/preparation01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-training-supervisor-preparation-20261108-01')
ROOTS={
 'original':Path('/var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T/ace-scientific-gpu-supervisor-w80gxpfu'),
 'initial-custody-copy':Path('/tmp/ace-scientific-gpu-supervisor-mczuvbj7'),
 'deltas':Path('/tmp/ace-scientific-gpu-schema-delta-7zphho50'),
 'intermediate-static-review':Path('/tmp/ace-scientific-training-gpu-review-55be5ba7'),
}
SINGLES={
 'reviews/initial-review.md':Path('/tmp/ace-scientific-training-gpu-initial-review-694a80b0.md'),
 'authoring/initial-source-generator.py':Path('/tmp/build-scientific-gpu-supervisor.py'),
 'authoring/reused-fragments.txt':Path('/tmp/ace-scientific-gpu-reused-fragments.txt'),
}
SOURCE_PINS={
 'supervise_llm_intervention_scientific_training_gpu.py':'55be5ba74fa16f3e05d5632a1d0ff4ac1b0a3187507849e82d6594810f3545fd',
 'test_supervise_llm_intervention_scientific_training_gpu.py':'040f8a8ce758517ed6db62898e765ee6c833b61f84cd1b587dfee961d1875ee1',
}
REVIEW=REPO/'docs/development/guidance/reviews/ace_llm_intervention_scientific_training_gpu_supervisor_review_2026-11-08.md'
START=time.monotonic();CPU=time.process_time()

def sha(b):return hashlib.sha256(b).hexdigest()
def js(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(p):
 before=p.lstat()
 assert stat.S_ISREG(before.st_mode) and before.st_nlink==1,('nonregular/link alias',str(p))
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  opened=os.fstat(f.fileno());assert (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino)
  b=f.read();after=os.fstat(f.fileno())
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 assert len(b)==before.st_size
 return b

def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert read(p)==b

def scan():
 result={}
 for label,root in ROOTS.items():
  assert root.is_dir() and not root.is_symlink()
  for p in sorted(root.rglob('*')):
   s=p.lstat()
   assert not stat.S_ISLNK(s.st_mode),('symlink',str(p))
   if stat.S_ISDIR(s.st_mode):continue
   assert stat.S_ISREG(s.st_mode) and s.st_nlink==1,('alias/special',str(p))
   result[label+'/'+p.relative_to(root).as_posix()]=p
 result.update(SINGLES)
 return result

assert EXT.parent.is_dir() and not EXT.parent.is_symlink()
assert not os.path.lexists(DEST) and not os.path.lexists(EXT),'exclusive destination already exists'
files=scan();entries={};root_counts={}
for name,p in files.items():
 b=read(p);s=p.lstat();entries[name]={'bytes':len(b),'sha256':sha(b),'original_path':str(p),'original_mtime_ns':s.st_mtime_ns,'type':'file'}
 label=name.split('/')[0];row=root_counts.setdefault(label,{'files':0,'bytes':0});row['files']+=1;row['bytes']+=len(b)
for n,pin in SOURCE_PINS.items():assert sha(read(REPO/'scripts/research'/n))==pin
review=read(REVIEW)
# Duplicate original/custody attempts are separate logical retained members, not
# a second execution or a filesystem hard-link alias.
for p in (ROOTS['original']/'attempt-01').rglob('*'):
 if p.is_file():assert read(p)==read(ROOTS['initial-custody-copy']/'attempt-01'/p.relative_to(ROOTS['original']/'attempt-01'))

DEST.mkdir(parents=True,exist_ok=False);EXT.mkdir(exist_ok=False)
write(DEST/'archive-members.json',js({'schema':'ace-source-preparation-archive-members-v1','files':entries,'count':len(entries),
 'regular_bytes':sum(v['bytes'] for v in entries.values()),'root_counts':root_counts,'symlinks':[],'hardlink_aliases':[],'special_files':[]}))
with (DEST/'attempts.tar.gz').open('xb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:
  with tarfile.open(fileobj=z,mode='w|',format=tarfile.PAX_FORMAT) as tar:
   for name,p in sorted(files.items()):
    b=read(p);assert sha(b)==entries[name]['sha256']
    item=tarfile.TarInfo(name);item.size=len(b);item.mode=0o600;item.mtime=0;item.type=tarfile.REGTYPE
    tar.addfile(item,io.BytesIO(b))
 f.flush();os.fsync(f.fileno())

def verify_archive(path):
 seen=set()
 with tarfile.open(path,'r:gz') as tar:
  for item in tar:
   assert item.isfile() and not item.issym() and not item.islnk(),('archive alias',item.name)
   assert item.name in entries and item.name not in seen;seen.add(item.name)
   assert not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
   b=tar.extractfile(item).read();expected=entries[item.name]
   assert len(b)==item.size==expected['bytes'] and sha(b)==expected['sha256']
   assert b==read(files[item.name]),('source/archive byte disagreement',item.name)
 assert seen==set(entries)
verify_archive(DEST/'attempts.tar.gz')
for n in SOURCE_PINS:write(DEST/'source'/n,read(REPO/'scripts/research'/n))
write(DEST/'review.md',review)
write(DEST/'packaging.py',Path(__file__).read_bytes())
# Copy every pre-receipt capsule file with fresh exclusive inodes and compare bytes.
for p in sorted(DEST.rglob('*')):
 if p.is_file():write(EXT/p.relative_to(DEST),read(p))
verify_archive(EXT/'attempts.tar.gz')
assert set(scan())==set(files)
for name,p in files.items():assert sha(read(p))==entries[name]['sha256']

history=[]
for label,p,methods in (
 ('original-ten',ROOTS['original']/'attempt-01/run.json',10),
 ('delta-02-failed',ROOTS['deltas']/'delta-02/run.json',1),
 ('delta-03-passed',ROOTS['deltas']/'delta-03/run.json',1),
 ('cleanup-04-passed',ROOTS['deltas']/'cleanup-04/run.json',1)):
 b=read(p);r=json.loads(b)
 history.append({'attempt':label,'original_run_path':str(p),'run_sha256':sha(b),'returncode':r['returncode'],
  'test_method_invocations':methods,'child_cpu_seconds':r['child_cpu_seconds'],'wall_seconds':r['wall_seconds'],
  'source_sha256':r['source_sha256'],'command':r['command'],'cwd':r['cwd']})
assert [r['returncode'] for r in history]==[0,1,0,0]
package_files={str(p.relative_to(DEST)):{'bytes':p.stat().st_size,'sha256':sha(read(p))} for p in sorted(DEST.rglob('*')) if p.is_file()}
receipt={'schema':'ace-scientific-training-supervisor-preparation-v1','utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Artificial/source-only implementation, fabricated boundary/lifecycle tests and independent static review custody. Not execution or scientific acceptance.',
 'original_roots':{k:str(v) for k,v in ROOTS.items()},'archive_members':len(entries),'regular_bytes':sum(v['bytes'] for v in entries.values()),
 'archive_exact_members_verified':True,'archive_original_bytes_compared':True,'both_archive_copies_reread_and_compared_to_originals':True,
 'symlink_members':0,'hardlink_members':0,'hardlink_files':0,'special_files':0,'fixture_alias_exceptions':[],
 'original_ten_test_copies_identical':True,'duplicate_original_custody_copy_is_not_another_test_run':True,
 'source_pins':SOURCE_PINS,'review_sha256':sha(review),'required_findings_remaining':0,
 'resolved_required_findings':['prospective CPU acceptance numerical/exposure pin fields','owned-child watchdog retained through uncertain cleanup'],
 'historical_test_attempts':history,'historical_child_cpu_seconds_total':sum(v['child_cpu_seconds'] for v in history),
 'historical_wall_seconds_total':sum(v['wall_seconds'] for v in history),'unique_test_methods':12,'test_method_invocations':13,
 'test_attempts':4,'new_test_runs_during_packaging':0,'prior_tests_repeated_during_packaging':0,
 'cost_scope':'Historical command costs counted once per original run, including failed schema assertion; excludes implementation/static review. Custody copy is not another run.',
 'packaging_process_cpu_seconds_through_verified_archives':time.process_time()-CPU,
 'packaging_wall_seconds_through_verified_archives':time.monotonic()-START,
 'packaging_measurement_excludes':'Final receipt publication and final capsule comparison; no allocation cost claimed.',
 'new_allocations':0,'gpu_slice_seconds':0,'new_environment_responses':0,'scientific_worlds_generated':0,
 'scientific_execution':False,'actual_runtime_stages':0,'model_or_tokenizer_execution':False,'remote_actions':0,
 'source_only_review_complete':True,'runtime_or_scheduler_qualified':False,'scientific_qualified':False,
 'repository_destination':str(DEST),'external_destination':str(EXT),'files':package_files}
write(DEST/'preparation.json',js(receipt));write(EXT/'preparation.json',read(DEST/'preparation.json'))

def capsule(root):
 result={};inodes=set()
 for p in sorted(root.rglob('*')):
  s=p.lstat();assert not stat.S_ISLNK(s.st_mode)
  if stat.S_ISDIR(s.st_mode):continue
  assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and (s.st_dev,s.st_ino) not in inodes
  inodes.add((s.st_dev,s.st_ino));b=read(p);result[str(p.relative_to(root))]=(len(b),sha(b))
 return result,inodes
left,li=capsule(DEST);right,ri=capsule(EXT)
assert left==right and not li.intersection(ri)
for name in left:assert read(DEST/name)==read(EXT/name)
print(json.dumps({'repository':str(DEST),'external':str(EXT),'capsule_files':len(left),'capsule_bytes':sum(v[0] for v in left.values()),
 'archive_members':len(entries),'original_member_bytes':sum(v['bytes'] for v in entries.values()),
 'preparation_sha256':sha(read(DEST/'preparation.json')),'archive_sha256':sha(read(DEST/'attempts.tar.gz')),
 'archive_members_sha256':sha(read(DEST/'archive-members.json')),'review_sha256':sha(review),
 'historical_child_cpu_seconds_total':receipt['historical_child_cpu_seconds_total'],
 'historical_wall_seconds_total':receipt['historical_wall_seconds_total'],'copies_byte_equal':True,'alias_free':True},indent=2))
