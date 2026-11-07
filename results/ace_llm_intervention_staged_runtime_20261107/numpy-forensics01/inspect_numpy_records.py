"""Read captured wheel metadata against the authenticated archive inventory.
No imports of copied packages, extraction, installs or environment modification.
Run only after complete terminal custody. Evidence, not runtime qualification.
"""
from pathlib import Path
import base64,csv,hashlib,io,json,posixpath,resource,tarfile,time
root=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-staged-runtime-terminal-20261107-01')
repo=Path('/Users/pat/code/ACE/results/ace_llm_intervention_staged_runtime_20261107')
out=root/'numpy-forensics01'
out.mkdir(exist_ok=False)
t=time.monotonic();cpu=time.process_time()
disposition=json.loads((root/'disposition.json').read_bytes())
raw=(root/'remote/qualification/packed/manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==disposition['files']['qualification/packed/manifest.json']['sha256']
manifest=json.loads(raw); index={r['path']:r for r in manifest['records']}
archive=root/'remote/qualification/packed/runtime.tar'
assert manifest['archive_sha256']==disposition['files']['qualification/packed/runtime.tar']['sha256']
assert archive.stat().st_size==manifest['archive_bytes']
base='lib/python3.10/site-packages'
names={base+'/numpy-'+v+'.dist-info/'+f for v in ('2.2.5','2.2.6') for f in ('METADATA','RECORD')}
names|={base+'/numpy/version.py'}
selected={}
with tarfile.open(archive,'r:') as tar:
 for entry in tar:
  if entry.name not in names: continue
  assert entry.isfile() and entry.size < 2*1024**2
  b=tar.extractfile(entry).read();r=index[entry.name]
  assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  selected[entry.name]=b
  target=out/entry.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(b)
assert set(selected)==names
records={}
for version in ('2.2.5','2.2.6'):
 data=selected[base+'/numpy-'+version+'.dist-info/RECORD'].decode()
 matches=[];mismatches=[];unhashed=[]
 for name,h,n in csv.reader(io.StringIO(data)):
  path=posixpath.normpath(posixpath.join(base,name))
  assert not path.startswith('../') and not path.startswith('/')
  r=index.get(path)
  if not h:unhashed.append({'path':path,'present':r is not None});continue
  algorithm,value=h.split('=',1)
  assert algorithm=='sha256'
  expected=base64.urlsafe_b64decode(value+'='*((4-len(value)%4)%4)).hex()
  result={'path':path,'expected_sha256':expected,'expected_bytes':int(n),
          'captured_sha256':r.get('sha256') if r else None,'captured_bytes':r.get('bytes') if r else None}
  (matches if r and r.get('sha256')==expected and r.get('bytes')==int(n) else mismatches).append(result)
 records[version]={'matched_hashed_records':len(matches),'mismatches':mismatches,'unhashed':unhashed,
                   'matches':matches,'metadata_version_lines':[line for line in selected[base+'/numpy-'+version+'.dist-info/METADATA'].decode().splitlines() if line.startswith(('Name:','Version:'))]}
v={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':manifest['archive_sha256'],
   'records':records,'scope':'metadata and RECORD byte comparisons only; no imported version or execution qualification',
   'new_package_imports':0,'environment_changes':0,'elapsed_seconds':time.monotonic()-t,
   'process_cpu_seconds':time.process_time()-cpu,'peak_self_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   'cost_scope':'this local forensic process only, not whole preparation or cycle'}
with (out/'forensics.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
import shutil
shutil.copyfile(__file__,out/'inspect_numpy_records.py')
shutil.copytree(out,repo/'numpy-forensics01')
print(json.dumps({k:x for k,x in v.items() if k!='records'},indent=2))
for version,x in records.items():print(version,'matches',x['matched_hashed_records'],'mismatches',len(x['mismatches']),'unhashed',len(x['unhashed']))
