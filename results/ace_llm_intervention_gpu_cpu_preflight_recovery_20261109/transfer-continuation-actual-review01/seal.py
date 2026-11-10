import pathlib,json,hashlib,tarfile,io,shutil,stat,datetime
W=pathlib.Path(__file__).parent;R=pathlib.Path('/Users/pat/code/ACE');B=pathlib.Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline');P=R/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109';G=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108';O=G/'launch02/transfer-continuation01';E=B/'ACE-LLM-intervention-scientific-campaign-gpu-20261109-02/transfer-continuation01'
D=P/'transfer-continuation-actual-review01';X=B/'ACE-LLM-GPU02-transfer-continuation-actual-review-20261110-01'
assert not D.exists() and not X.exists()
def raw(v):return (json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()
selected={}
for p in O.iterdir():
 assert p.is_file();selected['operation/'+p.name]=p
for p in (G/'transfer-continuation-command01').iterdir():selected['invocation/'+p.name]=p
for n in ['submission.intent.json','submission.result.json','transfer-root.json','launcher_failure.json','controller_released.json','readiness.json']:
 selected['launch02/'+n]=G/'launch02'/n
for folder in ['transport-failure01','transport-quiescence02']:
 for p in (G/folder).iterdir():
  if p.is_file():selected[folder+'/'+p.name]=p
s=E/'storage-operations'
for p in s.iterdir():
 if p.is_file():selected['external-storage/'+p.name]=p
for directory in ['verify-000','verify-001','fetch-log-028','fetch-log-091']:
 for n in ['capacity.json','command.json','intent.json','transport.json','receipt.json']:
  selected['external-storage/operations/'+directory+'/'+n]=s/'operations'/directory/n
archive=W/'saved-operational-metadata.tar.gz';memberpins={}
with archive.open('xb') as a,tarfile.open(fileobj=a,mode='w:gz',format=tarfile.PAX_FORMAT,dereference=False) as t:
 for name,p in sorted(selected.items()):
  st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1
  b=p.read_bytes();memberpins[name]=dict(bytes=len(b),sha256=sha(b),original_path=str(p));i=tarfile.TarInfo(name);i.size=len(b);i.mode=0o600;i.mtime=0;t.addfile(i,io.BytesIO(b))
observed={}
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  assert m.isfile() and not m.issym() and not m.islnk() and m.name not in observed and not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts
  b=t.extractfile(m).read();observed[m.name]=dict(bytes=len(b),sha256=sha(b))
assert observed=={k:{j:v[j] for j in ['bytes','sha256']} for k,v in memberpins.items()}
(W/'archive-members.json').write_bytes(raw(memberpins))
receipt=dict(scope='independent actual saved operational transfer-continuation lineage review',required_findings=0,scientific_acceptance=False,job_id='35516125',review_sha256=sha((W/'review.md').read_bytes()),source_sha256='a6996da4b7b4998fa0f576097a52cd72e25312d05697fc45ca83e25fccfab8c5',driver_commit='58df1d1b05d979e1f80e95c86cca0be530382174',packet_commit='463e5b1140e8aa957e07636c603c3057da81e7d6',failure_proof_commit='b476bc40bda7abb789416b37df6910a2196334aa',submission_result_sha256='ab4719a46cb68901829bc5617d1321a94021ffca7be89046554c69e9518f51fc',remote_intent_sha256='dd7148afb1fcad65834c717b39c1d4728860277e9e0ef75278c65378e6de0d1b',metadata_checks=190,failed_final_checks=0,originals_preserved=True,remote_actions=False,project_execution=False,tests_rerun=False,archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),members=len(memberpins),all_members_readback_verified=True),repo_path=str(D),external_path=str(X),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(W/'receipt.json').write_bytes(raw(receipt));D.mkdir();X.mkdir()
for p in W.iterdir():
 if p.is_file():
  for dest in [D,X]:
   with (dest/p.name).open('xb') as out:out.write(p.read_bytes())
inventory={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in D.iterdir() if p.is_file()}
for dest in [D,X]:(dest/'inventory.json').write_bytes(raw(inventory))
assert {p.name for p in D.iterdir()}=={p.name for p in X.iterdir()}
for p in D.iterdir():assert p.read_bytes()==(X/p.name).read_bytes() and p.stat().st_ino!=(X/p.name).stat().st_ino and p.stat().st_nlink==(X/p.name).stat().st_nlink==1
print(json.dumps(dict(repo=str(D),external=str(X),review_sha256=receipt['review_sha256'],receipt_sha256=sha((D/'receipt.json').read_bytes()),inventory_sha256=sha((D/'inventory.json').read_bytes()),files=len(inventory)+1,archive=receipt['archive']),indent=2))
