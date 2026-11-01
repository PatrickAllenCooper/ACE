from pathlib import Path
import hashlib,json,zipfile,tomllib,subprocess,datetime
r=Path('/Users/pat/code/ACE');b=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline');p=b/'ACE-submission-review-20261024-02';out=r/'results/delivery_submission_audit_20261101';out.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while q:=f.read(1024*1024):h.update(q)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
record=read(p/'artifact_record.json');checked=[]
for n,e in record['artifacts'].items():
 q=p/n;checked.append({'path':str(q),'expected_sha256':e['sha256'],'actual_sha256':sha(q),'bytes':q.stat().st_size,'matches':q.stat().st_size==e['bytes']and sha(q)==e['sha256']})
cap=read(r/'results/delivery_publication_integration_20261024/writing_capture/inventory.json');current=[]
for e in cap['files']:
 q=r/e['path'];current.append({'path':e['path'],'sha256':sha(q),'expected_sha256':e['sha256'],'matches':sha(q)==e['sha256']})
fr=read(r/'results/delivery_publication_release_20261024/successor02/final_review_receipt.json');proofs=[]
for n,e in fr['evidence'].items():
 q=Path(e['path']);proofs.append({'name':n,'path':str(q),'matches':q.exists()and sha(q)==e['sha256'],'sha256':sha(q)if q.exists()else None})
archives=[]
for n in ('ACE_manuscript_sources.zip','ACE_reproduction_with_attribution.zip'):
 with zipfile.ZipFile(p/n)as z:
  bytop={}
  for e in z.infolist():
   k=e.filename.split('/')[0];v=bytop.setdefault(k,{'members':0,'compressed_bytes':0,'uncompressed_bytes':0});v['members']+=1;v['compressed_bytes']+=e.compress_size;v['uncompressed_bytes']+=e.file_size
  archives.append({'name':n,'bytes':(p/n).stat().st_size,'members':len(z.infolist()),'within_100000000_bytes':(p/n).stat().st_size<=100000000,'within_100_MiB':(p/n).stat().st_size<=100*1024*1024,'compressed_breakdown':bytop})
automations=[]
for q in Path('/Users/pat/.codex/automations').glob('ace-*/automation.toml'):
 d=tomllib.loads(q.read_text());automations.append({'id':d['id'],'status':d['status'],'rrule':d['rrule']})
v={'audit_date':'2026-11-01','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'artifact_checks':checked,'writing_capture_checks':current,'final_receipt_checks':proofs,'archive_metadata':archives,'automations':automations,'scope':'Fresh read-only digests, membership metadata, source/receipt coherence and local automation state. No model/statistical execution, scientific rerun, compiler invocation or fresh Slurm query. Prior independent reviews are preserved evidence, not peer-review acceptance.'}
(out/'current_state.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'all_artifact_checks':all(x['matches']for x in checked),'artifact_count':len(checked),'all_writing_capture_checks':all(x['matches']for x in current),'writing_files':len(current),'all_receipt_checks':all(x['matches']for x in proofs),'receipt_count':len(proofs),'archives':archives,'automations':automations},indent=2))
