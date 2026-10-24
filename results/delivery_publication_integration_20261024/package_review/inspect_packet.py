"""Read-only identity/notice/membership review; no artifact imports or science."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,re,stat,datetime
from pypdf import PdfReader
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
PACKET=BASE/'ACE-submission-review-20261024-01'
ROOT=BASE/'delivery-publication-candidate-20261024-01'
OLD=BASE/'delivery-anonymous-B-candidate-20261008-03'
SNAP=BASE/'delivery-publication-source-snapshot-20261024-01'
PLAN=Path('/Users/pat/ACE_Study_Results/delivery-final-release-plan-20261024-01')
OUT=Path('/Users/pat/code/ACE/results/delivery_publication_integration_20261024/package_review')
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):return sha(p.read_bytes())
def read(p):return json.loads(p.read_text())
checks=[]
def ck(name,value,detail=None):
 checks.append(dict(name=name,passed=bool(value),detail=detail));return bool(value)
patterns={
 'owner':r'Patrick\s+Cooper|paco0228',
 'private_location':r'/Users/(?:pat|[^/\s]+)/|/scratch/alpine/|/projects/paco0228|/home/paco0228',
 'allocation_account':r'ucb736_asc1',
 'owner_url':r'github\.com/(?:paco0228|patrickcooper|patrick-cooper)|(?:pat(?:rick)?|paco0228)@',
}
compiled={k:re.compile(v,re.I) for k,v in patterns.items()}
def hits(text):
 return [{'kind':k,'text':m.group(0)} for k,p in compiled.items() for m in p.finditer(text)]
manifest=read(ROOT/'manifest.json'); old=read(OLD/'manifest.json')
ck('final manifest independently pinned',pin(ROOT/'manifest.json')=='87668a0095caa3202ca1402f1218bac2f82da42769e1aba716f39c016d263d4b')
ck('old manifest independently pinned',pin(OLD/'manifest.json')=='b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9')
oldfiles={f['path']:f for f in old['files']}; newfiles={f['path']:f for f in manifest['files']}
ck('3940 final files and5117 bindings',len(newfiles)==3940 and len(manifest['bindings'])==5117)
ck('3893 inherited manifest entries unchanged',len(oldfiles)==3893 and all(newfiles.get(k)==v for k,v in oldfiles.items()))
ck('5068 inherited bindings prefix unchanged',len(old['bindings'])==5068 and manifest['bindings'][:5068]==old['bindings'])
added=sorted(set(newfiles)-set(oldfiles));ck('47 additions',len(added)==47)
new_hits=[]; new_pins=[]
for rel in added:
 p=ROOT/rel; data=p.read_bytes();new_pins.append(dict(path=rel,sha256=sha(data),bytes=len(data)))
 ck('addition bytes '+rel,sha(data)==newfiles[rel]['sha256'])
 if p.suffix!='.pdf':
  h=hits(data.decode('utf8','replace'))
  if h:new_hits.append(dict(path=rel,hits=h))
ck('new additions no screened owner/private location/account',not new_hits,new_hits)
contract=read(ROOT/'publication/source_contract.json')
ck('contract independently pinned',pin(ROOT/'publication/source_contract.json')=='3cba05022130fb30d661f056427e29a0421811b4f7475cc19813d2b5e83be60d')
ck('no affirmative public/anonymous/training/submission flags',all(contract[k] is False for k in ['anonymity_review_complete','anonymous_execution_qualified','public_release_approved','submission_approved','training_reproduction_available']))
ck('historical input manifest explicitly preserved',pin(ROOT/'publication/replay_input_manifest.json')==pin(OLD/'manifest.json'))
member=read(PACKET/'source_membership.json'); inv=read(SNAP/'inventory.json')
ck('42file snapshot inventory independently pinned',pin(SNAP/'inventory.json')=='d17693410e124c67286a2b55914b4a404dbd26b6a35d24e6c083752560478238' and len(inv['files'])==42)
ck('ZIP membership binds inventory',member['inventory_sha256']==pin(SNAP/'inventory.json'))
ziprows=[]; ziphits=[];zipmeta=[]
with zipfile.ZipFile(PACKET/'ACE_manuscript_sources.zip') as z:
 infos=z.infolist(); names=[i.filename for i in infos]
 expected={f['path'].removeprefix('paper/aistats_ace_2027/'):f for f in member['files']}
 ck('29zip members exact',len(infos)==29 and set(names)==set(expected) and len(set(names))==len(names))
 ck('ZIP no comments',not z.comment and all(not i.comment for i in infos))
 for i in infos:
  safe=not PurePosixPath(i.filename).is_absolute() and '..' not in PurePosixPath(i.filename).parts and not stat.S_ISLNK(i.external_attr>>16)
  data=z.read(i); row=expected[i.filename]; rel=row['path']; current=(ROOT/rel).read_bytes(); snapshot=(SNAP/'source'/rel).read_bytes()
  ck('zip/source/snapshot '+i.filename,safe and sha(data)==row['sha256']==sha(current)==sha(snapshot) and len(data)==row['bytes'])
  ziprows.append(dict(path=i.filename,sha256=sha(data),bytes=len(data))); h=hits(data.decode('utf8','replace'))
  if h:ziphits.append(dict(path=i.filename,hits=h))
  zipmeta.append(dict(path=i.filename,extra_hex=i.extra.hex(),comment_hex=i.comment.hex(),date_time=list(i.date_time)))
ck('ZIP no screened owner/private location/account',not ziphits,ziphits)
private_snapshot=[r['path'] for r in inv['files'] if contract['snapshot_members'][r['path']]['disposition']=='private-only']
ck('all private-only snapshot files absent from final paths',all(p not in newfiles for p in private_snapshot),private_snapshot)
zipnames=set(expected)
ck('no private proof basenames in ZIP',all(Path(p).name not in zipnames for p in private_snapshot))
pdf=PACKET/'ACE_manuscript.pdf';r=PdfReader(pdf)
ck('PDF pin and exactpackage match',pin(pdf)=='00f31c47177898be45ed1d0222db8f6b38760584aa9a304c4396bf65a3fbab15' and pdf.read_bytes()==(ROOT/'paper/aistats_ace_2027/paper.pdf').read_bytes())
ck('PDF25pages',len(r.pages)==25)
ck('PDF anonymousauthor',r.metadata.get('/Author')=='Anonymous authors')
ck('PDF no XMP or attachments',r.xmp_metadata is None and len(r.attachments)==0)
pdfhits=[];uris=[]
for i,p in enumerate(r.pages):
 h=hits(p.extract_text() or '')
 if h:pdfhits.append(dict(page=i+1,hits=h))
 for a in p.get('/Annots',[]):
  d=a.get_object(); action=d.get('/A')
  if action and action.get('/URI'):uris.append(dict(page=i+1,uri=str(action['/URI'])))
ck('PDF no screened owner/private location/account text',not pdfhits,pdfhits)
ck('PDF metadata no screened identifiers',not hits(json.dumps(dict(r.metadata))))
ck('PDF URI no author/private links',not any(hits(u['uri']) for u in uris),uris)
notice=[]
for p in sorted((OLD/'notices').iterdir()):
 if p.is_file():
  target=ROOT/'notices'/p.name;data=p.read_bytes();h=hits(data.decode('utf8','replace'))
  ck('required notice byte preservation '+p.name,target.read_bytes()==data)
  notice.append(dict(path='notices/'+p.name,sha256=sha(data),bytes=len(data),owner_hits=h))
ck('Runner MIT independent pin',pin(ROOT/'notices/ACE_RUNNER_MIT.txt')=='206c4b6a50d0291622cae04f1ec98c6bf64609b53836f727b0d470e00528491c')
prov=read(ROOT/'notices/provenance.json')
ck('owner permission resolved in retained notice',prov['runner_grant_authority_resolved'] is True)
projections=[]
for kind,record in contract['projections'].items():
 p=ROOT/record['path'];j=read(p)
 ck('projection '+kind,j['schema']=='delivery-publication-evidence-projection-v1' and set(j['rules'])==set(j['values']) and j['original_sha256']==record['original_sha256'] and pin(p)==record['sha256'])
 projections.append(dict(kind=kind,path=record['path'],sha256=pin(p),selectors=sorted(j['rules']),scope=j['scope']))
# Compare omitted selector paths only; never run report/export/checker modules.
omitted=read(PLAN/'private/derivation.json')['omitted_selectors']
notice_status=[k for k in contract if any(t in k for t in ['notice','redistribution','anonymity','remaining'])]
result=dict(schema='ace-final-package-privacy-review-witness-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='in-progress; final parent offline checker and notice-status correction pending',checks=checks,passed=sum(x['passed'] for x in checks),total=len(checks),packet_pins={p.name:pin(p) for p in PACKET.iterdir() if p.is_file()},final_root=str(ROOT),manifest_sha256=pin(ROOT/'manifest.json'),source_contract_sha256=pin(ROOT/'publication/source_contract.json'),snapshot_sha256=pin(SNAP/'inventory.json'),added_files=new_pins,source_zip_rows=ziprows,zip_metadata=zipmeta,pdf_metadata=dict(r.metadata),pdf_uris=uris,pdf_pages=len(r.pages),required_notices=notice,private_snapshot_members=private_snapshot,public_projections=projections,omitted_selector_kinds=list(omitted),historical_notice_remaining=prov['remaining'],current_contract_notice_status_keys=notice_status,required_findings=[{'id':'P1','description':'Current authoritative publication contract does not explicitly supersede historical notices/provenance.json remaining list or bind a current notice/anonymity handling disposition.'}],human_decision='Approve venue-appropriate handling of the attribution-bearing full reproducibility package separately from the anonymous manuscript; keep required notices unchanged.',scope_limits=['read-only identity/notice/projection/membership review','no scientific/numerical tests, exporter, model, generic verifier, or packagebuilder executed','no legal or anonymity certification; source hashes and public lineage may permit identification','initial fullpackage review; final parent checker receipt pending'],public_release_approved=False)
(OUT/'witness01.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[x for x in checks if not x['passed']],'pins':result['packet_pins'],'new_hits':new_hits,'zip_hits':ziphits,'notice_status_keys':notice_status,'omitted_selector_kinds':list(omitted)},indent=2))
