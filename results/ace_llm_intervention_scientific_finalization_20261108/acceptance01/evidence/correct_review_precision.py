from pathlib import Path
import json,hashlib
O=Path(__file__).parent;D=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01');E=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-finalization-acceptance-20261110-01');S=O/'publication'
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def h(b):return hashlib.sha256(b).hexdigest()
def update(name,b):
 for root in (S,D,E):(root/name).write_bytes(b)
old={n:(S/n).read_bytes() for n in ('review.md','receipt.json','handoff.json','inventory.json')}
for root in (S,D,E):
 p=root/'review-predecessor01';p.mkdir()
 for n,b in old.items():(p/n).write_bytes(b)
text=old['review.md'].decode();rt=json.loads((O/'runtime-facts.json').read_bytes())
replacements={'83.551155':'83.551267','294.353026':'294.353463','179.052579':'179.053342'}
for before,after in replacements.items():assert before in text;text=text.replace(before,after)
update('review.md',text.encode());review=h(text.encode())
r=json.loads(old['receipt.json']);r['review_sha256']=review;r['review_revision_note']='Three prose timing decimal transcriptions corrected from the already preserved runtime-facts.json. All original publication bytes retained under review-predecessor01; no evidence/check/acceptance/baseline change.'
update('receipt.json',raw(r));receipt=h(raw(r));handoff=json.loads(old['handoff.json']);handoff['review_sha256']=review;handoff['independent_review_receipt_sha256']=receipt;update('handoff.json',raw(handoff))
correction=dict(scope='review prose only',required_findings=0,changes=replacements,authority='already inspected phase_timings in evidence/runtime-facts.json',acceptance_unchanged=True,baseline_unchanged=True,original_review_sha256=h(old['review.md']),current_review_sha256=review)
update('review-precision-correction.json',raw(correction));update('evidence/correct_review_precision.py',Path(__file__).read_bytes())
files={p.relative_to(S).as_posix():{'bytes':p.stat().st_size,'sha256':h(p.read_bytes())} for p in sorted(S.rglob('*')) if p.is_file() and p.relative_to(S).as_posix()!='inventory.json'}
inv=raw(dict(files=files,excludes=['inventory.json'],repository_root=str(D),external_root=str(E)));update('inventory.json',inv)
for n,v in files.items():
 assert h((D/n).read_bytes())==h((E/n).read_bytes())==v['sha256'];a=(D/n).stat();b=(E/n).stat();assert (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino)
result=dict(status='final_publication_verified',required_findings=0,acceptance_sha256=h((S/'acceptance.json').read_bytes()),baseline_sha256=h((S/'baseline.json').read_bytes()),review_sha256=review,receipt_sha256=receipt,handoff_sha256=h(raw(handoff)),inventory_sha256=h(inv),members_sha256=h((S/'report-members.json').read_bytes()),files=len(files)+1,bytes=sum(p.stat().st_size for p in S.rglob('*') if p.is_file()),repository=str(D),external=str(E))
(O/'publication-result-final.json').write_bytes(raw(result));print(json.dumps(result,sort_keys=True))
