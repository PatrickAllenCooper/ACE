"""One explicit pre-draw operational successor. No project imports or remote work."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,shutil,subprocess
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
OLD=REPO/'results/ace_llm_intervention_scientific_contract_20261108/candidate01'
FAIL=REPO/'results/ace_llm_intervention_scientific_realization_20261108/failure-acceptance01'
FROZEN='57a3e6dd27ded5cec007be235b799a2bec711fc2c5630ff16fef87d74273cd28'
FAIL_PIN='5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d'
ACCOUNT_PIN='63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'
CLAIM_PIN='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
CHANGED={'scripts/research/'+n for n in ('supervise_llm_intervention_scientific_cpu.py','collect_llm_intervention_scientific_training_cpu.py','prepare_llm_intervention_scientific_campaign_cpu.py','qualify_llm_intervention_scientific_campaign_gpu.py')}
GUIDE='docs/development/guidance/ace_llm_intervention_scientific_mount_recovery_2026-11-08.md'
REVIEW='docs/development/guidance/reviews/ace_llm_intervention_scientific_mount_alias_review_2026-11-08.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(c,m):
 if not c:raise ValueError(m)
def pinned(p,h):
 b=p.read_bytes();need(sha(b)==h,'pin differs '+str(p));return b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-commit',required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args()
 need(len(a.source_commit)==40 and all(c in '0123456789abcdef' for c in a.source_commit),'full source commit')
 def committed(p):
  b=subprocess.check_output(['git','--no-replace-objects','show',a.source_commit+':'+p],cwd=REPO)
  need(b==(REPO/p).read_bytes(),'current/committed mismatch '+p);return b
 oldraw=pinned(OLD/'scientific-freeze.json',FROZEN);old=json.loads(oldraw)
 failure_raw=pinned(FAIL/'acceptance.json',FAIL_PIN);failure=json.loads(failure_raw)
 account_raw=pinned(FAIL/'accounting.json',ACCOUNT_PIN);account=json.loads(account_raw)
 claim_path=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
 claim_raw=pinned(claim_path,CLAIM_PIN)
 need(failure['status']=='accepted_failed_before_draw' and failure['required_findings']==0
  and failure['complete_custody_verified'] is True and failure['scientific_worlds_generated']==0
  and failure['environment_responses']==0 and failure['realizer_called'] is False
  and failure['freeze_sha256']==FROZEN and failure['namespace_claim_sha256']==CLAIM_PIN
  and failure['job_id']=='35402725','independent exact pre-draw failure required')
 need(account['failure_acceptance_sha256']==FAIL_PIN and account['terminal_jobs']==39
  and account['submission_journals']==80 and account['closed_cpu_seconds']==11048
  and account['closed_gpu_slice_seconds']==1601 and account['outstanding_reservation']==0,
  'exact closed failure baseline required')
 source={p:committed(p) for p in old['source_pins']};pins={p:sha(b) for p,b in source.items()}
 need(len(source)==67 and {p for p in pins if pins[p]!=old['source_pins'][p]}==CHANGED,'only declared four infrastructure changes')
 proposal=json.loads(pinned(OLD/'proposal.json',old['original_proposal']['sha256']))
 need(all(pins[p]==h for p,h in proposal['source_pins'].items()),'original protocol changed')
 replacements={old['source_pins'][p]:pins[p] for p in CHANGED}
 def replace(v):
  if isinstance(v,str):return replacements.get(v,v)
  if isinstance(v,list):return [replace(x) for x in v]
  if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
  return v
 closure=copy.deepcopy(old['stage_closure'])
 for k in ('source_pins','stages','helper_metadata'):closure[k]=replace(closure[k])
 guide=committed(GUIDE);review=committed(REVIEW);need(sha(review)==a.review_sha256,'independent review pin differs')
 recovery=dict(schema='ace-scientific-pre-draw-operational-amendment-v1',predecessor_freeze_sha256=FROZEN,
  failure_acceptance_sha256=FAIL_PIN,failed_job_id='35402725',original_namespace_claim_sha256=CLAIM_PIN,
  original_local_registry=str(claim_path),original_remote_registry='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json',
  source_changes={p:dict(before=old['source_pins'][p],after=pins[p]) for p in sorted(CHANGED)},
  correction_guide=dict(path=GUIDE,sha256=sha(guide)),correction_review=dict(path=REVIEW,sha256=sha(review)),
  scientific_fields_unchanged=True,draws_before_amendment=0,responses_before_amendment=0,
  preserve_original_claim=True,exclusive_recovery_claim_required=True,automatic_retry_allowed=False)
 closure['operational_amendment']=recovery
 closure.update(observed_head=a.source_commit,git_binding_final=True,
  metadata_predecessor=dict(path='candidate01/closure-map.json',sha256=sha((OLD/'closure-map.json').read_bytes()),status='preserved original complete map'),
  source_delta=dict(changes=recovery['source_changes'],git_binding_status='all67files captured from source_commit',unchanged_entries_inherited_without_component_rereview=True),
  scope='Committed67-source operational successor. Prior AST derivations and unchanged role maps retained. Exactly four infrastructure source pins updated. No scientific execution.',
  status='committed_operational_successor_source_map',
  resource_contract_status='Unchanged seven stage caps. Closed11048CPU/1601GPU plus3600prep yields55568CPU/7181GPU total projection. No new reservation.',
  unresolved_authorization='Candidate only. Independent candidate adoption, recovery claim and actual lock/journal/site admission remain required.')
 successor=copy.deepcopy(old)
 successor.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=a.source_commit,
  source_pins=pins,stage_closure=closure,operational_recovery=recovery)
 successor['component_reviews'][REVIEW]=sha(review)
 for stage in successor['stages']:
  stage['entry_sha256']=pins['scripts/research/'+stage['entry']]
  if stage['stage']=='realize':
   need(stage['remote_root'].endswith('_01') and stage['exclusive_external_root'].endswith('-01'),'exact predecessor root')
   stage['remote_root']=stage['remote_root'][:-2]+'02'
   stage['exclusive_external_root']=stage['exclusive_external_root'][:-2]+'02'
 successor['admission_accounting'].update(baseline_sha256=ACCOUNT_PIN,closed_cpu_seconds=11048,max_projected_cpu_seconds=55568)
 need(11048+3600+sum(x['cpus']*x['wall_seconds'] for x in successor['stages'])==55568<=57600,'CPU projection')
 need(1601+sum(x['gpus']*x['wall_seconds'] for x in successor['stages'])==7181<=7200,'GPU projection')
 # This comparison seals every other top-level scientific field against the old freeze.
 allowed={'utc','source_commit','source_pins','stage_closure','component_reviews','stages','admission_accounting','operational_recovery'}
 need(all(successor[k]==v for k,v in old.items() if k not in allowed),'undeclared scientific delta')
 for prior,current in zip(old['stages'],successor['stages']):
  allowed_stage={'entry_sha256'}|({'remote_root','exclusive_external_root'} if prior['stage']=='realize' else set())
  need(all(current[k]==v for k,v in prior.items() if k not in allowed_stage),'resource/stage delta')
 ext=BASE/'ACE-LLM-intervention-scientific-contract-20261108-candidate02'
 local=REPO/'results/ace_llm_intervention_scientific_contract_20261108/candidate02'
 need(not ext.exists() and not local.exists(),'exclusive successor candidate required');ext.mkdir(mode=0o700)
 def save(n,b):
  p=ext/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
 for p,b in source.items():save('source/'+p,b)
 for p,h in old['component_reviews'].items():save('reviews/'+Path(p).name,pinned(OLD/'reviews'/Path(p).name,h))
 save('reviews/'+Path(REVIEW).name,review)
 payload={'scientific-freeze.json':raw(successor),'predecessor-freeze.json':oldraw,
  'failure-acceptance.json':failure_raw,'accounting.json':account_raw,'original-namespace-claim.json':claim_raw,
  'closure-map.json':raw(closure),'review-pins.json':raw(successor['component_reviews']),
  'proposal.json':(OLD/'proposal.json').read_bytes(),'design.md':pinned(OLD/'design.md',old['design']['sha256']),
  'operational-amendment.md':guide,'build_recovery_contract.py':Path(__file__).read_bytes()}
 for n,b in payload.items():save(n,b)
 files={p.relative_to(ext).as_posix():sha(p.read_bytes()) for p in sorted(ext.rglob('*')) if p.is_file()}
 save('inventory.json',raw(dict(files=files,source_commit=a.source_commit,scope='unadopted operational successor before first draw',scientific_execution=False)))
 shutil.copytree(ext,local)
 need({p.relative_to(local).as_posix():sha(p.read_bytes()) for p in local.rglob('*') if p.is_file()}==
  {p.relative_to(ext).as_posix():sha(p.read_bytes()) for p in ext.rglob('*') if p.is_file()},'candidate mirrors differ')
 print(json.dumps(dict(path=str(local),freeze_sha256=sha(raw(successor)),files=len(files)+1,source_files=67,adopted=False)))
if __name__=='__main__':main()
