"""Narrow actual02 status/record delta; does not execute package code or scan old payload."""
from pathlib import Path
import json,hashlib,copy,datetime
B=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
A=B/'delivery-publication-candidate-20261024-01';Z=B/'delivery-publication-candidate-20261024-02'
P=B/'ACE-submission-review-20261024-02';J=B/'delivery-final-release-launch-20261024-02'
R=Path('/Users/pat/code/ACE');O=R/'results/delivery_publication_integration_20261024/package_review'
PINS={'manifest':'dc034b705391d4a26eade8e2b863136bcae4a749d829d765c5922bd00a092e00','contract':'0ffe94be8bc1dae770346bac6160d5aad134476bfa6fb6e57deafe6723423596','checker':'f5c753caf846606018365b2011b209b9d3f369df1f89011a203d18cbc08b4be0','archive':'eeeca48f7621d9196a843cd5fb3b66e6397040d4c1b3bd85ac90ef9e590f9e9a','README':'d1708f43aaf7ffa9e6b1f347ed1d94433e24bcff1431be228edb1740bd1419d3'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def rd(p):return json.loads(p.read_text())
checks=[]
def ck(name,v):checks.append({'name':name,'passed':bool(v)})
m0=rd(A/'manifest.json');m=rd(Z/'manifest.json');c0=rd(A/'publication/source_contract.json');c=rd(Z/'publication/source_contract.json');n=c['notice_status'];fs0={x['path']:x for x in m0['files']};fs={x['path']:x for x in m['files']}
ck('actual manifest trusted pin',sha(Z/'manifest.json')==PINS['manifest'])
ck('actual contract trusted pin',sha(Z/'publication/source_contract.json')==PINS['contract'])
ck('actual checker trusted pin',sha(Z/'verify_delivery_publication_release.py')==PINS['checker'])
ck('candidate01 original manifest preserved',sha(A/'manifest.json')=='87668a0095caa3202ca1402f1218bac2f82da42769e1aba716f39c016d263d4b')
ck('3940 unchanged payload names',len(fs)==3940 and set(fs)==set(fs0))
changed=sorted(k for k in fs if fs[k]!=fs0[k]);ck('exactly contract and checker entries changed',changed==['publication/source_contract.json','verify_delivery_publication_release.py'])
ck('5121 bindings and all5117 prior bindings preserved',len(m['bindings'])==5121 and m['bindings'][:5117]==m0['bindings'])
expected=[{'record':'publication/source_contract.json','pointer':p,'artifact':a,'digest':'sha256'} for p,a in [('/notice_status/historical_notice/sha256','notices/provenance.json'),('/notice_status/retained_attribution/0/sha256','notices/ACE_RUNNER_MIT.txt'),('/notice_status/retained_attribution/1/sha256','notices/RUNNER_LICENSE_SCOPE.md'),('/notice_status/original_B/sha256','B/acceptance.json')]]
ck('four new explicit pointer edges',m['bindings'][5117:]==expected)
normalized=copy.deepcopy(c);del normalized['notice_status']
normalized['artifacts']['verify_delivery_publication_release.py']=c0['artifacts']['verify_delivery_publication_release.py']
for k in ['planner','request']:normalized['private_evidence'][k]=c0['private_evidence'][k]
ck('contract only notice-status and necessary provenance delta',normalized==c0)
ck('historical remaining selector superseded only',n['historical_notice']['remaining_selector']=='/remaining' and n['historical_notice']['remaining_is_historical'] is True and n['supersession_scope']=='Current status supersedes only the historical remaining list; all notice bytes and license terms are unchanged.')
ref=[n['historical_notice'],*n['retained_attribution'],n['original_B']]
for x in ref:
 path=x['path'];ck('notice/originalB byte identity '+path,sha(Z/path)==x['sha256']==sha(A/path)==fs0[path]['sha256'])
ck('MIT independent pin',n['retained_attribution'][0]['sha256']=='206c4b6a50d0291622cae04f1ec98c6bf64609b53836f727b0d470e00528491c')
ck('scope independent pin',n['retained_attribution'][1]['sha256']=='69d2ce3c2e0bc7e3297d01fb40ef6e54a1a9428f6e38b98fabc5e755b6cf0af8')
ck('copyright attribution still named','Patrick Cooper' in (Z/'notices/ACE_RUNNER_MIT.txt').read_text() and 'Patrick Cooper' in (Z/'notices/RUNNER_LICENSE_SCOPE.md').read_text())
ck('resolved authority not reopened',n['owner_authority']=='resolved; not reopened by anonymous-review handling')
ck('originalB inherited projected acceptance not relabeled',n['original_B']=={'path':'B/acceptance.json','sha256':'5808b925bb69e8b2fb6e9d15fd80880c5a32adeecf78791630d54decf7b8ee9e','status':'accepted'})
ck('old replay pin and explicit new-manifest exclusion',n['saved_checkpoint_replay']=={'status':'accepted privately against candidate03; not a numerical replay of this final manifest','manifest_sha256':'b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9','private_acceptance_sha256':'57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232'})
ck('both reports accepted private pin',n['reports']=={'status':'both registered and descriptive outputs accepted privately','private_acceptance_sha256':'86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10'})
ck('accounting unchanged pin cutoff and unknown scope',n['accounting']=={'path':'publication/accounting.json','sha256':'51865d9dbbada810cca2fa601deb08af95a5ce999b00464372e1a44783e27985','status':'reviewed identified-allocation cutoff and explicit unknowns; not measured whole-project CPU'})
ck('all new anonymity/training/approval flags false',all(n[k] is False for k in ['anonymous_original_training_qualified','strictly_anonymous','human_anonymity_handling_approved','human_release_submission_approved']))
ck('outer approvals remain false',all(c[k] is False for k in ['anonymity_review_complete','anonymous_execution_qualified','training_reproduction_available','public_release_approved','submission_approved']))
ck('named notice reason and concrete human action',n['anonymity_reason']=='Mandatory attribution names the rights holder; source hashes and provenance may also permit linkage.' and n['human_action']=='Approve how the venue receives the attribution-bearing full reproduction package; retain all required notices.')
code_review=R/'docs/development/guidance/reviews/delivery_notice_status_correction_review_2026-10-24.md';lineage=rd(J/'lineage.json')
ck('Dewey code review exact reviewed correction lineage',lineage['review_sha256']==sha(code_review) and lineage['source_revision']=='4bd4d09cc7d417dce832082cc3e191b533339ffb' and 'Verdict: 0 required fixes.' in code_review.read_text() and PINS['checker'] in code_review.read_text())
ck('private request and planner bind reviewed sources',c['private_evidence']['request']['sha256']==sha(J/'request.json') and c['private_evidence']['planner']['sha256']=='d8f52818f8041c902ac1a19d0e636f8ae6c5ebe56ca5bbf93d0cef4a02cb49c3')
build=rd(J/'build_stdout.json');offline=rd(J/'offline_stdout.json');oe=rd(J/'offline_execution.json');archive=rd(J/'archive.json')
ck('actual build verified current pin and membership',build['manifest_sha256']==PINS['manifest'] and build['files_verified']==3940 and build['bindings_verified']==5121 and rd(J/'build_execution.json')['exit_code']==0)
ck('actual isolated checker success current pin',oe['exit_code']==0 and all(x in oe['command'] for x in ['-I','-S','-B',PINS['manifest']]) and offline['integrity']['manifest_sha256']==PINS['manifest'])
ck('offline predecessor companions report outputs counts',all(offline[k]==v for k,v in {'inherited_files':3893,'inherited_bindings':5068,'companions':16,'report_outputs':13}.items()) and offline['integrity']['files_verified']==3940 and offline['integrity']['bindings_verified']==5121)
ck('offline semantic statuses correct',offline['historical_notice_status_superseded'] is True and offline['strictly_anonymous'] is False and offline['human_anonymity_handling_approved'] is False and offline['public_release_approved'] is False and offline['private_execution_revalidated'] is False)
ck('archive complete streamed membership receipt',archive['archive_sha256']==PINS['archive'] and archive['archive_bytes']==361670105 and archive['members']==3942 and archive['all_members_match_manifest_and_sidecars'] is True and archive['manifest_sha256']==PINS['manifest'])
record=rd(P/'artifact_record.json');expected_names={'ACE_manuscript.pdf','ACE_manuscript_sources.zip','ACE_reproduction_with_attribution.zip','source_membership.json','packaging_measurements.json','README.md'}
ck('packet exact record membership',set(record['artifacts'])==expected_names and {x.name for x in P.iterdir()}==expected_names|{'artifact_record.json'})
for name,v in record['artifacts'].items():ck('delivered artifact checksum/size '+name,sha(P/name)==v['sha256'] and (P/name).stat().st_size==v['bytes'])
ck('actual archive trusted pin',record['artifacts']['ACE_reproduction_with_attribution.zip']['sha256']==PINS['archive'])
ck('README unchanged reviewed text',record['artifacts']['README.md']['sha256']==PINS['README'])
oldchecks={'ACE_manuscript.pdf':'00f31c47177898be45ed1d0222db8f6b38760584aa9a304c4396bf65a3fbab15','ACE_manuscript_sources.zip':'92a767c534db1705f78ed6ad6b521e787d1bc248d2bca6fcfe7497bf01470b40','source_membership.json':'148357c90b4b789ce6e76373b202dc590b100e6ea859a93f14643e2eb434795b'}
ck('prior PDF ZIP membership reviews inherited unchanged',all(record['artifacts'][k]['sha256']==v for k,v in oldchecks.items()))
ck('record final three independent pins',record['final_manifest_sha256']==PINS['manifest'] and record['final_checker_sha256']==PINS['checker'] and record['final_source_contract_sha256']==PINS['contract'])
ck('record preserves writing/report pins',record['paper_source_sha256']=='49cbaabd42ee24ea7061233fa4aefa0c8f5121368075fe98726e4be19fcd1694' and record['writing_revision']=='72c8a0b1160d2369e4a4f5e840ad47e3a698cb35' and record['report_acceptance_sha256']==n['reports']['private_acceptance_sha256'])
ck('record no human/strict anonymous approval',all(record[k] is False for k in ['approved_for_submission','strictly_anonymous_reproduction_archive','human_anonymity_handling_approved']))
# Read recorded local measurements and their original small receipts; no rerun or sum.
measurement=rd(P/'packaging_measurements.json');receipts={}
for folder in [B/'delivery-publication-source-capture-20261024-01',B/'delivery-final-release-launch-20261024-01',J]:
 for p in folder.glob('*.json'):
  if p.name in {'execution.json','prepare_execution.json','build_execution.json','offline_execution.json','archive.json'}:receipts[sha(p)]=(p,rd(p))
ck('measurement nine stages both attempts capture once',len(measurement['measurements'])==9 and sum(x['stage']=='writing_capture' for x in measurement['measurements'])==1 and {x['attempt'] for x in measurement['measurements']}=={'publication01_preserved_predecessor','publication02_notice_status_correction'})
for entry in measurement['measurements']:
 pair=receipts.get(entry['receipt_sha256']);v=pair[1] if pair else {}
 fields=set(entry)-{'stage','attempt','receipt_sha256'}
 ck('measurement receipt/value '+entry['attempt']+'/'+entry['stage'],pair is not None and all(entry[k]==v[k] for k in fields))
ck('measurement scopes no inferred project total',measurement['whole_project_process_cpu_seconds'] is None and measurement['prior_chain_allocated_cpu_seconds']==132995 and all(measurement[k]==0 for k in ['new_fits','new_responses','new_allocations']) and 'not per-stage RSS' in measurement['scope'] and 'no whole-project CPU total' in measurement['scope'])
failed=[x for x in checks if not x['passed']]
result={'schema':'ace-P1-actual02-semantic-delta-review-v1','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed' if not failed else 'required-finding','P1_closed':not failed,'required_findings':failed,'checks':checks,'passed':len(checks)-len(failed),'total':len(checks),'actual_pins':PINS,'packet_artifact_record_sha256':sha(P/'artifact_record.json'),'packaging_measurements_sha256':sha(P/'packaging_measurements.json'),'changed_payload_entries':changed,'new_bindings':m['bindings'][5117:],'notice_status':n,'parent_receipts':{p.name:sha(p) for p in [J/'lineage.json',J/'prepare_execution.json',J/'build_execution.json',J/'build_stdout.json',J/'offline_execution.json',J/'offline_stdout.json',J/'archive.json']},'code_delta_review_sha256':sha(code_review),'prior_evidence':{p.name:sha(p) for p in [O/'witness01.json',O/'archive_witness01.json',O/'final_disposition01.json',O/'README02_wording_review.json',O/'P1_delta_expectation.json']},'prior_175_checks_repeated':False,'generic_package_check_or_builder_executed':False,'archive_members_rescanned':False,'science_models_exporters_executed':False,'human_anonymity_handling_pending':True,'human_submission_approval_pending':True,'publication_approved':False,'scope':'Only actual P1 notice/status/binding delta and final author packet/measurement coherence. Prior complete scientific/identity/notice/page checks are inherited; full build/offline/archive receipts inspected, not repeated. No legal or anonymity certification.'}
with (O/'P1_actual02_witness.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':failed,'P1_closed':result['P1_closed'],'witness_sha256':sha(O/'P1_actual02_witness.json')},indent=2))
