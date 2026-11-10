"""Exclusive metadata publication, guarded by all completed independent reviews.
No project imports or remote operations. Not invoked until the last gate closes.
"""
import hashlib,json,shutil,datetime
from pathlib import Path
HERE=Path(__file__).parent
R=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
A=BASE/'ACE-LLM-gpu02-35516125-postcustody-01'
def raw(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def load(p):return json.loads(Path(p).read_bytes())
def write(p,b):
 with p.open('xb') as f:f.write(b)
def save(p,x):write(p,raw(x));return sha(raw(x))
gate=load(HERE/'transfer-final-join.json')
need(gate['status']=='verified' and gate['required_findings']==0,'final actual transfer-content gate must close first')
facts=load(HERE/'attempt04/facts.json');runtime=load(HERE/'runtime02/runtime-facts.json')
need(facts['checks']>0 and len(runtime['records'])==11 and len(runtime['owned_cleanup'])==3,'actual independent gates')
c=load(A/'inputs/config.json')
def role(name):
 x=c['artifacts'][name];b=Path(x['path']).read_bytes();need(sha(b)==x['sha256'],'role pin '+name);return json.loads(b)
p=role('inner');outer=role('outer');cpu=role('cpu_acceptance');base=role('baseline');rec=role('remote_custody');full=rec['full_inventory']['files']
before=json.loads(role('before')['stdout']);after=json.loads(role('after')['stdout']);disposition=role('disposition');manifest=role('operational_manifest');context=role('recovery_context')['execution_context']
report=load(A/'audit-attempt04/report.json');need(sha((A/'audit-attempt04/report.json').read_bytes())=='eb0f711af3f98e997acec67cd56690c9b68146c112bd052412770f7bd4cd1060' and report['errors']==[],'final aid report')
T=Path(c['capsules'][1])
terminal=load(T/'remote/launch/gpu/terminal.json')
need(facts['accounting']['closed_cpu_seconds']==37073 and facts['accounting']['closed_gpu_slice_seconds']==4458,'verified actual accounting')
need(len(before['related_accounting'])==48 and len(before['journal_pins'])==98 and before['related_accounting']==after['related_accounting'] and before['journal_pins']==after['journal_pins'],'stable current history')
output=HERE/'publication';output.mkdir()
gpu_files={n[len('launch/gpu/'):]:v['sha256'] for n,v in full.items() if n.startswith('launch/gpu/')}
camp_prefix='cpu/cpu/worker/artifacts/campaign/'
campaign_files={n[len(camp_prefix):]:v['sha256'] for n,v in full.items() if n.startswith(camp_prefix)}
allocation=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',state='COMPLETED',exit_code='0:0')
acceptance=dict(schema='ace-scientific-campaign-gpu-acceptance-v1',status='development_gpu_phase_accepted',scope='development',gpu_packet_sha256=c['artifacts']['inner']['sha256'],cpu_packet_sha256=p['cpu_packet_sha256'],cpu_acceptance_sha256=c['artifacts']['cpu_acceptance']['sha256'],job_id='35516125',allocation=allocation,gpu_output=p['paths']['gpu_output'],campaign_root=p['paths']['campaign'],gpu_output_files=gpu_files,campaign_files=campaign_files,bundle_sha256=cpu['bundle_sha256'],attempt_pins=terminal['combined_attempt_pins'],accepted_checks=dict(scheduler=True,runtime=True,cleanup=True,custody=True),terminal_pins={n:gpu_files[n] for n in ['terminal.json','controller/phase/terminal.json']},scientific_qualified=False)
need(len(acceptance)==17 and len(gpu_files)==13262 and len(campaign_files)==27869,'exact existing core schema and full maps')
accept_pin=save(output/'acceptance.json',acceptance)
baseline=dict(base)
baseline.update(closed_cpu_seconds=37073,closed_gpu_slice_seconds=4458,cpu_admission_seconds=40673,input_pins=dict(predecessor=c['artifacts']['baseline']['sha256'],acceptance=accept_pin,disposition=c['artifacts']['disposition']['sha256']),journal_pins=before['journal_pins'],outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,related_accounting=before['related_accounting'],submission_journals=98,terminal_jobs=48)
need(set(baseline)==set(base),'same accepted accounting schema')
baseline_pin=save(output/'baseline.json',baseline)
review=(HERE/'review-final.md').read_bytes()
need(b'DRAFT' not in review and b'_PENDING' not in review,'final review required')
review+=('\nMachine records: acceptance SHA `'+accept_pin+'`; baseline SHA `'+baseline_pin+'`. The separate recovery supplement pins these and this review; no hash cycle is introduced.\n').encode()
write(output/'review.md',review);review_pin=sha(review)
role_names={'gpu-review':None,'gpu-disposition':'disposition','gpu-inventory':'before','gpu-inventory-after':'after','gpu-custody-inventory':'custody','gpu-recovery-review':'recovery_review','preflight-review':'preflight_review','operational-source-manifest':'operational_manifest','operational-source-review':'operational_review','gpu-failure-disposition':'failed_disposition','gpu-failure-seal':'failed_seal','gpu-failure-inventory':'failed_before','gpu-failure-inventory-after':'failed_after'}
ep={n:review_pin if k is None else c['artifacts'][k]['sha256'] for n,k in role_names.items()}
supp=dict(schema='ace-scientific-campaign-gpu-recovery-acceptance-v1',status='accepted',scope='development',required_findings=0,acceptance=True,scientific_qualified=False,historical_postcustody_verified=True,originals_preserved=True,job_id='35516125',gpu_acceptance_sha256=accept_pin,gpu_packet_sha256=c['artifacts']['inner']['sha256'],gpu_launch_packet_sha256=c['artifacts']['outer']['sha256'],cpu_acceptance_sha256=c['artifacts']['cpu_acceptance']['sha256'],authority_sha256=c['artifacts']['recovery_authority']['sha256'],preflight_context_sha256=c['artifacts']['recovery_context']['sha256'],preflight_receipt_sha256=c['artifacts']['preflight_receipt']['sha256'],readset_sha256=c['artifacts']['readset']['sha256'],failure_acceptance_sha256=c['artifacts']['failed_acceptance']['sha256'],failure_baseline_sha256=c['artifacts']['baseline']['sha256'],baseline_sha256=baseline_pin,source_files_sha256=context['source_files_sha256'],gpu_terminal_sha256=gpu_files['terminal.json'],phase_terminal_sha256=gpu_files['controller/phase/terminal.json'],postcustody_marker_sha256=gpu_files['controller/historical-postcustody-required.json'],before_observation_sha256=c['artifacts']['before']['sha256'],after_observation_sha256=c['artifacts']['after']['sha256'],custody_sha256=c['artifacts']['custody']['sha256'],control_root=outer['remote_root'],output_root=p['paths']['gpu_output'],raw_root_labels=dict(control=outer['remote_root'],launch=p['paths']['gpu_output'].removesuffix('/gpu')),cpu_campaign_root=p['paths']['campaign'],original_source_commit=outer['source_commit'],original_gpu_entry_sha256='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1',operational_source_commit=manifest['source_commit'],failed_gpu01_job_id='35498218',archive_receipt_sha256=c['artifacts']['remote_custody']['sha256'],failed_archive_receipt_sha256=c['artifacts']['failed_remote_custody']['sha256'],evidence_pins=ep)
supp_pin=save(output/'recovery-acceptance.json',supp)
evidence=output/'independent-evidence';evidence.mkdir()
for path in HERE.iterdir():
 if path==output:continue
 if path.is_dir():shutil.copytree(path,evidence/path.name,ignore=lambda directory,names: ['request.json'] if Path(directory)==HERE/'transfer-final-evidence/fetch-log' and 'request.json' in names else [])
 else:shutil.copyfile(path,evidence/path.name)
repo=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/acceptance01'
external=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-acceptance-20261110-02'
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),required_findings=0,job_id='35516125',acceptance_sha256=accept_pin,baseline_sha256=baseline_pin,recovery_acceptance_sha256=supp_pin,review_sha256=review_pin,scientific_utility_qualified=False,source_execution=False,remote_operations=0,repo=str(repo),external=str(external),original_aid_report_sha256='eb0f711af3f98e997acec67cd56690c9b68146c112bd052412770f7bd4cd1060',original_aid_seal_sha256='3465f27d525f4b5d5b6297349a11f8de00b33b0be28a691175f20b26995f5950',original_terminal_seal_sha256=c['artifacts']['custody']['sha256'])
save(output/'receipt.json',receipt)
files={str(q.relative_to(output)):dict(bytes=q.stat().st_size,sha256=sha(q.read_bytes())) for q in sorted(output.rglob('*')) if q.is_file()}
save(output/'inventory.json',dict(files=files))
for dest in [repo,external]:
 dest.mkdir()
 for path in sorted(output.rglob('*')):
  target=dest/path.relative_to(output)
  need(not path.is_symlink(),'no symlink in review publication')
  if path.is_dir():target.mkdir()
  else:write(target,path.read_bytes())
 for name,row in files.items():
  data=(dest/name).read_bytes();need(len(data)==row['bytes'] and sha(data)==row['sha256'],'publication copy '+name)
 need((dest/'inventory.json').read_bytes()==(output/'inventory.json').read_bytes(),'inventory copies')
print(json.dumps({**receipt,'receipt_sha256':sha((output/'receipt.json').read_bytes()),'inventory_sha256':sha((output/'inventory.json').read_bytes()),'files':len(files)+1},indent=2,sort_keys=True))
