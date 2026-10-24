from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
b=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline');repo=Path('/Users/pat/code/ACE');d=b/'delivery-prospective-reporting-launch-20261024-02';prep=b/'delivery-prospective-reporting-preparation-20261024-02';raw=b/'delivery-diagnostic-replay-terminal-20261024-12/raw';source=repo/'scripts/research'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (d/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
review=repo/'docs/development/guidance/reviews/delivery_report01_diagnostic_review_2026-10-24.md'
assert 'Final required findings remaining: 0.' in review.read_text()
source_names=['run_delivery_prospective_reporting.py','qualify_delivery_replay_runtime.py','supervise_delivery_qualified_replay.py','delivery_replay_supervision.py','runner_delivery_confirmation.py','audit_delivery_prospective_results.py','verify_delivery_release.py','replay_delivery_prospective_release.py','export_delivery_prospective_claims.py','prepare_delivery_prospective_supplement.py']
for n in source_names:
 with (d/n).open('xb') as f:f.write((source/n).read_bytes())
for n,p in [('runtime_inventory.json',raw/'runtime_inventory.json'),('replay_receipt.json',raw/'replay_receipt.json'),('replay_acceptance.json',prep/'replay_acceptance.json')]:
 with (d/n).open('xb') as f:f.write(p.read_bytes())
claim=json.loads((d/'output_claim.json').read_text());old=json.loads((raw/'freeze.json').read_text());original=b/'delivery-prospective-accepted-raw-20261008T0414Z'
auth={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authority':'Patrick blanket continuation approval; delivery_continuation_authorization_2026-10-24.md','scope':'both unchanged reporting interfaces after independently accepted replay12; no model loading/fits/updates/responses','accepted_replay_job':34863006,'accepted_replay_sha256':h(d/'replay_acceptance.json'),'review_sha256':h(review),'cpus':1,'rss_bytes':3*2**30,'wall_seconds':3600,'reserved_before_core_hours':93.368333,'reservation_after_submission_core_hours':94.368333,'ceiling_core_hours':150,'new_fits':0,'new_responses':0}
save('authorization.json',auth)
reg={'schema':'delivery-prospective-reporting-execution-v1','at':auth['at'],'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'output':claim['root'],'output_inode':claim['inode'],'output_claim_sha256':h(d/'output_claim.json'),'authorization_sha256':h(d/'authorization.json'),'source_pins':{n:h(d/n) for n in source_names},'metadata_pins':{n:h(d/n) for n in ['runtime_inventory.json','replay_receipt.json','replay_acceptance.json']},'original_results':'/scratch/alpine/paco0228/ACE/results/delivery_prospective_full_20261006','original_input_pins':{n:h(original/n) for n in ['registration.json','acceptance.json','scores.json','complete.json','fit_seal.json','audit_execution.json']},'package':old['package'],'manifest_sha256':old['manifest_sha256'],'environment':old['environment'],'python_sha256':old['base_python_sha256'],'cpus':1,'rss_bytes':3*2**30,'wall_seconds':3600,'slurm_wall_limit':'01:00:00','account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','new_fits':0,'new_optimizer_updates':0,'new_responses':0}
save('report_freeze.json',reg)
bootstrap='''import hashlib,json,os,stat,sys
from pathlib import Path
root,pin,worker_pin=sys.argv[1:];root=Path(root);trusted=False
try:
 raw=(root/'report_freeze.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('report freeze changed')
 reg=json.loads(raw);info=root.stat()
 if root.is_symlink() or info.st_ino!=reg['output_inode'] or info.st_uid!=os.geteuid() or info.st_mode&0o077:raise ValueError('report output changed')
 if hashlib.sha256((root/'output_claim.json').read_bytes()).hexdigest()!=reg['output_claim_sha256']:raise ValueError('report output claim changed')
 trusted=True
 p=root/'run_delivery_prospective_reporting.py';raw=p.read_bytes()
 if p.is_symlink() or hashlib.sha256(raw).hexdigest()!=worker_pin or reg['source_pins'][p.name]!=worker_pin:raise ValueError('report runner changed')
 sys.argv=[str(p),'--root',str(root),'--freeze-sha256',pin]
 exec(compile(raw,str(p),'exec',dont_inherit=True),{'__name__':'__main__','__file__':str(p)})
except BaseException as exc:
 record={'status':'launcher_failure','at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),'type':type(exc).__name__,'reason':str(exc),'freeze_sha256':pin,'new_fits':0,'new_responses':0}
 if trusted:
  try:
   with (root/'launcher_failure.json').open('x') as f:json.dump(record,f,indent=2);f.write('\\n')
  except BaseException as custody:record['custody_failure']=str(custody)
 print(json.dumps(record),file=sys.stderr);raise SystemExit(1)
'''
out=reg['output'];cmd=f"exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 {reg['environment']}/bin/python -I -B - {out} {h(d/'report_freeze.json')} {reg['source_pins']['run_delivery_prospective_reporting.py']} <<'PY_BOOTSTRAP'\n"
script=f'''#!/bin/bash
#SBATCH --job-name=ace_delB_report
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output={out}/slurm-%j.out
#SBATCH --error={out}/slurm-%j.err
set -euo pipefail
'''+cmd+bootstrap+'PY_BOOTSTRAP\n'
with (d/'run.sbatch').open('x') as f:f.write(script)
inputs=source_names+['runtime_inventory.json','replay_receipt.json','replay_acceptance.json','report_freeze.json','authorization.json','output_claim.json','run.sbatch'];save('input_pins.json',{n:h(d/n) for n in inputs})
save('packet_receipt.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_pins_sha256':h(d/'input_pins.json'),'source_revision':reg['source_revision'],'source_review_sha256':h(review),'independently_accepted_replay_sha256':h(d/'replay_acceptance.json'),'readiness_sha256':h(prep/'readiness.json'),'reconciliation_sha256':h(prep/'reconciliation.json'),'not_submitted':True})
print(json.dumps({'freeze':h(d/'report_freeze.json'),'sources':len(source_names),'input_pins':len(inputs),'script':h(d/'run.sbatch')},indent=2))
