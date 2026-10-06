"""One cause-specific pilot admission; preserve ambiguous outcomes without retry."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from datetime import datetime,timezone

ROOT=Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_importfix_20261006')
OLD=Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_20261006')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def exclusive(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

reg=read(ROOT/'pilot/registration.json');old=read(OLD/'pilot/registration.json')
decision=read(ROOT/'pilot_importfix_recovery_decision.json')
script=ROOT/'pilot_importfix_proposal.sbatch'
for key in ('worker_hashes','source_hashes','generator_hashes','dependencies','development_coefficient_seed',
            'action_seed','rows_per_development_history','scratch_epochs','online_tail_updates',
            'threads','rss_bytes','wall_seconds','account','full_stage_ceiling_cpu_core_hours',
            'confirmation_outcomes_permitted','test_losses_permitted'):
    if reg[key]!=old[key]:raise ValueError('frozen pilot setting changed: '+key)
if sha(script)!=decision['proposal_sbatch_sha256'] or decision['failed_pilot_retained_cpu_seconds']!=126:
    raise ValueError('reviewed environment correction or historical charge changed')
if Path(reg['project']).resolve()!=(OLD/'bundle').resolve() or Path(reg['source']).resolve()!=(OLD/'bundle/runner').resolve():
    raise ValueError('original source path changed')
for name,h in reg['worker_hashes'].items():
    if sha(OLD/'bundle/scripts/research'/name)!=h:raise ValueError('frozen worker changed')
for name,h in reg['source_hashes'].items():
    if sha(OLD/'bundle/runner'/name)!=h:raise ValueError('original learner changed')
for name,h in reg['generator_hashes'].items():
    if sha(OLD/'bundle'/name)!=h:raise ValueError('generator changed')
if (ROOT/'pilot/supervisor_started.json').exists():raise ValueError('already started; reconcile')
q=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T'],text=True,capture_output=True,timeout=15,check=True)
if any('ace_delB_pilot_fix' in line for line in q.stdout.splitlines()):raise ValueError('existing corrected pilot; do not duplicate')
a=subprocess.run(['sacct','-j','33505661','-X','-n','-P','-o','JobID,Account,State,CPUTimeRAW'],text=True,capture_output=True,timeout=15,check=True)
if not any(line.split('|')[:4]==['33505661','ucb736_asc1','FAILED','126'] for line in a.stdout.strip().splitlines()):
    raise ValueError('original terminal accounting not reconciled')
command=['sbatch','--parsable',str(script)]
exclusive(ROOT/'submission_started.json',{'at':utc(),'command':command,'registration_sha256':sha(ROOT/'pilot/registration.json'),
    'script_sha256':sha(script),'decision_sha256':sha(ROOT/'pilot_importfix_recovery_decision.json'),
    'submit_driver_sha256':sha(Path(__file__)),'account':'ucb736_asc1','prior_failed_cpu_seconds':126,'original_sacct':a.stdout})
try:
    result=subprocess.run(command,text=True,capture_output=True,timeout=30)
    output=result.stdout.strip();valid=re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',output)
    record={'at':utc(),'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'status':'submitted' if result.returncode==0 and valid else 'failed' if result.returncode else 'ambiguous',
        'job_id':output.split(';')[0] if result.returncode==0 and valid else None}
except subprocess.TimeoutExpired as e:
    record={'at':utc(),'status':'ambiguous_timeout','job_id':None,'stdout':str(e.stdout),'stderr':str(e.stderr)}
exclusive(ROOT/'submission_attempt.json',record)
print(json.dumps(record))
if record['status']!='submitted':raise SystemExit('submission unresolved; reconcile journal, never blind retry')
