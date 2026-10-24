#!/bin/bash
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
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02/slurm-%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02/slurm-%j.err
set -euo pipefail
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/bin/python -I -B - /scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02 00975b8ee41b5eca0cfb23a263bfdb4cdbe874eee24b68ac476ab662f92b2ca8 7b563c5846099a663490de3df70973e650473643b862375a78efdab331646176 <<'PY_BOOTSTRAP'
import hashlib,json,os,stat,sys
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
   with (root/'launcher_failure.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  except BaseException as custody:record['custody_failure']=str(custody)
 print(json.dumps(record),file=sys.stderr);raise SystemExit(1)
PY_BOOTSTRAP
