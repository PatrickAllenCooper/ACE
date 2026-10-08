#!/bin/bash
#SBATCH --job-name=ace_delB_env
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_01/slurm-%j.log
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_01/slurm-%j.log
set -euo pipefail
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd /tmp
/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B - /scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_01 <<'BOOTSTRAP'
import hashlib,json,sys,os
from pathlib import Path
root=Path(sys.argv[1]);path=root/'prepare_environment.py';pin='142bea17f79b8580302b3a1f10fa4fd186f3a6bed16562bd9dea31fc197aeb1f';worker_pin='a76c15749baa9c6170feeb82cb739b27d4d92954e6117609d13fe6e9930fb050';fd=None;trusted=False
try:
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);s=os.fstat(fd)
 if s.st_ino!=680102203 or s.st_uid!=os.geteuid() or s.st_mode&0o077:raise ValueError('output descriptor custody differs')
 def read(name):
  handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  with os.fdopen(handle,'rb') as stream:return stream.read()
 claim=read('output_claim.json')
 if hashlib.sha256(claim).hexdigest()!='2f3306b72144deb124e4b6ad1b6f8277130b97161c7bed594bfd178460b2bb99':raise ValueError('output claim digest differs')
 trusted=True
 raw=read('prepare_environment.py');actual=hashlib.sha256(raw).hexdigest()
 if actual!=worker_pin:raise ValueError('captured worker hash differs')
 f=read('freeze.json');freeze=root/'freeze.json'
 if hashlib.sha256(f).hexdigest()!=pin:raise ValueError('freeze hash differs')
 sys.argv=[str(path),'--freeze',str(freeze),'--freeze-sha256',pin,'--output',str(root)]
 exec(compile(raw,str(path),'exec',dont_inherit=True),{'__name__':'__main__','__file__':str(path),'__executed_source_sha256__':actual,'__output_dir_fd__':fd,'__output_inode__':680102203,'__output_claim_sha256__':'2f3306b72144deb124e4b6ad1b6f8277130b97161c7bed594bfd178460b2bb99','__logical_output__':str(root)})
except BaseException as e:
 if isinstance(e,SystemExit) and e.code in (None,0):raise
 record={'status':'failed','failure_type':type(e).__name__,'failure_reason':str(e),'job_id':os.environ.get('SLURM_JOB_ID'),'freeze_sha256':pin,'new_fits':0,'new_responses':0}
 if trusted:
  try:
   handle=os.open('bootstrap_failure.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
   with os.fdopen(handle,'w') as stream:json.dump(record,stream,indent=2)
  except BaseException as custody:print('terminal write failed: '+str(custody),file=sys.stderr)
 else:print(json.dumps(record),file=sys.stderr)
 raise
BOOTSTRAP
