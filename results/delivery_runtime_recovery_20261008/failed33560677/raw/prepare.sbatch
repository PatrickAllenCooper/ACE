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
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_02/slurm-%j.log
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_02/slurm-%j.log
set -euo pipefail
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd /tmp
/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B - /scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_02 <<'BOOTSTRAP'
import hashlib,json,sys,os
from pathlib import Path
root=Path(sys.argv[1]);path=root/'prepare_environment.py';pin='76b601ef875b9ae9316385225d3eb77c6230fadd42b97088fed5d810e5dc768c';worker_pin='859659f079f2de2c69ee85ef02e2de63b70ea8fb02104dd071a21ec9618d9008';fd=None;trusted=False
try:
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);s=os.fstat(fd)
 if s.st_ino!=680661409 or s.st_uid!=os.geteuid() or s.st_mode&0o077:raise ValueError('output descriptor custody differs')
 def read(name):
  handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  with os.fdopen(handle,'rb') as stream:return stream.read()
 claim=read('output_claim.json')
 if hashlib.sha256(claim).hexdigest()!='09cfb32f46fae39778a3f85a62b0cd8410d2e367be23b9b05a176801a7aef605':raise ValueError('output claim digest differs')
 trusted=True
 raw=read('prepare_environment.py');actual=hashlib.sha256(raw).hexdigest()
 if actual!=worker_pin:raise ValueError('captured worker hash differs')
 f=read('freeze.json');freeze=root/'freeze.json'
 if hashlib.sha256(f).hexdigest()!=pin:raise ValueError('freeze hash differs')
 sys.argv=[str(path),'--freeze',str(freeze),'--freeze-sha256',pin,'--output',str(root)]
 exec(compile(raw,str(path),'exec',dont_inherit=True),{'__name__':'__main__','__file__':str(path),'__executed_source_sha256__':actual,'__output_dir_fd__':fd,'__output_inode__':680661409,'__output_claim_sha256__':'09cfb32f46fae39778a3f85a62b0cd8410d2e367be23b9b05a176801a7aef605','__logical_output__':str(root)})
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
