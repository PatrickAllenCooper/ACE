#!/bin/bash
#SBATCH --job-name=ace_delB_replay
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261008_06/slurm-%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261008_06/slurm-%j.err
set -euo pipefail
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/bin/python -I -B - /scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261008_06 4f813ea15642925e0ee551075063206d947508c780619d92c7762ee1a2dba7e2 3e09f8e40cf3613a5f5642fcd8275039e048724bcc3441108634dce56610cf44 432817a92c52c49c1436295b155b16caa21da411ccf074dd249ee99ffa1d2a2c <<'PY_BOOTSTRAP'
import hashlib,json,os,stat,sys,traceback
launch_root,launch_pin,launch_worker_pin,launch_claim_pin=sys.argv[1:]
launch_fd=None;launch_trusted=False

def launch_read(name):
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=launch_fd)
 with os.fdopen(h,'rb') as f:
  if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular launcher input required')
  return f.read()
try:
 launch_fd=os.open(launch_root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 launch_raw=launch_read('freeze.json')
 if hashlib.sha256(launch_raw).hexdigest()!=launch_pin:raise ValueError('launcher freeze differs')
 launch_reg=json.loads(launch_raw);launch_info=os.fstat(launch_fd)
 if launch_info.st_ino!=launch_reg['output_inode'] or launch_info.st_uid!=os.geteuid() or launch_info.st_mode&0o077:raise ValueError('launcher output identity differs')
 if hashlib.sha256(launch_read('output_claim.json')).hexdigest()!=launch_claim_pin or launch_reg['output_claim_sha256']!=launch_claim_pin:raise ValueError('launcher claim differs')
 launch_trusted=True
 launch_source=launch_read('supervise_replay.py')
 if hashlib.sha256(launch_source).hexdigest()!=launch_worker_pin or launch_reg['supervision_worker_sha256']!=launch_worker_pin:raise ValueError('launcher worker differs')
 launch_path='/proc/self/fd/'+str(launch_fd)+'/supervise_replay.py'
 launch_context=vars(sys.modules['__main__']);launch_context.update(__name__='__main__',__file__=launch_path,__output_dir_fd__=launch_fd,__executed_source_sha256__=launch_worker_pin)
 sys.argv=[launch_path,'--freeze','/proc/self/fd/'+str(launch_fd)+'/freeze.json','--freeze-sha256',launch_pin]
 exec(compile(launch_source,launch_path,'exec',dont_inherit=True),launch_context)
except BaseException as launch_exc:
 launch_failure={'status':'launcher_failure','job_id':os.environ.get('SLURM_JOB_ID'),'freeze_sha256':launch_pin,'output_trusted':launch_trusted,'primary_type':type(launch_exc).__name__,'primary_reason':str(launch_exc),'new_fits':0,'new_responses':0}
 if launch_trusted:
  try:launch_failure['preserved_worker_terminal_sha256']=hashlib.sha256(launch_read('replay_execution.json')).hexdigest()
  except FileNotFoundError:pass
  except BaseException as inspect_exc:launch_failure['terminal_inspection_failure']={'type':type(inspect_exc).__name__,'reason':str(inspect_exc)}
  try:
   launch_h=os.open('launcher_failure.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=launch_fd)
   with os.fdopen(launch_h,'w') as launch_f:json.dump(launch_failure,launch_f,indent=2);launch_f.write('\n');launch_f.flush();os.fsync(launch_f.fileno())
  except BaseException as custody_exc:launch_failure['custody_failure']={'type':type(custody_exc).__name__,'reason':str(custody_exc)}
 print(json.dumps(launch_failure),file=sys.stderr)
 raise SystemExit(1)
PY_BOOTSTRAP
