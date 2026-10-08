#!/bin/bash
#SBATCH --job-name=ace_delB_replay
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_prospective_replay_20261008_03/slurm-%j.log
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_prospective_replay_20261008_03/slurm-%j.log
set -euo pipefail
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd /tmp
/projects/paco0228/miniconda3/envs/ace/bin/python -I -B - /scratch/alpine/paco0228/ACE/results/delivery_prospective_replay_20261008_03 <<'BOOTSTRAP'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);path=root/'supervise_replay.py';raw=path.read_bytes()
if hashlib.sha256(raw).hexdigest()!='d7067a563c42a1a315f0156901adc4149d0e32f367d2d5cc8c5667732ce47466':raise ValueError('launch worker digest mismatch')
freeze=root/'freeze.json';freeze_raw=freeze.read_bytes()
if hashlib.sha256(freeze_raw).hexdigest()!='41bf3ba3627b644667cdb20adabe93d36021cd804165be7af9e15e4457884a6f':raise ValueError('launch freeze digest mismatch')
sys.argv=[str(path),'--freeze',str(freeze),'--freeze-sha256','41bf3ba3627b644667cdb20adabe93d36021cd804165be7af9e15e4457884a6f']
exec(compile(raw,str(path),'exec',dont_inherit=True),{'__name__':'__main__','__file__':str(path)})
BOOTSTRAP
