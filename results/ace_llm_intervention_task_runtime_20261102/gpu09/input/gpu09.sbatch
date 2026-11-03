#!/bin/bash
#SBATCH --job-name=ace_llm_check09
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_checkpoint_gpu.py --freeze freeze.json --freeze-sha256 5a4289812cbd3181dee1b36e372cd1f3fbed4456eceaffa4adbaa9de1f0b603f --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09/gpu09
