#!/bin/bash
#SBATCH --job-name=ace_llm_gpu01
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_gpu.py --freeze freeze.json --freeze-sha256 717f77e9f663f356e68de6de36fd873380889a8398d4107912fd6c87f602e0d4 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01/gpu01
