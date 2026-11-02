#!/bin/bash
#SBATCH --job-name=ace_llm_gcpu01
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B prepare_llm_intervention_gpu_cpu.py --freeze freeze.json --freeze-sha256 62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01/cpu01
