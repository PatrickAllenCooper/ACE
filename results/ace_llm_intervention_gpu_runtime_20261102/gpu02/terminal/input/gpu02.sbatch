#!/bin/bash
#SBATCH --job-name=ace_llm_gpu02
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_02/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_02/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_02/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_gpu.py --freeze freeze.json --freeze-sha256 fd041a2401d93a5044ef4c01e37da3a83d7b6fe6fa97f83fde7999fe012a8dd8 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_02/gpu02
