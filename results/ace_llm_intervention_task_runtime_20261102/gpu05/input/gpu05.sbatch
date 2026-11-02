#!/bin/bash
#SBATCH --job-name=ace_llm_tgpu05
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
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_task_gpu.py --freeze freeze.json --freeze-sha256 b17599b6ed864c603270c097d86ada40eaebc696dfd9815dac80f7769be599c9 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05/gpu05
