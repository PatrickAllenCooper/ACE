#!/bin/bash
#SBATCH --job-name=ace_llm_gdiag06
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gradient_gpu_20261102_06/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gradient_gpu_20261102_06/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gradient_gpu_20261102_06/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_gradients.py --freeze freeze.json --freeze-sha256 0ff56c88449d9ae053b97bb83a6bc7799369294358d69f2700cfb70294daa6c3 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gradient_gpu_20261102_06/gpu06
