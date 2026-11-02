#!/bin/bash
#SBATCH --job-name=ace_llm_prec07
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
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_precision_gpu_20261102_07/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_precision_gpu_20261102_07/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_precision_gpu_20261102_07/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_precise_gpu.py --freeze freeze.json --freeze-sha256 61e4b5c48f8d6dbbb06abcdfa1645cbbac0c4ffa20e50854aa635006265daac4 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_precision_gpu_20261102_07/gpu07
