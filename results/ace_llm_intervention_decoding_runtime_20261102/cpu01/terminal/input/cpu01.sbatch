#!/bin/bash
#SBATCH --job-name=ace_llm_dcpu01
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_decoding_cpu_20261102_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_decoding_cpu_20261102_01/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_decoding_cpu_20261102_01/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B prepare_llm_intervention_decoding_cpu.py --freeze freeze.json --freeze-sha256 cf73af49666a76079c44cedbffcd93e63f19a561ae4da745438ba449019cf89c --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_decoding_cpu_20261102_01/cpu01
