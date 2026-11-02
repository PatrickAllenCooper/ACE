#!/bin/bash
#SBATCH --job-name=ace_llm_cpu01
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B qualify_llm_intervention_cpu.py --freeze freeze.json --freeze-sha256 9c114bc8d557f677dbd71bd36c6123daae9b3fff3d3bce96322c73b94ced6364 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/cpu01
