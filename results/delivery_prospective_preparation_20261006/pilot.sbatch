#!/bin/bash
#SBATCH --job-name=ace_delB_pilot
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --chdir=/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_20261006
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_20261006/slurm-%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/delivery_prospective_pilot_20261006/slurm-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/projects/paco0228/miniconda3/envs/ace/bin/python bundle/scripts/research/delivery_prospective_pilot.py run --out pilot
