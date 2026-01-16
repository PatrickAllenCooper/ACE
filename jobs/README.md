# ACE Job Scripts

This directory contains SLURM job scripts for running ACE experiments on HPC systems.

## Job Scripts

| Script | Purpose | Resources | Time |
|--------|---------|-----------|------|
| `run_ace_main.sh` | Main ACE DPO experiment | 32GB, 8 CPUs, 1 GPU | 12h |
| `run_baselines.sh` | All baseline methods | 32GB, 8 CPUs, 1 GPU | 8h |
| `run_duffing.sh` | Duffing oscillators (physics) | 32GB, 8 CPUs, 1 GPU | 4h |
| `run_phillips.sh` | Phillips curve (economics) | 16GB, 4 CPUs, 1 GPU | 2h |

## Usage

### Via Orchestrator (Recommended)
Use `run_all.sh` from the project root to submit all jobs:
```bash
./run_all.sh                    # Submit all 4 jobs in parallel
QUICK=true ./run_all.sh         # Quick test (10 episodes)
```

### Individual Job Submission
You can also submit individual jobs directly:
```bash
# ACE Main
sbatch --export=EPISODES=500,OUTPUT_DIR=results/ace jobs/run_ace_main.sh

# Baselines
sbatch --export=EPISODES=100,OUTPUT_DIR=results/baselines jobs/run_baselines.sh

# Duffing
sbatch --export=EPISODES=100,OUTPUT_DIR=results/duffing jobs/run_duffing.sh

# Phillips
sbatch --export=EPISODES=50,OUTPUT_DIR=results/phillips jobs/run_phillips.sh
```

## Environment Variables

Each job script accepts:
- `EPISODES` - Number of episodes to run
- `OUTPUT_DIR` - Directory for output files

## SLURM Directives

All scripts include:
- Partition: `aa100` (GPU partition)
- QOS: `normal`
- GPU: 1x A100
- Module: `cuda`
- Conda: `ace` environment

## Logs

Job outputs are written to:
- Standard output: `logs/<job>_<timestamp>_<jobid>.out`
- Standard error: `logs/<job>_<timestamp>_<jobid>.err`
