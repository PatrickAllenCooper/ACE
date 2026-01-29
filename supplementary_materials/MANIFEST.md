# ACE Supplementary Materials - File Manifest

This document lists all files included in the supplementary materials package.

## Core Code (12 files)

### Main Implementation
- `code/ace_experiments.py` (2943 lines) - Complete ACE training on 5-node SCM
- `code/baselines.py` (752 lines) - Baseline methods (Random, PPO, etc.)

### Complex SCM Experiments
- `code/experiments/complex_scm.py` (591 lines) - 15-node complex SCM definition
- `code/experiments/run_ace_complex_full.py` (780 lines) - ACE on complex SCM

## Experiment Scripts (6 files)

### Training Scripts
- `scripts/run_ace_5node.sh` - Run ACE on 5-node benchmark
- `scripts/run_ace_complex.sh` - Run ACE on 15-node benchmark
- `scripts/run_baselines.sh` - Run all baseline methods
- `scripts/run_ablations.sh` - Run ablation studies
- `scripts/run_multi_seed.sh` - Multi-seed statistical validation

### Analysis
- `scripts/analyze_results.py` - Compute statistics and compare methods

## Setup and Documentation (5 files)

- `README.md` - Complete usage instructions and API reference
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated environment setup
- `LICENSE` - MIT License
- `MANIFEST.md` - This file

## Total Package Size

- **Lines of Code:** ~5,000 (Python)
- **Core Implementation:** ~4,500 lines
- **Scripts:** ~500 lines
- **Documentation:** ~800 lines

## Verification Checklist

### Anonymization
- [x] No institution names
- [x] No HPC-specific paths
- [x] No user identifiers
- [x] No email addresses
- [x] No SLURM job IDs
- [x] No server hostnames

### Completeness
- [x] All core algorithms implemented
- [x] All baselines included
- [x] Ablation study scripts
- [x] Multi-seed validation
- [x] Result analysis tools
- [x] Complete documentation
- [x] Dependencies specified
- [x] Example usage provided

### Reproducibility
- [x] Exact hyperparameters documented
- [x] Random seeds specified
- [x] Training procedures detailed
- [x] Expected outputs described
- [x] Hardware requirements listed
- [x] Troubleshooting guide included

## File Sizes (Approximate)

```
code/ace_experiments.py           120 KB
code/baselines.py                  25 KB
code/experiments/complex_scm.py    20 KB
code/experiments/run_ace_complex_full.py  30 KB
scripts/*.sh                       15 KB (total)
scripts/analyze_results.py         8 KB
README.md                          20 KB
requirements.txt                   1 KB
setup.sh                          2 KB
LICENSE                           1 KB
MANIFEST.md                       2 KB

Total: ~245 KB (uncompressed code only)
```

## Usage Quick Reference

```bash
# Setup
./setup.sh
source ace_env/bin/activate

# Run main experiments
cd scripts/
./run_ace_5node.sh 42 results/ace 200

# Analyze results
python analyze_results.py results/ace

# Multi-seed validation
./run_multi_seed.sh results/validation 200
```

## Citation

This code package accompanies the paper:
"ACE: Active Causal Experimentation with Large Language Models"

For questions or issues, please refer to the main README.md file.
