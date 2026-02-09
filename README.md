# Active Causal Experimentalist (ACE)

**Learning Intervention Strategies via Direct Preference Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

ACE (Active Causal Experimentalist) learns experimental design strategies for causal discovery using Direct Preference Optimization (DPO). By learning from pairwise comparisons between candidate interventions rather than non-stationary reward magnitudes, ACE achieves stable training and strategic intervention selection.

**Key Results:**
- **70-71% improvement** over baselines at equal intervention budgets
- **Statistical significance:** p<0.001, Cohen's d ≈ 2.0
- **Scales to 15-node systems** while maintaining competitive performance
- **Multi-domain validation:** Synthetic benchmarks, physics simulations, economic data

---

## Repository Structure

```
ACE/
├── paper/                    # LaTeX source for ICML 2026 submission
│   ├── paper.tex
│   └── references.bib
├── ace_experiments.py        # Main ACE implementation (5-node)
├── baselines.py             # Baseline methods (Random, PPO, Max-Variance, etc.)
├── experiments/             # Domain-specific experiments
│   ├── complex_scm.py       # 15-node complex SCM
│   ├── duffing_oscillators.py
│   ├── phillips_curve.py
│   └── run_ace_complex_full.py  # Full ACE for 15-node
├── jobs/                    # SLURM job scripts for HPC
├── scripts/                 # Analysis and utility scripts
├── tests/                   # Comprehensive test suite
├── results/                 # Experimental results
├── guidance_documents/      # Development documentation
└── docs/                    # Technical documentation and archives
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/PatrickAllenCooper/ACE.git
cd ACE

# Create conda environment
conda env create -f environment.yml
conda activate ace

# Or install with pip
pip install -r requirements.txt
```

### Run ACE on 5-Node Benchmark

```bash
# Single seed
python ace_experiments.py --episodes 200 --seed 42

# Multi-seed validation (N=5)
bash jobs/workflows/run_ace_only.sh
```

### Run Ablation Studies

```bash
# Diversity ablation
python ace_experiments.py --no_diversity_reward --episodes 100 --seed 42

# Submit all ablations to HPC
bash jobs/workflows/submit_ablations_verified.sh
```

### Run Complex 15-Node SCM

```bash
# Local
python experiments/run_ace_complex_full.py --seed 42 --episodes 300

# HPC
sbatch jobs/run_ace_complex_single_seed.sh
```

---

## Key Components

### ACE Architecture
- **Policy:** Qwen2.5-1.5B language model for intervention generation
- **Training:** Direct Preference Optimization (DPO) from pairwise comparisons
- **Reward:** Information gain + node importance + exploration diversity
- **Lookahead:** K=4 candidates evaluated on cloned learners
- **Stability:** Reference policy updates, observational training

### Baseline Methods
- **Random:** Uniform random intervention selection
- **Round-Robin:** Systematic node coverage
- **Max-Variance:** Uncertainty sampling via MC Dropout
- **PPO:** Learned policy with value-based RL

---

## Experimental Results

| Experiment | N | ACE Result | Baseline | Improvement |
|------------|---|------------|----------|-------------|
| 5-Node SCM | 5 | 0.92 ± 0.73 | 2.03-2.10 | 70-71% |
| 15-Node SCM | 1 | 4.54 | 4.51-4.71 | Competitive |
| Lookahead Ablation | 5 | 0.61 | 2.10 ± 0.11 | 71% |
| Diversity Ablation | 2 | 2.82 ± 0.22 | - | +206% degrad. |

**Statistical Validation:**
- Paired t-tests with Bonferroni correction (α = 0.0125)
- Large effect sizes (Cohen's d: 2.0-2.5)
- Multiple independent seeds for robustness

---

## Documentation

- **`FINAL_PAPER_RESULTS_SUMMARY.md`:** Complete experimental results
- **`SUBMISSION_CHECKLIST.md`:** Paper submission preparation
- **`guidance_documents/`:** Development and experimental planning docs
- **`docs/`:** Technical documentation and archived status reports

---

## Citation

```bibtex
@inproceedings{ace2026,
  title={Active Causal Experimentalist (ACE): Learning Intervention Strategies via Direct Preference Optimization},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Supplementary Materials

See `supplementary_materials/` for:
- Complete code for all experiments
- Detailed setup instructions
- Result analysis scripts
- Additional documentation

Packaged archive: `ace_supplementary_materials.tar.gz`

---

## Contact

For questions about the code or experiments, please open an issue on GitHub.

**Paper Submission:** ICML 2026 (Submitted February 2026)
