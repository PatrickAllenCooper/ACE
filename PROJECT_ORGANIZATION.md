# ACE Project Organization & Verification Status

**Date:** January 21, 2026  
**Status:** Project cleanup and verification in progress

---

## Project Structure (Organized)

```
ACE/
├── README.md                    # Main project documentation
├── CHANGELOG.md                 # Version history
├── START_HERE.md                # Current work entry point
├── RUN_ALL_SUMMARY.md          # Experiment summaries
├── LICENSE                      # Project license
│
├── ace_experiments.py           # Main ACE (DPO) experiment
├── baselines.py                 # Baseline comparisons
├── visualize.py                 # Result visualization
├── compare_methods.py           # Method comparison & tables
├── clamping_detector.py         # Verify clamping strategy (paper claim)
├── regime_analyzer.py           # Verify regime selection (paper claim)
│
├── run_all.sh                   # HPC job orchestrator (MAIN WORKFLOW)
│
├── jobs/                        # SLURM job scripts
│   ├── run_ace_main.sh         # ACE experiment (Table 1, Section 3.4.1)
│   ├── run_baselines.sh        # Baselines (Table 1, Section 3.8)
│   ├── run_complex_scm.sh      # Complex 15-node (Section 3.4.2)
│   ├── run_duffing.sh          # Duffing oscillators (Section 3.6)
│   └── run_phillips.sh         # Phillips curve (Section 3.7)
│
├── experiments/                 # Additional experiments
│   ├── complex_scm.py          # 15-node hard benchmark
│   ├── duffing_oscillators.py  # Physics domain
│   └── phillips_curve.py       # Economics domain
│
├── scripts/                     # Utility scripts
│   ├── verify_claims.sh        # Verify all paper claims
│   ├── extract_ace.sh          # Extract ACE metrics for tables
│   ├── extract_baselines.sh    # Extract baseline metrics
│   ├── pipeline_test.sh        # Quick validation (30 min)
│   ├── test_jan21_fixes.sh     # Test latest improvements
│   ├── check_version.sh        # Version checking
│   ├── cleanup.sh              # Clean temporary files
│   ├── launch_training.sh      # Launch training wrapper
│   └── run_ace_experiments.sh  # Legacy wrapper
│
├── tests/                       # Test suite (77% coverage, 470 tests)
│   ├── conftest.py
│   ├── pytest.ini
│   ├── requirements-test.txt
│   ├── README.md
│   ├── test_*.py               # 30+ test files
│   ├── baselines/              # Baseline tests
│   └── experiments/            # Experiment tests
│
├── guidance_documents/          # Project documentation
│   └── guidance_doc.txt        # Comprehensive technical guide
│
├── paper/                       # Paper LaTeX source
│   └── paper.tex
│
├── results/                     # Experiment outputs
└── logs/                        # Job logs
```

---

## HPC Workflow Verification

### run_all.sh Job Submissions

**✅ Job 1: ACE Main** (jobs/run_ace_main.sh)
- **Purpose:** Core DPO-based causal discovery experiment
- **Paper Support:** Table 1 (ACE results), Section 3.4.1 (main experiment)
- **Output:** `results/paper_TIMESTAMP/ace/`
- **Status:** ✅ Ready to run

**✅ Job 2: Baselines** (jobs/run_baselines.sh)
- **Purpose:** Random, Round-Robin, Max-Variance, PPO comparisons
- **Paper Support:** Table 1 (baseline results), Section 3.8 (comparisons)
- **Output:** `results/paper_TIMESTAMP/baselines/`
- **Status:** ✅ Ready to run

**✅ Job 3: Complex SCM** (jobs/run_complex_scm.sh)
- **Purpose:** Hard 15-node benchmark
- **Paper Support:** Section 3.4.2 (complex SCM validation)
- **Output:** `results/paper_TIMESTAMP/complex_scm/`
- **Status:** ✅ Ready to run

**✅ Job 4: Duffing Oscillators** (jobs/run_duffing.sh)
- **Purpose:** Physics domain validation
- **Paper Support:** Section 3.6 (clamping strategy discovery)
- **Output:** `results/paper_TIMESTAMP/duffing/`
- **Status:** ✅ Ready to run

**✅ Job 5: Phillips Curve** (jobs/run_phillips.sh)
- **Purpose:** Economics domain validation
- **Paper Support:** Section 3.7 (regime selection)
- **Output:** `results/paper_TIMESTAMP/phillips/`
- **Status:** ✅ Ready to run

---

## Paper Claims Verification

### Claims Supported by run_all.sh Workflow

**Table 1: Method Comparison**
- ✅ ACE results → Job 1 (run_ace_main.sh)
- ✅ Random baseline → Job 2 (run_baselines.sh)
- ✅ Round-Robin baseline → Job 2 (run_baselines.sh)
- ✅ Max-Variance baseline → Job 2 (run_baselines.sh)
- ✅ PPO baseline → Job 2 (run_baselines.sh)
- ⚠️ Extraction: Need `scripts/extract_ace.sh` and `scripts/compare_methods.py`

**Section 3.4.1: Main ACE Experiment**
- ✅ 5-node SCM collider learning → Job 1
- ✅ Episode count, early stopping → Job 1
- ⚠️ Specific metrics extraction needed

**Section 3.4.2: Complex 15-Node SCM**
- ✅ Hard benchmark → Job 3
- ⚠️ Strategy comparison results extraction needed

**Section 3.6: Duffing Oscillators**
- ✅ Clamping strategy (do(X2=0)) → Job 4
- ✅ Detection: `clamping_detector.py` ✓ Exists
- ⚠️ Integration with job output needed

**Section 3.7: Phillips Curve**
- ✅ Regime selection → Job 5
- ✅ Detection: `regime_analyzer.py` ✓ Exists
- ⚠️ Integration with job output needed

**Section 3.8: Baseline Comparisons**
- ✅ All baselines → Job 2
- ✅ Comparison script: `compare_methods.py` ✓ Exists
- ⚠️ Automated table generation needed

---

## What's Missing or Needs Integration

### Critical (Required for Paper)

1. **Results Extraction Pipeline** ⚠️
   - `scripts/extract_ace.sh` exists but needs verification
   - `scripts/extract_baselines.sh` exists but needs verification
   - Need automated flow: run_all.sh → extract → compare → tables

2. **Claim Verification Integration** ⚠️
   - `scripts/verify_claims.sh` exists
   - `clamping_detector.py` exists
   - `regime_analyzer.py` exists
   - Need: Automated post-processing after jobs complete

3. **Table Generation** ⚠️
   - `compare_methods.py` exists
   - Need: Automated Table 1 generation from results
   - Need: LaTeX table output format

4. **Figure Generation** ⚠️
   - `visualize.py` exists
   - Need: Automated figure generation for all experiments
   - Need: Paper-ready figure formatting

### Important (Quality of Life)

5. **Post-Processing Workflow** ⚠️
   - Need: `process_results.sh` to orchestrate:
     - Extract metrics from all jobs
     - Run verification (clamping, regime)
     - Generate comparison tables
     - Create figures
     - Validate all claims

6. **Results Documentation** ⚠️
   - Need: `results/README.md` explaining output structure
   - Need: `results/RESULTS_LOG.md` tracking experimental findings

7. **Dependencies Documentation** ⚠️
   - Need: `requirements.txt` for pip users
   - Need: `environment.yml` for conda users

### Nice to Have

8. **CI/CD Integration** ⚠️
   - `.github/workflows/test.yml` for GitHub Actions
   - Automated test running on push
   - Coverage reporting

9. **Pre-commit Hooks** ⚠️
   - `.pre-commit-config.yaml`
   - Run tests before commit

10. **Docker Support** ⚠️
    - `Dockerfile` for reproducible environment
    - `docker-compose.yml` if needed

---

## Verification Checklist

### Can run_all.sh Confirm Paper Claims?

**Currently:**
- ✅ Runs all 5 experiments
- ✅ Generates output data
- ✅ Saves to organized directories
- ⚠️ Missing: Automated extraction → verification → table generation flow

**What's Needed:**

1. **After Jobs Complete** →
   ```bash
   # Extract results
   ./scripts/extract_ace.sh results/paper_TIMESTAMP/
   ./scripts/extract_baselines.sh results/paper_TIMESTAMP/
   
   # Verify claims
   ./scripts/verify_claims.sh results/paper_TIMESTAMP/
   
   # Generate tables
   python compare_methods.py results/paper_TIMESTAMP/
   
   # Generate figures
   python visualize.py results/paper_TIMESTAMP/*/
   ```

2. **Integrated Post-Processing Script** →
   ```bash
   ./scripts/process_all_results.sh results/paper_TIMESTAMP/
   ```
   Should:
   - Extract all metrics
   - Verify all claims
   - Generate all tables
   - Create all figures
   - Produce summary report

3. **Claim-to-Result Mapping** →
   - Document which paper claim is supported by which job/script
   - Automated verification that all claims have supporting data

---

## Recommended Actions

### Phase 1: Critical (Required for Paper Submission)

1. **Create `scripts/process_all_results.sh`**
   - Orchestrate all post-processing
   - Extract → Verify → Generate → Report

2. **Enhance `compare_methods.py`**
   - Output LaTeX table format
   - Automated Table 1 generation

3. **Create `results/README.md`**
   - Document output structure
   - Explain each subdirectory

4. **Create claim verification report**
   - Map claims to evidence
   - Automated checking

### Phase 2: Important (Best Practices)

5. **Create `requirements.txt`**
   - List all Python dependencies with versions

6. **Create `environment.yml`**
   - Conda environment specification

7. **Create `results/RESULTS_LOG.md`**
   - Track experimental findings
   - Document claim evidence

### Phase 3: Nice to Have

8. **CI/CD setup** (.github/workflows/)
9. **Pre-commit hooks** (.pre-commit-config.yaml)
10. **Docker support** (Dockerfile)

---

## Current Organization Status

**✅ Organized:**
- Main Python modules in root
- Job scripts in `jobs/`
- Utility scripts in `scripts/`
- Tests in `tests/`
- Paper in `paper/`
- Documentation consolidated

**⚠️ Needs Attention:**
- Results extraction automation
- Claim verification integration
- Table/figure generation automation
- Dependencies documentation
- Post-processing workflow

---

## Next Steps

1. Verify extraction scripts work with run_all.sh output
2. Create integrated post-processing workflow
3. Document claim-to-evidence mapping
4. Create requirements.txt
5. Test complete workflow: run → process → verify → tables

**Goal:** Single command to go from raw experiments → verified paper claims → publication-ready tables/figures
