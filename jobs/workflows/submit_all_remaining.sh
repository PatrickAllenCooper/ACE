#!/bin/bash

# ============================================================
# SUBMIT ALL REMAINING EXPERIMENTS FOR PAPER
# ============================================================
# This submits:
#   - 3 ablation jobs (no_convergence, no_root_learner, no_diversity)
#   - 1 no-oracle job (5 seeds)
#   - 1 complex SCM job (1 seed)
# Total: 5 jobs, ~15-18 hours runtime
# ============================================================

echo "============================================================"
echo "SUBMITTING ALL REMAINING EXPERIMENTS"
echo "============================================================"
echo ""

cd /projects/$USER/ACE || { echo "ERROR: Cannot find /projects/$USER/ACE"; exit 1; }

# ============================================================
# 1. ABLATIONS (3 jobs, 3 seeds each, ~3 hours each)
# ============================================================
echo "1. Submitting ABLATIONS (3 jobs)..."
echo "   - no_convergence (3 seeds)"
echo "   - no_root_learner (3 seeds)"
echo "   - no_diversity (3 seeds)"
echo ""

JOB_ABL_CONV=$(sbatch --parsable --export=ALL,ABLATION=no_convergence jobs/run_ablations_verified.sh)
echo "   Submitted: no_convergence (Job $JOB_ABL_CONV)"

JOB_ABL_ROOT=$(sbatch --parsable --export=ALL,ABLATION=no_root_learner jobs/run_ablations_verified.sh)
echo "   Submitted: no_root_learner (Job $JOB_ABL_ROOT)"

JOB_ABL_DIV=$(sbatch --parsable --export=ALL,ABLATION=no_diversity jobs/run_ablations_verified.sh)
echo "   Submitted: no_diversity (Job $JOB_ABL_DIV)"

echo ""

# ============================================================
# 2. NO-ORACLE ACE (1 job, 5 seeds sequential, ~15 hours)
# ============================================================
echo "2. Submitting NO-ORACLE ACE (5 seeds)..."
JOB_NO_ORACLE=$(sbatch --parsable jobs/run_ace_no_oracle.sh)
echo "   Submitted: Job $JOB_NO_ORACLE"
echo ""

# ============================================================
# 3. COMPLEX 15-NODE SCM (1 job, 1 seed, ~6-8 hours)
# ============================================================
echo "3. Submitting COMPLEX 15-NODE SCM ACE (seed 42)..."
JOB_COMPLEX=$(sbatch --parsable jobs/run_ace_complex_single_seed.sh)
echo "   Submitted: Job $JOB_COMPLEX"
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Ablation (no_convergence):  $JOB_ABL_CONV"
echo "  Ablation (no_root_learner): $JOB_ABL_ROOT"
echo "  Ablation (no_diversity):    $JOB_ABL_DIV"
echo "  No-Oracle ACE:              $JOB_NO_ORACLE"
echo "  Complex 15-Node SCM:        $JOB_COMPLEX"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Expected completion: 15-18 hours"
echo "============================================================"
