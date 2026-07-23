#!/bin/bash
# =============================================================================
# Quick verification that the gradient-checkpointing requires_grad fix works.
# Submits ONE small anon30 ACE job (5 episodes, 30 min wall) so we get a fast
# read on whether DPO is actually backpropagating through the model.
#
# Success criteria (check in the .err log when the job finishes):
#   1. NO occurrence of:
#        UserWarning: None of the inputs have requires_grad=True
#   2. NO occurrence of:
#        [WARNING] DPO loss near random chance (0.693)
#      after the first ~5-10 DPO updates.
#   3. node_losses.csv exists and total_loss decreases over the 5 episodes.
#
# If any of (1)-(3) fail, the fix is incomplete; do NOT launch the full
# 11-job clean resubmit until this passes.
#
# Usage (after `git pull` on CURC):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_verify_dpo_fix.sh
#   # Wait for it to start; tail the .err
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

# Override the seed worker's episode budget for a quick smoke run.
# We do this by writing a tiny wrapper that exports a different EPISODES.
WRAPPER="$OUT/_verify_wrapper_$(date +%s).sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/paco0228/ACE
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT}/anon30/ace/seed_42_VERIFY/job_${JOB_TAG}"
mkdir -p "$OUT_DIR"
python -u ace_experiments.py \
    --large_scale 30 \
    --anonymize_nodes \
    --episodes 5 \
    --seed 42 \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT_DIR"
EOF
chmod +x "$WRAPPER"

JOB=$(sbatch --parsable \
    --job-name="ace_verify" \
    --partition=aa100 --qos=normal \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G \
    --time=00:45:00 \
    --output="$OUT/logs/verify_dpo_fix_%j.out" \
    --error="$OUT/logs/verify_dpo_fix_%j.err" \
    --export=ALL,OUT=$OUT \
    "$WRAPPER")
echo "Submitted verification: Job $JOB"
echo ""
echo "Once it starts, tail the log:"
echo "  tail -f $OUT/logs/verify_dpo_fix_${JOB}.err"
echo ""
echo "After completion, check:"
echo "  grep -E 'requires_grad|0.693|near random' $OUT/logs/verify_dpo_fix_${JOB}.err"
echo "  -> empty output = fix works"
