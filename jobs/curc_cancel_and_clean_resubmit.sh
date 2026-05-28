#!/bin/bash
# =============================================================================
# Cancel the May 25 ace_a30c/ace_n50c clean batch (which would all OOM at
# anon30 due to longer prompts, and would all timeout at nodes50 due to
# 57 min/ep DPO at 50 nodes), then resubmit with the May 28 fixes:
#   - gradient checkpointing now also enabled at 30-node anon30 (OOM fix)
#   - nodes50 ACE episode budget reduced 120 -> 30 (fits in 20h wall time)
#   - anon30 ACE wall time bumped 14h -> 22h (checkpointing slowdown)
#
# Requires: ace_experiments.py with the May 28 gating fix (commit > b24f6f1).
#
# Usage:
#   cd /projects/paco0228/ACE
#   git pull origin main
#   bash jobs/curc_cancel_and_clean_resubmit.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

echo "================================================================"
echo " Cancelling stale May 25 clean batch (will all OOM/timeout)"
echo "================================================================"
TO_CANCEL=$(squeue -u "$USER" --format='%.10i %.20j' \
    | awk 'NR>1 && ($2 ~ /^ace_a30c_s/ || $2 ~ /^ace_n50c_s/) {print $1}')
if [ -z "$TO_CANCEL" ]; then
    echo "  (no ace_a30c / ace_n50c jobs in queue to cancel)"
else
    echo "$TO_CANCEL" | tr ' ' '\n' | sort -u | tr '\n' ' '
    echo ""
    echo "$TO_CANCEL" | tr ' ' '\n' | sort -u | xargs -r scancel
    echo "  cancelled."
fi

echo ""
echo "================================================================"
echo " Resubmitting clean batch with May 28 fixes"
echo "================================================================"
bash jobs/curc_clean_followup_resubmit.sh

echo ""
echo "Done. Next: verify a single job by waiting for first .out to appear,"
echo "  ls -lt results/curc_30node_followup/logs/ace_anon30_clean_seed42_*.out 2>/dev/null | head -1"
echo "and tail-following it for [PROGRESS] Episode 0/120 starting + non-OOM run."
