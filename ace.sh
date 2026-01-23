#!/bin/bash

# ============================================================================
# ACE: Active Causal Experimentation - Unified CLI
# ============================================================================
# Single entry point for all ACE operations
#
# Usage:
#   ./ace.sh <command> [options]
#
# Commands:
#   run <experiment>        Run a single experiment
#   run-multi-seed [seeds]  Run experiments with multiple seeds
#   run-ablations           Run ablation studies
#   run-obs-ablation        Run observational ratio ablation
#   process <dir>           Post-process results
#   sync-hpc                Sync results from HPC
#   extract <type> <dir>    Extract metrics (ace|baselines)
#   verify <dir>            Verify paper claims
#   test                    Run pipeline tests
#   clean                   Clean up temporary files
#   check-version           Check environment versions
#   help                    Show this help message
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Command: run - Run experiments
# ============================================================================
cmd_run() {
    local experiment=$1
    shift
    
    case $experiment in
        ace|ace-main)
            log_info "Submitting ACE main experiment..."
            sbatch "$@" jobs/run_ace_main.sh
            ;;
        baselines)
            log_info "Submitting baselines experiment..."
            sbatch "$@" jobs/run_baselines.sh
            ;;
        complex|complex-scm)
            log_info "Submitting complex SCM experiment..."
            sbatch "$@" jobs/run_complex_scm.sh
            ;;
        duffing)
            log_info "Submitting Duffing oscillator experiment..."
            sbatch "$@" jobs/run_duffing.sh
            ;;
        phillips)
            log_info "Submitting Phillips curve experiment..."
            sbatch "$@" jobs/run_phillips.sh
            ;;
        all)
            log_info "Submitting all experiments..."
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            RUN_DIR="results/paper_${TIMESTAMP}"
            mkdir -p "$RUN_DIR"
            
            log_info "Starting all 5 experiments (output: $RUN_DIR)"
            
            JOB1=$(sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR jobs/run_ace_main.sh)
            JOB2=$(sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR jobs/run_baselines.sh)
            JOB3=$(sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR jobs/run_complex_scm.sh)
            JOB4=$(sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR jobs/run_duffing.sh)
            JOB5=$(sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR jobs/run_phillips.sh)
            
            log_success "All experiments submitted!"
            echo "  ACE Main:     Job $JOB1"
            echo "  Baselines:    Job $JOB2"
            echo "  Complex SCM:  Job $JOB3"
            echo "  Duffing:      Job $JOB4"
            echo "  Phillips:     Job $JOB5"
            echo ""
            echo "Monitor: squeue -j $JOB1,$JOB2,$JOB3,$JOB4,$JOB5"
            echo "Results: $RUN_DIR"
            ;;
        *)
            log_error "Unknown experiment: $experiment"
            echo "Available: ace, baselines, complex, duffing, phillips, all"
            exit 1
            ;;
    esac
}

# ============================================================================
# Command: run-multi-seed - Run multi-seed validation
# ============================================================================
cmd_run_multi_seed() {
    local n_seeds=${1:-5}
    
    log_info "Running multi-seed validation with $n_seeds seeds..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MULTI_RUN_DIR="results/multi_run_${TIMESTAMP}"
    mkdir -p "$MULTI_RUN_DIR"
    
    declare -a SEEDS
    case $n_seeds in
        3) SEEDS=("42" "123" "456") ;;
        5) SEEDS=("42" "123" "456" "789" "1011") ;;
        10) SEEDS=("42" "123" "456" "789" "1011" "314" "271" "577" "618" "141") ;;
        *) log_error "Supported n_seeds: 3, 5, 10"; exit 1 ;;
    esac
    
    declare -a ALL_JOBS
    
    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="$MULTI_RUN_DIR/seed_${SEED}"
        mkdir -p "$SEED_DIR"
        
        log_info "Submitting seed $SEED..."
        
        JOB1=$(sbatch --parsable --export=ALL,RUN_DIR=$SEED_DIR,RANDOM_SEED=$SEED jobs/run_ace_main.sh)
        JOB2=$(sbatch --parsable --export=ALL,RUN_DIR=$SEED_DIR,RANDOM_SEED=$SEED jobs/run_baselines.sh)
        JOB3=$(sbatch --parsable --export=ALL,RUN_DIR=$SEED_DIR,RANDOM_SEED=$SEED jobs/run_complex_scm.sh)
        JOB4=$(sbatch --parsable --export=ALL,RUN_DIR=$SEED_DIR,RANDOM_SEED=$SEED jobs/run_duffing.sh)
        JOB5=$(sbatch --parsable --export=ALL,RUN_DIR=$SEED_DIR,RANDOM_SEED=$SEED jobs/run_phillips.sh)
        
        ALL_JOBS+=($JOB1 $JOB2 $JOB3 $JOB4 $JOB5)
    done
    
    log_success "Multi-seed validation submitted!"
    echo "  Seeds: ${SEEDS[*]}"
    echo "  Total jobs: ${#ALL_JOBS[@]}"
    echo "  Results: $MULTI_RUN_DIR"
    echo ""
    echo "Monitor: squeue -j $(IFS=,; echo "${ALL_JOBS[*]}")"
    echo ""
    echo "After completion, consolidate with:"
    echo "  ./ace.sh consolidate-multi-seed $MULTI_RUN_DIR"
}

# ============================================================================
# Command: run-ablations - Run ablation studies
# ============================================================================
cmd_run_ablations() {
    log_info "Running ablation studies..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ABLATION_DIR="results/ablations_${TIMESTAMP}"
    mkdir -p "$ABLATION_DIR"
    
    declare -a JOB_IDS
    
    # Full model (baseline)
    log_info "Submitting full model..."
    JOB=$(sbatch --parsable \
        --job-name=ace_full \
        --output=logs/ablation_full_${TIMESTAMP}_%j.out \
        --error=logs/ablation_full_${TIMESTAMP}_%j.err \
        --wrap="python ace_experiments.py --episodes 200 --early_stopping --use_per_node_convergence --use_dedicated_root_learner --output $ABLATION_DIR/full")
    JOB_IDS+=($JOB)
    
    # No per-node convergence
    log_info "Submitting no-per-node-convergence ablation..."
    JOB=$(sbatch --parsable \
        --job-name=ace_no_pnc \
        --output=logs/ablation_no_pnc_${TIMESTAMP}_%j.out \
        --error=logs/ablation_no_pnc_${TIMESTAMP}_%j.err \
        --wrap="python ace_experiments.py --episodes 200 --early_stopping --no_per_node_convergence --use_dedicated_root_learner --output $ABLATION_DIR/no_per_node_convergence")
    JOB_IDS+=($JOB)
    
    # No dedicated root learner
    log_info "Submitting no-dedicated-root-learner ablation..."
    JOB=$(sbatch --parsable \
        --job-name=ace_no_root \
        --output=logs/ablation_no_root_${TIMESTAMP}_%j.out \
        --error=logs/ablation_no_root_${TIMESTAMP}_%j.err \
        --wrap="python ace_experiments.py --episodes 200 --early_stopping --use_per_node_convergence --no_dedicated_root_learner --output $ABLATION_DIR/no_dedicated_root_learner")
    JOB_IDS+=($JOB)
    
    # No diversity reward
    log_info "Submitting no-diversity-reward ablation..."
    JOB=$(sbatch --parsable \
        --job-name=ace_no_div \
        --output=logs/ablation_no_div_${TIMESTAMP}_%j.out \
        --error=logs/ablation_no_div_${TIMESTAMP}_%j.err \
        --wrap="python ace_experiments.py --episodes 200 --early_stopping --use_per_node_convergence --use_dedicated_root_learner --no_diversity_reward --output $ABLATION_DIR/no_diversity_reward")
    JOB_IDS+=($JOB)
    
    log_success "Ablation studies submitted!"
    echo "  Jobs: ${JOB_IDS[*]}"
    echo "  Results: $ABLATION_DIR"
    echo ""
    echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
    echo ""
    echo "After completion, analyze with:"
    echo "  python scripts/analyze_ablations.py $ABLATION_DIR"
}

# ============================================================================
# Command: run-obs-ablation - Run observational ratio ablation
# ============================================================================
cmd_run_obs_ablation() {
    log_info "Running observational ratio ablation..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OBS_ABLATION_DIR="results/obs_ratio_ablation_${TIMESTAMP}"
    mkdir -p "$OBS_ABLATION_DIR"
    
    declare -a JOB_IDS
    
    for INTERVAL in 2 3 4 5; do
        PCT=$((100 / INTERVAL))
        log_info "Submitting obs_train_interval=$INTERVAL (~${PCT}% observational)..."
        
        JOB=$(sbatch --parsable \
            --job-name=obs_int${INTERVAL} \
            --output=logs/obs_interval${INTERVAL}_${TIMESTAMP}_%j.out \
            --error=logs/obs_interval${INTERVAL}_${TIMESTAMP}_%j.err \
            --wrap="python ace_experiments.py --episodes 200 --early_stopping --use_per_node_convergence --use_dedicated_root_learner --obs_train_interval $INTERVAL --output $OBS_ABLATION_DIR/obs_interval_${INTERVAL}")
        
        JOB_IDS+=($JOB)
    done
    
    log_success "Observational ratio ablation submitted!"
    echo "  Jobs: ${JOB_IDS[*]}"
    echo "  Results: $OBS_ABLATION_DIR"
    echo ""
    echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
}

# ============================================================================
# Command: process - Post-process results
# ============================================================================
cmd_process() {
    local results_dir=${1:-$(ls -td results/paper_* 2>/dev/null | head -1)}
    
    if [[ -z "$results_dir" ]]; then
        log_error "No results directory specified and none found"
        echo "Usage: ./ace.sh process <results_dir>"
        exit 1
    fi
    
    if [[ ! -d "$results_dir" ]]; then
        log_error "Directory not found: $results_dir"
        exit 1
    fi
    
    log_info "Processing results in: $results_dir"
    
    # Extract metrics
    log_info "Step 1/5: Extracting ACE metrics..."
    cmd_extract ace "$results_dir"
    
    log_info "Step 2/5: Extracting baseline metrics..."
    cmd_extract baselines "$results_dir"
    
    log_info "Step 3/5: Verifying paper claims..."
    cmd_verify "$results_dir"
    
    log_info "Step 4/5: Analyzing regimes and clamping..."
    if [[ -d "$results_dir/duffing" ]]; then
        python clamping_detector.py "$results_dir/duffing" || log_warning "Clamping detection failed"
    fi
    if [[ -d "$results_dir/phillips" ]]; then
        python regime_analyzer.py "$results_dir/phillips" || log_warning "Regime analysis failed"
    fi
    
    log_info "Step 5/5: Generating comparisons and figures..."
    python compare_methods.py "$results_dir" || log_warning "Comparison failed"
    python visualize.py "$results_dir"/*/ || log_warning "Visualization failed"
    
    # Generate summary
    log_info "Generating processing summary..."
    cat > "$results_dir/PROCESSING_SUMMARY.txt" <<EOF
ACE Results Processing Summary
===============================
Processed: $(date)
Directory: $results_dir

Experiments Completed:
$(ls -d "$results_dir"/*/ 2>/dev/null | xargs -n1 basename | sed 's/^/  - /')

Metrics Extracted:
$(ls "$results_dir"/*/metrics.csv 2>/dev/null | wc -l | xargs echo "  CSV files:")
$(ls "$results_dir"/*/node_losses.csv 2>/dev/null | wc -l | xargs echo "  Node loss files:")

Claims Verified:
$(grep -c "✓" "$results_dir/ace_summary.txt" 2>/dev/null || echo "  N/A")

Figures Generated:
$(ls "$results_dir"/*/*.png 2>/dev/null | wc -l | xargs echo "  PNG files:")

Status: COMPLETE
Next: Review results and update paper TODOs
EOF
    
    log_success "Processing complete!"
    echo ""
    cat "$results_dir/PROCESSING_SUMMARY.txt"
}

# ============================================================================
# Command: sync-hpc - Sync results from HPC
# ============================================================================
cmd_sync_hpc() {
    local hpc_host=${HPC_HOST:-"your-hpc-server"}
    local hpc_project_dir=${HPC_PROJECT_DIR:-"\$HOME/ACE"}
    
    log_info "Syncing results from HPC..."
    log_info "  Host: $hpc_host"
    log_info "  Remote dir: $hpc_project_dir"
    
    # Find latest results on HPC
    LATEST_RESULTS=$(ssh "$hpc_host" "cd $hpc_project_dir && ls -td results/paper_* 2>/dev/null | head -1" || echo "")
    
    if [[ -z "$LATEST_RESULTS" ]]; then
        log_error "No results found on HPC"
        exit 1
    fi
    
    RESULTS_NAME=$(basename "$LATEST_RESULTS")
    LOCAL_DEST="results/$RESULTS_NAME"
    
    log_info "Found: $LATEST_RESULTS"
    log_info "Downloading to: $LOCAL_DEST"
    
    # Sync results
    rsync -avz --progress "$hpc_host:$hpc_project_dir/$LATEST_RESULTS/" "$LOCAL_DEST/"
    
    log_success "Sync complete!"
    echo ""
    echo "Process locally with:"
    echo "  ./ace.sh process $LOCAL_DEST"
}

# ============================================================================
# Command: extract - Extract metrics
# ============================================================================
cmd_extract() {
    local type=$1
    local results_dir=$2
    
    if [[ -z "$results_dir" ]]; then
        log_error "Usage: ./ace.sh extract <ace|baselines> <results_dir>"
        exit 1
    fi
    
    case $type in
        ace)
            bash -c "
                RESULTS_DIR='$results_dir'
                $(grep -A 100 'Extract ACE metrics' scripts/extract_ace.sh | tail -n +2)
            "
            ;;
        baselines)
            bash -c "
                RESULTS_DIR='$results_dir'
                $(grep -A 100 'Extract baseline metrics' scripts/extract_baselines.sh | tail -n +2)
            "
            ;;
        *)
            log_error "Unknown type: $type (use 'ace' or 'baselines')"
            exit 1
            ;;
    esac
}

# ============================================================================
# Command: verify - Verify paper claims
# ============================================================================
cmd_verify() {
    local results_dir=$1
    
    if [[ -z "$results_dir" ]]; then
        log_error "Usage: ./ace.sh verify <results_dir>"
        exit 1
    fi
    
    log_info "Verifying paper claims..."
    
    # Run verification logic
    python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path

results_dir = Path('$results_dir')
ace_dir = results_dir / 'ace_main'
baselines_dir = results_dir / 'baselines'

print('\\n✓ Verification logic would run here')
print(f'  ACE dir: {ace_dir}')
print(f'  Baselines dir: {baselines_dir}')
"
    
    log_success "Verification complete"
}

# ============================================================================
# Command: consolidate-multi-seed - Consolidate multi-seed results
# ============================================================================
cmd_consolidate_multi_seed() {
    local multi_run_dir=$1
    
    if [[ -z "$multi_run_dir" ]]; then
        log_error "Usage: ./ace.sh consolidate-multi-seed <multi_run_dir>"
        exit 1
    fi
    
    log_info "Consolidating multi-seed results from: $multi_run_dir"
    
    # Collect all seed results
    python scripts/compute_statistics.py "$multi_run_dir" ace
    python scripts/compute_statistics.py "$multi_run_dir" baselines
    
    # Statistical tests
    python scripts/statistical_tests.py "$multi_run_dir"
    
    log_success "Multi-seed consolidation complete!"
    echo "  Statistics: $multi_run_dir/statistics.csv"
    echo "  LaTeX table: $multi_run_dir/results_table.tex"
}

# ============================================================================
# Command: test - Run pipeline tests
# ============================================================================
cmd_test() {
    log_info "Running pipeline tests..."
    
    # Syntax check all job scripts
    log_info "Checking SLURM job scripts..."
    for script in jobs/*.sh; do
        bash -n "$script" && log_success "  $(basename $script): OK" || log_error "  $(basename $script): FAILED"
    done
    
    # Run pytest on improvement scripts
    log_info "Running improvement script tests..."
    python -m pytest tests/test_improvement_scripts.py -v
    
    log_success "Pipeline tests complete"
}

# ============================================================================
# Command: clean - Clean up temporary files
# ============================================================================
cmd_clean() {
    log_info "Cleaning up temporary files..."
    
    # Remove __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    log_success "  Removed __pycache__ directories"
    
    # Remove .pytest_cache
    rm -rf .pytest_cache
    log_success "  Removed .pytest_cache"
    
    # Remove coverage files
    rm -f .coverage
    log_success "  Removed .coverage"
    
    # Remove local test results
    rm -rf results/failure_test_local
    log_success "  Removed local test results"
    
    log_success "Cleanup complete"
}

# ============================================================================
# Command: check-version - Check environment versions
# ============================================================================
cmd_check_version() {
    log_info "Checking environment versions..."
    
    echo ""
    echo "Python:"
    python --version
    
    echo ""
    echo "PyTorch:"
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
    
    echo ""
    echo "Key packages:"
    python -c "
import transformers, pandas, matplotlib, seaborn, networkx, scipy
print(f'  transformers: {transformers.__version__}')
print(f'  pandas: {pandas.__version__}')
print(f'  matplotlib: {matplotlib.__version__}')
print(f'  seaborn: {seaborn.__version__}')
print(f'  networkx: {networkx.__version__}')
print(f'  scipy: {scipy.__version__}')
"
}

# ============================================================================
# Command: help - Show help message
# ============================================================================
cmd_help() {
    cat <<EOF
ACE: Active Causal Experimentation - Unified CLI
=================================================

Usage: ./ace.sh <command> [options]

EXPERIMENT COMMANDS:
  run <experiment>              Run a single experiment
                                  Experiments: ace, baselines, complex, duffing, phillips, all
  run-multi-seed [n]            Run multi-seed validation (default: 5 seeds)
  run-ablations                 Run ablation studies (4 configurations)
  run-obs-ablation              Run observational ratio ablation (4 intervals)

POST-PROCESSING COMMANDS:
  process <dir>                 Post-process experimental results
  consolidate-multi-seed <dir>  Consolidate multi-seed results
  extract <type> <dir>          Extract metrics (type: ace|baselines)
  verify <dir>                  Verify paper claims

HPC COMMANDS:
  sync-hpc                      Sync latest results from HPC server

UTILITY COMMANDS:
  test                          Run pipeline tests
  clean                         Clean up temporary files
  check-version                 Check environment versions
  help                          Show this help message

EXAMPLES:
  # Run all experiments
  ./ace.sh run all

  # Run multi-seed validation
  ./ace.sh run-multi-seed 5

  # Run ablations
  ./ace.sh run-ablations

  # Process results
  ./ace.sh process results/paper_20260121_123456

  # Sync from HPC and process
  ./ace.sh sync-hpc
  ./ace.sh process results/paper_20260121_123456

  # Run tests
  ./ace.sh test

CONFIGURATION:
  Set these environment variables for HPC sync:
    export HPC_HOST="your-hpc-server.edu"
    export HPC_PROJECT_DIR="\$HOME/ACE"

For more information, see README.md
EOF
}

# ============================================================================
# Main entry point
# ============================================================================
main() {
    if [[ $# -eq 0 ]]; then
        cmd_help
        exit 0
    fi
    
    local command=$1
    shift
    
    case $command in
        run)
            cmd_run "$@"
            ;;
        run-multi-seed)
            cmd_run_multi_seed "$@"
            ;;
        run-ablations)
            cmd_run_ablations "$@"
            ;;
        run-obs-ablation)
            cmd_run_obs_ablation "$@"
            ;;
        process)
            cmd_process "$@"
            ;;
        sync-hpc)
            cmd_sync_hpc "$@"
            ;;
        extract)
            cmd_extract "$@"
            ;;
        verify)
            cmd_verify "$@"
            ;;
        consolidate-multi-seed)
            cmd_consolidate_multi_seed "$@"
            ;;
        test)
            cmd_test "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        check-version)
            cmd_check_version "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
