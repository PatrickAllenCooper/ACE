#!/usr/bin/env python3
"""
Local execution of all remaining experiments.
Runs on single GPU with comprehensive logging and incremental saves.

Designed for NVIDIA RTX 3080 (10GB VRAM).
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class ExperimentLogger:
    """Comprehensive logging for local experiments."""
    
    def __init__(self, log_dir="results/local_experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.results = []
        
    def log(self, message, level="INFO"):
        """Log message to both file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line, flush=True)
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def log_result(self, experiment, seed, status, elapsed_time, output_dir=None):
        """Log experiment result."""
        result = {
            'experiment': experiment,
            'seed': seed,
            'status': status,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(output_dir) if output_dir else None
        }
        self.results.append(result)
        
        # Save results incrementally
        results_file = self.log_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Result: {experiment} seed {seed} - {status} ({elapsed_time:.1f}s)")
    
    def summary(self):
        """Print summary of all experiments."""
        self.log("="*70)
        self.log("EXPERIMENT SUMMARY")
        self.log("="*70)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in self.results if r['status'] == 'FAILED')
        
        self.log(f"Total experiments: {total}")
        self.log(f"Successful: {success}")
        self.log(f"Failed: {failed}")
        
        total_time = sum(r['elapsed_time'] for r in self.results)
        self.log(f"Total time: {total_time/3600:.2f} hours")
        self.log("="*70)


def run_experiment(cmd, logger, experiment_name, seed, timeout_hours=2):
    """Run single experiment with logging."""
    logger.log(f"Starting: {experiment_name} seed {seed}")
    logger.log(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_hours * 3600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.log_result(experiment_name, seed, 'SUCCESS', elapsed)
            logger.log(f"Success in {elapsed/60:.1f} minutes")
            return True
        else:
            logger.log_result(experiment_name, seed, 'FAILED', elapsed)
            logger.log(f"FAILED with code {result.returncode}", level="ERROR")
            logger.log(f"STDERR: {result.stderr[:500]}", level="ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.log_result(experiment_name, seed, 'TIMEOUT', elapsed)
        logger.log(f"TIMEOUT after {elapsed/3600:.1f} hours", level="ERROR")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.log_result(experiment_name, seed, 'ERROR', elapsed)
        logger.log(f"Exception: {e}", level="ERROR")
        return False


def main():
    logger = ExperimentLogger()
    
    logger.log("="*70)
    logger.log("LOCAL EXPERIMENT EXECUTION - ALL REMAINING EXPERIMENTS")
    logger.log("="*70)
    logger.log(f"GPU: NVIDIA RTX 3080")
    logger.log(f"Log file: {logger.log_file}")
    logger.log("="*70)
    
    experiments = []
    
    # 1. Complex SCM with PPO (PRIORITY)
    logger.log("\n[1/3] COMPLEX 15-NODE SCM WITH PPO")
    logger.log("5 seeds × 5 methods, estimated 20 hours")
    
    complex_cmd = [
        sys.executable, "-u",
        "scripts/runners/run_critical_experiments.py",
        "--complex-scm",
        "--seeds", "42", "123", "456", "789", "1011",
        "--output-dir", "results/critical_experiments_local"
    ]
    
    if run_experiment(complex_cmd, logger, "complex_scm", "all", timeout_hours=24):
        logger.log("Complex SCM COMPLETE")
    else:
        logger.log("Complex SCM FAILED - continuing with ablations", level="WARN")
    
    # 2. Remaining Ablations
    logger.log("\n[2/3] REMAINING ABLATIONS")
    logger.log("3 ablations × 3 seeds, estimated 4-5 hours")
    
    for ablation in ['no_convergence', 'no_root_learner', 'no_diversity']:
        flag_map = {
            'no_convergence': '--custom --no_per_node_convergence',
            'no_root_learner': '--custom --no_dedicated_root_learner',
            'no_diversity': '--custom --no_diversity_reward'
        }
        
        flags = flag_map[ablation].split()
        
        for seed in [42, 123, 456]:
            output_dir = f"results/ablations_complete/{ablation}/seed_{seed}"
            
            cmd = [
                sys.executable, "-u",
                "ace_experiments.py",
                "--episodes", "100",
                "--seed", str(seed),
                "--output", output_dir
            ] + flags + [
                "--early_stopping",
                "--early_stop_patience", "15",
                "--early_stop_min_episodes", "30"
            ]
            
            success = run_experiment(cmd, logger, f"ablation_{ablation}", seed, timeout_hours=2)
            
            if not success:
                logger.log(f"Ablation {ablation} seed {seed} failed", level="WARN")
    
    # 3. ACE Without Oracle
    logger.log("\n[3/3] ACE WITHOUT ORACLE")
    logger.log("5 seeds, estimated 5 hours")
    
    for seed in [42, 123, 456, 789, 1011]:
        output_dir = f"results/ace_no_oracle/seed_{seed}"
        
        cmd = [
            sys.executable, "-u",
            "ace_experiments.py",
            "--custom",
            "--episodes", "200",
            "--seed", str(seed),
            "--output", output_dir,
            "--pretrain_steps", "0",
            "--early_stopping",
            "--use_per_node_convergence"
        ]
        
        success = run_experiment(cmd, logger, "ace_no_oracle", seed, timeout_hours=2)
        
        if not success:
            logger.log(f"No-oracle seed {seed} failed", level="WARN")
    
    # Final summary
    logger.summary()
    logger.log(f"All results saved to: {logger.log_dir}")
    logger.log(f"Detailed log: {logger.log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
