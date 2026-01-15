#!/usr/bin/env python3
"""
Run All Experiments for ACE Paper

This script orchestrates the complete experimental pipeline:
1. ACE (DPO) - Main experiment with LLM policy
2. Baselines - Random, Round-Robin, Max-Variance, PPO
3. Comparison Analysis - Generate figures and summary tables

Usage:
    # Full paper experiments (takes several hours)
    python run_all.py --full
    
    # Quick validation (few minutes)
    python run_all.py --quick
    
    # Custom configuration
    python run_all.py --ace_episodes 100 --baseline_episodes 50

Output:
    results/paper_run_YYYYMMDD_HHMMSS/
    ├── ace/                    # ACE (DPO) results
    │   ├── mechanism_contrast.png
    │   ├── training_curves.png
    │   ├── metrics.csv
    │   └── ...
    ├── baselines/              # Baseline comparison results
    │   ├── random_results.csv
    │   ├── round_robin_results.csv
    │   ├── max_variance_results.csv
    │   ├── ppo_results.csv
    │   └── baseline_comparison.png
    ├── comparison/             # Cross-method analysis
    │   ├── final_comparison.png
    │   ├── convergence_curves.png
    │   └── summary_table.csv
    └── paper_summary.txt       # Human-readable summary
"""

import argparse
import os
import sys
import subprocess
import shutil
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def setup_logging(log_file: str):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_command(cmd: list, description: str, timeout: int = None) -> bool:
    """Run a command and log output."""
    logging.info(f"Starting: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            logging.info(f"✓ Completed: {description}")
            if result.stdout:
                # Log last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    logging.info(f"  {line}")
            return True
        else:
            logging.error(f"✗ Failed: {description}")
            logging.error(f"  Exit code: {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-10:]:
                    logging.error(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"✗ Timeout: {description}")
        return False
    except Exception as e:
        logging.error(f"✗ Error: {description}: {e}")
        return False


def find_latest_run(base_dir: str, prefix: str) -> str:
    """Find the most recent run directory with given prefix."""
    if not os.path.exists(base_dir):
        return None
    
    dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not dirs:
        return None
    
    dirs.sort(reverse=True)  # Most recent first (timestamp-based names)
    return os.path.join(base_dir, dirs[0])


def generate_comparison_figures(ace_dir: str, baselines_dir: str, output_dir: str):
    """Generate cross-method comparison figures."""
    logging.info("Generating comparison figures...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ACE results
    ace_metrics = os.path.join(ace_dir, "node_losses.csv")
    if os.path.exists(ace_metrics):
        ace_df = pd.read_csv(ace_metrics)
        ace_df["method"] = "ACE (DPO)"
    else:
        logging.warning("ACE metrics not found, skipping ACE in comparison")
        ace_df = None
    
    # Load baseline results
    baseline_dfs = {}
    baseline_names = ["random", "round_robin", "max_variance", "ppo"]
    
    for name in baseline_names:
        path = os.path.join(baselines_dir, f"{name}_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            baseline_dfs[name] = df
            
    if not baseline_dfs:
        logging.warning("No baseline results found")
        return
        
    # --- Figure 1: Convergence Curves ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Mechanism Learning: Convergence Comparison", fontsize=14)
    
    colors = {
        "ACE (DPO)": "purple",
        "random": "red",
        "round_robin": "blue", 
        "max_variance": "green",
        "ppo": "orange"
    }
    
    display_names = {
        "random": "Random",
        "round_robin": "Round-Robin",
        "max_variance": "Max-Variance",
        "ppo": "PPO"
    }
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    
    # Total loss convergence
    ax = axes[0, 0]
    for name, df in baseline_dfs.items():
        mean_loss = df.groupby("step")["total_loss"].mean()
        ax.plot(mean_loss.index, mean_loss.values, 
                label=display_names.get(name, name), 
                color=colors.get(name, "gray"))
    
    if ace_df is not None:
        # ACE uses episode-step structure, aggregate by step
        ace_mean = ace_df.groupby("step")["total_loss"].mean()
        ax.plot(ace_mean.index, ace_mean.values, 
                label="ACE (DPO)", color=colors["ACE (DPO)"], linewidth=2)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Total MSE")
    ax.set_title("Total Loss Convergence")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    
    # Per-node loss
    for idx, node in enumerate(nodes):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        
        for name, df in baseline_dfs.items():
            col = f"loss_{node}"
            if col in df.columns:
                mean_loss = df.groupby("step")[col].mean()
                ax.plot(mean_loss.index, mean_loss.values,
                        label=display_names.get(name, name),
                        color=colors.get(name, "gray"))
        
        if ace_df is not None and f"loss_{node}" in ace_df.columns:
            mean_loss = ace_df.groupby("step")[f"loss_{node}"].mean()
            ax.plot(mean_loss.index, mean_loss.values,
                    label="ACE (DPO)", color=colors["ACE (DPO)"], linewidth=2)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.set_title(f"{node} Mechanism")
        ax.legend(fontsize=7)
        ax.set_yscale("log")
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Target')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_curves.png"), dpi=150)
    plt.close()
    logging.info(f"  Saved convergence_curves.png")
    
    # --- Figure 2: Final Performance Bar Chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect final losses
    final_data = []
    
    for name, df in baseline_dfs.items():
        final_step = df["step"].max()
        final_df = df[df["step"] == final_step]
        final_data.append({
            "method": display_names.get(name, name),
            "total_loss": final_df["total_loss"].mean(),
            "X3_loss": final_df["loss_X3"].mean() if "loss_X3" in final_df.columns else np.nan,
            "color": colors.get(name, "gray")
        })
    
    if ace_df is not None:
        final_step = ace_df["step"].max()
        final_df = ace_df[ace_df["step"] == final_step]
        final_data.append({
            "method": "ACE (DPO)",
            "total_loss": final_df["total_loss"].mean(),
            "X3_loss": final_df["loss_X3"].mean() if "loss_X3" in final_df.columns else np.nan,
            "color": colors["ACE (DPO)"]
        })
    
    final_df = pd.DataFrame(final_data)
    
    # Total loss bar chart
    ax = axes[0]
    bars = ax.bar(final_df["method"], final_df["total_loss"], 
                  color=final_df["color"])
    ax.set_ylabel("Final Total MSE")
    ax.set_title("Final Total Loss (Lower is Better)")
    ax.tick_params(axis='x', rotation=45)
    
    # X3 (collider) loss bar chart
    ax = axes[1]
    bars = ax.bar(final_df["method"], final_df["X3_loss"],
                  color=final_df["color"])
    ax.axhline(y=0.5, color='black', linestyle='--', label='Target (0.5)')
    ax.set_ylabel("Final X3 MSE")
    ax.set_title("Collider (X3) Learning (Target < 0.5)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_comparison.png"), dpi=150)
    plt.close()
    logging.info(f"  Saved final_comparison.png")
    
    # --- Summary Table ---
    summary_rows = []
    for name, df in baseline_dfs.items():
        final_step = df["step"].max()
        final_df = df[df["step"] == final_step]
        
        row = {
            "Method": display_names.get(name, name),
            "Total Loss": f"{final_df['total_loss'].mean():.3f} ± {final_df['total_loss'].std():.3f}",
        }
        for node in nodes:
            col = f"loss_{node}"
            if col in final_df.columns:
                mean = final_df[col].mean()
                row[f"{node} Loss"] = f"{mean:.3f}"
                row[f"{node} Pass"] = "✓" if mean < 0.5 else "✗"
        summary_rows.append(row)
    
    if ace_df is not None:
        final_step = ace_df["step"].max()
        final_df = ace_df[ace_df["step"] == final_step]
        row = {
            "Method": "ACE (DPO)",
            "Total Loss": f"{final_df['total_loss'].mean():.3f} ± {final_df['total_loss'].std():.3f}",
        }
        for node in nodes:
            col = f"loss_{node}"
            if col in final_df.columns:
                mean = final_df[col].mean()
                row[f"{node} Loss"] = f"{mean:.3f}"
                row[f"{node} Pass"] = "✓" if mean < 0.5 else "✗"
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    logging.info(f"  Saved summary_table.csv")
    
    return summary_df


def generate_paper_summary(run_dir: str, ace_dir: str, baselines_dir: str, 
                           summary_df: pd.DataFrame, duration: str):
    """Generate human-readable summary for paper."""
    
    summary_path = os.path.join(run_dir, "paper_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ACE PAPER EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Total Duration: {duration}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("Ground Truth SCM:\n")
        f.write("  X1 ~ N(0, 1)           [Root]\n")
        f.write("  X4 ~ N(2, 1)           [Root]\n")
        f.write("  X2 = 2*X1 + 1          [Linear]\n")
        f.write("  X3 = 0.5*X1 - X2 + sin(X2)  [Collider - KEY TARGET]\n")
        f.write("  X5 = 0.2*X4^2          [Nonlinear]\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 70 + "\n\n")
        
        if summary_df is not None:
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
        f.write("-" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("Success Criteria:\n")
        f.write("  1. X3 (Collider) Loss < 0.5\n")
        f.write("  2. Balanced intervention distribution (not >40% on single node)\n")
        f.write("  3. DPO loss decreasing from 0.693\n\n")
        
        f.write("Paper Claims to Validate:\n")
        f.write("  1. ACE (DPO) outperforms Random baseline\n")
        f.write("  2. ACE (DPO) outperforms Round-Robin systematic sampling\n")
        f.write("  3. ACE (DPO) outperforms Max-Variance greedy active learning\n")
        f.write("  4. ACE (DPO) outperforms PPO value-based RL\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("ACE Results:\n")
        if os.path.exists(ace_dir):
            for fname in sorted(os.listdir(ace_dir)):
                f.write(f"  - {fname}\n")
        f.write("\n")
        
        f.write("Baseline Results:\n")
        if os.path.exists(baselines_dir):
            for fname in sorted(os.listdir(baselines_dir)):
                f.write(f"  - {fname}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 70 + "\n")
    
    logging.info(f"Saved paper summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all ACE paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --quick              # Fast validation (~5 min)
  python run_all.py --full               # Full paper experiments (~2-4 hours)
  python run_all.py --ace_episodes 200   # Custom ACE episodes
        """
    )
    
    # Presets
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation run (few minutes)")
    parser.add_argument("--full", action="store_true",
                        help="Full paper experiments (several hours)")
    
    # ACE configuration
    parser.add_argument("--ace_episodes", type=int, default=None,
                        help="Number of ACE episodes (default: 50 quick, 500 full)")
    parser.add_argument("--ace_steps", type=int, default=25,
                        help="Steps per ACE episode")
    parser.add_argument("--use_custom", action="store_true",
                        help="Use custom transformer instead of LLM (faster, less capable)")
    
    # Baseline configuration
    parser.add_argument("--baseline_episodes", type=int, default=None,
                        help="Number of baseline episodes (default: 20 quick, 100 full)")
    parser.add_argument("--skip_ppo", action="store_true",
                        help="Skip PPO baseline (faster)")
    
    # Output
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Timeout per experiment in seconds (default: 4 hours)")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.quick:
        args.ace_episodes = args.ace_episodes or 10
        args.baseline_episodes = args.baseline_episodes or 10
        args.use_custom = True
    elif args.full:
        args.ace_episodes = args.ace_episodes or 500
        args.baseline_episodes = args.baseline_episodes or 100
    else:
        args.ace_episodes = args.ace_episodes or 50
        args.baseline_episodes = args.baseline_episodes or 50
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"paper_run_{timestamp}")
    ace_output = os.path.join(run_dir, "ace")
    baselines_output = os.path.join(run_dir, "baselines")
    comparison_output = os.path.join(run_dir, "comparison")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ace_output, exist_ok=True)
    os.makedirs(baselines_output, exist_ok=True)
    os.makedirs(comparison_output, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(run_dir, "run_all.log"))
    
    start_time = datetime.now()
    logging.info("=" * 70)
    logging.info("ACE PAPER EXPERIMENTS - STARTING")
    logging.info("=" * 70)
    logging.info(f"Output directory: {run_dir}")
    logging.info(f"ACE episodes: {args.ace_episodes}")
    logging.info(f"Baseline episodes: {args.baseline_episodes}")
    logging.info(f"Use custom transformer: {args.use_custom}")
    logging.info(f"Skip PPO: {args.skip_ppo}")
    
    results = {"ace": False, "baselines": False}
    
    # --- 1. Run ACE (DPO) Experiment ---
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 1: ACE (DPO) EXPERIMENT")
    logging.info("=" * 70)
    
    ace_cmd = [
        sys.executable, "ace_experiments.py",
        "--episodes", str(args.ace_episodes),
        "--steps", str(args.ace_steps),
        "--output", ace_output,
    ]
    
    if args.use_custom:
        ace_cmd.append("--custom")
    
    results["ace"] = run_command(
        ace_cmd,
        f"ACE experiment ({args.ace_episodes} episodes)",
        timeout=args.timeout
    )
    
    # Find the actual run directory (ace_experiments creates timestamped subdirs)
    ace_run_dir = find_latest_run(ace_output, "run_")
    if ace_run_dir:
        logging.info(f"ACE results in: {ace_run_dir}")
    
    # --- 2. Run Baselines ---
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 2: BASELINE EXPERIMENTS")
    logging.info("=" * 70)
    
    baseline_cmd = [
        sys.executable, "baselines.py",
        "--episodes", str(args.baseline_episodes),
        "--steps", str(args.ace_steps),
        "--output", baselines_output,
    ]
    
    if args.skip_ppo:
        baseline_cmd.append("--all")
    else:
        baseline_cmd.append("--all_with_ppo")
    
    results["baselines"] = run_command(
        baseline_cmd,
        f"Baseline experiments ({args.baseline_episodes} episodes)",
        timeout=args.timeout
    )
    
    # Find the actual baselines directory
    baselines_run_dir = find_latest_run(baselines_output, "baselines_")
    if baselines_run_dir:
        logging.info(f"Baseline results in: {baselines_run_dir}")
    
    # --- 3. Generate Comparison Figures ---
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 3: COMPARISON ANALYSIS")
    logging.info("=" * 70)
    
    summary_df = None
    if ace_run_dir and baselines_run_dir:
        try:
            summary_df = generate_comparison_figures(
                ace_run_dir, baselines_run_dir, comparison_output
            )
        except Exception as e:
            logging.error(f"Failed to generate comparison figures: {e}")
    
    # --- 4. Generate Summary ---
    end_time = datetime.now()
    duration = str(end_time - start_time).split('.')[0]  # Remove microseconds
    
    generate_paper_summary(
        run_dir, 
        ace_run_dir or ace_output,
        baselines_run_dir or baselines_output,
        summary_df,
        duration
    )
    
    # --- Final Report ---
    logging.info("\n" + "=" * 70)
    logging.info("EXPERIMENT COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Total duration: {duration}")
    logging.info(f"Results saved to: {run_dir}")
    logging.info("")
    logging.info("Status:")
    logging.info(f"  ACE (DPO):  {'✓ Success' if results['ace'] else '✗ Failed'}")
    logging.info(f"  Baselines: {'✓ Success' if results['baselines'] else '✗ Failed'}")
    logging.info("")
    logging.info("Key outputs:")
    logging.info(f"  {os.path.join(comparison_output, 'convergence_curves.png')}")
    logging.info(f"  {os.path.join(comparison_output, 'final_comparison.png')}")
    logging.info(f"  {os.path.join(comparison_output, 'summary_table.csv')}")
    logging.info(f"  {os.path.join(run_dir, 'paper_summary.txt')}")
    
    # Return success if at least baselines ran
    return 0 if results["baselines"] else 1


if __name__ == "__main__":
    sys.exit(main())
