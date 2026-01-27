#!/usr/bin/env python3
"""
Verification script for ablation setup.
Tests everything before HPC execution.
"""

import subprocess
import sys
import os
from pathlib import Path


def test_runner_help():
    """Test 1: Runner script has help."""
    result = subprocess.run(
        [sys.executable, "scripts/runners/run_ablations_fast.py", "--help"],
        capture_output=True,
        timeout=10
    )
    return result.returncode == 0


def test_valid_ablation_types():
    """Test 2: All ablation types accepted."""
    result = subprocess.run(
        [sys.executable, "scripts/runners/run_ablations_fast.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    required = ['no_dpo', 'no_convergence', 'no_root_learner', 'no_diversity']
    return all(abl in result.stdout for abl in required)


def test_ace_ablation_flags():
    """Test 3: Main script has ablation flags."""
    try:
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # If imports fail, check source code directly
        if result.returncode != 0 and 'torch' in result.stderr:
            content = Path("ace_experiments.py").read_text()
            required = ['--no_per_node_convergence', '--no_dedicated_root_learner', 
                       '--no_diversity_reward', '--custom']
            return all(('"' + flag + '"' in content or "'" + flag + "'" in content) for flag in required)
        
        required = ['--no_per_node_convergence', '--no_dedicated_root_learner', 
                    '--no_diversity_reward', '--custom']
        return all(flag in result.stdout for flag in required)
    except Exception:
        # Fallback: check source
        content = Path("ace_experiments.py").read_text()
        return '--no_per_node_convergence' in content


def test_job_script_qos():
    """Test 4: Job script has QoS."""
    script_path = Path("jobs/run_ablations_fast.sh")
    if not script_path.exists():
        return False
    
    content = script_path.read_text()
    return "#SBATCH --qos=" in content and "normal" in content


def test_job_script_path():
    """Test 5: Job script calls correct Python file."""
    script_path = Path("jobs/run_ablations_fast.sh")
    if not script_path.exists():
        return False
    
    content = script_path.read_text()
    return "scripts/runners/run_ablations_fast.py" in content


def test_scratch_script_import_fix():
    """Test 6: Scratch script stays in submit dir."""
    script_path = Path("jobs/run_ablations_scratch.sh")
    if not script_path.exists():
        return False
    
    content = script_path.read_text()
    lines = content.split('\n')
    
    # Find Python line
    python_idx = None
    for i, line in enumerate(lines):
        if 'python ace_experiments.py' in line and not line.strip().startswith('#'):
            python_idx = i
            break
    
    if python_idx is None:
        return False
    
    # Check previous 5 lines for cd SUBMIT_DIR
    previous = '\n'.join(lines[max(0, python_idx-5):python_idx])
    return 'cd "$SLURM_SUBMIT_DIR"' in previous or 'cd $SLURM_SUBMIT_DIR' in previous


def test_ablation_logic_implemented():
    """Test 7: Ablation logic exists in ace_experiments.py."""
    content = Path("ace_experiments.py").read_text()
    
    required = [
        'if args.no_diversity_reward:',
        'if args.no_dedicated_root_learner:',
        'if args.no_per_node_convergence:',
    ]
    
    return all(logic in content for logic in required)


def main():
    tests = [
        ("Runner script help", test_runner_help),
        ("Valid ablation types", test_valid_ablation_types),
        ("ACE ablation flags", test_ace_ablation_flags),
        ("Job script QoS", test_job_script_qos),
        ("Job script path", test_job_script_path),
        ("Scratch script fix", test_scratch_script_import_fix),
        ("Ablation logic", test_ablation_logic_implemented),
    ]
    
    print("="*70)
    print("ABLATION SYSTEM VERIFICATION")
    print("="*70)
    
    errors = 0
    for name, test_func in tests:
        try:
            passed = test_func()
            status = "[PASS]" if passed else "[FAIL]"
            padded_name = name + "." * (50 - len(name))
            print(padded_name + " " + status)
            if not passed:
                errors += 1
        except Exception as e:
            padded_name = name + "." * (50 - len(name))
            print(padded_name + " [ERROR]: " + str(e))
            errors += 1
    
    print("="*70)
    if errors == 0:
        print("[PASS] All verification tests passed!")
        print("System ready for HPC execution")
        print("="*70)
        return 0
    else:
        print("[FAIL] " + str(errors) + " test(s) failed")
        print("Fix errors before HPC submission")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
