#!/usr/bin/env python3
"""
Comprehensive verification of STRONG ACCEPT experiment scripts.
Tests everything without requiring GPU execution.
"""

import os
import sys
from pathlib import Path


def test_scripts_exist():
    """Test 1: Required scripts exist."""
    required = [
        "jobs/run_remaining_ablations.sh",
        "jobs/run_ace_no_oracle.sh",
        "jobs/workflows/submit_strong_accept_experiments.sh"
    ]
    return all(os.path.exists(f) for f in required)


def test_qos_specified():
    """Test 2: QoS specified in job scripts."""
    scripts = ["jobs/run_remaining_ablations.sh", "jobs/run_ace_no_oracle.sh"]
    for script in scripts:
        with open(script) as f:
            if "#SBATCH --qos=normal" not in f.read():
                return False
    return True


def test_ablation_logic_exists():
    """Test 3: Ablation logic implemented."""
    with open("ace_experiments.py") as f:
        content = f.read()
    required = [
        "if args.no_per_node_convergence:",
        "if args.no_dedicated_root_learner:",
        "if args.no_diversity_reward:"
    ]
    return all(logic in content for logic in required)


def test_pretrain_flag_exists():
    """Test 4: Pretrain_steps flag exists."""
    with open("ace_experiments.py") as f:
        content = f.read()
    return '"--pretrain_steps"' in content or "'--pretrain_steps'" in content


def test_unbuffered_output():
    """Test 5: Scripts use python -u."""
    scripts = ["jobs/run_remaining_ablations.sh", "jobs/run_ace_no_oracle.sh"]
    for script in scripts:
        with open(script) as f:
            if "python -u ace_experiments.py" not in f.read():
                return False
    return True


def test_working_directory():
    """Test 6: Scripts use SUBMIT_DIR not scratch."""
    scripts = ["jobs/run_remaining_ablations.sh", "jobs/run_ace_no_oracle.sh"]
    for script in scripts:
        with open(script) as f:
            content = f.read()
            if "cd $SLURM_SUBMIT_DIR" not in content:
                return False
            # Should NOT cd to scratch before python
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'python' in line and 'ace_experiments.py' in line:
                    prev_5 = '\n'.join(lines[max(0,i-5):i])
                    if 'cd.*scratch' in prev_5.lower() and 'SUBMIT_DIR' not in prev_5:
                        return False
    return True


def test_time_limits():
    """Test 7: Adequate time limits."""
    scripts = {
        "jobs/run_remaining_ablations.sh": 8,
        "jobs/run_ace_no_oracle.sh": 8
    }
    for script, min_hours in scripts.items():
        with open(script) as f:
            content = f.read()
            if f"#SBATCH --time=0{min_hours}:00:00" not in content:
                return False
    return True


def test_output_directories():
    """Test 8: Output directories specified."""
    scripts = ["jobs/run_remaining_ablations.sh", "jobs/run_ace_no_oracle.sh"]
    for script in scripts:
        with open(script) as f:
            if "BASE_OUTPUT=" not in f.read():
                return False
    return True


def test_ablation_flags_match():
    """Test 9: Ablation flags in script match implemented flags."""
    with open("jobs/run_remaining_ablations.sh") as f:
        content = f.read()
    
    flag_mapping = {
        "no_convergence": "--no_per_node_convergence",
        "no_root_learner": "--no_dedicated_root_learner",
        "no_diversity": "--no_diversity_reward"
    }
    
    for ablation, flag in flag_mapping.items():
        if ablation not in content or flag not in content:
            return False
    return True


def test_no_oracle_sets_zero():
    """Test 10: No-oracle script sets pretrain_steps=0."""
    with open("jobs/run_ace_no_oracle.sh") as f:
        content = f.read()
    return "--pretrain_steps 0" in content


def main():
    tests = [
        ("Scripts exist", test_scripts_exist),
        ("QoS specified", test_qos_specified),
        ("Ablation logic exists", test_ablation_logic_exists),
        ("Pretrain flag exists", test_pretrain_flag_exists),
        ("Unbuffered output", test_unbuffered_output),
        ("Working directory", test_working_directory),
        ("Time limits", test_time_limits),
        ("Output directories", test_output_directories),
        ("Ablation flags match", test_ablation_flags_match),
        ("No-oracle sets zero", test_no_oracle_sets_zero),
    ]
    
    print("="*70)
    print("STRONG ACCEPT SCRIPTS VERIFICATION")
    print("="*70)
    
    errors = 0
    for name, test_func in tests:
        try:
            passed = test_func()
            status = "[PASS]" if passed else "[FAIL]"
            padded = name + "." * (50 - len(name))
            print(padded + " " + status)
            if not passed:
                errors += 1
        except Exception as e:
            padded = name + "." * (50 - len(name))
            print(padded + " [ERROR]: " + str(e))
            errors += 1
    
    print("="*70)
    if errors == 0:
        print("[PASS] All verification tests passed!")
        print("Scripts are ready for HPC execution")
        print("="*70)
        return 0
    else:
        print(f"[FAIL] {errors} test(s) failed")
        print("Fix errors before HPC submission")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
