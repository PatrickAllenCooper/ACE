#!/usr/bin/env python3
"""
Static verification of PPO complex SCM implementation.
Tests without requiring GPU/ML dependencies.
"""

import sys


def test_ppo_function_exists():
    """Test 1: run_complex_ppo function exists."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    return "def run_complex_ppo(" in content


def test_ppo_runs_first():
    """Test 2: PPO runs before other baselines."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Find indices of method calls
    ppo_idx = content.find('print("\\nRunning PPO..."')
    if ppo_idx == -1:
        ppo_idx = content.find("print('\\nRunning PPO...")
    
    random_idx = content.find('print("\\nRunning Random..."', ppo_idx + 1)
    if random_idx == -1:
        random_idx = content.find("print('\\nRunning Random...", ppo_idx + 1)
    
    return ppo_idx != -1 and random_idx != -1 and ppo_idx < random_idx


def test_ppo_saves_immediately():
    """Test 3: PPO results saved immediately."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Check for save after PPO
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Running PPO' in line:
            next_20 = '\n'.join(lines[i:i+20])
            if '.to_csv' in next_20 and ('ppo' in next_20.lower() or 'PPO complete' in next_20):
                return True
    return False


def test_ppo_imports_from_baselines():
    """Test 4: PPO imports PPOPolicy from baselines."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Check for import in run_complex_ppo
    lines = content.split('\n')
    in_ppo_fn = False
    for line in lines:
        if 'def run_complex_ppo(' in line:
            in_ppo_fn = True
        if in_ppo_fn and 'def ' in line and 'run_complex_ppo' not in line:
            break
        if in_ppo_fn and 'from baselines import PPOPolicy' in line:
            return True
    return False


def test_incremental_saves_all_experiments():
    """Test 5: All experiment functions save incrementally."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Check extended baselines
    has_ext_save = 'extended_baselines_summary.csv' in content
    # Check lookahead
    has_look_save = 'lookahead_ablation_summary.csv' in content
    # Check complex
    has_complex_save = 'complex_scm_summary.csv' in content
    
    return has_ext_save and has_look_save and has_complex_save


def test_complex_scm_job_config():
    """Test 6: Complex SCM job properly configured."""
    with open("jobs/run_complex_scm_only.sh") as f:
        content = f.read()
    
    checks = [
        "#SBATCH --qos=normal",
        "#SBATCH --time=10:00:00",
        "python -u scripts/runners/run_critical_experiments.py",
        "--complex-scm",
        "cd $SLURM_SUBMIT_DIR"
    ]
    
    return all(check in content for check in checks)


def test_ppo_uses_correct_state_dim():
    """Test 7: PPO state dimension matches complex SCM."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Check for state_dim calculation
    lines = content.split('\n')
    in_ppo_fn = False
    for line in lines:
        if 'def run_complex_ppo(' in line:
            in_ppo_fn = True
        if in_ppo_fn and 'state_dim = len(scm.nodes) * 2' in line:
            return True
        if in_ppo_fn and 'def ' in line and 'run_complex_ppo' not in line:
            break
    return False


def test_ppo_returns_dict():
    """Test 8: PPO returns losses dict like other methods."""
    with open("scripts/runners/run_critical_experiments.py") as f:
        content = f.read()
    
    # Check return statement
    lines = content.split('\n')
    in_ppo_fn = False
    for line in lines:
        if 'def run_complex_ppo(' in line:
            in_ppo_fn = True
        if in_ppo_fn and 'return learner.evaluate()' in line:
            return True
        if in_ppo_fn and 'def ' in line and 'run_complex_ppo' not in line:
            in_ppo_fn = False
    return False


def main():
    tests = [
        ("PPO function exists", test_ppo_function_exists),
        ("PPO runs first", test_ppo_runs_first),
        ("PPO saves immediately", test_ppo_saves_immediately),
        ("PPO imports from baselines", test_ppo_imports_from_baselines),
        ("Incremental saves all experiments", test_incremental_saves_all_experiments),
        ("Complex SCM job configured", test_complex_scm_job_config),
        ("PPO state dimension correct", test_ppo_uses_correct_state_dim),
        ("PPO returns dict", test_ppo_returns_dict),
    ]
    
    print("="*70)
    print("PPO COMPLEX SCM VERIFICATION")
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
        print("[PASS] All PPO verification tests passed!")
        print("Complex SCM with PPO is ready for execution")
        print("="*70)
        return 0
    else:
        print(f"[FAIL] {errors} test(s) failed")
        print("Fix errors before HPC submission")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
