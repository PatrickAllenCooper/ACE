#!/usr/bin/env python3
"""
Static verification of ACE complex SCM implementation.
Tests without requiring GPU/execution.
"""

import sys
import os


def test_script_exists():
    """Test 1: run_ace_complex.py exists."""
    return os.path.exists("experiments/run_ace_complex.py")


def test_imports_declared():
    """Test 2: All necessary imports present."""
    with open("experiments/run_ace_complex.py") as f:
        content = f.read()
    
    required = [
        "ComplexGroundTruthSCM",
        "ComplexStudentSCM",
        "ComplexSCMLearner",
        "TransformerPolicy",
        "ExperimentalDSL",
        "EarlyStopping"
    ]
    
    return all(imp in content for imp in required)


def test_seed_setting():
    """Test 3: Random seeds are set."""
    with open("experiments/run_ace_complex.py") as f:
        content = f.read()
    
    return all(seed in content for seed in [
        "random.seed",
        "np.random.seed",
        "torch.manual_seed"
    ])


def test_early_stopping_used():
    """Test 4: Early stopping implemented."""
    with open("experiments/run_ace_complex.py") as f:
        content = f.read()
    
    return "EarlyStopping" in content and "early_stopper" in content


def test_results_saved():
    """Test 5: Results saved to CSV."""
    with open("experiments/run_ace_complex.py") as f:
        content = f.read()
    
    return ".to_csv" in content and "results.csv" in content


def test_custom_transformer_option():
    """Test 6: Custom transformer option available."""
    with open("experiments/run_ace_complex.py") as f:
        content = f.read()
    
    return "use_custom" in content and "TransformerPolicy" in content


def test_job_script_updated():
    """Test 7: Job script calls new script."""
    with open("jobs/run_ace_complex_scm.sh") as f:
        content = f.read()
    
    return "run_ace_complex.py" in content


def test_complex_scm_ace_added():
    """Test 8: ACE policy added to complex_scm.py choices."""
    with open("experiments/complex_scm.py") as f:
        content = f.read()
    
    # Should have ace in choices now
    return '"ace"' in content or "'ace'" in content


def main():
    tests = [
        ("Script exists", test_script_exists),
        ("All imports declared", test_imports_declared),
        ("Random seeds set", test_seed_setting),
        ("Early stopping used", test_early_stopping_used),
        ("Results saved", test_results_saved),
        ("Custom transformer option", test_custom_transformer_option),
        ("Job script updated", test_job_script_updated),
        ("ACE added to complex_scm.py", test_complex_scm_ace_added),
    ]
    
    print("="*70)
    print("ACE COMPLEX SCM VERIFICATION")
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
        print("ACE complex SCM is ready for HPC execution")
        print("="*70)
        return 0
    else:
        print(f"[FAIL] {errors} test(s) failed")
        print("Fix errors before HPC submission")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
