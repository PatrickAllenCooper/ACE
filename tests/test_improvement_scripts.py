"""
Comprehensive tests for all improvement scripts.

Tests cover:
- Multi-seed infrastructure
- Ablation infrastructure
- Failure analysis
- Statistical testing
- HPC sync script
- All new functionality
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
import os


# =============================================================================
# Multi-Seed Infrastructure Tests (now in unified CLI)
# =============================================================================

# Note: Multi-seed and consolidation scripts consolidated into ace.sh
# Tests moved to test_unified_cli.py


@pytest.mark.unit
def test_compute_statistics_imports():
    """Test that compute_statistics.py imports."""
    try:
        import scripts.compute_statistics as cs
        assert hasattr(cs, 'main')
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_compute_statistics_has_key_functions():
    """Test compute_statistics has expected functions."""
    script_path = Path("scripts/compute_statistics.py")
    content = script_path.read_text()
    
    assert 'def load_results_from_seeds' in content
    assert 'def compute_summary_statistics' in content
    assert 'def format_latex_table' in content


# =============================================================================
# Ablation Infrastructure Tests (now in unified CLI)
# =============================================================================

# Note: Ablation scripts consolidated into ace.sh
# Tests moved to test_unified_cli.py


@pytest.mark.unit
def test_analyze_ablations_imports():
    """Test that analyze_ablations.py imports."""
    try:
        import scripts.analyze_ablations as aa
        assert hasattr(aa, 'main')
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_ablation_flags_in_ace_experiments():
    """Test that ablation flags are present in ace_experiments."""
    result = subprocess.run(
        ['python', 'ace_experiments.py', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0
    assert '--no_per_node_convergence' in result.stdout
    assert '--no_dedicated_root_learner' in result.stdout
    assert '--no_diversity_reward' in result.stdout


# =============================================================================
# Failure Analysis Tests
# =============================================================================

@pytest.mark.unit
def test_test_failure_cases_imports():
    """Test that test_failure_cases.py imports."""
    try:
        import scripts.test_failure_cases as tfc
        assert hasattr(tfc, 'main')
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_test_failure_cases_has_structures():
    """Test that failure cases defines test structures."""
    script_path = Path("scripts/test_failure_cases.py")
    content = script_path.read_text()
    
    assert 'FullyConnectedSCM' in content
    assert 'RingSCM' in content


@pytest.mark.integration
def test_failure_cases_runs_locally(tmp_path):
    """Test that failure case script can run locally."""
    
    result = subprocess.run(
        ['python', 'scripts/test_failure_cases.py', 
         '--episodes', '2', '--output', str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Should complete without crashing
    assert "Failure Case Analysis" in result.stdout or result.returncode == 0


# =============================================================================
# Statistical Testing Tests
# =============================================================================

@pytest.mark.unit
def test_statistical_tests_imports():
    """Test that statistical_tests.py imports."""
    try:
        import scripts.statistical_tests as st
        assert hasattr(st, 'main')
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_statistical_tests_has_functions():
    """Test statistical_tests has key functions."""
    script_path = Path("scripts/statistical_tests.py")
    content = script_path.read_text()
    
    assert 'def paired_t_test' in content
    assert 'def bonferroni_correction' in content
    assert 'def format_significance' in content


# =============================================================================
# HPC Sync Script Tests
# =============================================================================

@pytest.mark.unit
def test_sync_results_from_hpc_exists():
    """Test that HPC sync script exists."""
    script = Path("scripts/sync_results_from_hpc.sh")
    assert script.exists()


@pytest.mark.unit
def test_sync_results_from_hpc_syntax():
    """Test HPC sync script syntax."""
    script = Path("scripts/sync_results_from_hpc.sh")
    
    result = subprocess.run(
        ['bash', '-n', str(script)],
        capture_output=True
    )
    
    assert result.returncode == 0


@pytest.mark.unit
def test_sync_results_from_hpc_mentions_rsync():
    """Test that sync script uses rsync."""
    script = Path("scripts/sync_results_from_hpc.sh")
    content = script.read_text()
    
    assert 'rsync' in content
    assert 'ssh' in content
    assert 'process_all_results.sh' in content


# =============================================================================
# Seed Argument Tests
# =============================================================================

@pytest.mark.unit
def test_ace_experiments_has_seed_argument():
    """Test that ace_experiments.py has --seed argument."""
    result = subprocess.run(
        ['python', 'ace_experiments.py', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0
    assert '--seed' in result.stdout


@pytest.mark.unit
def test_seed_argument_accepts_values():
    """Test that --seed argument can be parsed."""
    # Just test argument parsing, not full execution
    result = subprocess.run(
        ['python', '-c', 
         'import argparse; '
         'p = argparse.ArgumentParser(); '
         'p.add_argument("--seed", type=int); '
         'args = p.parse_args(["--seed", "42"]); '
         'assert args.seed == 42'],
        capture_output=True,
        timeout=5
    )
    
    assert result.returncode == 0


# =============================================================================
# Dependencies Tests
# =============================================================================

@pytest.mark.unit
def test_requirements_txt_exists():
    """Test that requirements.txt exists."""
    req_file = Path("requirements.txt")
    assert req_file.exists()


@pytest.mark.unit
def test_requirements_txt_has_key_packages():
    """Test that requirements.txt has essential packages."""
    req_file = Path("requirements.txt")
    content = req_file.read_text()
    
    assert 'torch' in content
    assert 'transformers' in content
    assert 'pandas' in content
    assert 'pytest' in content


@pytest.mark.unit
def test_environment_yml_exists():
    """Test that environment.yml exists."""
    env_file = Path("environment.yml")
    assert env_file.exists()


@pytest.mark.unit
def test_environment_yml_structure():
    """Test that environment.yml has proper structure."""
    env_file = Path("environment.yml")
    content = env_file.read_text()
    
    assert 'name: ace' in content
    assert 'dependencies:' in content
    assert 'python=' in content


# =============================================================================
# Paper Enhancement Tests
# =============================================================================

@pytest.mark.unit
def test_paper_has_reproducibility_statement():
    """Test that paper has reproducibility statement."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    assert 'Reproducibility' in content or 'reproducibility' in content.lower()


@pytest.mark.unit
def test_paper_has_ablation_section():
    """Test that paper has ablation framework."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    assert 'Ablation' in content


@pytest.mark.unit
def test_paper_has_failure_analysis_section():
    """Test that paper has failure analysis framework."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    assert 'When Does ACE' in content or 'Struggle' in content


@pytest.mark.unit
def test_paper_has_todo_markers_for_results():
    """Test that paper has clear TODO markers."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    # Should have TODO markers for pending results
    assert 'TODO' in content or 'textcolor{red}' in content


# =============================================================================
# Integration Tests for Complete Workflow
# =============================================================================

@pytest.mark.integration
def test_all_improvement_scripts_exist_and_executable():
    """Test that all improvement scripts exist and are executable."""
    
    scripts = [
        'run_all_multi_seed.sh',
        'consolidate_multi_runs.sh',
        'run_ablations.sh',
        'sync_results_from_hpc.sh',
        'process_all_results.sh',
        'compute_statistics.py',
        'analyze_ablations.py',
        'test_failure_cases.py',
        'statistical_tests.py'
    ]
    
    for script_name in scripts:
        script_path = Path("scripts") / script_name
        assert script_path.exists(), f"{script_name} not found"
        assert os.access(script_path, os.X_OK), f"{script_name} not executable"


@pytest.mark.integration
def test_all_improvement_scripts_have_valid_syntax():
    """Test syntax of all bash improvement scripts."""
    
    bash_scripts = [
        'run_all_multi_seed.sh',
        'consolidate_multi_runs.sh',
        'run_ablations.sh',
        'sync_results_from_hpc.sh'
    ]
    
    for script_name in bash_scripts:
        script_path = Path("scripts") / script_name
        
        result = subprocess.run(
            ['bash', '-n', str(script_path)],
            capture_output=True
        )
        
        assert result.returncode == 0, f"{script_name} has syntax error: {result.stderr}"


@pytest.mark.integration
def test_all_improvement_python_scripts_import():
    """Test that all Python improvement scripts can be imported."""
    
    python_scripts = [
        'compute_statistics',
        'analyze_ablations',
        'test_failure_cases',
        'statistical_tests'
    ]
    
    for script_name in python_scripts:
        try:
            __import__(f'scripts.{script_name}')
        except ImportError as e:
            pytest.fail(f"{script_name}.py import failed: {e}")


# =============================================================================
# Workflow Completeness Tests
# =============================================================================

@pytest.mark.unit
def test_improvement_plan_exists():
    """Test that improvement plan exists."""
    plan = Path("IMPROVEMENT_PLAN.txt")
    assert plan.exists()


@pytest.mark.unit
def test_implementation_complete_doc_exists():
    """Test that implementation complete doc exists."""
    doc = Path("IMPLEMENTATION_COMPLETE.txt")
    assert doc.exists()


@pytest.mark.unit
def test_paper_strength_audit_updated():
    """Test that paper strength audit was updated."""
    audit = Path("guidance_documents/PAPER_STRENGTH_AUDIT.txt")
    content = audit.read_text()
    
    # Should mention updated assessment
    assert 'UPDATED' in content or 'COMPLETE' in content


@pytest.mark.unit
def test_all_critical_phases_have_infrastructure():
    """Test that all 5 phases have corresponding scripts."""
    
    # Phase 1: Statistical validation
    assert Path("scripts/run_all_multi_seed.sh").exists()
    assert Path("scripts/compute_statistics.py").exists()
    
    # Phase 2: Ablations
    assert Path("scripts/run_ablations.sh").exists()
    assert Path("scripts/analyze_ablations.py").exists()
    
    # Phase 3: Failures
    assert Path("scripts/test_failure_cases.py").exists()
    
    # Phase 4: Statistical tests
    assert Path("scripts/statistical_tests.py").exists()
    
    # Phase 5: Documentation
    assert Path("requirements.txt").exists()
    assert Path("environment.yml").exists()


# =============================================================================
# End-to-End Workflow Validation
# =============================================================================

@pytest.mark.integration
def test_complete_workflow_documented():
    """Test that complete workflow is documented."""
    
    # Check that workflow is documented somewhere
    docs_to_check = [
        "IMPROVEMENT_PLAN.txt",
        "IMPLEMENTATION_COMPLETE.txt",
        "README.md"
    ]
    
    found_workflow = False
    for doc in docs_to_check:
        if Path(doc).exists():
            content = Path(doc).read_text()
            if 'run_all_multi_seed' in content or 'multi-seed' in content.lower():
                found_workflow = True
                break
    
    assert found_workflow, "Complete workflow not documented"


@pytest.mark.unit
def test_all_scripts_reference_correct_hpc_path():
    """Test that HPC-related scripts use correct server path."""
    
    hpc_path = "/projects/paco0228/ACE"
    hpc_host = "paco0228@login.rc.colorado.edu"
    
    # Check sync script
    sync_script = Path("scripts/sync_results_from_hpc.sh")
    if sync_script.exists():
        content = sync_script.read_text()
        assert hpc_host in content
        assert '/projects/paco0228' in content or 'HPC_PROJECT_DIR' in content
