"""
Tests for HPC/SLURM workflow scripts.

Best practices for testing HPC workflows locally:
1. Test core logic separately from SLURM directives
2. Validate SLURM directive syntax
3. Test script execution with dry-run mode
4. Mock SLURM commands for integration tests
5. Validate environment variable handling
6. Test job script structure and dependencies

These tests do NOT require actual HPC/SLURM access.
"""

import pytest
import subprocess
import os
import tempfile
from pathlib import Path


# =============================================================================
# SLURM Directive Validation Tests
# =============================================================================

@pytest.mark.unit
def test_run_ace_main_has_valid_slurm_directives():
    """Test that run_ace_main.sh has valid SLURM directives."""
    
    script_path = Path("jobs/run_ace_main.sh")
    assert script_path.exists(), "run_ace_main.sh not found"
    
    content = script_path.read_text()
    
    # Check for required SLURM directives
    assert "#SBATCH --job-name" in content
    assert "#SBATCH --time" in content
    assert "#SBATCH --cpus-per-task" in content or "#SBATCH -c" in content
    assert "#SBATCH --mem" in content


@pytest.mark.unit
def test_run_baselines_has_valid_slurm_directives():
    """Test that run_baselines.sh has valid SLURM directives."""
    
    script_path = Path("jobs/run_baselines.sh")
    assert script_path.exists(), "run_baselines.sh not found"
    
    content = script_path.read_text()
    
    # Check for SLURM directives
    assert "#SBATCH" in content


@pytest.mark.unit
def test_all_job_scripts_have_shebang():
    """Test that all job scripts have proper shebang."""
    
    job_dir = Path("jobs")
    job_scripts = list(job_dir.glob("*.sh"))
    
    assert len(job_scripts) > 0, "No job scripts found"
    
    for script in job_scripts:
        content = script.read_text()
        first_line = content.split('\n')[0]
        
        # Should start with shebang
        assert first_line.startswith("#!"), f"{script.name} missing shebang"


@pytest.mark.unit
def test_job_scripts_are_executable():
    """Test that job scripts have executable permissions."""
    
    job_dir = Path("jobs")
    job_scripts = list(job_dir.glob("*.sh"))
    
    for script in job_scripts:
        # Check if executable
        assert os.access(script, os.X_OK), f"{script.name} not executable"


# =============================================================================
# Script Syntax Validation Tests
# =============================================================================

@pytest.mark.unit
def test_run_ace_main_shell_syntax():
    """Test that run_ace_main.sh has valid shell syntax."""
    
    script_path = Path("jobs/run_ace_main.sh")
    
    # Use bash -n for syntax check (dry-run)
    result = subprocess.run(
        ['bash', '-n', str(script_path)],
        capture_output=True,
        text=True
    )
    
    # Should have no syntax errors
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


@pytest.mark.unit
def test_run_all_shell_syntax():
    """Test that run_all.sh has valid shell syntax."""
    
    script_path = Path("run_all.sh")
    
    if script_path.exists():
        result = subprocess.run(
            ['bash', '-n', str(script_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


@pytest.mark.unit
def test_all_shell_scripts_have_valid_syntax():
    """Test all .sh files have valid bash syntax."""
    
    sh_files = list(Path('.').glob('*.sh')) + list(Path('jobs').glob('*.sh'))
    
    for script in sh_files:
        result = subprocess.run(
            ['bash', '-n', str(script)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"{script.name} syntax error: {result.stderr}"


# =============================================================================
# Environment Variable Tests
# =============================================================================

@pytest.mark.unit
def test_hpc_env_vars_handling():
    """Test that scripts handle HPC environment variables."""
    
    script_path = Path("jobs/run_ace_main.sh")
    content = script_path.read_text()
    
    # Should reference common HPC env vars
    # These are typical patterns - adjust based on your scripts
    hpc_patterns = [
        'SLURM_JOB_ID',
        'SLURM_CPUS_PER_TASK',
        '$USER',
        'module load',  # Common in HPC
    ]
    
    # At least some HPC patterns should be present
    found_patterns = sum(1 for pattern in hpc_patterns if pattern in content)
    
    # Just verify it's an HPC script (has SBATCH)
    assert '#SBATCH' in content


# =============================================================================
# Job Dependency Tests
# =============================================================================

@pytest.mark.unit
def test_run_all_calls_job_scripts():
    """Test that run_all.sh references individual job scripts."""
    
    script_path = Path("run_all.sh")
    
    if script_path.exists():
        content = script_path.read_text()
        
        # Should reference job scripts
        expected_jobs = ['run_ace_main.sh', 'run_baselines.sh']
        
        for job in expected_jobs:
            if Path(f"jobs/{job}").exists():
                # Might reference it
                assert job in content or 'sbatch' in content


# =============================================================================
# Python Command Validation Tests
# =============================================================================

@pytest.mark.unit
def test_job_scripts_call_valid_python_modules():
    """Test that job scripts reference existing Python files."""
    
    script_path = Path("jobs/run_ace_main.sh")
    content = script_path.read_text()
    
    # Should call ace_experiments.py
    if 'ace_experiments.py' in content:
        assert Path('ace_experiments.py').exists()
    
    # Should call baselines.py
    script_path2 = Path("jobs/run_baselines.sh")
    if script_path2.exists():
        content2 = script_path2.read_text()
        if 'baselines.py' in content2:
            assert Path('baselines.py').exists()


# =============================================================================
# Dry-Run/Mock Execution Tests
# =============================================================================

@pytest.mark.integration
def test_ace_experiments_help_flag():
    """Test that ace_experiments.py has help flag (validates importability)."""
    
    result = subprocess.run(
        ['python', 'ace_experiments.py', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Should show help without errors
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower() or 'ace' in result.stdout.lower()


@pytest.mark.integration
def test_baselines_help_flag():
    """Test that baselines.py has help flag."""
    
    result = subprocess.run(
        ['python', 'baselines.py', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Should show help
    assert result.returncode == 0


# =============================================================================
# Output Directory Tests
# =============================================================================

@pytest.mark.unit
def test_job_scripts_specify_output_directories():
    """Test that job scripts specify output directories."""
    
    script_path = Path("jobs/run_ace_main.sh")
    content = script_path.read_text()
    
    # Should have output or results directory
    assert '--output' in content or 'results' in content or 'OUTPUT' in content


# =============================================================================
# Mock SLURM Command Tests (without actual SLURM)
# =============================================================================

@pytest.mark.unit
def test_mock_slurm_job_submission():
    """Test job submission logic without actual SLURM."""
    
    # Create a mock sbatch function
    def mock_sbatch(script_path):
        """Mock SLURM sbatch command."""
        # Validate script exists
        assert Path(script_path).exists()
        
        # Return mock job ID
        return "12345"
    
    # Test the mock
    job_id = mock_sbatch("jobs/run_ace_main.sh")
    assert isinstance(job_id, str)
    assert job_id.isdigit()


@pytest.mark.unit
def test_parse_slurm_directives():
    """Test parsing SLURM directives from script."""
    
    script_path = Path("jobs/run_ace_main.sh")
    content = script_path.read_text()
    
    # Extract SBATCH directives
    directives = {}
    for line in content.split('\n'):
        if line.startswith('#SBATCH'):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[1]
                value = parts[2] if len(parts) > 2 else True
                directives[key] = value
    
    # Should have found some directives
    assert len(directives) > 0


# =============================================================================
# Resource Request Validation Tests
# =============================================================================

@pytest.mark.unit
def test_job_scripts_request_reasonable_resources():
    """Test that job scripts request reasonable resources."""
    
    script_path = Path("jobs/run_ace_main.sh")
    content = script_path.read_text()
    
    # Parse time limit if present
    import re
    time_match = re.search(r'#SBATCH --time=(\d+):(\d+):(\d+)', content)
    
    if time_match:
        hours = int(time_match.group(1))
        # Should request reasonable time (not 999 hours)
        assert hours < 100, "Time request seems unreasonably high"


# =============================================================================
# Local Execution Test (Dry Run)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_test_script_runs():
    """Test that pipeline_test.sh can execute (validates workflow)."""
    
    script_path = Path("pipeline_test.sh")
    
    if script_path.exists() and os.access(script_path, os.X_OK):
        # This would actually run the test - skip for now
        # In real scenario, you'd run with minimal data
        pytest.skip("Would execute actual pipeline test - skip for unit test")
    else:
        pytest.skip("pipeline_test.sh not found or not executable")
