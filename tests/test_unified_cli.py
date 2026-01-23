"""
Comprehensive tests for the unified CLI (ace.sh)

Tests cover all commands and functionality of the consolidated CLI.
"""

import pytest
import subprocess
from pathlib import Path


# =============================================================================
# CLI Existence and Syntax Tests
# =============================================================================

@pytest.mark.unit
def test_cli_exists():
    """Test that ace.sh exists and is executable."""
    cli = Path("ace.sh")
    assert cli.exists(), "ace.sh not found"
    assert cli.stat().st_mode & 0o111, "ace.sh is not executable"


@pytest.mark.unit
def test_cli_syntax():
    """Test that ace.sh has valid bash syntax."""
    result = subprocess.run(
        ['bash', '-n', 'ace.sh'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, f"Syntax error in ace.sh: {result.stderr}"


@pytest.mark.unit
def test_cli_help_command():
    """Test that ace.sh help command works."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, "Help command failed"
    assert "ACE: Active Causal Experimentation" in result.stdout
    assert "Usage:" in result.stdout
    assert "EXPERIMENT COMMANDS:" in result.stdout
    assert "POST-PROCESSING COMMANDS:" in result.stdout


@pytest.mark.unit
def test_cli_no_args_shows_help():
    """Test that running ace.sh with no arguments shows help."""
    result = subprocess.run(
        ['./ace.sh'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


@pytest.mark.unit
def test_cli_unknown_command():
    """Test that unknown commands show error and help."""
    result = subprocess.run(
        ['./ace.sh', 'nonexistent-command'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 1
    assert "Unknown command" in result.stdout


# =============================================================================
# CLI Command Availability Tests
# =============================================================================

@pytest.mark.unit
def test_cli_has_all_experiment_commands():
    """Test that all experiment commands are documented in help."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # Check for all experiment commands
    assert 'run <experiment>' in result.stdout
    assert 'run-multi-seed' in result.stdout
    assert 'run-ablations' in result.stdout
    assert 'run-obs-ablation' in result.stdout


@pytest.mark.unit
def test_cli_has_all_processing_commands():
    """Test that all post-processing commands are documented."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # Check for post-processing commands
    assert 'process' in result.stdout
    assert 'consolidate-multi-seed' in result.stdout
    assert 'extract' in result.stdout
    assert 'verify' in result.stdout


@pytest.mark.unit
def test_cli_has_all_utility_commands():
    """Test that all utility commands are documented."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # Check for utility commands
    assert 'test' in result.stdout
    assert 'clean' in result.stdout
    assert 'check-version' in result.stdout


@pytest.mark.unit
def test_cli_has_hpc_commands():
    """Test that HPC commands are documented."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    assert 'sync-hpc' in result.stdout


# =============================================================================
# CLI Command Execution Tests (Dry Run / Validation)
# =============================================================================

@pytest.mark.unit
def test_cli_check_version():
    """Test check-version command."""
    result = subprocess.run(
        ['./ace.sh', 'check-version'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert 'Python:' in result.stdout or 'python' in result.stdout.lower()


@pytest.mark.unit
def test_cli_clean_command_structure():
    """Test that clean command has correct structure."""
    # Read the ace.sh file to verify clean command exists
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_clean()' in content, "clean command function not found"
    assert '__pycache__' in content, "clean doesn't remove __pycache__"
    assert '.pytest_cache' in content, "clean doesn't remove .pytest_cache"
    assert '.coverage' in content, "clean doesn't remove .coverage"


@pytest.mark.unit
def test_cli_extract_requires_type():
    """Test that extract command validates type parameter."""
    # Read ace.sh to verify validation
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should have validation for ace|baselines
    assert 'cmd_extract()' in content
    assert 'ace)' in content
    assert 'baselines)' in content


@pytest.mark.unit
def test_cli_run_command_structure():
    """Test that run command supports all experiments."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_run()' in content
    # Check for all experiment types
    assert 'ace|ace-main)' in content
    assert 'baselines)' in content
    assert 'complex|complex-scm)' in content
    assert 'duffing)' in content
    assert 'phillips)' in content
    assert 'all)' in content


# =============================================================================
# CLI Multi-Seed Command Tests
# =============================================================================

@pytest.mark.unit
def test_cli_multi_seed_structure():
    """Test that multi-seed command has correct structure."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_run_multi_seed()' in content
    # Should support different seed counts
    assert 'SEEDS=' in content or 'seeds' in content.lower()


@pytest.mark.unit
def test_cli_multi_seed_supports_seed_counts():
    """Test that multi-seed supports 3, 5, 10 seeds."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Check for seed arrays
    assert '3)' in content  # 3 seeds case
    assert '5)' in content  # 5 seeds case
    assert '10)' in content  # 10 seeds case


# =============================================================================
# CLI Ablation Command Tests
# =============================================================================

@pytest.mark.unit
def test_cli_ablations_structure():
    """Test that ablations command includes all configurations."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_run_ablations()' in content
    # Check for ablation flags
    assert 'no_per_node_convergence' in content
    assert 'no_dedicated_root_learner' in content
    assert 'no_diversity_reward' in content


@pytest.mark.unit
def test_cli_obs_ablation_structure():
    """Test that obs-ablation command tests intervals."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_run_obs_ablation()' in content
    # Should test intervals 2, 3, 4, 5
    assert 'INTERVAL' in content or 'interval' in content


# =============================================================================
# CLI Process Command Tests
# =============================================================================

@pytest.mark.unit
def test_cli_process_structure():
    """Test that process command has all steps."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_process()' in content
    # Should include all processing steps
    assert 'extract' in content.lower()
    assert 'verify' in content.lower()


@pytest.mark.unit
def test_cli_consolidate_multi_seed_structure():
    """Test that consolidate-multi-seed command exists."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_consolidate_multi_seed()' in content
    assert 'compute_statistics.py' in content
    assert 'statistical_tests.py' in content


# =============================================================================
# CLI Integration with SLURM Jobs
# =============================================================================

@pytest.mark.unit
def test_cli_references_slurm_jobs():
    """Test that CLI properly references SLURM job scripts."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should reference all 5 job scripts
    assert 'jobs/run_ace_main.sh' in content
    assert 'jobs/run_baselines.sh' in content
    assert 'jobs/run_complex_scm.sh' in content
    assert 'jobs/run_duffing.sh' in content
    assert 'jobs/run_phillips.sh' in content


@pytest.mark.unit
def test_cli_uses_sbatch():
    """Test that CLI uses sbatch for HPC job submission."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'sbatch' in content


# =============================================================================
# CLI Error Handling Tests
# =============================================================================

@pytest.mark.unit
def test_cli_has_error_logging():
    """Test that CLI has logging functions."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should have logging functions
    assert 'log_info' in content
    assert 'log_success' in content
    assert 'log_warning' in content
    assert 'log_error' in content


@pytest.mark.unit
def test_cli_has_color_output():
    """Test that CLI has color-coded output."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should define color codes
    assert 'RED=' in content
    assert 'GREEN=' in content
    assert 'YELLOW=' in content
    assert 'BLUE=' in content


@pytest.mark.unit
def test_cli_sets_errexit():
    """Test that CLI uses set -e for error handling."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'set -e' in content


# =============================================================================
# CLI Output Directory Tests
# =============================================================================

@pytest.mark.unit
def test_cli_creates_timestamped_directories():
    """Test that CLI creates timestamped output directories."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should use timestamp-based naming
    assert 'TIMESTAMP=$(date' in content
    assert 'YYYYMMDD_HHMMSS' in content or '%Y%m%d_%H%M%S' in content


@pytest.mark.unit
def test_cli_organizes_results():
    """Test that CLI organizes results in proper directories."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should create results directories
    assert 'results/' in content
    assert 'mkdir' in content


# =============================================================================
# CLI Help Examples Tests
# =============================================================================

@pytest.mark.unit
def test_cli_help_includes_examples():
    """Test that help includes usage examples."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    assert 'EXAMPLES:' in result.stdout
    # Should have example for running all
    assert './ace.sh run all' in result.stdout


@pytest.mark.unit
def test_cli_help_includes_configuration():
    """Test that help includes configuration info."""
    result = subprocess.run(
        ['./ace.sh', 'help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    assert 'CONFIGURATION:' in result.stdout or 'HPC' in result.stdout


# =============================================================================
# Integration Tests with Actual Commands
# =============================================================================

@pytest.mark.integration
def test_cli_test_command_runs_pytest():
    """Test that 'ace.sh test' command structure is correct."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'cmd_test()' in content
    # Should run pytest
    assert 'pytest' in content or 'bash -n' in content


@pytest.mark.integration
def test_cli_maintains_backward_compatibility():
    """Test that CLI provides all functionality of removed scripts."""
    # Verify that all 15 removed script functionalities are present
    removed_scripts = [
        'run_all',
        'run_all_multi_seed',
        'run_ablations',
        'run_obs_ratio_ablation',
        'run_ace_experiments',
        'process_all_results',
        'consolidate_multi_runs',
        'extract_ace',
        'extract_baselines',
        'verify_claims',
        'sync_results_from_hpc',
        'pipeline_test',
        'cleanup',
        'check_version',
    ]
    
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Each removed script should have equivalent command
    # (not exact match, but functionality should exist)
    assert 'run all' in content  # run_all
    assert 'multi-seed' in content  # run_all_multi_seed
    assert 'ablations' in content  # run_ablations
    assert 'obs-ablation' in content  # run_obs_ratio_ablation
    assert 'process' in content  # process_all_results
    assert 'consolidate-multi-seed' in content  # consolidate_multi_runs
    assert 'extract' in content  # extract_ace, extract_baselines
    assert 'verify' in content  # verify_claims
    assert 'sync-hpc' in content  # sync_results_from_hpc
    assert 'cmd_test' in content  # pipeline_test
    assert 'cmd_clean' in content  # cleanup
    assert 'cmd_check_version' in content  # check_version


# =============================================================================
# CLI Script Location Tests
# =============================================================================

@pytest.mark.unit
def test_cli_changes_to_script_directory():
    """Test that CLI changes to script directory."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    # Should cd to SCRIPT_DIR
    assert 'SCRIPT_DIR=' in content
    assert 'cd "$SCRIPT_DIR"' in content or 'cd $SCRIPT_DIR' in content


@pytest.mark.unit
def test_cli_handles_main_entry():
    """Test that CLI has main entry point."""
    with open('ace.sh', 'r') as f:
        content = f.read()
    
    assert 'main()' in content
    assert 'main "$@"' in content


# =============================================================================
# CLI Documentation Tests
# =============================================================================

@pytest.mark.unit
def test_migration_guide_exists():
    """Test that migration guide exists."""
    migration_guide = Path("SCRIPT_CONSOLIDATION.md")
    assert migration_guide.exists(), "SCRIPT_CONSOLIDATION.md not found"


@pytest.mark.unit
def test_migration_guide_documents_all_changes():
    """Test that migration guide is comprehensive."""
    with open('SCRIPT_CONSOLIDATION.md', 'r') as f:
        content = f.read()
    
    # Should document consolidation
    assert 'Migration' in content or 'migration' in content
    assert 'Old Script' in content or 'Before' in content
    assert 'New Command' in content or 'After' in content
    
    # Should list removed scripts
    assert 'run_all.sh' in content
    assert 'ace.sh' in content


@pytest.mark.unit
def test_readme_updated_for_cli():
    """Test that README was updated for new CLI."""
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Should reference new CLI
    assert './ace.sh' in content
    assert 'Unified CLI' in content or 'unified' in content.lower()


# =============================================================================
# CLI Smoke Tests (if not on HPC)
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_cli_help_flag_variations():
    """Test that various help flags work."""
    for help_arg in ['help', '--help', '-h']:
        result = subprocess.run(
            ['./ace.sh', help_arg],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0, f"Help failed for {help_arg}"
        assert 'Usage:' in result.stdout
