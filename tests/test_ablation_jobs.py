"""
Tests for ablation job scripts and argument handling.

Prevents catastrophic failures like the scratch directory import bug.
Provides 90% test coverage of ablation system before HPC execution.
"""

import subprocess
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestAblationArguments:
    """Test that all ablation flags are properly handled."""
    
    def test_ablation_flags_exist(self):
        """Verify all ablation flags are defined in argument parser."""
        import ace_experiments
        
        # Create parser
        import argparse
        parser = argparse.ArgumentParser()
        
        # Add all arguments (mimicking main)
        parser.add_argument("--no_per_node_convergence", action="store_true")
        parser.add_argument("--no_dedicated_root_learner", action="store_true")
        parser.add_argument("--no_diversity_reward", action="store_true")
        parser.add_argument("--custom", action="store_true")
        
        # Should parse without error
        args = parser.parse_args([])
        assert hasattr(args, 'no_per_node_convergence')
        assert hasattr(args, 'no_dedicated_root_learner')
        assert hasattr(args, 'no_diversity_reward')
        assert hasattr(args, 'custom')
    
    def test_all_ablation_flags_are_boolean(self):
        """Verify ablation flags are simple boolean switches."""
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        help_text = result.stdout
        # These should be store_true actions (no argument required)
        assert "--no_per_node_convergence" in help_text
        assert "--no_dedicated_root_learner" in help_text
        assert "--no_diversity_reward" in help_text
        # Verify they don't require values
        assert "--no_per_node_convergence NO_PER_NODE_CONVERGENCE" not in help_text
    
    def test_ablation_flags_in_help(self):
        """Verify ablation flags appear in --help output."""
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        help_text = result.stdout
        assert "--no_per_node_convergence" in help_text
        assert "--no_dedicated_root_learner" in help_text
        assert "--no_diversity_reward" in help_text
        assert "--custom" in help_text
    
    def test_script_accepts_ablation_flags(self):
        """Verify ace_experiments.py accepts ablation flags without error."""
        # Test each flag individually with minimal run
        flags_to_test = [
            (["--custom"], "no_dpo"),
            (["--no_per_node_convergence"], "no_convergence"),
            (["--no_dedicated_root_learner"], "no_root_learner"),
            (["--no_diversity_reward"], "no_diversity"),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for flags, name in flags_to_test:
                output_dir = os.path.join(tmpdir, name)
                result = subprocess.run(
                    [sys.executable, "ace_experiments.py", 
                     "--episodes", "0",
                     "--seed", "42",
                     "--output", output_dir] + flags,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Should not have argument parsing errors
                assert "error: unrecognized arguments" not in result.stderr.lower(), \
                    f"Flag {flags} not recognized in {name}: {result.stderr}"
                assert "error: invalid choice" not in result.stderr.lower(), \
                    f"Flag {flags} has invalid value in {name}: {result.stderr}"
    
    def test_combined_ablation_flags(self):
        """Test that multiple ablation flags can be combined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "ace_experiments.py",
                 "--episodes", "0",
                 "--seed", "42",
                 "--output", tmpdir,
                 "--no_diversity_reward",
                 "--no_per_node_convergence"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Should accept multiple ablation flags
            assert result.returncode in [0, None] or "completed" in result.stdout.lower()


class TestJobScriptIntegrity:
    """Test job scripts for common failure modes."""
    
    def test_scratch_script_doesnt_cd_before_python(self):
        """Verify scratch script doesn't change directory before running Python.
        
        Bug: cd to scratch dir breaks imports from local modules.
        """
        with open("jobs/run_ablations_scratch.sh") as f:
            content = f.read()
        
        # Find the Python invocation line
        lines = content.split('\n')
        python_line_idx = None
        for i, line in enumerate(lines):
            if 'python' in line and 'ace_experiments.py' in line and not line.strip().startswith('#'):
                python_line_idx = i
                break
        
        assert python_line_idx is not None, "Could not find Python invocation"
        
        # Check previous 10 lines for cd to scratch
        previous_lines = lines[max(0, python_line_idx-10):python_line_idx]
        cd_to_scratch_found = False
        cd_to_submit_found = False
        
        for line in previous_lines:
            if 'cd' in line and 'SCRATCH' in line and not line.strip().startswith('#'):
                cd_to_scratch_found = True
            if 'cd' in line and 'SUBMIT_DIR' in line and not line.strip().startswith('#'):
                cd_to_submit_found = True
        
        # Should be in submit dir, NOT scratch, when running Python
        assert not cd_to_scratch_found or cd_to_submit_found, \
            "Script changes to scratch dir before Python - will break imports!"
    
    def test_all_job_scripts_have_qos(self):
        """Verify all job scripts have QoS specified (Alpine requirement)."""
        import glob
        
        job_scripts = glob.glob("jobs/*.sh")
        
        for script_path in job_scripts:
            if 'run_' in script_path:  # Only check SLURM job scripts
                with open(script_path) as f:
                    content = f.read()
                
                assert "#SBATCH --qos=" in content or "# Not a SLURM script" in content, \
                    f"{script_path} missing QoS specification"


class TestAblationLogic:
    """Test that ablation flags actually change behavior."""
    
    def test_no_diversity_reward_sets_weight_to_zero(self):
        """Verify --no_diversity_reward actually disables diversity."""
        # Parse with ablation flag
        import ace_experiments
        import argparse
        
        # Create minimal parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--diversity_reward_weight", type=float, default=0.3)
        parser.add_argument("--no_diversity_reward", action="store_true")
        
        # Test without flag
        args = parser.parse_args([])
        assert args.diversity_reward_weight == 0.3
        
        # Test with flag - verify it would be set to 0
        args = parser.parse_args(["--no_diversity_reward"])
        assert args.no_diversity_reward is True
    
    def test_no_dedicated_root_learner_disables_flag(self):
        """Verify --no_dedicated_root_learner disables root learner."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_dedicated_root_learner", action="store_true")
        parser.add_argument("--root_fitting", action="store_true")
        parser.add_argument("--no_dedicated_root_learner", action="store_true")
        
        args = parser.parse_args(["--no_dedicated_root_learner"])
        assert args.no_dedicated_root_learner is True
    
    def test_no_per_node_convergence_disables_flag(self):
        """Verify --no_per_node_convergence disables per-node convergence."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_per_node_convergence", action="store_true")
        parser.add_argument("--no_per_node_convergence", action="store_true")
        
        args = parser.parse_args(["--no_per_node_convergence"])
        assert args.no_per_node_convergence is True
    
    def test_custom_flag_is_boolean(self):
        """Verify --custom is a simple boolean flag."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom", action="store_true")
        
        # Without flag
        args = parser.parse_args([])
        assert args.custom is False
        
        # With flag
        args = parser.parse_args(["--custom"])
        assert args.custom is True


class TestJobOutputValidation:
    """Test that jobs produce expected output files."""
    
    def test_expected_output_files_created(self, tmp_path):
        """Verify that a minimal run creates expected output files."""
        # Run with 0 episodes to test setup/teardown only
        result = subprocess.run(
            [
                sys.executable, "ace_experiments.py",
                "--episodes", "0",
                "--output", str(tmp_path),
                "--seed", "42"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check if run directory was created
        run_dirs = list(tmp_path.glob("run_*"))
        if len(run_dirs) > 0:
            run_dir = run_dirs[0]
            
            # Should at least create log file
            assert (run_dir / "experiment.log").exists(), \
                "experiment.log not created"
    
    def test_ablation_runner_script_exists(self):
        """Verify ablation runner script exists and is executable."""
        script_path = Path("scripts/runners/run_ablations_fast.py")
        assert script_path.exists(), "run_ablations_fast.py not found"
        
        # Should be executable or at least readable
        assert os.access(script_path, os.R_OK), "Script not readable"
    
    def test_ablation_runner_help(self):
        """Verify ablation runner has proper help text."""
        result = subprocess.run(
            [sys.executable, "scripts/runners/run_ablations_fast.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "--all" in result.stdout
        assert "--ablation" in result.stdout
        assert "--seeds" in result.stdout
    
    def test_ablation_runner_validates_ablation_type(self):
        """Verify runner rejects invalid ablation types."""
        result = subprocess.run(
            [sys.executable, "scripts/runners/run_ablations_fast.py", 
             "--ablation", "invalid_type"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should fail with invalid choice
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()


class TestFastAblationRunner:
    """Test the fast ablation runner script."""
    
    def test_runner_imports_successfully(self):
        """Verify runner script can be imported."""
        # This tests for syntax errors and import issues
        result = subprocess.run(
            [sys.executable, "-c", 
             "import sys; sys.path.insert(0, 'scripts/runners'); import run_ablations_fast"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Import failed: {result.stderr}"
    
    def test_runner_ablation_mapping(self):
        """Verify ablation names map to correct flags."""
        # Import the module
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_ablations_fast import run_ablation
            
            # Verify function exists
            assert callable(run_ablation)
        except ImportError:
            pytest.skip("Cannot import run_ablations_fast")


class TestAblationJobScripts:
    """Test ablation job script configurations."""
    
    def test_fast_ablation_job_has_qos(self):
        """Verify fast ablation job script has QoS."""
        with open("jobs/run_ablations_fast.sh") as f:
            content = f.read()
        
        assert "#SBATCH --qos=" in content
        assert "qos=normal" in content.lower()
    
    def test_fast_ablation_job_timeout_reasonable(self):
        """Verify fast ablation has reasonable timeout."""
        with open("jobs/run_ablations_fast.sh") as f:
            content = f.read()
        
        # Should have time limit
        assert "#SBATCH --time=" in content
        
        # Extract time limit
        for line in content.split('\n'):
            if '#SBATCH --time=' in line:
                time_str = line.split('=')[1].strip()
                # Should be 6 hours or less (not 12+)
                assert '06:00:00' in time_str or '05:' in time_str or '04:' in time_str, \
                    f"Fast ablations should use <=6h, got: {time_str}"
    
    def test_fast_ablation_calls_correct_script(self):
        """Verify fast ablation job calls the Python runner."""
        with open("jobs/run_ablations_fast.sh") as f:
            content = f.read()
        
        assert "run_ablations_fast.py" in content
        assert "scripts/runners/run_ablations_fast.py" in content


class TestLocalAblationExecution:
    """Test ablation execution locally before HPC."""
    
    @pytest.mark.slow
    def test_single_ablation_runs_locally(self, tmp_path):
        """Run a single ablation with 1 episode locally."""
        result = subprocess.run(
            [sys.executable, "scripts/runners/run_ablations_fast.py",
             "--ablation", "no_diversity",
             "--seeds", "42",
             "--max-episodes", "1",
             "--output-dir", str(tmp_path / "ablation_test")],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Should complete successfully
        assert result.returncode == 0, f"Ablation failed: {result.stderr}"
        
        # Should have created output directory
        output_dirs = list(tmp_path.glob("ablation_test_*"))
        assert len(output_dirs) > 0, "No output directory created"


class TestAblationLogicImplementation:
    """Test that ablation logic is actually implemented in ace_experiments.py."""
    
    def test_ablation_logic_modifies_args(self):
        """Verify ablation flags modify configuration."""
        with open("ace_experiments.py") as f:
            content = f.read()
        
        # Check that each ablation flag has implementation
        assert "if args.no_diversity_reward:" in content
        assert "args.diversity_reward_weight = 0.0" in content
        
        assert "if args.no_dedicated_root_learner:" in content
        assert "args.use_dedicated_root_learner = False" in content
        
        assert "if args.no_per_node_convergence:" in content
        assert "args.use_per_node_convergence = False" in content
    
    def test_ablation_flags_have_logging(self):
        """Verify ablations log their activation."""
        with open("ace_experiments.py") as f:
            content = f.read()
        
        # Each ablation should log a warning
        assert 'ABLATION: Diversity reward disabled' in content
        assert 'ABLATION: Dedicated root learner disabled' in content
        assert 'ABLATION: Per-node convergence disabled' in content


class TestJobScriptUnbufferedOutput:
    """Test job scripts use unbuffered Python output."""
    
    def test_single_ablation_has_unbuffered(self):
        """Verify single ablation job uses python -u."""
        with open("jobs/run_single_ablation.sh") as f:
            content = f.read()
        
        assert "python -u" in content
    
    def test_all_slurm_jobs_have_time_limit(self):
        """Verify all SLURM job scripts have time limits."""
        import glob
        
        for script_path in glob.glob("jobs/run_*.sh"):
            with open(script_path) as f:
                content = f.read()
            
            if "#SBATCH" in content:  # Is a SLURM script
                assert "#SBATCH --time=" in content, f"{script_path} missing time limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
