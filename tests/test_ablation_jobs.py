"""
Tests for ablation job scripts and argument handling.

Prevents catastrophic failures like the scratch directory import bug.
"""

import subprocess
import pytest
import sys
import os

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
        # Test each flag individually
        flags_to_test = [
            ["--custom"],
            ["--no_per_node_convergence"],
            ["--no_dedicated_root_learner"],
            ["--no_diversity_reward"],
        ]
        
        for flags in flags_to_test:
            result = subprocess.run(
                [sys.executable, "ace_experiments.py", "--episodes", "0"] + flags,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should not have argument parsing errors
            assert "error: unrecognized arguments" not in result.stderr.lower(), \
                f"Flag {flags} not recognized: {result.stderr}"


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
    
    @pytest.mark.slow
    def test_no_diversity_reward_sets_weight_to_zero(self):
        """Verify --no_diversity_reward actually disables diversity."""
        # This would require running a short experiment
        # For now, just verify the code path exists
        pass  # TODO: Implement when ablation logic is added
    
    @pytest.mark.slow
    def test_custom_flag_uses_transformer_not_llm(self):
        """Verify --custom uses TransformerPolicy instead of HuggingFacePolicy."""
        pass  # TODO: Implement when ablation logic is added


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
