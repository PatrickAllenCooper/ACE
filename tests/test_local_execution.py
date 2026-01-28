"""
Tests for local experiment execution.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestLocalExecutionScript:
    """Test local execution infrastructure."""
    
    def test_local_runner_exists(self):
        """Verify local runner script exists."""
        assert os.path.exists("scripts/runners/run_all_experiments_local.py")
    
    def test_local_runner_imports(self):
        """Verify local runner can be imported."""
        sys.path.insert(0, 'scripts/runners')
        try:
            import run_all_experiments_local
            assert hasattr(run_all_experiments_local, 'main')
            assert hasattr(run_all_experiments_local, 'ExperimentLogger')
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("Cannot import without ML dependencies")
            raise
    
    def test_experiment_logger_creates_directory(self, tmp_path):
        """Test ExperimentLogger creates log directory."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_all_experiments_local import ExperimentLogger
            
            log_dir = tmp_path / "test_logs"
            logger = ExperimentLogger(str(log_dir))
            
            assert log_dir.exists()
            assert logger.log_file.exists() or logger.log_file.parent.exists()
        except ImportError:
            pytest.skip("Cannot import without dependencies")
    
    def test_experiment_logger_saves_incrementally(self, tmp_path):
        """Test logger saves results incrementally."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_all_experiments_local import ExperimentLogger
            
            logger = ExperimentLogger(str(tmp_path / "test"))
            logger.log_result("test_exp", 42, "SUCCESS", 10.5)
            
            results_file = logger.log_dir / "experiment_results.json"
            assert results_file.exists()
            
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            assert len(results) == 1
            assert results[0]['experiment'] == 'test_exp'
            assert results[0]['status'] == 'SUCCESS'
        except ImportError:
            pytest.skip("Cannot import without dependencies")


class TestExperimentCommands:
    """Test that experiment commands are correct."""
    
    def test_complex_scm_command_structure(self):
        """Verify complex SCM command includes all required flags."""
        with open("scripts/runners/run_all_experiments_local.py") as f:
            content = f.read()
        
        # Should include these flags
        assert "--complex-scm" in content
        assert "--seeds" in content
        assert "run_critical_experiments.py" in content
    
    def test_ablation_commands_use_custom(self):
        """Verify ablations use --custom flag (no model download)."""
        with open("scripts/runners/run_all_experiments_local.py") as f:
            content = f.read()
        
        # All ablations should use --custom
        assert "'--custom --no_per_node_convergence'" in content or '"--custom --no_per_node_convergence"' in content
        assert "'--custom --no_dedicated_root_learner'" in content or '"--custom --no_dedicated_root_learner"' in content
        assert "'--custom --no_diversity_reward'" in content or '"--custom --no_diversity_reward"' in content
    
    def test_no_oracle_uses_custom(self):
        """Verify no-oracle uses --custom and pretrain_steps=0."""
        with open("scripts/runners/run_all_experiments_local.py") as f:
            content = f.read()
        
        # No-oracle should use --custom and --pretrain_steps 0
        assert '"--custom"' in content or "'--custom'" in content
        assert '"--pretrain_steps", "0"' in content or "'--pretrain_steps', '0'" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
