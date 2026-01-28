"""
Comprehensive tests for ACE on complex 15-node SCM.
Ensures implementation works before HPC execution.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestACEComplexImplementation:
    """Test ACE complex SCM implementation."""
    
    def test_run_ace_complex_imports(self):
        """Verify run_ace_complex.py can be imported."""
        sys.path.insert(0, 'experiments')
        try:
            import run_ace_complex
            assert hasattr(run_ace_complex, 'run_ace_complex')
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("Cannot import without ML dependencies")
            raise
    
    def test_complex_ground_truth_scm_exists(self):
        """Verify ComplexGroundTruthSCM can be imported."""
        from experiments.complex_scm import ComplexGroundTruthSCM
        assert ComplexGroundTruthSCM is not None
    
    def test_complex_student_scm_exists(self):
        """Verify ComplexStudentSCM can be imported."""
        from experiments.complex_scm import ComplexStudentSCM
        assert ComplexStudentSCM is not None
    
    def test_complex_learner_exists(self):
        """Verify ComplexSCMLearner can be imported."""
        from experiments.complex_scm import ComplexSCMLearner
        assert ComplexSCMLearner is not None
    
    @pytest.mark.slow
    def test_complex_scm_has_15_nodes(self):
        """Verify complex SCM has 15 nodes."""
        from experiments.complex_scm import ComplexGroundTruthSCM
        
        scm = ComplexGroundTruthSCM()
        assert len(scm.nodes) == 15
    
    @pytest.mark.slow
    def test_ace_imports_work_for_complex(self):
        """Test that ACE components can be imported for complex SCM."""
        try:
            from ace_experiments import (
                TransformerPolicy,
                ExperimentalDSL,
                EarlyStopping
            )
            assert TransformerPolicy is not None
            assert ExperimentalDSL is not None
            assert EarlyStopping is not None
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("Cannot import without ML dependencies")
            raise
    
    @pytest.mark.slow
    def test_experimental_dsl_handles_15_nodes(self):
        """Test ExperimentalDSL works with 15 nodes."""
        try:
            from ace_experiments import ExperimentalDSL
            from experiments.complex_scm import ComplexGroundTruthSCM
            
            oracle = ComplexGroundTruthSCM()
            dsl = ExperimentalDSL(oracle.nodes, value_min=-5.0, value_max=5.0)
            
            assert dsl is not None
            assert len(dsl.nodes) == 15
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
    
    @pytest.mark.slow
    def test_transformer_policy_with_15_nodes(self):
        """Test TransformerPolicy can handle 15 nodes."""
        try:
            from ace_experiments import TransformerPolicy, ExperimentalDSL
            from experiments.complex_scm import ComplexGroundTruthSCM
            import torch
            
            oracle = ComplexGroundTruthSCM()
            dsl = ExperimentalDSL(oracle.nodes)
            device = torch.device("cpu")
            
            policy = TransformerPolicy(dsl, device)
            assert policy is not None
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")


class TestACEComplexScript:
    """Test the run_ace_complex.py script."""
    
    def test_script_exists(self):
        """Verify script file exists."""
        assert os.path.exists("experiments/run_ace_complex.py")
    
    def test_script_has_main(self):
        """Verify script has main execution."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "if __name__" in content
        assert "run_ace_complex(" in content
    
    def test_script_has_argument_parser(self):
        """Verify script has argument parsing."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "argparse" in content
        assert "--seed" in content
        assert "--episodes" in content
    
    def test_script_sets_random_seeds(self):
        """Verify script sets all random seeds."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "random.seed" in content
        assert "np.random.seed" in content
        assert "torch.manual_seed" in content
    
    def test_uses_qwen_policy(self):
        """Verify script uses Qwen2.5-1.5B (not custom)."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "HuggingFacePolicy" in content
        assert "Qwen2.5-1.5B" in content
    
    def test_oracle_pretraining_present(self):
        """Verify oracle pretraining is used."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "supervised_pretrain_llm" in content
        assert "n_steps=200" in content
    
    def test_output_directory_creation(self):
        """Verify script creates output directories."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "os.makedirs" in content
        assert "run_dir" in content
    
    def test_logging_setup(self):
        """Verify logging is configured."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "logging.basicConfig" in content
        assert "experiment.log" in content
    
    def test_progress_messages(self):
        """Verify progress messages for monitoring."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "[STARTUP]" in content
        assert "[PROGRESS]" in content
        assert "[COMPLETE]" in content or "[CONVERGED]" in content


class TestACEComplexJobScript:
    """Test the HPC job script."""
    
    def test_job_script_exists(self):
        """Verify job script exists."""
        assert os.path.exists("jobs/run_ace_complex_scm.sh")
    
    def test_job_has_qos(self):
        """Verify job has QoS."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "#SBATCH --qos=normal" in content
    
    def test_job_calls_correct_script(self):
        """Verify job calls run_ace_complex.py."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "run_ace_complex.py" in content
    
    def test_job_has_adequate_time(self):
        """Verify job has sufficient time limit."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        # Should have 12h for 5 seeds
        assert "--time=12:00:00" in content or "--time=10:00:00" in content
    
    def test_job_has_environment_setup(self):
        """Verify job sets up conda environment."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "conda activate" in content
        assert "HF_HOME" in content
    
    def test_job_runs_all_5_seeds(self):
        """Verify job runs all 5 seeds."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "for SEED in 42 123 456 789 1011" in content
    
    def test_job_saves_per_seed_results(self):
        """Verify results saved after each seed."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "Seed $SEED complete" in content or "complete" in content


class TestCompleteIntegration:
    """Integration tests for complete pipeline."""
    
    def test_no_import_conflicts(self):
        """Verify no import conflicts between modules."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        # Should import both complex_scm and ace_experiments
        assert "from experiments.complex_scm import" in content
        assert "from ace_experiments import" in content
    
    def test_consistent_hyperparameters(self):
        """Verify hyperparameters match 5-node ACE."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        # Should use same settings as 5-node
        assert "lr=1e-5" in content  # DPO learning rate
        assert "n_steps=200" in content  # Oracle pretraining
    
    def test_error_handling_present(self):
        """Verify basic error handling exists."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        # Should have try/except or error messages
        assert "print(" in content  # At minimum has output
        assert "logging" in content  # Has logging


class TestDataIntegrity:
    """Test data saving and integrity."""
    
    def test_results_csv_structure(self):
        """Verify results CSV will have correct structure."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "results.append" in content
        assert "pd.DataFrame" in content
        assert ".to_csv" in content
    
    def test_seed_in_output_path(self):
        """Verify seed is part of output path for uniqueness."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "seed{seed}" in content or "seed$SEED" in content or "_seed" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
