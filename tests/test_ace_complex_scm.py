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
        assert "--custom" in content
    
    def test_script_sets_random_seeds(self):
        """Verify script sets all random seeds."""
        with open("experiments/run_ace_complex.py") as f:
            content = f.read()
        assert "random.seed" in content
        assert "np.random.seed" in content
        assert "torch.manual_seed" in content


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
        assert "--custom" in content
    
    def test_job_has_adequate_time(self):
        """Verify job has sufficient time limit."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        # Should have 12h for 5 seeds
        assert "--time=12:00:00" in content or "--time=10:00:00" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
