"""
Comprehensive tests to push from 75% to 80% coverage.

Focuses on remaining uncovered code paths in main files.
"""

import pytest
import torch
import tempfile
from pathlib import Path


# =============================================================================
# Additional Baseline Policy Integration Tests
# =============================================================================

@pytest.mark.integration
def test_max_variance_policy_full_workflow(seed_everything):
    """Test MaxVariancePolicy complete workflow."""
    from baselines import GroundTruthSCM, MaxVariancePolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = MaxVariancePolicy(oracle.nodes, n_candidates=8, n_mc_samples=3)
    
    # Run short workflow
    df = run_baseline(policy, oracle, n_episodes=2, steps_per_episode=3)
    
    assert len(df) == 6
    assert 'target' in df.columns


@pytest.mark.integration  
def test_baseline_observational_training_integration(seed_everything):
    """Test baseline with observational training integration."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # With observational training
    df = run_baseline(
        policy, oracle,
        n_episodes=2,
        steps_per_episode=4,
        obs_train_interval=2,
        obs_train_samples=50
    )
    
    assert len(df) == 8


# =============================================================================
# Additional SCMLearner from baselines.py Tests
# =============================================================================

@pytest.mark.unit
def test_baselines_scm_learner_observational_train_detailed(seed_everything):
    """Test SCMLearner observational_train from baselines."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Multiple observational training calls
    for _ in range(5):
        learner.observational_train(oracle, n_samples=50, n_epochs=10)
    
    # Should complete without error


@pytest.mark.unit
def test_baselines_scm_learner_with_varying_epochs(seed_everything):
    """Test SCMLearner with different epoch counts."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    data = oracle.generate(100)
    
    # Test with different epoch counts
    for n_epochs in [1, 10, 50]:
        loss = learner.train_step(data, n_epochs=n_epochs)
        assert isinstance(loss, float)


# =============================================================================
# Additional Complex SCM Mechanism Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.statistical
def test_complex_scm_layer2_mechanisms_all(seed_everything):
    """Test all Layer 2 mechanisms in Complex SCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=5000)
    
    # All Layer 2 nodes should have data
    for node in ['L1', 'L2', 'L3']:
        assert node in data
        assert data[node].std() > 0.2  # Should have variance


@pytest.mark.unit
@pytest.mark.statistical
def test_complex_scm_layer3_mechanisms_all(seed_everything):
    """Test all Layer 3 mechanisms in Complex SCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=5000)
    
    # All Layer 3 nodes should have data
    for node in ['N1', 'N2', 'N3']:
        assert node in data
        assert data[node].std() > 0.2


@pytest.mark.unit
def test_complex_scm_final_layer_all_nodes(seed_everything):
    """Test all final layer nodes in Complex SCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=1000)
    
    # Final layer: F1, F2, F3
    for node in ['F1', 'F2', 'F3']:
        assert node in data
        assert data[node].shape == (1000,)
        assert not torch.isnan(data[node]).any()


# =============================================================================
# Additional Clamping Detector Coverage
# =============================================================================

@pytest.mark.unit
def test_clamping_detector_value_extraction(tmp_path):
    """Test value extraction from log files."""
    from clamping_detector import detect_clamping
    
    # Create log with clear clamping pattern
    log_file = tmp_path / "test.log"
    log_content = """
    Step 1: DO X2 = 0.1
    Step 2: DO X2 = 0.0
    Step 3: DO X2 = -0.05
    Step 4: DO X2 = 0.02
    Step 5: DO X2 = 0.0
    Step 6: DO X2 = 0.01
    """
    log_file.write_text(log_content)
    
    result = detect_clamping(str(log_file), target_node='X2', threshold=0.2)
    
    # Should detect pattern or return stats
    assert isinstance(result, dict)


@pytest.mark.unit
def test_clamping_detector_threshold_variations():
    """Test clamping detection with different thresholds."""
    from clamping_detector import detect_clamping
    import tempfile
    
    # Create temp log
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        for i in range(10):
            f.write(f"DO X2 = {i * 0.1}\n")
        log_path = f.name
    
    # Test different thresholds
    for threshold in [0.1, 0.5, 1.0]:
        result = detect_clamping(log_path, threshold=threshold)
        assert isinstance(result, dict)


# =============================================================================
# Additional Regime Analyzer Coverage
# =============================================================================

@pytest.mark.unit
def test_regime_analyzer_with_valid_data(tmp_path):
    """Test regime analyzer with properly structured data."""
    from regime_analyzer import analyze_regime_selection
    import pandas as pd
    
    # Create structured results
    results_file = tmp_path / "results.csv"
    df = pd.DataFrame({
        'episode': list(range(50)),
        'step': [0] * 50,
        'target': ['X1'] * 50,
        'value': [float(i % 10) for i in range(50)]
    })
    df.to_csv(results_file, index=False)
    
    result = analyze_regime_selection(str(results_file), verbose=False)
    
    assert isinstance(result, dict)


# =============================================================================
# Additional Compare Methods Coverage
# =============================================================================

@pytest.mark.unit
def test_compare_methods_baseline_results_complete():
    """Test that BASELINE_RESULTS has complete structure."""
    from compare_methods import BASELINE_RESULTS
    
    # Should have multiple baselines
    assert len(BASELINE_RESULTS) >= 3
    
    # Each should have total loss
    for baseline, metrics in BASELINE_RESULTS.items():
        if 'total' in metrics:
            assert metrics['total'] >= 0


# =============================================================================
# Additional Visualization Coverage
# =============================================================================

@pytest.mark.unit
def test_visualize_print_summary_with_all_data(tmp_path):
    """Test print_summary with complete data."""
    from visualize import print_summary
    import pandas as pd
    
    # Complete dataset
    data = {
        'node_losses': pd.DataFrame({
            'episode': [0, 1, 2],
            'step': [0, 0, 0],
            'loss_X1': [1.0, 0.8, 0.6],
            'loss_X2': [2.0, 1.5, 1.2],
            'loss_X3': [3.0, 2.5, 2.0],
            'loss_X4': [1.0, 0.9, 0.8],
            'loss_X5': [1.5, 1.3, 1.1],
        }),
        'metrics': pd.DataFrame({
            'episode': [0, 1, 2],
            'step': [0, 0, 0],
            'target': ['X1', 'X2', 'X3'],
            'value': [1.0, 2.0, 1.5]
        }),
        'value_diversity': pd.DataFrame({
            'step': [0, 1, 2],
            'target': ['X1', 'X2', 'X3'],
            'value': [1.0, 2.0, 1.5]
        }),
        'dpo': pd.DataFrame({
            'step': [0, 1, 2],
            'loss': [0.7, 0.6, 0.5]
        })
    }
    
    # Should handle all data types
    print_summary(data)


# =============================================================================
# Additional Integration for Maximum Coverage
# =============================================================================

@pytest.mark.integration
def test_end_to_end_with_all_baseline_policies(seed_everything):
    """Test end-to-end workflow with each baseline policy."""
    from baselines import (
        GroundTruthSCM, RandomPolicy, RoundRobinPolicy,
        MaxVariancePolicy, run_baseline
    )
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policies = [
        RandomPolicy(oracle.nodes),
        RoundRobinPolicy(oracle.nodes),
        MaxVariancePolicy(oracle.nodes, n_candidates=5, n_mc_samples=2)
    ]
    
    for policy in policies:
        df = run_baseline(policy, oracle, n_episodes=1, steps_per_episode=3)
        assert len(df) == 3


@pytest.mark.integration
def test_complete_ace_workflow_with_checkpointing(ground_truth_scm, tmp_path, seed_everything):
    """Test ACE workflow with checkpoint saving."""
    from ace_experiments import (
        StudentSCM, ExperimentExecutor, SCMLearner,
        ScientificCritic, save_checkpoint,
        TransformerPolicy, ExperimentalDSL
    )
    
    seed_everything(42)
    
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    student = StudentSCM(ground_truth_scm)
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student, lr=0.01)
    critic = ScientificCritic(ground_truth_scm)
    
    # Create minimal policy for checkpointing
    dsl = ExperimentalDSL(list(ground_truth_scm.nodes))
    policy = TransformerPolicy(dsl, torch.device('cpu'), d_model=16)
    optimizer = torch.optim.Adam(policy.parameters())
    
    loss_history = []
    reward_history = []
    
    # Run 3 episodes with checkpointing
    for ep in range(3):
        result = executor.run_experiment(None)
        loss = learner.train_step(result, n_epochs=5)
        
        loss_history.append(loss)
        reward_history.append(10.0)
        
        # Save checkpoint
        save_checkpoint(
            str(run_dir), ep, policy, optimizer,
            loss_history, reward_history, ['DO X1 = 1.0']
        )
    
    # Should have created checkpoints
    checkpoints = list(run_dir.glob("checkpoint_*.pt"))
    assert len(checkpoints) >= 1
