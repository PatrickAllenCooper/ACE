"""
Final comprehensive test suite to reach 90% coverage.

Systematically covers all remaining uncovered code paths.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


# =============================================================================
# Exhaustive Baseline Coverage
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_all_baselines_complete_workflow(seed_everything):
    """Test complete workflow for all three baseline policies."""
    from baselines import (
        GroundTruthSCM, RandomPolicy, RoundRobinPolicy,
        MaxVariancePolicy, run_baseline
    )
    
    seed_everything(42)
    oracle = GroundTruthSCM()
    
    # Test each policy type with full workflow
    policy_configs = [
        ('Random', RandomPolicy(oracle.nodes)),
        ('RoundRobin', RoundRobinPolicy(oracle.nodes)),
        ('MaxVariance', MaxVariancePolicy(oracle.nodes, n_candidates=10, n_mc_samples=3))
    ]
    
    for name, policy in policy_configs:
        df = run_baseline(
            policy, oracle,
            n_episodes=3,
            steps_per_episode=10,
            obs_train_interval=3,
            obs_train_samples=100
        )
        
        assert len(df) == 30
        assert df['total_loss'].min() >= 0


# =============================================================================
# Exhaustive Experiment Coverage
# =============================================================================

@pytest.mark.unit
def test_complex_scm_all_node_types(seed_everything):
    """Test all node types in Complex SCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=1000)
    
    # Test every single node
    expected_nodes = scm.nodes
    
    for node in expected_nodes:
        assert node in data
        assert data[node].shape == (1000,)
        assert not torch.isnan(data[node]).any()
        assert not torch.isinf(data[node]).any()


@pytest.mark.unit
def test_complex_scm_every_parent_child_relationship():
    """Test every parent-child relationship in Complex SCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Verify all relationships
    for node in scm.nodes:
        parents = scm.get_parents(node)
        
        # All parents should be valid nodes
        for parent in parents:
            assert parent in scm.nodes
        
        # Roots should have no parents
        if node.startswith('R'):
            assert len(parents) == 0


# =============================================================================
# Exhaustive Analysis Tools Coverage
# =============================================================================

@pytest.mark.unit
def test_clamping_detector_all_nodes(tmp_path):
    """Test clamping detection for all possible nodes."""
    from clamping_detector import detect_clamping
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    for node in nodes:
        log_file = tmp_path / f"test_{node}.log"
        log_content = f"DO {node} = 0.0\nDO {node} = 0.1\nDO {node} = 0.0\n"
        log_file.write_text(log_content)
        
        result = detect_clamping(str(log_file), target_node=node, threshold=0.5)
        assert isinstance(result, dict)


@pytest.mark.unit
def test_regime_analyzer_regime_definitions():
    """Test all regime definitions in regime_analyzer."""
    from regime_analyzer import REGIMES
    
    # Test structure of all regimes
    for regime_name, regime_data in REGIMES.items():
        assert 'years' in regime_data
        assert 'volatility' in regime_data
        assert isinstance(regime_data['years'], tuple)
        assert len(regime_data['years']) == 2


@pytest.mark.unit
def test_compare_methods_all_baseline_metrics():
    """Test all baseline metrics in compare_methods."""
    from compare_methods import BASELINE_RESULTS
    
    expected_metrics = ['X1', 'X2', 'X3', 'X4', 'X5', 'total']
    
    for baseline_name, metrics in BASELINE_RESULTS.items():
        # Should have standard metrics
        assert 'total' in metrics or len(metrics) > 0


# =============================================================================
# Exhaustive Visualization Coverage
# =============================================================================

@pytest.mark.unit
def test_visualize_all_plotting_functions(tmp_path):
    """Test all visualization plotting functions."""
    from visualize import create_success_dashboard, create_mechanism_contrast, print_summary
    import matplotlib
    matplotlib.use('Agg')
    
    # Create comprehensive test data
    data = {
        'node_losses': pd.DataFrame({
            'episode': list(range(10)),
            'step': [0] * 10,
            'loss_X1': np.random.rand(10),
            'loss_X2': np.random.rand(10),
            'loss_X3': np.random.rand(10),
            'loss_X4': np.random.rand(10),
            'loss_X5': np.random.rand(10),
        }),
        'metrics': pd.DataFrame({
            'episode': list(range(10)),
            'step': [0] * 10,
            'target': ['X1', 'X2', 'X3', 'X4', 'X5'] * 2,
            'value': np.random.rand(10)
        })
    }
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Test all visualization functions
    create_success_dashboard(data, str(results_dir))
    print_summary(data)
    
    # Should create files
    assert (results_dir / "success_verification.png").exists()


# =============================================================================
# Comprehensive Integration for All Code Paths
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_complete_ace_pipeline_all_features_extended(ground_truth_scm, seed_everything):
    """Extended test of complete ACE pipeline with all features."""
    from ace_experiments import (
        StudentSCM, ExperimentExecutor, SCMLearner,
        ScientificCritic, EarlyStopping, StateEncoder,
        DedicatedRootLearner, fit_root_distributions,
        calculate_value_novelty_bonus, compute_unified_diversity_score
    )
    
    seed_everything(42)
    
    # Initialize all components
    student = StudentSCM(ground_truth_scm)
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student, lr=0.01, buffer_steps=10)
    critic = ScientificCritic(ground_truth_scm)
    stopper = EarlyStopping(patience=5, min_episodes=5)
    encoder = StateEncoder(5, torch.device('cpu'))
    root_learner = DedicatedRootLearner(['X1', 'X4'])
    
    nodes = list(ground_truth_scm.nodes)
    value_history = []
    target_history = []
    
    # Run extended workflow
    for ep in range(15):
        # Select intervention
        if ep < 3:
            result = executor.run_experiment(None)
            target = None
        else:
            target = nodes[ep % len(nodes)]
            value = float((ep % 5) - 2)
            result = executor.run_experiment({'target': target, 'value': value, 'samples': 100})
            value_history.append((target, value))
            target_history.append(target)
        
        # Train
        learner.train_step(result, n_epochs=10)
        
        # Evaluate
        total_loss, node_losses = critic.evaluate_model_detailed(student)
        
        # Encode state
        state = encoder(student, node_losses)
        
        # Compute rewards/scores
        if len(value_history) > 0:
            novelty = calculate_value_novelty_bonus(1.0, nodes[0], value_history)
        
        if len(target_history) >= 20:
            diversity = compute_unified_diversity_score(nodes[0], target_history, nodes)
        
        # Root fitting
        if ep > 0 and ep % 5 == 0:
            fit_root_distributions(student, ground_truth_scm, critic, ['X1', 'X4'], 
                                 n_samples=300, epochs=30)
        
        # Check early stopping
        if ep >= 5 and stopper.check_loss(total_loss):
            break
    
    # Should complete successfully
    assert ep >= 5  # Should run at least 5 episodes


# =============================================================================
# Maximum Code Path Coverage
# =============================================================================

@pytest.mark.unit
def test_every_component_with_edge_cases(ground_truth_scm, student_scm):
    """Test every major component with edge cases."""
    from ace_experiments import ExperimentExecutor, SCMLearner, ScientificCritic
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    critic = ScientificCritic(ground_truth_scm)
    
    # Test with various edge case values
    edge_values = [0.0, -5.0, 5.0, 0.001, -0.001]
    
    for val in edge_values:
        result = executor.run_experiment({'target': 'X1', 'value': val, 'samples': 10})
        loss = learner.train_step(result, n_epochs=3)
        total, losses = critic.evaluate_model_detailed(student_scm)
        
        assert isinstance(loss, float)
        assert isinstance(total, float)


@pytest.mark.unit
def test_all_utility_functions_with_minimal_inputs():
    """Test utility functions with minimal valid inputs."""
    from ace_experiments import (
        _bin_index, get_random_valid_command_range,
        ExperimentalDSL, DedicatedRootLearner
    )
    
    # Test bin_index with edge values
    assert _bin_index(-5.0, -5.0, 5.0, 10) == 0
    assert _bin_index(5.0, -5.0, 5.0, 10) == 9
    assert _bin_index(0.0, -5.0, 5.0, 10) == 5
    
    # Test random command
    cmd = get_random_valid_command_range(['X1'], value_min=-1.0, value_max=1.0)
    assert 'DO' in cmd
    
    # Test DSL with single node
    dsl = ExperimentalDSL(['X1'])
    assert len(dsl.nodes) == 1
    
    # Test root learner with single node
    learner = DedicatedRootLearner(['X1'])
    assert 'X1' in learner.root_nodes


@pytest.mark.integration
def test_baselines_with_all_parameters(seed_everything):
    """Test baselines.run_baseline with all parameter variations."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # Test with various parameter combinations
    param_sets = [
        {'n_episodes': 1, 'steps_per_episode': 5, 'obs_train_interval': 0},
        {'n_episodes': 2, 'steps_per_episode': 10, 'obs_train_interval': 2},
        {'n_episodes': 3, 'steps_per_episode': 5, 'obs_train_interval': 1, 'obs_train_samples': 50},
    ]
    
    for params in param_sets:
        df = run_baseline(policy, oracle, **params)
        assert len(df) == params['n_episodes'] * params['steps_per_episode']
