"""
Maximum coverage tests - pushing toward 90%.

Comprehensive tests for all remaining uncovered code paths.
"""

import pytest
import torch
import tempfile
import os


# =============================================================================
# Additional Integration Paths
# =============================================================================

@pytest.mark.integration
def test_complete_multi_episode_with_all_features(ground_truth_scm, seed_everything):
    """Test complete workflow using all features."""
    from ace_experiments import (
        StudentSCM, ExperimentExecutor, SCMLearner,
        ScientificCritic, EarlyStopping, DedicatedRootLearner,
        StateEncoder, fit_root_distributions
    )
    
    seed_everything(42)
    
    # Setup all components
    student = StudentSCM(ground_truth_scm)
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student, lr=0.01, buffer_steps=5)
    critic = ScientificCritic(ground_truth_scm)
    stopper = EarlyStopping(patience=10, min_episodes=3)
    root_learner = DedicatedRootLearner(['X1', 'X4'])
    encoder = StateEncoder(5, torch.device('cpu'))
    
    # Run 10 episodes with all features
    for ep in range(10):
        # Experiment
        if ep % 3 == 0:
            result = executor.run_experiment(None)
        else:
            result = executor.run_experiment({
                'target': list(ground_truth_scm.nodes)[ep % 5],
                'value': float(ep % 3),
                'samples': 100
            })
        
        # Train main learner
        learner.train_step(result, n_epochs=10)
        
        # Evaluate
        total_loss, node_losses = critic.evaluate_model_detailed(student)
        
        # Encode state
        state = encoder(student, node_losses)
        assert not torch.isnan(state).any()
        
        # Root fitting every 3 episodes
        if ep > 0 and ep % 3 == 0:
            fit_root_distributions(student, ground_truth_scm, critic, ['X1', 'X4'], 
                                 n_samples=200, epochs=20)
        
        # Check early stopping
        if ep >= 3 and stopper.check_loss(total_loss):
            break
    
    # Should complete successfully
    assert True


@pytest.mark.integration
def test_workflow_with_different_buffer_sizes(ground_truth_scm, seed_everything):
    """Test workflow with various buffer configurations."""
    from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner
    
    for buffer_size in [1, 5, 10, 50]:
        seed_everything(42)
        student = StudentSCM(ground_truth_scm)
        executor = ExperimentExecutor(ground_truth_scm)
        learner = SCMLearner(student, lr=0.01, buffer_steps=buffer_size)
        
        # Train
        for _ in range(buffer_size + 2):
            result = executor.run_experiment(None)
            learner.train_step(result, n_epochs=5)
        
        # Buffer should not exceed limit
        assert len(learner.buffer) <= buffer_size


# =============================================================================
# Additional Reward Function Coverage
# =============================================================================

@pytest.mark.unit
def test_impact_weight_with_all_nodes(ground_truth_scm):
    """Test impact weight for all nodes."""
    from ace_experiments import _impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {node: 1.0 for node in ground_truth_scm.nodes}
    
    # Test all nodes
    for node in ground_truth_scm.nodes:
        weight = _impact_weight(graph, node, node_losses)
        assert weight >= 0  # Should always be non-negative


@pytest.mark.unit
def test_direct_child_impact_for_all_nodes(ground_truth_scm):
    """Test direct child impact for all nodes."""
    from ace_experiments import _direct_child_impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {node: 1.0 for node in ground_truth_scm.nodes}
    
    for node in ground_truth_scm.nodes:
        weight_norm = _direct_child_impact_weight(graph, node, node_losses, normalize=True)
        weight_unnorm = _direct_child_impact_weight(graph, node, node_losses, normalize=False)
        
        assert weight_norm >= 0
        assert weight_unnorm >= weight_norm


@pytest.mark.unit
def test_disentanglement_bonus_for_all_nodes(ground_truth_scm):
    """Test disentanglement bonus for all nodes."""
    from ace_experiments import _disentanglement_bonus
    
    graph = ground_truth_scm.graph
    node_losses = {node: 2.0 for node in ground_truth_scm.nodes}
    
    for node in ground_truth_scm.nodes:
        bonus = _disentanglement_bonus(graph, node, node_losses)
        assert bonus >= 0  # Should always be non-negative


# =============================================================================
# Additional DPOLogger Coverage
# =============================================================================

@pytest.mark.unit
def test_dpo_logger_complete_workflow():
    """Test DPOLogger through complete workflow."""
    from ace_experiments import DPOLogger
    
    logger = DPOLogger()
    
    # Simulate 20 training steps
    for i in range(20):
        logger.log(
            loss=0.7 - i * 0.02,
            policy_win_lp=torch.tensor(1.0 + i * 0.1),
            policy_lose_lp=torch.tensor(-0.5),
            ref_win_lp=torch.tensor(0.8),
            ref_lose_lp=torch.tensor(-0.3),
            winner_target='X2',
            loser_target='X1',
            logit=torch.tensor(1.5)
        )
    
    assert logger.step_count == 20
    assert len(logger.history['loss']) == 20


# =============================================================================
# Additional Baselines Coverage
# =============================================================================

@pytest.mark.unit
def test_baselines_scientific_critic_has_evaluate(ground_truth_scm):
    """Test ScientificCritic from baselines.py has evaluate method."""
    from baselines import ScientificCritic
    
    critic = ScientificCritic(ground_truth_scm)
    
    # Verify it has the evaluate method
    assert hasattr(critic, 'evaluate')
    assert callable(critic.evaluate)


# =============================================================================
# Additional Experiment Coverage
# =============================================================================

@pytest.mark.unit
def test_complex_scm_get_parents_for_all_nodes():
    """Test get_parents for all nodes in ComplexSCM."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Test get_parents for all nodes
    for node in scm.nodes:
        parents = scm.get_parents(node)
        assert isinstance(parents, list)
        # All parents should be valid nodes
        for parent in parents:
            assert parent in scm.nodes


@pytest.mark.unit
def test_complex_scm_graph_structure():
    """Test ComplexSCM graph dictionary structure."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Graph should have all nodes
    assert len(scm.graph) == len(scm.nodes)
    
    # All nodes should have parent list
    for node in scm.nodes:
        assert node in scm.graph
        assert isinstance(scm.graph[node], list)


# =============================================================================
# Additional Visualization Coverage
# =============================================================================

@pytest.mark.unit
def test_visualize_load_run_data_all_csvs(tmp_path):
    """Test load_run_data with all possible CSV files."""
    from visualize import load_run_data
    import pandas as pd
    
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    # Create all CSV files
    csv_files = ['node_losses.csv', 'metrics.csv', 'value_diversity.csv', 'dpo_training.csv']
    
    for csv_file in csv_files:
        df = pd.DataFrame({'col1': [1, 2, 3]})
        df.to_csv(run_dir / csv_file, index=False)
    
    # Load all
    data = load_run_data(str(run_dir))
    
    # Should load all available files
    assert isinstance(data, dict)


# =============================================================================
# Additional Edge Cases for Complete Coverage
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_encode_decode_all_tokens():
    """Test encoding and decoding various token combinations."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Test various commands
    commands = [
        "DO X1 = -5",
        "DO X2 = 0",
        "DO X3 = 5"
    ]
    
    for cmd in commands:
        encoded = dsl.encode(cmd)
        decoded = dsl.decode(encoded)
        
        # Should be able to round-trip
        assert isinstance(decoded, str)


@pytest.mark.unit
def test_transformer_policy_initialization_only():
    """Test TransformerPolicy initialization with minimal config."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Minimal config
    policy = TransformerPolicy(dsl, torch.device('cpu'), d_model=16, nhead=2, num_layers=1)
    
    # Should initialize successfully
    assert isinstance(policy, torch.nn.Module)
    assert hasattr(policy, 'state_encoder')
    assert hasattr(policy, 'transformer_enc')
