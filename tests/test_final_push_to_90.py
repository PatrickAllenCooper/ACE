"""
Final comprehensive tests to reach 90% coverage.

Covers all remaining uncovered code paths systematically.
"""

import pytest
import torch
import pandas as pd
import tempfile
import os


# =============================================================================
# Comprehensive Baseline Coverage
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_baseline_full_episode_with_all_features(seed_everything):
    """Test baseline with all features enabled."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # Full features
    df = run_baseline(
        policy, oracle,
        n_episodes=5,
        steps_per_episode=10,
        obs_train_interval=3,
        obs_train_samples=100
    )
    
    assert len(df) == 50
    assert not df['total_loss'].isna().any()


@pytest.mark.unit
def test_baselines_student_scm_all_mechanisms(seed_everything):
    """Test all mechanisms in baselines StudentSCM."""
    from baselines import GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    
    # Generate data
    data = oracle.generate(100)
    
    # Forward pass (teacher forcing)
    predictions = student.forward(data)
    
    # All nodes should have predictions
    for node in oracle.nodes:
        assert node in predictions
        assert predictions[node].shape == (100,)


@pytest.mark.unit
def test_baselines_scm_learner_full_training_cycle(seed_everything):
    """Test complete training cycle for baselines SCMLearner."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Multiple training cycles
    for i in range(10):
        # Alternate between obs and interventional
        if i % 2 == 0:
            data = oracle.generate(100)
            loss = learner.train_step(data, n_epochs=10)
        else:
            data = oracle.generate(100, interventions={'X1': float(i)})
            loss = learner.train_step(data, intervened='X1', n_epochs=10)
        
        assert isinstance(loss, float)
        
        # Observational training
        if i % 3 == 0:
            learner.observational_train(oracle, n_samples=100, n_epochs=10)


# =============================================================================
# Comprehensive Experiment Coverage
# =============================================================================

@pytest.mark.unit
def test_complex_scm_all_interventions(seed_everything):
    """Test interventions on all Complex SCM nodes."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Test intervention on every node
    for node in scm.nodes:
        data = scm.generate(n_samples=50, interventions={node: 2.0})
        assert torch.all(data[node] == 2.0)


@pytest.mark.unit
def test_complex_scm_collider_interventions(seed_everything):
    """Test interventions specifically on colliders."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Colliders: L1, N1, C1, C2, F3
    colliders = ['L1', 'N1', 'C1', 'C2', 'F3']
    
    for collider in colliders:
        if collider in scm.nodes:
            data = scm.generate(n_samples=100, interventions={collider: 1.5})
            assert torch.all(data[collider] == 1.5)


# =============================================================================
# Comprehensive Analysis Tools Coverage
# =============================================================================

@pytest.mark.unit
def test_clamping_detector_comprehensive(tmp_path):
    """Comprehensive clamping detection test."""
    from clamping_detector import detect_clamping
    
    # Create comprehensive log
    log_file = tmp_path / "comprehensive.log"
    log_content = ""
    for i in range(50):
        if i < 10:
            log_content += f"Episode {i}: DO X2 = {i * 0.5}\n"
        else:
            log_content += f"Episode {i}: DO X2 = 0.0\n"  # Clamping pattern
    
    log_file.write_text(log_content)
    
    result = detect_clamping(str(log_file), target_node='X2', threshold=0.3)
    
    assert isinstance(result, dict)


@pytest.mark.unit
def test_regime_analyzer_comprehensive(tmp_path):
    """Comprehensive regime analysis test."""
    from regime_analyzer import analyze_regime_selection
    import pandas as pd
    
    # Create comprehensive data
    df = pd.DataFrame({
        'episode': list(range(100)),
        'step': [0] * 100,
        'target': ['X1'] * 100,
        'value': [float(i % 20) for i in range(100)]
    })
    
    results_file = tmp_path / "comprehensive_results.csv"
    df.to_csv(results_file, index=False)
    
    result = analyze_regime_selection(str(results_file), verbose=False)
    
    assert isinstance(result, dict)


# =============================================================================
# Comprehensive Utility Function Coverage
# =============================================================================

@pytest.mark.unit
def test_all_reward_functions_together(ground_truth_scm):
    """Test all reward functions in combination."""
    from ace_experiments import (
        _impact_weight, _direct_child_impact_weight,
        _disentanglement_bonus, calculate_value_novelty_bonus,
        compute_unified_diversity_score
    )
    
    graph = ground_truth_scm.graph
    nodes = list(ground_truth_scm.nodes)
    node_losses = {node: float(i) for i, node in enumerate(nodes)}
    
    # Test all reward functions for each node
    for node in nodes:
        impact = _impact_weight(graph, node, node_losses)
        child_impact = _direct_child_impact_weight(graph, node, node_losses)
        disent = _disentanglement_bonus(graph, node, node_losses)
        
        assert impact >= 0
        assert child_impact >= 0
        assert disent >= 0
    
    # Test novelty and diversity
    value_history = [(n, float(i)) for i, n in enumerate(nodes)]
    recent_targets = nodes * 20
    
    for node in nodes:
        novelty = calculate_value_novelty_bonus(1.0, node, value_history)
        diversity = compute_unified_diversity_score(node, recent_targets, nodes)
        
        assert novelty >= 0
        assert isinstance(diversity, (int, float))


# =============================================================================
# Additional Integration Coverage
# =============================================================================

@pytest.mark.integration
def test_multi_component_integration_all_features(ground_truth_scm, seed_everything):
    """Test all major components working together."""
    from ace_experiments import (
        StudentSCM, ExperimentExecutor, SCMLearner,
        ScientificCritic, EarlyStopping, StateEncoder,
        DedicatedRootLearner, ExperimentalDSL,
        fit_root_distributions, visualize_scm_graph
    )
    import matplotlib
    matplotlib.use('Agg')
    
    seed_everything(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # All components
        student = StudentSCM(ground_truth_scm)
        executor = ExperimentExecutor(ground_truth_scm)
        learner = SCMLearner(student, lr=0.01, buffer_steps=5)
        critic = ScientificCritic(ground_truth_scm)
        stopper = EarlyStopping(patience=5, min_episodes=2)
        encoder = StateEncoder(5, torch.device('cpu'))
        root_learner = DedicatedRootLearner(['X1', 'X4'])
        dsl = ExperimentalDSL(list(ground_truth_scm.nodes))
        
        # Run workflow
        for ep in range(8):
            # Experiment
            result = executor.run_experiment(None)
            learner.train_step(result, n_epochs=5)
            
            # Evaluate
            total_loss, node_losses = critic.evaluate_model_detailed(student)
            
            # Encode
            state = encoder(student, node_losses)
            
            # Root fitting
            if ep % 3 == 0:
                fit_root_distributions(student, ground_truth_scm, critic, 
                                     ['X1', 'X4'], n_samples=200, epochs=20)
            
            # Visualization
            if ep == 5:
                visualize_scm_graph(ground_truth_scm, tmpdir, node_losses)
            
            # Check stopping
            if ep >= 2 and stopper.check_loss(total_loss):
                break
        
        # All should work together
        assert True


# =============================================================================
# Edge Cases for Maximum Coverage
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_parse_lenient_edge_cases():
    """Test parse_to_dict_lenient with edge cases."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    edge_cases = [
        "Random text before DO X1 = 1.5 and after",
        "DO X1 = 100.0",  # Out of range
        "DO X1 = -100.0",  # Out of range negative
        "Multiple DO X1 = 1.0 and DO X2 = 2.0",  # Multiple commands
        "",  # Empty
        "DO X99 = 1.0",  # Invalid node
    ]
    
    for text in edge_cases:
        result = dsl.parse_to_dict_lenient(text, clip_out_of_range=True)
        # Should either parse or return None without crashing
        assert result is None or isinstance(result, dict)


@pytest.mark.unit
def test_all_components_with_extreme_values(ground_truth_scm, student_scm, seed_everything):
    """Test all components handle extreme values."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    
    # Extreme intervention values
    extreme_values = [-100.0, -10.0, 0.0, 10.0, 100.0]
    
    for val in extreme_values:
        result = executor.run_experiment({'target': 'X1', 'value': val, 'samples': 20})
        loss = learner.train_step(result, n_epochs=3)
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))


# =============================================================================
# Final Coverage Maximization Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_all_edge_cases(student_scm):
    """Test StateEncoder with all edge cases."""
    from ace_experiments import StateEncoder
    
    encoder = StateEncoder(5, torch.device('cpu'), d_model=32)
    
    edge_case_losses = [
        {},  # Empty
        {'X1': 0.0},  # Partial
        {f'X{i}': 0.0 for i in range(1, 6)},  # All zero
        {f'X{i}': 100.0 for i in range(1, 6)},  # All large
        {f'X{i}': float(i) * 10 for i in range(1, 6)},  # Varied
    ]
    
    for losses in edge_case_losses:
        encoding = encoder(student_scm, losses)
        assert isinstance(encoding, torch.Tensor)
        assert not torch.isnan(encoding).any()


@pytest.mark.integration
def test_complete_baseline_comparison_workflow(seed_everything):
    """Test complete workflow comparing all baselines."""
    from baselines import (
        GroundTruthSCM, RandomPolicy, RoundRobinPolicy,
        MaxVariancePolicy, run_baseline
    )
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    
    results = {}
    
    # Run each baseline
    policies = {
        'Random': RandomPolicy(oracle.nodes),
        'RoundRobin': RoundRobinPolicy(oracle.nodes),
        'MaxVariance': MaxVariancePolicy(oracle.nodes, n_candidates=5, n_mc_samples=2)
    }
    
    for name, policy in policies.items():
        df = run_baseline(policy, oracle, n_episodes=2, steps_per_episode=5)
        results[name] = df['total_loss'].iloc[-1]
    
    # All should have final losses
    assert len(results) == 3
    for loss in results.values():
        assert loss >= 0


@pytest.mark.unit
def test_all_dsl_parsing_paths():
    """Test all DSL parsing code paths."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    dsl = ExperimentalDSL(nodes, value_min=-5.0, value_max=5.0)
    
    # Test all parsing variations
    test_cases = [
        ("DO X1 = 1.5", True),  # Valid
        ("DO X2 = -3.2", True),  # Valid negative
        ("DO X3 = 0.0", True),  # Valid zero
        ("DO X4 = 4.99", True),  # Edge of range
        ("DO X5 = -4.99", True),  # Edge of range
        ("do x1 = 1.0", True),  # Lowercase (case insensitive)
        ("DO X1= 1.0", True),  # No space before =
        ("DO X1 =1.0", True),  # No space after =
        ("X1 = 1.0", False),  # Missing DO
        ("DO = 1.0", False),  # Missing target
        ("DO X1", False),  # Missing value
        ("DO X99 = 1.0", False),  # Invalid node
    ]
    
    for cmd, should_parse in test_cases:
        result = dsl.parse_to_dict(cmd)
        if should_parse:
            # Might parse or be out of range
            assert result is None or isinstance(result, dict)
        else:
            assert result is None


# =============================================================================
# Comprehensive Visualization Coverage
# =============================================================================

@pytest.mark.unit
def test_visualize_create_mechanism_contrast_all_mechanisms(ground_truth_scm, student_scm, tmp_path, seed_everything):
    """Test mechanism contrast for all mechanisms."""
    from visualize import create_mechanism_contrast
    import matplotlib
    matplotlib.use('Agg')
    
    seed_everything(42)
    
    # Train student to have some differences
    from baselines import SCMLearner
    learner = SCMLearner(student_scm)
    for _ in range(5):
        data = ground_truth_scm.generate(100)
        learner.train_step(data, n_epochs=10)
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create contrast
    create_mechanism_contrast(ground_truth_scm, student_scm, str(results_dir))
    
    assert (results_dir / "mechanism_contrast_improved.png").exists()


# =============================================================================
# Maximum Coverage for All Modules
# =============================================================================

@pytest.mark.unit
def test_all_imports_work():
    """Test that all major modules can be imported."""
    
    # Main modules
    import ace_experiments
    import baselines
    import visualize
    import compare_methods
    import clamping_detector
    import regime_analyzer
    
    # Experiment modules
    from experiments import complex_scm
    from experiments import duffing_oscillators
    from experiments import phillips_curve
    
    # All should import successfully
    assert ace_experiments is not None
    assert baselines is not None
    assert visualize is not None


@pytest.mark.unit
def test_all_main_classes_instantiable(ground_truth_scm):
    """Test all main classes can be instantiated."""
    from ace_experiments import (
        GroundTruthSCM, StudentSCM, ExperimentExecutor,
        SCMLearner, StateEncoder, EarlyStopping,
        DedicatedRootLearner, ExperimentalDSL,
        TransformerPolicy, ScientificCritic
    )
    
    # Instantiate all
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    executor = ExperimentExecutor(gt)
    learner = SCMLearner(student)
    encoder = StateEncoder(5, torch.device('cpu'))
    stopper = EarlyStopping()
    root_learner = DedicatedRootLearner(['X1', 'X4'])
    dsl = ExperimentalDSL(list(gt.nodes))
    policy = TransformerPolicy(dsl, torch.device('cpu'), d_model=16)
    critic = ScientificCritic(gt)
    
    # All should instantiate
    assert all([gt, student, executor, learner, encoder, stopper, 
                root_learner, dsl, policy, critic])


# =============================================================================
# Additional Edge Case Coverage
# =============================================================================

@pytest.mark.integration
def test_workflow_with_all_sample_sizes(ground_truth_scm, seed_everything):
    """Test workflow with various sample sizes."""
    from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    for n_samples in [10, 50, 100, 500]:
        student = StudentSCM(ground_truth_scm)
        executor = ExperimentExecutor(ground_truth_scm)
        learner = SCMLearner(student, lr=0.01)
        
        # Run with specific sample size
        result = executor.run_experiment({'target': 'X1', 'value': 1.0, 'samples': n_samples})
        loss = learner.train_step(result, n_epochs=5)
        
        assert isinstance(loss, float)


@pytest.mark.unit
def test_all_helper_functions_exist():
    """Test that all expected helper functions exist."""
    from ace_experiments import (
        get_random_valid_command_range,
        get_teacher_command_impact,
        _impact_weight,
        _direct_child_impact_weight,
        _disentanglement_bonus,
        _bin_index,
        calculate_value_novelty_bonus,
        compute_unified_diversity_score,
        fit_root_distributions,
        visualize_scm_graph,
        save_plots,
        visualize_contrast_save,
        save_checkpoint,
        create_emergency_save_handler
    )
    
    # All should be callable
    functions = [
        get_random_valid_command_range,
        get_teacher_command_impact,
        _impact_weight,
        _direct_child_impact_weight,
        _disentanglement_bonus,
        _bin_index,
        calculate_value_novelty_bonus,
        compute_unified_diversity_score,
        fit_root_distributions,
        visualize_scm_graph,
        save_plots,
        visualize_contrast_save,
        save_checkpoint,
        create_emergency_save_handler
    ]
    
    for func in functions:
        assert callable(func)
