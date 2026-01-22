"""
Integration tests for baselines.py workflow.

Tests cover:
- Full baseline execution
- run_baseline function with different policies
- Multi-episode training
- Results DataFrame structure
"""

import pytest
import pandas as pd


# =============================================================================
# run_baseline Integration Tests
# =============================================================================

@pytest.mark.integration
def test_run_baseline_with_random_policy_quick(seed_everything):
    """Test run_baseline with RandomPolicy (quick version)."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # Run short baseline
    df = run_baseline(
        policy=policy,
        oracle=oracle,
        n_episodes=2,
        steps_per_episode=5
    )
    
    # Should return DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10  # 2 episodes * 5 steps


@pytest.mark.integration
def test_run_baseline_dataframe_structure(seed_everything):
    """Test that run_baseline returns properly structured DataFrame."""
    from baselines import GroundTruthSCM, RoundRobinPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RoundRobinPolicy(oracle.nodes)
    
    df = run_baseline(policy, oracle, n_episodes=2, steps_per_episode=5)
    
    # Check required columns
    assert 'episode' in df.columns
    assert 'step' in df.columns
    assert 'target' in df.columns
    assert 'value' in df.columns
    assert 'total_loss' in df.columns
    
    # Check per-node loss columns
    for node in oracle.nodes:
        assert f'loss_{node}' in df.columns


@pytest.mark.integration
def test_run_baseline_loss_tracking(seed_everything):
    """Test that run_baseline tracks losses correctly."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    df = run_baseline(policy, oracle, n_episodes=3, steps_per_episode=5)
    
    # Check loss values are valid
    assert df['total_loss'].min() >= 0
    assert not df['total_loss'].isna().any()
    
    # Per-node losses should be >= 0
    for node in oracle.nodes:
        col = f'loss_{node}'
        assert df[col].min() >= 0


@pytest.mark.integration
def test_run_baseline_with_observational_training(seed_everything):
    """Test run_baseline with observational training enabled."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # With observational training every 2 steps
    df = run_baseline(
        policy, oracle,
        n_episodes=2,
        steps_per_episode=6,
        obs_train_interval=2,
        obs_train_samples=50
    )
    
    # Should complete without error
    assert len(df) == 12


@pytest.mark.integration
def test_run_baseline_multiple_episodes_fresh_student(seed_everything):
    """Test that run_baseline uses fresh student each episode."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    # Run 3 episodes
    df = run_baseline(policy, oracle, n_episodes=3, steps_per_episode=5)
    
    # Each episode should start fresh
    # Check that we have 3 distinct episodes
    assert df['episode'].nunique() == 3
    
    # Episode numbers should be 0, 1, 2
    assert set(df['episode'].unique()) == {0, 1, 2}


# =============================================================================
# Policy Reset Tests
# =============================================================================

@pytest.mark.integration
def test_round_robin_resets_between_episodes(seed_everything):
    """Test that RoundRobinPolicy resets between episodes."""
    from baselines import GroundTruthSCM, RoundRobinPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RoundRobinPolicy(oracle.nodes)
    
    # Run baseline
    df = run_baseline(policy, oracle, n_episodes=2, steps_per_episode=5)
    
    # First step of each episode should be same target (after reset)
    ep0_first = df[df['episode'] == 0].iloc[0]['target']
    ep1_first = df[df['episode'] == 1].iloc[0]['target']
    
    # Should be same (policy reset)
    assert ep0_first == ep1_first


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.integration
def test_run_baseline_with_single_episode(seed_everything):
    """Test run_baseline with single episode."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    df = run_baseline(policy, oracle, n_episodes=1, steps_per_episode=10)
    
    assert len(df) == 10
    assert df['episode'].unique() == [0]


@pytest.mark.integration
def test_run_baseline_with_many_steps(seed_everything):
    """Test run_baseline with many steps per episode."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    policy = RandomPolicy(oracle.nodes)
    
    df = run_baseline(policy, oracle, n_episodes=1, steps_per_episode=50)
    
    assert len(df) == 50
