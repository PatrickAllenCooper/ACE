"""Tests for aligned_history_dataframe.

Context: every 5-node budget-fairness resume on 2026-08-11 aborted at
episode 10 with ``ValueError: All arrays must be of the same length``.
Checkpoint restore only reloads loss/reward; cov_bonus/episode/step
histories start empty. The incremental metrics.csv save then built a
DataFrame from those unequal lists and killed the run. These tests lock
the helper that makes that save best-effort instead of fatal.
"""
import pandas as pd
import pytest


@pytest.mark.unit
def test_equal_lengths_pass_through():
    from ace_experiments import aligned_history_dataframe

    df = aligned_history_dataframe({
        "dpo_loss": [1.0, 0.5, 0.2],
        "reward": [0.1, 0.2, 0.3],
        "episode": [0, 0, 1],
    })
    assert len(df) == 3
    assert list(df["dpo_loss"]) == [1.0, 0.5, 0.2]


@pytest.mark.unit
def test_unequal_lengths_truncate_to_shortest_instead_of_raising():
    from ace_experiments import aligned_history_dataframe

    # Resume shape: loss/reward restored (N=5), episode/cov_bonus only
    # recorded since resume (N=2).
    df = aligned_history_dataframe({
        "dpo_loss": [1.0, 0.9, 0.8, 0.7, 0.6],
        "reward": [0.1, 0.1, 0.2, 0.2, 0.3],
        "cov_bonus": [0.0, 1.0],
        "episode": [8, 9],
        "step": [0, 1],
    })
    assert len(df) == 2
    assert list(df["dpo_loss"]) == [1.0, 0.9]
    assert list(df["episode"]) == [8, 9]


@pytest.mark.unit
def test_empty_columns_return_empty_frame():
    from ace_experiments import aligned_history_dataframe

    df = aligned_history_dataframe({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.unit
def test_none_column_treated_as_empty():
    from ace_experiments import aligned_history_dataframe

    df = aligned_history_dataframe({
        "dpo_loss": [1.0, 2.0],
        "cov_bonus": None,
    })
    assert len(df) == 0
