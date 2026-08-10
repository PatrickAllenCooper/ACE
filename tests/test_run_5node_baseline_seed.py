"""
Unit tests for scripts/runners/run_5node_baseline_seed.py.

Context: the 5-node budget-fairness Phase 2 submit script originally called
``baselines.py --all_with_ppo`` directly, which writes ``{method}_results.csv``
/ ``{method}_query_budget.json`` into one shared, timestamped directory per
seed -- a layout scripts/analysis/aggregate_budget_fairness.py cannot parse
(it expects ``{method}/seed_{seed}/{node_losses.csv,summary.csv,
query_budget.json}``, matching the 30-node pipeline's
run_30node_baseline_seed.py). This runner is the 5-node analog of that
30-node script; these tests guard the output contract the aggregator relies
on, since a regression here would only surface after burning the compute for
both phases of a budget-fairness rerun.
"""
import json
import subprocess
import sys

import pandas as pd
import pytest


CLI = ["scripts/runners/run_5node_baseline_seed.py"]
COMMON_ARGS = [
    "--episodes", "2",
    "--steps", "3",
    "--obs_train_interval", "1",
    "--obs_train_samples", "20",
]


def _run(tmp_path, method, seed=42, extra=None):
    out = tmp_path / "baselines"
    cmd = [sys.executable, *CLI, "--method", method, "--seed", str(seed),
           *COMMON_ARGS, "--output", str(out), *(extra or [])]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return out / method / f"seed_{seed}"


@pytest.mark.integration
@pytest.mark.parametrize("method", ["random", "round_robin", "max_variance"])
def test_writes_expected_files(tmp_path, method):
    run_dir = _run(tmp_path, method)

    for name in ("node_losses.csv", "summary.csv", "query_budget.json", "per_episode.csv"):
        assert (run_dir / name).exists(), f"missing {name} for method={method}"


@pytest.mark.integration
def test_node_losses_schema_matches_aggregator_expectations(tmp_path):
    """scripts/analysis/scaling_common.summarize_node_losses requires an
    `episode` column, a `total_loss` column, and at least one `loss_*`
    per-node column."""
    run_dir = _run(tmp_path, "random")
    df = pd.read_csv(run_dir / "node_losses.csv")

    assert "episode" in df.columns
    assert "total_loss" in df.columns
    loss_cols = [c for c in df.columns if c.startswith("loss_")]
    assert len(loss_cols) == 5  # 5-node GroundTruthSCM


@pytest.mark.integration
def test_summary_csv_has_n_nodes_and_loss_fields(tmp_path):
    run_dir = _run(tmp_path, "random")
    df = pd.read_csv(run_dir / "summary.csv")
    row = df.iloc[0]

    assert row["n_nodes"] == 5
    assert row["min_total_loss"] <= row["final_total_loss"] + 1e-9


@pytest.mark.integration
def test_query_budget_json_has_total_samples(tmp_path):
    run_dir = _run(tmp_path, "random")
    with open(run_dir / "query_budget.json") as fh:
        summary = json.load(fh)

    assert "total" in summary
    assert summary["total"]["samples"] > 0


@pytest.mark.integration
def test_query_budget_flag_stops_early(tmp_path):
    """--query_budget should cap total environment samples rather than the
    fixed --episodes count once the cumulative sample total reaches it."""
    run_dir = _run(tmp_path, "random", extra=["--query_budget", "50"])
    with open(run_dir / "query_budget.json") as fh:
        summary = json.load(fh)

    # run_baseline's own safety cap is generous (50x n_episodes); the point
    # here is only that a tiny budget does not silently run to completion
    # with an unrelated (much larger) total.
    assert summary["total"]["samples"] < 5000


@pytest.mark.integration
def test_output_is_readable_by_the_budget_fairness_aggregator(tmp_path):
    """End-to-end contract check against the actual downstream consumer."""
    sys.path.insert(0, "scripts/analysis")
    from scaling_common import summarize_seed_dir  # noqa: E402

    run_dir = _run(tmp_path, "round_robin")
    summary = summarize_seed_dir(str(run_dir))

    assert summary is not None
    assert summary["n_nodes"] == 5
    assert summary["per_node_best"] > 0


@pytest.mark.unit
def test_rejects_bayesian_oed_method():
    """Bayesian OED is intentionally not one of this script's choices (its
    5-node baseline lives in run_reviewer_experiments.py, with its own
    query-budget wiring); argparse should reject it rather than silently
    doing the wrong thing."""
    result = subprocess.run(
        [sys.executable, *CLI, "--method", "bayesian_oed", "--seed", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()
