"""Unit tests for scripts/analysis/aggregate_budget_fairness.py.

Covers the actual decision-gate arithmetic (query-budget-matched per-node
best MSE, ACE vs. best baseline) against synthetic ACE (node_losses.csv) and
baseline (summary.csv) result trees, since this script's output directly
determines the paper's headline framing.
"""
import csv
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "analysis", "aggregate_budget_fairness.py")


def _write_node_losses(seed_dir, per_node_best, per_node_final, n_nodes=5, n_episodes=3):
    os.makedirs(seed_dir, exist_ok=True)
    path = os.path.join(seed_dir, "node_losses.csv")
    fieldnames = ["episode", "total_loss"] + [f"loss_n{i}" for i in range(n_nodes)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Best (lowest) total_loss at episode 0, final (higher) at last episode,
        # matching summarize_node_losses' best=min, final=last-episode semantics.
        for ep in range(n_episodes):
            total = per_node_best * n_nodes if ep == 0 else per_node_final * n_nodes
            row = {"episode": ep, "total_loss": total}
            row.update({f"loss_n{i}": total / n_nodes for i in range(n_nodes)})
            writer.writerow(row)


def _write_query_budget(seed_dir, total_samples):
    with open(os.path.join(seed_dir, "query_budget.json"), "w") as fh:
        json.dump({"total": {"samples": total_samples}}, fh)


def _write_baseline_summary(seed_dir, min_total_loss, final_total_loss, n_nodes=5):
    os.makedirs(seed_dir, exist_ok=True)
    with open(os.path.join(seed_dir, "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["min_total_loss", "final_total_loss", "n_nodes"])
        writer.writeheader()
        writer.writerow({"min_total_loss": min_total_loss,
                          "final_total_loss": final_total_loss, "n_nodes": n_nodes})


@pytest.fixture
def budget_fairness_root(tmp_path):
    root = tmp_path / "curc_30node_budget_fairness"
    n_nodes = 5

    # ACE (env-lookahead): per-node best 0.20, matches a ~500k query budget.
    for seed in (42, 123):
        seed_dir = root / "ace_env" / f"seed_{seed}"
        _write_node_losses(str(seed_dir), per_node_best=0.20, per_node_final=0.25, n_nodes=n_nodes)
        _write_query_budget(str(seed_dir), total_samples=500_000)

    # ACE (student-lookahead): slightly worse per-node best, honest 1x budget.
    for seed in (42, 123):
        seed_dir = root / "ace_student" / f"seed_{seed}"
        _write_node_losses(str(seed_dir), per_node_best=0.30, per_node_final=0.35, n_nodes=n_nodes)
        _write_query_budget(str(seed_dir), total_samples=130_000)

    # Best baseline (random) at the matched ~500k budget, worse than ACE env.
    for seed in (42, 123):
        seed_dir = root / "baselines" / "random" / f"seed_{seed}"
        _write_baseline_summary(str(seed_dir), min_total_loss=0.40 * n_nodes,
                                 final_total_loss=0.45 * n_nodes, n_nodes=n_nodes)
        _write_query_budget(str(seed_dir), total_samples=500_000)

    # Weaker baseline (round_robin) also worse than ACE, to confirm "best
    # baseline" selection picks the lowest-loss one (random), not this one.
    for seed in (42, 123):
        seed_dir = root / "baselines" / "round_robin" / f"seed_{seed}"
        _write_baseline_summary(str(seed_dir), min_total_loss=0.60 * n_nodes,
                                 final_total_loss=0.65 * n_nodes, n_nodes=n_nodes)
        _write_query_budget(str(seed_dir), total_samples=500_000)

    return str(root)


@pytest.mark.unit
def test_ace_env_wins_decision_gate(budget_fairness_root, tmp_path):
    out_csv = str(tmp_path / "aggregate.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", budget_fairness_root, "--out", out_csv],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    # 2 ACE modes x 2 seeds + 2 baselines x 2 seeds = 8 rows.
    assert len(rows) == 8
    methods = {r["method"] for r in rows}
    assert methods == {"ace_env", "ace_student", "random", "round_robin"}

    stdout = result.stdout
    assert "ace_env: WINS vs best baseline (random" in stdout
    # round_robin must not be selected as "best baseline" -- random is lower-loss.
    assert "round_robin, " not in stdout.split("Decision gate")[-1]


@pytest.mark.unit
def test_ace_loses_when_baseline_is_better(tmp_path):
    root = tmp_path / "curc_5node_budget_fairness"
    n_nodes = 5
    seed_dir = root / "ace_env" / "seed_42"
    _write_node_losses(str(seed_dir), per_node_best=0.50, per_node_final=0.55, n_nodes=n_nodes)
    _write_query_budget(str(seed_dir), total_samples=100_000)

    baseline_dir = root / "baselines" / "random" / "seed_42"
    _write_baseline_summary(str(baseline_dir), min_total_loss=0.10 * n_nodes,
                             final_total_loss=0.15 * n_nodes, n_nodes=n_nodes)
    _write_query_budget(str(baseline_dir), total_samples=100_000)

    out_csv = str(tmp_path / "aggregate.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", str(root), "--out", out_csv],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "ace_env: LOSES to best baseline (random" in result.stdout


@pytest.mark.unit
def test_missing_root_exits_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", str(tmp_path / "does_not_exist")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode != 0


@pytest.mark.unit
def test_no_baselines_yet_reports_gate_unresolved(tmp_path):
    root = tmp_path / "curc_30node_budget_fairness"
    seed_dir = root / "ace_env" / "seed_42"
    _write_node_losses(str(seed_dir), per_node_best=0.20, per_node_final=0.25, n_nodes=5)
    _write_query_budget(str(seed_dir), total_samples=500_000)

    out_csv = str(tmp_path / "aggregate.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", str(root), "--out", out_csv],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "cannot resolve the decision gate" in result.stdout
