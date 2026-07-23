"""
Unit tests for environment-query budget accounting.

Context (NeurIPS 2026 review, reviewers wZrW / F4Cb / d6tT / TnpG): ACE's
lookahead queries the ground-truth environment once per candidate (K per
step) but only the executed (winning) candidate is charged against the
reported intervention budget. These tests cover the instrumentation added
to make every environment query explicit and taggable:

- ExperimentExecutor.query_log / query_summary (ace_experiments.py)
- generate_student_lookahead_batch, the zero-oracle-query lookahead mode
  used by --lookahead_on_student
- InstrumentedOracle (baselines.py), the equivalent counter for baselines,
  including Max-Variance's own hidden candidate-probe queries
- the --query_budget stopping condition on the ACE episode loop
"""

import pytest
import torch


# =============================================================================
# ExperimentExecutor query counting (ace_experiments.py)
# =============================================================================

@pytest.mark.unit
def test_executor_starts_with_empty_query_log(ground_truth_scm):
    from ace_experiments import ExperimentExecutor

    executor = ExperimentExecutor(ground_truth_scm)
    assert executor.query_log == {}
    assert executor.total_calls() == 0
    assert executor.total_samples() == 0


@pytest.mark.unit
def test_run_experiment_default_tag_is_executed(ground_truth_scm, seed_everything):
    from ace_experiments import ExperimentExecutor

    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    executor.run_experiment({"target": "X1", "value": 1.0, "samples": 30})

    assert executor.total_samples(tags=["executed"]) == 30
    assert executor.total_samples() == 30


@pytest.mark.unit
def test_run_experiment_respects_explicit_tag(ground_truth_scm, seed_everything):
    from ace_experiments import ExperimentExecutor

    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    executor.run_experiment({"target": "X2", "value": 1.0, "samples": 40}, tag="lookahead")

    assert executor.total_samples(tags=["executed"]) == 0
    assert executor.total_samples(tags=["lookahead"]) == 40
    assert executor.total_samples() == 40


@pytest.mark.unit
def test_query_log_accumulates_across_calls_and_tags(ground_truth_scm, seed_everything):
    from ace_experiments import ExperimentExecutor

    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)

    # 4 lookahead candidates, only 1 gets executed -- the exact asymmetry
    # flagged by reviewers.
    for _ in range(4):
        executor.run_experiment({"target": "X1", "value": 0.5, "samples": 100}, tag="lookahead")
    executor.run_experiment({"target": "X1", "value": 0.5, "samples": 100}, tag="executed")

    summary = executor.query_summary()
    assert summary["lookahead"]["calls"] == 4
    assert summary["lookahead"]["samples"] == 400
    assert summary["executed"]["calls"] == 1
    assert summary["executed"]["samples"] == 100
    assert summary["total"]["calls"] == 5
    assert summary["total"]["samples"] == 500

    # The headline "intervention budget" (executed only) understates the
    # true environment-query cost by 5x here -- this is exactly the gap
    # the query_budget/lookahead_on_student modes exist to close.
    assert executor.total_samples(tags=["executed"]) * 5 == executor.total_samples()


@pytest.mark.unit
def test_observational_none_plan_records_default_samples(ground_truth_scm):
    from ace_experiments import ExperimentExecutor

    executor = ExperimentExecutor(ground_truth_scm)
    result = executor.run_experiment(None, tag="observational")

    assert result["intervened"] is None
    assert executor.total_samples(tags=["observational"]) == 100


@pytest.mark.unit
def test_query_summary_is_independent_snapshot(ground_truth_scm):
    from ace_experiments import ExperimentExecutor

    executor = ExperimentExecutor(ground_truth_scm)
    executor.run_experiment({"target": "X1", "value": 1.0, "samples": 10}, tag="executed")
    summary1 = executor.query_summary()
    executor.run_experiment({"target": "X1", "value": 1.0, "samples": 10}, tag="executed")
    summary2 = executor.query_summary()

    # Mutating executor state after taking a summary must not retroactively
    # change the earlier snapshot.
    assert summary1["executed"]["samples"] == 10
    assert summary2["executed"]["samples"] == 20


# =============================================================================
# Student-only lookahead (zero oracle queries)
# =============================================================================

@pytest.mark.unit
def test_student_lookahead_does_not_touch_executor(ground_truth_scm, student_scm, seed_everything):
    from ace_experiments import ExperimentExecutor, generate_student_lookahead_batch

    seed_everything(0)
    executor = ExperimentExecutor(ground_truth_scm)
    before = executor.total_samples()

    generate_student_lookahead_batch(student_scm, {"target": "X2", "value": 1.5, "samples": 40})

    # The whole point of --lookahead_on_student: this must not be routed
    # through the executor at all.
    assert executor.total_samples() == before


@pytest.mark.unit
def test_student_lookahead_batch_structure(student_scm, seed_everything):
    from ace_experiments import generate_student_lookahead_batch

    seed_everything(0)
    batch = generate_student_lookahead_batch(student_scm, {"target": "X2", "value": 1.5, "samples": 40})

    assert set(batch.keys()) == {"data", "intervened"}
    assert batch["intervened"] == "X2"
    assert isinstance(batch["data"], dict)
    assert set(batch["data"].keys()) == set(student_scm.nodes)
    for node, tensor in batch["data"].items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 40


@pytest.mark.unit
def test_student_lookahead_applies_intervention_value(student_scm, seed_everything):
    from ace_experiments import generate_student_lookahead_batch

    seed_everything(0)
    batch = generate_student_lookahead_batch(student_scm, {"target": "X2", "value": 1.5, "samples": 20})

    assert torch.allclose(batch["data"]["X2"], torch.full((20,), 1.5))


@pytest.mark.unit
def test_student_lookahead_observational_plan(student_scm, seed_everything):
    from ace_experiments import generate_student_lookahead_batch

    seed_everything(0)
    batch = generate_student_lookahead_batch(student_scm, {"target": None, "samples": 25})

    assert batch["intervened"] is None
    for tensor in batch["data"].values():
        assert tensor.shape[0] == 25


@pytest.mark.unit
def test_student_lookahead_restores_training_mode(student_scm, seed_everything):
    from ace_experiments import generate_student_lookahead_batch

    seed_everything(0)
    student_scm.train()
    assert student_scm.training is True

    generate_student_lookahead_batch(student_scm, {"target": "X1", "value": 0.0, "samples": 5})

    # Must leave the student in the mode it found it in (train -> train).
    assert student_scm.training is True

    student_scm.eval()
    generate_student_lookahead_batch(student_scm, {"target": "X1", "value": 0.0, "samples": 5})
    assert student_scm.training is False


@pytest.mark.unit
def test_student_lookahead_data_is_detached(student_scm, seed_everything):
    from ace_experiments import generate_student_lookahead_batch

    seed_everything(0)
    batch = generate_student_lookahead_batch(student_scm, {"target": "X2", "value": 1.0, "samples": 10})

    for tensor in batch["data"].values():
        assert tensor.requires_grad is False


# =============================================================================
# --query_budget CLI wiring (ace_experiments.py)
# =============================================================================

@pytest.mark.unit
def test_ace_script_registers_query_budget_and_lookahead_flags():
    """The actual CLI (not a mirror parser) must expose these flags."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "ace_experiments.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert "--query_budget" in result.stdout
    assert "--lookahead_on_student" in result.stdout
    assert "--policy_update" in result.stdout


# =============================================================================
# InstrumentedOracle (baselines.py)
# =============================================================================

@pytest.mark.unit
def test_instrumented_oracle_wraps_and_counts():
    from baselines import GroundTruthSCM, InstrumentedOracle

    base = GroundTruthSCM()
    oracle = InstrumentedOracle(base)

    oracle.generate(n_samples=50, interventions={"X1": 1.0})
    assert oracle.total_samples() == 50
    assert oracle.query_summary()["executed"]["calls"] == 1


@pytest.mark.unit
def test_instrumented_oracle_delegates_attributes():
    from baselines import GroundTruthSCM, InstrumentedOracle

    base = GroundTruthSCM()
    oracle = InstrumentedOracle(base)

    # Non-generate attributes (nodes, graph, get_parents) must pass through.
    assert oracle.nodes == base.nodes
    assert oracle.get_parents("X2") == base.get_parents("X2")


@pytest.mark.unit
def test_instrumented_oracle_tags_are_independent():
    from baselines import GroundTruthSCM, InstrumentedOracle

    base = GroundTruthSCM()
    oracle = InstrumentedOracle(base)

    for _ in range(64):
        oracle.generate(n_samples=50, interventions={"X1": 0.0}, tag="candidate_probe")
    oracle.generate(n_samples=50, interventions={"X1": 0.0}, tag="executed")

    summary = oracle.query_summary()
    assert summary["candidate_probe"]["calls"] == 64
    assert summary["candidate_probe"]["samples"] == 3200
    assert summary["executed"]["calls"] == 1
    # Max-Variance's default n_candidates=64: its hidden query cost is 64x
    # the single executed step it reports, mirroring ACE's lookahead ratio.
    assert summary["total"]["samples"] == 3250


@pytest.mark.unit
def test_max_variance_probe_queries_are_tagged_when_wrapped():
    from baselines import GroundTruthSCM, InstrumentedOracle, MaxVariancePolicy, StudentSCM

    base = GroundTruthSCM()
    oracle = InstrumentedOracle(base)
    student = StudentSCM(base)
    policy = MaxVariancePolicy(base.nodes, n_candidates=8, n_mc_samples=2)

    policy.select_intervention(student, oracle=oracle)

    summary = oracle.query_summary()
    assert "candidate_probe" in summary
    assert summary["candidate_probe"]["calls"] == 8


@pytest.mark.unit
def test_scientific_critic_validation_set_bypasses_counter():
    from baselines import GroundTruthSCM, InstrumentedOracle, ScientificCritic

    base = GroundTruthSCM()
    oracle = InstrumentedOracle(base)
    ScientificCritic(oracle)

    # The one-time held-out validation set must not appear in the budget
    # accounting, mirroring ace_experiments.py's ScientificCritic.
    assert oracle.query_log == {}
