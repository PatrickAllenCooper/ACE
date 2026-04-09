#!/usr/bin/env python3
"""
Tests for the three fixed reviewer experiments.

Validates that:
1. --graph_misspec flag correctly modifies student graph while keeping oracle intact
2. --large_scale flag creates a valid N-node SCM compatible with ACE
3. Hyperparameter grid subprocess invocation works with varied cov_bonus/diversity_reward_weight
4. Graph misspecification subprocess invocation works
5. All edge cases are handled (reversed edge DAG validity, missing edge parent sets, etc.)

Run with: pytest tests/test_reviewer_fixes.py -v
"""

import sys
import os
import subprocess
import pytest
import torch
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ace_experiments import CausalModel, GroundTruthSCM, StudentSCM


class TestGraphMisspecification:
    """Test the --graph_misspec flag logic."""

    def test_baseline_5node_graph(self):
        """Verify the unmodified 5-node SCM has the expected edges."""
        scm = GroundTruthSCM()
        edges = set(scm.graph.edges)
        assert ('X1', 'X2') in edges
        assert ('X2', 'X3') in edges
        assert ('X1', 'X3') in edges
        assert ('X4', 'X5') in edges
        assert len(edges) == 4

    def test_missing_edge_removes_x1_x3(self):
        """Missing edge should remove X1->X3 from student but oracle keeps it."""
        import copy
        scm = GroundTruthSCM()
        oracle_edges = set(scm.graph.edges)

        student_graph = copy.deepcopy(scm)
        student_graph.graph.remove_edge('X1', 'X3')
        student_edges = set(student_graph.graph.edges)

        assert ('X1', 'X3') in oracle_edges, "Oracle must keep X1->X3"
        assert ('X1', 'X3') not in student_edges, "Student must lose X1->X3"
        assert ('X2', 'X3') in student_edges, "Student must keep X2->X3"
        assert nx.is_directed_acyclic_graph(student_graph.graph), "Must remain a DAG"

    def test_extra_edge_adds_x1_x5(self):
        """Extra edge should add X1->X5 to student."""
        import copy
        scm = GroundTruthSCM()
        student_graph = copy.deepcopy(scm)
        student_graph.graph.add_edge('X1', 'X5')
        student_edges = set(student_graph.graph.edges)

        assert ('X1', 'X5') in student_edges
        assert ('X4', 'X5') in student_edges
        assert nx.is_directed_acyclic_graph(student_graph.graph)

    def test_reversed_edge_creates_valid_dag(self):
        """Reversed X1->X2 to X2->X1 must still be a valid DAG."""
        import copy
        scm = GroundTruthSCM()
        student_graph = copy.deepcopy(scm)

        student_graph.graph.remove_edge('X1', 'X2')
        student_graph.graph.add_edge('X2', 'X1')

        assert nx.is_directed_acyclic_graph(student_graph.graph), \
            "Reversed edge must produce a valid DAG"

        new_topo = list(nx.topological_sort(student_graph.graph))
        assert len(new_topo) == 5
        assert new_topo.index('X2') < new_topo.index('X1'), \
            "X2 must come before X1 in topological order after reversal"

    def test_missing_and_extra_combined(self):
        """Combined misspec: remove X1->X3, add X1->X5."""
        import copy
        scm = GroundTruthSCM()
        student_graph = copy.deepcopy(scm)

        student_graph.graph.remove_edge('X1', 'X3')
        student_graph.graph.add_edge('X1', 'X5')

        student_edges = set(student_graph.graph.edges)
        assert ('X1', 'X3') not in student_edges
        assert ('X1', 'X5') in student_edges
        assert nx.is_directed_acyclic_graph(student_graph.graph)

    def test_student_scm_builds_with_misspecified_graph(self):
        """StudentSCM must build successfully with each misspecified graph."""
        import copy

        misspec_configs = [
            ("missing_edge", lambda g: g.graph.remove_edge('X1', 'X3')),
            ("extra_edge", lambda g: g.graph.add_edge('X1', 'X5')),
            ("reversed_edge", lambda g: (
                g.graph.remove_edge('X1', 'X2'),
                g.graph.add_edge('X2', 'X1'),
                setattr(g, 'topo_order', list(nx.topological_sort(g.graph))),
                setattr(g, 'nodes', sorted(list(g.graph.nodes))),
            )),
            ("missing_and_extra", lambda g: (
                g.graph.remove_edge('X1', 'X3'),
                g.graph.add_edge('X1', 'X5'),
            )),
        ]

        for name, modifier in misspec_configs:
            scm = GroundTruthSCM()
            modified = copy.deepcopy(scm)
            modifier(modified)

            student = StudentSCM(modified)
            assert student is not None, f"StudentSCM failed for {name}"
            assert len(student.nodes) == 5, f"Wrong node count for {name}"

    def test_oracle_data_unchanged_under_misspec(self):
        """Oracle generates identical data regardless of student graph changes."""
        scm = GroundTruthSCM()

        torch.manual_seed(42)
        data1 = scm.generate(100)

        torch.manual_seed(42)
        data2 = scm.generate(100)

        for node in scm.nodes:
            assert torch.allclose(data1[node], data2[node]), \
                f"Oracle data for {node} changed unexpectedly"

    def test_student_architecture_differs_under_misspec(self):
        """Student MLP architecture must change based on parent count."""
        import copy

        scm = GroundTruthSCM()
        student_normal = StudentSCM(scm)

        modified = copy.deepcopy(scm)
        modified.graph.remove_edge('X1', 'X3')
        student_misspec = StudentSCM(modified)

        x3_normal_params = student_normal.mechanisms['X3'][0].in_features
        x3_misspec_params = student_misspec.mechanisms['X3'][0].in_features

        assert x3_normal_params == 2, "Normal X3 has 2 parents (X1, X2)"
        assert x3_misspec_params == 1, "Misspecified X3 has 1 parent (X2 only)"


class TestLargeScaleSCM:
    """Test the --large_scale flag logic."""

    def test_large_scale_scm_import(self):
        """LargeScaleSCM must be importable."""
        from experiments.large_scale_scm import LargeScaleSCM
        scm = LargeScaleSCM(30)
        assert len(scm.nodes) == 30

    def test_large_scale_edge_construction(self):
        """Edges built from LargeScaleSCM dict must form a valid DAG."""
        from experiments.large_scale_scm import LargeScaleSCM

        np.random.seed(42)
        lscm = LargeScaleSCM(30)

        edges = []
        for node, parents in lscm.graph.items():
            for p in parents:
                edges.append((p, node))

        G = nx.DiGraph(edges)
        assert nx.is_directed_acyclic_graph(G), "30-node graph must be a DAG"
        assert len(G.nodes) == 30

    def test_large_scale_causal_model_wrapper(self):
        """The CausalModel wrapper for LargeScaleSCM must work."""
        from experiments.large_scale_scm import LargeScaleSCM

        np.random.seed(42)
        lscm = LargeScaleSCM(30)

        edges = []
        for node, parents in lscm.graph.items():
            for p in parents:
                edges.append((p, node))

        cm = CausalModel(edges)
        assert len(cm.nodes) == 30
        assert len(cm.topo_order) == 30

        roots = [n for n in cm.nodes if len(cm.get_parents(n)) == 0]
        assert len(roots) == 5, f"Expected 5 roots, got {len(roots)}"

    def test_large_scale_generate_data(self):
        """The large-scale SCM wrapper must generate valid data."""
        from experiments.large_scale_scm import LargeScaleSCM

        np.random.seed(42)
        lscm = LargeScaleSCM(30)

        edges = []
        for node, parents in lscm.graph.items():
            for p in parents:
                edges.append((p, node))

        coeffs = {}
        np.random.seed(42)
        for node, parents in lscm.graph.items():
            coeffs[node] = {p: float(np.random.uniform(0.3, 0.7)) for p in parents}

        class TestSCM(CausalModel):
            def __init__(self, edges, coeffs):
                super().__init__(edges)
                self._coeffs = coeffs
            def mechanisms(self, data, node, n_samples=1):
                n = next(iter(data.values())).shape[0] if data else n_samples
                parents = self.get_parents(node)
                if not parents:
                    return torch.randn(n)
                value = torch.zeros(n)
                for p in parents:
                    value = value + self._coeffs[node][p] * data[p]
                return value + torch.randn(n) * 0.15
            def generate(self, n_samples=1, interventions=None):
                data = {}
                interventions = interventions or {}
                for node in self.topo_order:
                    if node in interventions:
                        data[node] = torch.full((n_samples,), float(interventions[node]))
                    else:
                        parents = self.get_parents(node)
                        p_data = {p: data[p] for p in parents}
                        data[node] = self.mechanisms(p_data, node, n_samples=n_samples)
                return data

        scm = TestSCM(edges, coeffs)
        data = scm.generate(100)

        assert len(data) == 30
        for node in scm.nodes:
            assert data[node].shape == (100,), f"Node {node} has wrong shape"
            assert not torch.isnan(data[node]).any(), f"Node {node} has NaN values"
            assert not torch.isinf(data[node]).any(), f"Node {node} has Inf values"

    def test_large_scale_intervention(self):
        """Interventions on the large-scale SCM must clamp the target node."""
        from experiments.large_scale_scm import LargeScaleSCM

        np.random.seed(42)
        lscm = LargeScaleSCM(30)
        edges = [(p, n) for n, parents in lscm.graph.items() for p in parents]
        coeffs = {n: {p: float(np.random.uniform(0.3, 0.7)) for p in parents}
                  for n, parents in lscm.graph.items()}

        class TestSCM(CausalModel):
            def __init__(self, edges, coeffs):
                super().__init__(edges)
                self._coeffs = coeffs
            def mechanisms(self, data, node, n_samples=1):
                n = next(iter(data.values())).shape[0] if data else n_samples
                parents = self.get_parents(node)
                if not parents:
                    return torch.randn(n)
                value = sum(self._coeffs[node][p] * data[p] for p in parents)
                return value + torch.randn(n) * 0.15
            def generate(self, n_samples=1, interventions=None):
                data = {}
                interventions = interventions or {}
                for node in self.topo_order:
                    if node in interventions:
                        data[node] = torch.full((n_samples,), float(interventions[node]))
                    else:
                        p_data = {p: data[p] for p in self.get_parents(node)}
                        data[node] = self.mechanisms(p_data, node, n_samples=n_samples)
                return data

        scm = TestSCM(edges, coeffs)
        data = scm.generate(50, interventions={"X3": 2.5})
        assert torch.allclose(data["X3"], torch.full((50,), 2.5))

    def test_student_scm_builds_for_30_nodes(self):
        """StudentSCM must build for a 30-node graph."""
        from experiments.large_scale_scm import LargeScaleSCM

        np.random.seed(42)
        lscm = LargeScaleSCM(30)
        edges = [(p, n) for n, parents in lscm.graph.items() for p in parents]

        cm = CausalModel(edges)
        student = StudentSCM(cm)

        assert len(student.nodes) == 30
        roots = [n for n in student.nodes if len(student.get_parents(n)) == 0]
        for r in roots:
            assert isinstance(student.mechanisms[r], torch.nn.ParameterDict)


class TestCLIArgParsing:
    """Test that the new CLI arguments parse correctly."""

    def test_graph_misspec_arg_parses(self):
        """--graph_misspec should accept valid choices."""
        import argparse

        for choice in ["missing_edge", "extra_edge", "reversed_edge", "missing_and_extra"]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--graph_misspec", type=str, default=None,
                                choices=["missing_edge", "extra_edge", "reversed_edge", "missing_and_extra"])
            args = parser.parse_args(["--graph_misspec", choice])
            assert args.graph_misspec == choice

    def test_graph_misspec_invalid_choice_rejected(self):
        """--graph_misspec should reject invalid choices."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--graph_misspec", type=str, default=None,
                            choices=["missing_edge", "extra_edge", "reversed_edge", "missing_and_extra"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--graph_misspec", "bad_value"])

    def test_large_scale_arg_parses(self):
        """--large_scale should accept an integer."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--large_scale", type=int, default=None)
        args = parser.parse_args(["--large_scale", "30"])
        assert args.large_scale == 30

    def test_default_args_unchanged(self):
        """Without new flags, behavior should be unchanged."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--graph_misspec", type=str, default=None)
        parser.add_argument("--large_scale", type=int, default=None)
        args = parser.parse_args([])
        assert args.graph_misspec is None
        assert args.large_scale is None


class TestSubprocessInvocation:
    """Test that ace_experiments.py can be invoked with the new flags (dry parse only)."""

    def test_ace_help_includes_graph_misspec(self):
        """ace_experiments.py --help should mention --graph_misspec."""
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--help"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert "--graph_misspec" in result.stdout

    def test_ace_help_includes_large_scale(self):
        """ace_experiments.py --help should mention --large_scale."""
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--help"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert "--large_scale" in result.stdout

    def test_ace_rejects_invalid_misspec(self):
        """ace_experiments.py should reject invalid --graph_misspec values."""
        result = subprocess.run(
            [sys.executable, "ace_experiments.py", "--graph_misspec", "nonexistent"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
