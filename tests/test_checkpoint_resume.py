#!/usr/bin/env python3
"""Tests for checkpoint save and resume behavior."""

import os
import sys
import copy
import shutil
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ace_experiments import CausalModel, GroundTruthSCM, StudentSCM, save_checkpoint


class TestCheckpointSave:
    """Test save_checkpoint writes a loadable file in run_dir."""

    def test_saves_to_run_dir(self, tmp_path):
        """Checkpoint goes to run_dir/checkpoint.pt, not /scratch."""
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        from collections import deque

        save_checkpoint(str(tmp_path), 10, student, opt, [0.5, 0.4], [1.0, 0.9], deque([0, 1]))
        assert (tmp_path / "checkpoint.pt").exists()

    def test_checkpoint_is_loadable(self, tmp_path):
        """Saved checkpoint can be loaded back with torch.load."""
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        from collections import deque

        save_checkpoint(str(tmp_path), 42, student, opt, [0.1], [2.0], deque([1]))
        ckpt = torch.load(str(tmp_path / "checkpoint.pt"), map_location='cpu')

        assert ckpt['episode'] == 42
        assert 'policy_state_dict' in ckpt
        assert 'optimizer_state_dict' in ckpt
        assert ckpt['loss_history'] == [0.1]
        assert ckpt['reward_history'] == [2.0]

    def test_checkpoint_overwrites_previous(self, tmp_path):
        """Saving twice overwrites instead of accumulating files."""
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        from collections import deque

        save_checkpoint(str(tmp_path), 5, student, opt, [], [], deque())
        save_checkpoint(str(tmp_path), 10, student, opt, [], [], deque())

        files = list(tmp_path.glob("checkpoint*.pt"))
        assert len(files) == 1, f"Expected 1 checkpoint file, got {len(files)}"

        ckpt = torch.load(str(files[0]), map_location='cpu')
        assert ckpt['episode'] == 10


class TestRunDirResume:
    """Test that main() detects existing checkpoints and reuses run dirs."""

    def test_no_checkpoint_creates_new_dir(self, tmp_path):
        """When no checkpoint exists, a new timestamped run dir is created."""
        # Simulate empty output dir (no prior runs)
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        existing_run_dir, checkpoint_to_load = _detect_checkpoint(output_dir)
        assert existing_run_dir is None
        assert checkpoint_to_load is None

    def test_existing_checkpoint_detected(self, tmp_path):
        """When a run dir with checkpoint.pt exists, it is detected."""
        output_dir = str(tmp_path / "output")
        run_dir = os.path.join(output_dir, "run_20260101_000000_seed42")
        os.makedirs(run_dir)

        # Create a dummy checkpoint
        torch.save({'episode': 25}, os.path.join(run_dir, "checkpoint.pt"))

        existing_run_dir, checkpoint_to_load = _detect_checkpoint(output_dir)
        assert existing_run_dir == run_dir
        assert checkpoint_to_load == os.path.join(run_dir, "checkpoint.pt")

    def test_most_recent_run_dir_used(self, tmp_path):
        """When multiple run dirs exist, the latest one with a checkpoint is used."""
        output_dir = str(tmp_path / "output")

        for ts in ["run_20260101_000000", "run_20260102_000000", "run_20260103_000000"]:
            d = os.path.join(output_dir, ts)
            os.makedirs(d)
            torch.save({'episode': int(ts[-6:])}, os.path.join(d, "checkpoint.pt"))

        existing_run_dir, checkpoint_to_load = _detect_checkpoint(output_dir)
        assert "20260103" in existing_run_dir

    def test_run_dir_without_checkpoint_not_used(self, tmp_path):
        """Abandoned run dirs without checkpoint.pt are ignored."""
        output_dir = str(tmp_path / "output")
        run_dir = os.path.join(output_dir, "run_20260101_000000_seed42")
        os.makedirs(run_dir)
        # No checkpoint.pt written

        existing_run_dir, checkpoint_to_load = _detect_checkpoint(output_dir)
        assert existing_run_dir is None
        assert checkpoint_to_load is None


def _detect_checkpoint(output_dir):
    """Mirror of the detection logic in ace_experiments.main()."""
    existing_run_dir = None
    checkpoint_to_load = None
    if os.path.isdir(output_dir):
        candidates = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("run_")
        ])
        for candidate in reversed(candidates):
            candidate_path = os.path.join(output_dir, candidate)
            checkpoint_path = os.path.join(candidate_path, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                existing_run_dir = candidate_path
                checkpoint_to_load = checkpoint_path
                break
    return existing_run_dir, checkpoint_to_load


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
