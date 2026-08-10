"""Unit tests for resolve_local_hf_snapshot (HF 429 avoidance)."""

import os

import pytest


@pytest.mark.unit
def test_passthrough_when_already_a_directory(tmp_path):
    from ace_experiments import resolve_local_hf_snapshot

    d = tmp_path / "my_model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert resolve_local_hf_snapshot(str(d)) == str(d)


@pytest.mark.unit
def test_passthrough_when_no_cache(tmp_path):
    from ace_experiments import resolve_local_hf_snapshot

    assert resolve_local_hf_snapshot(
        "Qwen/Qwen2.5-3B", hf_home=str(tmp_path)
    ) == "Qwen/Qwen2.5-3B"


@pytest.mark.unit
def test_resolves_hub_layout_snapshot(tmp_path):
    from ace_experiments import resolve_local_hf_snapshot

    snap = (tmp_path / "hub" / "models--Qwen--Qwen2.5-3B" / "snapshots" / "abcd1234")
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"model_type": "qwen2"}')
    resolved = resolve_local_hf_snapshot("Qwen/Qwen2.5-3B", hf_home=str(tmp_path))
    assert resolved == str(snap)


@pytest.mark.unit
def test_prefers_newest_snapshot(tmp_path):
    from ace_experiments import resolve_local_hf_snapshot
    import time

    root = tmp_path / "hub" / "models--Qwen--Qwen2.5-1.5B" / "snapshots"
    old = root / "oldsha"
    new = root / "newsha"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    (new / "config.json").write_text("{}")
    # Ensure new is newer.
    os.utime(old, (1_000_000_000, 1_000_000_000))
    os.utime(new, (2_000_000_000, 2_000_000_000))
    resolved = resolve_local_hf_snapshot("Qwen/Qwen2.5-1.5B", hf_home=str(tmp_path))
    assert resolved == str(new)


@pytest.mark.unit
def test_passthrough_non_hub_id():
    from ace_experiments import resolve_local_hf_snapshot

    assert resolve_local_hf_snapshot("gpt2") == "gpt2"
