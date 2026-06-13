#!/usr/bin/env python3
"""Shared helpers for scaling-sweep and K-ablation aggregation/plots."""
from __future__ import annotations

import os
from glob import glob
from typing import Any

import pandas as pd

SCALING_METHODS = ["ace", "zero_shot_lm", "random"]
SCALING_SCALES = [15, 30, 50]
K_VALUES = [4, 8, 16]
PLATEAU_EP_BUDGET = 40


def latest_file(seed_dir: str, name: str) -> str | None:
    candidates = glob(os.path.join(seed_dir, "**", name), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def summarize_node_losses(nl_path: str) -> dict[str, Any] | None:
    df = pd.read_csv(nl_path)
    if "episode" not in df.columns or "total_loss" not in df.columns:
        return None
    n_nodes = sum(1 for c in df.columns if c.startswith("loss_"))
    per_ep = (
        df.groupby("episode")["total_loss"]
        .last()
        .reset_index()
        .sort_values("episode")
    )
    if per_ep.empty or n_nodes == 0:
        return None
    best = float(per_ep["total_loss"].min())
    final = float(per_ep["total_loss"].iloc[-1])
    n_ep = int(per_ep["episode"].nunique())
    best_ep = int(per_ep.loc[per_ep["total_loss"].idxmin(), "episode"])
    return {
        "best": best,
        "final": final,
        "n_ep": n_ep,
        "n_nodes": n_nodes,
        "best_episode": best_ep,
        "per_node_best": best / n_nodes,
        "per_node_final": final / n_nodes,
    }


def summarize_summary_csv(path: str) -> dict[str, Any] | None:
    df = pd.read_csv(path)
    if "min_total_loss" not in df.columns:
        return None
    row = df.iloc[0]
    n_nodes = int(row.get("n_nodes", 0)) or 1
    best = float(row["min_total_loss"])
    final = float(row.get("final_total_loss", row.get("last_total_loss", best)))
    n_ep = int(row.get("n_episodes", row.get("episodes", 0)))
    return {
        "best": best,
        "final": final,
        "n_ep": n_ep,
        "n_nodes": n_nodes,
        "best_episode": int(row.get("best_episode", 0)),
        "per_node_best": best / n_nodes,
        "per_node_final": final / n_nodes,
    }


def summarize_seed_dir(seed_dir: str) -> dict[str, Any] | None:
    nl = latest_file(seed_dir, "node_losses.csv")
    if nl:
        return summarize_node_losses(nl)
    summ = latest_file(seed_dir, "summary.csv")
    if summ:
        return summarize_summary_csv(summ)
    return None


def metrics_cost(seed_dir: str) -> dict[str, float]:
    met = latest_file(seed_dir, "metrics.csv")
    out = {"prompt_tokens_mean": float("nan"), "peak_vram_gb": float("nan")}
    if not met or not os.path.isfile(met):
        return out
    try:
        mdf = pd.read_csv(met)
    except Exception:
        return out
    for col, key in (("prompt_tokens_mean", "prompt_tokens_mean"),
                     ("peak_vram_gb", "peak_vram_gb")):
        if col in mdf.columns and not mdf[col].empty:
            out[key] = float(mdf[col].iloc[-1])
    return out


def discover_seeds(root: str, scale: int, method: str) -> list[int]:
    base = os.path.join(root, f"nodes{scale}", method)
    if not os.path.isdir(base):
        return []
    seeds = []
    for d in glob(os.path.join(base, "seed_*")):
        try:
            seeds.append(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(seeds))
