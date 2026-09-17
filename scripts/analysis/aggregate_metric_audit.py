#!/usr/bin/env python3
"""Put ACE and baseline runs on ONE metric (Sept 2026 metric audit).

Two per-step files exist in the repo:

* ACE runs (``node_losses.csv`` from ace_experiments.py): ``total_loss`` is ACE's
  broad-range, root-weighted score; ``loss_<node>`` are unweighted per-node
  values on that same broad-range evaluation; ``obs_total_loss``/``obs_loss_*``
  (runs after 10 Sept 2026) are the observational, unweighted convention.
* Baseline runs (``results.csv``/``node_losses.csv`` from baselines.py or the
  scripts/runners/*_baseline_seed.py runners): ``total_loss``/``loss_*`` are the
  observational, unweighted convention; ``ace_total_loss``/``ace_loss_*`` (runs
  after 10 Sept 2026) are ACE's evaluator on the same student.

For every run this script derives, per step, the same four quantities:

  broad_w   root-weighted broad-range total (what ACE's Table 1/2 numbers are)
  broad_nr  broad-range loss summed over NON-ROOT nodes (the learnable part)
  broad_u   unweighted broad-range total
  obs_u     observational unweighted total (NaN when the run predates the audit)

and reports best-over-steps and final for each, per arm, with mean +/- sd and
the per-node version of broad_nr (divided by the number of non-root nodes).
Roots are identified structurally (5-node: X1, X4; LargeScaleSCM: the first
layer, X1..X_r with r = 5 at N=30, else max(2, N//6)); a run with anonymised
node names falls back to "loss never below 0.5 and nearly constant", and says so.

Usage
  python scripts/analysis/aggregate_metric_audit.py --reruns <root>            # the audit re-run tree
  python scripts/analysis/aggregate_metric_audit.py --arm ACE30=results/curc_20260415_152624/large_scale \
         --arm Random30=results/audit_reruns/t2_30node/random --primary broad_nr_pn_best
Any number of --arm LABEL=DIR; DIR is searched recursively for per-step files.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

STEP_FILES = ("node_losses.csv", "results.csv")


def find_runs(root: str) -> list[str]:
    out = []
    for name in STEP_FILES:
        out += glob.glob(os.path.join(root, "**", name), recursive=True)
    # one file per run dir: prefer node_losses.csv when both exist
    by_dir: dict[str, str] = {}
    for f in sorted(out):
        d = os.path.dirname(f)
        if d not in by_dir or f.endswith("node_losses.csv"):
            by_dir[d] = f
    return sorted(by_dir.values())


def structural_roots(nodes: list[str]) -> set[str] | None:
    if not all(re.fullmatch(r"X\d+", n) for n in nodes):
        return None
    n = max(int(x[1:]) for x in nodes)
    if n <= 5:
        return {"X1", "X4"}
    r = 5 if n == 30 else max(2, n // 6)
    return {f"X{i}" for i in range(1, r + 1)}


def heuristic_roots(df: pd.DataFrame, cols: list[str]) -> set[str]:
    out = set()
    for c in cols:
        v = df[c]
        if v.min() > 0.5 and (v.std() / max(v.mean(), 1e-9)) < 0.15:
            out.add(c.split("_", 1)[1] if c.startswith("loss_") else c.split("_", 2)[2])
    return out


MAX_EPISODE: int | None = None  # set from --max-episode


def node_of(col: str) -> str:
    return col.split("loss_", 1)[1]


def summarize_run(path: str) -> dict | None:
    df = pd.read_csv(path)
    if "total_loss" not in df.columns:
        return None
    if MAX_EPISODE is not None and "episode" in df.columns:
        df = df[df["episode"] <= MAX_EPISODE]
        if df.empty:
            return None
    loss_cols = [c for c in df.columns if c.startswith("loss_")]
    ace_cols = [c for c in df.columns if c.startswith("ace_loss_")]
    obs_cols_ = [c for c in df.columns if c.startswith("obs_loss_")]
    if not loss_cols:
        return None
    nodes = [node_of(c) for c in loss_cols]
    roots = structural_roots(nodes)
    root_mode = "structural"
    if roots is None:
        roots = heuristic_roots(df, loss_cols)
        root_mode = "heuristic"
    roots = {r for r in roots if r in nodes}
    non_roots = [n for n in nodes if n not in roots]

    # Which convention does 'total_loss' follow? Decide by reconstruction, never by
    # file name: ACE's total is 0.2*roots + non-roots; the baselines' is the plain sum.
    R = sum(df[c] for c in loss_cols if node_of(c) in roots) if roots else 0.0
    NR = sum(df[c] for c in loss_cols if node_of(c) not in roots)
    err_w = float((0.2 * R + NR - df["total_loss"]).abs().max())
    err_u = float((R + NR - df["total_loss"]).abs().max())
    if ace_cols:                      # post-audit baseline run: obs in loss_*, broad in ace_loss_*
        kind, broad_cols, obs_cols = "baseline", ace_cols, loss_cols
    elif obs_cols_:                   # post-audit ACE run: broad in loss_*, obs in obs_loss_*
        kind, broad_cols, obs_cols = "ace", loss_cols, obs_cols_
    elif err_w < 1e-5:                # legacy ACE run
        kind, broad_cols, obs_cols = "ace", loss_cols, []
    elif err_u < 1e-5:                # legacy baseline run
        kind, broad_cols, obs_cols = "baseline", [], loss_cols
    else:
        print(f"[warn] cannot classify {path}: total_loss matches neither convention "
              f"(weighted err {err_w:.3g}, unweighted err {err_u:.3g}); skipped", file=sys.stderr)
        return None

    rec = {"path": path, "kind": kind, "n_nodes": len(nodes), "n_roots": len(roots),
           "root_mode": root_mode, "steps": len(df),
           "episodes": int(df["episode"].max()) + 1 if "episode" in df.columns else np.nan,
           "recon_err": min(err_w, err_u)}
    if broad_cols:
        b = {node_of(c): df[c] for c in broad_cols}
        broad_w = sum((0.2 if n in roots else 1.0) * v for n, v in b.items())
        broad_nr = sum(v for n, v in b.items() if n not in roots)
        broad_u = sum(v for v in b.values())
        for name, ser in (("broad_w", broad_w), ("broad_nr", broad_nr), ("broad_u", broad_u)):
            rec[f"{name}_best"], rec[f"{name}_final"] = float(ser.min()), float(ser.iloc[-1])
        rec["broad_nr_pn_best"] = rec["broad_nr_best"] / max(len(non_roots), 1)
        rec["broad_nr_pn_final"] = rec["broad_nr_final"] / max(len(non_roots), 1)
        # End-of-campaign: the student is re-initialised at the start of every
        # episode and trained through a fixed number of steps, so the loss at an
        # episode's last step is the outcome of one full campaign. Averaging it
        # over episodes (and over the last 20) is a min-free summary that does
        # not depend on picking the single best step of the whole run.
        if "episode" in df.columns and "step" in df.columns:
            last = df.loc[df.groupby("episode")["step"].idxmax()]
            eoc = (broad_nr.loc[last.index] / max(len(non_roots), 1))
            rec["eoc_nr_pn_mean"] = float(eoc.mean())
            rec["eoc_nr_pn_last20"] = float(eoc.iloc[-20:].mean())
            rec["eoc_nr_pn_median"] = float(eoc.median())
    if obs_cols:
        o = {node_of(c): df[c] for c in obs_cols}
        obs_u = sum(o.values())
        obs_nr = sum(v for n, v in o.items() if n not in roots)
        rec["obs_u_best"], rec["obs_u_final"] = float(obs_u.min()), float(obs_u.iloc[-1])
        rec["obs_nr_pn_best"] = float(obs_nr.min()) / max(len(non_roots), 1)
        rec["obs_nr_pn_final"] = float(obs_nr.iloc[-1]) / max(len(non_roots), 1)
    m = re.search(r"seed_?(\d+)", path)
    rec["seed"] = int(m.group(1)) if m else -1
    return rec


def welch(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy import stats
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="LABEL=DIR (repeatable)")
    ap.add_argument("--reruns", help="root of the audit re-run tree; adds one arm per <suite>/<method>")
    ap.add_argument("--primary", default="eoc_nr_pn_mean",
                    help="metric for the pairwise tests (default: end-of-campaign non-root broad-range MSE per node, mean over episodes)")
    ap.add_argument("--only", default=None, help="comma-separated subset of metric columns to print")
    ap.add_argument("--csv", help="write the per-run table here")
    ap.add_argument("--max-episode", type=int, help="truncate every run to episodes <= this before scoring (budget-matched comparisons)")
    args = ap.parse_args()
    global MAX_EPISODE
    MAX_EPISODE = args.max_episode

    arms: list[tuple[str, str]] = [tuple(a.split("=", 1)) for a in args.arm]  # type: ignore[misc]
    if args.reruns:
        for suite in sorted(os.listdir(args.reruns)):
            sdir = os.path.join(args.reruns, suite)
            if not os.path.isdir(sdir) or suite == "logs":
                continue
            for method in sorted(os.listdir(sdir)):
                mdir = os.path.join(sdir, method)
                if os.path.isdir(mdir) and method != "logs":
                    arms.append((f"{suite}/{method}", mdir))
    if not arms:
        ap.error("give --arm LABEL=DIR and/or --reruns ROOT")

    rows = []
    for label, d in arms:
        runs = find_runs(d)
        if not runs:
            print(f"[warn] no per-step files under {d}", file=sys.stderr)
        for f in runs:
            r = summarize_run(f)
            if r:
                r["arm"] = label
                rows.append(r)
    if not rows:
        print("nothing to aggregate"); return 1
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False); print(f"per-run table -> {args.csv}")

    metrics = ["broad_w_best", "broad_w_final", "broad_nr_pn_best", "broad_nr_pn_final",
               "eoc_nr_pn_mean", "eoc_nr_pn_last20", "eoc_nr_pn_median",
               "broad_u_best", "broad_u_final", "obs_u_best", "obs_u_final", "obs_nr_pn_best", "obs_nr_pn_final"]
    metrics = [m for m in metrics if m in df.columns]
    if args.only:
        keep = [m.strip() for m in args.only.split(",")]
        metrics = [m for m in metrics if m in keep]
    pd.set_option("display.width", 220)
    print("\n== per-arm mean +/- sd (n runs) ==")
    g = df.groupby("arm")
    tbl = pd.DataFrame({m: g[m].apply(lambda s: f"{s.mean():.4f}+/-{s.std(ddof=1):.4f}" if s.notna().sum() > 1 else (f"{s.mean():.4f}" if s.notna().any() else "--")) for m in metrics})
    tbl.insert(0, "n", g.size()); tbl.insert(1, "nodes", g["n_nodes"].mean().round(1))
    tbl.insert(2, "episodes", g["episodes"].mean().round(0))
    print(tbl.to_string())
    bad = df[df["recon_err"] > 1e-5]
    if len(bad):
        print(f"\n[warn] {len(bad)} run(s) whose total_loss is not reproduced from per-node columns (root set wrong?):")
        print(bad[["arm", "seed", "recon_err", "root_mode"]].to_string(index=False))
    if any(df["root_mode"] == "heuristic"):
        print(f"\n[note] {int((df['root_mode']=='heuristic').sum())} run(s) used heuristic root detection (anonymised names)")

    ace_arms = [a for a in df["arm"].unique() if (df[df.arm == a]["kind"] == "ace").all()]
    base_arms = [a for a in df["arm"].unique() if a not in ace_arms]
    if ace_arms and base_arms and args.primary in df.columns:
        print(f"\n== Welch p, ACE arm vs baseline arm, on {args.primary} (lower is better; direction shown) ==")
        for a in ace_arms:
            va = df[df.arm == a][args.primary].to_numpy(float)
            for b in base_arms:
                vb = df[df.arm == b][args.primary].to_numpy(float)
                if np.isnan(vb).all():
                    continue
                sign = "<" if np.nanmean(va) < np.nanmean(vb) else ">"
                print(f"  {a:<34} {np.nanmean(va):.4f} {sign} {np.nanmean(vb):.4f} {b:<34} p={welch(va, vb):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
