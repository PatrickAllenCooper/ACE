"""Predicted declared-endpoint contrast between target-risk-like sequences and GOOD static schedules
(leave-one-out public schedule per graph x cost cell; plan-once from the post-init state). Model-internal."""
import math
import numpy as np, pandas as pd
from acelib import *
import static_designs as sd
sd.K = 2000
import importlib
def shared_greedy(S, worlds, cost, seed_base=31000, budget=24):
    bs = {w: sd.Batch(*S[w], seed_base + w) for w in worlds}
    cur = {w: float(bs[w].risk().mean()) for w in worlds}
    design, left = [], budget
    while True:
        feas = [a for a in ACTIONS if action_cost(a, cost) <= left]
        if not feas: break
        best = None
        for a in feas:
            vals = {w: float(bs[w].copy().add(a).risk().mean()) for w in worlds}
            score = np.mean([math.log(cur[w]) - math.log(vals[w]) for w in worlds]) / action_cost(a, cost)
            if best is None or score > best[0] + 1e-15: best = (score, a, vals)
        design.append(best[1]); left -= action_cost(best[1], cost)
        for w in worlds: bs[w].add(best[1])
        cur = best[2]
    return design
S, SEQ = {}, {}
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    S[wi] = (posterior(parents, noise, 1.0, hist[:8]), parents, noise)
    SEQ[wi] = [h["action_id"] for h in load_arm(3, wi)["terminal"]["final_state"]["history"][8:]]
cells = {}
for wi in range(24): cells.setdefault(meta(wi)[1:], []).append(wi)
LOO, PLAN = {}, {}
for (graph, cost), ws in cells.items():
    for w in ws:
        LOO[w] = shared_greedy(S, [x for x in ws if x != w], cost)
        PLAN[w] = shared_greedy(S, [w], cost, seed_base=77000)
sd.K = 4000
K = 4000
def paths(heads, parents, noise, seq, cost, seed, eps_seed):
    b = sd.Batch(heads, parents, noise, seed)
    b.eps = np.random.default_rng(eps_seed).standard_normal((K, 40, 3))
    truth = []
    for q in QUERY_CLAMPS:
        vals = {}
        for n in NODES:
            if n in q: vals[n] = np.full(K, q[n]); continue
            phi = np.stack([np.ones(K)] + [vals[p] for p in parents[n]], 1)
            vals[n] = np.einsum("kd,kd->k", phi, b.theta[n])
        truth.append(vals["Z"])
    truth = np.stack(truth, 1)
    def nm():
        m = {n: np.linalg.solve(b.P[n], b.r[n][:, :, None])[:, :, 0] for n in NODES}
        preds = []
        for q in QUERY_CLAMPS:
            vals = {}
            for n in NODES:
                if n in q: vals[n] = np.full(K, q[n]); continue
                phi = np.stack([np.ones(K)] + [vals[p] for p in parents[n]], 1)
                vals[n] = np.einsum("kd,kd->k", phi, m[n])
            preds.append(vals["Z"])
        return (((np.stack(preds, 1) - truth) / SCALE) ** 2).mean(1)
    first = np.full(K, 24.0)
    e = nm(); hit = e <= 0.04; first[hit] = 0; crossed = hit.copy(); spent = 0
    for a in seq:
        b.add(a); spent += action_cost(a, cost)
        e = nm(); hit = (e <= 0.04) & ~crossed
        first[hit] = spent; crossed |= hit
    return first, e, b.risk()
R = {}
for wi in range(24):
    for j, (n, seq) in enumerate([("TR", SEQ[wi]), ("LOO_public_static", LOO[wi]), ("plan_once", PLAN[wi])]):
        R[(wi, n)] = paths(*S[wi], seq, meta(wi)[2], 4242 + wi, 2000003 * (j + 1) + wi)
for n in ["TR", "LOO_public_static", "plan_once"]:
    print(f"{n:20s} expected capped total {np.mean([R[(wi, n)][0].mean() + 8 for wi in range(24)]):.3f} ; expected final NMSE {np.mean([R[(wi, n)][1].mean() for wi in range(24)]):.4f} ; expected final-ok count {np.sum([(R[(wi, n)][1] <= 0.04).mean() for wi in range(24)]):.2f}")
for a, b in [("TR", "LOO_public_static"), ("plan_once", "LOO_public_static"), ("TR", "plan_once")]:
    line = f"{a} - {b}"
    for label, idx, tf in [("capped cost", 0, None), ("log final realized NMSE", 1, np.log), ("log final posterior risk", 2, np.log)]:
        dm, dv = [], []
        for wi in range(24):
            A, B = R[(wi, a)][idx], R[(wi, b)][idx]
            if tf is not None: A, B = tf(A), tf(B)
            dm.append((A - B).mean()); dv.append((A - B).var(ddof=1))
        mean = float(np.mean(dm)); se = math.sqrt(np.sum(dv) / 24 ** 2 + np.var(dm, ddof=1) / 24)
        need = (2.8 * se * math.sqrt(24) / abs(mean)) ** 2
        line += f" | {label}: {mean:+.3f}, SE(24) {se:.3f}, z {mean/se:+.2f}, n*={need:.0f}"
    print(line)
