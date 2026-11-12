import math, io, contextlib
import numpy as np, pandas as pd
from acelib import *
import static_designs as sd
K = 4000
sd.K = K
S, SEQ = {}, {}
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    S[wi] = (posterior(parents, noise, 1.0, hist[:8]), parents, noise)
    SEQ[wi] = {pol: [h["action_id"] for h in load_arm(pi, wi)["terminal"]["final_state"]["history"][8:]] for pi, pol in enumerate(POLICIES)}

def paths(heads, parents, noise, seq, cost, seed, eps_seed):
    b = sd.Batch(heads, parents, noise, seed)                 # same seed => same parameter draws (same fantasy system)
    b.eps = np.random.default_rng(eps_seed).standard_normal((K, 40, 3))   # arm-specific acquisition noise
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

designs = lambda wi: {"TR": SEQ[wi]["target_risk"], "LI": SEQ[wi]["local_information"], "coverage": SEQ[wi]["menu_coverage"],
                      "random": SEQ[wi]["uniform_random"], "ranker": SEQ[wi]["learned_ranker"], "frozen(obs24)": ["observe_all"] * 24, "DPO(xneg24)": ["set_x_neg"] * 24}
names = ["TR", "LI", "coverage", "random", "ranker", "frozen(obs24)", "DPO(xneg24)"]
R = {}
for wi in range(24):
    for j, (n, seq) in enumerate(designs(wi).items()):
        R[(wi, n)] = paths(*S[wi], seq, meta(wi)[2], 4242 + wi, 1000003 * (j + 1) + wi)
print("== predicted paired contrasts, one episode per system, screen-like pairing (shared system+init, independent acquisition noise) ==")
print("contrast | endpoint: expected mean diff, predicted SE(n=24), z, systems needed for z=2.8 with 1 episode each")
obs = {("TR", "coverage"): (-0.833, 0.627), ("TR", "random"): (-2.625, 1.108), ("TR", "DPO(xneg24)"): (-3.542, 1.313), ("TR", "frozen(obs24)"): (-2.0, 1.381),
       ("TR", "LI"): (-0.333, 1.268), ("TR", "ranker"): (-1.958, 1.353), ("DPO(xneg24)", "frozen(obs24)"): (1.542, 1.558), ("DPO(xneg24)", "random"): (0.917, 1.616)}
rows = []
for (a, b), (om, ose) in obs.items():
    line = f"{a:12s} - {b:14s}"
    for label, idx, tf in [("capped cost", 0, None), ("log final realized NMSE", 1, np.log), ("log final posterior risk", 2, np.log)]:
        dm, dv = [], []
        for wi in range(24):
            A, B = R[(wi, a)][idx], R[(wi, b)][idx]
            if tf is not None: A, B = tf(A), tf(B)
            dm.append((A - B).mean()); dv.append((A - B).var(ddof=1))
        mean = float(np.mean(dm)); se = math.sqrt(np.sum(dv)) / 24
        # between-system heterogeneity of the expected difference also contributes when systems are sampled:
        se_tot = math.sqrt(np.sum(dv) / 24 ** 2 + np.var(dm, ddof=1) / 24)
        need = (2.8 * se_tot * math.sqrt(24) / abs(mean)) ** 2 if mean != 0 else float("inf")
        line += f" | {label}: {mean:+.3f}, SE {se_tot:.3f}, z {mean/se_tot:+.2f}, n*={need:.0f}"
        rows.append(dict(a=a, b=b, endpoint=label, exp_diff=mean, se24=se_tot, z=mean / se_tot, n_needed=need))
    print(line + f" | OBSERVED capped: {om:+.3f} ± {ose:.3f}")
pd.DataFrame(rows).to_csv("out_endpoint_power_model.csv", index=False)
print("\nwithin-system SD of a single-episode capped cost (sqrt mean variance over systems):", {n: round(math.sqrt(np.mean([R[(wi, n)][0].var(ddof=1) for wi in range(24)])), 2) for n in names})
print("CV of final realized NMSE within system (median over systems):", {n: round(float(np.median([R[(wi, n)][1].std(ddof=1) / R[(wi, n)][1].mean() for wi in range(24)])), 2) for n in names})
print("CV of final posterior risk within system (median over systems):", {n: round(float(np.median([R[(wi, n)][2].std(ddof=1) / R[(wi, n)][2].mean() for wi in range(24)])), 2) for n in names})
