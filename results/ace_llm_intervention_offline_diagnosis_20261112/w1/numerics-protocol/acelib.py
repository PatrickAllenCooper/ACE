"""Independent re-implementation (read-only analysis) of the ACE linear-Gaussian learner,
structural moments and integrated risk, for offline replay of SAVED histories only.
No environment responses are generated here."""
import json, os
import numpy as np

CAMP = ("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/"
        "terminal02/remote/launch/cpu/worker/artifacts/campaign")
POLICIES = ["uniform_random", "menu_coverage", "local_information", "target_risk", "learned_ranker"]
NODES = ["X", "Y", "Z"]
QUERY_CLAMPS = [{}, {"X": -1.0}, {"X": 0.0}, {"X": 1.0}, {"Y": -1.0}, {"Y": 0.0}, {"Y": 1.0}, {"X": -1.0, "Y": 1.0}]
ACTIONS = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos", "set_y_neg", "set_y_zero", "set_y_pos", "set_xy"]
ACTION_CLAMPS = dict(zip(ACTIONS, QUERY_CLAMPS))
SCALE = 2.0


def load_arm(pi, wi):
    base = f"{CAMP}/policy-{pi:02d}/world-{wi:02d}"
    world = json.load(open(base + "/world.json"))
    learner = json.load(open(base + "/learner.json"))
    term = json.load(open(base + "/episode/terminal.json"))
    plan = json.load(open(base + "/episode/plan.json"))
    decisions = []
    k = 0
    while os.path.exists(f"{base}/episode/decision-{k:04d}.input.json"):
        inp = json.load(open(f"{base}/episode/decision-{k:04d}.input.json"))
        proc = json.load(open(f"{base}/episode/decision-{k:04d}.process.json"))
        decisions.append((inp, proc))
        k += 1
    return {"world": world, "learner": learner, "terminal": term, "plan": plan, "decisions": decisions, "base": base}


def graph_parents(world_or_learner):
    return {n["name"]: list(n["parents"]) for n in world_or_learner["graph"]}


def posterior(parents, noise, prior_var, history):
    """Batch conjugate posterior per head from accepted receipts (same eligibility rule as the learner)."""
    heads = {}
    for node in NODES:
        pa = parents[node]
        d = 1 + len(pa)
        P = np.eye(d) / prior_var
        r = np.zeros(d)
        n = 0
        for rec in history:
            if node in rec["targets"] or node not in rec["values"]:
                continue
            avail = dict(rec["values"], **rec["targets"])
            if not set(pa) <= set(avail):
                continue
            phi = np.array([1.0] + [avail[p] for p in pa])
            P += np.outer(phi, phi) / noise[node]
            r += phi * rec["values"][node] / noise[node]
            n += 1
        V = np.linalg.inv(P)
        V = (V + V.T) / 2
        heads[node] = (V @ r, V, n)
    return heads


def moments(heads, parents, clamps):
    """Exact mean/covariance of the noise-disabled composed mapping (mirrors target_risk.structural_moments)."""
    idx = {n: i for i, n in enumerate(NODES)}
    mu = np.zeros(3)
    C = np.zeros((3, 3))
    for i, node in enumerate(NODES):
        if node in clamps:
            mu[i] = clamps[node]
            continue
        m, V = heads[node][0], heads[node][1]
        pa = [idx[p] for p in parents[node]]
        phim = np.r_[1.0, mu[pa]]
        phiC = np.zeros((len(pa) + 1, len(pa) + 1))
        phiC[1:, 1:] = C[np.ix_(pa, pa)]
        mu[i] = m @ phim
        if pa:
            C[i, :i] = m[1:] @ C[pa, :i]
        C[:i, i] = C[i, :i]
        C[i, i] = np.sum(V * phiC) + phim @ V @ phim + m @ phiC @ m
    return mu, (C + C.T) / 2


def risk(heads, parents, per_query=False):
    v = np.array([max(0.0, moments(heads, parents, q)[1][2, 2]) for q in QUERY_CLAMPS]) / SCALE ** 2
    return (float(v.mean()), v) if per_query else float(v.mean())


def predict(heads, parents, clamps):
    vals = dict(clamps)
    for node in NODES:
        if node not in vals:
            m = heads[node][0]
            vals[node] = float(m @ np.r_[1.0, [vals[p] for p in parents[node]]])
    return vals


def truth_mean(world, clamps):
    vals = dict(clamps)
    for mech in world["mechanisms"]:
        if mech["node"] not in vals:
            vals[mech["node"]] = mech["intercept"] + sum(c * vals[p] for p, c in mech["coefficients"])
    return vals


def nmse(heads, parents, world):
    e = [(predict(heads, parents, q)["Z"] - truth_mean(world, q)["Z"]) / SCALE for q in QUERY_CLAMPS]
    return float(np.mean(np.square(e)))


def meta(wi):
    regime = ["matched_low", "matched_high", "shifted_low", "shifted_high"][wi // 6]
    graph = ["chain", "fork", "collider"][(wi % 6) // 2]
    cost = ["uniform", "heterogeneous"][wi % 2]
    return regime, graph, cost


def action_cost(action, cost):
    return 2 if (cost == "heterogeneous" and ACTIONS.index(action) >= 4) else 1
