"""Model-internal (working-model) preposterior risk of OPEN-LOOP designs, starting from the SAVED
post-initialization public state of each development system.

Nothing here is an environment response: parameters are drawn from the saved public posterior and
hypothetical ("fantasy") responses are drawn from the learner's own predictive law, exactly the kind of
computation choose_target_risk performs one step ahead. No private truth is used. Fixed seeds.
J(d) = E[ integrated_risk(posterior after saved 8 rows + fantasy rows of design d) | saved state ]."""
import json, math, sys
import numpy as np, pandas as pd
from acelib import *

OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/"
K = 4000
IDX = {n: i for i, n in enumerate(NODES)}


class Batch:
    """K parallel fantasy posteriors sharing the saved initial state."""
    def __init__(self, heads, parents, noise, seed):
        self.parents, self.noise = parents, noise
        rng = np.random.default_rng(seed)
        self.theta, self.P, self.r = {}, {}, {}
        for n in NODES:
            m, V, _ = heads[n]
            L = np.linalg.cholesky(V)
            self.theta[n] = m + rng.standard_normal((K, len(m))) @ L.T
            P0 = np.linalg.inv(V)
            self.P[n] = np.repeat(P0[None], K, 0)
            self.r[n] = np.repeat((P0 @ m)[None], K, 0)
        self.eps = rng.standard_normal((K, 40, 3))  # common random numbers by row position
        self.rows = 0

    def copy(self):
        b = Batch.__new__(Batch)
        b.parents, b.noise, b.theta, b.eps, b.rows = self.parents, self.noise, self.theta, self.eps, self.rows
        b.P = {n: v.copy() for n, v in self.P.items()}
        b.r = {n: v.copy() for n, v in self.r.items()}
        return b

    def add(self, action):
        clamps = ACTION_CLAMPS[action]
        vals = {}
        for n in NODES:
            if n in clamps:
                vals[n] = np.full(K, clamps[n])
                continue
            phi = np.stack([np.ones(K)] + [vals[p] for p in self.parents[n]], 1)
            y = np.einsum("kd,kd->k", phi, self.theta[n]) + math.sqrt(self.noise[n]) * self.eps[:, self.rows, IDX[n]]
            vals[n] = y
            self.P[n] += np.einsum("ki,kj->kij", phi, phi) / self.noise[n]
            self.r[n] += phi * y[:, None] / self.noise[n]
        self.rows += 1
        return self

    def risk(self):
        V = {n: np.linalg.inv(self.P[n]) for n in NODES}
        m = {n: np.einsum("kij,kj->ki", V[n], self.r[n]) for n in NODES}
        tot = np.zeros(K)
        for clamps in QUERY_CLAMPS:
            mu = np.zeros((K, 3)); C = np.zeros((K, 3, 3))
            for i, n in enumerate(NODES):
                if n in clamps:
                    mu[:, i] = clamps[n]
                    continue
                pa = [IDX[p] for p in self.parents[n]]
                d = len(pa) + 1
                phim = np.concatenate([np.ones((K, 1)), mu[:, pa]], 1)
                phiC = np.zeros((K, d, d))
                if pa:
                    phiC[:, 1:, 1:] = C[:, pa][:, :, pa]
                    C[:, i, :i] = np.einsum("kp,kpj->kj", m[n][:, 1:], C[:, pa, :i])
                    C[:, :i, i] = C[:, i, :i]
                mu[:, i] = np.einsum("kd,kd->k", m[n], phim)
                C[:, i, i] = (np.einsum("kij,kij->k", V[n], phiC) + np.einsum("ki,kij,kj->k", phim, V[n], phim)
                              + np.einsum("ki,kij,kj->k", m[n], phiC, m[n]))
            tot += np.maximum(C[:, 2, 2], 0) / SCALE ** 2
        return tot / len(QUERY_CLAMPS)


def J(heads, parents, noise, design, seed):
    b = Batch(heads, parents, noise, seed)
    for a in design:
        b.add(a)
    r = b.risk()
    return float(r.mean()), float(r.std(ddof=1)), float(r.std(ddof=1) / math.sqrt(K))


def plan_once(heads, parents, noise, cost, seed, budget=24, rule="ratio"):
    """Greedy open-loop plan computed ONLY from the saved post-initialization state (no later data)."""
    b = Batch(heads, parents, noise, seed)
    cur = float(b.risk().mean())
    design, left = [], budget
    while True:
        feas = [a for a in ACTIONS if action_cost(a, cost) <= left]
        if not feas:
            break
        best = None
        for a in feas:
            val = float(b.copy().add(a).risk().mean())
            score = (cur - val) / action_cost(a, cost)
            if best is None or score > best[0] + 1e-15:
                best = (score, a, val)
        design.append(best[1]); b.add(best[1]); cur = best[2]; left -= action_cost(best[1], cost)
    return design, cur


if __name__ == "__main__":
    states = {}
    for wi in range(24):
        fin = {}
        for pi, pol in enumerate(POLICIES):
            arm = load_arm(pi, wi)
            hist = arm["terminal"]["final_state"]["history"]
            parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]; pv = arm["learner"]["prior_variance"]
            fin[pol] = dict(design=[h["action_id"] for h in hist[8:]], final_post=risk(posterior(parents, noise, pv, hist), parents))
            if pi == 0:
                h8 = posterior(parents, noise, pv, hist[:8])
        states[wi] = dict(heads=h8, parents=parents, noise=noise, fin=fin)
    rows = []
    for wi in range(24):
        regime, graph, cost = meta(wi)
        s = states[wi]
        h, pa, nz = s["heads"], s["parents"], s["noise"]
        seed = 9000 + wi
        rec = dict(world=wi, regime=regime, graph=graph, cost=cost, R8=risk(h, pa))
        for pol in POLICIES:
            rec[f"real_{pol}"] = s["fin"][pol]["final_post"]
            j, sd, se = J(h, pa, nz, s["fin"][pol]["design"], seed)
            rec[f"J_{pol}"], rec[f"sd_{pol}"] = j, sd
        partner = (wi + 12) % 24
        rec["J_TRpartner"], rec["sd_TRpartner"], _ = J(h, pa, nz, states[partner]["fin"]["target_risk"]["design"], seed)
        # other two systems in the same graph x cost cell (different noise level): additional public-only transplants
        others = [w for w in range(24) if w != wi and meta(w)[1:] == (graph, cost) and w != partner]
        rec["J_TRothercell_mean"] = float(np.mean([J(h, pa, nz, states[w]["fin"]["target_risk"]["design"], seed)[0] for w in others]))
        rec["J_obs24"], rec["sd_obs24"], _ = J(h, pa, nz, ["observe_all"] * 24, seed)
        rec["J_xneg24"], rec["sd_xneg24"], _ = J(h, pa, nz, ["set_x_neg"] * 24, seed)
        rec["J_xpos24"], _, _ = J(h, pa, nz, ["set_x_pos"] * 24, seed)
        rec["J_obs12_xneg6_xpos6"], _, _ = J(h, pa, nz, ["observe_all"] * 12 + ["set_x_neg"] * 6 + ["set_x_pos"] * 6, seed)
        rec["J_obs8_xneg8_xpos8"], _, _ = J(h, pa, nz, ["observe_all"] * 8 + ["set_x_neg"] * 8 + ["set_x_pos"] * 8, seed)
        d, jplan = plan_once(h, pa, nz, cost, seed)
        # evaluate the plan with an independent seed to avoid optimizer's-curse bias
        rec["J_planonce"], rec["sd_planonce"], _ = J(h, pa, nz, d, seed + 500000)
        rec["planonce_insample"] = jplan
        rec["planonce_design"] = " ".join(f"{a}:{d.count(a)}" for a in ACTIONS if d.count(a))
        rows.append(rec)
        print(wi, graph, cost, regime, "R8=%.4f realTR=%.4f J_TR=%.4f J_TRpartner=%.4f J_cov=%.4f realcov=%.4f J_plan=%.4f | %s" % (
            rec["R8"], rec["real_target_risk"], rec["J_target_risk"], rec["J_TRpartner"], rec["J_menu_coverage"], rec["real_menu_coverage"], rec["J_planonce"], rec["planonce_design"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "out_static_designs.csv", index=False)
    df.to_pickle(OUT + "static_designs.pkl")
