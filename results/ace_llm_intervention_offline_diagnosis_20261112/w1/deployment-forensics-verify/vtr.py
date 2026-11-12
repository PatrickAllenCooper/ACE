"""Verifier's own re-implementation of the one-step target-risk score for the public linear-Gaussian SCM.

Written from the mathematical definition (posterior variance of the composed noise-disabled target mean,
averaged over the public query law, before and after one fantasised response), not by importing the repository.
Two versions:
  * seq_scores: sequential, mirrors numpy RNG consumption so saved 32-draw values can be replayed exactly.
  * vec_scores: vectorised over draws with its own RNG, used for large-draw reference values.
"""
import sys, site
site.addsitedir('/Users/pat/Library/Python/3.9/lib/python/site-packages')
import hashlib
import numpy as np

def heads_from_state(state):
    graph = [(n['name'], list(n['parents'])) for n in state['graph']]
    summ = {s['node']: s for s in state['scm']}
    heads = {}
    for name, parents in graph:
        s = summ[name]
        assert s['terms'] == ['intercept'] + parents
        heads[name] = (np.array(s['parameter_mean'], float), np.array(s['parameter_covariance'], float), float(s['noise_variance']))
    return graph, heads

def target_variance(graph, heads, clamps, target):
    """Posterior variance of the composed structural mean of `target` under hard clamps (exact moments)."""
    idx = {name: i for i, (name, _) in enumerate(graph)}
    n = len(graph)
    m = np.zeros(n); C = np.zeros((n, n))
    for i, (name, parents) in enumerate(graph):
        if name in clamps:
            m[i] = clamps[name]; continue
        mu, S, _ = heads[name]
        P = [idx[p] for p in parents]
        phibar = np.r_[1.0, m[P]]
        Phi = np.zeros((len(P) + 1, len(P) + 1)); Phi[1:, 1:] = C[np.ix_(P, P)]
        m[i] = mu @ phibar
        if P:
            C[i, :i] = mu[1:] @ C[P, :i]
        C[:i, i] = C[i, :i]
        C[i, i] = np.sum(S * Phi) + phibar @ S @ phibar + mu @ Phi @ mu
    C = (C + C.T) / 2
    return max(0.0, float(C[idx[target], idx[target]]))

def risk(graph, heads, task):
    scale = task['scale']
    tot = 0.0
    for q, w in zip(task['queries'], task['weights']):
        tot += w * target_variance(graph, heads, {k: v for k, v in q['targets']}, task['target']) / scale ** 2
    return float(tot)

def seq_scores(envelope, seed, draws):
    """Mirror of the deployed estimator's sampling order; returns dict of per-action (score, after, se)."""
    state, task = envelope['state'], envelope['task']
    graph, heads = heads_from_state(state)
    before = risk(graph, heads, task)
    out = {}
    feas = sorted((a for a in state['menu'] if a['cost_units'] <= state['remaining_units']), key=lambda a: a['action_id'])
    for a in feas:
        aseed = int.from_bytes(hashlib.sha256(a['action_id'].encode()).digest()[:16], 'big')
        rng = np.random.default_rng(np.random.SeedSequence([seed, aseed]))
        clamps = {k: v for k, v in a['targets']}
        rs = []
        for _ in range(draws):
            vals = dict(clamps)
            for name, parents in graph:
                if name not in vals:
                    mu, S, nv = heads[name]
                    coef = rng.multivariate_normal(mu, S, check_valid='raise')
                    phi = np.array([1.0] + [vals[p] for p in parents])
                    vals[name] = float(phi @ coef + rng.normal(0, np.sqrt(nv)))
            new = dict(heads)
            for name, parents in graph:
                if name in clamps: continue
                mu, S, nv = heads[name]
                phi = np.array([1.0] + [vals[p] for p in parents])
                pr = S @ phi
                den = nv + phi @ pr
                mu2 = mu + pr * ((vals[name] - phi @ mu) / den)
                S2 = S - np.outer(pr, pr) / den
                S2 = (S2 + S2.T) / 2
                new[name] = (mu2, S2, nv)
            rs.append(risk(graph, new, task))
        after = float(np.mean(rs))
        out[a['action_id']] = ((before - after) / a['cost_units'], after, float(np.std(rs, ddof=1) / np.sqrt(draws) / a['cost_units']))
    return before, out

# ---------------- vectorised version ----------------
def _bt_target_variance(graph, bheads, clamps, target, B):
    idx = {name: i for i, (name, _) in enumerate(graph)}
    n = len(graph)
    m = np.zeros((B, n)); C = np.zeros((B, n, n))
    for i, (name, parents) in enumerate(graph):
        if name in clamps:
            m[:, i] = clamps[name]; continue
        mu, S = bheads[name]            # (B,d), (B,d,d)
        P = [idx[p] for p in parents]
        d = len(P) + 1
        phibar = np.concatenate([np.ones((B, 1)), m[:, P]], axis=1)
        Phi = np.zeros((B, d, d))
        if P:
            Phi[:, 1:, 1:] = C[:, P][:, :, P]
        m[:, i] = np.einsum('bd,bd->b', mu, phibar)
        if P and i > 0:
            C[:, i, :i] = np.einsum('bp,bpj->bj', mu[:, 1:], C[:, P, :i])
            C[:, :i, i] = C[:, i, :i]
        C[:, i, i] = (np.einsum('bij,bij->b', S, Phi) + np.einsum('bi,bij,bj->b', phibar, S, phibar)
                      + np.einsum('bi,bij,bj->b', mu, Phi, mu))
    return np.maximum(0.0, C[:, idx[target], idx[target]])

def vec_scores(envelope, draws, rng, antithetic=False):
    """Large-draw reference. Returns before, {action: (score, after, se)} for every menu action that is affordable."""
    state, task = envelope['state'], envelope['task']
    graph, heads = heads_from_state(state)
    before = risk(graph, heads, task)
    out = {}
    feas = sorted((a for a in state['menu'] if a['cost_units'] <= state['remaining_units']), key=lambda a: a['action_id'])
    B = draws
    for a in feas:
        clamps = {k: v for k, v in a['targets']}
        vals = {k: np.full(B, float(v)) for k, v in clamps.items()}
        bheads = {}
        for name, parents in graph:
            mu, S, nv = heads[name]
            d = len(mu)
            if name in clamps:
                bheads[name] = (np.broadcast_to(mu, (B, d)).copy(), np.broadcast_to(S, (B, d, d)).copy())
                continue
            # sample coefficients with a symmetric square root of S
            w, V = np.linalg.eigh(S)
            root = V * np.sqrt(np.clip(w, 0, None))
            z = rng.standard_normal((B, d))
            coef = mu + z @ root.T
            phi = np.stack([np.ones(B)] + [vals[p] for p in parents], axis=1)
            y = np.einsum('bd,bd->b', phi, coef) + rng.standard_normal(B) * np.sqrt(nv)
            vals[name] = y
            pr = phi @ S.T                      # (B,d)   S symmetric
            den = nv + np.einsum('bd,bd->b', phi, pr)
            mu2 = mu + pr * ((y - phi @ mu) / den)[:, None]
            S2 = S - np.einsum('bi,bj->bij', pr, pr) / den[:, None, None]
            S2 = (S2 + np.swapaxes(S2, 1, 2)) / 2
            bheads[name] = (mu2, S2)
        r = np.zeros(B)
        for q, w_ in zip(task['queries'], task['weights']):
            r += w_ * _bt_target_variance(graph, bheads, {k: v for k, v in q['targets']}, task['target'], B) / task['scale'] ** 2
        after = float(r.mean())
        out[a['action_id']] = ((before - after) / a['cost_units'], after, float(r.std(ddof=1) / np.sqrt(B) / a['cost_units']))
    return before, out
