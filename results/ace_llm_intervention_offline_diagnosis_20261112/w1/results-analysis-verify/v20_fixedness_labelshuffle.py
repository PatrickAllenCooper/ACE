"""Verifier step 20: a third candidate 'shuffle baseline' for the fixed-sequence agreement: permute the CELL LABELS across systems
(keeping cell sizes) instead of permuting action order within runs. Tests whether the claimed baselines (0.38 TR, 0.47 LI) correspond to that."""
import pandas as pd, numpy as np, pickle, collections
meta = pd.read_csv("out_meta_mux.csv", index_col=0)
seqs = pickle.load(open("out_seqs.pkl", "rb"))
W = meta.index.tolist()
def norm_sign(w, seq, xy=False):
    far = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
    near = "set_x_pos" if far == "set_x_neg" else "set_x_neg"
    return ["x_far" if a == far else "x_near" if a == near else (("x_far" if far == "set_x_neg" else "x_near") if (xy and a == "set_xy") else a) for a in seq]
def loo(S, cells, tie="alpha"):
    hit = tot = 0
    for w in S:
        others = [S[v] for v in S if v != w and cells[v] == cells[w]]
        if not others: continue
        for k, tok in enumerate(S[w]):
            col = [o[k] for o in others if len(o) > k]
            if not col: continue
            cnt = collections.Counter(col); best = max(cnt.values())
            cands = sorted(t for t, c in cnt.items() if c == best)
            pred = cands[0] if tie == "alpha" else next(t for t in col if t in cands)
            hit += tok == pred; tot += 1
    return hit / tot
rng = np.random.default_rng(5)
celldefs = {"graph x cost": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"]), "graph": lambda w: meta.loc[w, "graph_stratum"], "cost": lambda w: meta.loc[w, "cost_stratum"]}
for p in ("target_risk", "local_information"):
    for sn in ("raw", "sign", "sign+xy"):
        S = {w: (list(seqs[(w, p)]) if sn == "raw" else norm_sign(w, seqs[(w, p)], sn == "sign+xy")) for w in W}
        for cn, cf in celldefs.items():
            for tie in ("alpha", "first"):
                cells = {w: cf(w) for w in W}
                a = loo(S, cells, tie)
                labs = [cells[w] for w in W]
                sh = []
                for _ in range(500):
                    perm = rng.permutation(len(W))
                    sh.append(loo(S, {w: labs[perm[i]] for i, w in enumerate(W)}, tie))
                sh = np.array(sh)
                print(f"{p:18s} tokens={sn:8s} cell={cn:13s} tie={tie:5s} agreement={a:.3f} cell-label-shuffle mean={sh.mean():.3f} p(>=obs)={(np.sum(sh >= a) + 1) / (len(sh) + 1):.3f}")
