"""Verifier step 11: two more variants of the fixed-sequence agreement (tie -> overall most frequent token in the other runs of the cell;
and a position-free 'always predict the cell's most frequent token' baseline)."""
import pandas as pd, numpy as np, pickle, collections
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
seqs = pickle.load(open(OUT + "out_seqs.pkl", "rb"))
W = meta.index.tolist()
def norm_sign(w, seq, xy_as_x=False):
    far = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
    near = "set_x_pos" if far == "set_x_neg" else "set_x_neg"
    return ["x_far" if a == far else "x_near" if a == near else ("x_far" if (xy_as_x and a == "set_xy" and far == "set_x_neg") else "x_near" if (xy_as_x and a == "set_xy") else a) for a in seq]
def loo(S, cells, mode="pos_tie_overall"):
    hit = tot = 0
    for w in S:
        others = [S[v] for v in S if v != w and cells[v] == cells[w]]
        overall = collections.Counter(t for o in others for t in o)
        rank = {t: (-c, t) for t, c in overall.items()}
        top = min(overall, key=lambda t: rank[t])
        for k, tok in enumerate(S[w]):
            if mode == "mix_only":
                pred = top
            else:
                col = [o[k] for o in others if len(o) > k]
                if not col: continue
                cnt = collections.Counter(col); best = max(cnt.values())
                cands = [t for t, c in cnt.items() if c == best]
                pred = min(cands, key=lambda t: rank[t])
            hit += tok == pred; tot += 1
    return hit / tot
rng = np.random.default_rng(11)
defs = {"graph x cost": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"]), "graph": lambda w: meta.loc[w, "graph_stratum"],
        "cost": lambda w: meta.loc[w, "cost_stratum"], "all": lambda w: 0}
for p in ("target_risk", "local_information"):
    for sn in ("raw", "sign", "sign+xy"):
        S = {w: (list(seqs[(w, p)]) if sn == "raw" else norm_sign(w, seqs[(w, p)], sn == "sign+xy")) for w in W}
        for cn, cf in defs.items():
            cells = {w: cf(w) for w in W}
            a = loo(S, cells)
            sh = np.mean([loo({w: list(rng.permutation(S[w])) for w in W}, cells) for _ in range(300)])
            mo = loo(S, cells, "mix_only")
            print(f"{p:18s} tokens={sn:8s} cell={cn:13s} agreement(tie->overall)={a:.3f} shuffle={sh:.3f} mix-only(always top token)={mo:.3f}")
