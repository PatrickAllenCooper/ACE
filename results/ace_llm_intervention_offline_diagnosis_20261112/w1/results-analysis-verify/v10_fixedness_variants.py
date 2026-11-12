"""Verifier step 10: sensitivity of the leave-one-out 'fixed sequence' agreement to definitional choices
(tie-breaking for the positional mode; what is shuffled; treatment of set_xy in sign normalization)."""
import pandas as pd, numpy as np, pickle, collections
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
seqs = pickle.load(open(OUT + "out_seqs.pkl", "rb"))
W = meta.index.tolist()
MENU = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos", "set_y_neg", "set_y_zero", "set_y_pos", "set_xy"]

def modal_seq(others, tie, rng=None):
    L = max(len(o) for o in others)
    out = []
    for k in range(L):
        col = [o[k] for o in others if len(o) > k]
        cnt = collections.Counter(col)
        best = max(cnt.values())
        cands = [t for t, c in cnt.items() if c == best]
        if tie == "alpha": out.append(sorted(cands)[0])
        elif tie == "first": out.append(next(t for t in col if t in cands))
        elif tie == "last": out.append(next(t for t in reversed(col) if t in cands))
        elif tie == "random": out.append(cands[rng.integers(len(cands))])
        elif tie == "frac": out.append({t: 1 / len(cands) for t in cands})
    return out

def loo(S, cells, tie, rng=None, shuffle=None):
    """shuffle None: no shuffling; 'all': every run's order permuted; 'heldout': only the held-out run permuted, modal from intact others;
       'others': modal built from shuffled others, held-out intact."""
    hit = tot = 0.0
    Sall = {w: list(rng.permutation(S[w])) for w in S} if shuffle in ("all", "others") else S
    for w in S:
        src = Sall if shuffle in ("all", "others") else S
        others = [src[v] for v in S if v != w and cells[v] == cells[w]]
        ms = modal_seq(others, tie, rng)
        test = S[w]
        if shuffle == "all": test = Sall[w]
        if shuffle == "heldout": test = list(rng.permutation(S[w]))
        for k, tok in enumerate(test):
            if k >= len(ms): continue
            m = ms[k]
            hit += (m.get(tok, 0.0) if isinstance(m, dict) else float(tok == m)); tot += 1
    return hit / tot

def norm_sign(w, seq, xy_as_x=False):
    far = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
    near = "set_x_pos" if far == "set_x_neg" else "set_x_neg"
    out = []
    for a in seq:
        if a == far: out.append("x_far")
        elif a == near: out.append("x_near")
        elif a == "set_xy" and xy_as_x: out.append("x_far" if far == "set_x_neg" else "x_near")
        else: out.append(a)
    return out

rng = np.random.default_rng(7)
cells_gc = {w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"]) for w in W}
cells_g = {w: meta.loc[w, "graph_stratum"] for w in W}
cells_c = {w: meta.loc[w, "cost_stratum"] for w in W}
rows = []
for p in ("target_risk", "local_information"):
    for cname, cells in (("graph x cost", cells_gc), ("graph", cells_g), ("cost", cells_c)):
        for sn in ("raw", "sign", "sign+xy"):
            S = {w: (list(seqs[(w, p)]) if sn == "raw" else norm_sign(w, seqs[(w, p)], sn == "sign+xy")) for w in W}
            for tie in ("alpha", "first", "last", "frac"):
                a = loo(S, cells, tie, rng)
                b_all = np.mean([loo(S, cells, tie, rng, "all") for _ in range(300)])
                b_held = np.mean([loo(S, cells, tie, rng, "heldout") for _ in range(300)])
                rows.append(dict(policy=p, cell=cname, tokens=sn, tie=tie, agreement=a, shuffle_all=b_all, shuffle_heldout_only=b_held))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R.round(3).to_string(index=False))
R.to_csv(OUT + "out_fixedness_variants.csv", index=False)
for p in ("target_risk", "local_information"):
    r = R[R.policy == p]
    print(p, "agreement range over variants:", round(r.agreement.min(), 3), "-", round(r.agreement.max(), 3),
          "| shuffle(all) range:", round(r.shuffle_all.min(), 3), "-", round(r.shuffle_all.max(), 3),
          "| shuffle(held-out only) range:", round(r.shuffle_heldout_only.min(), 3), "-", round(r.shuffle_heldout_only.max(), 3),
          "| excess over shuffle(all) range:", round((r.agreement - r.shuffle_all).min(), 3), "-", round((r.agreement - r.shuffle_all).max(), 3),
          "| excess over shuffle(held-out) range:", round((r.agreement - r.shuffle_heldout_only).min(), 3), "-", round((r.agreement - r.shuffle_heldout_only).max(), 3))
