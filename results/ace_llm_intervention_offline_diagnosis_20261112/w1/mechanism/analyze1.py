import pickle, collections, statistics as S, json
rows=pickle.load(open("decisions.pkl","rb"))
by=collections.defaultdict(list)
for r in rows: by[r["arm"]].append(r)
for arm,rs in by.items():
    rs.sort(key=lambda r:r["ordinal"])
    c=collections.Counter(r["completion"] for r in rs)
    pt=[r["prompt_tokens"] for r in rs]; uc=[r["user_chars"] for r in rs]
    print("=====",arm,"n",len(rs))
    print(" completions:",dict(c))
    print(" prompt_tokens min/median/mean/max:",min(pt),S.median(pt),round(S.mean(pt),1),max(pt))
    print(" user_chars   min/median/mean/max:",min(uc),S.median(uc),round(S.mean(uc),1),max(uc))
    print(" legal_commands counter:",dict(collections.Counter(r["legal"] for r in rs)))
    print(" n_new tokens counter:",dict(collections.Counter(r["n_new"] for r in rs)))
    print(" distinct task payloads:",len({r["task"] for r in rs}), "distinct task_sha256:",len({r["task_sha256"] for r in rs}))
    print(" distinct envelopes:",len({r["envelope_sha256"] for r in rs}))
    # episodes: start when requests==8
    eps=[]; 
    for r in rs:
        if r["requests"]==8: eps.append([])
        eps[-1].append(r)
    print(" episodes:",len(eps),"decisions per episode:",dict(collections.Counter(len(e) for e in eps)))
    print(" episode (graph,cost,noise) order:",[ (e[0]["graph"][:2],e[0]["cost"][:3],e[0]["noise"]) for e in eps])
    # tokens by history rows
    d=collections.defaultdict(list)
    for r in rs: d[r["requests"]].append(r["prompt_tokens"])
    print(" tokens by history rows (rows: min-max):", {k:(min(v),max(v)) for k,v in sorted(d.items()) if k in (8,9,12,16,20,24,28,31)})
    print(" elapsed seconds per decision mean:", round(S.mean(r["elapsed"] for r in rs),3))
# all arms share first envelope?
first={arm:by[arm][0]["envelope_sha256"] for arm in by}
print("first envelope per arm:",first)
