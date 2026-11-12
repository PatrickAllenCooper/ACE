import pickle, collections, json
rows=pickle.load(open("decisions.pkl","rb"))
short=lambda c: json.loads(c).get("action_id","STOP")
for arm in ("sft_llm","frozen_llm"):
    rs=sorted([r for r in rows if r["arm"]==arm], key=lambda r:r["ordinal"])
    eps=[]
    for r in rs:
        if r["requests"]==8: eps.append([])
        eps[-1].append(r)
    print("=====",arm)
    for i,e in enumerate(eps):
        seq=[short(r["completion"]) for r in e]
        # run-length encode
        rle=[]; 
        for s in seq:
            if rle and rle[-1][0]==s: rle[-1][1]+=1
            else: rle.append([s,1])
        last=e[-1]
        print(f" ep{i:02d} {e[0]['graph'][:5]:5s} {e[0]['cost'][:3]} noise={e[0]['noise']:<4} n={len(e):2d} last(remaining={last['remaining']},requests={last['requests']}) :", " ".join(f"{s}x{n}" for s,n in rle))
    if arm=="sft_llm":
        stops=[r for r in rs if short(r["completion"])=="STOP"]
        print(" stop decisions: remaining_units counter:", dict(collections.Counter(r["remaining"] for r in stops)), " requests counter:", dict(collections.Counter(r["requests"] for r in stops)), " spent counter:", dict(collections.Counter(r["spent"] for r in stops)))
        # by decision index
        byidx=collections.defaultdict(collections.Counter)
        for e in eps:
            for k,r in enumerate(e): byidx[k][short(r["completion"])]+=1
        for k in sorted(byidx): print("  decision",k, dict(byidx[k]))
        bygraph=collections.defaultdict(collections.Counter)
        for r in rs: bygraph[(r["graph"],r["cost"])][short(r["completion"])]+=1
        for k,v in bygraph.items(): print("  ",k,dict(v))
