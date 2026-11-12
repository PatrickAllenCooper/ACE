import pickle, collections, statistics as S, numpy as np
for arm in ("frozen_llm","sft_llm","dpo_llm"):
    o=[x for x in pickle.load(open(f"planner_{arm}.pkl","rb")) if x["scores"]]
    best=collections.Counter(x["best"] for x in o)
    nonstop=[x for x in o if x["llm"]!="STOP" and x["llm"] in x["scores"]]
    agree=sum(1 for x in nonstop if x["llm"]==x["best"])
    reg=[max(x["scores"].values())-x["scores"][x["llm"]] for x in nonstop]
    bestv=[max(x["scores"].values()) for x in nonstop]
    chosenv=[x["scores"][x["llm"]] for x in nonstop]
    rk=collections.Counter(sorted(x["scores"].values(), reverse=True).index(x["scores"][x["llm"]])+1 for x in nonstop)
    # is LLM choice significantly worse than best? (gap > 2*sqrt(se_b^2+se_c^2))
    sig=sum(1 for x in nonstop if (max(x["scores"].values())-x["scores"][x["llm"]]) > 2*np.hypot(x["se"][x["best"]], x["se"][x["llm"]]))
    neg=sum(1 for x in nonstop if x["scores"][x["llm"]]<=0)
    print(f"== {arm}: states with planner scores={len(o)} ; planner-best action on these states: {dict(best.most_common())}")
    print(f"   LLM acquisitions={len(nonstop)} agree with planner-best={agree} ({100*agree/len(nonstop):.1f}%); rank of LLM action among feasible: {dict(sorted(rk.items()))}")
    print(f"   planner-estimated one-step risk reduction per cost: LLM action mean={S.mean(chosenv):.5f}, planner-best mean={S.mean(bestv):.5f}, ratio={S.mean(chosenv)/S.mean(bestv):.3f}; LLM action significantly below best (>2 MC SE) in {sig}/{len(nonstop)}; LLM action has estimated gain<=0 in {neg}")
    byk=collections.defaultdict(list)
    for x in nonstop: byk[min((x["requests"]-8)//6,3)].append((x["scores"][x["llm"]], max(x["scores"].values()), x["llm"]==x["best"]))
    print("   by acquisition-step block [0-5,6-11,12-17,18+]: (mean LLM gain, mean best gain, agree%)", {k:(round(S.mean(a for a,_,_ in v),5), round(S.mean(b for _,b,_ in v),5), round(100*S.mean(c for _,_,c in v),1)) for k,v in sorted(byk.items())})
    if arm=="dpo_llm":
        # first decision of each episode
        f=[x for x in o if x["requests"]==8]
        print("   first decision of each of 24 episodes: planner-best:", dict(collections.Counter(x["best"] for x in f)))
        # distinct best actions per episode
        eps=[]; 
        for x in o:
            if x["requests"]==8: eps.append([])
            eps[-1].append(x["best"])
        print("   distinct planner-best actions per DPO episode:", [len(set(e)) for e in eps])
