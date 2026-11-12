import json, collections
P="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/inputs/"
d=json.load(open(P+"prepared.json"))
prop=json.load(open(P+"proposal.json"))
tw={w["world_id"]:w for w in prop["training_worlds"]}
recs=d["training_plan"]["records"]
sched=d["training_plan"]["schedule"]
short=lambda c: json.loads(c).get("action_id","STOP")
rows=[]
for r in recs:
    wid=int(r["record_id"].split("/")[1]); si=int(r["record_id"].split("-")[-1])
    raw=json.loads(r["raw_record_utf8"])
    st=raw["envelope"]["state"]
    w=tw[wid]
    rows.append(dict(rid=r["record_id"], wid=wid, si=si, sup=short(r["supervised_command"]), teacher=w["training"]["snapshots"][si]["teacher_method"],
        behavior=w["training"]["behavior_method"], graph=w["graph_stratum"], cost=w["cost_stratum"], regime=w["regime"],
        remaining=st["remaining_units"], requests=st["requests"], terminal=st["terminal"], ncmd=len(r["commands"]),
        prompt_tokens=r["token_metadata"]["prompt_tokens"], strict=[(short(a),short(b)) for a,b in r["strict_pairs"]], ntied=len(r["tied_pairs"]),
        hist=[h["action_id"] for h in st["history"]]))
print("raw record top-level keys:", sorted(json.loads(recs[0]["raw_record_utf8"]).keys()))
print("SFT label counts (160):", dict(collections.Counter(x["sup"] for x in rows)))
print("by snapshot index:")
for si in range(5):
    sub=[x for x in rows if x["si"]==si]
    print("  si",si,"n",len(sub),"remaining:",dict(collections.Counter(x["remaining"] for x in sub)),"terminal:",dict(collections.Counter(x["terminal"] for x in sub)),"ncmd:",dict(collections.Counter(x["ncmd"] for x in sub)),"labels:",dict(collections.Counter(x["sup"] for x in sub)))
print("by teacher:")
for t in sorted({x["teacher"] for x in rows}):
    sub=[x for x in rows if x["teacher"]==t and x["si"]<4]
    print("  ",t,"n(nonterminal)",len(sub),dict(collections.Counter(x["sup"] for x in sub)))
print("by graph (nonterminal):")
for g in ("chain","fork","collider"):
    sub=[x for x in rows if x["graph"]==g and x["si"]<4]
    print("  ",g,len(sub),dict(collections.Counter(x["sup"] for x in sub)))
print("by cost (nonterminal):")
for c in ("uniform","heterogeneous"):
    sub=[x for x in rows if x["cost"]==c and x["si"]<4]
    print("  ",c,len(sub),dict(collections.Counter(x["sup"] for x in sub)))
print("prompt_tokens min/max over training records:", min(x["prompt_tokens"] for x in rows), max(x["prompt_tokens"] for x in rows))
print("requests by si:", {si:dict(collections.Counter(x["requests"] for x in rows if x["si"]==si)) for si in range(5)})
# DPO scheduled pairs
bykey={x["rid"]:x for x in rows}
ch=collections.Counter(); rj=collections.Counter(); pr=collections.Counter()
for s in sched:
    if s["stage"]=="dpo":
        a,b=bykey[s["record_id"]]["strict"][s["pair_index"]]
        ch[a]+=1; rj[b]+=1; pr[(a,b)]+=1
print("DPO scheduled steps:", sum(ch.values()))
print(" chosen counts  :", dict(ch.most_common()))
print(" rejected counts:", dict(rj.most_common()))
print(" net (chosen-rejected):", {k: ch[k]-rj[k] for k in sorted(set(ch)|set(rj))})
print(" top pairs:", pr.most_common(12))
# all strict pairs
ch2=collections.Counter(); rj2=collections.Counter()
for x in rows:
    for a,b in x["strict"]: ch2[a]+=1; rj2[b]+=1
print("ALL strict pairs:", sum(ch2.values()))
print(" chosen:", dict(ch2.most_common())); print(" rejected:", dict(rj2.most_common()))
print(" net:", {k: ch2[k]-rj2[k] for k in sorted(set(ch2)|set(rj2))})
print("SFT order (first 12 schedule entries):", [ (s["record_id"].split("/")[1][-3:]+"/"+s["record_id"][-1]) for s in sched[:12]])
print("DPO schedule by snapshot index:", dict(collections.Counter(bykey[s["record_id"]]["si"] for s in sched if s["stage"]=="dpo")))
json.dump(rows, open("train_rows.json","w"))
