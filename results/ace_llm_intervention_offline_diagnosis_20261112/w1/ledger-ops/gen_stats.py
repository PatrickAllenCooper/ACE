import json, os, re, statistics as st, sys, collections
base='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services'
out={}
for svc in ('frozen_llm','sft_llm','dpo_llm'):
    d=os.path.join(base,svc,'worker')
    fs=sorted(f for f in os.listdir(d) if re.fullmatch(r'generation-\d+\.json',f))
    pt=[];el=[];cu=[];comp=collections.Counter();newtok=[];legal=collections.Counter()
    for f in fs:
        g=json.load(open(os.path.join(d,f)))
        pt.append(g['prompt_tokens']); el.append(g['elapsed_seconds']); cu.append(g['cuda_interval_ms']/1000.0)
        comp[g['completion']]+=1
        newtok.append(len(g['sequence_ids'])-len(g['prompt_ids']))
        legal[g['legal_commands']]+=1
    def q(v,p): 
        s=sorted(v); return s[min(len(s)-1,int(p*len(s)))]
    out[svc]=dict(n=len(fs),prompt_tokens=dict(min=min(pt),p50=st.median(pt),mean=round(st.mean(pt),1),p90=q(pt,.9),max=max(pt)),
                  elapsed_s=dict(min=round(min(el),3),p50=round(st.median(el),3),mean=round(st.mean(el),3),p90=round(q(el,.9),3),max=round(max(el),3),sum=round(sum(el),1)),
                  cuda_s=dict(p50=round(st.median(cu),3),mean=round(st.mean(cu),3),sum=round(sum(cu),1)),
                  new_tokens=dict(mean=round(st.mean(newtok),2),sum=sum(newtok),dist=dict(collections.Counter(newtok))),
                  completions=dict(comp), legal_commands=dict(legal))
    # correlation prompt tokens vs cuda
    mx=st.mean(pt); my=st.mean(cu)
    cov=sum((a-mx)*(b-my) for a,b in zip(pt,cu)); vx=sum((a-mx)**2 for a in pt); vy=sum((b-my)**2 for b in cu)
    out[svc]['corr_prompt_tokens_vs_cuda']=round(cov/((vx*vy)**0.5),3) if vx>0 and vy>0 else None
print(json.dumps(out,indent=1))
json.dump(out,open('/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/ledger-ops/gen_stats.json','w'),indent=1)
