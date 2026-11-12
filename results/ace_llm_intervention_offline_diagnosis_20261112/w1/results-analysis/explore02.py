import json
P='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque/cpu/cpu/worker/artifacts/campaign/private/'
def show(o,depth=0,maxd=4,maxk=40):
    if isinstance(o,dict):
        for k,v in list(o.items())[:maxk]:
            t=type(v).__name__
            s=''
            if isinstance(v,(str,int,float,bool)) or v is None:
                s=repr(v)[:160]
            elif isinstance(v,list):
                s='list len %d'%len(v)
                if v and not isinstance(v[0],(dict,list)):
                    s+=' '+repr(v[:10])[:200]
            elif isinstance(v,dict):
                s='dict keys %d'%len(v)
            print('  '*depth+str(k)+': '+t+' '+s)
            if depth<maxd and isinstance(v,(dict,list)):
                show(v,depth+1,maxd,maxk)
    elif isinstance(o,list):
        if o and isinstance(o[0],(dict,list)):
            print('  '*depth+'[0]:')
            show(o[0],depth+1,maxd,maxk)
r=json.load(open(P+'world-00/arm-03/cost-report.json'))
show(r,maxd=3)
print('======== world summary')
s=json.load(open(P+'world-00/summary.json'))
show(s,maxd=2,maxk=30)
