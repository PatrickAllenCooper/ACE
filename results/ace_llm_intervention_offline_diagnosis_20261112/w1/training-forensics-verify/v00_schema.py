"""Schema peek only: exposure.json, llm-prepared.json, plan.json, reference-scores.json, final-before-reload.json."""
import json, sys
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
PREP=TG+'/terminal01/remote/cpu/preparation/session/worker/artifacts'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
def show(o,ind=0,maxlist=2,depth=0,maxdepth=4):
    if isinstance(o,dict):
        for k,v in list(o.items())[:45]:
            if isinstance(v,(dict,list)):
                print(' '*ind+str(k)[:80]+': '+type(v).__name__+'('+str(len(v))+')')
                if depth<maxdepth: show(v,ind+2,maxlist,depth+1,maxdepth)
            else:
                print(' '*ind+str(k)[:80]+': '+repr(v)[:200])
        if len(o)>45: print(' '*ind+'... (%d keys)'%len(o))
    elif isinstance(o,list):
        for i,v in enumerate(o[:maxlist]):
            if isinstance(v,(dict,list)):
                print(' '*ind+'['+str(i)+'] '+type(v).__name__+'('+str(len(v))+')')
                if depth<maxdepth: show(v,ind+2,maxlist,depth+1,maxdepth)
            else: print(' '*ind+'['+str(i)+'] '+repr(v)[:200])
for name in sys.argv[1:]:
    p={'exposure':PREP+'/exposure.json','prepared':PREP+'/llm-prepared.json','plan':FIT+'/training/plan.json',
       'ref':FIT+'/training/reference-scores.json','final':FIT+'/final-before-reload.json','term':FIT+'/training/terminal.json',
       'sftreload':FIT+'/sft-reload-scores.json','finalreload':FIT+'/final-reload-scores.json','refsnap':FIT+'/training/reference-snapshot.json',
       'numplan':PREP+'/numerical-plan.json'}[name]
    print('=====',name,p)
    show(json.load(open(p)))
