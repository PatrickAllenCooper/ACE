import json, pandas as pd, numpy as np
D='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/'
cells=pd.read_csv(D+'cells.csv')
curves=pd.read_csv(D+'curves.csv')
print(cells.dtypes)
print(cells.shape, curves.shape)
print(cells.nunique())
print(cells.groupby(['cost_stratum','graph_stratum','regime']).world_id.nunique())
print(cells[['world_id','cost_stratum','graph_stratum','regime']].drop_duplicates().to_string())
print(cells.termination_reason.value_counts())
print(curves.dtypes)
print(curves.action_id.value_counts(dropna=False))
print(curves.groupby('policy_id').size())
with open(D+'derivation.json') as f:
    der=json.load(f)
def show(o,depth=0,maxd=3,pref=''):
    if isinstance(o,dict):
        for k,v in list(o.items())[:40]:
            t=type(v).__name__
            s=''
            if isinstance(v,(str,int,float,bool)) or v is None:
                s=repr(v)[:200]
            elif isinstance(v,list):
                s='list len %d'%len(v)
            elif isinstance(v,dict):
                s='dict keys %d'%len(v)
            print('  '*depth+pref+str(k)+': '+t+' '+s)
            if depth<maxd and isinstance(v,(dict,list)):
                show(v,depth+1,maxd)
    elif isinstance(o,list):
        if o:
            print('  '*depth+'[0]: '+type(o[0]).__name__+' '+(repr(o[0])[:200] if not isinstance(o[0],(dict,list)) else ''))
            if depth<maxd and isinstance(o[0],(dict,list)):
                show(o[0],depth+1,maxd)
show(der,maxd=2)
