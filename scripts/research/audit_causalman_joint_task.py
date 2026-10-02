"""Static private-evaluator task audit; generates no simulator responses."""
import argparse, hashlib, json, pathlib, sys

def digest(b):
    return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=pathlib.Path,required=True)
    p.add_argument('--probe',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args()
    spec=json.loads(a.probe.read_text())
    for entry in spec['source_inventory']:
        assert digest((a.cache/entry['path']).read_bytes())==entry['sha256']
    sys.path.insert(0,str(a.cache.resolve()))
    from causalman.utils.serialization import load_pickle
    import networkx as nx
    import sympy as sp
    with (a.cache/spec['graph']).open('rb') as f:
        graph=load_pickle(f)
    targets=sorted(spec['levels'])
    public=set(spec['public_schema'])
    descendants={t:sorted(nx.descendants(graph,t)&public) for t in targets}
    shared=sorted(set.intersection(*(set(v) for v in descendants.values())))
    checks=[]
    for t in targets:
        for value in spec['levels'][t]:
            gates=[]
            for suffix in ['_LTL','_UTL']:
                threshold=graph.nodes[t+suffix]['source_distribution'].rhs
                expr=graph.nodes[t+suffix+'_MpGood']['term'].rhs
                gates.append(int(expr.subs({sp.Symbol(t):value,sp.Symbol(t+suffix):threshold})))
            checks.append({'target':t,'value':value,'lower_gate':gates[0],'upper_gate':gates[1]})
    assert all(x['lower_gate']==1 and x['upper_gate']==1 for x in checks)
    result={'upstream_revision':spec['upstream_revision'],'probe_sha256':digest(a.probe.read_bytes()),'public_descendants':descendants,'shared_public_descendants':shared,'shared_equations':{n:str(graph.nodes[n].get('term')) for n in shared},'threshold_checks':checks,'decision':'Probe menu does not establish nontrivial joint-action headroom. Retain as interface smoke only.','scope':'Private graph-based task construction audit; no learner sees equations; no policy result. Single actions versus observation may still change quality.','generated_rows':0,'model_calls':0}
    a.output.mkdir(parents=True,exist_ok=False)
    b=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    (a.output/'audit.json').write_bytes(b)
    (a.output/'complete.json').write_text(json.dumps({'audit_sha256':digest(b),'script_sha256':digest(pathlib.Path(__file__).read_bytes()),'generated_rows':0},indent=2)+'\n')
    print(result['decision'])

if __name__=='__main__':
    main()
