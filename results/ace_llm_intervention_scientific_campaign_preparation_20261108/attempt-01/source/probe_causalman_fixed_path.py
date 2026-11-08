"""Bounded engineering probe; no policy fitting or confirmation data."""
import argparse, hashlib, importlib.metadata, json, pathlib, platform, resource, sys, time

REV = '17529dad5ec8b8c691494c617b9af4533aa44bf8'
GRAPH = 'causalman/dataset_objects/causalman_micro/batch_data/batch_1_0_111_8_1/dag_level_1_0.pkl'
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=pathlib.Path,required=True); ap.add_argument('--output',type=pathlib.Path,required=True); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    tree=json.loads((a.cache/'upstream_tree.json').read_text()); inventory=[]
    for x in tree['tree']:
        if x['path']==GRAPH or (x['path'].startswith('causalman/') and x['path'].endswith('.py')):
            b=(a.cache/x['path']).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
            inventory.append({'path':x['path'],'sha256':sha(b)})
    sys.path.insert(0,str(a.cache.resolve()))
    from causalman.utils.serialization import load_pickle
    from causalman.utils.graph import sample_CausalGraph
    import numpy as np
    with (a.cache/GRAPH).open('rb') as f: g=load_pickle(f)
    def fingerprint(graph):
        return sha(json.dumps({'edges':sorted(graph.edges),'nodes':{n:{k:str(v) for k,v in d.items() if k != 'NodeModel'} for n,d in graph.nodes(data=True)}},sort_keys=True).encode())
    before=fingerprint(g); public=sorted(n for n,d in g.nodes(data=True) if d.get('Observable') is True)
    assert public
    # Monitoring thresholds establish numerical probe levels only, not actuator feasibility.
    names=['PF_M1_T1_sgrad','PF_M1_T2_sgrad']; levels={}
    for n in names:
        assert n in public
        levels[n]=[float(g.nodes[n+s]['source_distribution'].rhs) for s in ['_LTL','_UTL']]
    actions=[{}]+[{n:v} for n in names for v in levels[n]]+[{names[0]:v,names[1]:w} for v in levels[names[0]] for w in levels[names[1]]]
    runs=[]; start=time.time()
    for i,act in enumerate(actions):
        for k,v in act.items(): assert k in names and type(v) is float and np.isfinite(v) and v in levels[k]
        t=time.perf_counter(); df=sample_CausalGraph(g,sample_size=16,random_state=81000+i,interventions=act)
        assert len(df)==16 and set(df.columns)==set(g.nodes)
        assert all(np.allclose(df[n].to_numpy(dtype=float),v) for n,v in act.items())
        values=df[public].to_numpy(dtype=float); assert np.isfinite(values).all()
        assert fingerprint(g)==before
        runs.append({'action':act,'seed':81000+i,'generated_rows':len(df),'returned_rows':len(values),'seconds':time.perf_counter()-t,'public_response_sha256':sha(values.tobytes())})
    result={'upstream_revision':REV,'graph':GRAPH,'nodes':len(g),'edges':len(g.edges),'public_schema':public,'levels':levels,'runs':runs,'total_generated_rows':sum(r['generated_rows'] for r in runs),'start_unix':start,'end_unix':time.time(),'max_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_units':'bytes' if sys.platform=='darwin' else 'KiB','platform':platform.platform(),'python':sys.version,'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','sympy','networkx','matplotlib']},'source_inventory':inventory,'graph_equations_and_metadata_unchanged':True,'fingerprint_excludes':'NodeModel objects; canonical symbolic terms/distributions checked directly','scope':'one fixed external simulator path; engineering only; no fitted learner; fresh seed for every action'}
    b=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(); (a.output/'probe.json').write_bytes(b)
    (a.output/'complete.json').write_text(json.dumps({'probe_sha256':sha(b),'script_sha256':sha(pathlib.Path(__file__).read_bytes()),'model_calls':0,'gpu_seconds':0,'generated_rows':result['total_generated_rows']},indent=2)+'\n')
    print(json.dumps({'nodes':len(g),'public_columns':len(public),'rows':result['total_generated_rows'],'sampling_seconds':time.time()-start,'max_rss_native':result['max_rss_native']}))
if __name__=='__main__': main()
