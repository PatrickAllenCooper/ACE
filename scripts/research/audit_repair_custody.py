"""Audit archived observations and recorded acquisition block boundaries."""
import csv,hashlib,json,pathlib
import numpy as np
root=pathlib.Path('results/local_connected_transfer_heldout_dev_20260930');rows=[]
for seed in range(2000,2012):
 for n in [16,64]:
  for policy in ['factorial_hub_pair','risk_pair']:
   p=root/f'seed_{seed}'/f'source_n_{n}'/policy;c=json.loads((p/'complete.json').read_text());hashes={}
   for name in ['actions.csv','acquired.npz']:
    hashes[name]=hashlib.sha256((p/name).read_bytes()).hexdigest();assert hashes[name]==c[name+'_sha256']
   for file,key in [(p.parents[1]/'source_and_assay.npz','source_and_assay_sha256'),(p.parents[1]/'system.json','system_sha256')]:assert hashlib.sha256(file.read_bytes()).hexdigest()==c[key]
   acts=list(csv.DictReader((p/'actions.csv').open()));counts=[int(a['trajectories']) for a in acts]
   a=np.load(p/'acquired.npz');assert sum(counts)==len(a['values'])==44;assert counts==[4,8,8,8,8,8]
   assert np.isfinite(a['values']).all() and np.isfinite(a['motif_features']).all()
   ids=np.repeat(np.arange(len(acts)),counts);mask=a['natural_mask'];assert mask.shape==(44,10)
   per=[len(set(ids[mask[:,j]])) for j in range(10)]
   rows.append({'seed':seed,'source_n':n,'policy':policy,'hashes':hashes,'natural_blocks_per_motif':per,'rows':44,'blocks':6})
out=pathlib.Path('results/repair_custody_audit_20261002');out.mkdir(exist_ok=False)
r={'campaigns':len(rows),'recorded_rows':sum(x['rows'] for x in rows),'min_natural_blocks':min(min(x['natural_blocks_per_motif']) for x in rows),'max_natural_blocks':max(max(x['natural_blocks_per_motif']) for x in rows),'cells':rows,'new_queries':0,'decision':'Raw data and block boundaries available for debugging; exposed systems and six adaptively collected blocks do not supply fresh independent validation.'};b=(json.dumps(r,indent=2)+'\n').encode();(out/'audit.json').write_bytes(b);(out/'complete.json').write_text(json.dumps({'audit_sha256':hashlib.sha256(b).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print({k:v for k,v in r.items() if k!='cells'})
