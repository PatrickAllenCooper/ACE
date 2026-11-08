"""Corrected published relative-angle control; reused development split."""
import pathlib,hashlib,zipfile,io,json
import numpy as np
import pandas as pd
b=pathlib.Path('/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip').read_bytes();assert hashlib.sha256(b).hexdigest()=='584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9'
with zipfile.ZipFile(io.BytesIO(b)) as z:d=pd.read_csv(z.open('lt_malus_v1/white_64.csv'))
a=d[['pol_1','pol_2']].to_numpy(float);y=d.vis_3.to_numpy(float);r=np.deg2rad(a);idx=np.floor((a+90)/30).astype(int);test=(idx[:,0]+2*idx[:,1])%5==0;train=~test
x=np.column_stack([np.ones(len(y)),np.cos(r[:,0]-r[:,1])**2]);coef,_,rank,_=np.linalg.lstsq(x[train],y[train],rcond=None);pred=np.einsum('ij,j->i',x,coef);assert np.isfinite(pred).all();var=np.var(y[train]);assert var>0
out=pathlib.Path('results/chambers_published_physics_screen_20261002');out.mkdir(exist_ok=False)
result={'protocol_commit':'ba4f94b9','upstream_revision':'9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6','train_rows':int(train.sum()),'test_rows':int(test.sum()),'coefficients':coef.tolist(),'rank':int(rank),'train_nmse':float(np.mean((pred[train]-y[train])**2)/var),'test_nmse':float(np.mean((pred[test]-y[test])**2)/var),'scope':'corrected control, reused development split, one condition','new_queries':0,'model_calls':0}
raw=(json.dumps(result,indent=2)+'\n').encode();(out/'result.json').write_bytes(raw);(out/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(raw).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'upstream_source_sha256':hashlib.sha256(pathlib.Path('results/chambers_published_model_source_20261002/light_tunnel_models.py').read_bytes()).hexdigest()},indent=2)+'\n');print(raw.decode())
