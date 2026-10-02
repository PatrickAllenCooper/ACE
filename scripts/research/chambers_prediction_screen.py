"""Execute the frozen white_64 blocked-angle CPU screen."""
import pathlib,hashlib,json,zipfile,io,time
import numpy as np
import pandas as pd
p=pathlib.Path('/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip')
b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9'
start=time.time()
with zipfile.ZipFile(io.BytesIO(b)) as z:
 d=pd.read_csv(z.open('lt_malus_v1/white_64.csv'))
a=d[['pol_1','pol_2']].to_numpy(float); y=d.vis_3.to_numpy(float)
assert np.isfinite(a).all() and np.isfinite(y).all() and ((a>=-90)&(a<90)).all()
regions=np.floor((a+90)/30).astype(int);test=(regions[:,0]+2*regions[:,1])%5==0;train=~test
r=np.deg2rad(a)
def basis(x):return np.column_stack([np.ones(len(x)),np.sin(2*x),np.cos(2*x),np.sin(4*x),np.cos(4*x)])
f,g=basis(r[:,0]),basis(r[:,1]);var=float(np.var(y[train]));assert var>0
features={'constant':f[:,:1],'additive_fourier':np.column_stack([f,g[:,1:]]),'tensor_fourier':np.einsum('ni,nj->nij',f,g).reshape(len(y),-1),'physical':np.column_stack([np.ones(len(y)),np.cos(r[:,0])**2*np.cos(r[:,1]-r[:,0])**2])}
results={}
for name,x in features.items():
 coef,_,rank,_=np.linalg.lstsq(x[train],y[train],rcond=None);pred=np.einsum("ij,j->i",x[test],coef)
 assert np.isfinite(coef).all() and np.isfinite(pred).all()
 results[name]={'rank':int(rank),'features':x.shape[1],'normalized_mse':float(np.mean((pred-y[test])**2)/var),'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest()}
out=pathlib.Path('results/chambers_prediction_screen_verified_20261002');out.mkdir(exist_ok=False)
result={'protocol_commit':'c9649ecb','condition':'white_64','train_rows':int(train.sum()),'test_rows':int(test.sum()),'training_variance':var,'results':results,'elapsed_seconds':time.time()-start,'new_physical_queries':0,'model_api_calls':0,'numpy':np.__version__,'pandas':pd.__version__}
data=(json.dumps(result,indent=2)+'\n').encode();(out/'result.json').write_bytes(data)
(out/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(data).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(b).hexdigest()},indent=2)+'\n')
print(json.dumps(result,indent=2))
