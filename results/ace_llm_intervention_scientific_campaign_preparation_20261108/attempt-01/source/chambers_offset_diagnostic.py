"""Training-only fitting of the frozen offset diagnostic."""
import hashlib,io,json,pathlib,time,zipfile,itertools
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
start=time.time(); b=pathlib.Path('/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip').read_bytes()
assert hashlib.sha256(b).hexdigest()=='584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9'
with zipfile.ZipFile(io.BytesIO(b)) as z:d=pd.read_csv(z.open('lt_malus_v1/white_64.csv'))
a=d[['pol_1','pol_2']].to_numpy(float);y=d.vis_3.to_numpy(float);r=np.deg2rad(a)
regions=np.floor((a+90)/30).astype(int);test=(regions[:,0]+2*regions[:,1])%5==0;train=~test
sd=np.std(y[train]);assert sd>0 and np.isfinite(y).all()
def predict(p,x):return p[0]+p[1]*np.cos(x[:,0]+p[2])**2*np.cos(x[:,1]-x[:,0]+p[3])**2
fits=[]
for o1,o2 in itertools.product([-np.pi/2,-np.pi/4,0,np.pi/4],repeat=2):
 f=least_squares(lambda p:(predict(p,r[train])-y[train])/sd,[min(y[train]),np.ptp(y[train]),o1,o2],bounds=([-np.inf,0,-np.pi,-np.pi],[np.inf,np.inf,np.pi,np.pi]),max_nfev=500)
 fits.append({'parameters':f.x.tolist(),'training_nmse':float(np.mean(f.fun**2)),'success':bool(f.success),'nfev':f.nfev})
best=min(fits,key=lambda f:f['training_nmse']);pred=predict(best['parameters'],r[test]);assert np.isfinite(pred).all()
result={'protocol_commit':'ec2fbfad','scope':'post hoc development; reused white_64 split','train_rows':int(train.sum()),'test_rows':int(test.sum()),'selected':best,'test_nmse':float(np.mean(((pred-y[test])/sd)**2)),'fits':fits,'seconds':time.time()-start,'scipy':scipy.__version__,'new_queries':0,'model_calls':0}
p=pathlib.Path('results/chambers_offset_diagnostic_20261002');p.mkdir(exist_ok=False);raw=(json.dumps(result,indent=2)+'\n').encode();(p/'result.json').write_bytes(raw)
(p/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(raw).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='fits'},indent=2))
