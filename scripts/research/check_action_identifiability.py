"""Response-free finite-menu model-class separation diagnostic.

This checks representational separation, not statistical or causal identification.
No simulator observations, learned labels or private system parameters are used.
"""
import hashlib,itertools,json,pathlib
import numpy as np

def residual_fraction(design, candidate):
    coef=np.linalg.lstsq(design,candidate,rcond=None)[0]
    residual=candidate-np.einsum('ij,j->i',design,coef)
    denom=float(np.sum(candidate*candidate))
    return float(np.sum(residual*residual)/denom) if denom else 0.

def inspect(levels):
    points=np.array(list(itertools.product(levels,repeat=2)),float)
    x,y=points.T
    # Stronger comparator than the existing three-feature baseline: includes intercept.
    linear=np.column_stack([np.ones(len(x)),x,y,x*y])
    candidates={'quadratic_x':x*x,'quadratic_y':y*y,'cubic_x':x**3,
                'saturating_odd':np.tanh(1.7*x+.8*y),
                'radial':np.exp(-.5*(x*x+y*y))}
    return {'points':points.tolist(),'baseline_rank':int(np.linalg.matrix_rank(linear)),
            'residual_energy_fractions':{n:residual_fraction(linear,v) for n,v in candidates.items()}}

def main():
    results={n:inspect(v) for n,v in [('corners',[-2.,2.]),('three_level',[-2.,0.,2.]),('five_level',[-2.,-1.,0.,1.,2.])]}
    assert all(v<1e-20 for v in results['corners']['residual_energy_fractions'].values())
    assert results['three_level']['residual_energy_fractions']['saturating_odd']>1e-3
    assert results['three_level']['residual_energy_fractions']['cubic_x']<1e-20
    assert results['five_level']['residual_energy_fractions']['cubic_x']>1e-3
    p=pathlib.Path('results/action_identifiability_diagnostic_20261002');p.mkdir(exist_ok=False)
    r={'menus':results,'scope':'function-class checks on declared mathematical action menus; not empirical performance','simulator_queries':0,'model_calls':0,'checks':'passed'}
    b=(json.dumps(r,indent=2)+'\n').encode();(p/'result.json').write_bytes(b)
    (p/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(b).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({k:v['residual_energy_fractions'] for k,v in results.items()},indent=2))
if __name__=='__main__':main()
