"""Analytical DAG bounds and registered paired analysis; no empirical certificates."""
from __future__ import annotations
import math


def dag_bound(order, parents, local_bounds, lipschitz, clamped=()):
    """Path-sum bound in topological order. Caller must verify support assumptions."""
    bounds={};clamped=set(clamped)
    for node in order:
        if any(p not in bounds for p in parents[node]):raise ValueError('not topological')
        epsilon=local_bounds[node]
        if epsilon<0 or not math.isfinite(epsilon):raise ValueError('invalid local bound')
        terms=[]
        for p in parents[node]:
            L=lipschitz[node,p]
            if L<0 or not math.isfinite(L):raise ValueError('invalid Lipschitz constant')
            terms.append(L*bounds[p])
        bounds[node]=0.0 if node in clamped else epsilon+sum(terms)
    return bounds


def quantization_margin(value, levels):
    if len(levels)<2 or any(a>=b for a,b in zip(levels,levels[1:])):
        raise ValueError('need ordered distinct levels')
    return min(abs(value-(a+b)/2) for a,b in zip(levels,levels[1:]))


def holm(pvalues):
    """Adjusted p-values returned in original order (including nonrejections)."""
    if any(not 0<=p<=1 for p in pvalues):raise ValueError('invalid p value')
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);adjusted=[0.]*len(order);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,(len(order)-rank)*pvalues[i]));adjusted[i]=last
    return adjusted


def paired_log_ratio(delivery,control,floor=1e-12):
    """One paired value per independently parameterized system, NOT per row/init.

    Inputs here are paired errors for ONE history per system. For Stage B's
    two-history design use delivery_prospective_analysis, which averages log
    ratios (not errors) within each system. Two-sided paired t on log ratios;
    CI is marginal. Holm and the practical threshold are separate gates.
    """
    if len(delivery)!=len(control) or len(delivery)<2:raise ValueError('unpaired study')
    if any(not math.isfinite(v) or v<0 for v in [*delivery,*control]):raise ValueError('invalid error')
    import numpy as np
    from scipy import stats
    x=np.log(np.maximum(delivery,floor)/np.maximum(control,floor));n=len(x)
    mean=float(np.mean(x));sd=float(np.std(x,ddof=1));se=sd/math.sqrt(n)
    if se==0:p=1. if mean==0 else 0.;half=0.
    else:p=float(2*stats.t.sf(abs(mean/se),n-1));half=float(stats.t.ppf(.975,n-1)*se)
    return {'n_systems':n,'ratio':math.exp(mean),'ci95':[math.exp(mean-half),math.exp(mean+half)],
            'p_raw':p,'floor':floor,'log_mean':mean,'log_sd':sd}
