"""Training-only finite mixture selection; no model loading, environments or file IO.

A caller must pass calibration rows, never private evaluation outcomes. This
module can enforce row/intervention semantics, not establish data provenance.
"""
from dataclasses import dataclass
import math
from typing import Sequence

WEIGHTS=(0.,.25,.5,.75,1.)
FIT_INDICES=tuple(range(6))+tuple(range(8,17))+tuple(range(20,29))
CALIBRATION_INDICES=(6,7,17,18,19,29,30,31)

@dataclass(frozen=True)
class Row:
    x: float
    m: float
    y: float
    clamp: str

@dataclass(frozen=True)
class Choice:
    weights: tuple[float,...]
    mse: float
    candidates: tuple[tuple[tuple[float,...],float],...]


def finite(values):
    result=tuple(float(v) for v in values)
    if not result or not all(math.isfinite(v) for v in result):
        raise ValueError('nonempty finite vector required')
    return result


def predict(model, x):
    result=finite(model(tuple(x)))
    if len(result)!=len(x):raise ValueError('prediction length mismatch')
    return result


def mse(pred, y):
    pred=finite(pred); y=finite(y)
    if len(pred)!=len(y):raise ValueError('target length mismatch')
    errors=tuple((a-b)*(a-b) for a,b in zip(pred,y))
    if not all(math.isfinite(v) for v in errors):raise ValueError('nonfinite squared error')
    value=math.fsum(v/len(y) for v in errors)
    if not math.isfinite(value):raise ValueError('nonfinite MSE')
    return value


def blend(g,f,weight):
    if len(g)!=len(f):raise ValueError('expert length mismatch')
    return finite((1-weight)*a+weight*b for a,b in zip(g,f))


def split_history(rows: Sequence[Row]):
    if len(rows)!=32:raise ValueError('fixed32response history required')
    expected=('none',)*8+('X',)*12+('M',)*12
    if tuple(r.clamp for r in rows)!=expected:raise ValueError('history intervention layout mismatch')
    for r in rows:finite((r.x,r.m,r.y))
    return tuple(rows[i] for i in FIT_INDICES),tuple(rows[i] for i in CALIBRATION_INDICES)


def root_calibration(rows: Sequence[Row]):
    if len(rows)!=8 or tuple(r.clamp for r in rows)!=('none','none','X','X','X','M','M','M'):
        raise ValueError('fixed stratified calibration layout required')
    for r in rows:finite((r.x,r.m,r.y))
    selected=tuple(r for r in rows if r.clamp!='M')
    return tuple(r.x for r in selected),tuple(r.y for r in selected)


def terminal_choice(calibration, grammar_m, grammar_y, pfn_m, pfn_y):
    x,y=root_calibration(calibration)
    g=predict(grammar_y,predict(grammar_m,x));f=predict(pfn_y,predict(pfn_m,x))
    candidates=tuple(((w,),mse(blend(g,f,w),y)) for w in WEIGHTS)
    weights,error=min(candidates,key=lambda item:item[1])
    return Choice(weights,error,candidates)


def mechanism_choice(calibration, grammar_m, grammar_y, pfn_m, pfn_y):
    x,y=root_calibration(calibration)
    gm=predict(grammar_m,x);fm=predict(pfn_m,x)
    candidates=[]
    for wm in WEIGHTS:
        parents=blend(gm,fm,wm)
        gy=predict(grammar_y,parents);fy=predict(pfn_y,parents)
        for wy in WEIGHTS:
            candidates.append(((wm,wy),mse(blend(gy,fy,wy),y)))
    weights,error=min(candidates,key=lambda item:item[1])
    return Choice(weights,error,tuple(candidates))


def terminal_forecast(x,grammar_m,grammar_y,pfn_m,pfn_y,weight):
    if weight not in WEIGHTS:raise ValueError('weight outside frozen grid')
    x=finite(x)
    return blend(predict(grammar_y,predict(grammar_m,x)),predict(pfn_y,predict(pfn_m,x)),weight)


def mechanism_forecast(x,grammar_m,grammar_y,pfn_m,pfn_y,weights):
    if len(weights)!=2 or any(w not in WEIGHTS for w in weights):raise ValueError('weights outside frozen grid')
    x=finite(x);parents=blend(predict(grammar_m,x),predict(pfn_m,x),weights[0])
    return blend(predict(grammar_y,parents),predict(pfn_y,parents),weights[1])
