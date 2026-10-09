"""Compile a fit-only scalar grid into fixed numerical data. No model imports or IO.

The caller must authenticate fit/teacher/phase provenance. Trusted-runtime value
immutability is not adversarial security. Compiled predictions are NEW methods.
"""
from dataclasses import dataclass
from bisect import bisect_left
import hashlib
import json
import math


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError('finite scalar required')
    try:
        value = float(x)
    except (OverflowError, ValueError):
        raise ValueError('finite scalar required') from None
    if not math.isfinite(value):
        raise ValueError('finite scalar required')
    return value


def vector(values):
    out = tuple(finite(x) for x in values)
    if not out:
        raise ValueError('nonempty scalar vector required')
    return out


def grid(parents):
    parents = vector(parents)
    a, b = min(parents), max(parents)
    if a == b:
        return (a,)
    span = finite(b-a)
    points = (a,) + tuple(finite(a+span*(k/128)) for k in range(1,128)) + (b,)
    if any(y <= x for x, y in zip(points, points[1:])):
        raise ValueError('collapsed grid points')
    return points


@dataclass(frozen=True)
class RetainedParameters:
    family: str
    coefficients: tuple

    def __post_init__(self):
        sizes = {'linear': 2, 'quadratic': 3, 'tanh': 2}
        if self.family not in sizes:
            raise ValueError('unknown retained grammar')
        coef = vector(self.coefficients)
        if len(coef) != sizes[self.family]:
            raise ValueError('coefficient count')
        object.__setattr__(self, 'coefficients', coef)

    def scalar(self, x):
        x = finite(x)
        c = self.coefficients
        if self.family == 'tanh':
            return finite(c[0] + c[1]*math.tanh(x))
        if self.family == 'linear':
            return finite(c[0] + c[1]*x)
        return finite(c[0] + c[1]*x + c[2]*x*x)

    def __call__(self, values):
        return tuple(self.scalar(x) for x in vector(values))


@dataclass(frozen=True)
class FixedTable:
    knots: tuple
    values: tuple
    retained: RetainedParameters
    provider: str
    fit_parent_sha256: str

    def __post_init__(self):
        knots, values = vector(self.knots), vector(self.values)
        if len(knots) != len(values) or len(knots) not in (1,129):
            raise ValueError('fixed grid/output shape required')
        if knots != grid((knots[0],knots[-1])):
            raise ValueError('fixed grid policy mismatch')
        if type(self.retained) is not RetainedParameters:
            raise ValueError('copied retained parameters required')
        if self.provider not in ('grammar','rbf','pfn'):
            raise ValueError('unknown provider')
        if not isinstance(self.fit_parent_sha256,str) or len(self.fit_parent_sha256)!=64 or any(x not in '0123456789abcdef' for x in self.fit_parent_sha256):
            raise ValueError('fit parent digest required')
        object.__setattr__(self,'knots',knots)
        object.__setattr__(self,'values',values)

    def scalar(self,x):
        x=finite(x)
        if x<self.knots[0] or x>self.knots[-1]:
            return self.retained.scalar(x)
        i=bisect_left(self.knots,x)
        if self.knots[i]==x:
            return self.values[i]
        left,right=self.knots[i-1],self.knots[i]
        w=(x-left)/(right-left)
        return finite((1-w)*self.values[i-1]+w*self.values[i])

    def __call__(self,values):
        return tuple(self.scalar(x) for x in vector(values))

    def payload(self):
        return {'schema':'ace-fixed-scalar-table-v1','knots':[x.hex() for x in self.knots], 'values':[x.hex() for x in self.values], 'retained':{'family':self.retained.family,'coefficients':[x.hex() for x in self.retained.coefficients]}, 'provider':self.provider,'fit_parent_sha256':self.fit_parent_sha256}

    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compile_table(teacher, parents, retained, provider):
    """Exactly one fixed-grid teacher query; no teacher is held by returned object.

    Only call BEFORE validation. The caller records inference/resources/failures.
    This interface alone cannot prove temporal/source provenance.
    """
    parents=vector(parents)
    points=grid(parents)
    if type(retained) is not RetainedParameters or provider not in ('grammar','rbf','pfn') or not callable(teacher):
        raise ValueError('invalid teacher/retained/provider')
    parent_digest=hashlib.sha256(json.dumps([x.hex() for x in parents],separators=(',',':')).encode()).hexdigest()
    outputs=vector(teacher(points))
    return FixedTable(points,outputs,retained,provider,parent_digest)


def restore(payload, expected_sha256):
    """Restore data under an independent pin; the pin's trust belongs to caller."""
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=expected_sha256:
        raise ValueError('payload pin mismatch')
    if set(payload)!= {'schema','knots','values','retained','provider','fit_parent_sha256'} or payload['schema']!='ace-fixed-scalar-table-v1':
        raise ValueError('schema mismatch')
    r=payload['retained']
    if set(r)!= {'family','coefficients'}:
        raise ValueError('retained schema mismatch')
    table = FixedTable(tuple(float.fromhex(x) for x in payload['knots']), tuple(float.fromhex(x) for x in payload['values']), RetainedParameters(r['family'],tuple(float.fromhex(x) for x in r['coefficients'])),payload['provider'],payload['fit_parent_sha256'])
    if table.payload() != payload or table.digest() != digest:
        raise ValueError('noncanonical payload reconstruction')
    return table
