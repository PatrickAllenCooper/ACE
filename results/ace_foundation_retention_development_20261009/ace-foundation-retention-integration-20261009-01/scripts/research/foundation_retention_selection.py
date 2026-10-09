"""Finite training-only SCM selection. No fitting, model imports or file IO.

The caller owns provenance and supplies deterministic, fixed pointwise heads.
Row layouts are checked here; this cannot authenticate that rows are training
data. Exceptions invalidate the selection, never silently remove a candidate.
"""
from dataclasses import dataclass
import math

ORDER = ('retained', 'grammar', 'rbf', 'pfn')
MODES = ('raw', 'local', 'interval', 'combined')


def vector(values):
    values = tuple(float(x) for x in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('nonempty finite vector required')
    return values


def predict(head, x):
    x = vector(x)
    y = vector(head(x))
    if len(x) != len(y):
        raise ValueError('prediction length mismatch')
    return y


def mse(prediction, target):
    prediction, target = vector(prediction), vector(target)
    if len(prediction) != len(target):
        raise ValueError('target length mismatch')
    losses = vector((a-b)*(a-b) for a, b in zip(prediction, target))
    value = math.fsum(x / len(losses) for x in losses)
    if not math.isfinite(value):
        raise ValueError('nonfinite mean loss')
    return value


@dataclass(frozen=True)
class Row:
    x: float
    m: float
    y: float
    clamp: str


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self):
        vector((self.lower, self.upper))
        if self.lower > self.upper:
            raise ValueError('reversed interval')

    @classmethod
    def from_parents(cls, parents):
        parents = vector(parents)
        return cls(min(parents), max(parents))

    def contains(self, value):
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class GatedHead:
    retained: object
    updated: object
    interval: Interval

    def __call__(self, x):
        x = vector(x)
        inside = tuple(i for i, v in enumerate(x) if self.interval.contains(v))
        outside = tuple(i for i, v in enumerate(x) if not self.interval.contains(v))
        output = [None] * len(x)
        for indices, head in ((inside, self.updated), (outside, self.retained)):
            if indices:
                values = predict(head, tuple(x[i] for i in indices))
                for i, value in zip(indices, values):
                    output[i] = value
        return tuple(output)


@dataclass(frozen=True)
class Selection:
    mode: str
    names: tuple
    intervals: tuple
    local_errors: tuple  # (node, candidate name, MSE) for every candidate
    admitted: tuple  # two ordered candidate-name tuples
    composed_errors: tuple  # every admitted pair, MSE
    selected: tuple
    composed_mse: float
    calibration_inside: tuple  # eligible local points inside each fit interval


def training_layout(fit, calibration):
    fit, calibration = tuple(fit), tuple(calibration)
    for rows, layout in ((fit, ('none',)*6 + ('X',)*9 + ('M',)*9),
                         (calibration, ('none',)*2 + ('X',)*3 + ('M',)*3)):
        if len(rows) != len(layout) or tuple(r.clamp for r in rows) != layout:
            raise ValueError('fixed stratified fit/calibration layout required')
        for row in rows:
            vector((row.x, row.m, row.y))
    natural_fit = tuple(r for r in fit if r.clamp != 'M')
    natural_cal = tuple(r for r in calibration if r.clamp != 'M')
    intervals = (Interval.from_parents(r.x for r in natural_fit),
                 Interval.from_parents(r.m for r in fit))
    local = ((tuple(r.x for r in natural_cal), tuple(r.m for r in natural_cal)),
             (tuple(r.m for r in calibration), tuple(r.y for r in calibration)))
    root = (tuple(r.x for r in natural_cal), tuple(r.y for r in natural_cal))
    return intervals, local, root


def heads_for_mode(heads, names, mode, intervals):
    if mode not in MODES or tuple(names) not in (ORDER, ORDER[:-1]):
        raise ValueError('undeclared mode or candidate set')
    if len(heads) != 2 or len(intervals) != 2:
        raise ValueError('two node heads and intervals required')
    effective = []
    for node, interval in zip(heads, intervals):
        if set(node) != set(names) or not all(callable(node[n]) for n in names):
            raise ValueError('complete callable candidate dictionary required')
        effective.append({n: (GatedHead(node['retained'], node[n], interval)
                              if mode in ('interval', 'combined') and n != 'retained'
                              else node[n]) for n in names})
    return tuple(effective)


def select(fit, calibration, heads, mode='combined', include_pfn=True):
    if type(include_pfn) is not bool:
        raise ValueError('include_pfn must be boolean')
    names = ORDER if include_pfn else ORDER[:-1]
    intervals, local, root = training_layout(fit, calibration)
    effective = heads_for_mode(heads, names, mode, intervals)
    errors, admitted = [], []
    for node, candidates, (x, y) in zip(('M', 'Y'), effective, local):
        scores = {n: mse(predict(candidates[n], x), y) for n in names}
        errors.extend((node, n, scores[n]) for n in names)
        admitted.append(tuple(n for n in names if n == 'retained'
                              or mode in ('raw', 'interval')
                              or scores[n] < scores['retained']))
    x, y = root
    pairs = []
    for m_name in admitted[0]:
        parent = predict(effective[0][m_name], x)
        for y_name in admitted[1]:
            error = mse(predict(effective[1][y_name], parent), y)
            pairs.append(((m_name, y_name), error))
    def rank(item):
        (m_name, y_name), error = item
        return (error, int(m_name != 'retained') + int(y_name != 'retained'),
                names.index(m_name), names.index(y_name))
    chosen, error = min(pairs, key=rank)
    counts = tuple(sum(interval.contains(v) for v in x)
                   for interval, (x, _) in zip(intervals, local))
    return Selection(mode, names, intervals, tuple(errors), tuple(admitted),
                     tuple(pairs), chosen, error, counts)


def selected_heads(selection, heads):
    effective = heads_for_mode(heads, selection.names, selection.mode, selection.intervals)
    if len(selection.selected) != 2 or any(n not in selection.admitted[i]
                                          for i, n in enumerate(selection.selected)):
        raise ValueError('selected head outside admitted set')
    return tuple(effective[i][n] for i, n in enumerate(selection.selected))


def forecast(selection, heads, x):
    m, y = selected_heads(selection, heads)
    return predict(y, predict(m, x))
