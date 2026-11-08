"""Pure fixed-family paired-loss selector. No models, environment calls or certificate of IID provenance."""
from __future__ import annotations

import math
from itertools import product

NAMES = ('retained', 'grammar', 'rbf', 'pfn')
OBJECTIVES = ('M_local', 'Y_local', 'Y_composed')
BUDGETS = (8, 32, 128)
DELTA = 0.05
TESTS = 135  # 15 nonreference complete pairs x 3 objectives x 3 fixed looks
REFERENCE = ('retained', 'retained')


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('finite real scalar required')
    return float(x)


def bounded_squared_loss(prediction, observation, scale_squared):
    """Caller supplies a positive fit-only scale; overflow saturates the declared loss."""
    prediction, observation, scale_squared = map(_number, (prediction, observation, scale_squared))
    if scale_squared <= 0:
        raise ValueError('positive squared scale required')
    residual = abs(prediction - observation)
    if residual >= math.sqrt(scale_squared):
        return 1.0
    return (residual / math.sqrt(scale_squared)) ** 2


def select(losses, *, budget, rule='simultaneous', include_pfn=True):
    """Accept exactly one prefix, with every fixed candidate pair present.

    losses maps (M_name,Y_name) to objective -> sequence of losses in [0,1].
    Values must be computed with the frozen effective heads at the same paired
    observations. Local identity is inferred ONLY from retained head identifiers;
    matching observed samples is checked but cannot prove predictor identity.
    Runtime/source/independence/model-failure qualification belongs to a runner.
    """
    if type(budget) is not int or budget not in BUDGETS:
        raise ValueError('unregistered budget')
    if rule not in ('empirical', 'simultaneous') or type(include_pfn) is not bool:
        raise ValueError('unknown rule/family')
    names = NAMES if include_pfn else NAMES[:-1]
    pairs = list(product(names, repeat=2))
    if not isinstance(losses, dict) or set(losses) != set(pairs):
        raise ValueError('exact complete candidate family required; no silent dropping')
    checked = {}
    for pair in pairs:
        entry = losses[pair]
        if not isinstance(entry, dict) or set(entry) != set(OBJECTIVES):
            raise ValueError('all objectives required')
        checked[pair] = {}
        for obj in OBJECTIVES:
            seq = entry[obj]
            if not isinstance(seq, (list, tuple)) or len(seq) != budget:
                raise ValueError('exact prefix length required')
            values = [_number(x) for x in seq]
            if any(x < 0 or x > 1 for x in values):
                raise ValueError('loss outside [0,1]')
            checked[pair][obj] = values
    reference = checked[REFERENCE]
    # An objective's local predictions cannot depend on the other selected head.
    for obj, head in (('M_local', 0), ('Y_local', 1)):
        by_name = {}
        for pair in pairs:
            value = checked[pair][obj]
            if pair[head] in by_name and value != by_name[pair[head]]:
                raise ValueError('inconsistent fixed local head losses')
            by_name[pair[head]] = value
    epsilon = math.sqrt(2 * math.log(TESTS / DELTA) / budget)
    records = []
    admissible = []
    for pair in pairs:
        record = {'pair': list(pair), 'objectives': {}}
        for obj in OBJECTIVES:
            vals = checked[pair][obj]
            diffs = [x - y for x, y in zip(vals, reference[obj])]
            mean_diff = math.fsum(diffs) / budget
            identity = pair == REFERENCE or (obj == 'M_local' and pair[0] == 'retained') or (obj == 'Y_local' and pair[1] == 'retained')
            if identity and any(d != 0 for d in diffs):
                raise ValueError('claimed retained identity contradicts paired losses')
            upper = 0.0 if identity else mean_diff + (epsilon if rule == 'simultaneous' else 0.0)
            record['objectives'][obj] = {'mean_loss': math.fsum(vals) / budget, 'mean_difference': mean_diff, 'upper': upper, 'identity': identity}
        values = record['objectives']
        accept = pair != REFERENCE and values['M_local']['upper'] <= 0 and values['Y_local']['upper'] <= 0 and values['Y_composed']['upper'] < 0
        record['admissible'] = accept
        records.append(record)
        if accept:
            admissible.append((values['Y_composed']['mean_loss'], sum(x != 'retained' for x in pair), tuple(NAMES.index(x) for x in pair), pair))
    choice = min(admissible)[-1] if admissible else REFERENCE
    return {'selected': list(choice), 'fallback': not bool(admissible), 'fallback_reason': None if admissible else 'no_candidate_passed_all_objectives', 'budget': budget, 'rule': rule, 'include_pfn': include_pfn, 'delta_per_world': DELTA, 'simultaneous_tests': TESTS, 'epsilon': epsilon if rule == 'simultaneous' else 0.0, 'records': records, 'scope': 'bounded losses under declared IID laws conditional on fixed candidates; assumptions not authenticated by this module'}
