"""Stage B model kernels; only training rows enter fitting.

The rolling comparator invokes the archived ACE update function directly, with
its duplicate winner, persistent Adam, fast adaptation and masked replay intact.
Shared joint-root histories have no directly intervened nonroot, so its known
fast-adaptation masking limitation is inactive. No oracle constructor, policy,
query selector or model API is invoked.
"""
from collections import deque
import importlib.metadata
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def runtime(source):
    import torch
    source = str(Path(source).resolve())
    if 'ace.oracle' in sys.modules and not Path(sys.modules['ace.oracle'].__file__).is_relative_to(source):
        raise ValueError('a different oracle implementation is already imported')
    sys.path.insert(0, source)
    from ace.oracle import ACEOracle, MLPSurrogate
    return torch, ACEOracle, MLPSurrogate


def validate_input(data):
    import numpy as np
    order, parents, rows = data['order'], data['parents'], data['rows']
    if len(set(order)) != len(order) or set(parents) != set(order) or len(rows) < 50:
        raise ValueError('incomplete training graph/history')
    seen = set()
    for node in order:
        if len(set(parents[node])) != len(parents[node]) or not set(parents[node]) <= seen:
            raise ValueError('non-topological graph')
        seen.add(node)
    roots = [node for node in order if not parents[node]]
    if data['roots'] != roots or data['target'] not in order or data['target'] in roots:
        raise ValueError('root/target mismatch')
    if [r['query_index'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('query IDs must charge every response, including duplicate actions')
    for row in rows:
        if set(row['node_values']) != set(order) or set(row['clamps']) != set(roots):
            raise ValueError('only complete joint-root interventions are registered')
        if not np.isfinite(list(row['node_values'].values())).all():
            raise ValueError('nonfinite training response')
        if any(row['node_values'][root] != row['clamps'][root] or not -3 <= row['clamps'][root] <= 3 for root in roots):
            raise ValueError('root clamp mismatch')
    return roots


def normalizers(data):
    """A common training-only calibration prefix for every arm.

    The first50 observations are available before any algorithmic update.
    Replaying these observations is retrospective; it does not claim a strict
    streaming algorithm before the calibration prefix has been collected.
    Root ranges come from the specified action support; measured-parent ranges
    come only from that prefix and are never refreshed with test outcomes.
    """
    import numpy as np
    roots = validate_input(data)
    rows = data['rows'][:50]
    result = {}
    for node in data['order']:
        if not data['parents'][node]:
            continue
        lo, hi = [], []
        for parent in data['parents'][node]:
            values = np.array([row['node_values'][parent] for row in rows])
            a, b = (-3., 3.) if parent in roots else (float(values.min()), float(values.max()))
            lo.append(a); hi.append(b if b > a else a + 1.)
        result[node] = {'lo': lo, 'hi': hi}
    result['flat'] = {'lo': [-3.] * len(roots), 'hi': [3.] * len(roots)}
    return result


def initial_models(data, source, init, flat=False):
    torch, _, MLP = runtime(source)
    ranges = normalizers(data)
    nodes = ['flat'] if flat else [n for n in data['order'] if data['parents'][n]]
    models, optimizers = {}, {}
    for node in nodes:
        # One fixed documented scheme, shared across scratch/rolling SCM arms.
        offset = len(data['order']) if node == 'flat' else data['order'].index(node)
        torch.random.default_generator.manual_seed(init + offset)
        scale = ranges[node]
        model = MLP(len(scale['lo']), scale['lo'], scale['hi'])
        models[node] = model
        optimizers[node] = torch.optim.Adam(model.parameters(), lr=.002)
    return models, optimizers, ranges


def online_context(data):
    import torch
    def loss(pred, target):
        value = torch.nn.functional.mse_loss(pred, target)
        if not torch.isfinite(value):
            raise ValueError('nonfinite online training loss; retain failed attempt')
        return value
    return SimpleNamespace(replay_buffer=deque(maxlen=50), causal_dag=data['parents'],
                           device=torch.device('cpu'), loss_fn=loss)


def online_entry(data, row):
    return {'X': row['clamps'], 'node_values': row['node_values'], 'y': row['node_values'][data['target']],
            # Joint roots are clamped, and no modeled nonroot is intervened on.
            'intervened': None, '_query_index': row['query_index']}


def original_update(context, models, optimizers, entry, source, epochs=100):
    _, Oracle, _ = runtime(source)
    context.replay_buffer.append(entry)  # winner already present when original helper appends it again
    Oracle._train_node_mlps_on(context, models, optimizers, entry, epochs)


def fit(data, source, arm, init=0, epochs=None, online_tail=None):
    """Return states and accounting, without any evaluation responses.

    Epoch/tail overrides exist only for the explicitly labeled development
    timing pilot. The prospective batch must bind its fixed configuration and
    prohibit those overrides before release.
    """
    torch, _, _ = runtime(source)
    validate_input(data)
    if arm not in ('delivery', 'online', 'simpler', 'ablation'):
        raise ValueError('unregistered arm')
    if init not in (0, 1, 2) or (arm in ('online', 'ablation') and init != 0):
        raise ValueError('unregistered initialization')
    models, optimizers, ranges = initial_models(data, source, init, arm == 'simpler')
    costs = {}
    start, cpu0 = time.monotonic(), time.process_time()
    if arm == 'online':
        context = online_context(data)
        rows = data['rows']
        if online_tail is not None:
            if not 1 <= online_tail <= len(rows):
                raise ValueError('invalid development timing tail')
            for row in rows[max(0, len(rows) - online_tail - 50):len(rows) - online_tail]:
                context.replay_buffer.append(online_entry(data, row))
            rows = rows[-online_tail:]
        for row in rows:
            original_update(context, models, optimizers, online_entry(data, row), source, epochs=100)
        for node, model in models.items():
            costs[node] = {'updates': len(rows) * 120, 'eligible_rows': len(data['rows']),
                           'parameters': sum(p.numel() for p in model.parameters())}
    else:
        count = epochs if epochs is not None else 100 if arm == 'ablation' else 30000
        if type(count) is not int or count <= 0:
            raise ValueError('positive epoch count required')
        for node, model in models.items():
            inputs = data['roots'] if node == 'flat' else data['parents'][node]
            output = data['target'] if node == 'flat' else node
            x = torch.tensor([[r['node_values'][n] for n in inputs] for r in data['rows']], dtype=torch.float32)
            y = torch.tensor([r['node_values'][output] for r in data['rows']], dtype=torch.float32)
            head_cpu = time.process_time()
            for _ in range(count):
                optimizers[node].zero_grad()
                loss = torch.nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite scratch training loss; retain failed attempt')
                loss.backward(); optimizers[node].step()
            costs[node] = {'updates': count, 'eligible_rows': len(data['rows']),
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'cpu_seconds': time.process_time() - head_cpu}
    return {n: m.state_dict() for n, m in models.items()}, {
        'complete': True, 'arm': arm, 'init': init, 'heads': costs, 'normalizers': ranges,
        'cpu_seconds': time.process_time() - cpu0, 'wall_seconds': time.monotonic() - start,
        'unique_paid_rows': len(data['rows']), 'calibration_rows': 50,
        'new_queries_in_fit': 0, 'evaluation_responses_read': 0,
        'online_source': 'ACEOracle._train_node_mlps_on; append then duplicate winner; Adam persists',
        'development_epoch_override': epochs, 'development_online_tail': online_tail,
    }


def predict_graph(order, parents, models, clamps):
    """Never substitute measured test intermediates into a free-running chain."""
    import torch
    values = dict(clamps)
    for node in order:
        if node in clamps:
            continue
        if not parents[node]:
            raise ValueError('missing root clamp')
        x = torch.tensor([[values[p] for p in parents[node]]], dtype=torch.float32)
        with torch.no_grad():
            values[node] = float(models[node](x).item())
    return values


def dependencies():
    return {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'pandas', 'sympy', 'PyYAML')}
