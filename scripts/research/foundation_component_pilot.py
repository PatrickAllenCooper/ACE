#!/usr/bin/env python3
"""Frozen, CPU-only component development screen; no accepted artifacts used."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'

import numpy as np

FAMILIES = ('linear', 'quadratic', 'tanh')
METHODS = ('polynomial', 'extra_trees', 'tabpfn_v2', 'grammar', 'language')
SEEDS = tuple(range(91000, 91006))
WEIGHT_PIN = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def basis(x, family):
    x = np.asarray(x).reshape(-1)
    if family == 'linear':
        return np.column_stack((np.ones_like(x), x))
    if family == 'quadratic':
        return np.column_stack((np.ones_like(x), x, x*x))
    if family == 'tanh':
        return np.column_stack((np.ones_like(x), np.tanh(x)))
    raise ValueError('family outside grammar')


class Grammar:
    def __init__(self, family=None):
        self.family = family

    def fit(self, x, y):
        def coefficients(xx, yy, family):
            b = basis(xx, family)
            return np.linalg.solve(b.T @ b + 1e-6*np.eye(b.shape[1]), b.T @ yy)
        if self.family is None:
            k = int(0.75 * len(y))
            self.validation = {
                f: float(np.mean((basis(x[k:], f) @ coefficients(x[:k], y[:k], f) - y[k:])**2))
                for f in FAMILIES
            }
            self.family = min(FAMILIES, key=self.validation.get)
        self.coef = coefficients(x, y, self.family)
        return self

    def predict(self, x):
        return basis(x, self.family) @ self.coef


def world(seed):
    """Evaluator owns truth; learner receives only returned training arrays."""
    rng = np.random.default_rng(seed)
    index = seed - SEEDS[0]
    families = (FAMILIES[index % 3], FAMILIES[(index+1) % 3])
    coeffs = []
    for family in families:
        c = [float(rng.uniform(-0.3, 0.3)), float(rng.uniform(0.6, 1.4))]
        if family == 'quadratic':
            c.append(float(rng.uniform(-0.7, 0.7)))
        coeffs.append(np.array(c))

    def truth(v, node):
        return basis(v, families[node]) @ coeffs[node]

    menu = np.array([-1., -0.5, 0., 0.5, 1.])
    x = np.concatenate((rng.uniform(-1, 1, 8), np.resize(menu, 12), rng.uniform(-1, 1, 12)))
    m = truth(x, 0) + rng.normal(0, 0.05, 32)
    m[20:] = np.resize(menu, 12)
    y = truth(m, 1) + rng.normal(0, 0.05, 32)
    clamp = np.array(['none']*8 + ['X']*12 + ['M']*12)
    train = np.column_stack((x, m, y))
    test_x = np.random.default_rng(seed+1000000).uniform(-1, 1, 256)
    test_m = truth(test_x, 0)
    test_y = truth(test_m, 1)
    return train, clamp, np.column_stack((test_x, test_m, test_y)), {
        'families': families, 'coefficients': [c.tolist() for c in coeffs],
        'noise_sd': 0.05, 'test_noise_sd': 0., 'seed': seed,
    }


def eligible(train, clamp):
    keep = clamp != 'M'
    return ((train[keep, 0:1].copy(), train[keep, 1].copy()),
            (train[:, 1:2].copy(), train[:, 2].copy()))


def prompt_for(data):
    pairs = {n: np.column_stack((x[:8, 0], y[:8])).round(6).tolist()
             for n, (x, y) in zip(('M', 'Y'), data)}
    return ('Choose one regression family per mechanism from these measured parent-child pairs. '
            'Each mechanism has a free intercept and fitted coefficients. Allowed names: '
            'linear (1,x), quadratic (1,x,x*x), tanh (1,tanh(x)). '
            'Return ONLY a JSON object with exactly keys M and Y and an allowed string value. '
            'No explanation or code. Data: ' + json.dumps(pairs))


def parse_proposal(raw):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'M', 'Y'}:
        raise ValueError('exact M,Y mapping required')
    if any(type(v) is not str or v not in FAMILIES for v in obj.values()):
        raise ValueError('invalid family')
    return obj


class LanguageProposer:
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32).to('cpu').eval()

    def __call__(self, prompt):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt')
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        tokens = result[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)


def make_model(method, checkpoint, family=None):
    if method in ('grammar', 'language'):
        return Grammar(family)
    if method == 'polynomial':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        return make_pipeline(PolynomialFeatures(3), Ridge(alpha=0.01))
    if method == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=128, min_samples_leaf=2, random_state=0, n_jobs=1)
    if method == 'tabpfn_v2':
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(model_path=str(checkpoint), device='cpu', n_estimators=1,
                              random_state=0, n_jobs=1)
    raise ValueError(method)


def evaluate(models, test, data):
    pred_m = np.asarray(models[0].predict(test[:, 0:1])).reshape(-1)
    pred_y_local = np.asarray(models[1].predict(test[:, 1:2])).reshape(-1)
    pred_y_composed = np.asarray(models[1].predict(pred_m[:, None])).reshape(-1)
    predictions = (pred_m, pred_y_local, pred_y_composed)
    targets = (test[:, 1], test[:, 2], test[:, 2])
    variances = [float(np.var(data[i][1])) for i in (0, 1, 1)]
    out = {}
    for name, pred, target, variance in zip(('M_local', 'Y_local', 'Y_composed'), predictions, targets, variances):
        if not np.isfinite(pred).all():
            raise ValueError('nonfinite predictions')
        mse = float(np.mean((pred-target)**2))
        out[name] = {'mse': mse, 'nmse': mse/max(variance, 1e-12),
                     'training_variance': variance, 'floor_active': variance < 1e-12}
    return out, np.column_stack(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--language-model', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    assert digest(__file__) == freeze['source_sha256']
    assert digest(args.checkpoint) == WEIGHT_PIN == freeze['checkpoint_sha256']
    for name, pin in freeze['language_files'].items():
        assert digest(args.language_model/name) == pin
    for name, version in freeze['dependencies'].items():
        assert importlib.metadata.version(name) == version, name
    args.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start, cpu = time.monotonic(), time.process_time()
    write(args.output/'started.json', {'freeze_sha256': digest(args.freeze), 'pid': os.getpid(),
          'at_unix': time.time(), 'planned_cells': 30, 'cpu_threads': 1, 'gpu': False})
    proposer, language_error = None, None
    try:
        proposer = LanguageProposer(str(args.language_model))
    except Exception:
        language_error = traceback.format_exc()
        (args.output/'language_load_failure.txt').write_text(language_error)
    rows = []
    for seed in SEEDS:
        train, clamp, test, truth = world(seed)
        d = args.output/str(seed)
        d.mkdir()
        np.savez(d/'arrays.npz', train=train, clamp=clamp, test=test)
        write(d/'private_truth.json', truth)
        data = eligible(train, clamp)
        for method in METHODS:
            if time.monotonic()-start > 1800:
                raise TimeoutError('pilot wall limit')
            before, pcpu = time.monotonic(), time.process_time()
            row = {'seed': seed, 'method': method, 'eligible_counts': [len(y) for _, y in data],
                   'training_responses': 32, 'status': 'failed'}
            try:
                proposal = None
                if method == 'language':
                    if proposer is None:
                        raise RuntimeError('language model failed to load; no silent substitution')
                    prompt = prompt_for(data)
                    (d/'prompt.txt').write_text(prompt)
                    raw, tokens = proposer(prompt)
                    (d/'raw_response.txt').write_text(raw)
                    row['generated_tokens'] = tokens
                    try:
                        proposal = parse_proposal(raw)
                        row['proposal_valid'] = True
                        row['proposal'] = proposal
                    except Exception as exc:
                        row['proposal_valid'] = False
                        row['parse_error'] = str(exc)
                        row['fallback'] = 'grammar'
                models = [make_model(method, args.checkpoint,
                          proposal[n] if proposal else None).fit(x, y)
                          for n, (x, y) in zip(('M', 'Y'), data)]
                row['selected_families'] = [getattr(m, 'family', None) for m in models]
                row['metrics'], predictions = evaluate(models, test, data)
                np.save(d/(method+'_predictions.npy'), predictions)
                row['status'] = 'complete'
            except Exception:
                row['error'] = traceback.format_exc()
            row['elapsed_s'] = time.monotonic()-before
            row['process_cpu_s'] = time.process_time()-pcpu
            write(d/(method+'.json'), row)
            rows.append(row)
            print(seed, method, row['status'], round(row['elapsed_s'], 3), flush=True)
    write(args.output/'complete.json', {'cells': rows, 'planned_cells': 30,
          'completed_cells': sum(r['status']=='complete' for r in rows),
          'elapsed_s': time.monotonic()-start, 'process_cpu_s': time.process_time()-cpu,
          'maxrss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'training_responses_total': 192, 'test_actions_total': 1536,
          'scope': 'synthetic development; fixed shared histories; no acquisition inference'})


if __name__ == '__main__':
    main()
