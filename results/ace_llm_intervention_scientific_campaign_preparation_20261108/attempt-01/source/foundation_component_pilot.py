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
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'

def reject_process_spawn(event, args):
    if event in ('subprocess.Popen', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.system'):
        raise RuntimeError('component worker permits one process only: '+event)

sys.addaudithook(reject_process_spawn)
import numpy as np

FAMILIES = ('linear', 'quadratic', 'tanh')
METHODS = ('polynomial', 'extra_trees', 'tabpfn_v2', 'grammar', 'language')
SEEDS = tuple(range(91000, 91006))
WEIGHT_PIN = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    tmp = Path(str(path)+'.pending')
    with tmp.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


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
        if not np.isfinite([mse, variance, mse/max(variance, 1e-12)]).all():
            raise ValueError('nonfinite metric or normalization')
        out[name] = {'mse': mse, 'nmse': mse/max(variance, 1e-12),
                     'training_variance': variance, 'floor_active': variance < 1e-12}
    return out, np.column_stack(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--language-model', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--mode', choices=('pilot','tabpfn-smoke','language-smoke'), required=True)
    parser.add_argument('--freeze-sha256',required=True)
    args = parser.parse_args()
    if os.environ.get('ACE_COMPONENT_SUPERVISED') != '1' or not (args.output/'plan.json').is_file():
        raise RuntimeError('required supervisor and prior durable plan absent')
    freeze_bytes=args.freeze.read_bytes()
    if hashlib.sha256(freeze_bytes).hexdigest()!=args.freeze_sha256:
        raise ValueError('worker captured freeze pin mismatch')
    freeze = json.loads(freeze_bytes)
    if args.mode not in freeze.get('allowed_modes',[]):
        raise ValueError('mode not in frozen stage')
    if freeze.get('stage') != ('pilot' if args.mode=='pilot' else 'compatibility'):
        raise ValueError('wrong frozen stage')
    required = {'numpy','torch','transformers','tabpfn','scikit-learn','scipy','safetensors','tokenizers','huggingface-hub'}
    if freeze.get('schema') != 'ace-component-freeze-v1' or not required.issubset(freeze['dependencies']):
        raise ValueError('required freeze/dependency schema')
    if digest(__file__) != freeze['source_sha256'] or digest(freeze['protocol_path']) != freeze['protocol_sha256']:
        raise ValueError('source/protocol mismatch')
    if digest(args.checkpoint) != WEIGHT_PIN or freeze['checkpoint_sha256'] != WEIGHT_PIN:
        raise ValueError('checkpoint mismatch')
    if freeze['tabpfn_revision'] != '4972a65a1b30806315c6f92499959ffbfc69a673':
        raise ValueError('TabPFN provenance revision')
    if freeze['language_revision'] != 'c1899de289a04d12100db370d81485cdf75e47ca':
        raise ValueError('language provenance revision')
    if args.language_model.name != freeze['language_revision']:
        raise ValueError('language cache revision directory mismatch')
    required_files = {'config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'}
    actual_files = {str(f.relative_to(args.language_model)) for f in args.language_model.rglob('*') if f.is_file()}
    if actual_files != required_files or set(freeze['language_files']) != actual_files:
        raise ValueError('incomplete or unexpected language load closure')
    for name, pin in freeze['language_files'].items():
        if digest(args.language_model/name) != pin:
            raise ValueError('language file mismatch: '+name)
    actual = {n: importlib.metadata.version(n) for n in required}
    if any(actual[n] != freeze['dependencies'][n] for n in required):
        raise ValueError('runtime mismatch')
    write(args.output/'preflight.json', {'actual_dependencies': actual, 'language_files':freeze['language_files'],
          'protocol_sha256':freeze['protocol_sha256'],'python':sys.version,'at_unix':time.time(),
          'configuration':freeze['configuration']})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start, cpu = time.monotonic(), time.process_time()
    write(args.output/'started.json', {'freeze_sha256': args.freeze_sha256, 'pid': os.getpid(),
          'at_unix': time.time(), 'planned_cells': 30, 'cpu_threads': 1, 'gpu': False})
    if args.mode != 'pilot':
        d=args.output/'smoke'; d.mkdir()
        write(d/(args.mode+'.started.json'), {'at_unix':time.time()})
        if args.mode == 'tabpfn-smoke':
            x=np.linspace(-1,1,16)[:,None]; y=2*x[:,0]+.3
            model=make_model('tabpfn_v2',args.checkpoint).fit(x,y)
            pred=np.asarray(model.predict(np.array([[-.25],[.25]])))
            if pred.shape != (2,) or not np.isfinite(pred).all():
                raise ValueError('smoke output invalid')
            result={'status':'complete','kind':'API compatibility only','predictions':pred.tolist()}
        else:
            proposer=LanguageProposer(str(args.language_model))
            raw,tokens=proposer('Return only a JSON object with keys M and Y, each set to linear.')
            result={'status':'complete','kind':'API compatibility only','raw':raw,'tokens':tokens}
        write(d/(args.mode+'.json'),result)
        write(args.output/'smoke.json',result)
        return
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
            write(d/(method+'.started.json'), {'seed':seed,'method':method,'at_unix':time.time()})
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
          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
          'resource_scope':'worker phase after preflight; terminal receipt covers whole process',
          'training_responses_total': 192, 'test_actions_total': 1536,
          'scope': 'synthetic development; fixed shared histories; no acquisition inference'})


if __name__ == '__main__':
    main()
