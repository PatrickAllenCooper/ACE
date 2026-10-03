import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
import hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BUNDLE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/host-mirror/results/slot0/ace_results/5f89033d')
REPO = Path('/Users/pat/code/ACE-Runner')
GRID = REPO / 'studies/2026-09-full-set/grid/tlam_mission1_full_grid.npz'
SOURCE = '0767be28fa2ff4ab72277349abe60359e8585d24'
def now(): return datetime.now(timezone.utc).isoformat()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, data):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

if '--worker' not in sys.argv:
    started = time.monotonic()
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    assert head == SOURCE, head
    files = [BUNDLE / n for n in ['meta.json', 'observations.ndjson','mlp.pt','report.yaml']]
    files += sorted((BUNDLE / 'mlps').glob('*.pt')) + [GRID, REPO/'ace/delivery.py', REPO/'ace/grid_eval.py', Path(__file__)]
    frozen = {'source':SOURCE,'bundle':str(BUNDLE),'grid':str(GRID),
              'hashes':{str(p):digest(p) for p in files}, 'frozen_at':now(),
              'epochs':30000,'inits':[0,1,2],'lr':0.002,'schedule':'constant',
              'cpu_threads':6,'elapsed_ceiling_seconds':7200,'simulator_calls':0,'llm_calls':0,
              'primary':'causal-chain exact-level error on full exposed grid',
              'scope':'single exposed seed development; no fresh confirmation or sampling claim'}
    assert not (OUT/'frozen.json').exists(), 'do not overwrite prior execution'
    write('frozen.json', frozen)
    with (OUT/'run.log').open('w') as log:
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                  cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=max(1,7200-(time.monotonic()-started)))
            status = 'complete' if proc.returncode == 0 else 'failed'
            code = proc.returncode
        except subprocess.TimeoutExpired:
            status, code = 'incomplete_time_ceiling', None
    write('execution.json', {'status':status,'exit_code':code,'ended_at':now(),
                            'elapsed_seconds':time.monotonic()-started})
    print(status, flush=True)
    sys.exit(0 if status == 'complete' else 1)

import numpy as np
import torch
from ace.result import ACEResult
from ace.delivery import refit_delivery
from ace.oracle import MLPSurrogate
from ace.grid_eval import GridTruth, score_bundle, chain_predict, flat_predict, score

torch.set_num_threads(6)
torch.set_num_interop_threads(1)
truth = GridTruth.load(GRID, target='engagement_rate', env_id='tlam_mission1')
assert truth.n_points == 390625
assert truth.sha256 == 'a62aeac6f1c1e5e1c835cf537b3de3bfd87b6cb53c582dfe6c32cc090c49e090'
result = ACEResult.load(BUNDLE)
assert result.observations.counts()['ace']['total'] == 4803
online = score_bundle(BUNDLE, truth)
write('online.json', {'metrics':online,'scored_at':now(),'grid_sha256':truth.sha256})
print('original scored', now(), online, flush=True)
results = []
for init in [0,1,2]:
    start = time.monotonic()
    print('fit_start',init,now(),flush=True)
    receipt = refit_delivery(result, OUT/f'init-{init}', custody_meta=BUNDLE/'meta.json', epochs=30000, seed=init)
    fit_seconds = time.monotonic()-start
    print('fit_complete',init,now(),fit_seconds,flush=True)
    states = torch.load(OUT/f'init-{init}'/'models.pt', map_location='cpu', weights_only=True)
    def predictor(state):
        model = MLPSurrogate.from_state_dict(state).eval()
        def predict(x):
            with torch.inference_mode():
                return np.concatenate([model(torch.tensor(x[i:i+8192],dtype=torch.float32)).numpy()
                                       for i in range(0,len(x),8192)])
        return predict
    levels = truth.levels()
    chain = score(chain_predict({n:predictor(s) for n,s in states.items() if n!='flat'},result.causal_dag,truth),truth.y,levels)
    flat = score(flat_predict(predictor(states['flat']),truth),truth.y,levels)
    row = {'init':init,'fit_seconds':fit_seconds,'total_seconds':time.monotonic()-start,
           'chain':chain,'flat':flat,'receipt':receipt,'scored_at':now()}
    results.append(row)
    write(f'init-{init}-scores.json', row)
    print('score_complete',init,now(),chain,flat,flush=True)
summary = {'scope':'exposed single-bundle development only','online':online,'inits':results,
           'median_chain_exact_error':float(np.median([1-r['chain']['exact'] for r in results])),
           'median_chain_mse':float(np.median([r['chain']['mse'] for r in results])),
           'median_flat_exact':float(np.median([r['flat']['exact'] for r in results])),
           'completed_at':now(),'torch_version':torch.__version__,'threads':torch.get_num_threads(),
           'gpu_used':False,'simulator_calls':0,'llm_calls':0}
write('summary.json',summary)
