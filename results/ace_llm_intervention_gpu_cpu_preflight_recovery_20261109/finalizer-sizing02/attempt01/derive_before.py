"""One new local metadata derivation; no producer/scientific functions executed."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
HELPER_SHA256 = '9ce042cbd024f92266090221e8f6c20486bc0811aaa511e9c7fa56a50e2e7d72'
BEFORE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-terminal-20261109-02/inventory_transport.json')
BEFORE_SHA256 = '2ca9bf1fd7f73a6964d9a10e3e4b239329c4bd4b53b2723a0131ded9a0395d66'

source = (HERE/'derive_inventory_timing.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == HELPER_SHA256
scope = {'__name__':'captured_metadata_derivation', '__file__':str(HERE/'derive_inventory_timing.py')}
exec(compile(source, scope['__file__'], 'exec'), scope)
raw = scope['capture'](BEFORE, BEFORE_SHA256)
summary, files, stderr = scope['observation'](raw)
out = Path(sys.argv[1]);out.mkdir(mode=0o700,exist_ok=False)
(out/'before.stderr').write_text(stderr)
result = dict(schema='ace-finalizer-sizing-before-derivation-v1',
    input=dict(path=str(BEFORE),bytes=len(raw),sha256=BEFORE_SHA256),
    helper_sha256=HELPER_SHA256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    observation=summary,after_observation_pending=True,actual_gpu_acceptance_pending=True,
    full_path_timing_supported=False,resource_admission_accepted=False,scientific_qualified=False)
(out/'derived.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ('observation_utc','transport_utc','elapsed_utc_bracket_seconds','full_files','full_bytes','roots')},indent=2))
print(json.dumps({k:{field:v[field] for field in ('files','bytes','first_counter','last_counter','wholly_contained_sample_seconds','intersecting_sample_seconds')} for k,v in summary['groups'].items()},indent=2))
