"""New fabricated metadata/loader fixtures only; no scientific source execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('history_adapter_under_test', HERE/'llm_intervention_cpu_history_admission.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)  # Only the newly authored stdlib adapter.

FAKE_SOURCE = b'''VERSION = "ace-supplied-plan-campaign-v1"
def _load_attempt(c, campaign_pin, path, pid, expected):
    calls.append((c, campaign_pin, path, pid, expected))
    return {"synthetic_gpu_passthrough": pid}, None
def _inventory(path):
    raise AssertionError("no historical tree scan permitted by this fixture")
def _closure(path, record):
    return None
def _need(ok, message):
    if not ok:
        raise ValueError(message)
def _keys(value, keys):
    return None
def _decode(raw):
    return None
def _source_pins(sources):
    return None
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


class Fixture:
    def __init__(self, directory):
        self.base = Path(directory).resolve()
        self.cpu = self.base/'cpu'
        self.root = self.cpu/'worker/artifacts/campaign'
        self.root.mkdir(parents=True)
        sources = []
        for wid in A.WORLD_IDS:
            # Identifier-only fabricated plans; no world, coefficient, outcome or response is drawn.
            plan = {'world': {'world_id': wid}, 'synthetic_metadata_only': True}
            sources.append({'plan': plan, 'plan_sha256': sha(A._canonical(plan)), 'strata': {}})
        self.c = dict(version=A.VERSION, scope='development', sources=sources,
                      policy_ids=list(A.CPU_IDS+A.GPU_IDS),
                      policy_phases={p: 'cpu' if p in A.CPU_IDS else 'gpu' for p in A.CPU_IDS+A.GPU_IDS},
                      services={p: {} for p in A.GPU_IDS})
        self.cpin = sha(A._canonical(self.c))
        self.sp = A._source_pins(self.c)
        self.bundle = raw(dict(campaign=self.c, campaign_sha256=self.cpin,
                               scope='development', source_plan_pins=self.sp))
        self.files = {'campaign.json': raw(self.c), 'source-plan-pins.json': raw(self.sp),
                      'start.json': raw(dict(utc='fabricated', campaign_sha256=self.cpin, retry_allowed=False))}
        self.payloads = {}
        for i, pid in enumerate(A.CPU_IDS):
            prefix = f'policy-{i:02d}/'
            arms = []
            for j, s in enumerate(sources):
                name = prefix+f'world-{j:02d}/arm-receipt.json'
                self.files[name] = raw(dict(plan_sha256=s['plan_sha256'], policy_id=pid, synthetic=True))
                arms.append(dict(world_id=s['plan']['world']['world_id'], status='completed',
                                 path=f'world-{j:02d}', receipt_sha256=sha(self.files[name]), reason=None))
            # Unselected opaque historical file exists in the accepted map but must never be opened by the adapter.
            self.payloads[prefix+'unused-payload.bin'] = b'fabricated-unused-history'
            evidence = {n[len(prefix):]: sha(b) for n, b in {**self.files, **self.payloads}.items() if n.startswith(prefix)}
            header = dict(version=A.VERSION, campaign_sha256=self.cpin, source_plan_pins=self.sp,
                          policy_id=pid, phase='cpu', arms=arms,
                          service=dict(status='not_started', file=None, sha256=None), errors=[],
                          evidence_sha256=evidence, retry_allowed=False, private_scores_computed=False)
            self.files[prefix+'attempt.json'] = raw(header)
            self.files[prefix+'attempt.candidate.json'] = raw(header)
        self.write()
        self.module = types.ModuleType('synthetic_campaign_fixture')
        self.module.__file__ = str(self.base/'synthetic_campaign_fixture.py')
        exec(compile(FAKE_SOURCE, self.module.__file__, 'exec', dont_inherit=True), self.module.__dict__)
        self.module.calls = []

    def write(self):
        for n, b in {**self.files, **self.payloads}.items():
            p = self.root/n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)

    def inputs(self):
        pins = {pid: sha(self.files[f'policy-{i:02d}/attempt.json']) for i, pid in enumerate(A.CPU_IDS)}
        a = dict(schema='ace-scientific-campaign-cpu-acceptance-v1', status='development_cpu_phase_accepted',
                 cpu_packet_sha256='1'*64, job_id='1',
                 allocation=dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal',
                                 cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'),
                 cpu_output=str(self.cpu), campaign_root=str(self.root),
                 cpu_output_files={A.BUNDLE_PATH: sha(self.bundle), 'worker/artifacts/cpu-phase/terminal.json': '2'*64},
                 cpu_output_excluded_directory=A.CAMPAIGN_SUFFIX,
                 campaign_files={n: sha(b) for n, b in {**self.files, **self.payloads}.items()},
                 bundle_path=A.BUNDLE_PATH, bundle_sha256=sha(self.bundle),
                 phase_terminal_path='worker/artifacts/cpu-phase/terminal.json', phase_terminal_sha256='2'*64,
                 attempt_pins=pins, scientific_qualified=False)
        ar = raw(a)
        rs = raw(dict(schema=A.SCHEMA, cpu_acceptance_sha256=sha(ar), campaign_root=str(self.root),
                      files={n: dict(sha256=sha(b), bytes=len(b), utf8=b.decode()) for n, b in self.files.items()}))
        return dict(acceptance_raw=ar, acceptance_sha256=sha(ar), bundle_raw=self.bundle, bundle_sha256=sha(self.bundle),
                    readset_raw=rs, readset_sha256=sha(rs), campaign_source_raw=FAKE_SOURCE,
                    campaign_source_sha256=sha(FAKE_SOURCE))

    def capture(self, **changes):
        kwargs = self.inputs(); kwargs.update(changes)
        # This is a test-only source-identity substitution for the entirely synthetic module above.
        with patch.object(A, 'CAMPAIGN_SOURCE_SHA256', sha(FAKE_SOURCE)):
            return A.capture_history(**kwargs)

    def call(self, history, pid=None, **changes):
        pid = pid or A.CPU_IDS[0]
        index = (A.CPU_IDS+A.GPU_IDS).index(pid) if pid in A.CPU_IDS+A.GPU_IDS else 0
        values = dict(c=self.c, campaign_pin=self.cpin,
                      path=self.root/f'policy-{index:02d}', pid=pid,
                      expected=history._attempt_pins[pid] if pid in A.CPU_IDS else '3'*64)
        values.update(changes)
        return self.module._load_attempt(**values)

    def change_header(self, edit):
        name = 'policy-00/attempt.json'; value = json.loads(self.files[name]); edit(value)
        self.files[name] = self.files['policy-00/attempt.candidate.json'] = raw(value)


