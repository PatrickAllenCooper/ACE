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


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ace-history-fabricated-')
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_exact_roster_cpu_semantics_small_reads_and_gpu_passthrough(self):
        f = self.f; h = f.capture(); original = f.module._load_attempt; inventory = f.module._inventory
        self.assertEqual(len(h._files), 133)
        read_names = []
        read = A._Live.read
        def recording(live, name):
            read_names.append(name)
            return read(live, name)
        with patch.object(A._Live, 'read', recording), h.install(f.module):
            read_names.clear()
            r, closure = f.call(h)
            self.assertIsNone(closure); self.assertEqual(len(r['arms']), 24)
            for name in f.files:
                if name.startswith('policy-00/'):
                    self.assertEqual(read_names.count(name), 1)
            self.assertFalse(any(n.endswith('.bin') for n in read_names))
            self.assertEqual(f.module.calls, [])
            r['arms'][0]['status'] = 'failed'
            self.assertEqual(f.call(h)[0]['arms'][0]['status'], 'completed')
            (f.root/'policy-05').mkdir()
            result = f.call(h, A.GPU_IDS[0])
            self.assertEqual(result[0]['synthetic_gpu_passthrough'], A.GPU_IDS[0])
            self.assertEqual(len(f.module.calls), 1)
            self.assertIs(f.module.calls[0][0], f.c)
            self.assertIs(f.module._inventory, inventory)
        self.assertIs(f.module._load_attempt, original)

    def test_changed_live_attempt_candidate_arm_and_headers_reject(self):
        f = self.f; h = f.capture()
        for name in ('policy-00/attempt.json', 'policy-00/attempt.candidate.json',
                     'policy-00/world-00/arm-receipt.json', *A.HEADER_FILES):
            with self.subTest(name=name):
                with h.install(f.module):
                    p = f.root/name; old = p.read_bytes(); p.write_bytes(b'!'+old[1:])
                    try:
                        with self.assertRaises(ValueError): f.call(h)
                    finally: p.write_bytes(old)

    def test_accepted_but_semantically_invalid_headers_and_arms_reject(self):
        f = self.f; saved = dict(f.files)
        edits = [lambda h: h.update(phase='gpu'), lambda h: h.update(errors=['synthetic failure']),
                 lambda h: h['arms'][0].update(status='failed', reason='synthetic failure'),
                 lambda h: h.update(source_plan_pins=[]),
                 lambda h: h.update(service=dict(status='recorded', file='service/closed.json', sha256='1'*64))]
        for edit in edits:
            f.files = dict(saved); f.change_header(edit)
            with self.subTest(edit=edits.index(edit)), self.assertRaises(ValueError): f.capture()
        f.files = dict(saved)
        name = 'policy-00/world-00/arm-receipt.json'
        arm = json.loads(f.files[name]); arm['policy_id'] = A.GPU_IDS[0]; f.files[name] = raw(arm)
        def repin(h):
            h['arms'][0]['receipt_sha256'] = sha(f.files[name])
            h['evidence_sha256']['world-00/arm-receipt.json'] = sha(f.files[name])
        f.change_header(repin)
        with self.assertRaisesRegex(ValueError, 'arm source'): f.capture()

    def test_independent_pins_roster_candidate_and_bounds(self):
        f = self.f; inp = f.inputs()
        with self.assertRaises(ValueError): f.capture(acceptance_sha256='0'*64)
        with self.assertRaises(ValueError): f.capture(bundle_sha256='0'*64)
        with self.assertRaises(ValueError): f.capture(readset_sha256='0'*64)
        with self.assertRaises(ValueError): f.capture(campaign_source_raw=FAKE_SOURCE+b'\n')
        for edit in (lambda r: r['files'].pop('start.json'),
                     lambda r: r['files']['start.json'].update(bytes=True),
                     lambda r: r.update(campaign_root=str(f.root/'other')),
                     lambda r: r.update(cpu_acceptance_sha256='0'*64)):
            r = json.loads(inp['readset_raw']); edit(r); b = raw(r)
            with self.assertRaises(ValueError): f.capture(readset_raw=b, readset_sha256=sha(b))
        f.files['policy-00/attempt.candidate.json'] += b' '
        with self.assertRaises(ValueError): f.capture()
        big = b' '*(A.MAX_READSET+1)
        with self.assertRaises(ValueError): f.capture(readset_raw=big, readset_sha256=sha(big))

    def test_wrong_routing_campaign_and_live_self_hash_reject_before_original(self):
        f = self.f; h = f.capture()
        with h.install(f.module):
            for changes in (dict(expected='0'*64), dict(campaign_pin='0'*64),
                            dict(c={**f.c, 'scope': 'policy_train'}), dict(pid='unknown'),
                            dict(path=f.root/'policy-05'), dict(pid=A.GPU_IDS[0], path=f.root/'policy-00'),
                            dict(path=f.root/'policy-00/../policy-00')):
                with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError): f.call(h, **changes)
            p = f.root/'policy-00/attempt.json'; p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'CPU expected pin'):
                f.call(h, expected=sha(p.read_bytes()))
            self.assertEqual(f.module.calls, [])

    def test_symlink_root_ancestor_and_file_rejected(self):
        f = self.f; h = f.capture()
        for name in ('policy-00/attempt.json', 'policy-00/world-00', 'policy-00'):
            with self.subTest(name=name), h.install(f.module):
                p = f.root/name; other = p.with_name(p.name+'-original'); p.rename(other); p.symlink_to(other)
                try:
                    with self.assertRaises((OSError, ValueError)): f.call(h)
                finally: p.unlink(); other.rename(p)
        worker = f.cpu/'worker'; saved = f.cpu/'worker-original'; worker.rename(saved); worker.symlink_to(saved)
        try:
            with self.assertRaises((OSError, ValueError)):
                with h.install(f.module): pass
        finally: worker.unlink(); saved.rename(worker)

    def test_parent_replacement_and_service_or_finalization_appearance_reject(self):
        f = self.f; h = f.capture()
        with h.install(f.module):
            worker = f.cpu/'worker'; saved = f.cpu/'worker-original'; worker.rename(saved)
            shutil.copytree(saved, worker)
            try:
                with self.assertRaisesRegex(ValueError, 'identity changed'): f.call(h)
            finally: shutil.rmtree(worker); saved.rename(worker)
        for name in ('policy-04/service', 'finalization-intent.json', 'private'):
            with self.subTest(name=name), h.install(f.module):
                p = f.root/name; p.symlink_to(f.root/'missing-target')
                try:
                    with self.assertRaises(ValueError): f.call(h)
                finally: p.unlink()

    def test_original_identity_restore_error_retirement_and_nested_install(self):
        f = self.f; h = f.capture(); original = f.module._load_attempt
        with self.assertRaisesRegex(RuntimeError, 'synthetic body failure'):
            with h.install(f.module):
                callback = f.module._load_attempt
                with self.assertRaises(ValueError):
                    with h.install(f.module): pass
                raise RuntimeError('synthetic body failure')
        self.assertIs(f.module._load_attempt, original)
        with self.assertRaises(ValueError): callback(f.c, f.cpin, f.root/'policy-00', A.CPU_IDS[0], h._attempt_pins[A.CPU_IDS[0]])
        f.module._load_attempt = lambda *args: None
        with self.assertRaises(ValueError):
            with h.install(f.module): pass
        self.assertIsNot(f.module._load_attempt, original)
        f.module._load_attempt = original
        inventory = f.module._inventory; f.module._inventory = lambda p: {}
        with self.assertRaises(ValueError):
            with h.install(f.module): pass
        f.module._inventory = inventory

    def test_captured_inputs_detached_and_hardlinked_attempt_pair_legal(self):
        f = self.f; h = f.capture()
        p = f.root/'policy-00/attempt.candidate.json'; p.unlink(); os.link(f.root/'policy-00/attempt.json', p)
        f.files.clear()  # Mutable producer-side fixture objects confer no later authority.
        with h.install(f.module):
            self.assertEqual(f.call(h)[0]['policy_id'], A.CPU_IDS[0])
        with self.assertRaises(TypeError): h._files['start.json'] = b'changed'


if __name__ == '__main__':
    unittest.main(verbosity=2)
