"""Only new acceptance/accounting boundaries. Every packet below is FABRICATED.

The earlier Packet helper supplies synthetic publication metadata, never its test
suite. No actual custody, report, package, model, remote host or scientific code
is run. Independent pins are simulated explicitly; no production bypass exists.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_delivery_final_release_plan as planner
import verify_delivery_publication_release as public
from test_delivery_final_release import Packet

E, H = public.encoded, public.digest
AT = '2026-10-24T23:30:00+00:00'


def scheduler(job, success=True):
    state, exit_, elapsed = ('COMPLETED', '0:0', 2513) if success else ('FAILED', '1:0', 1792)
    rows = []
    for suffix in ('', '.batch', '.extern'):
        rows.append([str(job)+suffix]*2 + [state, exit_, '2026-10-24T15:27:34', '2026-10-24T16:09:27',
            str(elapsed), str(elapsed), '00:10.000', '1', '3G', '12K' if suffix == '.batch' else '',
            'cpu=1,mem=3G,node=1', 'ucb736_asc1', 'acpu', 'cpu-normal', 'fixture-node'])
    columns = 'JobIDRaw,JobID,State,ExitCode,Start,End,ElapsedRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AllocTRES,Account,Partition,QOS,NodeList'
    return {'commands': {'sacct': {'returncode': 0, 'argv': ['sacct', '-j', str(job), '--format='+columns],
        'stdout': '\n'.join('|'.join(row) for row in rows)+'\n'}}}


def accounting_fixture():
    attempts, old_rows, later = [], [], []
    for i in range(57):
        name = 'original_'+str(i+1).zfill(2) if i < 46 else 'replay'+str(i-43).zfill(2)
        seconds = 100 if i < 56 else 128690-5600
        row = {'attempt_id': name, 'kind': 'pilot' if i < 46 else 'replay', 'state': 'FAILED', 'exit_code': '1:0',
            'allocated_cpus': 1, 'allocated_cpu_seconds': seconds, 'elapsed_allocation_seconds': seconds,
            'requested_wall_seconds': None if i < 46 else 150000, 'slurm_total_cpu_seconds': 1.0,
            'wrapper_elapsed_seconds': None, 'raw_scheduler_row_sha256': H(('fixture row '+name).encode())}
        attempts.append(row)
        if i < 46:
            old_rows.append({'phase': row['kind'], 'state': row['state'], 'exit_code': row['exit_code'],
                'allocated_cpus': 1, 'allocated_cpu_seconds': seconds, 'elapsed_seconds': seconds,
                'slurm_total_cpu_raw': '00:01.000', 'raw_parent_row': 'fixture row '+name})
        else:
            later.append({'attempt': name, 'kind': row['kind'], 'state': row['state'], 'exit_code': row['exit_code'],
                'allocated_cpus': 1, 'allocated_cpu_seconds': seconds, 'elapsed_allocation_seconds': seconds,
                'wall_cap_seconds': 150000, 'slurm_total_cpu_seconds_as_recorded': 1.0,
                'raw_parent_row': 'fixture row '+name})
    measures = {}
    for name, job, state, seconds, wall in (('report01', 34864950, 'FAILED', 1792, 1800),
                                           ('report02', 34867191, 'COMPLETED', 2513, 3600)):
        remote = scheduler(job, name == 'report02'); row, _ = planner.scheduler_row(remote, job)
        attempts.append({'attempt_id': name, 'kind': 'reporting', 'state': state, 'exit_code': row[3],
            'allocated_cpus': 1, 'allocated_cpu_seconds': seconds, 'elapsed_allocation_seconds': seconds,
            'requested_wall_seconds': wall, 'slurm_total_cpu_seconds': 10.0, 'wrapper_elapsed_seconds': 30.0,
            'raw_scheduler_row_sha256': H('|'.join(row).encode())})
        measures[name] = {'state': state, 'allocated_cpu_seconds': seconds, 'slurm_total_cpu_seconds': 10.0,
            'wrapper_elapsed_seconds': 30.0, 'child_process_cpu_seconds': None if name == 'report01' else 2.0,
            'child_elapsed_seconds': None if name == 'report01' else 20.0,
            'child_peak_self_rss_kib': None if name == 'report01' else 10,
            'supervised_peak_tree_rss_bytes': 10240, 'slurm_batch_max_rss_kib': 12,
            'requested_cpus': 1, 'requested_memory_bytes': 3*2**30, 'requested_wall_seconds': wall,
            'observed_allocated_cpus': 1, 'observed_allocated_memory_bytes': 3*2**30, 'observed_gpus': 0,
            'scope': public.REPORT_MEASUREMENT_SCOPE}
    source = {'schema': 'ace-publication-incremental-accounting-review-v1',
        'original': {'allocation_rows': old_rows}, 'incremental_attempts': later,
        'reconciliation': {'total_distinct_allocations': 57, 'cumulative_through_replay12_allocated_cpu_seconds': 128690}}
    value = {'schema': public.ACCOUNTING_SCHEMA, 'scope': public.ACCOUNTING_SCOPE,
        'cutoff': {'attempt': 'report02', 'terminal_utc': '2026-10-24T22:09:27Z', 'allocation_count': 59,
                   'allocated_cpu_seconds': 132995}, 'attempts': attempts,
        'reservation': {'unit': 'CPU-core-hour', 'scope': 'delivery study and reproduction/reporting caps',
            'reserved': 94.36833333333334, 'ceiling': 150, 'basis': 'requested'},
        'original_fit_measurements': {'process_cpu_seconds': 110802.107301787, 'summed_elapsed_seconds': 113596.44237109905,
            'peak_tree_rss_bytes': 1400913920, 'scope': '640 original completed fits'},
        'report_measurements': measures,
        'unknowns': dict.fromkeys(('whole_sprint_process_cpu_seconds', 'confirmation_process_cpu_seconds',
            'complete_preparation_process_cpu_seconds', 'interrupted_returned_responses')),
        'response_accounting': {'charged': 60962, 'persisted': 57636, 'interrupted_reserved': 3326,
            'scope': 'inherited confirmation charge/reservation accounting; not Stage B totals'},
        'measurement_scopes_overlap': True, 'source_receipt_sha256s': [H(b'one'), H(b'two'), H(b'three')],
        'publication_approved': False, 'measurement_caveats': list(public.ACCOUNTING_CAVEATS)}
    return value, source


class PrivateFixture:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.rawroot = self.root/'raw'; self.rawroot.mkdir()
        self.packet = Packet(); p = self.packet
        self.f = copy.deepcopy(p.evidence['report_freeze'])
        self.r = copy.deepcopy(p.evidence['report_receipt'])
        self.e = copy.deepcopy(p.evidence['report_execution'])
        self.rt = copy.deepcopy(p.evidence['report_runtime'])
        self.raw = {n: p.writing[target] for n, target in public.REPORT_OUTPUTS.items()}
        for name in public.REPORT_SOURCES:
            self.raw[name] = p.writing['scripts/research/'+name] if name in public.PUBLIC_REPORT_SOURCES else ('# fabricated '+name).encode()
        self.f.update(schema='delivery-prospective-reporting-execution-v1', source_revision='a'*40,
            output='/private/fabricated', source_pins={n: H(self.raw[n]) for n in public.REPORT_SOURCES},
            cpus=1, rss_bytes=3*2**30, wall_seconds=3600, slurm_wall_limit='01:00:00',
            account='ucb736_asc1', partition='acpu', qos='cpu-normal',
            authorization_sha256=H(b'authorization'), output_claim_sha256=H(b'claim'))
        self.raw.update({'authorization.json': b'authorization', 'output_claim.json': b'claim',
            'run.sbatch': b'not executable fixture', 'report_exports_started.json': E({'at': AT}),
            'replay_receipt.json': E(p.evidence['replay'])})
        loaded = {}
        for i in range(618):
            module = ('numpy', 'scipy')[i] if i < 2 else 'numpy.fixture_'+str(i)
            name = module.split('.')[0]
            loaded[module] = {'path': 'lib/python3.11/site-packages/'+name+'/'+('__init__.py' if i < 2 else 'fixture_'+str(i)+'.py'),
                             'sha256': H(('statistics fixture '+str(i)).encode())}
        objects = [{'path': row['path'], 'kind': 'file', 'sha256': row['sha256'], 'bytes': 1} for row in loaded.values()]
        objects += [{'path': 'fixture/'+str(i), 'kind': 'file', 'sha256': H(b'x'), 'bytes': 1} for i in range(28577-len(objects))]
        self.inventory = {'prefix': '/private/environment', 'files': objects}
        self.r.update(loaded_statistics_files=loaded, process_cpu_seconds=2.0, elapsed_seconds=20.0, peak_self_rss_kib=10)
        self.rt['dependencies'] = {n: {'version': public.reports.DEPENDENCIES[n], 'sha256': loaded[n]['sha256'],
            'origin': '/private/environment/'+loaded[n]['path']} for n in ('numpy', 'scipy')}
        self.r['statistics_dependencies'] = self.rt['dependencies']; self.rt['environment_objects'] = 28577
        fields = {'JobId': '34867191', 'Account': 'ucb736_asc1', 'Partition': 'acpu', 'QOS': 'cpu-normal',
            'NumNodes': '1', 'NumCPUs': '1', 'NumTasks': '1', 'CPUs/Task': '1', 'TimeLimit': '01:00:00',
            'MinMemoryNode': '3G', 'AllocTRES': 'cpu=1,mem=3G,node=1', 'TresPerNode': None, 'TresPerTask': 'cpu=1', 'Gres': None}
        raw_control = ' '.join(k+'='+v for k, v in fields.items() if v is not None)
        self.e.update(job_id='34867191', elapsed_seconds=30.0,
            allocation={'raw_scontrol': raw_control, 'raw_scontrol_sha256': H(raw_control.encode()), 'verified_fields': fields},
            deadline_policy={'cleanup_reserve_seconds': 50, 'rounding_reserve_seconds': 2,
                'maximum_wrapper_relative_child_seconds': 3550, 'remaining_child_seconds_at_check': 3500})
        self.e['child'].update(primary_failure=None, secondary_failures=[], samples=[{'rss_bytes': 10240}])
        self.remote = scheduler(34867191)
        self.accounting, self.accounting_source = accounting_fixture()
        self.rebuild()

    def rebuild(self):
        self.raw['runtime_inventory.json'] = E(self.inventory)
        replay = {'accepted': True, 'job_id': 34863006, 'manifest_sha256': self.f['manifest_sha256'],
            'replay_receipt_sha256': H(self.raw['replay_receipt.json']), 'runtime_inventory_sha256': H(self.raw['runtime_inventory.json'])}
        self.raw['replay_acceptance.json'] = E(replay)
        self.f['metadata_pins'] = {n: H(self.raw[n]) for n in public.REPORT_METADATA}
        self.raw['report_freeze.json'] = E(self.f)
        for k in ('source_pins', 'metadata_pins', 'original_input_pins'):
            self.r[k] = copy.deepcopy(self.f[k])
        self.r.update(freeze_sha256=H(E(self.f)), outputs={n: H(self.raw[n]) for n in public.REPORT_OUTPUTS})
        self.raw['report_receipt.json'] = E(self.r)
        self.e.update(freeze_sha256=H(E(self.f)), report_receipt_sha256=H(E(self.r)))
        self.raw['report_execution.json'] = E(self.e)
        self.rt['inventory_sha256'] = H(self.raw['runtime_inventory.json'])
        self.raw['report_runtime_verified.json'] = E(self.rt)
        self.frozen = {n: H(self.raw[n]) for n in (*public.REPORT_SOURCES, *public.REPORT_METADATA,
            'report_freeze.json', 'authorization.json', 'output_claim.json', 'run.sbatch')}
        self.raw.pop('padding.bin', None)
        while len(self.raw) < 49:
            self.raw['extra_'+str(len(self.raw))+'.txt'] = b'fabricated'
        self.raw['padding.bin'] = b'x'*(9147453-sum(len(v) for v in self.raw.values()))
        for n, raw in self.raw.items():
            path = self.rawroot/n; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        self.remote.update(root=self.f['output'], file_count=50, bytes=9147453,
            files=[{'path': n, 'sha256': H(v), 'bytes': len(v)} for n, v in self.raw.items()])
        custody = {'at': AT, 'job_id': 34867191, 'parent_state': 'COMPLETED', 'files_verified': 50,
            'bytes_verified': 9147453, 'frozen_inputs_verified': 17, 'scientific_outcomes_decoded_by_collection': False,
            'remote_inventory_sha256': H(E(self.remote))}
        self.originals = {k: self.raw[n] for k, n in planner.REPORT_RECORDS.items()}
        self.originals.update(report_custody=E(custody), remote_inventory=E(self.remote), report_input_pins=E(self.frozen),
            report_review=b'FABRICATED independent review, zero required findings',
            accounting_review=b'FABRICATED reviewed accounting', accounting_source=E(self.accounting_source),
            failed_report_inventory=E(scheduler(34864950, False)))
        row, rows = planner.scheduler_row(self.remote, 34867191)
        v = {'at': AT, 'job_id': 34867191, 'actual_execution_checks_pass': True,
            'independent_scientific_output_review_pending': True,
            **{k: H(self.originals[n]) for k, n in planner.ACCEPTANCE_EDGES.items() if n != 'execution_verification'},
            'primary_index_sha256': H(self.raw['primary/claim_index.json']),
            'descriptive_index_sha256': H(self.raw['descriptive/output_index.json']), 'counts': public.COUNTS,
            'new_fits': 0, 'new_optimizer_updates': 0, 'new_responses': 0, 'allocation_seconds': 2513,
            'known_cumulative_allocated_seconds': 132995, 'known_cumulative_allocated_core_hours': 132995/3600,
            'reservation_core_hours': 94.36833333333334, 'slurm_total_cpu': row[8],
            'slurm_max_rss_by_step': {r[1]: r[11] for r in rows}, 'child_process_cpu_seconds': 2.0,
            'child_elapsed_seconds': 20.0, 'child_peak_rss_kib': 10, 'wrapper_elapsed_seconds': 30.0,
            'measurements_overlap_not_additive': True, 'whole_sprint_cpu_unknown': True}
        self.originals['execution_verification'] = E(v)
        witness = {'schema': 'ace-report02-independent-review-witness-v1', 'job_id': 34867191, 'required_findings': [],
            'checks': {'numerical': {'counts': public.COUNTS}, 'reporting': {'outputs': 13},
                       'execution': {'source_revision': self.f['source_revision']}},
            'launch_input_sha256': self.frozen, 'output_sha256': self.r['outputs'],
            'historical_input_sha256': self.f['original_input_pins']}
        self.originals['report_review_witness'] = E(witness)
        self.acceptance = {**v, 'schema': planner.REPORT_ACCEPTANCE_SCHEMA, 'accepted': True,
            'independent_scientific_output_review_pending': False, 'review_sha256': H(self.originals['report_review']),
            'review_witness_sha256': H(E(witness)), 'execution_verification_sha256': H(E(v)),
            'review_required_findings': 0, 'reviewed_output_counts': public.COUNTS, 'publication_approved': False,
            **{k: self.f[k] for k in ('source_pins', 'metadata_pins', 'original_input_pins')}}
        self.originals['report_acceptance'] = E(self.acceptance)
        self.accounting['source_receipt_sha256s'] = [H(self.originals[k]) for k in
            ('accounting_source', 'failed_report_inventory', 'remote_inventory')]
        self.originals['accounting'] = E(self.accounting)
        self.inputs = {k: {'path': str(self.root/k), 'sha256': H(v)} for k, v in self.originals.items()}
        self.inputs['report_freeze']['path'] = str(self.rawroot/'report_freeze.json')

    def check(self, accepted_pin=None):
        with patch.object(public, 'CANDIDATE_SHA', self.f['manifest_sha256']), \
             patch.object(public, 'REPLAY_ACCEPTANCE_SHA', H(self.raw['replay_acceptance.json'])):
            a = planner.require_report_acceptance(self.originals['report_acceptance'], self.inputs, self.originals,
                    accepted_pin if accepted_pin is not None else H(self.originals['report_acceptance']))
            planner.validate_private_accounting(self.originals, a)
            return a


class AcceptanceTests(unittest.TestCase):
    def test_complete_fabricated_private_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp); a = p.check()
            self.assertTrue(a['accepted']); self.assertFalse(a['publication_approved'])
            self.assertEqual(len(p.r['loaded_statistics_files']), 618)

    def test_external_pin_and_review_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp)
            for value in (None, '', H(b'not independent acceptance')):
                with self.subTest(pin=value), self.assertRaises(ValueError):
                    planner.require_report_acceptance(p.originals['report_acceptance'], p.inputs, p.originals, value)
            for key, value in [('accepted', False), ('job_id', 34864950), ('job_id', True),
                               ('review_required_findings', True), ('review_required_findings', 1),
                               ('publication_approved', True), ('independent_scientific_output_review_pending', True)]:
                changed = {**p.acceptance, key: value}; p.originals['report_acceptance'] = E(changed)
                p.inputs['report_acceptance']['sha256'] = H(E(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    p.check()
            p.rebuild(); p.originals['report_review'] += b' changed'
            with self.assertRaisesRegex(ValueError, 'review differs'): p.check()
            p.rebuild(); w = public.decode(p.originals['report_review_witness']); w['required_findings'] = ['open']
            p.originals['report_review_witness'] = E(w)
            with self.assertRaisesRegex(ValueError, 'review not closed/bound'): p.check()

    def test_resealed_scheduler_cleanup_and_resource_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp)
            mutations = [lambda: p.e['child']['cleanup'].update(complete=False),
                lambda: p.e.update(job_id='34864950'), lambda: p.f.update(cpus=True),
                lambda: p.e['allocation']['verified_fields'].update(NumCPUs='2'),
                lambda: p.e['deadline_policy'].update(cleanup_reserve_seconds=0),
                lambda: p.remote['commands']['sacct'].update(stdout=p.remote['commands']['sacct']['stdout'].replace('COMPLETED', 'FAILED'))]
            original = copy.deepcopy((p.f, p.e, p.remote))
            for i, mutate in enumerate(mutations):
                p.f, p.e, p.remote = copy.deepcopy(original); mutate(); p.rebuild()
                with self.subTest(case=i), self.assertRaises(ValueError): p.check()

    def test_resealed_runtime_and_output_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp)
            original = copy.deepcopy((p.r, p.inventory, p.raw))
            mutations = [lambda: p.r['loaded_statistics_files']['numpy.fixture_2'].update(sha256=H(b'wrong')),
                lambda: p.r['loaded_statistics_files'].pop('numpy.fixture_2'),
                lambda: p.inventory['files'].append(p.inventory['files'][0]),
                lambda: p.r['loaded_statistics_files']['numpy'].update(path='../numpy.py'),
                lambda: p.raw.update({'primary/claim_index.json': E({'output_hashes': {}})})]
            for i, mutate in enumerate(mutations):
                p.r, p.inventory, p.raw = copy.deepcopy(original); mutate(); p.rebuild()
                with self.subTest(case=i), self.assertRaises(ValueError): p.check()

    def test_custody_membership_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp)
            extra = p.rawroot/'unreviewed'; extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'custody membership'): p.check()
            extra.unlink(); target = p.rawroot/'padding.bin'; target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'pinned input changed'): p.check()
            p.rebuild(); frozen = copy.deepcopy(p.frozen); frozen['run.sbatch'] = H(b'wrong')
            p.originals['report_input_pins'] = E(frozen)
            with self.assertRaises(ValueError): p.check()

    def test_accounting_schema_mutations(self):
        original, _ = accounting_fixture(); public.validate_accounting(original)
        mutations = [lambda v: v['attempts'].pop(), lambda v: v['attempts'][1].update(attempt_id='original_01'),
            lambda v: v['attempts'][0].update(allocated_cpus=True),
            lambda v: v['attempts'][0].update(allocated_cpu_seconds=-1),
            lambda v: v['attempts'][0].update(slurm_total_cpu_seconds=float('nan')),
            lambda v: v['attempts'][0].update(requested_wall_seconds=0),
            lambda v: v['attempts'][-2].update(state='COMPLETED', exit_code='0:0'),
            lambda v: v['reservation'].update(unit='CPU-second'),
            lambda v: v['unknowns'].update(whole_sprint_process_cpu_seconds=0),
            lambda v: v['cutoff'].update(allocated_cpu_seconds=132994),
            lambda v: v.update(measurement_scopes_overlap=False),
            lambda v: v['measurement_caveats'].pop(),
            lambda v: v['report_measurements']['report01'].update(child_process_cpu_seconds=0),
            lambda v: v['report_measurements']['report02'].update(observed_gpus=True)]
        for i, mutate in enumerate(mutations):
            value = copy.deepcopy(original); mutate(value)
            with self.subTest(case=i), self.assertRaises(ValueError): public.validate_accounting(value)

    def test_private_accounting_preserves_failures_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PrivateFixture(tmp); p.check()
            for mutate in (lambda v: v['attempts'][0].update(state='COMPLETED', exit_code='0:0'),
                           lambda v: v['source_receipt_sha256s'].__setitem__(0, H(b'wrong')),
                           lambda v: v['report_measurements']['report02'].update(child_process_cpu_seconds=3.0)):
                value = copy.deepcopy(p.accounting); mutate(value); p.originals['accounting'] = E(value)
                with self.assertRaises(ValueError): planner.validate_private_accounting(p.originals, p.acceptance)

    def test_public_binding_and_scope_without_replaying_old_suite(self):
        p = Packet(); accounting, _ = accounting_fixture()
        p.evidence['report_execution'].update(job_id='34867191', elapsed_seconds=30.0)
        p.evidence['report_receipt'].update(process_cpu_seconds=2.0, elapsed_seconds=20.0, peak_self_rss_kib=10)
        p.evidence['report_execution']['report_receipt_sha256'] = H(E(p.evidence['report_receipt']))
        p.pins['accounting'] = H(E(accounting))
        for key, value in zip(('accounting_source', 'failed_report_inventory', 'remote_inventory'), accounting['source_receipt_sha256s']):
            p.pins[key] = value
        raw, contract, manifest, _ = planner.publication_payload(p.writing, p.old_raw, p.old_source,
            {k: E(v) for k, v in p.evidence.items()}, p.pins, b'%PDF-fixture', b'# fixture', E(accounting))
        raw.update(p.inherited)
        with patch.object(public, 'CANDIDATE_SHA', p.old_pin), patch.object(public, 'SOURCE_CONTRACT_SHA', H(p.old_source)):
            result = public.validate_publication(manifest, p.old_raw, contract, raw)
            self.assertTrue(result['accounting_disclosure_complete'])
            self.assertFalse(result['private_execution_revalidated'])
            self.assertEqual(manifest['files'][:3893], p.old['files'])
            self.assertEqual(manifest['bindings'][:5068], p.old['bindings'])
            for name in public.FALSE_FLAGS: self.assertIs(contract[name], False)
            self.assertEqual(raw['notices/ACE_RUNNER_MIT.txt'], p.inherited['notices/ACE_RUNNER_MIT.txt'])
            changed = copy.deepcopy(contract); changed['accounting_disclosure']['sha256'] = H(b'forged')
            with self.assertRaisesRegex(ValueError, 'accounting artifact binding'):
                public.validate_publication(manifest, p.old_raw, changed, raw)
            changed = copy.deepcopy(manifest)
            changed['files'] = [v for v in changed['files'] if v['path'] != public.ACCOUNTING]
            with self.assertRaises((ValueError, KeyError)):
                public.validate_publication(changed, p.old_raw, contract, raw)

    def test_request_requires_external_pins_before_reading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            values = {k: {'path': str(root/k), 'sha256': H(k.encode())} for k in planner.INPUTS}
            values['manifest']['sha256'] = public.CANDIDATE_SHA; values['plan']['sha256'] = public.PLAN_SHA
            values['replay_acceptance']['sha256'] = public.REPLAY_ACCEPTANCE_SHA
            request = E({'schema': 'delivery-final-release-inputs-v1', 'inputs': values})
            path = root/'request'; path.write_bytes(request)
            for a, b in ((H(b'wrong'), values['accounting']['sha256']),
                         (values['report_acceptance']['sha256'], H(b'wrong'))):
                with self.assertRaisesRegex(ValueError, 'independent acceptance/accounting pin differs'):
                    planner.request_inputs(path, H(request), a, b)
            with self.assertRaisesRegex(ValueError, 'pinned input changed'):
                planner.request_inputs(path, H(b'wrong'), H(b'a'), H(b'b'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
