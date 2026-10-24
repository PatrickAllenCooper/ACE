"""New integration boundaries only. All studies/evidence here are fabricated.

The production acceptance gate is never overridden; positive coverage exercises
pure downstream construction/checking, not a qualified preparation or package.
"""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_delivery_final_release_plan as planner
import verify_delivery_publication_release as public

E, H = public.encoded, public.digest
# Arbitrary synthetic identity, never a real report01/report02 execution claim.
SYNTHETIC_REPORT_JOB = '900000001'


def entry(name, raw):
    return {'path': name, 'sha256': H(raw), 'original_sha256': H(raw), 'bytes': len(raw),
            'role': 'fixture', 'transform': {'kind': 'identity'}}


def put(value, selector, item):
    parts = selector[1:].split('/')
    for part in parts[:-1]:
        value = value.setdefault(part, {})
    value[parts[-1]] = copy.deepcopy(item)


class Packet:
    def __init__(self):
        self.writing = {n: ('% artificial '+n+'\n').encode() for n in public.SNAPSHOT_MAP}
        self.primary_analysis = {'fixture': [1.0, 2.0, 3.0, 4.0]}
        self.secondary = {'fixture': [5.0, 6.0]}
        historical = {n: H(('historical '+n).encode()) for n in public.ORIGINAL_INPUTS}
        historical['registration.json'] = public.reports.REGISTRATION
        acceptance = E({'primary_analysis': self.primary_analysis, 'secondary_descriptive_analysis': self.secondary})
        scores = E({'fixture': 'complete projected score metadata'})
        contract = {'registration_sha256': public.reports.REGISTRATION,
            'upstream_gate': {'acceptance_sha256': historical['acceptance.json'],
                'scores_sha256': historical['scores.json'], 'audit_execution_sha256': historical['audit_execution.json']},
            'original_complete_sha256': historical['complete.json'], 'fit_seal_sha256': historical['fit_seal.json'],
            'original_acceptance_sha256': historical['acceptance.json'], 'original_scores_sha256': historical['scores.json'],
            'metadata_projections': {}}
        for name, raw in (('acceptance', acceptance), ('scores', scores)):
            path = 'B/'+name+'.json'
            contract['metadata_projections'][path] = {'original_path': path,
                'original_sha256': historical[name+'.json'], 'derived_sha256': H(raw)}
        exporter_sha = H(self.writing['scripts/research/export_delivery_prospective_claims.py'])
        source = {'runner_grant_authority_resolved': True,
            'bindings': [{'original_location': f'fixture/{i}.py', 'original_sha256': H(str(i).encode())} for i in range(24)]}
        source['bindings'][0] = {'original_location': 'B/scripts/research/export_delivery_prospective_claims.py',
                                'original_sha256': exporter_sha}
        self.old_source = E(source)
        self.inherited = {'B/acceptance.json': acceptance, 'B/scores.json': scores,
            'B/replay_contract.json': E(contract), 'reproduction/source_contract.json': self.old_source,
            'reproduction/source_contract_A_C_F.json': E({'historical': True}),
            'prepare_delivery_prospective_supplement.py': self.writing['scripts/research/prepare_delivery_prospective_supplement.py'],
            'notices/ACE_RUNNER_MIT.txt': b'fixture required notice\n'}
        files = [entry(n, raw) for n, raw in self.inherited.items()]
        files += [entry(f'fixture/{i}.bin', b'x') for i in range(public.OLD_FILES-len(files))]
        self.old = {'schema': 'delivery-anonymous-derived-v1', 'files': files,
            'bindings': [{'record': 'B/replay_contract.json', 'pointer': '/registration_sha256',
                          'artifact': 'fixture/0.bin', 'digest': 'original_sha256'} for _ in range(public.OLD_BINDINGS)]}
        self.old_raw = E(self.old); self.old_pin = H(self.old_raw)
        self.evidence = {}
        for kind, specs in public.SPECS.items():
            value = {}
            for selector, rule in specs.items():
                default = 0 if type(rule) is dict else {
                    'hash': H(b'pin'), 'analysis': {}, 'int': 0, 'true': True,
                    'version': '1.0', 'report-job': SYNTHETIC_REPORT_JOB, 'sources': {},
                    'metadata': {}, 'outputs': {}, 'originals': {}, 'statistics-files': {}}.get(rule, rule)
                put(value, selector, default)
            self.evidence[kind] = value
        freeze = self.evidence['replay_freeze']
        freeze.update(manifest_sha256=self.old_pin, package_files=3893, package_bindings=5068,
                      exact_dependencies=public.reports.DEPENDENCIES, package='/private/fixture/package')
        replay = self.evidence['replay']
        replay.update(runtime=public.reports.DEPENDENCIES, checkpoints_replayed=640,
            primary_cells_recomputed=240, cached_responses_checked=48000,
            primary_analysis=self.primary_analysis, secondary_descriptive_analysis=self.secondary)
        replay['integrity'].update(manifest_sha256=self.old_pin, files_verified=3893, bindings_verified=5068)
        execution = self.evidence['replay_execution']
        execution.update(manifest_sha256=self.old_pin, freeze_sha256=H(E(freeze)), replay_receipt_sha256=H(E(replay)))
        execution['authenticated_runtime'].update(environment_objects_verified=28577)
        execution['child_execution']['command'] = ['/private/fixture/python', 'private-script.py']
        provenance = {'manifest_sha256': self.old_pin, 'supplemental_replay_sha256': H(E(replay)),
            'original_registration_sha256': public.reports.REGISTRATION,
            'acceptance_sha256': H(acceptance), 'scores_sha256': H(scores),
            'tool_sha256': H(self.inherited['prepare_delivery_prospective_supplement.py'])}
        supplement = {k: [None]*n for k, n in public.COUNTS.items()}
        supplement.update(provenance=provenance, new_fits=0, new_responses=0, additional_significance_tests=0)
        self.writing[public.PAPER+'delivery_prospective_supplement.json'] = E(supplement)
        primary = {'registration_sha256': public.reports.REGISTRATION, 'source_revision': public.reports.REVISION,
            'acceptance_sha256': historical['acceptance.json'], 'scores_sha256': historical['scores.json'],
            'exporter_sha256': exporter_sha, 'deployment_initialization': 0, 'supervision_and_compute_matched': False,
            'primary_analysis': self.primary_analysis, 'secondary_descriptive_analysis': self.secondary,
            'output_hashes': {n: H(self.writing[public.PAPER+n]) for n in public.reports.PRIMARY}}
        descriptive = {'provenance': provenance, 'actual_numerical_replay_performed_here': False,
            'output_hashes': {n: H(self.writing[public.PAPER+(('delivery_prospective_'+n) if n in public.reports.DATA else n)])
                              for n in public.reports.DESCRIPTIVE+public.reports.DATA}}
        self.writing[public.PAPER+public.reports.PRIMARY_INDEX] = E(primary)
        self.writing[public.PAPER+public.reports.DESCRIPTIVE_INDEX] = E(descriptive)
        self.writing[public.PAPER+'tmlr_style_provenance.json'] = E({'files':
            {n: {'sha256': H(self.writing[public.PAPER+n])} for n in public.STYLES}})
        self.writing[public.PAPER+'claim_index.json'] = E({'generator_sha256': H(self.writing['scripts/research/generate_delivery_claims.py'])})
        chunks = ['% BEGIN EDITOR COMPANION BUNDLE (generated; edit companion files)']
        for n in public.COMPANIONS:
            raw = self.writing[public.PAPER+n]
            chunks += [f'% {n}: SHA256 {H(raw)}', f'\\begin{{filecontents*}}[overwrite]{{{n}}}',
                       raw.decode().rstrip('\n'), '\\end{filecontents*}']
        chunks.append('% END EDITOR COMPANION BUNDLE')
        self.writing[public.PAPER+'paper.tex'] = '\n'.join(chunks).encode()
        report_freeze = self.evidence['report_freeze']
        report_freeze.update(manifest_sha256=self.old_pin, source_pins={
            **{name: H(('fabricated private source '+name).encode()) for name in public.REPORT_SOURCES},
            'export_delivery_prospective_claims.py': exporter_sha,
            'prepare_delivery_prospective_supplement.py': provenance['tool_sha256']},
            original_input_pins=historical, metadata_pins={'replay_receipt.json': H(E(replay)),
            'replay_acceptance.json': public.REPLAY_ACCEPTANCE_SHA, 'runtime_inventory.json': H(b'pin')})
        receipt = self.evidence['report_receipt']
        receipt.update(freeze_sha256=H(E(report_freeze)), source_pins=report_freeze['source_pins'],
            original_input_pins=historical, metadata_pins=report_freeze['metadata_pins'], counts=public.COUNTS,
            outputs={n: H(self.writing[target]) for n, target in public.REPORT_OUTPUTS.items()})
        runtime = self.evidence['report_runtime']; runtime['environment_objects'] = 28577
        for name in ('numpy', 'scipy'):
            receipt['statistics_dependencies'][name].update(version=public.reports.DEPENDENCIES[name], origin='/private/runtime/'+name)
            runtime['dependencies'][name].update(version=public.reports.DEPENDENCIES[name], origin='/private/runtime/'+name)
            receipt['loaded_statistics_files'][name] = {
                'path': 'lib/python3.11/site-packages/'+name+'/__init__.py',
                'sha256': runtime['dependencies'][name]['sha256']}
        receipt['loaded_statistics_files']['numpy.fixture'] = {
            'path': 'lib/python3.11/site-packages/numpy/fixture.py', 'sha256': H(b'fabricated module')}
        # Artificial quantities only; these are never actual run measurements.
        report_freeze.update(cpus=1, rss_bytes=3*2**30, wall_seconds=3600)
        receipt.update(freeze_sha256=H(E(report_freeze)), process_cpu_seconds=12.5,
                       elapsed_seconds=30.5, peak_self_rss_kib=123456)
        runtime.update(process_cpu_seconds=2.5, elapsed_seconds=8.5)
        report_execution = self.evidence['report_execution']
        report_execution.update(freeze_sha256=H(E(report_freeze)), report_receipt_sha256=H(E(receipt)))
        report_execution['child']['cleanup']['private_process'] = '/private/process'
        for kind, key in (('replay_freeze', 'freeze'), ('replay_execution', 'execution'), ('replay', 'replay')):
            self.writing[public.PAPER+public.reports.PROOFS[key]] = E(self.evidence[kind])
        self.writing[public.PAPER+public.reports.PROOFS['manifest']] = self.old_raw
        self.writing[public.PAPER+public.reports.PROOFS['contract']] = E(contract)
        self.pins = {n: H(('private '+n).encode()) for n in public.PRIVATE_KEYS}
        self.pins.update(plan=public.PLAN_SHA, replay_acceptance=public.REPLAY_ACCEPTANCE_SHA,
                         bundle=H(self.writing[public.PAPER+public.reports.BUNDLE]))

    def construct(self):
        self.raw, self.contract, self.manifest, self.omissions = planner.publication_payload(
            self.writing, self.old_raw, self.old_source, {k: E(v) for k, v in self.evidence.items()},
            self.pins, b'%PDF-1.7\nfixture', b'# artificial checker source\n')
        self.raw.update(self.inherited)

    def validate(self):
        with patch.object(public, 'CANDIDATE_SHA', self.old_pin), patch.object(public, 'SOURCE_CONTRACT_SHA', H(self.old_source)):
            return public.validate_publication(self.manifest, self.old_raw, self.contract, self.raw)


class FinalReleaseTests(unittest.TestCase):
    def test_bootstrap_authenticates_before_dependency_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checker = root/public.CHECKER
            checker.write_bytes(Path(public.__file__).read_bytes())
            sources = {name: (Path(public.__file__).parent/name).read_bytes() for name in public.BOOTSTRAP_PINS}
            for changed in (None, *sources):
                for name, raw in sources.items():
                    if name == changed:
                        raw = b"from pathlib import Path\nPath(__file__).with_name('EXECUTED').touch()\n"
                    (root/name).write_bytes(raw)
                result = subprocess.run([sys.executable, '-I', '-B', str(checker), '--help'],
                                        cwd=root, capture_output=True, text=True)
                # Retain every subprocess stream in the outer captured test log.
                print('bootstrap case:', changed, 'exit:', result.returncode)
                print(result.stdout, end=''); print(result.stderr, end='')
                self.assertEqual(result.returncode, 0 if changed is None else 1)
                self.assertFalse((root/'EXECUTED').exists())
                if changed:
                    self.assertIn('bootstrap dependency digest differs: '+changed, result.stderr)
            # All hashes must pass before *either* captured dependency executes.
            with patch.object(public, '__file__', str(checker)), patch('builtins.exec') as execute:
                with self.assertRaisesRegex(ValueError, 'bootstrap dependency digest differs'):
                    public.authenticated_dependencies()
                execute.assert_not_called()
        # An already-imported substitute cannot satisfy a pinned byte load.
        with patch.dict(sys.modules, {'delivery_prospective_publication': object(),
                                     'verify_delivery_release': object()}):
            loaded = public.authenticated_dependencies()
            self.assertEqual(loaded['delivery_prospective_publication.py'].digest(b'x'), H(b'x'))

    def test_exact_report_source_and_metadata_memberships(self):
        for kind in ('report_freeze', 'report_receipt'):
            for key, name in (('source_pins', 'runner_delivery_confirmation.py'),
                              ('metadata_pins', 'runtime_inventory.json')):
                for operation in ('remove', 'add'):
                    p = Packet(); row = p.evidence[kind][key]
                    if operation == 'remove':
                        del row[name]
                    else:
                        row['unreviewed_import.py' if key == 'source_pins' else 'unreviewed_evidence.json'] = H(b'unknown')
                    with self.subTest(kind=kind, key=key, operation=operation), self.assertRaisesRegex(ValueError, 'exact .* membership'):
                        p.construct()
        p = Packet(); p.construct(); p.validate()
        self.assertEqual(len(p.evidence['report_receipt']['source_pins']), 10)
        self.assertEqual(len(p.evidence['report_receipt']['metadata_pins']), 3)

    def test_loaded_statistics_paths_hashes_and_source_dispositions(self):
        p = Packet(); p.construct(); p.validate()
        projection = public.decode(p.raw['publication/evidence/report_receipt.json'])
        self.assertEqual(projection['values']['/loaded_statistics_files'],
                         p.evidence['report_receipt']['loaded_statistics_files'])
        self.assertFalse(any(s.startswith('/loaded_statistics_files') for s in p.omissions['report_receipt']))
        dispositions = p.contract['report_source_dispositions']
        self.assertEqual(sum(row['disposition'] == 'public-artifact' for row in dispositions.values()), 2)
        for name in set(public.REPORT_SOURCES)-set(public.PUBLIC_REPORT_SOURCES):
            self.assertEqual(dispositions[name]['custody_sha256'], p.pins['report_custody'])
        for mutate in (
            lambda m: m.pop('scipy'),
            lambda m: m.update({'other.fixture': m['numpy']}),
            lambda m: m['numpy'].update(path='/private/numpy/__init__.py'),
            lambda m: m['numpy'].update(path='../numpy/__init__.py'),
            lambda m: m['numpy'].update(path='lib//numpy/__init__.py'),
            lambda m: m['numpy'].update(origin='/private/environment'),
            lambda m: m['numpy'].update(sha256='0'*64),
        ):
            p = Packet(); mutate(p.evidence['report_receipt']['loaded_statistics_files'])
            p.evidence['report_execution']['report_receipt_sha256'] = H(E(p.evidence['report_receipt']))
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                p.construct(); p.validate()
        for mutate in (
            lambda p: p.contract['report_source_dispositions'].pop('runner_delivery_confirmation.py'),
            lambda p: p.contract['report_source_dispositions']['runner_delivery_confirmation.py'].update(custody_sha256='0'*64),
            lambda p: p.contract['report_source_dispositions']['export_delivery_prospective_claims.py'].update(path='unbound.py'),
        ):
            p = Packet(); p.construct(); mutate(p)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, 'source dispositions'):
                p.validate()

    def test_resource_quantities_units_scopes_and_unknowns(self):
        p = Packet(); p.construct(); p.validate()
        for kind, fields in public.RESOURCE_FIELDS.items():
            result = public.decode(p.raw['publication/evidence/'+kind+'.json'])
            for selector, (_, unit, scope, basis) in fields.items():
                with self.subTest(kind=kind, selector=selector):
                    value = result['values'][selector]
                    self.assertEqual(value, {'value': p.evidence[kind][selector[1:]], 'status': 'recorded',
                                             'unit': unit, 'scope': scope, 'basis': basis})
                    self.assertNotIn(selector, p.omissions[kind])
                    for unknown in ('missing', None):
                        raw = copy.deepcopy(p.evidence[kind])
                        if unknown == 'missing': del raw[selector[1:]]
                        else: raw[selector[1:]] = None
                        derived = public.project(kind, E(raw))
                        public.projection(derived, kind)
                        self.assertEqual(derived['values'][selector]['status'], 'unknown')
                        self.assertIsNone(derived['values'][selector]['value'])
        disclosure = p.contract['accounting_disclosure']
        self.assertIsNone(disclosure['whole_sprint_process_cpu']['value'])
        self.assertIsNone(disclosure['reporting_cutoff'])
        self.assertIn('34864950', disclosure['failed_report_jobs'])
        self.assertFalse(p.validate()['accounting_disclosure_complete'])
        disclosure['status'] = 'complete'
        with self.assertRaisesRegex(ValueError, 'disclosure requirement'): p.validate()

    def test_resource_quantities_reject_nonfinite_negative_bool_and_relabeling(self):
        p = Packet()
        for kind, fields in public.RESOURCE_FIELDS.items():
            good = public.project(kind, E(p.evidence[kind]))
            for selector, (numeric, *_) in fields.items():
                for invalid in (True, False, -1, -0.5, float('inf'), float('nan'), '12.5', [], {}):
                    bad = copy.deepcopy(good); bad['values'][selector]['value'] = invalid
                    with self.subTest(kind=kind, selector=selector, invalid=invalid), self.assertRaises(ValueError):
                        public.projection(bad, kind)
                if numeric == 'int':
                    bad = copy.deepcopy(good); bad['values'][selector]['value'] = 1.5
                    with self.assertRaises(ValueError): public.projection(bad, kind)
                for key, value in (('unit', 'hours'), ('scope', 'whole sprint'), ('basis', 'inferred'), ('status', 'unknown')):
                    bad = copy.deepcopy(good); bad['values'][selector][key] = value
                    with self.subTest(key=key), self.assertRaises(ValueError): public.projection(bad, kind)
        # Malformed raw recorded values must fail at derivation too.
        for value in (True, -1, '12.5'):
            raw = copy.deepcopy(p.evidence['report_receipt']); raw['process_cpu_seconds'] = value
            with self.assertRaises(ValueError): public.project('report_receipt', E(raw))

    def test_report_job_identity_does_not_qualify_execution(self):
        p = Packet(); p.evidence['report_execution']['job_id'] = '34864950'
        # The otherwise complete fabricated packet must not relabel report01.
        with self.assertRaisesRegex(ValueError, 'failed report attempt'): p.construct()
        for job in (SYNTHETIC_REPORT_JOB, '900000002'):
            p = Packet(); p.evidence['report_execution']['job_id'] = job
            p.construct(); result = p.validate()
            self.assertFalse(result['private_execution_revalidated'])
            self.assertEqual(public.decode(p.raw['publication/evidence/report_execution.json'])['values']['/job_id'], job)
            # A well-formed different identity still cannot open production.
            with self.assertRaisesRegex(ValueError, 'schema pending'):
                planner.require_report_acceptance(E({'accepted': True, 'job_id': job}), {})

    def test_complete_downstream_contract_with_private_evidence(self):
        packet = Packet(); packet.construct(); result = packet.validate()
        self.assertEqual((result['inherited_files'], result['inherited_bindings'], result['companions']), (3893, 5068, 16))
        self.assertFalse(result['private_execution_revalidated'])
        self.assertEqual(len(packet.contract['snapshot_members']), 42)
        self.assertEqual(len(packet.contract['private_evidence']), 8)
        self.assertIn('/child/cleanup/private_process', packet.omissions['report_execution'])
        self.assertNotIn(b'/private/', E(packet.contract))
        self.assertTrue(all(b'/private/' not in raw for name, raw in packet.raw.items() if name.startswith('publication/evidence/')))

    def test_production_missing_unaccepted_and_unknown_acceptance_reject_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); request = root/'request.json'; acceptance = root/'acceptance.json'
            data = {'schema': 'delivery-final-release-inputs-v1', 'inputs': {
                n: {'path': str(root/(n+'.json')), 'sha256': H(b'absent')} for n in planner.INPUTS}}
            data['inputs']['manifest']['sha256'] = public.CANDIDATE_SHA
            data['inputs']['plan']['sha256'] = public.PLAN_SHA
            data['inputs']['replay_acceptance']['sha256'] = public.REPLAY_ACCEPTANCE_SHA
            data['inputs']['report_acceptance']['path'] = str(acceptance)
            request.write_bytes(E(data))
            with self.assertRaises(FileNotFoundError): planner.prepare(request, H(E(data)), root/'out')
            for value, message in (({'accepted': False}, 'accepted report'), ({'accepted': True, 'schema': 'invented'}, 'schema pending')):
                acceptance.write_bytes(E(value)); data['inputs']['report_acceptance']['sha256'] = H(E(value)); request.write_bytes(E(data))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                    planner.prepare(request, H(E(data)), root/'out')
            self.assertFalse((root/'out').exists())
            with self.assertRaisesRegex(ValueError, 'pinned input changed'):
                planner.prepare(request, '0'*64, root/'out')

    def test_exact_inherited_entries_bindings_and_notices(self):
        for mutate in (
            lambda p: p.manifest['files'][0].update(original_sha256='0'*64),
            lambda p: p.manifest['files'].remove(next(e for e in p.manifest['files'] if e['path']=='notices/ACE_RUNNER_MIT.txt')),
            lambda p: p.manifest['bindings'][0].update(digest='sha256'),
            lambda p: p.manifest['bindings'].pop(),
        ):
            p = Packet(); p.construct(); mutate(p)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): p.validate()

    def test_nested_projection_rules_and_types_reject_leaks(self):
        p = Packet()
        value = public.project('report_execution', E(p.evidence['report_execution']))
        self.assertNotIn('private_process', E(value).decode())
        for mutate in (
            lambda v: v['values'].update({'/child/cleanup/private_process': '/private/process'}),
            lambda v: v['values'].update({'/child/exit_code': False}),
            lambda v: v['values'].update({'/child/cleanup/complete': {'path': '/private'}}),
            lambda v: v['rules'].update({'/new_fits': 'hash'}),
        ):
            bad = copy.deepcopy(value); mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): public.projection(bad, 'report_execution')
        bad = copy.deepcopy(p.evidence['report_freeze']); bad['source_pins']['/private/source.py'] = H(b'source')
        with self.assertRaises(ValueError): public.project('report_freeze', E(bad))

    def test_report_lineage_versions_counters_and_cleanup(self):
        mutations = [
            ('report_freeze', '/manifest_sha256', '0'*64),
            ('report_receipt', '/original_input_pins/fit_seal.json', '0'*64),
            ('report_receipt', '/counts/cells', 639),
            ('report_execution', '/child/cleanup/complete', False),
            ('report_runtime', '/dependencies/scipy/version', '1.16.2'),
            ('report_execution', '/job_id', 'different-job'),
        ]
        for kind, selector, value in mutations:
            p = Packet(); put(p.evidence[kind], selector, value)
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                p.construct(); p.validate()

    def test_full_report_and_companion_membership(self):
        for name in (public.PAPER+public.reports.PRIMARY_INDEX,
                     public.PAPER+'delivery_prospective_all_cells.csv', public.PAPER+'delivery_theory.tex'):
            p = Packet(); del p.writing[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'complete source snapshot'): p.construct()
        p = Packet(); p.writing[public.PAPER+'delivery_theory.tex'] += b'changed\n'; p.construct()
        with self.assertRaisesRegex(ValueError, '16 companion'): p.validate()

    def test_original_projected_selectors_not_rebased(self):
        p = Packet(); name = public.PAPER+public.reports.PRIMARY_INDEX
        index = public.decode(p.writing[name]); index['acceptance_sha256'] = H(p.inherited['B/acceptance.json'])
        p.writing[name] = E(index)
        p.evidence['report_receipt']['outputs']['primary/claim_index.json'] = H(E(index))
        p.evidence['report_execution']['report_receipt_sha256'] = H(E(p.evidence['report_receipt']))
        p.construct()
        with self.assertRaisesRegex(ValueError, 'historical/projected'): p.validate()

    def test_recorded_analysis_uses_frozen_tolerance_and_schema(self):
        for delta, accepted in ((1e-11, True), (1e-3, False)):
            p = Packet(); name = public.PAPER+public.reports.PRIMARY_INDEX
            record = public.decode(p.writing[name]); record['primary_analysis']['fixture'][0] += delta
            p.writing[name] = E(record)
            p.evidence['report_receipt']['outputs']['primary/claim_index.json'] = H(E(record))
            p.evidence['report_execution']['report_receipt_sha256'] = H(E(p.evidence['report_receipt']))
            p.construct()
            with self.subTest(delta=delta):
                if accepted:
                    p.validate()
                else:
                    with self.assertRaisesRegex(ValueError, 'analysis numeric'): p.validate()
        p = Packet(); p.construct(); path = p.contract['projections']['replay']['path']
        projected = public.decode(p.raw[path]); projected['values']['/primary_analysis']['private_extra'] = '/private/path'
        p.raw[path] = E(projected)
        p.contract['projections']['replay']['sha256'] = p.contract['artifacts'][path] = H(p.raw[path])
        record = next(e for e in p.manifest['files'] if e['path'] == path)
        record.update(sha256=H(p.raw[path]), original_sha256=H(p.raw[path]), bytes=len(p.raw[path]))
        # Even mutually updated public hashes cannot extend the frozen science
        # tree's shape to carry new private metadata.
        with self.assertRaisesRegex(ValueError, 'analysis keys'): p.validate()

    def test_private_source_scope_and_authoritative_contract(self):
        for mutate in (
            lambda p: p.contract.update(authoritative_path='reproduction/source_contract.json'),
            lambda p: p.contract.update(training_reproduction_available=True),
            lambda p: p.contract['private_evidence']['report_acceptance'].update(path='/private/acceptance.json'),
            lambda p: p.contract['original_source_bindings'].pop(),
            lambda p: p.contract['source_usage'].update({'invented.py': 'executable'}),
        ):
            p = Packet(); p.construct(); mutate(p)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): p.validate()

    def test_plan_preserves_original_projection_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); original = root/'original.json'; original.write_bytes(E({'public': 1, 'private': '/private/path'}))
            plain = root/'identity'; plain.write_bytes(b'x')
            derived = E({'public': 1}); special = entry('A/protocol.json', derived)
            special.update(original_sha256=H(original.read_bytes()), transform={'kind': 'project-json', 'keys': ['public']})
            manifest = {'files': [special]+[entry(f'dummy/{i}', b'x') for i in range(3892)], 'bindings': [{}]*5068}
            plan = {'files': [{'path': e['path'], 'role': e['role'], 'transform': e['transform'],
                'original_sha256': e['original_sha256'], 'sources': [str(original if i==0 else plain)]}
                for i, e in enumerate(manifest['files'])], 'bindings': manifest['bindings']}
            planner.inherited_plan(plan, manifest)
            plan['files'][0].update(original_sha256=H(derived), transform={'kind': 'identity'})
            with self.assertRaisesRegex(ValueError, 'transformation semantics'): planner.inherited_plan(plan, manifest)


if __name__ == '__main__':
    unittest.main()
