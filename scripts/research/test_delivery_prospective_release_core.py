"""Pre-score rejection and exact extracted numerical-body checks; no B outcomes."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_delivery_prospective_release_core as core


def write(p, value):
    p.write_text(json.dumps(value))


class ReleaseCoreTests(unittest.TestCase):
    def fixture(self, root):
        auditor = 'a'*64; cached_scores = 'b'*64
        reg = {'source_revision': core.REVISION, 'dependencies': core.DEPENDENCIES,
               'matrix_fits': 640, 'primary_cells': 240, 'scope': 'fixture only',
               'resources': {'rss_bytes': 1024, 'audit_wall_seconds': 900},
               'worker_hashes': {'audit_delivery_prospective_results.py': auditor}}
        write(root/'registration.json',reg); rh = core.sha(root/'registration.json')
        write(root/'fit_seal.json', {'fixture': True})
        complete = {'n_fits': 640, 'n_primary_cells': 240, 'charged_responses': 48000,
                    'account': 'ucb736_asc1', 'registration_sha256': rh,
                    'fit_seal_sha256': core.sha(root/'fit_seal.json'), 'scores_sha256': cached_scores}
        write(root/'complete.json',complete)
        accepted = {'full_acceptance': True, 'fits_checked': 640, 'checkpoints_replayed': 640,
                    'primary_cells_checked': 240, 'charged_cached_responses': 48000,
                    'new_simulator_responses': 0, 'source_revision': core.REVISION,
                    'runtime': core.DEPENDENCIES, 'study_registration_sha256': rh,
                    'complete_sha256': core.sha(root/'complete.json'),
                    'fit_seal_sha256': complete['fit_seal_sha256'], 'scores_sha256': cached_scores,
                    'auditor_sha256': auditor, 'scope': 'fixture only', 'at':'2026-10-07T00:00:00+00:00',
                    'primary_analysis':{'sentinel':1900180066.125},
                    'secondary_descriptive_analysis':{'nested':[1900180066.125,'unopened-scientific-string']},
                    'replay_tolerance':{'rtol':1e-6,'atol':1e-7,'roots':'exact'},
                    'replay_max_abs_deltas':{str(i):0. for i in range(640)},
                    'audit_cpu_seconds':1.,'audit_wall_seconds':2.,
                    'scoring_predictions':'unchanged original cached arrays',
                    'phase_execution_hashes':{n:'c'*64 for n in ('qualification_execution.json','collect_execution.json','evaluate_execution.json')}}
        worlds={f'{size}:{i:02d}' for size in (5,30) for i in range(20)}
        files={n:'d'*64 for n in ('input.json','queries.ndjson','receipt.json')}
        accepted['training_receipt_hashes']={c+'/'+h:files for c in worlds for h in ('balanced_varied_value','matched_random')}
        accepted['evaluation_receipt_hashes']={c:files for c in worlds}
        write(root/'acceptance.json', accepted)
        supervisor = {'status': 'complete', 'exit_code': 0, 'account': 'ucb736_asc1',
                      'registration_sha256': rh, 'acceptance_sha256': core.sha(root/'acceptance.json'),
                      'peak_tree_rss_bytes': 512, 'elapsed_seconds': 20}
        write(root/'audit_execution.json', supervisor)
        return rh, supervisor, accepted

    def test_successful_upstream_gate_does_not_open_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rh, _, _ = self.fixture(root)
            original_decoder=core.json.JSONDecoder
            class Decoder(original_decoder):
                def raw_decode(self,s,idx=0):
                    if s[idx:].lstrip().startswith(('1900180066','"unopened-scientific-string')):
                        raise AssertionError('scientific outcome value decoded')
                    return super().raw_decode(s,idx)
            with patch.object(core.json,'JSONDecoder',Decoder):
                result = core.accepted_before_scores(root,rh)
            self.assertFalse(result['outcomes_opened'])
            self.assertFalse((root/'scores.json').exists())
            projected=result['custody_bound_acceptance_projection']
            self.assertFalse(projected['scientific_outcome_values_decoded'])
            self.assertNotIn('primary_analysis',projected['metadata'])

    def test_failed_supervisor_never_parses_acceptance_or_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rh, supervisor, _ = self.fixture(root)
            supervisor['exit_code'] = 1; write(root/'audit_execution.json',supervisor)
            opened = []; original = core.read
            def read(path):
                opened.append(Path(path).name); return original(path)
            with patch.object(core,'read',side_effect=read):
                with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)
            self.assertNotIn('acceptance.json',opened); self.assertNotIn('scores.json',opened)

    def test_changed_acceptance_hash_rejects_before_parsing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rh, _, accepted = self.fixture(root)
            accepted['scope']='drift'; write(root/'acceptance.json',accepted)
            with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)

    def test_incomplete_acceptance_and_stop_flag_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rh, supervisor, accepted = self.fixture(root)
            accepted['checkpoints_replayed']=560; write(root/'acceptance.json',accepted)
            supervisor['acceptance_sha256']=core.sha(root/'acceptance.json')
            write(root/'audit_execution.json',supervisor)
            with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)
            rh, _, _ = self.fixture(root); (root/'stop_new_fits.json').write_text('{}')
            with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)

    def test_missing_completion_and_wrong_runtime_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rh, _, _ = self.fixture(root)
            (root/'complete.json').unlink()
            with self.assertRaises(FileNotFoundError): core.accepted_before_scores(root,rh)
        with patch.object(core.importlib.metadata,'version',return_value='other-runtime'):
            with self.assertRaises(ValueError): core.runtime_gate()

    def test_malformed_supervisor_measurements_reject_before_projection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for field, value in [('exit_code',False),('peak_tree_rss_bytes',True),
                                 ('peak_tree_rss_bytes',-1),('elapsed_seconds',float('nan')),
                                 ('elapsed_seconds',float('inf')),('elapsed_seconds',-1),
                                 ('elapsed_seconds',False)]:
                rh, supervisor, _=self.fixture(root); supervisor[field]=value
                write(root/'audit_execution.json',supervisor)
                with patch.object(core,'acceptance_projection',side_effect=AssertionError('too early')):
                    with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)

    def test_omitted_evidence_bad_digest_and_replay_indices_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for failure in ('missing','digest','indices','nonfinite','bool-count'):
                rh, supervisor, accepted=self.fixture(root)
                if failure=='missing': del accepted['training_receipt_hashes']
                if failure=='digest': accepted['phase_execution_hashes']['collect_execution.json']='not-a-digest'
                if failure=='indices': del accepted['replay_max_abs_deltas']['639']
                if failure=='nonfinite': accepted['audit_cpu_seconds']=float('nan')
                if failure=='bool-count': accepted['new_simulator_responses']=False
                write(root/'acceptance.json',accepted)
                supervisor['acceptance_sha256']=core.sha(root/'acceptance.json')
                write(root/'audit_execution.json',supervisor)
                with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)

    def test_stop_flag_and_malformed_skipped_analysis_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rh, _, _=self.fixture(root)
            (root/'stop_new_fits.json').write_text('{}')
            with patch.object(core,'acceptance_projection',side_effect=AssertionError('too early')):
                with self.assertRaises(ValueError): core.accepted_before_scores(root,rh)
            (root/'stop_new_fits.json').unlink()
            p=root/'acceptance.json'; text=p.read_text().replace('1900180066.125','NaN')
            p.write_text(text)
            with self.assertRaises(ValueError): core.acceptance_projection(p,core.sha(p))

    def test_projection_hash_and_metadata_use_one_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); _, _, accepted=self.fixture(root)
            p=root/'acceptance.json'; old_hash=core.sha(p); original_read=Path.read_bytes
            accepted['full_acceptance']=False
            def replace_after_read(path):
                raw=original_read(path)
                if path==p: write(p,accepted)
                return raw
            with patch.object(Path,'read_bytes',replace_after_read):
                projected=core.acceptance_projection(p,old_hash)
            self.assertTrue(projected['metadata']['full_acceptance'])
            self.assertEqual(projected['original_acceptance_sha256'],old_hash)
            self.assertNotEqual(projected['original_acceptance_sha256'],core.sha(p))
            with self.assertRaises(ValueError): core.acceptance_projection(p,old_hash)

    def test_nonascii_number_in_skipped_analysis_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); self.fixture(root); p=root/'acceptance.json'
            p.write_text(p.read_text().replace('1900180066.125','1900180066.١٢٥'))
            with self.assertRaises(ValueError): core.acceptance_projection(p,core.sha(p))

    def test_exact_function_bodies_and_analytic_primary_statistics(self):
        here=Path(__file__).resolve().parent
        auditor=here/'audit_delivery_prospective_results.py'; guard=here/'runner_delivery_confirmation.py'
        code, segments=core.extracted_core(auditor,core.sha(auditor),guard,core.sha(guard))
        self.assertEqual(set(segments),set(core.FUNCTIONS+core.GLOBALS+('read','sha','journal_count')))
        self.assertNotIn('def audit(',code)
        module={}; exec(compile(code,'derived-test-core','exec'),module)
        tree=ast.parse(code)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in segments:
                digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest,segments[node.name]['ast_sha256'])
        rows=[]
        for case in module['WORLD_IDS']:
            for history in module['HISTORIES']:
                for arm, value in [('delivery',1.),('online',4.),('simpler',2.)]:
                    rows.append({'graph_size':int(case.split(':')[0]),'system_id':case,
                                 'history':history,'arm':arm,'init':0,'nmse':value,'status':'complete'})
        result=module['primary_statistics'](rows)
        self.assertEqual(result['n_primary_cells'],240)
        for c in result['contrasts']:
            self.assertAlmostEqual(c['ratio'],.25 if c['control']=='online' else .5)
            self.assertTrue(c['superiority'])
        with self.assertRaises(ValueError): module['primary_statistics'](rows[:-1])

    def test_changed_original_source_rejects_extraction(self):
        here=Path(__file__).resolve().parent
        with self.assertRaises(ValueError):
            core.extracted_core(here/'audit_delivery_prospective_results.py','different',
                                here/'runner_delivery_confirmation.py',core.sha(here/'runner_delivery_confirmation.py'))

    def test_source_extraction_hashes_and_parses_same_snapshot(self):
        here=Path(__file__).resolve().parent; original_read=Path.read_bytes
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'auditor.py'; p.write_bytes((here/'audit_delivery_prospective_results.py').read_bytes())
            expected=core.sha(p); guard=here/'runner_delivery_confirmation.py'
            def mutate_after_read(path):
                raw=original_read(path)
                if path==p: p.write_text('raise RuntimeError("changed")')
                return raw
            with patch.object(Path,'read_bytes',mutate_after_read):
                code,segments=core.extracted_core(p,expected,guard,core.sha(guard))
            self.assertIn('def replay_predictions(',code)
            self.assertNotIn('raise RuntimeError("changed")',code)
            self.assertNotEqual(core.sha(p),expected)


if __name__=='__main__': unittest.main()
