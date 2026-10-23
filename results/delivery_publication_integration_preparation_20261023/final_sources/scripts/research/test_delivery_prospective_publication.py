"""Artificial report/proof packets only; no actual B scores or model imports."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivery_prospective_publication as reporting
import sync_delivery_manuscript as sync
import prepare_delivery_publication_snapshot as snapshot


def encoded(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.primary = self.root/'primary';self.primary.mkdir()
        self.descriptive = self.root/'descriptive';self.descriptive.mkdir()
        self.proofs = self.root/'proofs';self.proofs.mkdir()
        self.destination = self.root/'assembled'
        self.proof_paths = {n: self.proofs/(n+'.json') for n in reporting.PROOFS}
        self.expected = {n: reporting.digest(('fixture '+n).encode()) for n in reporting.EXPECTED_KEYS}
        repo = Path(__file__).resolve().parents[2]
        self.exporter = (repo/snapshot.PROSPECTIVE_SOURCES[0]).read_bytes()
        self.reporter = (repo/snapshot.PROSPECTIVE_SOURCES[1]).read_bytes()
        self.expected.update(exporter=reporting.digest(self.exporter), reporter=reporting.digest(self.reporter))
        self.freeze = {'manifest_sha256': self.expected['manifest'], 'exact_dependencies': reporting.DEPENDENCIES,
            'new_fits': 0, 'new_responses': 0, 'package_files': 3, 'package_bindings': 5068,
            'base_python_sha256': 'a'*64}
        self.replay = {'full_supplemental_replay': True, 'runtime': reporting.DEPENDENCIES,
            'checkpoints_replayed': 640, 'primary_cells_recomputed': 240, 'cached_responses_checked': 48000,
            'new_optimizer_updates': 0, 'new_responses': 0,
            'integrity': {'expected_digest_supplied': True, 'manifest_sha256': self.expected['manifest'],
                          'files_verified': 3, 'bindings_verified': 5068},
            'primary_analysis': {'artificial': 1.0}, 'secondary_descriptive_analysis': {'artificial': 2.0}}
        self.execution = {'status': 'complete', 'stage': 'qualified', 'exit_code': 0,
            'manifest_sha256': self.expected['manifest'], 'new_fits': 0, 'new_responses': 0,
            'child_execution': {'status': 'complete', 'exit_code': 0, 'cleanup': {'complete': True}},
            'authenticated_runtime': {'runtime_qualification_sha256': 'b'*64, 'runtime_inventory_sha256': 'c'*64,
                'pre_checkpoint_runtime_sha256': 'd'*64, 'python_sha256': 'a'*64, 'environment_objects_verified': 10}}
        for n in reporting.PRIMARY:
            (self.primary/n).write_text('% artificial report '+n+'\n')
        for n in reporting.DESCRIPTIVE+reporting.DATA:
            (self.descriptive/n).write_text('% artificial report '+n+'\n')
        self.primary_index = {'output_hashes': {}, 'exporter_sha256': self.expected['exporter'],
            'registration_sha256': reporting.REGISTRATION, 'source_revision': reporting.REVISION,
            'acceptance_sha256': 'e'*64, 'scores_sha256': 'f'*64, 'deployment_initialization': 0,
            'supervision_and_compute_matched': False, 'primary_analysis': self.replay['primary_analysis'],
            'secondary_descriptive_analysis': self.replay['secondary_descriptive_analysis']}
        self.provenance = {'manifest_sha256': self.expected['manifest'], 'tool_sha256': self.expected['reporter'],
            'acceptance_sha256': '1'*64, 'scores_sha256': '2'*64, 'original_registration_sha256': reporting.REGISTRATION}
        self.manifest = {'schema': 'delivery-anonymous-derived-v1', 'files': [
            {'path': 'B/acceptance.json', 'original_sha256': '1'*64, 'sha256': '1'*64, 'transform': {'kind': 'identity'}},
            {'path': 'B/scores.json', 'original_sha256': '2'*64, 'sha256': '2'*64, 'transform': {'kind': 'identity'}},
            {'path': 'B/replay_contract.json', 'original_sha256': '', 'sha256': '', 'transform': {'kind': 'identity'}}]}
        # Actual planner: identity entries hold already projected bytes; only
        # the replay contract connects historical and derived receipt hashes.
        self.contract = {'schema': 'delivery-prospective-replay-v1', 'registration_sha256': reporting.REGISTRATION,
            'original_revision': reporting.REVISION, 'dependencies': reporting.DEPENDENCIES,
            'matrix_fits': 640, 'primary_cells': 240, 'original_acceptance_sha256': 'e'*64,
            'original_scores_sha256': 'f'*64, 'metadata_projections': {
                'B/acceptance.json': {'original_path': 'B/acceptance.json', 'original_sha256': 'e'*64, 'derived_sha256': '1'*64},
                'B/scores.json': {'original_path': 'B/scores.json', 'original_sha256': 'f'*64, 'derived_sha256': '2'*64}}}
        self.descriptive_index = {'provenance': self.provenance, 'output_hashes': {}, 'actual_numerical_replay_performed_here': False}
        self.supplement = {n: [None]*count for n,count in {'cells': 640, 'fixed_init0': 320,
            'system_log_ratios': 80, 'history_specific_ratios': 8, 'short_fit_ablation': 2, 'primary_floor_counts': 4}.items()}
        self.supplement.update(new_fits=0, new_responses=0, additional_significance_tests=0, provenance=self.provenance)
        self.refresh()

    def refresh(self):
        raw=encoded(self.contract);self.proof_paths['contract'].write_bytes(raw);self.expected['contract']=reporting.digest(raw)
        self.manifest['files'][2].update(original_sha256=self.expected['contract'],sha256=self.expected['contract'])
        raw=encoded(self.manifest);self.proof_paths['manifest'].write_bytes(raw);self.expected['manifest']=reporting.digest(raw)
        self.freeze['manifest_sha256']=self.expected['manifest'];self.execution['manifest_sha256']=self.expected['manifest']
        self.replay['integrity']['manifest_sha256']=self.expected['manifest'];self.provenance['manifest_sha256']=self.expected['manifest']
        for n in ('freeze', 'replay'):
            raw=encoded(getattr(self,n));self.proof_paths[n].write_bytes(raw);self.expected[n]=reporting.digest(raw)
        self.execution.update(freeze_sha256=self.expected['freeze'], replay_receipt_sha256=self.expected['replay'])
        raw=encoded(self.execution);self.proof_paths['execution'].write_bytes(raw);self.expected['execution']=reporting.digest(raw)
        self.provenance['supplemental_replay_sha256']=self.expected['replay']
        (self.descriptive/'supplement.json').write_bytes(encoded(self.supplement))
        self.primary_index['output_hashes']={n: reporting.digest((self.primary/n).read_bytes()) for n in reporting.PRIMARY}
        self.descriptive_index['output_hashes']={n: reporting.digest((self.descriptive/n).read_bytes()) for n in reporting.DESCRIPTIVE+reporting.DATA}
        for name,folder,index,filename in [('primary_index',self.primary,self.primary_index,'claim_index.json'),
                                         ('descriptive_index',self.descriptive,self.descriptive_index,'output_index.json')]:
            raw=encoded(index);(folder/filename).write_bytes(raw);self.expected[name]=reporting.digest(raw)

    def assemble(self):
        return reporting.prepare(self.primary,self.descriptive,self.proof_paths,self.expected,self.destination)

    def test_complete_namespace_and_exclusive_output(self):
        prior=(self.primary/'claim_index.json').read_bytes();result=self.assemble()
        self.assertEqual(result['files'],19);self.assertEqual(result['companions'],5)
        self.assertEqual(set(p.name for p in self.destination.iterdir()),set(reporting.FILES))
        self.assertEqual((self.destination/reporting.PRIMARY_INDEX).read_bytes(),prior)
        self.assertEqual((self.primary/'claim_index.json').read_bytes(),prior)
        self.assertFalse((self.destination/'claim_index.json').exists())
        self.assertFalse(result['actual_execution_validated_here'])
        reporting.read_bundle(self.destination,result['bundle_sha256'])
        with self.assertRaises(FileExistsError):self.assemble()

    def test_missing_pin_symlink_and_nested_destination_reject_before_write(self):
        valid=dict(self.expected);self.expected['replay']='0'*64
        with self.assertRaises(ValueError):self.assemble()
        self.expected=valid
        original=self.destination;self.destination=self.primary/'nested'
        with self.assertRaisesRegex(ValueError,'outside'):self.assemble()
        self.destination=original
        source=self.primary/reporting.PRIMARY[0];copy=self.root/'report';copy.write_bytes(source.read_bytes())
        source.unlink();source.symlink_to(copy)
        with self.assertRaisesRegex(ValueError,'symlink'):self.assemble()
        self.assertFalse(self.destination.exists())

    def test_failed_or_partial_replay_rejects_before_index_decode(self):
        for key,value in [('full_supplemental_replay',False),('checkpoints_replayed',639),
                          ('primary_cells_recomputed',True),('new_optimizer_updates',1),('new_responses',1)]:
            original=self.replay[key];self.replay[key]=value;self.refresh()
            (self.primary/'claim_index.json').write_bytes(b'NOT JSON')
            self.expected['primary_index']=reporting.digest(b'NOT JSON')
            with self.subTest(key=key),self.assertRaises(ValueError):self.assemble()
            self.assertFalse(self.destination.exists());self.replay[key]=original
        self.refresh();self.execution['child_execution']['cleanup']['complete']=False;self.refresh()
        with self.assertRaisesRegex(ValueError,'cleanup'):self.assemble()
        self.assertFalse(self.destination.exists())

    def test_source_cross_report_and_analysis_lineage_rejected(self):
        for obj,key,value in [(self.primary_index,'exporter_sha256','0'*64),
                              (self.primary_index,'acceptance_sha256','0'*64),
                              (self.primary_index,'deployment_initialization',1),
                              (self.primary_index,'primary_analysis',{'artificial':9.0})]:
            original=obj[key];obj[key]=value;self.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):self.assemble()
            self.assertFalse(self.destination.exists());obj[key]=original
        self.supplement['fixed_init0'].pop();self.refresh()
        with self.assertRaisesRegex(ValueError,'matrix length'):self.assemble()

    def base_writing(self,root):
        root.mkdir(exist_ok=True)
        root.chmod(0o700)
        for n in sync.FILES:(root/n).write_text('% artificial base '+n+'\n')
        (root/'tmlr_style_provenance.json').write_bytes(encoded({'files': {}}))

    def test_sync_requires_pin_and_embeds_verified_captured_bytes(self):
        result=self.assemble();self.base_writing(self.destination)
        with patch.object(sync,'ROOT',self.destination):
            with self.assertRaisesRegex(ValueError,'explicit'):sync.bundled()
            original=reporting.capture_bundle
            def mutate_after_capture(root,pin):
                raw=original(root,pin)
                (root/reporting.PRIMARY[0]).chmod(0o600)
                (root/reporting.PRIMARY[0]).write_text('% changed after capture\n')
                return raw
            with patch.object(reporting,'capture_bundle',mutate_after_capture):
                text=sync.bundled(result['bundle_sha256'])
            self.assertEqual(text.count('\\begin{filecontents*}'),16)
            self.assertIn('% artificial report '+reporting.PRIMARY[0],text)
            self.assertNotIn('changed after capture',text)
            with self.assertRaisesRegex(ValueError,'changed'):sync.bundled(result['bundle_sha256'])

    def test_base_mode_retains_eleven_companions(self):
        self.base_writing(self.destination)
        with patch.object(sync,'ROOT',self.destination):
            self.assertEqual(sync.bundled().count('\\begin{filecontents*}'),11)

    def test_committed_capture_includes_all_reports_and_rejects_silent_omission(self):
        result=self.assemble();pin=result['bundle_sha256'];repo=self.root/'git';repo.mkdir()
        raw={n: ('% artificial '+n+'\n').encode() for n in snapshot.FILES}
        raw[snapshot.SOURCES[-1]]=Path(reporting.__file__).read_bytes()
        raw[snapshot.SOURCES[0]]=('FILES = '+repr(snapshot.COMPANIONS)+'\n').encode()
        styles=('tmlr.sty','fancyhdr.sty','tmlr.bst','TMLR_STYLE_LICENSE','archive/tmlr_official_template.tex')
        raw[snapshot.PAPER+'tmlr_style_provenance.json']=encoded({'files':{n:{'sha256':reporting.digest(raw[snapshot.PAPER+n])} for n in styles}})
        raw[snapshot.PAPER+'claim_index.json']=encoded({'generator_sha256':reporting.digest(raw[snapshot.SOURCES[1]])})
        raw.update({snapshot.PAPER+n:(self.destination/n).read_bytes() for n in reporting.FILES})
        raw[snapshot.PROSPECTIVE_SOURCES[0]]=self.exporter;raw[snapshot.PROSPECTIVE_SOURCES[1]]=self.reporter
        chunks=[snapshot.BEGIN]
        for n in snapshot.COMPANIONS+reporting.COMPANIONS:
            b=raw[snapshot.PAPER+n]
            chunks += [f'% {n}: SHA256 {reporting.digest(b)}',f'\\begin{{filecontents*}}[overwrite]{{{n}}}',b.decode().rstrip('\n'),'\\end{filecontents*}']
        raw[snapshot.PAPER+'paper.tex']=('\n'.join(chunks+[snapshot.END])+'\n').encode()
        for n,b in raw.items():
            path=repo/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE).decode().strip()
        git('init','-q');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
        git('add','.');git('commit','-qm','fabricated reports');revision=git('rev-parse','HEAD')
        (repo/snapshot.PAPER/'paper.tex').write_text('dirty checkout')
        captured,blobs=snapshot.capture(repo,revision,pin);snapshot.check_bundle(captured,pin)
        self.assertEqual(len(captured),42);self.assertEqual(captured,raw)
        with self.assertRaisesRegex(ValueError,'explicit'):snapshot.capture(repo,revision)
        captured[snapshot.PROSPECTIVE_SOURCES[1]]=b'altered reporter'
        with self.assertRaisesRegex(ValueError,'reporter source'):snapshot.check_bundle(captured,pin)

    def test_review_derivation_mapping_and_captured_helper_rejection(self):
        # The valid fixture deliberately has DIFFERENT original and released hashes.
        self.assertNotEqual(self.primary_index['scores_sha256'],self.provenance['scores_sha256'])
        self.contract['metadata_projections']['B/scores.json']['original_sha256']='0'*64;self.refresh()
        with self.assertRaisesRegex(ValueError,'original/derived'):self.assemble()
        self.contract['metadata_projections']['B/scores.json']['original_sha256']='f'*64
        self.manifest['files'][1]['transform']['kind']='project-json';self.refresh()
        with self.assertRaisesRegex(ValueError,'identity entry'):self.assemble()
        with self.assertRaisesRegex(ValueError,'helper'):
            snapshot.check_bundle({snapshot.SOURCES[-1]:b'from another helper version'})

    def test_contract_pin_projection_and_manifest_disagreement_reject(self):
        for obj,key,value in [
                (self.contract['metadata_projections']['B/scores.json'],'original_path','wrong'),
                (self.contract['metadata_projections']['B/scores.json'],'derived_sha256','0'*64),
                (self.contract,'original_acceptance_sha256','0'*64),
                (self.contract,'matrix_fits',639)]:
            original=obj[key];obj[key]=value;self.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):self.assemble()
            self.assertFalse(self.destination.exists());obj[key]=original
        self.refresh();self.expected['contract']='0'*64
        with self.assertRaisesRegex(ValueError,'independent input pin'):self.assemble()
        self.refresh()
        self.manifest['files'][2]['sha256']='0'*64
        raw=encoded(self.manifest);self.proof_paths['manifest'].write_bytes(raw);self.expected['manifest']=reporting.digest(raw)
        self.freeze['manifest_sha256']=self.expected['manifest'];self.execution['manifest_sha256']=self.expected['manifest']
        self.replay['integrity']['manifest_sha256']=self.expected['manifest']
        for n in ('freeze','replay','execution'):
            if n=='execution':self.execution.update(freeze_sha256=self.expected['freeze'],replay_receipt_sha256=self.expected['replay'])
            raw=encoded(getattr(self,n));self.proof_paths[n].write_bytes(raw);self.expected[n]=reporting.digest(raw)
        with self.assertRaisesRegex(ValueError,'manifest-bound replay contract'):self.assemble()

    def test_review_output_ancestor_swap_is_not_followed(self):
        parent=self.root/'movable';parent.mkdir();victim=self.root/'victim';victim.mkdir()
        self.destination=parent/'out';original=reporting.validate
        def swap_after_validation(*args):
            result=original(*args)
            parent.rename(self.root/'held');parent.symlink_to(victim,target_is_directory=True)
            return result
        with patch.object(reporting,'validate',swap_after_validation):
            with self.assertRaises((ValueError,OSError)):self.assemble()
        self.assertFalse((victim/'out').exists());self.assertFalse((self.root/'held'/'out').exists())


if __name__=='__main__':unittest.main(verbosity=2)
