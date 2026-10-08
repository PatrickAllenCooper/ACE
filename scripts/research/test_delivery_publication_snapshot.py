"""Fabricated Git fixtures only; no numerical or historical-worker execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_delivery_publication_snapshot as snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'Fixture')
        self.run_git('config', 'user.email', 'fixture@example.invalid')
        self.raw = {n: ('fixture ' + n + '\n').encode() for n in snapshot.FILES}
        self.raw[snapshot.SOURCES[0]] = ('FILES = ' + repr(snapshot.COMPANIONS) + '\n').encode()
        styles = ('tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'TMLR_STYLE_LICENSE',
                  'archive/tmlr_official_template.tex')
        self.raw[snapshot.PAPER + 'tmlr_style_provenance.json'] = json.dumps({
            'files': {n: {'sha256': snapshot.digest(self.raw[snapshot.PAPER+n])}
                      for n in styles}}).encode()
        self.raw[snapshot.PAPER + 'claim_index.json'] = json.dumps({
            'generator_sha256': snapshot.digest(self.raw[snapshot.SOURCES[1]])}).encode()
        chunks = [snapshot.BEGIN]
        for n in snapshot.COMPANIONS:
            raw = self.raw[snapshot.PAPER+n]
            chunks += [f'% {n}: SHA256 {snapshot.digest(raw)}',
                       f'\\begin{{filecontents*}}[overwrite]{{{n}}}',
                       raw.decode().rstrip('\n'), '\\end{filecontents*}']
        self.raw[snapshot.PAPER+'paper.tex'] = (
            '\n'.join(chunks + [snapshot.END]) + '\nactive manuscript\n').encode()
        for name, raw in self.raw.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.commit = self.commit_all()

    def run_git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True,
                              capture_output=True).stdout.decode().strip()

    def commit_all(self):
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'fabricated fixture')
        return self.run_git('rev-parse', 'HEAD')

    def test_complete_committed_bytes_ignore_dirty_checkout_and_new_head(self):
        (self.repo / snapshot.PAPER / 'paper.tex').write_text('dirty and unsynchronized')
        self.commit_all()
        out = self.root / 'snapshot'
        receipt = snapshot.prepare(self.repo, self.commit, out)
        inventory = json.loads((out/'inventory.json').read_bytes())
        # Independent fixed published-path oracle, including required style notice.
        expected = {'paper.tex','tmlr.sty','fancyhdr.sty','tmlr.bst','delivery_references.bib',
                    'delivery_claims.tex','delivery_attribution_table.tex','delivery_history_table.tex',
                    'delivery_physical_table.tex','delivery_theory.tex','delivery_method_diagram.tex',
                    'delivery_evaluation_design.tex','claim_index.json','tmlr_style_provenance.json',
                    'TMLR_STYLE_LICENSE','archive/tmlr_official_template.tex'}
        expected = {snapshot.PAPER+n for n in expected} | {
            'scripts/research/sync_delivery_manuscript.py',
            'scripts/research/generate_delivery_claims.py',
            'scripts/research/verify_delivery_table_data.py',
            'docs/development/guidance/delivery_reviewer_commands_2026-10-08.md'}
        self.assertEqual({f['path'] for f in inventory['files']}, expected)
        self.assertEqual(len(inventory['files']), 20)
        self.assertEqual(receipt['inventory_sha256'], snapshot.digest((out/'inventory.json').read_bytes()))
        for row in inventory['files']:
            raw = (out/'source'/row['path']).read_bytes()
            self.assertEqual(raw, self.raw[row['path']])
            self.assertEqual(row['sha256'], snapshot.digest(raw))
            self.assertEqual(row['bytes'],len(raw))
        for field in ('scientific_inputs_included', 'entry_points_executed',
                      'prospective_report_qualification_performed', 'anonymous_release_approved', 'submission_approved'):
            self.assertIs(inventory[field], False)
        before = (out/'inventory.json').read_bytes()
        with self.assertRaises(ValueError):
            snapshot.prepare(self.repo, self.commit, out)
        self.assertEqual((out/'inventory.json').read_bytes(), before)

    def test_invalid_pin_missing_file_and_git_symlink_no_outputs(self):
        out = self.root/'rejected'
        for pin in ('HEAD', self.commit[:8], self.commit.upper(), '0'*40):
            with self.subTest(pin=pin), self.assertRaises((ValueError,subprocess.CalledProcessError)):
                snapshot.prepare(self.repo, pin, out)
            self.assertFalse(out.exists())
        path = self.repo/snapshot.PAPER/'delivery_method_diagram.tex'
        path.unlink()
        missing = self.commit_all()
        with self.assertRaises(ValueError): snapshot.prepare(self.repo,missing,out)
        self.assertFalse(out.exists())
        path.symlink_to('delivery_theory.tex')
        alias = self.commit_all()
        with self.assertRaises(ValueError): snapshot.prepare(self.repo,alias,out)
        self.assertFalse(out.exists())

    def test_external_exclusive_output_and_symlink_ancestors(self):
        for out in (self.repo/'new',self.repo,self.repo/'paper'):
            with self.subTest(path=str(out)), self.assertRaises(ValueError):
                snapshot.prepare(self.repo,self.commit,out)
        alias=self.root/'alias'; alias.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError): snapshot.prepare(self.repo,self.commit,alias/'new')
        self.assertFalse((self.root/'new').exists())
        with self.assertRaises(ValueError):
            snapshot.prepare(self.repo/'paper',self.commit,self.root/'nested-root')

    def test_changed_bundle_membership_source_notice_and_duplicate_metadata(self):
        mutations = [
            (snapshot.PAPER+'delivery_theory.tex', b'changed unembedded theory'),
            (snapshot.SOURCES[0], b'FILES = ()'),
            (snapshot.PAPER+'TMLR_STYLE_LICENSE', b'changed required notice'),
            (snapshot.SOURCES[1], b'changed generator'),
            (snapshot.PAPER+'claim_index.json', b'{"generator_sha256":"a","generator_sha256":"b"}'),
            (snapshot.PAPER+'paper.tex', self.raw[snapshot.PAPER+'paper.tex']+snapshot.BEGIN.encode())]
        for name,value in mutations:
            with self.subTest(name=name):
                raw=copy.deepcopy(self.raw);raw[name]=value
                with self.assertRaises(ValueError): snapshot.check_bundle(raw)

    def test_git_commit_and_blob_replacements_cannot_override_pin(self):
        before = self.run_git('rev-parse',self.commit+':'+snapshot.PAPER+'delivery_theory.tex')
        path = self.repo/snapshot.PAPER/'paper.tex'
        path.write_text('replacement has incoherent source')
        new = self.commit_all()
        self.run_git('replace',self.commit,new)
        # Also replace an individual original blob with unrelated source bytes.
        replacement = self.run_git('rev-parse',new+':'+snapshot.PAPER+'paper.tex')
        self.run_git('replace',before,replacement)
        out=self.root/'replacement-proof'
        snapshot.prepare(self.repo,self.commit,out)
        for name in snapshot.FILES:
            self.assertEqual((out/'source'/name).read_bytes(),self.raw[name])

    def test_ancestor_symlink_swap_after_preflight_cannot_redirect_writes(self):
        parent=self.root/'outputs';parent.mkdir()
        captured=snapshot.capture
        def swapped(*args):
            result=captured(*args)
            parent.rename(self.root/'old-outputs')
            parent.symlink_to(self.repo,target_is_directory=True)
            return result
        with patch.object(snapshot,'capture',side_effect=swapped), self.assertRaises(OSError):
            snapshot.prepare(self.repo,self.commit,parent/'new-snapshot')
        self.assertFalse((self.repo/'new-snapshot').exists())
        self.assertFalse((self.root/'old-outputs/new-snapshot').exists())

    def test_private_read_only_snapshot_under_permissive_umask(self):
        old=os.umask(0)
        try:
            out=self.root/'private-snapshot'
            snapshot.prepare(self.repo,self.commit,out)
        finally:
            os.umask(old)
        for path in (out,*out.rglob('*')):
            self.assertEqual(path.stat().st_mode & 0o777,0o500 if path.is_dir() else 0o400)

    def test_untrusted_shared_writable_parent_and_corrupt_tree_reject(self):
        parent=self.root/'shared';parent.mkdir();parent.chmod(0o777)
        with self.assertRaisesRegex(ValueError,'trusted'):
            snapshot.prepare(self.repo,self.commit,parent/'new')
        self.assertFalse((parent/'new').exists())
        original=snapshot.git
        def corrupted(repo,*args):
            raw=original(repo,*args)
            if args[:2]==('cat-file','tree'):
                return raw[:-1]+bytes([raw[-1]^1])
            return raw
        with patch.object(snapshot,'git',side_effect=corrupted), self.assertRaisesRegex(ValueError,'tree bytes'):
            snapshot.prepare(self.repo,self.commit,self.root/'corrupt')
        self.assertFalse((self.root/'corrupt').exists())

    def test_real_mac_acl_allow_and_inheritance_reject_deny_only_accepts(self):
        parent=self.root/'acl-parent';parent.mkdir(mode=0o700)
        self.addCleanup(lambda: subprocess.run(['/bin/chmod','-N',str(parent)],
                                              check=True,capture_output=True))
        def chmod_acl(entry):
            subprocess.run(['/bin/chmod','+a',entry,str(parent)],check=True,capture_output=True)
        chmod_acl('everyone allow add_file,add_subdirectory,delete_child,file_inherit,directory_inherit')
        with self.assertRaisesRegex(ValueError,'ACL'):
            snapshot.prepare(self.repo,self.commit,parent/'rejected')
        self.assertFalse((parent/'rejected').exists())
        inherited=parent/'inherited';inherited.mkdir()
        fd=os.open(inherited,snapshot.DIRECTORY_FLAGS)
        try:
            with self.assertRaisesRegex(ValueError,'ACL'):snapshot.reject_allow_acl(fd)
        finally:os.close(fd)
        subprocess.run(['/bin/chmod','-N',str(parent)],check=True,capture_output=True)
        chmod_acl('everyone deny delete')
        snapshot.prepare(self.repo,self.commit,parent/'accepted')


if __name__ == '__main__':
    unittest.main()
