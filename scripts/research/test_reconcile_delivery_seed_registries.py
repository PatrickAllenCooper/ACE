import json
from pathlib import Path
import tempfile
import unittest
import reconcile_delivery_seed_registries as r

class Reconciliation(unittest.TestCase):
    def test_seed_fields_text_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meta.json').write_text(json.dumps({'seed':17,'optimizer_seed':3}))
            (root/'seed_23.json').write_text('{}');(root/'summary.csv').write_text('seed,value\n19,0\n')
            d=r.scan([tmp],[17,23])
            self.assertEqual(d['structured_or_declared_intersection'],[17])
            self.assertIn(19,d['known_seed_values'])
            self.assertEqual({v for m in d['proposed_mentions'] for v in m['seeds']},{17,23})
            self.assertTrue(all(len(f['sha256'])==64 for f in d['files']))
    def test_failures_and_symlinks_are_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'bad.json').write_text('{');(root/'link.json').symlink_to(root/'bad.json')
            (root/'huge.json').write_bytes(b' '* (r.MAX_BYTES+1))
            d=r.scan([tmp,str(root/'missing')],[17])
            self.assertEqual({e['reason'] for e in d['unread']},{'oversized','json_parse_failure_seed_fields','root_missing'})
            self.assertEqual(d['symlinks_not_followed'],[str(root/'link.json')])
    def test_registry_only_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meta.json').write_text('{"seed":17}');(root/'raw.csv').write_text('seed\n23\n')
            d=r.scan([tmp],[17,23],registry_only=True)
            self.assertEqual(len(d['files']),1);self.assertEqual(d['structured_or_declared_intersection'],[17])

if __name__=='__main__':unittest.main()
