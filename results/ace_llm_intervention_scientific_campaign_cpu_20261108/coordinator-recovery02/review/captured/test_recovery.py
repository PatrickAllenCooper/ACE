"""New fabricated coordinator recovery checks. No SSH, Slurm or science."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parent
NEW=(ROOT/'authoring/launch_once.py').read_bytes()
assert hashlib.sha256(NEW).hexdigest()=='0bb50f3048efaa3bcab96fcec8c24029be00c991d4a7d9b4cdf7a1a433bddca4'
BUILDER=(ROOT/'authoring/build_packet.py').read_bytes()
assert hashlib.sha256(BUILDER).hexdigest()=='3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'
b=types.ModuleType('build_packet');exec(compile(BUILDER,'<captured-metadata-builder>','exec'),b.__dict__)
sys.modules['build_packet']=b
controller=types.ModuleType('recovery_under_test');exec(compile(NEW,'<captured-recovery>','exec'),controller.__dict__)


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='fixture-',dir=ROOT/'test01')).resolve()
        self.previous=(b.LOCAL,b.EXT)
        b.LOCAL=self.root/'repo';b.EXT=self.root/'external'
        for d in (b.LOCAL,b.EXT):d.mkdir()
        values={'controller.json':{'pid':123},'controller_released.json':{'released':True},
                'launcher_failure.json':{'retry_allowed':False},'readiness_transport.json':{
                    'returncode':1,'stdout':'','stderr':'TimeoutError: absolute remote coordinator deadline'}}
        self.proof={'scientific_packet_unchanged':'a'*64,'no_sbatch_reached':True,'no_new_allocation':True,'original_failed_members':{}}
        for n,v in values.items():
            data=b.raw(v);self.proof['original_failed_members'][n]={'bytes':len(data),'sha256':b.sha(data)}
            for d in (b.LOCAL,b.EXT):(d/n).write_bytes(data)
        self.original={str(p):p.read_bytes() for d in (b.LOCAL,b.EXT) for p in d.iterdir()}

    def tearDown(self):b.LOCAL,b.EXT=self.previous

    def test_only_exact_unsubmitted_failure_is_admitted(self):
        controller.admit_recovery(self.proof,'a'*64)
        for key in ('no_sbatch_reached','no_new_allocation'):
            altered=dict(self.proof,**{key:False})
            with self.subTest(key=key),self.assertRaises(ValueError):controller.admit_recovery(altered,'a'*64)
        with self.assertRaises(ValueError):controller.admit_recovery(self.proof,'b'*64)
        for n in ('recovery02','submission.intent.json','submission.result.json'):
            p=b.LOCAL/n;p.write_bytes(b'owned fabricated marker')
            with self.subTest(marker=n),self.assertRaises(ValueError):controller.admit_recovery(self.proof,'a'*64)
            p.unlink()
        for p,data in self.original.items():self.assertEqual(Path(p).read_bytes(),data)

    def test_failure_changed_in_either_copy_blocks(self):
        (b.EXT/'launcher_failure.json').write_bytes(b'changed artificial evidence')
        with self.assertRaises(ValueError):controller.admit_recovery(self.proof,'a'*64)

    def test_new_journals_never_overwrite_original_failure(self):
        tree=ast.parse(NEW);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        save=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='save')
        env={'b':b,'RECOVERY_NAME':controller.RECOVERY_NAME,'CANONICAL_RECORDS':controller.CANONICAL_RECORDS}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[save],type_ignores=[])),'<new-save>','exec'),env)
        for d in (b.LOCAL,b.EXT):(d/'recovery02').mkdir()
        env['save']('launcher_failure.json',{'new':True})
        env['save']('submission.intent.json',{'new_intent':True})
        for d in (b.LOCAL,b.EXT):
            self.assertEqual((d/'launcher_failure.json').read_bytes(),self.original[str(d/'launcher_failure.json')])
            self.assertEqual((d/'submission.intent.json').read_bytes(),(d/'recovery02/submission.intent.json').read_bytes())
        with self.assertRaises(FileExistsError):env['save']('submission.intent.json',{'overwrite':True})

    def test_original_remote_gates_and_child_are_unchanged(self):
        old=(ROOT.parent/'launch01/launch_once.py').read_text()
        def literal(src):
            return next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets))
        original=literal(old);recovery=literal(NEW)
        a={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
        z={n.name:n for n in ast.parse(recovery).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(a),set(z))
        for name in a:
            if name=='read':
                z[name].body=[n for n in z[name].body if not isinstance(n,ast.Global)
                    and not (isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='_READ_COUNT')
                    and not (isinstance(n,ast.If) and '_READ_COUNT' in ast.unparse(n.test))]
            self.assertEqual(ast.dump(a[name]),ast.dump(z[name]),name)
        self.assertIn('signal.alarm(1200)',recovery)
        self.assertIn('timeout=1230',NEW.decode())
        self.assertEqual(old.count("subprocess.run(['sbatch'"),1)
        self.assertEqual(NEW.decode().count("subprocess.run(['sbatch'"),1)
        self.assertIn('timeout=25)',NEW.decode())
        self.assertEqual(b.DEADLINES,dict(startup=180,work=5940,hard=5980))
        self.assertEqual(b.RESOURCES['wall_seconds'],6000)


if __name__=='__main__':unittest.main(verbosity=2)
