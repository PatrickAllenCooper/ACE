"""Only NEW artificial metadata seams; never import collector or call its main."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT/'pull_terminal.py').read_bytes()
OLD = (ROOT/'predecessor622/pull_terminal.py').read_bytes()
TREE = ast.parse(SOURCE)
MAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def definitions(tree, names, env):
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError('requested metadata-only definitions missing')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<selected-artificial-metadata-seams>', 'exec'), env)
    return env


def need(ok, message):
    if not ok:
        raise ValueError(message)


def remote_literal(source):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == 'REMOTE_OBSERVATION' for t in n.targets))


class MetadataDelta(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix='fixture-', dir=ROOT/'attempt01')).resolve()

    def test_separate_git_commit_rejects_crossed_bytes_and_preserves_packet_binding(self):
        packet_commit = '1'*40; collector_commit = '2'*40; calls = []
        objects = {packet_commit: b'original packet source', collector_commit: b'new collector source'}
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if command[2:4] == ['cat-file', '-t']:
                return types.SimpleNamespace(returncode=0, stdout=b'commit\n')
            self.assertEqual(command[2], 'show')
            revision, path = command[3].split(':', 1)
            self.assertEqual(path, 'sidecars/collector.py')
            return types.SimpleNamespace(returncode=0, stdout=objects[revision])
        env = definitions(TREE, {'require_collector_commit'}, dict(need=need, Path=Path,
            re=__import__('re'), subprocess=types.SimpleNamespace(run=fake_run), REPO=Path('/fabricated/repo')))
        check = env['require_collector_commit']
        check(collector_commit, 'sidecars/collector.py', objects[collector_commit])
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            check(packet_commit, 'sidecars/collector.py', objects[collector_commit])
        with self.assertRaisesRegex(ValueError, 'full collector'):
            check(collector_commit[:8], 'sidecars/collector.py', objects[collector_commit])
        env['subprocess'] = types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=b'tree\n'))
        with self.assertRaisesRegex(ValueError, 'must be a commit'):
            check(collector_commit, 'sidecars/collector.py', objects[collector_commit])
        new_calls = [n for n in ast.walk(MAIN) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'require_collector_commit']
        self.assertEqual(len(new_calls), 3)
        self.assertTrue(all(ast.unparse(n.args[0]) == 'a.collector_source_commit' for n in new_calls))
        self.assertEqual([ast.unparse(n.args[1]) for n in new_calls],
                         ['collector_path', 'helper_path', 'collector_review.relative_to(REPO)'])
        old_main = next(n for n in ast.parse(OLD).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        def packet_gate(main):
            return next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'need'
                        and any(isinstance(a, ast.Constant) and a.value == 'current local submission lineage differs' for a in n.args))
        self.assertEqual(ast.dump(packet_gate(old_main)), ast.dump(packet_gate(MAIN)))
        shows = [n for n in ast.walk(MAIN) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'subprocess.run'
                 and n.args and isinstance(n.args[0], ast.List) and len(n.args[0].elts) > 2
                 and isinstance(n.args[0].elts[2], ast.Constant) and n.args[0].elts[2].value == 'show']
        self.assertEqual(len(shows), 2)  # Original packet-member loop and packet review.
        self.assertTrue(all('a.packet_commit' in ast.unparse(n.args[0]) for n in shows))
        required_args = next(n for n in MAIN.body if isinstance(n, ast.For)
                             and isinstance(n.iter, ast.Tuple) and 'collector-source-commit' in ast.unparse(n.iter))
        self.assertTrue(any(isinstance(k, ast.keyword) and k.arg == 'required' and k.value.value is True
                            for n in ast.walk(required_args) if isinstance(n, ast.Call) for k in n.keywords))
        (self.fixture/'fabricated-git-calls.json').write_text(json.dumps(
            [dict(command=c, kwargs={k: str(v) for k, v in kw.items()}) for c, kw in calls], indent=2)+'\n')

    def test_absolute_deadline_bypasses_per_file_errors_and_progress_preserves_gates(self):
        remote = ast.parse(remote_literal(SOURCE)); old_remote = ast.parse(remote_literal(OLD))
        alarms = [n for n in ast.walk(remote) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'signal.alarm']
        self.assertEqual(len(alarms), 1); self.assertEqual(ast.literal_eval(alarms[0].args[0]), 1200)
        env = definitions(remote, {'need', 'RemoteObservationDeadline', 'remote_deadline', 'fingerprint'},
            dict(os=os, stat=stat, hashlib=hashlib, json=json, time=time, sys=__import__('sys'),
                 _FINGERPRINT_COUNT=999, _OBSERVATION_STARTED=time.monotonic()))
        p = self.fixture/'opaque-artificial.txt'; p.write_bytes(b'fabricated byte fixture\n')
        capture = io.StringIO()
        with contextlib.redirect_stderr(capture):
            info, st = env['fingerprint'](p)
        observed = json.loads(capture.getvalue())
        self.assertEqual(observed['phase'], 'terminal_inventory_fingerprint')
        self.assertEqual(observed['files_read'], 1000)
        self.assertGreaterEqual(observed['elapsed_seconds'], 0)
        self.assertEqual(info, dict(bytes=st.st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        (self.fixture/'progress.stderr.log').write_text(capture.getvalue())
        with self.assertRaises(env['RemoteObservationDeadline']):
            try:
                env['remote_deadline'](None, None)
            except (OSError, ValueError):
                self.fail('absolute alarm swallowed by the original per-file error handler')
        # The actual alarm is never installed and no remote/scheduler body runs.
        outer_try = next(n for n in remote.body if isinstance(n, ast.Try))
        handler = next(h for h in outer_try.handlers if ast.unparse(h.type) == 'BaseException')
        partial = {'files': {'launch/toy': info}}
        exc = env['RemoteObservationDeadline']('absolute remote metadata deadline')
        exec(compile(ast.fix_missing_locations(ast.Module(body=handler.body, type_ignores=[])),
                     '<partial-observation-publication-only>', 'exec'), dict(v=partial, exc=exc))
        self.assertEqual(partial['files']['launch/toy'], info)
        self.assertEqual(partial['observation_error']['type'], 'RemoteObservationDeadline')
        (self.fixture/'partial-observation.json').write_text(json.dumps(partial, indent=2)+'\n')
        # All pre-existing protected remote statements remain byte-semantic AST identical.
        new_nodes = []
        for n in remote.body:
            if isinstance(n, ast.Import):
                n.names = [a for a in n.names if a.name not in ('signal', 'sys', 'time')]
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('remote_deadline', 'RemoteObservationDeadline'):
                continue
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) in ('signal.signal', 'signal.alarm'):
                continue
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('_FINGERPRINT_COUNT', '_OBSERVATION_STARTED') for t in n.targets):
                continue
            if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint':
                n.body = [x for x in n.body if not isinstance(x, ast.Global)
                          and not (isinstance(x, ast.AugAssign) and ast.unparse(x.target) == '_FINGERPRINT_COUNT')
                          and not (isinstance(x, ast.If) and '_FINGERPRINT_COUNT' in ast.unparse(x.test))]
            new_nodes.append(n)
        self.assertEqual(ast.dump(ast.Module(body=new_nodes, type_ignores=[])), ast.dump(old_remote))

    def test_local_timeout_retains_partial_streams_and_observe_rejects_partial(self):
        saved = []
        def save(n, v):
            saved.append((n, v))
            with (self.fixture/n).open('x') as f:
                json.dump(v, f, sort_keys=True, indent=2); f.write('\n')
        def expired(command, **kwargs):
            self.assertEqual(kwargs['timeout'], 1230)
            raise subprocess.TimeoutExpired(command, 1230, output=b'{"partial":', stderr=b'progress1000\n\xfftail')
        env = definitions(MAIN, {'run_saved'}, dict(subprocess=types.SimpleNamespace(run=expired,
            TimeoutExpired=subprocess.TimeoutExpired), save=save, need=need, utc=lambda: 'fabricated-utc'))
        with self.assertRaisesRegex(ValueError, 'failed/partial custody retained'):
            env['run_saved']('transport.json', ['fabricated-no-execution'], source='fabricated-source', timeout=1230)
        v = saved[0][1]
        self.assertIsNone(v['returncode']); self.assertTrue(v['timeout'])
        self.assertFalse(v['retry_allowed']); self.assertTrue(v['partial_custody_preserved'])
        self.assertEqual(v['stdout'], '{"partial":'); self.assertEqual(v['stderr'], 'progress1000\n\ufffdtail')
        self.assertEqual(v['python_source'], 'fabricated-source')
        observations = []
        def fake_saved(n, command, **kwargs):
            observations.append(kwargs)
            return {'stdout': json.dumps({'terminal_gate': True, 'observation_error': {'type': 'RemoteObservationDeadline'}})}
        env = definitions(MAIN, {'observe'}, dict(run_saved=fake_saved, HELPER='fabricated-helper', source='fabricated-source',
            decode=json.loads, need=need))
        with self.assertRaisesRegex(ValueError, 'terminal inventory unavailable'):
            env['observe']('fake.json')
        self.assertEqual(observations[0]['timeout'], 1230)


if __name__ == '__main__':
    unittest.main(verbosity=2)
