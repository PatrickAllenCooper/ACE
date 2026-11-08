"""New worker contract tests only: fabricated tokens and mocked execution.

Run directly, avoiding pytest/conftest. Set ACE_SCREEN_PROMPT_CPU_TEST_OUTPUT
to a new external directory; every attempt and rejected input remains there.
No real tokenizer, numerical/model import, scheduler, process or runtime stage.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch


class ForbidRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'torch', 'transformers', 'tokenizers'}:
            raise AssertionError('actual numerical/tokenizer import forbidden: '+fullname)


sys.meta_path.insert(0, ForbidRuntime())
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/research'
sys.path.insert(0, str(SOURCE))
import qualify_llm_intervention_screen_prompt_cpu as gate


class FabricatedTokenSource:
    eos_token_id, pad_token_id = 151645, 151643

    def __init__(self, commands, lengths=None):
        self.table = {c: [1000+i] for i, c in enumerate(commands)}
        self.lengths = list(lengths or [])
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [42] * (self.lengths.pop(0) if self.lengths else 100)

    def encode(self, text, **kwargs):
        return self.table[text]

    def decode(self, ids, **kwargs):
        return next(c for c, value in self.table.items() if list(ids) == value)


class ScreenCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = Path(os.environ['ACE_SCREEN_PROMPT_CPU_TEST_OUTPUT'])
        cls.output.mkdir(parents=True, exist_ok=False)
        custody = ROOT/'results/ace_llm_intervention_screen_prompt_preparation_20261108/preparation01/cli-packet'
        manifest_raw = (custody/'manifest.json').read_bytes()
        cls.manifest = gate.B.decode(manifest_raw)
        cls.data = {'fixture-manifest.json': manifest_raw}
        for row in cls.manifest['fixtures']:
            for key in ('envelope_file', 'prompt_file'):
                name = row[key]['path']
                cls.data[name] = (custody/name).read_bytes()
        cls.service = gate.B.module('test_service', SOURCE/gate.SERVICE, (SOURCE/gate.SERVICE).read_bytes())
        cls.campaign = gate.B.module('test_campaign', SOURCE/gate.CAMPAIGN, (SOURCE/gate.CAMPAIGN).read_bytes())
        cls.diagnostic = gate.B.module('test_diagnostic', SOURCE/gate.DIAGNOSTIC, (SOURCE/gate.DIAGNOSTIC).read_bytes())
        for name in set(gate.HELPER_PINS) | cls.service.SOURCE_NAMES | {gate.ENTRY}:
            cls.data[name] = b'# fabricated cleanup helper; never executed\n' if name == 'probe.py' else (SOURCE/name).read_bytes()
        cls.probe_pin = gate.B.sha(cls.data['probe.py'])

    def setUp(self):
        self.root = self.output/self._testMethodName
        self.root.mkdir()

    def context(self):
        inputs = self.root/'inputs'
        inputs.mkdir()
        for name, value in self.data.items():
            (inputs/name).write_bytes(value)
        p = dict(schema=gate.VERSION, scope='artificial', request_mode='all_120_frozen_fixtures',
            source_commit='613c3d0432ac64937e6322fba0720e089cfbe04e',
            source_names=sorted(self.service.SOURCE_NAMES | {gate.ENTRY}),
            files={n: gate.B.sha(v) for n, v in self.data.items()}, fixture_manifest_sha256=gate.FIXTURE_MANIFEST_SHA,
            archive=dict(path=str(self.root/'archive.tar'), sha256=self.diagnostic.ARCHIVE_SHA),
            manifest=dict(path=str(self.root/'runtime-manifest.json'), sha256=self.diagnostic.MANIFEST_SHA),
            runtime_source=self.diagnostic.RUNTIME_SOURCE, runtime=self.service.RUNTIME,
            model_directory=str(self.root/'model'), model_files=self.campaign.MODEL_FILES,
            resources=gate.RESOURCES, output=str(self.root/'output'))
        args = types.SimpleNamespace(packet=str(inputs/'packet.json'), source_commit=p['source_commit'],
            output=p['output'], started=time.monotonic(), process_started=time.process_time())
        args.packet_sha256 = gate.B.save(args.packet, p)
        return args, p

    def authenticate(self, args):
        # Only the cleanup helper is fabricated, and is never executed here.
        with patch.dict(gate.HELPER_PINS, {'probe.py': self.probe_pin}), \
             patch.object(gate, '__file__', str(Path(args.packet).parent/gate.ENTRY)):
            return gate.authenticate(args)

    def run_tokens(self, *, lengths=None, prepare_change=None, deadline=None):
        rows = self.manifest['fixtures']
        context = dict(rows=rows, data=self.data)
        tokenizer = FabricatedTokenSource(rows[0]['legal_commands'], lengths)
        journal = self.service.Journal(self.root/'artifacts')
        by_envelope = {row['envelope_sha256']: row for row in rows}
        def prompt(envelope):
            row = by_envelope[gate.B.sha(gate.B.raw(envelope)[:-1])]
            return self.data[row['prompt_file']['path']].decode()
        def prepare(envelope, capture, *, max_tokens, max_new_tokens):
            self.assertEqual((max_tokens, max_new_tokens), (8192, 32))
            row = by_envelope[gate.B.sha(gate.B.raw(envelope)[:-1])]
            ids = capture.apply_chat_template([{'role': 'user', 'content': prompt(envelope)}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            commands = []
            for command in row['legal_commands']:
                tokens = capture.encode(command, add_special_tokens=False)
                self.assertEqual(capture.decode(tokens, skip_special_tokens=False), command)
                commands.append(tokens+[capture.eos_token_id])
            if prepare_change:
                prepare_change(row, capture)
            if len(ids)+32 > 8192 or max(map(len, commands)) > 32:
                raise ValueError('public task or complete action exceeds frozen token limits')
            metadata = dict(version='ace-shared-task-language-policy-v1',
                envelope_sha256=row['envelope_sha256'], prompt_sha256=row['prompt_sha256'],
                task_sha256=gate.B.sha(gate.B.raw(envelope['task'])[:-1]), prompt_tokens=len(ids),
                longest_command_tokens=max(map(len, commands)), max_tokens=8192, max_new_tokens=32,
                enable_thinking=False, legal_commands=len(commands))
            return types.SimpleNamespace(prompt_ids=ids, completion_ids=commands), metadata
        result = gate.tokenize_all(context, tokenizer, prepare, prompt, journal,
                                  deadline if deadline is not None else time.monotonic()+30)
        return result, gate.B.decode((journal.output/'tokenization.json').read_bytes()), tokenizer

    def test_authentication_admits_exact_large_flat_roster_and_distinct_pins(self):
        args, packet = self.context()
        ctx = self.authenticate(args)
        self.assertGreater(len(packet['files']), 250)
        self.assertLess(len(packet['files']), 320)
        self.assertEqual(len(ctx['rows']), 120)
        self.assertNotEqual(ctx['manifest_pins']['fixture_manifest_file_sha256'],
                            ctx['manifest_pins']['fixture_manifest_canonical_sha256'])
        (self.root/'observed-roster.json').write_bytes(gate.B.raw(dict(count=len(packet['files']), **ctx['manifest_pins'])))
        self.assertFalse({'numpy', 'torch', 'transformers', 'tokenizers'} & set(sys.modules))

    def test_missing_extra_and_traversal_inputs_rejected(self):
        args, packet = self.context()
        original = copy.deepcopy(packet)
        for index, mode in enumerate(('missing', 'extra', 'traversal')):
            p = copy.deepcopy(original)
            if mode == 'missing':
                del p['files'][self.manifest['fixtures'][0]['prompt_file']['path']]
            else:
                name = 'extra.json' if mode == 'extra' else '../escape.json'
                p['files'][name] = gate.B.sha(b'{}')
                if mode == 'extra':
                    (Path(args.packet).parent/name).write_bytes(b'{}')
            candidate = Path(args.packet).parent/f'request-{index}.json'
            args.packet, args.packet_sha256 = str(candidate), gate.B.save(candidate, p)
            with self.assertRaises((ValueError, KeyError)):
                self.authenticate(args)

    def test_pin_changes_commit_drift_and_manifest_reserialization_rejected(self):
        args, packet = self.context()
        name = self.manifest['fixtures'][0]['prompt_file']['path']
        (Path(args.packet).parent/name).write_bytes(self.data[name]+b' ')
        with self.assertRaisesRegex(ValueError, 'pin differs'):
            self.authenticate(args)
        (Path(args.packet).parent/name).write_bytes(self.data[name])
        args.source_commit = '0'*40
        with self.assertRaisesRegex(ValueError, 'fixed capacity-only'):
            self.authenticate(args)
        altered = dict(self.data)
        altered['fixture-manifest.json'] = json.dumps(self.manifest, indent=2).encode()
        self.assertEqual(gate.B.decode(altered['fixture-manifest.json']), self.manifest)
        with self.assertRaisesRegex(ValueError, 'manifest pin'):
            gate.fixture_inputs(packet, altered)

    def test_all_120_fabricated_results_keep_exact_ids_commands_and_flags(self):
        summary, saved, tokenizer = self.run_tokens()
        self.assertEqual(summary, dict(fixtures=120, fits=120, overflows=0, failures=0, capacity_qualified=True))
        self.assertEqual([r['fixture_id'] for r in saved['rows']], [r['fixture_id'] for r in self.manifest['fixtures']])
        self.assertTrue(all(r['prompt_ids'] == [42]*100 for r in saved['rows']))
        self.assertTrue(all(call[1]['truncation'] is False for call in tokenizer.calls))
        self.assertEqual(saved['scientific_qualified'], False)
        for result, row in zip(saved['rows'], self.manifest['fixtures']):
            self.assertEqual(list(result['commands']), row['legal_commands'])
            self.assertTrue(all(ids[-1] == 151645 for ids in result['commands'].values()))

    def test_exact_context_boundary_and_overflow_ids_are_preserved_without_retry(self):
        summary, saved, tokenizer = self.run_tokens(lengths=[8160, 8161])
        self.assertEqual((summary['fits'], summary['overflows'], summary['failures']), (119, 1, 0))
        self.assertFalse(summary['capacity_qualified'])
        self.assertEqual(saved['rows'][0]['status'], 'fits')
        self.assertEqual(saved['rows'][1]['status'], 'overflow')
        self.assertEqual(len(saved['rows'][1]['prompt_ids']), 8161)
        self.assertEqual(len(saved['rows'][1]['commands']), 9)
        self.assertEqual(len(tokenizer.calls), 120)

    def test_partial_command_failure_keeps_failed_row_and_continues(self):
        def change(row, capture):
            if row is self.manifest['fixtures'][0]:
                capture.commands.pop('{"stop":true}')
                raise ValueError('public task or complete action exceeds frozen token limits')
        summary, saved, tokenizer = self.run_tokens(prepare_change=change)
        self.assertEqual((summary['fits'], summary['overflows'], summary['failures']), (119, 0, 1))
        self.assertEqual(saved['rows'][0]['status'], 'failed')
        self.assertNotIn('{"stop":true}', saved['rows'][0]['commands'])
        self.assertEqual(len(tokenizer.calls), 120)

    def test_deadline_preserves_all_unattempted_slots(self):
        with self.assertRaises(gate.Deadline):
            self.run_tokens(deadline=time.monotonic()-1)
        saved = gate.B.decode((self.root/'artifacts/tokenization.json').read_bytes())
        self.assertEqual(len(saved['rows']), 120)
        self.assertEqual({r['status'] for r in saved['rows']}, {'not_attempted'})

    def test_missing_eos_wrong_command_duplicate_tokens_and_metadata_rejected(self):
        row = self.manifest['fixtures'][0]
        valid = dict(prompt_ids=[7], commands={c: [1000+i, 151645] for i, c in enumerate(row['legal_commands'])})
        for kind in ('missing_eos', 'extra_command', 'duplicate', 'bool_token'):
            value = copy.deepcopy(valid)
            if kind == 'missing_eos':
                value['commands'][row['legal_commands'][0]].pop()
            elif kind == 'extra_command':
                value['commands']['{"action_id":"unknown"}'] = [99, 151645]
            elif kind == 'duplicate':
                value['commands'][row['legal_commands'][1]] = value['commands'][row['legal_commands'][0]]
            else:
                value['prompt_ids'] = [True]
            gate.B.save(self.root/(kind+'.json'), value)
            with self.assertRaises(ValueError):
                gate.validate_tokens(value, row)
        valid['metadata'] = {'max_tokens': 9000}
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            gate.validate_metadata(valid, row, {'task': {}})

    def test_accelerators_and_changed_cpu_controls_rejected(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(gate.glob, 'glob', return_value=[]):
            self.assertEqual(gate.no_accelerators()['devices'], [])
            for name in ('SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
                with patch.dict(os.environ, {name: '0'}), self.assertRaises(ValueError):
                    gate.no_accelerators()
        with patch.dict(os.environ, {}, clear=True), patch.object(gate.glob, 'glob', return_value=['/dev/nvidia0']):
            with self.assertRaises(ValueError):
                gate.no_accelerators()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'controls differ'):
            gate.cpu_controls(Path('/fabricated/runtime'))

    def test_saved_acceptance_requires_full_cpu_guard_and_rejects_mapping_exemption(self):
        args, _ = self.context()
        ctx = self.authenticate(args)
        summary, saved, _ = self.run_tokens()
        out = Path(args.output)/'worker/artifacts'
        out.parent.mkdir(parents=True)
        (self.root/'artifacts').rename(out)
        prefix = Path('/fabricated/staged-runtime')
        stage, stage_pin = {'staged_prefix': str(prefix)}, 'a'*64
        controls = dict(CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=str(prefix/'lib'),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='GNU')
        inputs = dict(models={n: dict(sha256=p, bytes=1) for n, p in ctx['packet']['model_files'].items()},
                      accelerators=dict(devices=[], environment={}), controls=controls)
        for name in ('inputs-before.json', 'inputs-after.json'):
            gate.B.save(out/name, inputs)
        gate.B.save(out/'runtime-stage-binding.json', {'gpu_mapping_enabled': False})
        projects = {Path(n).stem: {'sha256': ctx['packet']['files'][n]}
                    for n in ctx['packet']['source_names'] if n != gate.ENTRY}
        for phase in ('after-service-imports', 'after-cpu-tokenization'):
            gate.B.save(out/('runtime-'+phase+'.json'), {'captured_project_modules': projects})
        for i, phase in enumerate(('service_cpu_import_start', 'service_cpu_import_complete',
                                   'cpu_tokenizer_load_start', 'cpu_tokenization_complete')):
            gate.B.save(out/f'phase-{i:03d}.json', {'phase': phase, 'elapsed_seconds': float(i)})
        terminal = dict(schema=gate.VERSION, packet_sha256=args.packet_sha256, source_commit=args.source_commit,
            stage_sha256=stage_pin, tokenization_sha256=gate.B.sha((out/'tokenization.json').read_bytes()),
            **summary, **ctx['manifest_pins'], **gate.FLAGS)
        gate.B.save(out/'terminal.json', terminal)
        with patch.object(ctx['cpu'], 'validate_runtime_records') as guard:
            self.assertEqual(gate.validate_worker(ctx, args, stage, stage_pin, {}), terminal)
            self.assertEqual(guard.call_count, 1)
            self.assertEqual(guard.call_args.args[1:3], (stage, stage_pin))
            guard.side_effect = ValueError('fabricated mixed native runtime rejection')
            with self.assertRaisesRegex(ValueError, 'mixed native runtime'):
                gate.validate_worker(ctx, args, stage, stage_pin, {})
            guard.side_effect = None
            (out/'runtime-stage-binding.json').write_bytes(gate.B.raw({'gpu_mapping_enabled': True}))
            with self.assertRaisesRegex(ValueError, 'mapping exemption'):
                gate.validate_worker(ctx, args, stage, stage_pin, {})
            gate.B.save(self.root/'rejected-mapping-binding.json', {'gpu_mapping_enabled': True})
            (out/'runtime-stage-binding.json').write_bytes(gate.B.raw({'gpu_mapping_enabled': False}))
            saved['rows'][0]['metadata']['max_tokens'] = 9000
            (out/'tokenization.json').write_bytes(gate.B.raw(saved))
            (out/'fixture-000.json').write_bytes(gate.B.raw(saved['rows'][0]))
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                gate.validate_worker(ctx, args, stage, stage_pin, {})

    def test_parent_reuses_bounded_supervisor_and_bootstrap_without_launching(self):
        args, _ = self.context()
        ctx = self.authenticate(args)
        out = Path(args.output)
        (out/'worker/artifacts').mkdir(parents=True)
        gate.B.save(out/'worker/artifacts/terminal.json', {'fabricated': True})
        scratch = self.root/'scratch'
        scratch.mkdir()
        allocation = ('JobId=17 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal '
            'NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:05:00 Requeue=0 Restarts=0 UserId=paco0228(1) '
            'AllocTRES=cpu=2,node=1,mem=7680M ReqTRES=cpu=2,node=1,mem=7680M')
        prefix = scratch/'ace-screen-prompt-cpu-17/runtime'
        d = ctx['diagnostic']
        probe = types.SimpleNamespace(kill_owned=Mock())
        cleanup = dict(returncode=0, group_still_exists=False, kill_sent=False, errors=[])
        with patch.dict(os.environ, {'SLURM_JOB_ID': '17', 'SLURM_SCRATCH': str(scratch)}, clear=True), \
             patch.object(gate.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=allocation, stderr='')), \
             patch.object(gate, 'no_accelerators', return_value={}), \
             patch.object(gate, 'check_inputs', return_value={'opaque': 'hash'}), \
             patch.object(d, 'stage_runtime', return_value={'staged_prefix': str(prefix)}), \
             patch.object(d, 'provenance_contract', return_value={}), patch.object(d, 'events', return_value=Mock()), \
             patch.object(gate.B, 'module', return_value=probe), \
             patch.object(ctx['training'], 'cpu_environment', return_value={'fixed': 'cpu'}), \
             patch.object(gate.B, 'supervise', return_value=cleanup) as supervise, \
             patch.object(gate, 'validate_worker', return_value={'capacity_qualified': False}), \
             patch.object(gate.signal, 'setitimer') as alarm:
            gate.parent(args, ctx)
        self.assertEqual((gate.STARTUP, gate.WORK, gate.CLEANUP, gate.HARD), (120, 260, 280, 290))
        self.assertEqual(gate.RESOURCES['memory_mib'], 7680)
        command, env, output, start, pin, kill = supervise.call_args.args
        self.assertEqual(command[:5], [str(prefix/'bin/python'), '-I', '-B', '-c', d.BOOTSTRAP])
        self.assertEqual(command[6], ctx['packet']['files'][gate.ENTRY])
        self.assertIs(kill, probe.kill_owned)
        self.assertEqual((env, output, start, pin), ({'fixed': 'cpu'}, out, args.started, args.packet_sha256))
        self.assertLessEqual(alarm.call_args.args[1], 260)
        terminal = gate.B.decode((out/'terminal.json').read_bytes())
        self.assertEqual(terminal['status'], 'capacity_rejected')
        self.assertIs(terminal['scientific_qualified'], False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
