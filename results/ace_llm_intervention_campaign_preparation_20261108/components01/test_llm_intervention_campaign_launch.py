"""New pure tests for the fixed-command / later-runtime binding boundary."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import llm_intervention_campaign_launch as launch


class LaunchTests(unittest.TestCase):
    def objects(self):
        c = {'schema': launch.VERSION, 'policy_id': 'sft_llm',
            'entry': '/packet/llm_intervention_model_service.py', 'entry_sha256': 'a'*64,
            'freeze': '/packet/sft.json', 'freeze_sha256': 'b'*64,
            'worker_output': '/results/workers/sft', 'launch_output': '/results/launches/sft',
            'binding_path': '/results/stage-binding.json', 'stage_path': '/results/stage.json',
            'archive_sha256': 'c'*64, 'manifest_sha256': 'd'*64,
            'bootstrap': 'captured entry bytes only',
            'bootstrap_sha256': launch._sha(b'captured entry bytes only')}
        cp = launch._sha(launch._raw(c))
        b = {'schema': launch.VERSION, 'job_id': '12345', 'stage_path': c['stage_path'],
             'stage_sha256': 'e'*64, 'staged_prefix': '/local/job123/runtime',
             'contracts': {'frozen_llm': 'f'*64, 'sft_llm': cp, 'dpo_llm': '0'*64}}
        s = {'version': 'ace-runtime-archive-v1', 'archive_sha256': c['archive_sha256'],
             'manifest_sha256': c['manifest_sha256'], 'staged_prefix': b['staged_prefix'],
             'objects': 49966, 'file_bytes': 8397575073, 'model_loaded': False, 'runtime_qualified': False}
        return c, cp, b, s

    def test_exact_argv_and_unchanged_contract_across_runtime_locations(self):
        c, cp, b, s = self.objects()
        before = launch._raw(c)
        args = launch.derive_command(c, contract_sha256=cp, binding=b, stage=s)
        self.assertEqual(args, ['/local/job123/runtime/bin/python', '-I', '-B', '-c', c['bootstrap'],
            c['entry'], c['entry_sha256'], '--freeze', c['freeze'], '--freeze-sha256', c['freeze_sha256'],
            '--output', c['worker_output'], '--runtime-stage', c['stage_path'], '--runtime-stage-sha256', 'e'*64])
        b['staged_prefix'] = s['staged_prefix'] = '/different/local/runtime'
        changed = launch.derive_command(c, contract_sha256=cp, binding=b, stage=s)
        self.assertEqual(changed[1:], args[1:])
        self.assertEqual(launch._raw(c), before)

    def test_missing_or_conflicting_stage_contract_and_injected_command_rejected(self):
        for target, key, value in [('binding','command',['sh','-c','anything']),
                ('binding','job_id','1; sh'), ('binding','stage_path','/other/stage'),
                ('binding','staged_prefix','/tmp/../runtime'), ('stage','objects',49965),
                ('stage','file_bytes',1), ('stage','model_loaded',True),
                ('stage','archive_sha256','0'*64), ('stage','staged_prefix','/different')]:
            with self.subTest(target=target,key=key):
                c, cp, b, s = self.objects()
                (b if target == 'binding' else s)[key] = value
                with self.assertRaises(ValueError):
                    launch.derive_command(c, contract_sha256=cp, binding=b, stage=s)
        c, cp, b, s = self.objects()
        b['contracts']['sft_llm'] = '0'*64
        with self.assertRaises(ValueError): launch.derive_command(c, contract_sha256=cp, binding=b, stage=s)

    def test_contract_paths_bootstrap_and_role_rejected(self):
        for key, value in [('policy_id','policy_train'), ('entry','/packet/other.py'),
                ('freeze','relative'), ('worker_output','/packet'),
                ('launch_output','/results/workers/sft/subdir'),
                ('bootstrap','unbound script'), ('entry_sha256','bad')]:
            with self.subTest(key=key):
                c, cp, b, s = self.objects()
                c[key] = value
                with self.assertRaises(ValueError): launch.validate_contract(c)

    def test_byte_read_rejects_pin_symlink_oversize_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'input'
            p.write_bytes(b'{"a":1}')
            pin = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(launch.read_pinned(p,pin),b'{"a":1}')
            with self.assertRaises(ValueError): launch.read_pinned(p,'0'*64)
            alias = Path(directory)/'alias'; alias.symlink_to(p)
            with self.assertRaises(OSError): launch.read_pinned(alias,pin)
            with patch.object(launch, 'MAX_BYTES', 2), self.assertRaises(ValueError): launch.read_pinned(p,pin)
        with self.assertRaises(ValueError): launch._decode(b'{"a":1,"a":2}')

    def test_cli_records_exact_exec_without_launching_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, _, b, s = self.objects()
            entry = root/'llm_intervention_model_service.py'; entry.write_bytes(b'no execution')
            freeze = root/'sft.json'; freeze.write_bytes(b'{}')
            c.update(entry=str(entry), entry_sha256=launch._sha(entry.read_bytes()),
                     freeze=str(freeze), freeze_sha256=launch._sha(freeze.read_bytes()),
                     worker_output=str(root/'worker'), launch_output=str(root/'launch'),
                     binding_path=str(root/'binding.json'), stage_path=str(root/'stage.json'))
            cp = launch._sha(launch._raw(c))
            b.update(stage_path=c['stage_path'])
            b['contracts']['sft_llm'] = cp
            Path(c['stage_path']).write_bytes(launch._raw(s))
            b['stage_sha256'] = launch._sha(launch._raw(s))
            Path(c['binding_path']).write_bytes(launch._raw(b))
            contract = root/'contract.json'; contract.write_bytes(launch._raw(c))
            env = {launch.BINDING_ENV: launch._sha(launch._raw(b)), 'SLURM_JOB_ID':'12345',
                   'CUDA_VISIBLE_DEVICES':'MIG-test', 'MKL_THREADING_LAYER':'GNU'}
            argv = ['shim','--contract',str(contract),'--contract-sha256',cp]
            with patch.dict(os.environ,env,clear=True), patch('sys.argv',argv), patch.object(launch.os,'execv') as execute:
                launch.main()
                command = launch.derive_command(c,contract_sha256=cp,binding=b,stage=s)
                execute.assert_called_once_with(command[0],command)
                receipt = json.loads((root/'launch/execution.json').read_bytes())
                self.assertEqual(receipt['command'],command)
                self.assertEqual(receipt['stage_binding_sha256'],env[launch.BINDING_ENV])
                with self.assertRaises(FileExistsError): launch.main()
            self.assertFalse((root/'worker').exists())


if __name__ == '__main__': unittest.main()
