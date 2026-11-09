"""Exercise only new authority joins with fabricated metadata and a checker stub."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

HERE = Path(__file__).parent
raw = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
sha = lambda value: hashlib.sha256(value).hexdigest()
digest = lambda value: sha(raw(value))


def need(ok, message):
    if not ok:
        raise ValueError(message)


class AuthorityBinding(unittest.TestCase):
    def test_independent_authority_joins(self):
        tree = ast.parse((HERE/'qualify_llm_intervention_scientific_campaign_gpu.py').read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'recovery_contract')
        names = {'authority':'gpu-recovery-authority.json', 'context':'cpu-history-expected.json',
                 'receipt':'cpu-history-preflight.json', 'read_set':'cpu-history-readset.json'}
        env = dict(RECOVERY_FILES=names, PREFLIGHT='llm_intervention_cpu_preflight_receipt.py',
                   PREFLIGHT_SHA='a'*64, decode=json.loads, sha=sha, json=json, need=need,
                   module=lambda *args: types.SimpleNamespace(validate_receipt=lambda *args: {'fixture': True}))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<new-authority-fixture>', 'exec'), env)
        packet = {'files': {'source.py':'b'*64, names['authority']:'c'*64,
                            names['context']:'d'*64, names['receipt']:'e'*64, names['read_set']:'f'*64},
                  'source_names':['source.py'], 'cpu_acceptance_sha256':'1'*64,
                  'paths': {'cpu_output':'/fixture/cpu', 'campaign':'/fixture/cpu/worker/artifacts/campaign',
                            'gpu_output':'/fixture/gpu01/gpu'}, 'resources':{'fixture':4}, 'limits':{'fixture':240}}
        authority = {'schema':'ace-gpu-cpu-history-recovery-authority-v1', 'status':'operationally_adopted',
            'historical_postcustody_required':True, 'unchanged_scientific_inputs':True,
            'original_entry_sha256':'2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1',
            'original_source_commit':'a9346022f145cc29439325b6f291597b2e6ec037',
            'original_gpu01_packet_sha256':'436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7',
            'original_gpu01_job_id':'35498218', 'control_root':'/fixture/gpu02',
            'allowed_new_output':'/fixture/gpu01/gpu', 'failed_gpu01_root':'/fixture/gpu01',
            'baseline':{'sha256':'2'*64,'terminal_jobs':47,'submission_journals':96},
            'failed_gpu01_acceptance_sha256':'3'*64, 'namespace_sha256':'4'*64,
            'original_namespace_sha256':'5'*64, 'historical_verifier_sha256':'6'*64,
            'historical_proof_pins':{'read_set_artifact_sha256':'f'*64},
            'cpu_acceptance_sha256':'1'*64, 'operational_source_files':{'source.py':'b'*64}}
        context = {'lineage':{'cpu_acceptance_sha256':'1'*64,'recovery_authority_sha256':'c'*64,
                    'checker_sha256':'a'*64,'failed_gpu01_acceptance_sha256':'3'*64,
                    'namespace_sha256':'4'*64,'original_namespace_sha256':'5'*64,'source_sha256':'6'*64},
            'control_root':'/fixture/gpu02', 'gpu_output':'/fixture/gpu01/gpu','failed_gpu01_root':'/fixture/gpu01',
            'cpu_roots': {k:packet['paths'][k] for k in ('cpu_output','campaign')},
            'baseline':authority['baseline'], 'proof_pins':authority['historical_proof_pins'],
            'source_files_sha256':digest(authority['operational_source_files']),
            'model_files_sha256':digest({'fixture':'7'*64}),
            'resources_sha256':digest(packet['resources']), 'limits_sha256':digest(packet['limits'])}
        expected = {'execution_context':context,'cpu_inventories':{
            'cpu_output':{'files':{'terminal.json':{'sha256':'8'*64}}},
            'campaign':{'files':{'campaign.json':{'sha256':'9'*64}}}}}
        acceptance = {'cpu_output_files':{'terminal.json':'8'*64},'campaign_files':{'campaign.json':'9'*64}}
        def call(a, e):
            data={names['authority']:raw(a),names['context']:raw(e), names['receipt']:b'{}',
                  env['PREFLIGHT']:b'# artificial'}
            return env['recovery_contract'](packet,data,Path('/fixture/gpu02/inputs'),acceptance,
                                             {'model_files':{'fixture':'7'*64}})
        self.assertEqual(call(authority, expected)['authority_sha256'], 'c'*64)
        for key, replacement in [('historical_verifier_sha256','0'*64),
                                 ('historical_proof_pins',{'read_set_artifact_sha256':'0'*64}),
                                 ('cpu_acceptance_sha256','0'*64),
                                 ('control_root','/fixture/other'),
                                 ('original_gpu01_job_id','999'),
                                 ('unchanged_scientific_inputs',False)]:
            altered=copy.deepcopy(authority); altered[key]=replacement
            with self.subTest(field=key), self.assertRaises(ValueError):
                call(altered,expected)


if __name__ == '__main__':
    unittest.main()
