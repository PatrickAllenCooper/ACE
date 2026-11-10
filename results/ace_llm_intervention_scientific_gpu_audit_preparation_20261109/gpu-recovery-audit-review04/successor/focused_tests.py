"""New fabricated path-proof and actual mapping-predicate cases only."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import json
import types
import unittest

HERE = Path(__file__).resolve().parent
data = (HERE/'dependencies/gpu_recovery_delta.py').read_bytes()
assert hashlib.sha256(data).hexdigest() == '0991abb13267cac61c954d87c0bc510fddb256259118dde882a14b1812f09bb6'
mod = types.ModuleType('focused_path_proof')
exec(compile(data, '<pure-path-proof>', 'exec'), mod.__dict__)
aid_bytes = (HERE/'audit_saved_gpu72.py').read_bytes()
assert hashlib.sha256(aid_bytes).hexdigest() == '97361592d83e7ca23e84e1732c430fbd6c16eb62ffe6bf0741fbd214f369d852'
# Execute only the exact pure mapping need(...) expression, extracted from the
# pinned successor AST. Do not import/invoke the aid or any producer.
tree = ast.parse(aid_bytes)
generation = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'generation')
guard = next(n for n in generation.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id == 'need'
             and len(n.value.args) == 2 and isinstance(n.value.args[1], ast.Constant)
             and n.value.args[1].value == 'exact episode/session mapping')
params = ['self', 'mapping', 'ctx', 'session', 'i', 'g', 'ip']
fn = ast.FunctionDef(name='mapping_gate',
    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in params],
                       vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
    body=[copy.deepcopy(guard)], decorator_list=[], returns=None, type_comment=None)
module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
scope = dict(need=mod.need, POLICIES=('uniform_random','menu_coverage','local_information',
    'target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm'))
exec(compile(module, '<exact-mapping-expression>', 'exec'), scope)
mapping_gate = scope['mapping_gate']


def fixture():
    logical = '/scratch/alpine/fixture/cpu'
    resolved = '/gpfs/alpine1/scratch/fixture/cpu'
    suffix = '/worker/artifacts/campaign'
    producer = 'prepare_llm_intervention_scientific_campaign_cpu.py'
    pin = 'a' * 64
    old = dict(schema='ace-scientific-campaign-cpu-v1', cpu_output=logical,
               resources={'cpus': 2}, deadlines={'work': 10})
    proof = dict(schema=old['schema'], packet_sha256=pin,
                 resources=copy.deepcopy(old['resources']),
                 deadlines=copy.deepcopy(old['deadlines']), retry_allowed=False,
                 output_path_binding=dict(logical=logical, resolved=resolved,
                     mount_alias=True, root_device=63, root_inode=8388611))
    data = (json.dumps(proof,sort_keys=True,indent=2)+'\n').encode()
    info = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    packet = dict(files={producer:'ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893'},
                  cpu_packet_sha256=pin, paths=dict(cpu_output=logical,campaign=logical+suffix))
    cpu = dict(cpu_packet_sha256=pin,cpu_output=logical,campaign_root=logical+suffix,
               cpu_output_files={n:info['sha256'] for n in ('start.json','worker/start.json')})
    prior = {'launch/cpu/'+n:copy.deepcopy(info) for n in ('start.json','worker/start.json')}
    current = {'cpu/cpu/'+n:copy.deepcopy(info) for n in ('start.json','worker/start.json')}
    return proof, info, packet, cpu, old, prior, current


def gate_fixture():
    proof = mod.output_path_proof(*fixture())
    owner = SimpleNamespace(recorded_campaign_root=proof['resolved_campaign_root'])
    ctx = dict(index=0, decision=3)
    session = dict(policy_id='frozen_llm')
    generation = dict(task_sha256='c'*64)
    identity = 'd'*64
    mapping = dict(episode_directory=owner.recorded_campaign_root+'/policy-05/world-00/episode',
                   decision_index=3,session_request_id=3,task_sha256=generation['task_sha256'],
                   task_binding_index=0,identity_sha256=identity,status='planned_before_transmission',
                   scope='client_only_episode_to_session_mapping')
    return [owner,mapping,ctx,session,3,generation,identity]


class RecordedPathCases(unittest.TestCase):
    def test_exact_accepted_proof_and_full_mapping(self):
        proof = mod.output_path_proof(*fixture())
        self.assertEqual(proof['resolved_campaign_root'],
                         '/gpfs/alpine1/scratch/fixture/cpu/worker/artifacts/campaign')
        self.assertFalse(proof['acceptance'])
        self.assertFalse(proof['live_resolution_performed'])
        mapping_gate(*gate_fixture())

    def test_reject_wrong_root_or_binding(self):
        changes = [
            lambda x: x[0]['output_path_binding'].update(logical='/scratch/alpine/other/cpu'),
            lambda x: x[0]['output_path_binding'].update(resolved='/unknown/fixture/cpu'),
            lambda x: x[0]['output_path_binding'].update(resolved='/gpfs/alpine1/scratch/other/cpu'),
            lambda x: x[0]['output_path_binding'].update(resolved='/gpfs/alpine1/scratch/fixture/../cpu'),
            lambda x: x[0]['output_path_binding'].update(mount_alias=False),
            lambda x: x[0]['output_path_binding'].update(root_device=True),
            lambda x: x[0]['output_path_binding'].update(root_inode=0),
            lambda x: x[0]['output_path_binding'].update(extra='unbound'),
            lambda x: x[2]['paths'].update(campaign='/scratch/alpine/fixture/cpu/wrong'),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                args=fixture();change(args)
                with self.assertRaises(ValueError):mod.output_path_proof(*args)

    def test_reject_accepted_proof_pin_and_source_drift(self):
        changes = [
            lambda x: x[1].update(sha256='e'*64),
            lambda x: x[1].update(bytes=x[1]['bytes']+1),
            lambda x: x[5]['launch/cpu/start.json'].update(sha256='e'*64),
            lambda x: x[6].pop('cpu/cpu/worker/start.json'),
            lambda x: x[3]['cpu_output_files'].update({'worker/start.json':'e'*64}),
            lambda x: x[2]['files'].update({'prepare_llm_intervention_scientific_campaign_cpu.py':'e'*64}),
            lambda x: x[0].update(packet_sha256='e'*64),
            lambda x: x[0].update(schema='wrong-schema'),
            lambda x: x[0].update(retry_allowed=True),
            lambda x: x[0]['resources'].update(cpus=99),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                args=fixture();change(args)
                with self.assertRaises(ValueError):mod.output_path_proof(*args)

    def test_exact_mapping_rejects_path_decision_and_rpc_drift(self):
        changes = [
            dict(episode_directory='/scratch/alpine/fixture/cpu/worker/artifacts/campaign/policy-05/world-00/episode'),
            dict(episode_directory='/other/policy-05/world-00/episode'),
            dict(episode_directory='/gpfs/alpine1/scratch/fixture/cpu/worker/artifacts/campaign/policy-05/world-01/episode'),
            dict(decision_index=4),dict(session_request_id=4),dict(task_binding_index=1),
            dict(task_sha256='e'*64),dict(identity_sha256='e'*64),
            dict(status='other'),dict(scope='other'),dict(extra='unbound'),
        ]
        for change in changes:
            with self.subTest(change=change):
                args=gate_fixture();args[1].update(change)
                with self.assertRaises(ValueError):mapping_gate(*args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
