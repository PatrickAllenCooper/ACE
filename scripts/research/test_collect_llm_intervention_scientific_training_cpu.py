"""NEW artificial glue tests; real collector and scientific draws never run.

All descriptors have explicitly handwritten zero coefficients. Fake collector
outputs exercise wrapper contracts, not the closed collector implementation.
Retain every input/output under ACE_SCIENTIFIC_COLLECTION_TEST_OUTPUT.
"""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parent))
import collect_llm_intervention_scientific_training_cpu as worker


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(raw(value)[:-1]).hexdigest()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw(value))


ORIGINAL_CAPTURE = worker._captured


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'tokenizers', 'safetensors'}:
            raise AssertionError('model/tokenizer import forbidden: '+fullname)
        return None


@contextmanager
def guarded_capture(source):
    with ORIGINAL_CAPTURE(source):
        import llm_intervention_world as world
        import llm_intervention_scm as scm
        import llm_intervention_contract as contract
        import llm_intervention_training_collection as collection
        import llm_intervention_training_dataset as dataset
        import llm_intervention_target_risk as risk
        import numpy as np
        def forbidden(*args, **kwargs):
            raise AssertionError('real draw/response/collection/update forbidden')
        with ExitStack() as stack:
            for obj, name in [(world, 'generate_world'), (world, '_normal'), (world, '_evaluate'),
                    (world, 'private_structural_means'), (world.ResponseAdapter, '__init__'),
                    (world, 'training_callbacks'), (scm.LinearGaussianSCM, 'update'),
                    (contract.ChargedExecutor, '__init__'), (collection, 'collect_training_batch'),
                    (collection, 'assemble_training_dataset'), (dataset, 'assemble_training_dataset'),
                    (risk, 'choose_target_risk'), (np.random, 'default_rng')]:
                stack.enter_context(patch.object(obj, name, forbidden))
            yield


def fixture(source):
    """Whole proposal, 32 supplied zero-law fake descriptors; no realization."""
    source_pins = {n: sha(b) for n, b in source.items()}
    with guarded_capture(source):
        from llm_intervention_screen_protocol import build_proposal, assembly_recipe
        from llm_intervention_training_collection import configuration_payload
        from llm_intervention_world import WorldDescriptor, LinearMechanism
        protocol_pins = {n: source_pins[n] for n in worker.PROTOCOL_SOURCES}
        proposal = build_proposal(source_pins=protocol_pins)
        inputs = {'proposal.json': raw(proposal)}
        worlds, configurations, entries = {}, {}, []
        for i, row in enumerate(proposal['training_worlds']):
            wid = row['world_id']
            config = assembly_recipe(proposal, world_id=wid, expected_source_pins=protocol_pins)
            graph = config['assembly_parameters']['graph']
            world = WorldDescriptor('policy_train', wid, graph, tuple(LinearMechanism(n.name, 0.,
                tuple((p, 0.) for p in n.parents), row['public']['learner']['noise_variances'][n.name]) for n in graph))
            inputs[f'descriptor-{i:03d}.json'] = raw(world.private_payload())
            worlds[str(wid)] = world.sha256
            configurations[str(wid)] = digest(configuration_payload(config))
            entries.append({'scope': 'policy_train', 'world_id': wid, 'path': f'world-{i:03d}.json',
                'status': 'complete', 'descriptor_sha256': world.sha256, 'file_sha256': sha(inputs[f'descriptor-{i:03d}.json'])})
        for i, row in enumerate(proposal['development_worlds'], 32):
            # Metadata only. No development descriptor is constructed or staged.
            entries.append({'scope': 'development', 'world_id': row['world_id'], 'path': f'world-{i:03d}.json',
                'status': 'complete', 'descriptor_sha256': digest(['fake dev', i]), 'file_sha256': digest(['fake bytes', i])})
        realization = {'version': 'ace-first-screen-descriptor-realization-v1', 'status': 'complete',
            'proposal_sha256': digest(proposal), 'source_pins': protocol_pins,
            'freeze_sha256': digest('fabricated freeze'), 'namespace_admission_sha256': digest('fabricated namespace'),
            'retry_allowed': False, 'environment_responses': 0, 'policy_updates': 0,
            'scientific_execution_qualified': False, 'worlds_generated': 56, 'entries': entries}
        inputs['realization-terminal.json'] = raw(realization)
    packet = {'schema': worker.VERSION, 'scope': 'policy_train', 'files': {n: sha(b) for n,b in inputs.items()},
        'source_pins': source_pins, 'proposal_sha256': digest(proposal), 'world_roster': list(worker.WORLD_IDS),
        'roster_sha256': digest(worker.WORLD_IDS), 'world_pins': worlds, 'configuration_pins': configurations,
        'freeze_sha256': realization['freeze_sha256'], 'namespace_admission_sha256': realization['namespace_admission_sha256']}
    return packet, inputs


def admission(packet, output):
    return {'schema': worker.ADMISSION_VERSION, 'stage': 'policy_train_collection', 'status': 'admitted',
        'packet_sha256': sha(raw(packet)), 'output': str(output), 'freeze_sha256': packet['freeze_sha256'],
        'namespace_admission_sha256': packet['namespace_admission_sha256'],
        'realization_terminal_sha256': packet['files']['realization-terminal.json'],
        'source_runtime_scheduler_qualified': True, 'namespace_owned': True, 'accounting_admitted': True,
        'max_response_attempts': 9216, 'max_reserved_cost_units': 10240, 'gpus': 0}


def fake_collection(**kw):
    """Fabricated inventory/accounting boundary. Never executes old collection."""
    import llm_intervention_training_collection as collection
    out = kw['output'];out.mkdir()
    planned = read(out.parent/'planned-records.json')
    source_pins = kw['expected_source_pins']
    worlds, observed, accounts = [], [], {}
    for i, wid in enumerate(worker.WORLD_IDS):
        config = kw['configurations'][wid]
        menu = config['assembly_parameters']['menu']
        journal_plan = collection._journal_plan(config)
        # Explicit all-observe trajectory: 8 initial + 24 acquisitions; no values.
        account = {}
        reservations = []
        for kind, count, cost in [('trajectory', 32, 32), ('branch', 128, 16*sum(a.cost_units for a in menu)), ('scoring', 128, 128)]:
            account[kind] = dict(reserved_cost_units=cost, reserved_response_attempts=count,
                known_returns=count, executor_accepted_responses=count, unknown_return_attempts=0)
        reservations = [dict(kind='trajectory', request_id=j, action_id='observe_all', cost_units=1) for j in range(32)]
        account.update(total_reserved_cost_units=sum(account[k]['reserved_cost_units'] for k in ('trajectory','branch','scoring')),
            reserved_response_attempts=288, known_returns=288, executor_accepted_responses=288, unknown_return_attempts=0,
            reservations=reservations, accounting_complete=True, accounting_errors=[], counts_are_lower_bounds=False)
        accounts[f'world-{i:03d}'] = account
        worlds.append(dict(world_id=wid, path=f'world-{i:03d}', status='complete', journal_plan=journal_plan, accounting=account))
        save(out/f'world-{i:03d}'/'fabricated-accounting.json', account)
    for row in planned:
        record = {'fixture_only': True, 'record_id': row['record_id'], 'preferences': row['pairs']}
        receipt = {'fixture_only': True, 'world_id': row['world_id'], 'pairs': len(row['pairs'])}
        save(out/row['path']/'record.json', record);save(out/row['path']/'receipt.json', receipt)
        observed.append(dict(row, status='complete', record_sha256=sha(raw(record)), supervision_receipt_sha256=sha(raw(receipt))))
    plan = dict(version=collection.VERSION, world_roster=list(worker.WORLD_IDS), worlds=worlds, records=planned,
        maximum_response_attempts=9216, maximum_reserved_cost_units=10240, source_file_sha256=source_pins)
    save(out/'plan.json', plan)
    inventory = dict(version=collection.VERSION, scope='policy_train', record_roster=[r['record_id'] for r in planned],
        records=observed, record_pins={r['record_id']:r['record_sha256'] for r in observed},
        supervision_receipt_pins={r['record_id']:r['supervision_receipt_sha256'] for r in observed}, collection_plan_sha256=sha(raw(plan)))
    save(out/'inventory.json', inventory)
    terminal = dict(version=collection.VERSION, scope='policy_train', status='complete', world_roster=list(worker.WORLD_IDS),
        worlds=worlds, record_roster=inventory['record_roster'], planned_records=planned,
        inventory_sha256=sha(raw(inventory)), accounting={k:sum(a[k] for a in accounts.values()) for k in worker.ACCOUNT_FIELDS},
        accounting_complete=True, accounting_totals_are_lower_bounds=False, exposure_ready=True, retry_allowed=False,
        partial_records_authorized_for_training=False, complete_case_filtering_allowed=False,
        scientific_evaluation_performed=False, policy_training_performed=False)
    save(out/'terminal.json', terminal)
    def inventory_boundary(root, planned_rows, supplied_worlds):
        # Only the old matched-record validator is stubbed. Inspect the complete
        # fabricated bytes independently so dropped pairs/record bytes surface.
        for row in observed:
            r = read(root/row['path']/'record.json')
            assert r['preferences'] == row['pairs']
            assert sha((root/row['path']/'record.json').read_bytes()) == row['record_sha256']
        return copy.deepcopy(observed)
    collection._inventory = inventory_boundary
    collection._world_accounting = lambda directory, journal_plan: read(directory/'fabricated-accounting.json')
    return terminal


class ScientificCollectionGlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ACE_SCIENTIFIC_COLLECTION_TEST_OUTPUT'])
        cls.root.mkdir(exist_ok=False)
        save(cls.root/'ARTIFICIAL-ONLY.json', {'scientific_realization': False, 'real_collection_calls': 0,
            'responses': 0, 'models': 0, 'description': 'handwritten zero laws and stubbed collection evidence only'})
        cls.source = {n: (Path(worker.__file__).parent/Path(n).name).read_bytes() for n in worker.SOURCE_PATHS}
        cls.packet, cls.inputs = fixture(cls.source)

    def setUp(self):
        self.base = self.root/self._testMethodName;self.base.mkdir()
        self.p, self.inputs = copy.deepcopy((self.packet, self.__class__.inputs))
        self.guard = patch.object(worker, '_captured', guarded_capture);self.guard.start();self.addCleanup(self.guard.stop)
        self.no_models = NoModels();sys.meta_path.insert(0,self.no_models);self.addCleanup(sys.meta_path.remove,self.no_models)
        self.calls = []

    def params(self, label='run'):
        output = self.base/label
        a = admission(self.p, output)
        return dict(packet_raw=raw(self.p), packet_sha256=sha(raw(self.p)), inputs=self.inputs,
            admission_raw=raw(a), admission_sha256=sha(raw(a)), output=output)

    def run_fake(self, args, fake=fake_collection):
        def wrapped(**kw):
            self.calls.append(kw)
            return fake(**kw)
        with patch.object(worker, '_invoke_collection', wrapped):
            return worker.run_collection(**args)

    def repin_input(self, name, value):
        self.inputs[name] = raw(value);self.p['files'][name] = sha(self.inputs[name])

    def reject(self, label):
        args = self.params(label)
        with self.assertRaises(ValueError):self.run_fake(args)
        self.assertEqual(self.calls, [])
        self.assertFalse(read(args['output']/'failure.json')['collection_invoked'])

    def test_complete_supplied_roster_delegates_once_retains_all_pairs_and_is_exclusive(self):
        args = self.params();result = self.run_fake(args)
        self.assertEqual(len(self.calls),1)
        kw = self.calls[0]
        self.assertEqual(kw['world_roster'], worker.WORLD_IDS)
        self.assertEqual(set(kw['worlds']), set(worker.WORLD_IDS))
        self.assertTrue(all(m.intercept == 0 and all(v == 0 for _,v in m.coefficients) for w in kw['worlds'].values() for m in w.mechanisms))
        self.assertEqual((result['record_count'],result['retained_pair_count']), (160,3584))
        self.assertEqual(result['accounting']['total_reserved_cost_units'],10240)
        self.assertEqual(result['accounting']['reserved_response_attempts'],9216)
        self.assertEqual(len(list((args['output']/'collection').glob('world-*/snapshot-*/record.json'))),160)
        before = {str(p):sha(p.read_bytes()) for p in args['output'].rglob('*') if p.is_file()}
        with self.assertRaises(FileExistsError):self.run_fake(args)
        self.assertEqual(before,{str(p):sha(p.read_bytes()) for p in args['output'].rglob('*') if p.is_file()})
        self.assertEqual(len(self.calls),1)
        self.assertFalse(any(n.startswith('llm_intervention_') for n in sys.modules))

    def test_independent_pins_source_and_exact_input_membership_gate_imports(self):
        cases=['packet_pin','admission_pin','missing_source','changed_source','missing_descriptor','extra_development','raw_descriptor_drift']
        for label in cases:
            self.p,self.inputs=copy.deepcopy((self.packet,self.__class__.inputs))
            if label=='missing_source':self.p['source_pins'].pop('scripts/research/llm_intervention_target_risk.py')
            if label=='changed_source':self.p['source_pins']['scripts/research/llm_intervention_world.py']='0'*64
            if label=='missing_descriptor':self.inputs.pop('descriptor-031.json')
            if label=='extra_development':self.inputs['development.json']=b'{}'
            if label=='raw_descriptor_drift':self.inputs['descriptor-031.json']+=b' '
            args=self.params(label)
            if label=='packet_pin':args['packet_sha256']='0'*64
            if label=='admission_pin':args['admission_sha256']='0'*64
            with self.subTest(label=label),patch.object(worker,'_captured',side_effect=AssertionError('imports preceded admission')):
                with self.assertRaises(ValueError):self.run_fake(args)
            self.assertEqual(self.calls,[])

    def test_stage_namespace_accounting_and_output_admission_are_specific(self):
        changes=[('status','pending'),('stage','tokenization'),('namespace_owned',False),
            ('source_runtime_scheduler_qualified',False),('accounting_admitted',False),('gpus',1),
            ('max_reserved_cost_units',10239),('max_response_attempts',9215),('output','/wrong/output'),
            ('freeze_sha256','0'*64),('namespace_admission_sha256','0'*64),('realization_terminal_sha256','0'*64)]
        for key,value in changes:
            args=self.params(key);a=json.loads(args['admission_raw']);a[key]=value
            args.update(admission_raw=raw(a),admission_sha256=sha(raw(a)))
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_fake(args)
        self.assertEqual(self.calls,[])

    def test_complete_realization_identity_order_and_original_namespace_are_required(self):
        changes=[('partial',lambda t:t.update(status='failed')),
            ('missing',lambda t:t['entries'].pop()),('order',lambda t:t['entries'].reverse()),
            ('namespace',lambda t:t.update(namespace_admission_sha256='0'*64)),
            ('freeze',lambda t:t.update(freeze_sha256='0'*64)),
            ('unfinished',lambda t:t['entries'][55].update(status='unstarted')),
            ('descriptor_pin',lambda t:t['entries'][31].update(descriptor_sha256='0'*64))]
        for label,change in changes:
            self.p,self.inputs=copy.deepcopy((self.packet,self.__class__.inputs))
            t=json.loads(self.inputs['realization-terminal.json']);change(t);self.repin_input('realization-terminal.json',t)
            with self.subTest(label=label):self.reject(label)

    def test_all_descriptors_and_configuration_pins_checked_before_collection(self):
        for label in ['scope','identity','known_noise','configuration','world_membership','proposal_namespace']:
            self.p,self.inputs=copy.deepcopy((self.packet,self.__class__.inputs))
            if label in ('scope','identity','known_noise'):
                name='descriptor-031.json';world=json.loads(self.inputs[name])
                if label=='scope':world['scope']='development'
                elif label=='identity':world['world_id']=202611070100
                else:world['mechanisms'][0]['noise_variance']=9.
                self.repin_input(name,world);self.p['world_pins'][str(worker.WORLD_IDS[-1])]=digest(world)
                t=json.loads(self.inputs['realization-terminal.json'])
                t['entries'][31].update(file_sha256=sha(raw(world)),descriptor_sha256=digest(world))
                self.repin_input('realization-terminal.json',t)
            elif label=='configuration':self.p['configuration_pins'][str(worker.WORLD_IDS[-1])]='0'*64
            elif label=='world_membership':self.p['world_pins']['202611070100']='0'*64
            else:
                prop=json.loads(self.inputs['proposal.json']);prop['training_worlds'][-1]['training']['request_namespace']='changed'
                self.repin_input('proposal.json',prop);self.p['proposal_sha256']=digest(prop)
                t=json.loads(self.inputs['realization-terminal.json']);t['proposal_sha256']=digest(prop)
                self.repin_input('realization-terminal.json',t)
            with self.subTest(label=label):self.reject(label)

    def test_saved_counts_missing_pairs_and_charged_failures_cannot_publish_success(self):
        for label in ['missing_record','missing_pair','branch_charge','unknown_return','initial_charge','returned_summary']:
            args=self.params(label)
            def corrupt(**kw):
                result=fake_collection(**kw);root=kw['output']
                if label in ('missing_record','missing_pair'):
                    inv=read(root/'inventory.json')
                    if label=='missing_record':inv['records'].pop()
                    else:inv['records'][0]['pairs'].pop()
                    (root/'inventory.json').write_bytes(raw(inv));result['inventory_sha256']=sha(raw(inv))
                elif label!='returned_summary':
                    a=result['worlds'][0]['accounting']
                    if label=='branch_charge':a['branch']['reserved_cost_units']-=1
                    if label=='unknown_return':a['branch']['unknown_return_attempts']=1
                    if label=='initial_charge':a['reservations'][0]['action_id']='set_x_neg'
                    (root/'world-000/fabricated-accounting.json').write_bytes(raw(a))
                (root/'terminal.json').write_bytes(raw(result))
                if label=='returned_summary':result['accounting']['total_reserved_cost_units']-=1
                return result
            with self.subTest(label=label),self.assertRaises(ValueError):self.run_fake(args,corrupt)
            self.assertFalse((args['output']/'terminal.json').exists())
            self.assertTrue((args['output']/'collection/terminal.json').exists())
            self.assertEqual(read(args['output']/'failure.json')['stage'],'saved_evidence')

    def test_original_failed_attempt_and_unknown_charge_survive_first_exception(self):
        args=self.params()
        original=raw({'status':'failed','accounting':{'total_reserved_cost_units':2,'unknown_return_attempts':1}})
        def fail(**kw):
            kw['output'].mkdir();(kw['output']/'failure.json').write_bytes(original)
            (kw['output']/'trajectory.jsonl').write_bytes(b'fabricated durable reservation; unknown return\n')
            raise RuntimeError('original collector failure')
        with self.assertRaisesRegex(RuntimeError,'original collector failure'):self.run_fake(args,fail)
        self.assertEqual((args['output']/'collection/failure.json').read_bytes(),original)
        failure=read(args['output']/'failure.json')
        self.assertTrue(failure['collection_invoked']);self.assertNotIn('accounting',failure)
        self.assertFalse((args['output']/'terminal.json').exists())
        self.assertEqual(len(self.calls),1)

    def test_initial_publication_failures_are_preserved_before_collection(self):
        original=worker._write
        for name in ['intent.json','packet.json','admission.json']:
            args=self.params(name)
            def fail(path,data):
                if path.name==name:raise OSError('initial publication '+name)
                return original(path,data)
            with patch.object(worker,'_write',fail),self.assertRaisesRegex(OSError,'initial publication'):
                self.run_fake(args)
            self.assertFalse(read(args['output']/'failure.json')['collection_invoked'])
        self.assertEqual(self.calls,[])

    def test_terminal_publication_ambiguity_preserves_committed_terminal(self):
        args=self.params();link=os.link
        def interrupt(source,dest):
            link(source,dest)
            raise KeyboardInterrupt('after terminal link')
        with patch.object(worker.os,'link',interrupt),self.assertRaises(KeyboardInterrupt):self.run_fake(args)
        self.assertTrue((args['output']/'terminal.json').exists())
        self.assertFalse((args['output']/'failure.json').exists())
        self.assertEqual(read(args['output']/'publication-interruption.json')['status'],'publication_interrupted_reconciliation_required')


if __name__=='__main__':
    unittest.main(verbosity=2)
