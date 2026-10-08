"""Fabricated source-transition metadata; no inference or real B outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_delivery_release as builder
import delivery_prospective_source_contract as successor
import extend_delivery_prospective_release_plan as planner
from extend_delivery_source_contract import ROLES, INTERFACES
from test_delivery_prospective_release_assembly import fixture, put
from verify_delivery_release import sha, verify


def add_predecessor(args, root):
    plan = json.loads(args['plan_file'].read_text())
    def add(name, value):
        path = root/'predecessor'/name
        digest = put(path, value)
        plan['files'].append(dict(path=name,sources=[str(path)],original_sha256=digest,
                                 role='synthetic-source-contract-fixture',transform={'kind':'identity'}))
        return digest
    notices = {name:add(name,{'synthetic_notice':name}) for name in
               ('notices/ACE_RUNNER_MIT.txt','notices/ACE_APACHE_2_0.txt')}
    files = {f['path']:f for f in plan['files']}
    interfaces = []
    for name,(role,scope,runtimes) in INTERFACES.items():
        digest = add(name,{'synthetic_interface':name})
        runtime_records = []
        for runtime in runtimes:
            if runtime not in files:
                h = add(runtime, {'synthetic_runtime':runtime,'private_fixture':'not released'})
                entry = plan['files'][-1]
                entry['transform'] = {'kind':'project-json','keys':['synthetic_runtime']}
                projected = root/'projected-fixture-runtime.json'
                projected.write_text(json.dumps({'synthetic_runtime':runtime},indent=2,sort_keys=True)+'\n')
                files[runtime] = {'original_sha256':h,'packaged_sha256':sha(projected)}
            runtime_records.append(dict(path=runtime,sha256=files[runtime].get('packaged_sha256',files[runtime]['original_sha256'])))
        interfaces.append(dict(path=name,sha256=digest,role=role,scope=scope,
                               runtime_records=runtime_records))
    frozen = {f['path']:f for f in json.loads(args['source_inventory'].read_text())['files']}
    rows = [dict(binding_id=str(i),original_location=name,stage=name[0],
                 original_sha256=frozen[name]['sha256'] if name in frozen else 'a'*64,
                 disposition='private-original-only',released_original_path=None,
                 released_original_sha256=None,anonymous_execution_qualified=False)
            for i,name in enumerate(ROLES)]
    contract = dict(schema='delivery-source-dispositions-v1',bindings=rows,interfaces=interfaces,
        B_included=False,training_reproduction_available=False,public_release_approved=False,
        anonymity_review_complete=False,runner_grant_authority_resolved=True,
        runner_notice_sha256=notices['notices/ACE_RUNNER_MIT.txt'],
        ACE_source_notice_sha256=notices['notices/ACE_APACHE_2_0.txt'])
    digest = add(successor.ACTIVE,contract)
    plan['files'][-1]['role'] = 'derived-source-disposition-contract'
    for i,interface in enumerate(interfaces):
        plan['bindings'].append(dict(record=successor.ACTIVE,pointer=f'/interfaces/{i}/sha256',
                                    artifact=interface['path'],digest='sha256'))
    historical = root/'historical-binding.json'
    put(historical,{'predecessor_sha256':digest})
    plan['files'].append(dict(path='reproduction/historical-binding.json',sources=[str(historical)],
        original_sha256=sha(historical),role='synthetic-digest-binding',transform={'kind':'identity'}))
    plan['bindings'].append(dict(record='reproduction/historical-binding.json',
        pointer='/predecessor_sha256',artifact=successor.ACTIVE,digest='sha256'))
    put(args['plan_file'],plan)
    args['expected_source_contract_sha256'] = digest
    return plan,contract,digest


class SourceTransitionTests(unittest.TestCase):
    def test_positive_full_fabricated_transition_build_and_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve(); root=base/'input'; root.mkdir()
            args=fixture(root); inherited,old,pin=add_predecessor(args,root)
            original_plan=sha(args['plan_file'])
            old_raw=Path(next(f for f in inherited['files'] if f['path']==successor.ACTIVE)['sources'][0]).read_bytes()
            result=planner.extend(**args)
            self.assertEqual(sha(args['plan_file']),original_plan)
            self.assertEqual(result['source_contract_transition']['interfaces'],6)
            destination=base/'package'
            built=builder.build(args['out'],destination,base/'private-build.json')
            active=json.loads((destination/successor.ACTIVE).read_text())
            self.assertEqual((destination/successor.PREDECESSOR).read_bytes(),old_raw)
            self.assertNotEqual(sha(destination/successor.ACTIVE),pin)
            self.assertEqual(active['schema'],'delivery-source-dispositions-v2')
            self.assertEqual(active['predecessor']['sha256'],pin)
            self.assertEqual(active['prepared_against_plan_sha256'],original_plan)
            self.assertTrue(active['B_included'])
            for flag in ('B_numerical_replay_qualified','public_release_approved',
                         'anonymity_review_complete','training_reproduction_available'):
                self.assertIs(active[flag],False)
            self.assertEqual({r['original_location']:r['original_sha256'] for r in active['bindings']},
                             {r['original_location']:r['original_sha256'] for r in old['bindings']})
            included=[r for r in active['bindings'] if r['released_original_path'] is not None]
            self.assertEqual([r['original_location'] for r in included],[successor.DESIGN])
            self.assertEqual(included[0]['released_original_sha256'],sha(destination/'delivery_prospective_design.py'))
            self.assertTrue(all(r['anonymous_execution_qualified'] is False for r in active['bindings']))
            self.assertEqual(active['derived_sources'][0]['source_bindings'],[successor.AUDITOR,successor.GUARD])
            self.assertEqual(active['B_upstream']['runtime_record_sha256'],sha(destination/'B/replay_contract.json'))
            manifest=json.loads((destination/'manifest.json').read_text())
            for original_edge in inherited['bindings']:
                expected=copy.deepcopy(original_edge)
                for key in ('record','artifact'):
                    if expected[key]==successor.ACTIVE: expected[key]=successor.PREDECESSOR
                self.assertIn(expected,manifest['bindings'])
            for i,interface in enumerate(active['interfaces']):
                self.assertEqual(interface['sha256'],sha(destination/interface['path']))
                self.assertIn(dict(record=successor.ACTIVE,pointer=f'/interfaces/{i}/sha256',
                                   artifact=interface['path'],digest='sha256'),manifest['bindings'])
                for j,runtime in enumerate(interface['runtime_records']):
                    self.assertEqual(runtime['sha256'],sha(destination/runtime['path']))
                    self.assertIn(dict(record=successor.ACTIVE,pointer=f'/interfaces/{i}/runtime_records/{j}/sha256',
                                       artifact=runtime['path'],digest='sha256'),manifest['bindings'])
            original_runtime=next(e for e in inherited['files'] if e['path']=='F/runtime.json')
            self.assertNotEqual(sha(destination/'F/runtime.json'),original_runtime['original_sha256'])
            moved=base/'relocated'; destination.rename(moved)
            self.assertEqual(verify(moved,built['manifest_sha256'])['files_verified'],result['files'])
            (moved/successor.PREDECESSOR).write_bytes(b'changed predecessor')
            with self.assertRaisesRegex(ValueError,'changed|digest|hash'):
                verify(moved,built['manifest_sha256'])

    def test_wrong_predecessor_pin_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'contract.json'; put(path,{'unused':True})
            plan={'files':[dict(path=successor.ACTIVE,sources=[str(path)],original_sha256=sha(path),
                role='derived-source-disposition-contract',transform={'kind':'identity'})]}
            with patch.object(planner,'snapshot',side_effect=AssertionError('decoded wrong pin')):
                with self.assertRaisesRegex(ValueError,'unsupported inherited'):
                    successor.preflight(plan,planner.snapshot,'b'*64)

    def test_original_gate_precedes_transition_or_input(self):
        with patch.object(planner.core,'accepted_before_scores',side_effect=ValueError('original blocked')):
            with patch.object(successor,'preflight',side_effect=AssertionError('transition before gate')):
                with self.assertRaisesRegex(ValueError,'original blocked'):
                    planner.extend('plan','study','inv','srcinv','src','core','cc','a'*64,'private','out')

    def test_predecessor_conflict_and_missing_adapter_before_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); args=fixture(root); plan,_,pin=add_predecessor(args,root)
            duplicate=copy.deepcopy(plan); duplicate['files'].append(dict(plan['files'][-1],path=successor.PREDECESSOR))
            with self.assertRaisesRegex(ValueError,'already transitioned'):
                successor.preflight(duplicate,planner.snapshot,pin)
            args['replay']=None
            original=planner.snapshot
            def guarded(path,*a,**kw):
                if Path(path).name=='scores.json': raise AssertionError('scores decoded before source gate')
                return original(path,*a,**kw)
            with patch.object(planner,'snapshot',guarded):
                with self.assertRaisesRegex(ValueError,'explicit B replay interface'):
                    planner.extend(**args)
            self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())

    def test_source_failures_reject_before_score_decode_and_outputs(self):
        cases=('notice','interface','runtime','B-source','missing-adapter','symlink-adapter','helper')
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve(); args=fixture(root); plan,old,pin=add_predecessor(args,root)
                if case in ('notice','interface','runtime'):
                    name={'notice':'notices/ACE_RUNNER_MIT.txt','interface':old['interfaces'][0]['path'],
                          'runtime':'F/runtime.json'}[case]
                    entry=next(e for e in plan['files'] if e['path']==name)
                    changed=root/'changed.json'; put(changed,{'changed':True})
                    entry.update(sources=[str(changed)],original_sha256=sha(changed),transform={'kind':'identity'})
                    put(args['plan_file'],plan)
                elif case=='B-source':
                    old['bindings'][4]['original_sha256']='b'*64
                    entry=next(e for e in plan['files'] if e['path']==successor.ACTIVE)
                    pin=put(Path(entry['sources'][0]),old); entry['original_sha256']=pin
                    put(args['plan_file'],plan); args['expected_source_contract_sha256']=pin
                elif case=='missing-adapter':
                    args['replay']=root/'missing.py'
                elif case=='symlink-adapter':
                    alias=root/'alias.py'; alias.symlink_to(args['replay']); args['replay']=alias
                elif case=='helper':
                    changed=root/'wrong-helper.py'; changed.write_text('# incorrect identity source\n')
                    plan['files'].append(dict(path='delivery_prospective_design.py',sources=[str(changed)],
                        original_sha256=sha(changed),role='original-safe-helper',transform={'kind':'identity'}))
                    put(args['plan_file'],plan)
                original=planner.snapshot
                def guarded(path,*a,**kw):
                    if Path(path).name=='scores.json': raise AssertionError('scores decoded before source failure')
                    return original(path,*a,**kw)
                with patch.object(planner,'snapshot',guarded):
                    with self.assertRaises((ValueError,FileNotFoundError)):
                        planner.extend(**args)
                self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())


if __name__=='__main__': unittest.main()
