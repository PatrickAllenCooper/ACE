"""Positive relative B assembly using fabricated artifacts, never real outcomes.

Exercises full membership and actual planner/builder/verifier interfaces. Dummy
checkpoint/prediction bytes cannot qualify scientific inference or a runtime.
The only patched replay pins belong to this synthetic registration/contract.
"""
import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import build_delivery_release as builder
import extend_delivery_prospective_release_plan as planner
import prepare_delivery_prospective_release_core as core
import replay_delivery_prospective_release as replay
import test_delivery_prospective_release_core as fixtures
from verify_delivery_release import sha, verify


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    return sha(path)


def fixture(root):
    here = Path(__file__).resolve().parent
    study, source = root/'study', root/'source'
    study.mkdir(); source.mkdir()
    reg = json.loads((here.parents[1]/'results/delivery_prospective_preparation_20261006/full_registration.json').read_text())
    reg = copy.deepcopy(reg)
    reg.update(output='/synthetic/study', source='/synthetic/runner', project='/synthetic/project',
               scope='synthetic assembly only, no scientific inference')
    entries = {}
    def artifact(name, value=None, raw=None):
        p = root/'objects'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        if raw is None: put(p, value)
        else: p.write_bytes(raw)
        entries[name] = {'path': name, 'sha256': sha(p), 'sources': [str(p)]}
        return sha(p)
    frozen = []
    committed = {}
    for names, prefix in [(reg['worker_hashes'], 'scripts/research/'), (reg['generator_hashes'], '')]:
        for name in names:
            rel = prefix+name
            original = here.parents[1]/rel
            raw = original.read_bytes()
            names[name] = artifact('B/project/'+rel, raw=raw)
            committed[rel] = names[name]
            p = source/'B'/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
            frozen.append({'stage':'B', 'path':'B/'+rel, 'sha256':names[name],
                           'source_revision':core.REVISION, 'transformation':'identity'})
    reg['source_commit_receipt_sha256'] = artifact('B/project/source_commit_receipt.json',
        {'source_revision':core.REVISION, 'files':committed})
    # No archived learner imported or checkpoint serialized by this fixture.
    for name in reg['source_hashes']:
        reg['source_hashes'][name] = artifact('source/runner/'+name, raw=b'# synthetic learner placeholder\n')
    descriptors = {}
    for case in planner.WORLDS:
        size = int(case.split(':')[0]); folder = 'B/descriptors/'+case.replace(':','-')
        world = dict(size=size, seed=int(case.split(':')[1]), family='fixture', order=[],
                     parents={}, coefficients={}, noise_sd=0, source='/synthetic/generator')
        if size == 30: world['nonlinear_nodes'] = []
        descriptors[case] = {'world_sha256':artifact(folder+'/world.json', world),
                             'actions_sha256':artifact(folder+'/actions.json', {'fixture': True})}
    reg['descriptor_manifest_sha256'] = artifact('B/descriptor_manifest.json', {'worlds':descriptors})
    for name, key in planner.PROTOCOL_FILES.items():
        if name == 'descriptor_manifest.json': continue
        value = {'state':'retained synthetic failure', 'allocated_cpu_seconds':126} if name == 'historical_pilot_failure.json' else {'fixture':True}
        reg[key] = artifact('B/'+name, value)
    rh = artifact('B/registration.json', reg)
    put(study/'registration.json', reg)
    training, evaluation = {}, {}
    for case in planner.WORLDS:
        for history in planner.HISTORIES:
            folder = 'B/training/'+case.replace(':','-')+'/'+history
            training[case+'/'+history] = {n:artifact(folder+'/'+n, raw=(json.dumps({'attempt':1,'synthetic_bundle':folder+'/'+n})+'\n').encode())
                                         for n in ('input.json','queries.ndjson','receipt.json')}
        folder = 'B/evaluation/'+case.replace(':','-')
        evaluation[case] = {n:artifact(folder+'/'+n, raw=(json.dumps({'attempt':1,'synthetic_bundle':folder+'/'+n})+'\n').encode())
                            for n in ('input.json','queries.ndjson','receipt.json')}
    matrix, seal, scores = [], {}, {}
    primary = []
    for case in planner.WORLDS:
        attempts = []
        for history in planner.HISTORIES:
            for arm, init in planner.ARMS:
                i = len(matrix); folder = f'B/fits/{case.replace(":","-")}/{history}/{arm}-i{init}'
                binding = {'protocol_sha256':rh, 'input_sha256':training[case+'/'+history]['input.json'],
                           'kernel_sha256':reg['worker_hashes']['delivery_prospective_models.py'],
                           'source_hashes':reg['source_hashes'], 'dependencies':reg['dependencies']}
                cell = dict(index=i, case=case, graph_size=int(case.split(':')[0]), history=history,
                            arm=arm, init=init, binding=binding,
                            input_dir='/synthetic/study/training/'+case.replace(':','-')+'/'+history,
                            out='/synthetic/study/'+folder[2:])
                matrix.append(cell)
                model = artifact(folder+'/models.pt', raw=f'not a model; synthetic cell {i}'.encode())
                receipt = artifact(folder+'/receipt.json', dict(complete=True, binding=binding, arm=arm,
                    init=init, model_sha256=model, new_queries_in_fit=0, evaluation_responses_read=0,
                    finished_at='2026-10-07T00:00:00+00:00'))
                seal[cell['out']] = {'model_sha256':model, 'receipt_sha256':receipt}
                prediction = artifact(f'B/evaluation/{case.replace(":","-")}/cell-{i}-predictions.npz',
                                      raw=f'not predictions; synthetic cell {i}'.encode())
                scores[str(i)] = {'cell':cell, 'metric':{'synthetic':True},
                    'evaluation_input_sha256':evaluation[case]['input.json'], 'predictions_sha256':prediction}
                if init == 0 and arm in ('delivery','online','simpler'): primary.append({'synthetic':i})
                attempts.append({'cell_index':i, 'status':'complete', 'exit_code':0, 'peak_tree_rss_bytes':1024})
        folder = 'B/world_execution/'+case.replace(':','-')
        artifact(folder+'/complete.json', {'at':'2026-10-07T00:00:00+00:00','n_fits':16,'protocol_sha256':rh})
        artifact(folder+'/execution.json', {'attempts':attempts})
    mh = artifact('B/matrix.json', {'registration_sha256':rh,'cells':matrix})
    sh = artifact('B/fit_seal.json', {'registration_sha256':rh,'matrix_sha256':mh,'artifacts':seal})
    (study/'fit_seal.json').write_bytes(Path(entries['B/fit_seal.json']['sources'][0]).read_bytes())
    artifact('B/evaluation_started.json', {'at':'2026-10-07T00:00:01+00:00','fit_seal_sha256':sh})
    phases = {}
    for name in ('qualification','collect','evaluate'):
        phases[name+'_execution.json'] = artifact('B/'+name+'_execution.json', dict(status='complete',exit_code=0,
            account='ucb736_asc1',registration_sha256=rh,peak_tree_rss_bytes=1024,elapsed_seconds=1))
    log = artifact('B/qualification.log', raw=b'synthetic qualification placeholder\n')
    artifact('B/qualification_complete.json', dict(registration_sha256=rh,dependencies=reg['dependencies'],
        confirmation_responses_evaluated=0,execution_sha256=phases['qualification_execution.json'],log_sha256=log))
    artifact('B/collection_complete.json', dict(registration_sha256=rh,charged_responses=32000,
        matrix_sha256=mh,artifacts=training))
    score_hash = artifact('B/scores.json', dict(cells=scores, primary_analysis={'synthetic':True},
        primary_rows=primary,scope=reg['scope'],new_training_responses=32000,
        new_shared_evaluation_responses=16000,fit_seal_sha256=sh,evaluation_cpu_seconds=1,evaluation_wall_seconds=1))
    complete = dict(n_fits=640,n_primary_cells=240,charged_responses=48000,account='ucb736_asc1',
                    registration_sha256=rh,fit_seal_sha256=sh,scores_sha256=score_hash,fit_cpu_core_hours=0)
    ch = artifact('B/complete.json', complete); put(study/'complete.json', complete)
    scratch = root/'gate-fixture'; scratch.mkdir()
    _, _, accepted = fixtures.ReleaseCoreTests().fixture(scratch)
    accepted.update(study_registration_sha256=rh,complete_sha256=ch,fit_seal_sha256=sh,scores_sha256=score_hash,
        auditor_sha256=reg['worker_hashes']['audit_delivery_prospective_results.py'],scope=reg['scope'],
        training_receipt_hashes=training,evaluation_receipt_hashes=evaluation,phase_execution_hashes=phases,
        primary_analysis={'synthetic':True},secondary_descriptive_analysis={'synthetic':True})
    ah = artifact('B/acceptance.json', accepted); put(study/'acceptance.json', accepted)
    supervisor = dict(status='complete',exit_code=0,account='ucb736_asc1',registration_sha256=rh,
                      acceptance_sha256=ah,peak_tree_rss_bytes=1024,elapsed_seconds=1)
    artifact('B/audit_execution.json', supervisor); put(study/'audit_execution.json', supervisor)
    inv = root/'inventory.json'
    put(inv, dict(schema='delivery-prospective-composite-v1',completed=True,conflict_policy='reject',
                  conflicts=[],missing_indices=[],cells_verified=640,registration_sha256=rh,matrix_sha256=mh,
                  files=list(entries.values())))
    src_inv = root/'source-inventory.json'; src_hash = put(src_inv, {'protocol_hashes':{'B':rh},'files':frozen})
    code, segments = core.extracted_core(here/'audit_delivery_prospective_results.py',
        reg['worker_hashes']['audit_delivery_prospective_results.py'], here/'runner_delivery_confirmation.py',
        reg['worker_hashes']['runner_delivery_confirmation.py'])
    core_file = root/'prepared.py'; core_file.write_text(code)
    contract_file = root/'core-contract.json'
    cch = put(contract_file, dict(registration_sha256=rh,original_revision=core.REVISION,target_dependencies=reg['dependencies'],
        matrix_fits=640,primary_cells=240,frozen_source_inventory_sha256=src_hash,frozen_source_bindings_checked=20,
        source_guards_bypassed=False,B_outcomes_opened=False,new_fits=0,new_responses=0,
        original_auditor_sha256=reg['worker_hashes']['audit_delivery_prospective_results.py'],
        original_utility_sha256=reg['worker_hashes']['runner_delivery_confirmation.py'],
        preparation_and_gate_tool_sha256=sha(core.__file__),derived_core_sha256=sha(core_file),exact_extracted_segments=segments))
    inherited = root/'inherited.json'
    inherited_object = root/'prior.json'
    inherited_hash = put(inherited_object, {'synthetic_prior':True})
    # Carry the new A/C/F notice-plan interface into the real B planner. This
    # is deliberately synthetic metadata, not a substitution for candidate11.
    metadata = root/'prior-pyproject.toml'
    metadata.write_text('[tool.poetry]\nname = "fixture"\nauthors = ["Decisive AI Team"]\n'
                        'homepage = "https://fixture.invalid"\nrepository = "https://fixture.invalid/repo"\n'
                        'license = "MIT"\n[tool.poetry.dependencies]\npython = ">=3.11"\n')
    original_metadata = sha(metadata)
    metadata_transform = {'kind':'project-toml','omit':['tool.poetry.authors','tool.poetry.homepage','tool.poetry.repository']}
    projected_metadata = root/'projected-pyproject.toml'
    projected_metadata.write_bytes(builder.project_toml_bytes(metadata.read_bytes(),metadata_transform))
    prior_protocol = root/'prior-protocol.json'
    put(prior_protocol, {'source_hashes':{'pyproject.toml':original_metadata}})
    prior_notice = root/'prior-notice.json'
    put(prior_notice, {'original_sha256':original_metadata,'derived_sha256':sha(projected_metadata),
                      'public_release_approved':False,'runner_grant_authority_resolved':False})
    license_notice = root/'prior-license.txt'
    license_notice.write_text('Synthetic license notice only; not a redistribution grant.\n')
    put(inherited, {'files':[{'path':'F/prior-fixture.json','sources':[str(inherited_object)],
         'role':'synthetic-prior-artifact','original_sha256':inherited_hash,'transform':{'kind':'identity'}}],
         'bindings':[], 'status':'synthetic fixture only'})
    inherited_plan = json.loads(inherited.read_text())
    for name, path, role, transform in (
        ('source/runner/pyproject.toml',metadata,'derived-runtime-metadata',metadata_transform),
        ('F/protocol.json',prior_protocol,'synthetic-original-protocol',{'kind':'identity'}),
        ('notices/provenance.json',prior_notice,'synthetic-notice-provenance',{'kind':'identity'}),
        ('notices/fixture-license.txt',license_notice,'synthetic-license-notice',{'kind':'identity'})):
        inherited_plan['files'].append({'path':name,'sources':[str(path)],'role':role,
                                       'original_sha256':sha(path),'transform':transform})
    inherited_plan['bindings'] = [
        {'record':'F/protocol.json','pointer':'/source_hashes/pyproject.toml','artifact':'source/runner/pyproject.toml','digest':'original_sha256'},
        {'record':'notices/provenance.json','pointer':'/original_sha256','artifact':'source/runner/pyproject.toml','digest':'original_sha256'},
        {'record':'notices/provenance.json','pointer':'/derived_sha256','artifact':'source/runner/pyproject.toml','digest':'sha256'}]
    inherited.write_text(json.dumps(inherited_plan,sort_keys=True)+'\n')
    return dict(plan_file=inherited,prospective=study,inventory=inv,source_inventory=src_inv,source_root=source,
                core_file=core_file,core_contract=contract_file,expected_core_contract_sha256=cch,
                private_dir=root.parent/(root.name+'-private'),out=root.parent/(root.name+'-plan.json'),
                replay=here/'replay_delivery_prospective_release.py',expected_registration=rh)


class PositiveAssemblyTests(unittest.TestCase):
    def test_full_synthetic_planner_build_relocation_and_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); root = base/'inputs'; root.mkdir()
            args = fixture(root)
            input_hashes = {n:sha(args[n]) for n in ('plan_file','inventory','source_inventory','core_file','core_contract')}
            result = planner.extend(**args)
            self.assertEqual(input_hashes, {n:sha(args[n]) for n in input_hashes})
            self.assertEqual((result['fits'],result['worlds'],result['predictions']), (640,40,640))
            self.assertEqual((result['new_fits'],result['new_responses']), (0,0))
            destination = base/'package'
            built = builder.build(args['out'], destination, base/'private-build.json')
            manifest = json.loads((destination/'manifest.json').read_text())
            contract = json.loads((destination/'B/replay_contract.json').read_text())
            replay.byte_integrity(destination, manifest)
            # Independent membership oracle, not planner-reported counts.
            expected = {'F/prior-fixture.json','F/protocol.json','source/runner/pyproject.toml',
                        'notices/provenance.json','notices/fixture-license.txt','delivery_prospective_design.py',
                        'delivery_prospective_replay_core.py','replay_delivery_prospective_release.py',
                        'B/core_contract.json','B/replay_contract.json'}
            expected.update('B/'+n for n in ('registration.json','scores.json','acceptance.json','complete.json',
                'audit_execution.json','evaluation_started.json','matrix.json','fit_seal.json',
                'historical_pilot_failure.json','qualification_execution.json','collect_execution.json','evaluate_execution.json'))
            identity = set()
            for case in planner.WORLDS:
                case_path = case.replace(':','-')
                for history in planner.HISTORIES:
                    identity.update(f'B/training/{case_path}/{history}/{n}' for n in ('input.json','queries.ndjson','receipt.json'))
                    for arm, init in planner.ARMS:
                        identity.update(f'B/fits/{case_path}/{history}/{arm}-i{init}/{n}' for n in ('models.pt','receipt.json'))
                identity.update(f'B/evaluation/{case_path}/{n}' for n in ('input.json','queries.ndjson','receipt.json'))
                start = planner.WORLDS.index(case)*16
                identity.update(f'B/evaluation/{case_path}/cell-{i}-predictions.npz' for i in range(start,start+16))
                identity.add(f'B/descriptors/{case_path}/actions.json')
                expected.add(f'B/descriptors/{case_path}/world.json')
                expected.update(f'B/world_execution/{case_path}/{n}' for n in ('complete.json','execution.json'))
            original_inventory = json.loads(args['inventory'].read_text())
            originals = {e['path']:e['sha256'] for e in original_inventory['files']}
            identity.update(n for n in originals if n.startswith('source/runner/'))
            expected.update(identity)
            published = {e['path']:e['sha256'] for e in manifest['files']}
            self.assertEqual(set(published),expected)
            self.assertEqual({n:published[n] for n in identity},{n:originals[n] for n in identity})
            with patch.object(replay,'REGISTRATION',args['expected_registration']), \
                 patch.object(replay,'CORE_CONTRACT_SHA',args['expected_core_contract_sha256']):
                replay.gate(contract)
            self.assertTrue(json.loads((destination/'F/prior-fixture.json').read_text())['synthetic_prior'])
            inherited_plan = json.loads(args['plan_file'].read_text())
            prior_metadata = next(f for f in inherited_plan['files'] if f['path']=='source/runner/pyproject.toml')
            original_toml = tomllib.loads(Path(prior_metadata['sources'][0]).read_text())
            expected_toml = copy.deepcopy(original_toml)
            for field in ('authors','homepage','repository'): del expected_toml['tool']['poetry'][field]
            self.assertEqual(tomllib.loads((destination/'source/runner/pyproject.toml').read_text()),expected_toml)
            self.assertEqual(json.loads((destination/'F/protocol.json').read_text())['source_hashes']['pyproject.toml'],prior_metadata['original_sha256'])
            self.assertNotEqual(published['source/runner/pyproject.toml'],prior_metadata['original_sha256'])
            self.assertEqual(sha(destination/'notices/fixture-license.txt'),
                             next(f['original_sha256'] for f in inherited_plan['files'] if f['path']=='notices/fixture-license.txt'))
            self.assertEqual(len(contract['cells']),640)
            self.assertEqual(len(contract['world_attempt_elapsed_seconds_unknown']),640)
            self.assertEqual(len(contract['training']),80)
            self.assertEqual(len(contract['evaluation']),40)
            self.assertNotIn('input_dir',contract['cells'][0])
            self.assertNotIn('source', json.loads((destination/'B/descriptors/5-00/world.json').read_text()))
            original = json.loads((args['private_dir']/'original/B/scores.json').read_text())
            projected = json.loads((destination/'B/scores.json').read_text())
            self.assertIn('input_dir',original['cells']['0']['cell'])
            self.assertNotIn('input_dir',projected['cells']['0']['cell'])
            self.assertEqual(original['cells']['0']['metric'],projected['cells']['0']['metric'])
            moved = base/'renamed-package'; destination.rename(moved)
            self.assertEqual(verify(moved,built['manifest_sha256'])['files_verified'],result['files'])
            # Corruption after relocation is detected before model parsing.
            (moved/'B/fits/5-00/balanced_varied_value/delivery-i0/models.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact bytes changed'):
                replay.byte_integrity(moved, manifest)


if __name__ == '__main__': unittest.main()
