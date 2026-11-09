"""NEW artificial saved-metadata cases only; no production imports or main calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pure_supplement', ROOT/'audit_supplement.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


class View:
    def __init__(self):
        self.data = {}; self.opaque = {}; self.full = {}; self.p = {}; self.packet_pin = 'a'*64

    def j(self, n):
        return copy.deepcopy(self.data[n])

    def h(self, n, expected=None):
        value = self.opaque[n] if n in self.opaque else s.sha(s.raw(self.data[n]))
        if expected is not None:
            s.need(value == expected, 'artificial inventory pin differs')
        return value


def oracle():
    diagnostic = (ROOT/'oracles'/s.DIAGNOSTIC).read_bytes()
    helper = (ROOT/'oracles/llm_intervention_service_runtime.py').read_bytes()
    return dict(generated_module=s.constant(diagnostic, 'GENERATED_MODULE'),
        cache_pairs=s.constant(diagnostic, 'CACHE_PAIRS'), provenance_sources=s.constant(diagnostic, 'PROVENANCE_SOURCES'),
        bootstrap=s.constant(diagnostic, 'BOOTSTRAP'), original_runtime=s.constant(diagnostic, 'RUNTIME_SOURCE'),
        extra_packages=s.constant(helper, 'EXTRA_PACKAGES'))


def runtime_fixture():
    v = View(); o = oracle(); prefix = '/fabricated/runtime'; tmp = '/fabricated/service-tmp-123'; folder = tmp+'/tmpfake'
    module_path = folder+'/'+o['generated_module']+'.py'; text = '# fabricated saved source, never executed\n'
    versions = dict(numpy='2.2.5', scipy='1.15.3', torch='2.9.1+cu128', transformers='4.57.3',
                    safetensors='0.7.0', huggingface_hub='0.36.0', tokenizers='0.22.1', jinja2='3.1.6')
    candidates = {k.replace('_', '-'): [val] for k, val in versions.items()}; candidates['numpy'] += ['2.2.6']
    contract = dict(archive_sha256='b'*64, manifest_sha256='c'*64, imported_versions=versions, metadata_candidates=candidates)
    stage = dict(version='ace-runtime-archive-v1', archive_sha256='b'*64, manifest_sha256='c'*64,
        staged_prefix=prefix, objects=49966, file_bytes=8397575073, model_loaded=False, runtime_qualified=False,
        cache_source_sha256={n: s.sha(n.encode()) for pair in o['cache_pairs'] for n in pair})
    v.p = dict(runtime=contract)
    v.data[s.CPU+'stage.json'] = stage
    root_stat = dict(dev=1, ino=2, uid=123, gid=123, mode=stat.S_IFDIR | 0o700)
    bound = dict(stage_sha256=v.h(s.CPU+'stage.json'), stage=stage, contract=contract, gpu_mapping_enabled=False,
        temporary_root=tmp, temporary_root_stat=root_stat)
    v.data[s.ART+'runtime-stage-binding.json'] = bound
    provenance = dict(expected_source=text, expected_source_sha256=s.sha(text.encode()), sources={
        n: dict(path=prefix+'/lib/python3.10/site-packages/'+relative, sha256=h)
        for n, (relative, h) in o['provenance_sources'].items()})
    v.data[s.CPU+'independent-provenance-sources.json'] = provenance
    v.data[s.ART+'runtime-provenance-sources.json'] = provenance
    before = dict(prefix=prefix, base_prefix=prefix, executable=prefix+'/bin/python', python_version='3.10.19',
        sys_path=[prefix+'/lib/python3.10'], packages={}, mixed_module_paths=[], mixed_native_paths=[], generated_provenance={'present': False})
    v.data[s.ART+'runtime-before-imports.json'] = before
    proof = dict(present=True, verified=True, temporary_root=tmp, directory=folder, path=module_path,
        module_file=module_path, spec_origin=module_path, owner_type='tempfile.TemporaryDirectory',
        temporary_root_stat=root_stat, directory_stat=root_stat,
        file_stat=dict(uid=123, mode=stat.S_IFREG | 0o600, nlink=1, size=len(text.encode())),
        source_pins={n: r['sha256'] for n, r in provenance['sources'].items() if n.startswith('torch.')},
        expected_source_sha256=provenance['expected_source_sha256'], source_sha256=provenance['expected_source_sha256'], source_content=text,
        function_origins={n: dict(module=o['generated_module'], code_filename=module_path) for n in ('forward', 'forward_async', '_remote_forward')})
    observed = dict(before, packages={n: dict(version=versions[n]) for n in ('numpy', 'scipy', 'torch', 'transformers')},
        additional_package_versions={n: versions[n] for n in o['extra_packages']}, sys_path=before['sys_path']+[folder],
        generated_provenance=proof, mixed_modules={o['generated_module']: module_path}, mixed_module_paths=[module_path])
    for n in ('runtime-after-service-imports.json', 'runtime-after-cpu-phase.json'):
        v.data[s.ART+n] = copy.deepcopy(observed)
    distributions = [dict(name=n, version=val, metadata_path=prefix+'/metadata/'+n+'/'+val,
                          distribution_root=prefix+'/site-packages') for n, vals in candidates.items() for val in vals]
    v.data[s.ART+'runtime-metadata.json'] = dict(distributions=distributions, numpy_entries=[{'version': x} for x in ('2.2.5', '2.2.6')],
        resolved_defaults={n: dict(metadata_path=prefix+'/metadata/'+n) for n in candidates})
    v.data[s.ART+'cache-reconciliation.json'] = dict(no_cache_modified=True, pairs=[dict(cache_path=prefix+'/'+a,
        source_path=prefix+'/'+b, cache_sha256=stage['cache_source_sha256'][a], source_sha256=stage['cache_source_sha256'][b],
        matches_stage_pins=True, admissible_for_diagnosis=True, cache_executed_by_comparison=False,
        cpython_timestamp_reject_expected=True, code_equivalent=None) for a, b in o['cache_pairs']])
    return v, o


def bundle_fixture():
    v, o = runtime_fixture(); rows = []; bindings = []; worlds = {}
    p = v.p
    p.update(cpu_output='/fabricated/campaign/cpu', source_commit='d'*40,
        files={n: s.sha(n.encode()) for n in s.SCREEN_SOURCES | {s.ENTRY}},
        world_pins={}, numeric_timeout_seconds=55, services={}, service_identity_pins={}, persistent_specs={},
        protocol_source_pins={'toy.py': 'e'*64}, outer_provenance_receipt_sha256='f'*64)
    arms = list(s.POLICIES[:4])+['numerical_ranker', 'llm_frozen', 'llm_sft', 'llm_dpo']
    for wi, wid in enumerate(s.WORLD_IDS):
        # Explicit small hand-built descriptors only; no world generator or response.
        world = dict(world_id=wid, scope='development', graph=[{'name': 'x', 'parents': []}], coefficients=[0, 1])
        worlds[wid] = world; p['world_pins'][str(wid)] = s.digest(world)
        v.data['launch/inputs/'+f'descriptor-{wi:03d}.json'] = world
        public = dict(graph=world['graph'], task={'target': 'x', 'queries': []}, menu=[{'action_id': 'observe_all', 'cost_units': 1}],
            learner=dict(noise_variances={'x': 1}, prior_variance=1), initial_actions=['observe_all']*8,
            initial_cost_units=8, objective='fabricated', domain_notes='fabricated', additional_cost_units=24)
        row = dict(world_id=wid, public=public, initial_binding=dict(request_namespace='initial-'+str(wid), replicate_id='shared'),
            regime='regime-'+str(wi%4), graph_stratum='graph', cost_stratum='cost-'+str(wi%2), development_episodes=[
                dict(arm=arm, measurement_binding=dict(request_namespace='measurement-'+str(wid)+'-'+arm, replicate_id='r')) for arm in arms])
        rows.append(row)
        for pid in s.POLICIES[:4]:
            binding = dict(world_id=wid, policy_id=pid, seed=109)
            bindings.append(dict(world_id=wid, policy_id=pid, binding=binding, binding_sha256=s.digest(binding)))
    for pid in s.POLICIES[5:]:
        p['services'][pid] = dict(identity={'role': pid}, max_requests=576)
        p['service_identity_pins'][pid] = s.digest(p['services'][pid]['identity'])
        p['persistent_specs'][pid] = dict(policy_id=pid, kind='persistent', command=['fabricated-service', pid],
            timeout_seconds=55, identity_sha256=p['service_identity_pins'][pid], policy_provenance_sha256='b'*64)
    proposal = {'development_worlds': rows}; v.data['launch/inputs/proposal.json'] = proposal
    o['proposal_sha256'] = s.digest(proposal); o['proposal_byte_sha256'] = v.h('launch/inputs/proposal.json')
    v.data['launch/inputs/numeric-bindings.json'] = bindings
    v.data[s.CPU+'worker.started.json'] = {'deadline_monotonic': 15940.0}
    v.data[s.PHASE+'ranker/training_complete.json'] = {'fabricated': True}
    training_pin = v.h(s.PHASE+'ranker/training_complete.json')
    v.opaque[s.PHASE+'ranker/dpo.ranker.pt'] = 'c'*64
    ranker = dict(exports={'dpo': dict(path='dpo.ranker.pt', sha256='c'*64, training_receipt_sha256=training_pin)})
    v.data[s.PHASE+'ranker/terminal.json'] = ranker
    prefixes = s.expected_prefixes(v, o); v.data[s.ART+'numeric-prefixes.json'] = prefixes
    export = dict(path=p['cpu_output']+'/worker/artifacts/cpu-phase/ranker/dpo.ranker.pt', sha256='c'*64, training_receipt_sha256=training_pin)
    specs = s.policy_specs(p, bindings, prefixes, export, v.h(s.PHASE+'ranker/terminal.json'))
    sources = []
    for row in rows:
        wid = row['world_id']; public = row['public']
        plan = dict(world=worlds[wid], world_sha256=p['world_pins'][str(wid)], task=public['task'], menu=public['menu'],
            learner=dict(graph=public['graph'], noise_variances=public['learner']['noise_variances'], prior_variance=public['learner']['prior_variance']),
            initial_actions=public['initial_actions'], initial_cost_units=8, acquisition_cap_units=24,
            initial_binding=row['initial_binding'], measurement_bindings={pid: e['measurement_binding'] for pid, e in zip(s.POLICIES, row['development_episodes'])},
            policy_specs=specs[wid], objective=public['objective'], domain_notes=public['domain_notes'],
            outer_provenance_receipt_sha256=p['outer_provenance_receipt_sha256'], scientific_provenance_qualified=False, retry_allowed=False)
        sources.append(dict(plan=copy.deepcopy(plan), plan_sha256=s.digest(plan), strata={k: row[k] for k in ('regime', 'graph_stratum', 'cost_stratum')}))
    c = dict(scope='development', sources=sources, strata_fields=['regime', 'graph_stratum', 'cost_stratum'],
        policy_ids=list(s.POLICIES), policy_phases={pid: 'cpu' if pid in s.POLICIES[:5] else 'gpu' for pid in s.POLICIES},
        services=p['services'], scientific_provenance_qualified=False, retry_allowed=False)
    bundle = dict(campaign=c, campaign_sha256=s.digest(c), scope='development', proposal_sha256=o['proposal_sha256'],
        numeric_policy_bindings=bindings, service_identity_pins=p['service_identity_pins'], source_pins=p['protocol_source_pins'],
        world_pins=[dict(world_id=w, sha256=p['world_pins'][str(w)]) for w in s.WORLD_IDS],
        worlds_generated=0, responses_generated=0, proposal_arm_mapping=dict(zip(s.POLICIES, arms)),
        source_plan_pins=[dict(world_id=x['plan']['world']['world_id'], sha256=x['plan_sha256']) for x in sources])
    for k in ('execution_authorized', 'scientific_protocol_frozen', 'scientific_provenance_qualified', 'source_qualified', 'runtime_qualified', 'scheduler_qualified'):
        bundle[k] = False
    v.data[s.PHASE+'campaign-bundle.json'] = bundle
    refresh(v)
    for pi, pid in enumerate(s.POLICIES[:5]):
        for wi, wid in enumerate(s.WORLD_IDS):
            member = s.CAMPAIGN+f'policy-{pi:02d}/world-{wi:02d}/episode/decision-0000.process.json'
            v.data[member] = {'command': copy.deepcopy(specs[wid][pi]['command'])}; v.full[member] = {'sha256': v.h(member), 'bytes': len(s.raw(v.data[member]))}
    return v, o


def refresh(v):
    bundle = v.data[s.PHASE+'campaign-bundle.json']; c = bundle['campaign']
    for source in c['sources']:
        source['plan_sha256'] = s.digest(source['plan'])
    bundle['campaign_sha256'] = s.digest(c)
    bundle['source_plan_pins'] = [dict(world_id=x['plan']['world']['world_id'], sha256=x['plan_sha256']) for x in c['sources']]
    v.data[s.CAMPAIGN+'campaign.json'] = copy.deepcopy(c)
    v.data[s.CAMPAIGN+'source-plan-pins.json'] = copy.deepcopy(bundle['source_plan_pins'])
    v.data[s.PHASE+'terminal.json'] = {'campaign_sha256': bundle['campaign_sha256']}
    for pi in range(5):
        v.data[s.CAMPAIGN+f'policy-{pi:02d}/attempt.json'] = dict(campaign_sha256=bundle['campaign_sha256'], source_plan_pins=bundle['source_plan_pins'])


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix='fixture-', dir=ROOT/'attempt01'))

    def test_runtime_missing_mixed_cache_and_generated_provenance(self):
        v, o = runtime_fixture(); result = s.audit_runtime(v, o)
        self.assertTrue(result['saved_runtime_predicates_passed'])
        cases = {
            'missing-package': lambda x: x.data[s.ART+'runtime-after-cpu-phase.json']['packages'].pop('torch'),
            'mixed-native': lambda x: x.data[s.ART+'runtime-after-cpu-phase.json']['mixed_native_paths'].append('/ambient/lib.so'),
            'changed-search-path': lambda x: x.data[s.ART+'runtime-before-imports.json']['sys_path'].append('/ambient'),
            'missing-numpy-metadata': lambda x: x.data[s.ART+'runtime-metadata.json']['numpy_entries'].pop(),
            'mixed-metadata': lambda x: x.data[s.ART+'runtime-metadata.json']['distributions'][0].update(metadata_path='/ambient/dist-info'),
            'modified-cache': lambda x: x.data[s.ART+'cache-reconciliation.json'].update(no_cache_modified=False),
            'missing-cache-pair': lambda x: x.data[s.ART+'cache-reconciliation.json']['pairs'].pop(),
            'cache-executed': lambda x: x.data[s.ART+'cache-reconciliation.json']['pairs'][0].update(cache_executed_by_comparison=True),
            'cache-pin': lambda x: x.data[s.ART+'cache-reconciliation.json']['pairs'][0].update(source_sha256='0'*64),
            'generated-source': lambda x: x.data[s.ART+'runtime-after-cpu-phase.json']['generated_provenance'].update(source_content='changed'),
            'generated-origin': lambda x: x.data[s.ART+'runtime-after-cpu-phase.json']['generated_provenance'].update(spec_origin='/ambient/generated.py'),
        }
        outcomes = {}
        for name, mutate in cases.items():
            altered = copy.deepcopy(v); mutate(altered)
            with self.subTest(name=name), self.assertRaises((ValueError, KeyError)):
                s.audit_runtime(altered, o)
            outcomes[name] = 'rejected'
        (self.fixture/'runtime-case-results.json').write_bytes(s.raw(outcomes))

    def test_exact24_semantics_reject_rehashed_plan_menu_checkpoint_namespace_changes(self):
        v, o = bundle_fixture(); result = s.audit_bundle(v, o)
        self.assertEqual((result['worlds'], result['declared_cells'], result['numeric_policy_bindings']), (24, 192, 96))
        def plan(x): return x.data[s.PHASE+'campaign-bundle.json']['campaign']['sources'][0]['plan']
        cases = {
            'menu-after-rehash': lambda x: plan(x)['menu'][0].update(cost_units=2),
            'task-after-rehash': lambda x: plan(x)['task'].update(target='different'),
            'graph-after-rehash': lambda x: plan(x)['learner']['graph'][0].update(name='different'),
            'checkpoint-after-rehash': lambda x: plan(x)['policy_specs'][4]['command'].__setitem__(-5, '/wrong/sft.ranker.pt'),
            'namespace-after-rehash': lambda x: plan(x)['measurement_bindings'][s.POLICIES[0]].update(replicate_id='changed'),
            'coefficients-hash-only': lambda x: plan(x)['world']['coefficients'].__setitem__(0, 1),
            'stratum-after-rehash': lambda x: x.data[s.PHASE+'campaign-bundle.json']['campaign']['sources'][0]['strata'].update(regime='changed'),
            'policy-order-after-rehash': lambda x: plan(x)['policy_specs'].reverse(),
            'changed-original-deadline': lambda x: x.data[s.CPU+'worker.started.json'].update(deadline_monotonic=15941.0),
        }
        outcomes = {}
        for name, mutate in cases.items():
            altered = copy.deepcopy(v); mutate(altered); refresh(altered)
            with self.subTest(name=name), self.assertRaises((ValueError, KeyError)):
                s.audit_bundle(altered, o)
            outcomes[name] = 'rejected'
        # Equal plan bytes are insufficient if a recorded executed command differs.
        altered = copy.deepcopy(v); member = next(iter(altered.full)); altered.data[member]['command'] = ['wrong-process']
        with self.assertRaisesRegex(ValueError, 'executed decision argv'):
            s.audit_bundle(altered, o)
        (self.fixture/'bundle-case-results.json').write_bytes(s.raw(outcomes))
        (self.fixture/'artificial-success-facts.json').write_bytes(s.raw(result))

    def test_unfinished_or_different_aid_report_blocks_before_evidence_access(self):
        report = dict(machine_checks_complete=True, errors=[], acceptance=False, scientific_qualified=False,
            audit_source_sha256=s.AID_SHA, config_sha256='a'*64,
            completed_checks=['custody', 'packet', 'ancestor', 'accounting', 'runtime', 'ranker', 'attempts', 'numeric', 'terminal'])
        def forbidden(*args): self.fail('evidence accessed before the prior-aid gate')
        aid = types.SimpleNamespace(Audit=forbidden)
        for update in ({'machine_checks_complete': False}, {'config_sha256': 'b'*64},
                       {'audit_source_sha256': 'c'*64}, {'completed_checks': ['custody']}, {'errors': [{'failed': True}]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                s.open_completed_aid(aid, {}, 'a'*64, dict(report, **update))


if __name__ == '__main__':
    unittest.main(verbosity=2)
