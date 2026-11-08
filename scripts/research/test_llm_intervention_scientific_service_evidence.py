"""New stdlib-only scientific bounds and unchanged join-core comparison."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import types
import unittest

HERE=Path(__file__).parent
NEW=HERE/'llm_intervention_scientific_service_evidence.py'
OLD=HERE/'llm_intervention_campaign_service_evidence.py'
NAMES={'_need','_scientific_shape','_members'}
source=ast.parse(NEW.read_bytes())
body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
api=types.SimpleNamespace()
ns=dict(WORLD_IDS=tuple(range(202611070100,202611070124)),MAX_REQUESTS=576,re=re,
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest())
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(NEW),'exec'),ns)
api.__dict__.update(ns)

class ScientificServiceTests(unittest.TestCase):
    def fixture(self):
        policies=['uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm']
        return dict(scope='development',policy_ids=policies,
            policy_phases={p:'gpu' if p in policies[-3:] else 'cpu' for p in policies},
            services={p:dict(max_requests=576,identity=dict(fixture=p)) for p in policies[-3:]},
            sources=[dict(plan=dict(world=dict(scope='development',world_id=w),initial_actions=[None]*8,
                initial_cost_units=8,acquisition_cap_units=24,menu=[dict(cost_units=1)]*8)) for w in api.WORLD_IDS])

    def check(self,c):return api._scientific_shape(c,'dpo_llm',api.digest(c['services']['dpo_llm']['identity']))

    def test_full_accounting_all_arms_and_expensive_menu(self):
        c=self.fixture();self.assertEqual(self.check(c),(6144,6144))
        for s in c['sources']:s['plan']['menu']=[dict(cost_units=2)]*8
        self.assertEqual(self.check(c),(3840,6144))

    def test_scope_roster_cost_and_request_mutations_rejected(self):
        mutations=[lambda c:c.update(scope='artificial'),lambda c:c['sources'].reverse(),
            lambda c:c['sources'].pop(),lambda c:c['services']['dpo_llm'].update(max_requests=575),
            lambda c:c['services']['sft_llm'].update(max_requests=577),
            lambda c:c['sources'][0]['plan'].update(initial_cost_units=0),
            lambda c:c['sources'][0]['plan'].update(acquisition_cap_units=25),
            lambda c:c['sources'][0]['plan']['menu'][0].update(cost_units=0)]
        for mutate in mutations:
            c=copy.deepcopy(self.fixture());mutate(c)
            with self.assertRaises(ValueError):self.check(c)

    def test_ordinal_575_allowed_and_576_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'request-0575.json').write_text('{}')
            self.assertEqual(set(api._members(root,'request-*.json',r'request-(\d+)\.json')),{575})
            (root/'request-0576.json').write_text('{}')
            with self.assertRaises(ValueError):api._members(root,'request-*.json',r'request-(\d+)\.json')

    def test_original_generation_trajectory_and_custody_functions_unchanged(self):
        old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(OLD.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        new={n.name:ast.dump(n,include_attributes=False) for n in source.body if isinstance(n,ast.FunctionDef)}
        for name in ('_need','_read','_same','_inventory','_members','_prefix','_episode_plan','_arm','_generation'):
            self.assertEqual(new[name],old[name],name)
        self.assertNotIn('numpy',__import__('sys').modules)
        self.assertNotIn('torch',__import__('sys').modules)

if __name__=='__main__':unittest.main()
