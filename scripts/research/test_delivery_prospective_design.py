import unittest
from delivery_prospective_design import structural_values,shared_actions,action_block,reserved,world_spec,validate_world


class ProspectiveDesignTests(unittest.TestCase):
    def fixture(self):
        return {'size':5,'order':['X1','X4','X2','X3','X5'],
            'parents':{'X1':[],'X4':[],'X2':['X1'],'X3':['X1','X2'],'X5':['X4']},
            'coefficients':{'a':2.,'b':1.,'c':.5,'d':1.,'s':1.,'q':.2}}

    def test_base_equations_and_correct_clamp(self):
        import math
        s=self.fixture();v=structural_values(s,{'X1':1.,'X4':2.})
        self.assertEqual(v['X2'],3.);self.assertAlmostEqual(v['X3'],.5-3+math.sin(3));self.assertAlmostEqual(v['X5'],.8)
        w=structural_values(s,{'X1':1.,'X4':2.,'X2':4.},{'X2':100.})
        self.assertEqual(w['X2'],4.);self.assertAlmostEqual(w['X3'],.5-4+math.sin(4))

    def test_blocks_reserved_before_collection(self):
        s=self.fixture();train=shared_actions(s,'random',17,100);test=shared_actions(s,'evaluation',18,100)
        tb={action_block(list(a.values())) for a in train};eb={action_block(list(a.values())) for a in test}
        self.assertFalse(tb&eb);self.assertTrue(all(not reserved(b) for b in tb));self.assertTrue(all(reserved(b) for b in eb))

    def test_balanced_values_have_bounded_quota_imbalance(self):
        from collections import Counter
        actions=shared_actions(self.fixture(),'balanced',19,100)
        for root in ('X1','X4'):
            counts=Counter(a[root] for a in actions)
            self.assertEqual(len(counts),11);self.assertLessEqual(max(counts.values())-min(counts.values()),2)

    def test_large_generator_is_deterministic_without_global_rng_damage(self):
        import numpy as np
        from pathlib import Path
        project=Path(__file__).resolve().parents[2]
        state=np.random.get_state()
        expected=np.random.random(3);np.random.set_state(state)
        # Qualification fixture seed is not a selected prospective world.
        a=world_spec(30,0,project);b=world_spec(30,0,project)
        self.assertEqual(a,b)
        self.assertTrue(np.array_equal(np.random.random(3),expected))
        self.assertEqual(len(validate_world(a)),5)

    def test_world_rejects_graph_and_mechanism_drift(self):
        import copy
        from pathlib import Path
        s=world_spec(5,0,Path(__file__).resolve().parents[2]);validate_world(s)
        bad=copy.deepcopy(s);bad['parents']['X2']=['X3']
        with self.assertRaises(ValueError):validate_world(bad)
        bad=copy.deepcopy(s);bad['coefficients']['q']=float('nan')
        with self.assertRaises(ValueError):validate_world(bad)


if __name__=='__main__':unittest.main()
