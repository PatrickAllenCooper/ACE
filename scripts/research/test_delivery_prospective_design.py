import unittest
from delivery_prospective_design import structural_values,shared_actions,action_block,reserved


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


if __name__=='__main__':unittest.main()
