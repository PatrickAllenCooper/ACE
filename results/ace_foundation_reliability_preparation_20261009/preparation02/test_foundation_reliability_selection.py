"""Artificial loss fixtures only; no model fit, environment or scientific seed."""
import copy
import math
import unittest
from itertools import product
from foundation_reliability_selection import select, bounded_squared_loss, NAMES, OBJECTIVES


def fixture(n, names=NAMES):
    return {pair: {o: [0.9] * n for o in OBJECTIVES} for pair in product(names, repeat=2)}


def set_head(x, name, node, value):
    obj = ('M_local', 'Y_local')[node]
    for pair in x:
        if pair[node] == name:
            x[pair][obj] = [value] * len(x[pair][obj])


class ReliabilityTests(unittest.TestCase):
    def test_bounded_loss(self):
        self.assertEqual(bounded_squared_loss(2, 0, 16), .25)
        self.assertEqual(bounded_squared_loss(1e308, -1e308, 1), 1)
        self.assertEqual(bounded_squared_loss(0, 0, 1e-300), 0)
        for bad in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError): bounded_squared_loss(1, 0, bad)

    def test_small_n_cannot_certify_even_best_possible_difference(self):
        x=fixture(8)
        for p in x:
            for o in OBJECTIVES: x[p][o]=[1.0]*8
        set_head(x, 'grammar', 0, 0.0)
        x[('grammar','retained')]['Y_composed']=[0.0]*8
        self.assertGreater(math.sqrt(2*math.log(2700)/8), 1)
        self.assertTrue(select(x,budget=8)['fallback'])
        self.assertFalse(select(x,budget=8,rule='empirical')['fallback'])

    def test_large_margin_passes_with_exact_retained_local_identity(self):
        x=fixture(128);set_head(x,'grammar',0,0.1)
        x[('grammar','retained')]['Y_composed']=[.1]*128
        r=select(x,budget=128)
        self.assertEqual(r['selected'],['grammar','retained'])
        rec=next(z for z in r['records'] if z['pair']==r['selected'])
        self.assertEqual(rec['objectives']['Y_local']['upper'],0)
        self.assertAlmostEqual(rec['objectives']['M_local']['upper'],-.8+math.sqrt(2*math.log(2700)/128))

    def test_sample_tie_is_not_identity(self):
        x=fixture(128);x[('grammar','retained')]['Y_composed']=[.1]*128
        self.assertTrue(select(x,budget=128)['fallback'])
        self.assertFalse(select(x,budget=128,rule='empirical')['fallback'])

    def test_local_harm_blocks_composed_gain(self):
        x=fixture(128);set_head(x,'grammar',0,1)
        x[('grammar','retained')]['Y_composed']=[0]*128
        self.assertTrue(select(x,budget=128,rule='empirical')['fallback'])

    def test_retention_first_then_declared_name_ties(self):
        x=fixture(128)
        for name in ('grammar','rbf'):
            set_head(x,name,0,.1);set_head(x,name,1,.1)
            x[(name,'retained')]['Y_composed']=[.1]*128
            x[(name,name)]['Y_composed']=[.1]*128
        self.assertEqual(select(x,budget=128)['selected'],['grammar','retained'])

    def test_missing_or_malformed_candidate_invalidates(self):
        original=fixture(32)
        x=copy.deepcopy(original);del x[('pfn','pfn')]
        with self.assertRaises(ValueError):select(x,budget=32)
        for bad in (float('nan'),float('inf'),-0.1,1.1,True):
            x=copy.deepcopy(original);x[('pfn','pfn')]['Y_composed'][0]=bad
            with self.assertRaises(ValueError):select(x,budget=32)
        x=copy.deepcopy(original);x[('pfn','pfn')]['Y_composed'].append(.9)
        with self.assertRaises(ValueError):select(x,budget=32)
        with self.assertRaises(ValueError):select(original,budget=16)

    def test_inconsistent_local_head_rejected(self):
        x=fixture(32);x[('grammar','retained')]['Y_local'][0]=.1
        with self.assertRaises(ValueError):select(x,budget=32)

    def test_no_pfn_preserves_multiplicity(self):
        x=fixture(128,NAMES[:-1]);r=select(x,budget=128,include_pfn=False)
        self.assertEqual(len(r['records']),9)
        self.assertEqual(r['simultaneous_tests'],135)
        self.assertAlmostEqual(r['epsilon'],math.sqrt(2*math.log(2700)/128))

if __name__=='__main__': unittest.main()
