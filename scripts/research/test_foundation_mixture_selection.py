"""Artificial selector fixtures, not the proposed92000–92005 worlds."""
import unittest
import foundation_mixture_selection as m


def calibration(y=1):
    return tuple(m.Row(1,123,y,c) for c in ('none','none','X','X','X'))+tuple(m.Row(999,999,-999,'M') for _ in range(3))

class Selection(unittest.TestCase):
    def test_split_and_eligibility_counts(self):
        rows=tuple(m.Row(i,i,i,c) for i,c in enumerate(('none',)*8+('X',)*12+('M',)*12))
        fit,cal=m.split_history(rows)
        self.assertEqual(len(fit),24);self.assertEqual(len(cal),8)
        self.assertEqual(sum(r.clamp!='M' for r in fit),15)
        self.assertEqual(set(fit)&set(cal),set())
        self.assertEqual(set(fit)|set(cal),set(rows))
        self.assertEqual(len(m.root_calibration(cal)[0]),5)
    def test_terminal_complementary_error(self):
        g=lambda x:[0]*len(x);f=lambda x:[2]*len(x);identity=lambda x:x
        choice=m.terminal_choice(calibration(),g,identity,f,identity)
        self.assertEqual(choice.weights,(.5,));self.assertEqual(choice.mse,0)
        self.assertEqual(len(choice.candidates),5)
    def test_mechanism_choice_uses_predicted_not_measured_parents(self):
        g=lambda x:[0]*len(x);f=lambda x:[2]*len(x);identity=lambda x:x
        choice=m.mechanism_choice(calibration(),g,identity,f,identity)
        self.assertEqual(choice.weights,(.5,0));self.assertEqual(choice.mse,0)
        changed=tuple(m.Row(r.x,-777,r.y,r.clamp) for r in calibration())
        self.assertEqual(choice,m.mechanism_choice(changed,g,identity,f,identity))
        self.assertEqual(len(choice.candidates),25)
    def test_ties_favor_grammar_and_grid_order(self):
        identity=lambda x:x
        self.assertEqual(m.terminal_choice(calibration(),identity,identity,identity,identity).weights,(0,))
        self.assertEqual(m.mechanism_choice(calibration(),identity,identity,identity,identity).weights,(0,0))
    def test_nodewise_counterexample(self):
        gm=lambda x:[.9]*len(x);fm=lambda x:[2]*len(x)
        gy=lambda x:[2*v for v in x];fy=lambda x:[.5*v for v in x]
        self.assertAlmostEqual(m.terminal_forecast([1],gm,gy,fm,fy,.5)[0],1.4)
        y=m.mechanism_forecast([1],gm,gy,fm,fy,(.5,.5))[0]
        self.assertAlmostEqual(y,1.8125);self.assertGreater((y-1)**2,.64)
    def test_bad_vectors_and_layout_fail(self):
        with self.assertRaises(ValueError):m.root_calibration(calibration()[:-1])
        with self.assertRaises(ValueError):m.predict(lambda x:[float('nan')],(1,))
        with self.assertRaises(ValueError):m.predict(lambda x:[1],(1,2))
        with self.assertRaises(ValueError):m.terminal_forecast([1],lambda x:x,lambda x:x,lambda x:x,lambda x:x,.3)
        with self.assertRaises(ValueError):m.mse((float('inf'),),(0,))
        with self.assertRaises(ValueError):m.mse((),())
        with self.assertRaises(ValueError):m.mse((1e200,),(0,))

if __name__=='__main__':unittest.main()
