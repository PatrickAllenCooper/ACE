"""Artificial arithmetic teachers only; no fitted or pretrained models."""
import copy
import math
import unittest
from dataclasses import FrozenInstanceError
from foundation_fixed_mechanisms import RetainedParameters, compile_table, grid, restore


class FixedMechanismTests(unittest.TestCase):
    def test_linear_oracle_and_exact_knots(self):
        old=RetainedParameters('linear',[1,2])
        table=compile_table(lambda x:[3+4*u for u in x],[-1,1],old,'rbf')
        self.assertEqual(len(table.knots),129)
        self.assertEqual(table(table.knots),table.values)
        self.assertEqual(table([-.75,0,.125]),(0,3,3.5))

    def test_quadratic_interpolation_remainder_oracle(self):
        table=compile_table(lambda x:[u*u for u in x],[-1,1],RetainedParameters('linear',[0,0]),'pfn')
        for a,b in zip(table.knots,table.knots[1:]):
            mid=(a+b)/2
            self.assertEqual(table.scalar(mid)-mid*mid,(b-a)**2/4)

    def test_teacher_called_once_no_reference_or_batch_dependence(self):
        calls=[]
        def teacher(x):
            calls.append(tuple(x))
            return [u+len(x) for u in x]
        table=compile_table(teacher,[-1,1],RetainedParameters('linear',[0,1]),'pfn')
        x=(-2,-.4,0,.6,2)
        y=table(x)
        self.assertEqual(table(tuple(reversed(x))),tuple(reversed(y)))
        self.assertEqual(table(x[:2])+table(x[2:]),y)
        self.assertEqual(table((x[2],))[0],y[2])
        self.assertEqual(len(calls),1)
        self.assertNotIn('teacher',vars(table))

    def test_copy_and_frozen_state(self):
        coefficients=[2,3];old=RetainedParameters('linear',coefficients);coefficients[0]=200
        returned=[4.]*129
        table=compile_table(lambda x:returned,[-1,1],old,'grammar');pin=table.digest();returned[0]=900
        self.assertEqual(table.scalar(-1),4)
        self.assertEqual(table.scalar(2),8)
        self.assertEqual(table.digest(),pin)
        with self.assertRaises(FrozenInstanceError):table.values=(0,)

    def test_outside_retention_and_boundary_jump(self):
        old=RetainedParameters('quadratic',[1,2,3])
        table=compile_table(lambda x:[100.]*len(x),[-1,1],old,'rbf')
        self.assertEqual(table.scalar(1),100)
        right=math.nextafter(1,math.inf)
        self.assertEqual(table.scalar(right),old.scalar(right))
        self.assertEqual(table.scalar(-2),9)

    def test_degenerate_interval(self):
        table=compile_table(lambda x:[7], [2,2],RetainedParameters('tanh',[1,2]),'grammar')
        self.assertEqual(table.knots,(2,))
        self.assertEqual(table.scalar(2),7)
        self.assertEqual(table.scalar(0),1)

    def test_pinned_roundtrip_and_corruption(self):
        table=compile_table(lambda x:[u*u for u in x],[-1,1],RetainedParameters('linear',[0,1]),'pfn')
        restored=restore(table.payload(),table.digest())
        self.assertEqual(restored.digest(),table.digest())
        self.assertEqual(restored([-.9,0,1.5]),table([-.9,0,1.5]))
        bad=copy.deepcopy(table.payload());bad['values'][0]=(3.).hex()
        with self.assertRaises(ValueError):restore(bad,table.digest())

    def test_failures_do_not_fallback(self):
        old=RetainedParameters('linear',[0,1])
        for teacher in (lambda x:[0],lambda x:[float('nan')]*len(x),lambda x:[True]*len(x)):
            with self.assertRaises(ValueError):compile_table(teacher,[-1,1],old,'pfn')
        def failing(x):raise RuntimeError('teacher failure')
        with self.assertRaises(RuntimeError):compile_table(failing,[-1,1],old,'pfn')
        with self.assertRaises(ValueError):old.scalar(float('inf'))
        with self.assertRaises(ValueError):RetainedParameters('quadratic',[1,1,1]).scalar(1e308)

    def test_grid_rejection(self):
        for parents in ([],[True,1],[0,float('inf')],[-1e308,1e308],[1,math.nextafter(1,math.inf)]):
            with self.assertRaises(ValueError):grid(parents)

if __name__=='__main__':unittest.main()
