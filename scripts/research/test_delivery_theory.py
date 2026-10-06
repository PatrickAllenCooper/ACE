import unittest
from delivery_theory import dag_bound,quantization_margin,holm,paired_log_ratio


class TheoryTests(unittest.TestCase):
    def test_chain_amplifies_local_error(self):
        # True x=1,y=2x,z=3y; learned y=2x+.1,z=3y+.2.
        b=dag_bound(['x','y','z'],{'x':[],'y':['x'],'z':['y']},
                    {'x':0,'y':.1,'z':.2},{('y','x'):2,('z','y'):3},['x'])
        self.assertAlmostEqual(b['z'],.5)
        self.assertAlmostEqual(abs((3*(2+.1)+.2)-6),b['z'])

    def test_fork_collider_and_clamp(self):
        pa={'x':[],'a':['x'],'b':['x'],'c':['a','b']}
        eps={'x':0,'a':.1,'b':.2,'c':.3}
        L={('a','x'):2,('b','x'):1,('c','a'):4,('c','b'):5}
        bound=dag_bound(list(pa),pa,eps,L,['x'])
        self.assertAlmostEqual(bound['c'],1.7)
        clamped=dag_bound(list(pa),pa,eps,L,['x','a'])
        self.assertEqual(clamped['a'],0);self.assertAlmostEqual(clamped['c'],1.3)

    def test_quantization_strict_margin(self):
        levels=[0.,1.,2.];self.assertEqual(quantization_margin(1.,levels),.5)
        self.assertEqual(quantization_margin(.5,levels),0)
        # Same squared error: .49 is right for true0, .51 is wrong for true1.
        self.assertEqual(min(levels,key=lambda v:abs(v-.49)),0.)
        self.assertEqual(min(levels,key=lambda v:abs(v-1.51)),2.)

    def test_more_data_can_hurt(self):
        # Constant hypothesis, true test response=0; adding noisy observation 2.
        self.assertLess(0**2,((0+2)/2)**2)

    def test_more_optimization_can_hurt(self):
        # Train on noisy y=1, test y=0; loss-minimizer overfits the noise.
        self.assertLess(.1**2,1**2)

    def test_observed_parents_do_not_certify_chain(self):
        # Child correct only at observed parent0; upstream predicts .1.
        child=lambda x:100*x
        self.assertEqual(child(0),0);self.assertEqual(child(.1),10)

    def test_root_support_ambiguity_amplifies_in_box_upstream_error(self):
        true=lambda r,m:r+m
        alternative=lambda r,m,k:r+m+k*(m-r)
        for k in (-100.,0.,1000.):
            for r in (-1.,-.25,0.,.25,1.):
                self.assertEqual(true(r,r),alternative(r,r,k))
        r,delta,k=.25,.01,1000.
        self.assertTrue(-1<=r<=1 and -1<=r+delta<=1)
        error=abs(alternative(r,r+delta,k)-true(r,r))
        self.assertAlmostEqual(error,10.01)
        self.assertAlmostEqual(error,abs((1+k)*delta))
        # An internal clamp off the diagonal separates the two mechanisms.
        self.assertNotEqual(true(r,.5),alternative(r,.5,k))

    def test_noisy_expectation_differs_from_exact_chain_of_means(self):
        # do(W=0), X=W+U, Y=X^2, U uniform on {-1,+1}.
        noise=[-1.,1.];conditional_mean_x=sum(noise)/2
        chain_prediction=conditional_mean_x**2
        interventional_mean=sum(u**2 for u in noise)/2
        self.assertEqual(chain_prediction,0.)
        self.assertEqual(interventional_mean,1.)

    def test_response_selection_biases_an_eligible_mechanism(self):
        # No clamp of X, measured parent W=0, mean-zero noise; retaining
        # positive-response probes alone destroys zero conditional mean.
        all_responses=[-1.,1.]
        selected=[x for x in all_responses if x>0]
        self.assertEqual(sum(all_responses)/len(all_responses),0.)
        self.assertEqual(sum(selected)/len(selected),1.)

    def test_holm_and_paired_unit(self):
        self.assertEqual(holm([.01,.04,.2]),[.03,.08,.2])
        r=paired_log_ratio([.2,.4,.3],[1.,2.,1.5])
        self.assertAlmostEqual(r['ratio'],.2);self.assertEqual(r['n_systems'],3)
        with self.assertRaises(ValueError):paired_log_ratio([1],[1,2])


if __name__=='__main__':unittest.main()
