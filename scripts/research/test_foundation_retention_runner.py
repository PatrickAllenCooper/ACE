"""New runner boundary fixtures using artificial functions; no PFN model loads."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import foundation_retention_pilot as w
import foundation_mixture_selection as split
import foundation_retention_selection as ret

class Linear:
    def fit(self,x,y):
        self.coef=np.linalg.lstsq(np.column_stack((np.ones(len(x)),x[:,0])),y,rcond=None)[0]
        return self
    def predict(self,x):return self.coef[0]+self.coef[1]*x[:,0]
class RBFStub(Linear):
    def fit(self,x,y):
        self.center_=float(np.mean(x));self.scale_=float(np.std(x));self.target_center_=float(np.mean(y))
        return super().fit(x.reshape(-1,1),y)
    def __call__(self,x):return self.predict(np.asarray(x).reshape(-1,1))
def basis(x,f):
    x=np.asarray(x).reshape(-1)
    return np.column_stack((np.ones(len(x)),x,x*x)) if f=='quadratic' else np.column_stack((np.ones(len(x)),np.tanh(x) if f=='tanh' else x))
def eligible(train,clamp):return ((train[clamp!='M',0:1],train[clamp!='M',1]),(train[:,1:2],train[:,2]))
def setup_worker():
    w.np=np;w.sel=split;w.ret=ret;w.flex=types.SimpleNamespace(RBFControl=RBFStub)
    w.base=types.SimpleNamespace(FAMILIES=('linear','quadratic','tanh'),basis=basis,eligible=eligible,make_model=lambda *args:Linear())
class Runner(unittest.TestCase):
    def setUp(self):setup_worker()
    def test_fresh_identity_and_learner_private_boundary(self):
        with self.assertRaises(ValueError):w.specification(92000)
        with self.assertRaises(ValueError):w.specification(123456,True)
        spec=w.specification(223456,True);rows=w.history(spec,'null',223456,1)
        fit,_=split.split_history(rows);pre,_=w.fit_or_failure(fit,'grammar',Path('/unused'))
        with patch.object(w,'private_probes',side_effect=AssertionError('private access')):
            heads,choices,errors,events=w.make_predictors(rows,pre,Path('/unused'))
        self.assertFalse(errors);self.assertEqual(set(choices),set(w.METHODS[4:]))
        self.assertEqual([h.fit_rows for h in heads['rbf24']],[15,24])
        self.assertEqual([h.fit_rows for h in heads['pfn24']],[15,24])
        self.assertEqual(choices['combined_no_pfn'].names,ret.ORDER[:-1])
        self.assertEqual(choices['combined'].intervals[0],ret.Interval(-1.,1.))
    def test_failed_pfn_does_not_remove_candidate_from_four_head_selectors(self):
        spec=w.specification(223456,True);rows=w.history(spec,'null',223456,1)
        fit,_=split.split_history(rows);pre,_=w.fit_or_failure(fit,'grammar',Path('/unused'))
        def maker(method,checkpoint):
            if method=='tabpfn_v2':raise RuntimeError('injected pfn fit failure')
            return Linear()
        w.base.make_model=maker
        heads,choices,errors,events=w.make_predictors(rows,pre,Path('/unused'))
        self.assertEqual(set(errors),{'pfn24','raw','local','interval','combined'})
        self.assertEqual(set(choices),{'combined_no_pfn'})
        self.assertEqual(events['pfn24']['nodes'][1]['status'],'unattempted')
    def test_partition_and_mutation_failure_prevents_use(self):
        class BatchDependent(Linear):
            def predict(self,x):return np.full(len(x),len(x),dtype=float)
        w.base.make_model=lambda *args:BatchDependent()
        with self.assertRaisesRegex(ValueError,'partition'):w.Head('grammar',np.array([[0.],[1.]]),np.array([0.,1.]),Path('/unused'))
        class Mutating(Linear):
            calls=0
            def predict(self,x):self.calls+=1;return np.full(len(x),self.calls,dtype=float)
        w.base.make_model=lambda *args:Mutating()
        with self.assertRaisesRegex(ValueError,'repeat'):w.Head('grammar',np.array([[0.],[1.]]),np.array([0.,1.]),Path('/unused'))
    def test_full_artificial_layout_saved_parents_and_empty_support(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w.run(root,Path('/unused'),'fixture')
            complete=json.loads((root/'complete.json').read_text())
            self.assertEqual(complete['completed_cells'],36)
            self.assertEqual(complete['planned_cells'],36)
            self.assertEqual(complete['training_responses_total'],160)
            self.assertEqual(complete['private_responses_total'],3072)
            for row in complete['cells']:
                folder=root/str(row['seed'])/row['variant']
                parents=np.load(folder/(row['method']+'_parents.npy'))
                self.assertEqual(parents.shape,(256,1))
                self.assertEqual(w.sha(folder/(row['method']+'_parents.npy')),row['parent_sha256'])
                self.assertEqual(row['diagnostics']['local']['M_local']['outside'],{'count':0,'mse':None})
                self.assertEqual(row['diagnostics']['composed_Y_parent_inside']+row['diagnostics']['composed_Y_parent_outside'],256)
                seal=json.loads((folder/'selection_seal.json').read_text())
                reservation=json.loads((folder/'evaluation.reserved.json').read_text())
                self.assertLessEqual(seal['at_unix'],reservation['at_unix'])
                self.assertEqual(reservation['selection_seal_sha256'],w.sha(folder/'selection_seal.json'))

if __name__=='__main__':unittest.main(verbosity=2)
