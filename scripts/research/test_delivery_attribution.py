import unittest
from delivery_attribution import reconstruct,eligible


def row(q,role,step=None,selected=False,do=None):
    do=do or {}
    return {'query_index':q,'method':'ace','role':role,'step':step,'selected':selected,
            'params':{'x':float(q)},'intermediates':{},'outcome':float(q),
            'interventions':do,'intervention_target':next(iter(do),None)}


class ReplayTests(unittest.TestCase):
    def test_final_refresh_is_not_used_and_winner_duplicates(self):
        rows=[row(0,'seed'),row(1,'lookahead',1,True,{'y':1}),row(2,'obs_refresh',1)]
        m={'query_counts':{'ace':{'seed':1,'lookahead':1,'obs_refresh':1,'total':3,'executed':1}},
           'node_mlps':['y'],'total_steps':1}
        r=reconstruct(rows,m)
        self.assertEqual(r['online_admitted'],[0,1]);self.assertEqual(r['final_buffer'],[0,1,2])
        self.assertEqual(r['final_buffer_never_admitted'],[2]);self.assertEqual(r['online_use_counts'][1],2)
        self.assertEqual(r['clamped_fast_adapt_updates'],{'y':1})

    def test_masks_and_paid_counter(self):
        m={'feature_names':['x'],'target_name':'y','causal_dag':{'x':[],'y':['x']}}
        rows=[row(0,'seed'),row(1,'lookahead',1,True,{'y':1}),row(2,'lookahead',2,True,{'x':2})]
        self.assertEqual(eligible(rows,m,'y')[1],[0,2]);self.assertEqual(eligible(rows,m,'flat')[1],[0,2])
        with self.assertRaises(ValueError):reconstruct(rows,{'query_counts':{'ace':{'total':1}}})

    def test_duplicate_indices_rejected(self):
        rows=[row(0,'seed'),row(0,'seed')]
        with self.assertRaises(ValueError):reconstruct(rows,{'query_counts':{'ace':{'seed':2,'total':2}},'node_mlps':[],'total_steps':0})


if __name__=='__main__':unittest.main()
