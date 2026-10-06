import copy
import math
import tempfile
from pathlib import Path
import unittest

from audit_delivery_prospective_results import (WORLD_IDS,HISTORIES,ARMS,canonical_cells,
    primary_statistics,secondary_summary,replay_predictions,calibration_ranges,history_receipt,match)
from delivery_prospective_analysis import analyze
from delivery_prospective_models import runtime
from runner_delivery_confirmation import read,write,sha,Journal
from test_delivery_prospective_models import SOURCE


class ResultsAuditTests(unittest.TestCase):
    def test_independent_statistics_keep_world_unit_history_logs_and_holm(self):
        rows=[]
        for case in WORLD_IDS:
            size=int(case.split(':')[0]);i=int(case.split(':')[1])
            for history,d,c in (('balanced_varied_value',.1,1.),('matched_random',90.,100.)):
                for arm,v in (('delivery',d*math.exp(i/20)),('online',c),('simpler',c*1.2)):
                    rows.append({'graph_size':size,'system_id':case,'history':history,'arm':arm,
                        'init':0,'nmse':v,'status':'complete'})
        computed=primary_statistics(rows)
        match(computed,analyze(rows,{s:[c for c in WORLD_IDS if c.startswith(str(s)+':')] for s in (5,30)}),'primary')
        self.assertAlmostEqual(computed['contrasts'][0]['ratio'],.3*math.exp(.475))
        for bad in (rows[:-1],rows+[rows[0]],[{**rows[0],'init':1},*rows[1:]]):
            with self.assertRaises(ValueError):primary_statistics(bad)

    def test_complete_matrix_preserves_canonical_paths_in_local_custody(self):
        reg={'worlds':WORLD_IDS,'histories':HISTORIES,'cells':[list(c) for c in ARMS],
             'output':'/scratch/alpine/paco0228/ACE/results/frozen',
             'worker_hashes':{'delivery_prospective_models.py':'kernel'},'source_hashes':{},'dependencies':{}}
        inputs={c+'/'+h:'paid-sha' for c in WORLD_IDS for h in HISTORIES}
        cells=canonical_cells(reg,inputs,'protocol-sha')
        self.assertEqual(len(cells),640);self.assertEqual(cells[0]['index'],0)
        self.assertTrue(cells[0]['out'].startswith(reg['output']+'/fits/'))
        self.assertEqual(len({(c['case'],c['history'],c['arm'],c['init']) for c in cells}),640)
        bad=copy.deepcopy(reg);bad['worlds'][-1]='30:18'
        with self.assertRaises(ValueError):canonical_cells(bad,inputs,'protocol-sha')
        with self.assertRaises(ValueError):canonical_cells(reg,{},'protocol-sha')

    def test_checkpoint_replay_uses_predictions_and_calibration_prefix_only(self):
        import numpy as np
        torch,_,MLP=runtime(SOURCE);torch.set_num_threads(1)
        data={'order':['R','M','Y'],'parents':{'R':[],'M':['R'],'Y':['M']},'roots':['R'],'target':'Y',
              'rows':[{'query_index':i+1,'clamps':{'R':i/100},'node_values':{'R':i/100,'M':999.,'Y':0.}}
                      for i in range(60)]}
        states={}
        for node in ('M','Y'):
            model=MLP(1,[-3.],[3.])
            with torch.no_grad():
                for p in model.parameters():p.zero_()
                model.net[4].bias[0]=2. if node=='M' else 0.
                if node=='Y':
                    # Input scaling makes this chain return M exactly.
                    model.net[0].weight[0,0]=6.;model.net[0].bias[0]=-3.
                    model.net[2].weight[0,0]=1.;model.net[4].weight[0,0]=1.
            states[node]=model.state_dict()
        prediction,_=replay_predictions(data,states,MLP,torch,'delivery')
        self.assertTrue(np.all(prediction['M']==2.))
        np.testing.assert_allclose(prediction['Y'],2.,rtol=0.,atol=1e-6)
        changed=copy.deepcopy(data)
        for r in changed['rows']:r['node_values']['M']=-999.
        again,_=replay_predictions(changed,states,MLP,torch,'delivery')
        self.assertTrue(np.array_equal(prediction['Y'],again['Y']))
        before=calibration_ranges(data)
        changed=copy.deepcopy(data)
        for r in changed['rows'][50:]:r['node_values']['M']=1e8
        self.assertEqual(before,calibration_ranges(changed))
        with self.assertRaises(ValueError):match({'mse':1.},{'mse':2.},'tampered metric')

    def test_cached_response_identity_is_audited_without_new_simulator_calls(self):
        from delivery_prospective_design import shared_actions,world_spec
        from test_delivery_prospective_models import PROJECT
        spec=world_spec(5,0,PROJECT);actions=shared_actions(spec,'random',17,400)
        # Cached analytic placeholder responses; no structural_values call.
        roots=[n for n in spec['order'] if not spec['parents'][n]]
        data={'order':spec['order'],'parents':spec['parents'],'roots':roots,'target':'X3',
              'rows':[{'query_index':i+1,'clamps':a,'node_values':{n:a.get(n,0.) for n in spec['order']}}
                      for i,a in enumerate(actions)]}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);j=Journal(out/'queries.ndjson',400)
            for _ in actions:j.reserve()
            write(out/'input.json',data)
            receipt={'complete':True,'evaluation':False,'rows':400,'charged_responses':400,
                'input_sha256':sha(out/'input.json'),'journal_sha256':sha(out/'queries.ndjson'),
                'source_world_coefficients_in_input':False,'noise':'disabled in both collection and evaluation'}
            write(out/'receipt.json',receipt)
            self.assertEqual(history_receipt(out,spec,actions,False),data)
            # Even if the surrounding file hash is updated, wrong action IDs fail.
            data['rows'][0]['clamps']=actions[1];write(out/'input.json',data)
            receipt['input_sha256']=sha(out/'input.json');write(out/'receipt.json',receipt)
            with self.assertRaises(ValueError):history_receipt(out,spec,actions,False)
            with self.assertRaises(ValueError):history_receipt(out,spec,actions,True)

    def test_secondary_sensitivity_preserves_all_inits_without_selecting_a_model(self):
        metrics=[]
        for case in WORLD_IDS:
            for history in HISTORIES:
                for arm,init in ARMS:
                    value={'delivery':2.,'online':4.,'simpler':3.,'ablation':8.}[arm]/(init+1)
                    metrics.append(({'case':case,'history':history,'arm':arm,'init':init},value))
        result=secondary_summary(metrics)
        self.assertEqual(result['primary_deployment_init'],0)
        for stratum in result['strata'].values():
            self.assertAlmostEqual(stratum['delivery_vs_short_fit']['ratio'],.25)
            for arm in ('delivery','simpler'):
                self.assertEqual(set(stratum['initialization_sensitivity'][arm]),{'0','1','2'})
                for init in (0,1,2):
                    r=stratum['initialization_sensitivity'][arm][str(init)]
                    self.assertAlmostEqual(r['ratio_to_fixed_init0'],1/(init+1))
                    self.assertEqual(r['systems_better_than_init0'],0 if init==0 else 20)
                    self.assertNotIn('p_raw',r)
        with self.assertRaises(ValueError):secondary_summary(metrics[:-1])


if __name__=='__main__':unittest.main()
